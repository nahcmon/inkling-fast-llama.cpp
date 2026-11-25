"""Realistic serving test: the launcher's exact server settings, then a chat the way a user has one
(short turns, two photos, a long paste), measuring TTFT, decode tok/s and VRAM/shared memory.

  python realtest.py NAME --exe ENGINE --model GGUF --ctx 131072 [--kv-global q3_0]

The prefill release gate is paste prefill >= 1500 tok/s, follow-up decode >= 56 tok/s,
and peak server shared memory <= 64 MiB above the loaded sample.
"""
import argparse, base64, io, json, os, random, signal, subprocess, sys, threading, time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vspeed_long import HERE, HTTP, gpu_lock, request, wait_until_ready, memory_sample  # noqa: E402
from PIL import Image  # noqa: E402


def photo(w, h, seed):
    # photo-like content (smooth gradients + noise); the vision cost depends on the size, not the content
    rnd = random.Random(seed)
    img = Image.new("RGB", (64, 48))
    img.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(64 * 48)])
    img = img.resize((w, h), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def stream_chat(url, messages, max_tokens, deadline, **options):
    payload = {"messages": messages, "max_tokens": max_tokens, "ignore_eos": True,
               "temperature": 1.0, "seed": 42, **options,
               "stream": True, "stream_options": {"include_usage": True}}
    req = Request(url + "/v1/chat/completions", json.dumps(payload).encode("utf-8"),
                  {"Content-Type": "application/json", "Accept": "text/event-stream"})
    parts = {"content": [], "reasoning_content": []}
    first_ms = dict.fromkeys(parts)
    usage = timings = None
    done = False
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("realtest wall-clock limit reached")
    start = time.perf_counter()
    try:
        with HTTP.open(req, timeout=remaining) as response:
            for line in response:
                received = time.perf_counter()
                if received - start >= remaining:
                    raise TimeoutError("realtest wall-clock limit reached")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"chat stream failed: {chunk['error']}")
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    for key in parts:
                        value = delta.get(key)
                        if value:
                            if first_ms[key] is None:
                                first_ms[key] = (received - start) * 1000
                            parts[key].append(value)
    except HTTPError as exc:
        raise RuntimeError(f"chat completion: HTTP {exc.code}: {exc.read().decode()}") from exc
    if not done or usage is None or timings is None:
        raise RuntimeError("chat stream ended without completion, usage, or server timings")
    return {"content": "".join(parts["content"]), "first_ms": first_ms,
            "usage": usage, "timings": timings}


def probe_turns(url, paste, slots):
    yield "probe hi", [{"role": "user", "content": "Hi! What can you help me with? Answer in a few sentences."}], 32, {}
    for i in range(10):
        yield f"short follow-up {i + 1:02}", [{"role": "user", "content": "What else?"}], 12, {}
    # Use the tokenizer to size the shared system text, then send it through chat templating.
    if not paste.strip():
        raise ValueError("TTFT probes require a nonempty paste file")
    text = paste
    while True:
        ids = request(url, "/tokenize", {"content": text, "add_special": False})["tokens"]
        if len(ids) >= 2048:
            break
        text += "\n" + paste
    system = request(url, "/detokenize", {"tokens": ids[:2048]})["content"]
    for i in range(5):
        yield f"shared system {i + 1}", [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Give me one fact from section {i + 1}."},
        ], 12, {"id_slot": i % slots}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--model-dir", default=str(HERE.parent),
                    help="folder with the GGUFs, the mmproj and run-inkling.ps1 (default: parent of this script)")
    ap.add_argument("--paste", default="", help="text file for the ~5k-token paste turn (default: MODEL_DIR/cmp/eval.txt)")
    ap.add_argument("--exe", default="", help="llama-server.exe (default: the engine run-inkling.ps1 uses)")
    ap.add_argument("--model", default="", help="GGUF (default: MODEL_DIR/Inkling-2-Mini-fast-9.3-tuned.gguf)")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--kv-global", default="")
    ap.add_argument("--np", type=int, default=8)
    ap.add_argument("--ub", type=int, default=256)
    ap.add_argument("--extra", default="", help="extra server args, space separated")
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--timeout", type=float, default=165, help="wall-clock limit in seconds after GPU lock acquisition")
    ap.add_argument("--ttft-probe", action="store_true", help="run short follow-up and shared system prompt probes instead")
    ap.add_argument("--cpu-sampler", action="store_true", help="disable per-request GPU sampling for profiling")
    ap.add_argument("--probe-effort", choices=["none"], help="explicitly request minimal reasoning in the TTFT probe")
    ap.add_argument("--launcher", default="",
                    help='start the server with ../run-inkling.ps1 and these args instead (e.g. "-Mode Standard"); '
                         'it is stopped with stop-inkling.ps1 afterwards')
    a = ap.parse_args()
    ROOT = Path(a.model_dir).resolve()
    a.model = a.model or str(ROOT / "Inkling-2-Mini-fast-9.3-tuned.gguf")
    if not a.exe:
        import re
        launcher = (ROOT / "run-inkling.ps1").read_text(encoding="utf-8-sig")
        m = re.search(r'^\$exe = "([^"]+)"', launcher, re.M)
        a.exe = m.group(1) if m else str(ROOT.parent / "build-cuda/bin/llama-server.exe")

    env = dict(os.environ)
    env.pop("LLAMA_KV_TYPE_GLOBAL", None)
    if a.kv_global:
        env["LLAMA_KV_TYPE_GLOBAL"] = a.kv_global
    url = f"http://127.0.0.1:{a.port}"
    cmd = [a.exe, "-m", a.model, "-c", str(a.ctx), "-ub", str(a.ub), "-t", "2", "-ctk", "q4_0", "-ctv", "q4_0",
           "-bs", "--jinja", "-ngl", "99", "--fit", "off", "-fa", "on", "-np", str(a.np), "--ctx-checkpoints", "2",
           "-lm", "none", "--cache-ram", "0", "--temp", "1.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0.0",
           "--mmproj", str(ROOT / "mmproj-Inkling-2-Mini-Q8_0.gguf"), "--image-max-tokens", "1024", "--host", "127.0.0.1", "--port", str(a.port)]
    if a.np > 1:
        cmd.append("-kvu")
    cmd += a.extra.split()

    paste = Path(a.paste or ROOT / "cmp" / "eval.txt").read_text(encoding="utf-8")[:22000]
    turns = [] if a.ttft_probe else [
        ("hi", [{"type": "text", "text": "Hi! What can you help me with? Answer in a few sentences."}], 200),
        ("photo 4032x3024", [{"type": "image_url", "image_url": {"url": photo(4032, 3024, 1)}},
                             {"type": "text", "text": "Describe this photo in detail."}], 300),
        ("screenshot 2560x1440", [{"type": "image_url", "image_url": {"url": photo(2560, 1440, 2)}},
                                  {"type": "text", "text": "And this one? Compare both."}], 300),
        ("paste ~5k tok", [{"type": "text", "text": "Summarize this document:\n\n" + paste}], 300),
        ("follow-up", [{"type": "text", "text": "Now write a long, detailed story based on all of the above."}], 500),
    ]

    log_path = HERE / f"realtest_{a.name}.log"
    lock_dir = ROOT / "shrink"
    lock_path = (lock_dir if lock_dir.is_dir() else HERE) / "gpu.lock"
    rows = []
    with gpu_lock(lock_path), open(log_path, "w", encoding="utf-8") as log:
        deadline = time.monotonic() + a.timeout
        if a.launcher:  # test exactly what the user runs: the deployed launcher script
            ps = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
            r = subprocess.run(ps + [str(ROOT / "run-inkling.ps1"), *a.launcher.split(), "-LocalOnly", "-Port", str(a.port)],
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=min(60, a.timeout))
            # (no pipes: the server inherits the launcher's handles and would keep a pipe open forever)
            if r.returncode != 0:
                raise RuntimeError(f"launcher failed, see {log_path}")

            class Launched:
                pid = int((ROOT / "inkling-server.pid").read_text().strip())
                def poll(self): return None
                def terminate(self):
                    try:
                        os.kill(self.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                def kill(self): subprocess.run(ps + [str(ROOT / "stop-inkling.ps1")], stdin=subprocess.DEVNULL)
                def wait(self): pass
            proc = Launched()
        else:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        watchdog = threading.Timer(max(0, deadline - time.monotonic()), proc.terminate)
        watchdog.start()
        try:
            wait_until_ready(proc, url)
            time.sleep(2)
            m = memory_sample("loaded", proc.pid)
            rows.append(("loaded", 0, {}, m, {"used": m["used_mib"], "srv": m["server_shared_mib"]}, {}))
            msgs = []
            requests = (probe_turns(url, paste, request(url, "/props")["total_slots"]) if a.ttft_probe else
                        ((label, [{"role": "user", "content": content}], n, {}) for label, content, n in turns))
            for label, messages, n, options in requests:
                if "id_slot" in options:
                    msgs = []
                msgs.extend(messages)
                peak = {"used": 0, "srv": 0.0}
                stop = threading.Event()

                def watch():
                    while not stop.is_set():
                        try:
                            s = memory_sample("x", proc.pid)
                            peak["used"] = max(peak["used"], s["used_mib"])
                            peak["srv"] = max(peak["srv"], s["server_shared_mib"])
                        except Exception:
                            pass
                        stop.wait(1.0)
                if not a.ttft_probe:
                    t = threading.Thread(target=watch, daemon=True)
                    t.start()
                try:
                    if a.cpu_sampler:
                        options["backend_sampling"] = False
                    if a.ttft_probe and a.probe_effort:
                        options["chat_template_kwargs"] = {"reasoning_effort": a.probe_effort}
                    r = stream_chat(url, msgs, n, deadline, **options)
                finally:
                    if not a.ttft_probe:
                        stop.set()
                        t.join()
                msgs.append({"role": "assistant", "content": r["content"]})
                if not a.ttft_probe:
                    m = memory_sample(label, proc.pid)
                    peak["used"] = max(peak["used"], m["used_mib"])
                    peak["srv"] = max(peak["srv"], m["server_shared_mib"])
                ctx_used = r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
                rows.append((label, ctx_used, r["timings"], m, peak, r["first_ms"]))
        finally:
            watchdog.cancel()
            watchdog.join()
            proc.kill(); proc.wait()

    if a.launcher:
        description = f"run-inkling.ps1 {a.launcher} | port {a.port}"
    else:
        description = (f"{Path(a.model).name} ctx {a.ctx} kv_global={a.kv_global or 'q4_0'} "
                       f"np {a.np} ub {a.ub} {a.extra}")
    if a.probe_effort:
        description += f" | probe reasoning effort: {a.probe_effort}"
    out = [f"# {a.name}: {description}",
           f"{'turn':22} {'ctx':>6} {'new':>6} {'pp t/s':>7} {'tg t/s':>7} {'used':>6} {'peak':>6} {'srvShared':>9} "
           f"{'peakSrvSh':>9} {'allShared':>9} {'TTFT ms':>9} {'reason ms':>9} {'prompt ms':>9}"]

    def fmt_ms(value):
        return f"{value:9.1f}" if value is not None else f"{'-':>9}"

    for label, ctx_used, tm, m, peak, first_ms in rows:
        pn = tm.get("prompt_n", 0)
        pps = tm.get("prompt_per_second", 0)
        tps = tm.get("predicted_per_second", 0)
        if a.ttft_probe and label != "loaded":
            memory = f"{'-':>6} {'-':>6} {'-':>9} {'-':>9} {'-':>9}"
        else:
            memory = (f"{m['used_mib']:>6} {peak['used']:>6} {m['server_shared_mib']:>9.0f} "
                      f"{peak['srv']:>9.0f} {m['shared_mib']:>9.0f}")
        out.append(f"{label:22} {ctx_used:>6} {pn or 0:>6} {pps or 0:>7.0f} {tps or 0:>7.1f} {memory} "
                   f"{fmt_ms(first_ms.get('content'))} {fmt_ms(first_ms.get('reasoning_content'))} {fmt_ms(tm.get('prompt_ms'))}")
    text = "\n".join(out)
    print(text)
    with open(HERE / "realtest.tsv", "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


if __name__ == "__main__":
    main()
