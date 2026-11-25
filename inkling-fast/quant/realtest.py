"""Realistic serving test: the launcher's exact server settings, then a chat the way a user has one
(short turns, two photos, a long paste), measuring decode tok/s and VRAM/shared memory after every turn.

  python realtest.py NAME --exe ENGINE --model GGUF --ctx 131072 [--kv-global q3_0]

The prefill release gate is paste prefill >= 1500 tok/s, follow-up decode >= 56 tok/s,
and peak server shared memory <= 64 MiB above the loaded sample.
"""
import argparse, base64, io, json, os, random, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vspeed_long import HERE, gpu_lock, request, wait_until_ready, memory_sample  # noqa: E402
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


def main():
    global ROOT
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
    turns = [
        ("hi", [{"type": "text", "text": "Hi! What can you help me with? Answer in a few sentences."}], 200),
        ("photo 4032x3024", [{"type": "image_url", "image_url": {"url": photo(4032, 3024, 1)}},
                             {"type": "text", "text": "Describe this photo in detail."}], 300),
        ("screenshot 2560x1440", [{"type": "image_url", "image_url": {"url": photo(2560, 1440, 2)}},
                                  {"type": "text", "text": "And this one? Compare both."}], 300),
        ("paste ~5k tok", [{"type": "text", "text": "Summarize this document:\n\n" + paste}], 300),
        ("follow-up", [{"type": "text", "text": "Now write a long, detailed story based on all of the above."}], 500),
    ]

    log_path = HERE / f"realtest_{a.name}.log"
    rows = []
    with gpu_lock(), open(log_path, "w", encoding="utf-8") as log:
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
                def kill(self): subprocess.run(ps + [str(ROOT / "stop-inkling.ps1")], stdin=subprocess.DEVNULL)
                def wait(self): pass
            proc = Launched()
        else:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            wait_until_ready(proc, url)
            time.sleep(2)
            m = memory_sample("loaded", proc.pid)
            rows.append(("loaded", 0, 0, None, None, m))
            msgs = []
            for label, content, n in turns:
                msgs.append({"role": "user", "content": content})
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
                t = threading.Thread(target=watch, daemon=True)
                t.start()
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("realtest wall-clock limit reached")
                    r = request(url, "/v1/chat/completions", {"messages": msgs, "max_tokens": n, "ignore_eos": True,
                                                            "temperature": 1.0}, timeout=remaining)
                finally:
                    stop.set()
                    t.join()
                tm = r.get("timings", {})
                msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
                m = memory_sample(label, proc.pid)
                peak["used"] = max(peak["used"], m["used_mib"])
                peak["srv"] = max(peak["srv"], m["server_shared_mib"])
                ctx_used = r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"]
                rows.append((label, ctx_used, tm.get("prompt_n"), tm.get("prompt_per_second"),
                             tm.get("predicted_per_second"), m, peak))
        finally:
            proc.kill(); proc.wait()

    if a.launcher:
        description = f"run-inkling.ps1 {a.launcher} | port {a.port}"
    else:
        description = (f"{Path(a.model).name} ctx {a.ctx} kv_global={a.kv_global or 'q4_0'} "
                       f"np {a.np} ub {a.ub} {a.extra}")
    out = [f"# {a.name}: {description}",
           f"{'turn':22} {'ctx':>6} {'new':>6} {'pp t/s':>7} {'tg t/s':>7} {'used':>6} {'peak':>6} {'srvShared':>9} {'peakSrvSh':>9} {'allShared':>9}"]
    for row in rows:
        label, ctx_used, pn, pps, tps, m = row[:6]
        peak = row[6] if len(row) > 6 else {"used": m["used_mib"], "srv": m["server_shared_mib"]}
        out.append(f"{label:22} {ctx_used:>6} {pn or 0:>6} {pps or 0:>7.0f} {tps or 0:>7.1f} {m['used_mib']:>6} "
                   f"{peak['used']:>6} {m['server_shared_mib']:>9.0f} {peak['srv']:>9.0f} {m['shared_mib']:>9.0f}")
    text = "\n".join(out)
    print(text)
    with open(HERE / "realtest.tsv", "a", encoding="utf-8") as f:
        f.write(text + "\n\n")


if __name__ == "__main__":
    main()
