"""Measure one uncached long completion with the vision settings from vspeed.sh."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCK = HERE / "gpu.lock"
CONTEXT = 262144
DECODE_TOKENS = 128
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@contextmanager
def gpu_lock(lock_path=LOCK):
    while True:
        try:
            lock = lock_path.open("x")
            break
        except FileExistsError:
            print(f"Waiting for {lock_path}", flush=True)
            time.sleep(10)
    try:
        with lock:
            print(os.getpid(), file=lock)
        yield
    finally:
        lock_path.unlink()


def request(url, route, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url + route, body, {"Content-Type": "application/json"})
    try:
        with HTTP.open(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{route}: HTTP {exc.code}: {exc.read().decode()}") from exc


def wait_until_ready(proc, url):
    deadline = time.monotonic() + 60
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {proc.returncode}; see server log")
        try:
            if request(url, "/health", timeout=2)["status"] == "ok":
                return
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if getattr(getattr(exc, "reason", exc), "winerror", None) == 10013:
                raise RuntimeError(f"Localhost HTTP access was denied: {exc}") from exc
        time.sleep(1)
    raise RuntimeError(f"llama-server health check timed out: {last_error}")


def prompt_tokens(url, count, corpus_path):
    text = corpus_path.read_text(encoding="utf-8")
    corpus = request(url, "/tokenize", {"content": text, "add_special": False})["tokens"]
    suffix = request(
        url, "/tokenize",
        {"content": "\n\nWrite a long story about a lighthouse keeper.\n", "add_special": False},
    )["tokens"]
    repeated = corpus * ((count - len(suffix) + len(corpus) - 1) // len(corpus))
    return repeated[:count - len(suffix)] + suffix


def memory_sample(phase, pid=0):
    start = time.monotonic()
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, timeout=10,
    )
    counters = subprocess.check_output(
        [
            "powershell.exe", "-NoProfile", "-Command",
            "$ErrorActionPreference='Stop'; "
            "$samples = (Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage',"
            "'\\GPU Process Memory(*)\\Shared Usage').CounterSamples; "
            "$adapter = ($samples | Where-Object Path -Like '*gpu adapter memory*' | "
            "Measure-Object CookedValue -Sum).Sum; "
            f"$server = ($samples | Where-Object InstanceName -Like 'pid_{pid}_*' | "
            "Measure-Object CookedValue -Sum).Sum; "
            "[Console]::WriteLine(('{0} {1}' -f "
            "[math]::Round($adapter/1MB,1), [math]::Round($server/1MB,1)))",
        ],
        text=True, timeout=15,
    )
    shared, server_shared = (float(value.replace(",", ".")) for value in counters.split())
    return {
        "phase": phase, "monotonic_s": round(start, 2),
        "used_mib": int(used.strip()), "shared_mib": shared, "server_shared_mib": server_shared,
    }


def run(args):
    model = args.model.resolve()
    url = f"http://127.0.0.1:{args.port}"
    stem = HERE / f"vspeed_long_{args.name}"
    env = dict(os.environ)
    env.pop("LLAMA_KV_TYPE_GLOBAL", None)
    if args.kv_global == "q3_0":
        env["LLAMA_KV_TYPE_GLOBAL"] = args.kv_global
    cmd = [
        str(args.exe.resolve()), "-m", str(model), "-c", str(args.context),
        "-ub", str(args.ub), "-t", "2", "-ctk", "q4_0", "-ctv", "q4_0",
        "-bs", "--jinja", "-ngl", "99", "--fit", "off", "-fa", "on",
        "-np", "8", "--ctx-checkpoints", "2", "-kvu",
        "--mmproj", str(args.model_dir / "mmproj-Inkling-2-Mini-Q8_0.gguf"),
        "--image-max-tokens", "1024",
        "-lm", "none", "--cache-ram", "0", "--host", "127.0.0.1", "--port", str(args.port),
    ]
    with gpu_lock(), stem.with_suffix(".log").open("x", encoding="utf-8") as log:
        with stem.with_suffix(".memory.jsonl").open("x", encoding="utf-8") as memory_log:
            stop = threading.Event()

            def sample(phase, pid=0):
                values = memory_sample(phase, pid)
                print(json.dumps(values), file=memory_log, flush=True)
                return values

            def sample_periodically():
                while not stop.wait(10):
                    try:
                        sample("completion", proc.pid)
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        print(json.dumps({"sample_error": str(exc)}), file=memory_log, flush=True)

            baseline = sample("baseline")
            deadline = time.monotonic() + args.timeout
            proc = subprocess.Popen(cmd, cwd=args.model_dir, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
            sampler = None
            try:
                wait_until_ready(proc, url)
                tokens = prompt_tokens(url, args.tokens, args.corpus)
                sample("ready", proc.pid)
                print(f"{model.name}: sending {len(tokens):,} uncached tokens", flush=True)
                sampler = threading.Thread(target=sample_periodically)
                sampler.start()
                result = request(
                    url, "/completion",
                    {"prompt": tokens, "cache_prompt": False, "n_predict": DECODE_TOKENS,
                     "temperature": 0, "ignore_eos": True, "seed": 42},
                    timeout=max(1, deadline - time.monotonic()),
                )
                stop.set()
                sampler.join()
                filled = sample("filled", proc.pid)
                timing = result["timings"]
                if (result.get("truncated") or timing.get("cache_n", 0) != 0
                        or timing["prompt_n"] != args.tokens
                        or timing["predicted_n"] != DECODE_TOKENS):
                    raise RuntimeError(f"Unexpected token counts or truncation: {timing}")
                metrics = {
                    "model": model.name, "context": args.context, "ub": args.ub, "timings": timing,
                    "baseline": baseline, "filled": filled,
                }
                with stem.with_suffix(".json").open("x", encoding="utf-8") as output:
                    json.dump(metrics, output, indent=2)
                print(json.dumps(metrics), flush=True)
            finally:
                stop.set()
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                if sampler is not None:
                    sampler.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("name", help="unique label for the result and logs")
    parser.add_argument("--context", type=int, default=CONTEXT)
    parser.add_argument("--ub", type=int, default=256)
    parser.add_argument("--model-dir", type=Path, default=ROOT)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--tokens", type=int, default=200000)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--timeout", type=int, default=165, help="wall-clock limit in seconds after server launch")
    parser.add_argument("--kv-global", choices=("q3_0", "q4_0"), default="q3_0")
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    args.corpus = (args.corpus or args.model_dir / "cmp" / "eval.txt").resolve()
    if not 64 <= args.tokens <= args.context - DECODE_TOKENS:
        parser.error("prompt must leave room for 128 generated tokens")
    try:
        run(args)
    except (OSError, RuntimeError) as exc:
        parser.exit(1, f"{args.name}: {exc}\n")
