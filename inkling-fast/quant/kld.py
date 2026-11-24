"""Quality check against reference logits with llama-perplexity (KL divergence vs the best available model).

  base:    python kld.py base    --exe llama-perplexity.exe --model ref.gguf  --logits base_logits.bin [-- extra args]
  compare: python kld.py compare --exe llama-perplexity.exe --model cand.gguf --logits base_logits.bin --name mixB [-- extra]

base   = run ONCE with the highest-precision file you can run (BF16/Q8_0; use `-- --n-cpu-moe 30` or `-ngl 0` if it
         doesn't fit in VRAM - slow is fine). Writes the reference logits (~GBs for big vocabularies).
compare= every candidate quant. Prints PPL, mean KLD, 99th-percentile KLD, top-1 agreement; appends a line to kld.tsv.
Both use assets/corpus.txt with the same -c / --chunks, which must stay identical between base and compare.
Default extra args are for a candidate that fits fully on the GPU; override after '--'.
"""
import argparse, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _server import split_args

HERE = os.path.dirname(os.path.abspath(__file__))
own, extra = split_args(sys.argv[1:])
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["base", "compare"])
ap.add_argument("--exe", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--logits", required=True); ap.add_argument("--name", default="")
ap.add_argument("--corpus", default=os.path.join(HERE, "..", "assets", "corpus.txt"))
ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--chunks", type=int, default=24)
ap.add_argument("--logdir", default=os.getcwd())
a = ap.parse_args(own)
name = a.name or os.path.splitext(os.path.basename(a.model))[0]

cmd = [a.exe, "-m", a.model, "-f", a.corpus, "-c", str(a.ctx), "--chunks", str(a.chunks), "-b", "512", "-ub", "256",
       "-t", "2", "--kl-divergence-base", a.logits]
if a.mode == "compare":
    cmd.append("--kl-divergence")
cmd += extra or ["-ngl", "99", "--fit", "off", "-fa", "on"]
log = os.path.join(a.logdir, f"kld_{a.mode}_{name}.log")
with open(log, "w", encoding="utf-8") as f:
    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
t = open(log, encoding="utf-8", errors="replace").read()
if rc != 0:
    print(f"llama-perplexity failed (exit {rc}) - see {log}\n" + t[-1500:]); sys.exit(1)
if a.mode == "base":
    print(f"reference logits written: {a.logits} ({os.path.getsize(a.logits) / 2**30:.2f} GiB)")
    m = re.search(r"Final estimate: PPL = ([\d.]+)", t)
    print("reference PPL", m.group(1) if m else "?"); sys.exit(0)

def g(p):
    m = re.search(p, t)
    return m.group(1) if m else "?"
row = dict(name=name, ppl=g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), kld=g(r"Mean\s+KLD:\s*([\d.]+)"),
           kld99=g(r"99\.0%\s+KLD:\s*([\d.]+)"), top1=g(r"Same top p:\s*([\d.]+)"),
           size=f"{os.path.getsize(a.model) / 2**30:.2f}")
print(f"{name:<24} PPL {row['ppl']}  KLD {row['kld']}  KLD99 {row['kld99']}  top1 {row['top1']}%  {row['size']} GiB")
tsv = os.path.join(a.logdir, "kld.tsv")
new = not os.path.exists(tsv)
with open(tsv, "a", encoding="utf-8") as f:
    if new:
        f.write("\t".join(row) + "\n")
    f.write("\t".join(row.values()) + "\n")
