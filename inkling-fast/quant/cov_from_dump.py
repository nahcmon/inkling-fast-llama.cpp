"""Layer input covariance of expert gate_up (mean of x x^T, f32 n x n, the IMAT_COV_DIR format gptq_build.py reads)
from the f16 rows llama-imatrix wrote with IMAT_DUMP_DIR. Much faster than accumulating it inside the imatrix run.

python cov_from_dump.py [--dump dump] [--out covg2]
"""
import argparse, glob, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="dump")
ap.add_argument("--out", default="covg2")
ap.add_argument("--n", type=int, default=2304, help="row length (n_embd)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for f in sorted(glob.glob(f"{a.dump}/*.ffn_gate_up_exps.weight.x16")):
    X = np.memmap(f, np.float16, "r").reshape(-1, a.n)
    C = np.zeros((a.n, a.n), np.float64)
    for r in range(0, len(X), 8192):
        x = np.asarray(X[r:r + 8192], np.float32)
        C += (x.T @ x).astype(np.float64)
    C /= len(X)
    name = os.path.basename(f)[:-4]
    C.astype(np.float32).tofile(f"{a.out}/{name}.cov")
    print(f"{name}: {len(X)} rows", flush=True)
