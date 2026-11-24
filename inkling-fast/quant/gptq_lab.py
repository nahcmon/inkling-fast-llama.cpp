"""Quantizer lab: does error compensation over the full input covariance (GPTQ-style, done per 256-column
super-block so the IQ/K quantizers stay untouched) beat plain imatrix quantization of the expert weights?

Proxy loss per expert: tr(E H E^T) / tr(W H W^T), E = Q - W, H = expected x x^T of that expert's input.
H_e is approximated as S_e C S_e: C = input correlation of the whole layer (covg/<name>.cov, shared by all
experts), S_e = sqrt(per-expert imatrix diagonal). KLD on the real model decides in the end.

python gptq_lab.py --layer 14 --tensor gate_up --type IQ2_S --experts 0,17,64
"""
import argparse, ctypes, os, sys, time
import numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from gguf import GGUFReader

BIN = os.environ.get("LLAMA_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build-cuda", "bin"))
os.add_dll_directory(BIN)
G = ctypes.CDLL(os.path.join(BIN, "ggml-base.dll"))
G.ggml_quantize_chunk.restype = ctypes.c_size_t
G.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                  ctypes.c_int64, ctypes.c_void_p]
G.ggml_row_size.restype = ctypes.c_size_t
G.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
G.ggml_get_type_traits.restype = ctypes.c_void_p
G.ggml_get_type_traits.argtypes = [ctypes.c_int]
TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
TYPES = {"IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19, "IQ4_NL": 20, "IQ2_S": 22, "IQ1_M": 29,
         "NVFP4": 40, "Q2_0": 42, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12}


def to_float(tid, raw, n):
    fn = TO_FLOAT(ctypes.c_void_p.from_address(G.ggml_get_type_traits(tid) + 40).value)
    out = np.empty(n, np.float32)
    raw = np.ascontiguousarray(raw)
    fn(raw.ctypes.data, out.ctypes.data, n)
    return out


def quant(tid, W, imat):
    """quantize + dequantize W (rows x n) with per-column importance imat"""
    W = np.ascontiguousarray(W, np.float32); rows, n = W.shape
    buf = np.empty(rows * G.ggml_row_size(tid, n), np.uint8)
    im = np.ascontiguousarray(imat, np.float32)
    G.ggml_quantize_chunk(tid, W.ctypes.data, buf.ctypes.data, 0, rows, n, im.ctypes.data)
    return to_float(tid, buf, rows * n).reshape(rows, n)


def loss(E, H, W):
    return float(np.einsum("ij,jk,ik->", E, H, E) / np.einsum("ij,jk,ik->", W, H, W))


def gptq(tid, W, H, imat, blk=256, schur=False):
    """quantize super-block by super-block; after each, re-aim the not-yet-quantized columns so they cancel the
    error made so far under H (exact least squares given the fixed errors)"""
    rows, n = W.shape
    Q = np.zeros_like(W); T = W.astype(np.float64).copy()   # T = current target for unquantized columns
    for s in range(0, n, blk):
        B = slice(s, s + blk); R = slice(s + blk, n)
        w = imat[B]
        if schur:   # importance of a column given that later columns can still compensate (GPTQ's 1/Hinv_jj)
            Hi = np.linalg.inv(H[s:, s:]); w = 1.0 / np.diag(Hi)[:blk]
        Q[:, B] = quant(tid, T[:, B], w)
        if s + blk < n:
            F = slice(0, s + blk)
            Ef = Q[:, F] - W[:, F]
            T[:, R] = W[:, R] - np.linalg.solve(H[R, R], H[R, F] @ Ef.T).T
    return Q


ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=14)
ap.add_argument("--tensor", choices=["gate_up", "down"], default="gate_up")
ap.add_argument("--type", default="IQ2_S")
ap.add_argument("--experts", default="0,17,64")
ap.add_argument("--rows", type=int, default=0, help="only the first N output rows (speed)")
ap.add_argument("--damp", type=float, default=0.01)
a = ap.parse_args()
tid = TYPES[a.type]
name = f"blk.{a.layer}.ffn_{a.tensor}_exps.weight"

src = "../cmp/Inkling-2-Mini-UD-Q8_K_XL.gguf" if a.tensor == "gate_up" else "down_f16.gguf"
t = next(x for x in GGUFReader(src).tensors if x.name == name)
n_in, n_out, n_exp = (int(v) for v in t.shape)
ttype = int(t.tensor_type)
raw = np.asarray(t.data).reshape(n_exp, -1)
im = {x.name: np.asarray(x.data) for x in GGUFReader("imatrix_q8.gguf").tensors if x.name.startswith(name)}
s2 = im[name + ".in_sum2"].reshape(n_exp, -1); cnt = im[name + ".counts"].reshape(n_exp)

cov = np.fromfile(f"covg/{name}.cov", np.float32).reshape(-1, int(np.sqrt(np.fromfile(f"covg/{name}.cov", np.float32).size)))
nc = cov.shape[0]
dg = np.sqrt(np.maximum(np.diag(cov), 1e-30))
C = cov / np.outer(dg, dg)
print(f"{name}: in {n_in} out {n_out} experts {n_exp}, src {t.tensor_type.name}, cov {nc}, "
      f"mean |offdiag corr| {np.abs(C - np.eye(nc)).mean():.4f}")

for e in (int(x) for x in a.experts.split(",")):
    W = to_float(ttype, raw[e], n_in * n_out).reshape(n_out, n_in).astype(np.float64)
    if a.rows: W = W[:a.rows]
    d = (s2[e] / max(cnt[e], 1)).astype(np.float64)
    if n_in > nc:   # padded down: pad columns carry nothing
        d = np.concatenate([d, np.zeros(n_in - d.size)]) if d.size < n_in else d
    dd = d[:nc]
    H = np.zeros((n_in, n_in))
    H[:nc, :nc] = np.sqrt(np.outer(dd, dd)) * C
    H += np.eye(n_in) * (a.damp * np.mean(np.diag(H)[:nc]) + 1e-12)
    imat = np.maximum(d, 1e-8 * d.max()).astype(np.float32)
    res = []
    t0 = time.time(); Qb = quant(tid, W, imat); res.append(("imatrix", loss(Qb - W, H, W), time.time() - t0))
    t0 = time.time(); Qg = gptq(tid, W, H, imat); res.append(("gptq", loss(Qg - W, H, W), time.time() - t0))
    t0 = time.time(); Qs = gptq(tid, W, H, imat, schur=True); res.append(("gptq+schur", loss(Qs - W, H, W), time.time() - t0))
    diag_only = np.diag(np.diag(H))
    print(f" expert {e:3d} (tokens {int(cnt[e])}): " + "  ".join(
        f"{k} {v:.5f} ({v / res[0][1] * 100 - 100:+.1f}%, {s:.1f}s)" for k, v, s in res)
        + f"  | diag-only loss imatrix {loss(Qb - W, diag_only, W):.5f} gptq {loss(Qg - W, diag_only, W):.5f}")
