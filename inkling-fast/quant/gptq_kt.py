"""GPTQ with ik_llama.cpp's IQ2_KT trellis codes (expert gate_up). Same compensation as gptq_build.py, but the
per-row float scale is fitted once on the original row by ik's stock encoder and frozen; then super-block by
super-block (256 columns) the compensated target is encoded with ik_iq2kt_encode_fixed ([TAG_GPTQ_KT] export of the
patched ik build: 4-bit sub-block scale + trellis search under that row scale) and the rest re-aimed.

Writes kt/gu_GKT_<layer>.gguf as Q8_0 (decoded values) for a KLD test on mainline; --raw also saves the IQ2_KT bytes
(kt/gu_GKT_<layer>.npy) for a later native build.

python gptq_kt.py --layers 6-17 [--damp 1.0] [--plain] [--experts 4]
"""
import argparse, ctypes, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import kt_test as KT   # ik() loader, to_float, G
import gguf
from gguf import GGUFReader

BLK = 256
Q8_0, IQ2_KT = 8, 153


def load():
    K = KT.ik()
    f = ctypes.c_void_p
    K.ik_iq2kt_encode_fixed.argtypes = [f, f, f, ctypes.c_int64, f]; K.ik_iq2kt_encode_fixed.restype = None
    K.ik_iq2kt_decode_fixed.argtypes = [f, f, ctypes.c_int64, f]; K.ik_iq2kt_decode_fixed.restype = None
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--plain", action="store_true", help="no compensation (checks the fixed-scale encoder vs stock)")
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--out", default="kt")
    ap.add_argument("--cov", default="covg")
    ap.add_argument("--imatrix", default="imatrix_q8.gguf")
    ap.add_argument("--tag", default="GKT")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    K = load()
    layers = []
    for part in a.layers.split(","):
        lo, _, hi = part.partition("-"); layers += list(range(int(lo), int(hi or lo) + 1))
    os.makedirs(a.out, exist_ok=True)
    rs = {t.name: t for t in GGUFReader("../cmp/Inkling-2-Mini-UD-Q8_K_XL.gguf").tensors}
    ims = {t.name: np.asarray(t.data) for t in GGUFReader(a.imatrix).tensors}
    G, to_float = KT.G, KT.to_float
    pool = ThreadPoolExecutor(a.workers)
    for l in layers:
        t0 = time.time()
        name = f"blk.{l}.ffn_gate_up_exps.weight"
        outf = f"{a.out}/gu_{a.tag}_{l}.gguf"
        if os.path.exists(outf) and not a.experts: print(f"{outf} exists"); continue
        t = rs[name]; n_in, n_out, n_exp = (int(v) for v in t.shape)
        ne = a.experts or n_exp
        s2 = ims[name + ".in_sum2"].reshape(n_exp, -1).astype(np.float64)
        cnt = ims[name + ".counts"].reshape(n_exp).astype(np.float64)
        cov = np.fromfile(f"{a.cov}/{name}.cov", np.float32)
        nc = int(round(np.sqrt(cov.size))); cov = cov.reshape(nc, nc).astype(np.float64)
        gd = np.diag(cov).copy()
        dd = np.where(cnt[:, None] > 0, s2 / np.maximum(cnt, 1)[:, None], np.pad(gd, (0, n_in - nc))[None, :])
        dg = np.sqrt(np.maximum(gd, 1e-30))
        C = np.eye(n_in); C[:nc, :nc] = cov / np.outer(dg, dg)
        Cu = C.copy()
        C += a.damp * np.eye(n_in)
        Ks = [np.linalg.solve(C[st:, st:], C[st:, :st]).T for st in range(BLK, n_in, BLK)]
        raw = np.asarray(t.data).reshape(n_exp, -1)
        rb = K.ggml_row_size(IQ2_KT, n_in); bsz = (rb - 4) // (n_in // BLK)
        rb8 = G.ggml_row_size(Q8_0, n_in)
        out8 = np.empty((ne, n_out, rb8), np.uint8)
        outk = np.empty((ne, n_out, rb), np.uint8) if a.raw else None
        stats = [None] * ne

        def one(e):
            W = to_float(int(t.tensor_type), raw[e], n_in * n_out).reshape(n_out, n_in).astype(np.float64)
            d = dd[e]
            imat = np.maximum(d, 1e-8 * max(d.max(), 1e-30)).astype(np.float32)
            sq = np.sqrt(np.maximum(d, 1e-6 * d.max()))
            Wf = np.ascontiguousarray(W, np.float32)
            stock = np.empty(n_out * rb, np.uint8)
            K.quantize_iq2_kt(Wf.ctypes.data, stock.ctypes.data, n_out, n_in, imat.ctypes.data, None)
            stock = stock.reshape(n_out, rb)
            drow = np.ascontiguousarray(stock[:, :4]).view(np.float32).reshape(-1).copy()
            T = W.copy(); Qf = np.empty_like(W)
            row = np.empty((n_out, rb), np.uint8); row[:, :4] = stock[:, :4]
            for k, st in enumerate(range(0, n_in, BLK)):
                B = slice(st, st + BLK)
                Tb = np.ascontiguousarray(T[:, B], np.float32)
                blk = np.empty((n_out, bsz), np.uint8)
                K.ik_iq2kt_encode_fixed(Tb.ctypes.data, np.ascontiguousarray(imat[B]).ctypes.data, drow.ctypes.data, n_out, blk.ctypes.data)
                Qb = np.empty((n_out, BLK), np.float32)
                K.ik_iq2kt_decode_fixed(blk.ctypes.data, drow.ctypes.data, n_out, Qb.ctypes.data)
                row[:, 4 + k * bsz: 4 + (k + 1) * bsz] = blk
                Qf[:, B] = Qb
                if not a.plain and st + BLK < n_in:
                    F = slice(0, st + BLK); R = slice(st + BLK, n_in)
                    T[:, R] = W[:, R] - ((Qf[:, F] - W[:, F]) * sq[F]) @ Ks[k] / sq[R]
            # the assembled rows must decode with ik's own dequantizer to exactly what the loop used
            D = np.empty((n_out, n_in), np.float32)
            for r in range(0, n_out, 97):
                K.dequantize_row_iq2_kt(row[r].ctypes.data, D[r].ctypes.data, n_in)
                assert np.allclose(D[r], Qf[r], rtol=1e-5, atol=1e-7 * np.abs(Qf[r]).max()), f"decode mismatch e{e} r{r}"
            E = (Qf - W) * sq
            Es = None
            if a.experts:
                Sf = np.empty((n_out, n_in), np.float32)
                for r in range(n_out): K.dequantize_row_iq2_kt(stock[r].ctypes.data, Sf[r].ctypes.data, n_in)
                Es = (Sf - W) * sq
                stats[e] = (float(np.einsum("ij,ij->", E @ Cu, E)), float(np.einsum("ij,ij->", Es @ Cu, Es)))
            Q32 = np.ascontiguousarray(Qf, np.float32)
            o = np.empty(n_out * rb8, np.uint8)
            G.ggml_quantize_chunk(Q8_0, Q32.ctypes.data, o.ctypes.data, 0, n_out, n_in, None)
            out8[e] = o.reshape(n_out, rb8)
            if outk is not None: outk[e] = row
        list(pool.map(one, range(ne)))
        if a.experts:
            g, s = sum(x[0] for x in stats), sum(x[1] for x in stats)
            print(f"{name}: proxy loss {'plain' if a.plain else 'gptq'}-fixed {g:.6g} vs stock IQ2_KT {s:.6g} ({g / s - 1:+.1%}), {time.time() - t0:.0f}s")
            continue
        w = gguf.GGUFWriter(outf + ".tmp", "inkling2")
        w.add_tensor(name, out8, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        os.replace(outf + ".tmp", outf)
        if outk is not None: np.save(f"{a.out}/gu_{a.tag}_{l}.npy", outk)
        print(f"{outf}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
