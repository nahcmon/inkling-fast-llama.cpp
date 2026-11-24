"""Quality test of ik_llama.cpp's trellis types (IQ2_KT / IQ3_KT / IQ4_KT) on expert gate_up, before any CUDA port.
Quantizes with ik's own encoder (imatrix per expert), decodes, and stores the result as Q8_0 so this llama.cpp build
can run it. The incumbent goes through the same Q8_0 round trip (--from FILE: decode that file's tensor instead).

python kt_test.py --type IQ2_KT --layers 6-17 --out kt
python kt_test.py --from ../Inkling-2-Mini-fast-9.2-gptq.gguf --layers 6-17 --out kt     (incumbent, round trip)
"""
import argparse, ctypes, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "engines", "llama.cpp-src", "gguf-py"))
import gguf
from gguf import GGUFReader


def _lab():
    p = os.path.join(HERE, "gptq_lab.py")
    src = open(p, encoding="utf-8").read().split("ap = argparse")[0]
    g = {"__name__": "lab", "__file__": p}; exec(src, g); return g


L = _lab()
to_float, G = L["to_float"], L["G"]
Q8_0 = 8
IK_BIN = os.environ.get("IK_BIN", os.path.join(HERE, "..", "..", "..", "..", "engines", "ik_src", "ik_llama.cpp-main", "build-cpu", "bin"))
KT = {"IQ2_KT": 153, "IQ3_KT": 154, "IQ4_KT": 155}


def ik():
    os.add_dll_directory(os.path.abspath(IK_BIN))
    K = ctypes.CDLL(os.path.join(os.path.abspath(IK_BIN), "ggml.dll"))
    K.ggml_row_size.restype = ctypes.c_size_t; K.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    for t in ("iq2_kt", "iq3_kt", "iq4_kt"):
        q = getattr(K, f"quantize_{t}"); q.restype = ctypes.c_size_t
        q.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
        d = getattr(K, f"dequantize_row_{t}"); d.restype = None
        d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    if hasattr(K, "ggml_cpu_init"): K.ggml_cpu_init()
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=list(KT))
    ap.add_argument("--from", dest="src")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", default="kt")
    ap.add_argument("--experts", type=int, default=0, help="test: first N experts, print weighted error vs source")
    a = ap.parse_args()
    layers = []
    for part in a.layers.split(","):
        lo, _, hi = part.partition("-"); layers += list(range(int(lo), int(hi or lo) + 1))
    os.makedirs(a.out, exist_ok=True)
    K = ik() if a.type else None
    rs = {t.name: t for t in GGUFReader("../cmp/Inkling-2-Mini-UD-Q8_K_XL.gguf").tensors}
    fs = {t.name: t for t in GGUFReader(a.src).tensors} if a.src else None
    ims = {t.name: np.asarray(t.data) for t in GGUFReader("imatrix_q8.gguf").tensors}
    tag = a.type or "rt"
    pool = ThreadPoolExecutor(6)
    for l in layers:
        t0 = time.time()
        name = f"blk.{l}.ffn_gate_up_exps.weight"
        outf = f"{a.out}/gu_{tag}_{l}.gguf"
        if os.path.exists(outf) and not a.experts: print(f"{outf} exists"); continue
        t = rs[name]; n_in, n_out, n_exp = (int(v) for v in t.shape)
        ne = a.experts or n_exp
        s2 = ims[name + ".in_sum2"].reshape(n_exp, -1).astype(np.float64)
        cnt = ims[name + ".counts"].reshape(n_exp).astype(np.float64)
        d = s2 / np.maximum(cnt, 1)[:, None]
        imat = np.maximum(d, 1e-8 * np.maximum(d.max(1, keepdims=True), 1e-30)).astype(np.float32)
        raw = np.asarray(t.data).reshape(n_exp, -1)
        if fs: fraw = np.asarray(fs[name].data).reshape(n_exp, -1); ftype = int(fs[name].tensor_type)
        rb8 = G.ggml_row_size(Q8_0, n_in)
        out = np.empty((ne, n_out, rb8), np.uint8)
        errs = [0.0] * ne

        def one(e):
            W = to_float(int(t.tensor_type), raw[e], n_in * n_out).reshape(n_out, n_in)
            if fs:
                D = to_float(ftype, fraw[e], n_in * n_out).reshape(n_out, n_in)
            else:
                tid = KT[a.type]; low = a.type.lower()
                rb = K.ggml_row_size(tid, n_in)
                buf = np.empty(n_out * rb, np.uint8)
                x = np.ascontiguousarray(W, np.float32); im = np.ascontiguousarray(imat[e])
                getattr(K, f"quantize_{low}")(x.ctypes.data, buf.ctypes.data, n_out, n_in, im.ctypes.data, None)
                D = np.empty((n_out, n_in), np.float32)
                dq = getattr(K, f"dequantize_row_{low}")
                for r in range(n_out):
                    dq(buf.ctypes.data + r * rb, D[r].ctypes.data, n_in)
            errs[e] = float((((D - W) ** 2) * imat[e]).sum() / ((W ** 2) * imat[e]).sum())
            D = np.ascontiguousarray(D, np.float32)
            o = np.empty(n_out * rb8, np.uint8)
            G.ggml_quantize_chunk(Q8_0, D.ctypes.data, o.ctypes.data, 0, n_out, n_in, None)
            out[e] = o.reshape(n_out, rb8)
        list(pool.map(one, range(ne)))
        print(f"{name} {tag}: imatrix-weighted rel err {np.mean(errs):.5f} over {ne} experts ({time.time() - t0:.0f}s)", flush=True)
        if a.experts: continue
        w = gguf.GGUFWriter(outf + ".tmp", "inkling2")
        w.add_tensor(name, out, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        os.replace(outf + ".tmp", outf)


if __name__ == "__main__":
    main()
