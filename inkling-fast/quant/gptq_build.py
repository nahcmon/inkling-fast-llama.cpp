"""Build GPTQ-compensated expert tensors (see gptq_lab.py) for whole layers, one small GGUF per layer and type:
gptq/<gu|dn>_<TYPE>_<layer>.gguf, holding only that layer's ffn_<tensor>_exps.weight. Splice in with
assemble.py --take "blk\\.L\\.ffn_gate_up_exps\\.weight=gptq/gu_IQ2_S_L.gguf".

Per expert e: H_e = S_e (C + damp I) S_e, C = layer input correlation (covg/*.cov), S_e = sqrt(imatrix diag of e).
Super-block by super-block (256 columns): quantize with the stock ggml quantizer (imatrix weights), then re-aim
the remaining columns to cancel the error so far: T_R = W_R - ((E_F * s_F) @ K) / s_R, K = C_FR (C_RR)^-1.

python gptq_build.py --tensor gate_up --type IQ2_S --layers 6-17 [--workers 6] [--plain] [--verify]
  --plain   no compensation (must reproduce llama-quantize's bytes; --verify checks expert 0 against a pool file)
"""
import argparse, os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\spiel\Documents\AI\engines\llama.cpp-src\gguf-py")
import gguf
from gguf import GGUFReader


def _lab():
    src = open(os.path.join(HERE, "gptq_lab.py"), encoding="utf-8").read().split("ap = argparse")[0]
    g = {"__name__": "lab"}; exec(src, g); return g


L = _lab()
quant, to_float, TYPES, G = L["quant"], L["to_float"], L["TYPES"], L["G"]
BLK = 256
_W = {}


def init(state):
    _W.update(state)
    t = next(x for x in GGUFReader(state["src"]).tensors if x.name == state["name"])
    _W["raw"] = np.asarray(t.data).reshape(state["n_exp"], -1)
    if state.get("covx"):
        n = state["n_in"]
        _W["cx"] = np.memmap(state["covx"], np.float32, "r", shape=(state["n_exp"], n, n))


def expert_K(e):
    """Per-expert correlation shrunk toward the pooled one by routed sample count (Codex sketch):
    M_e = (1-a) C_pool + a C_e + damp I, a = n_e / (n_e + prior)."""
    s = _W
    a = s["cnt_x"][e] / (s["cnt_x"][e] + s["prior"])
    M = s["Cp"].copy()
    if a > 0:
        Ce = np.array(s["cx"][e], np.float64)
        dg = np.sqrt(np.maximum(np.diag(Ce), 1e-30))
        M = (1 - a) * M + a * (Ce / np.outer(dg, dg))
    M += s["damp"] * np.eye(M.shape[0])
    return [np.linalg.solve(M[st:, st:], M[st:, :st]).T for st in range(BLK, M.shape[0], BLK)]


def do_expert(e):
    s = _W
    n_in, n_out = s["n_in"], s["n_out"]
    W = to_float(s["ttype"], s["raw"][e], n_in * n_out).reshape(n_out, n_in)
    d = s["d"][e]
    imat = np.maximum(d, 1e-8 * max(d.max(), 1e-30)).astype(np.float32)
    tid = s["tid"]
    if s["plain"]:
        Qf = None; T = W
    else:
        sq = np.sqrt(np.maximum(d, 1e-6 * d.max())).astype(np.float64)
        Ks = expert_K(e) if s.get("covx") else s["K"]
        T = W.astype(np.float64).copy(); Qf = np.empty_like(T)
    rb = G.ggml_row_size(tid, BLK)
    out = np.empty((n_out, n_in // BLK, rb), np.uint8)
    for k, st in enumerate(range(0, n_in, BLK)):
        B = slice(st, st + BLK)
        Tb = np.ascontiguousarray(T[:, B], np.float32)
        buf = np.empty(n_out * rb, np.uint8)
        G.ggml_quantize_chunk(tid, Tb.ctypes.data, buf.ctypes.data, 0, n_out, BLK, imat[B].ctypes.data)
        if s.get("refit") and Qf is not None:
            # [refit] codes frozen, per-row super-block scale d (fp16, first 2 bytes of each 256-col block) re-solved
            # for the GPTQ loss of this block given optimal later compensation: P = ((M_rem^-1)_BB)^-1
            Qb = to_float(tid, buf, n_out * BLK).reshape(n_out, BLK).astype(np.float64)
            bb = buf.reshape(n_out, rb)
            dold = bb[:, :2].copy().view(np.float16).astype(np.float64).reshape(-1)
            ok = np.abs(dold) > 0
            qt = (Qb / np.where(ok, dold, 1)[:, None]) * sq[B]
            tt = T[:, B] * sq[B]
            P = s["P"][k]
            qP = qt @ P
            den = np.einsum("ij,ij->i", qP, qt); num = np.einsum("ij,ij->i", qP, tt)
            dnew = np.where(ok & (den > 0), num / np.maximum(den, 1e-30), dold).astype(np.float16)
            dnew = np.where(np.isfinite(dnew), dnew, dold.astype(np.float16))
            bb[:, :2] = dnew.reshape(-1, 1).view(np.uint8).reshape(n_out, 2)
            buf = bb.reshape(-1)
        out[:, k, :] = buf.reshape(n_out, rb)
        if Qf is not None and (st + BLK < n_in or s.get("proxy")):
            Qf[:, B] = to_float(tid, buf, n_out * BLK).reshape(n_out, BLK)
        if Qf is not None and st + BLK < n_in:
            F = slice(0, st + BLK); R = slice(st + BLK, n_in)
            T[:, R] = W[:, R] - ((Qf[:, F] - W[:, F]) * sq[F]) @ Ks[k] / sq[R]
    if s.get("proxy"):
        E = (Qf - W) * sq if Qf is not None else (to_float(tid, out.reshape(-1), n_out * n_in).reshape(n_out, n_in) - W) * np.sqrt(np.maximum(d, 1e-6 * d.max()))
        return e, out.reshape(n_out, -1), float(np.einsum("ij,ij->", E @ s["Cu"], E))
    return e, out.reshape(n_out, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", choices=["gate_up", "down"], required=True)
    ap.add_argument("--type", required=True)
    ap.add_argument("--layers", required=True, help="e.g. 0-5 or 18,19")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--verify", help="pool GGUF to compare expert 0 bytes with (use with --plain)")
    ap.add_argument("--experts", type=int, default=0, help="only the first N experts (testing)")
    ap.add_argument("--out", default="gptq")
    ap.add_argument("--refit", action="store_true", help="re-solve each block's fp16 scale with codes frozen")
    ap.add_argument("--imatrix", help="imatrix file (default imatrix_q8.gguf / imatrix_q8_pad.gguf for down)")
    ap.add_argument("--cov", default="covg", help="directory with <tensor>.cov layer covariances")
    ap.add_argument("--covx", action="store_true", help="per-expert covariance from covx/<name>.covx, shrunk toward pooled")
    ap.add_argument("--prior", type=float, default=0, help="shrinkage prior count for --covx (default n_in)")
    a = ap.parse_args()
    tid = TYPES[a.type]
    layers = []
    for part in a.layers.split(","):
        lo, _, hi = part.partition("-"); layers += list(range(int(lo), int(hi or lo) + 1))
    src = "../cmp/Inkling-2-Mini-UD-Q8_K_XL.gguf" if a.tensor == "gate_up" else "down_f16.gguf"
    imf = a.imatrix or ("imatrix_q8.gguf" if a.tensor == "gate_up" else "imatrix_q8_pad.gguf")
    rs = {t.name: t for t in GGUFReader(src).tensors}
    ims = {t.name: np.asarray(t.data) for t in GGUFReader(imf).tensors}
    os.makedirs(a.out, exist_ok=True)
    tag = ("gu" if a.tensor == "gate_up" else "dn") + ("p" if a.plain else "")
    for l in layers:
        name = f"blk.{l}.ffn_{a.tensor}_exps.weight"
        outf = f"{a.out}/{tag}_{a.type}_{l}.gguf"
        if os.path.exists(outf) and not a.experts: print(f"{outf} exists"); continue
        t0 = time.time()
        t = rs[name]; n_in, n_out, n_exp = (int(v) for v in t.shape)
        s2 = ims[name + ".in_sum2"].reshape(n_exp, -1).astype(np.float64)
        cnt = ims[name + ".counts"].reshape(n_exp).astype(np.float64)
        cov = np.fromfile(f"{a.cov}/{name}.cov", np.float32)
        nc = int(round(np.sqrt(cov.size))); cov = cov.reshape(nc, nc).astype(np.float64)
        gd = np.diag(cov).copy()
        d = np.where(cnt[:, None] > 0, s2 / np.maximum(cnt, 1)[:, None], np.pad(gd, (0, n_in - nc))[None, :])
        K = []
        if not a.plain:
            dg = np.sqrt(np.maximum(gd, 1e-30))
            C = np.eye(n_in); C[:nc, :nc] = cov / np.outer(dg, dg)
            C += a.damp * np.eye(n_in)
            for st in range(BLK, n_in, BLK):
                K.append(np.linalg.solve(C[st:, st:], C[st:, :st]).T)   # C_FR C_RR^-1  (|F| x |R|)
            if a.refit:
                extra_P = [np.linalg.inv(np.linalg.inv(C[st:, st:])[:BLK, :BLK]) for st in range(0, n_in, BLK)]
            if a.experts:
                Cu = C - a.damp * np.eye(n_in)
        if a.covx and not a.plain:
            cxf = f"covx/{name}.covx"
            with open(cxf, "rb") as f:
                f.seek(n_exp * n_in * n_in * 4); cnt_x = np.frombuffer(f.read(n_exp * 8), np.int64).astype(np.float64)
            Cp = np.eye(n_in); Cp[:nc, :nc] = cov / np.outer(dg, dg)
            extra = dict(covx=cxf, cnt_x=cnt_x, prior=a.prior or n_in, Cp=Cp, damp=a.damp)
            K = []
        else:
            extra = {}
        if a.refit and not a.plain: extra["refit"] = True; extra["P"] = extra_P
        if a.experts:
            extra["proxy"] = True
            if a.plain:
                dg = np.sqrt(np.maximum(gd, 1e-30)); Cu = np.eye(n_in); Cu[:nc, :nc] = cov / np.outer(dg, dg)
            extra["Cu"] = Cu
        state = dict(n_in=n_in, n_out=n_out, ttype=int(t.tensor_type), src=src, name=name, n_exp=n_exp, d=d, tid=tid, plain=a.plain, K=K, **extra)
        ne = a.experts or n_exp
        res = [None] * ne
        with Pool(a.workers, initializer=init, initargs=(state,)) as p:
            loss = []
            for r in p.imap_unordered(do_expert, range(ne)):
                res[r[0]] = r[1]
                if len(r) > 2: loss.append(r[2])
        if loss: print(f"  proxy loss (sum over {ne} experts, normalized-correlation units): {sum(loss):.6g}")
        data = np.stack(res)
        if a.verify:
            ref = next(x for x in GGUFReader(a.verify).tensors if x.name == name)
            rb = np.asarray(ref.data).reshape(n_exp, -1)
            print(f"  verify vs {a.verify}: type {ref.tensor_type.name}, expert0 identical: {np.array_equal(rb[0], data[0].reshape(-1))}, "
                  f"all: {np.array_equal(rb[:ne], data.reshape(ne, -1))}")
        if a.experts: print(f"  {ne} experts in {time.time() - t0:.0f}s (test, not written)"); continue
        w = gguf.GGUFWriter(outf + ".tmp", "inkling2")
        w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType(tid))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        os.replace(outf + ".tmp", outf)
        print(f"{outf}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
