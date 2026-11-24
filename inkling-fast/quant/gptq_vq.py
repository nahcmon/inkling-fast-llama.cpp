"""Full GPTQ for IQ2_XXS / IQ2_S / IQ3_XXS expert tensors: error feedback inside each 256-column super-block too.

gptq_build.py compensates only between super-blocks (the stock quantizer picks all codes of a block at once).
Here, per super-block: the stock ggml quantizer picks the scales (fp16 d + 4-bit sub-block scales) for the current
compensated target, then the grid codes are re-chosen group by group (8 columns; IQ3_XXS: 4) in column order, each
group minimizing the exact GPTQ group loss (q - t)^T P (q - t), P = (Hinv_GG)^-1 of the not-yet-quantized columns,
and its error is fed forward to every later column (standard GPTQ via the upper Cholesky factor U of H^-1).
Everything runs in column-scaled space W' = W * s_e (s_e = sqrt imatrix diag of expert e), where H_e = S_e Cn S_e
becomes the shared Cn = C + damp*I, so U, P are shared by all experts and all experts run as one batch on the GPU.

python gptq_vq.py --type IQ2_S --layers 6-17 --damp 1.0 --out gptq_vq1.0 [--experts N] [--device cuda]
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "engines", "llama.cpp-src", "gguf-py"))
import gguf
from gguf import GGUFReader


def _lab():
    p = os.path.join(HERE, "gptq_lab.py")
    src = open(p, encoding="utf-8").read().split("ap = argparse")[0]
    g = {"__name__": "lab", "__file__": p}; exec(src, g); return g


L = _lab()
to_float, TYPES, G = L["to_float"], L["TYPES"], L["G"]
BLK = 256
F16 = lambda b: b.view(np.float16).astype(np.float32)


# ---------------------------------------------------------------- grids (read back through ggml's own dequantizer)
def _grids():
    ks = np.array([s | ((bin(s).count("1") & 1) << 7) for s in range(128)], np.uint8)   # ksigns_iq2xs
    one = np.frombuffer(np.float16(1.0).tobytes(), np.uint8)
    # IQ2_XXS: 66 B, db = d*(0.5+s)*0.25 with s=1 -> 0.375
    blocks = []
    for b in range(8):
        q = np.zeros(16, np.uint32)
        for ib in range(8):
            q[2 * ib] = int.from_bytes(bytes(range(32 * b + 4 * ib, 32 * b + 4 * ib + 4)), "little")
            q[2 * ib + 1] = 1 << 28
        blocks.append(np.concatenate([one, q.view(np.uint8)]))
    g2xxs = (to_float(TYPES["IQ2_XXS"], np.concatenate(blocks), 8 * 256) / 0.375).reshape(256, 8)
    # IQ2_S: 82 B = d, qs[32] low bits, signs[32], qh[8], scales[8]; scale nibble 1 -> 0.375
    blocks = []
    for b in range(32):
        idx = np.arange(32 * b, 32 * b + 32)
        qs = (idx & 255).astype(np.uint8)
        hi = (idx >> 8).reshape(8, 4)
        qh = (hi[:, 0] | hi[:, 1] << 2 | hi[:, 2] << 4 | hi[:, 3] << 6).astype(np.uint8)
        blocks.append(np.concatenate([one, qs, np.zeros(32, np.uint8), qh, np.full(8, 0x11, np.uint8)]))
    g2s = (to_float(TYPES["IQ2_S"], np.concatenate(blocks), 32 * 256) / 0.375).reshape(1024, 8)
    # IQ3_XXS: 98 B = d, qs[64] grid idx, 8 x u32 (signs | scale<<28); db = d*(0.5+s)*0.5, s=1 -> 0.75
    blocks = []
    for b in range(4):
        qs = np.arange(64 * b, 64 * b + 64).astype(np.uint8)
        ss = np.full(8, 1 << 28, np.uint32)
        blocks.append(np.concatenate([one, qs, ss.view(np.uint8)]))
    g3 = (to_float(TYPES["IQ3_XXS"], np.concatenate(blocks), 4 * 256) / 0.75).reshape(256, 4)
    return ks, g2xxs, g2s, g3


KS, GRID2XXS, GRID2S, GRID3 = _grids()
assert np.allclose(GRID2XXS, np.round(GRID2XXS)) and GRID2XXS.min() > 0 and GRID2S.min() > 0 and GRID3.min() > 0


# ---------------------------------------------------------------- per-type parse / encode
def parse_db(tid, blk):
    """blk: (R, rb) uint8 of one super-block per row -> per-column scale (R, 256) float32, as the dequantizer uses it"""
    d = F16(blk[:, :2].copy()).reshape(-1, 1)
    if tid == TYPES["IQ2_XXS"]:
        aux = blk[:, 2:66].copy().view(np.uint32).reshape(-1, 8, 2)[:, :, 1]
        db = d * (0.5 + (aux >> 28)) * 0.25                      # (R, 8) per 32
        return np.repeat(db, 32, axis=1).astype(np.float32)
    if tid == TYPES["IQ2_S"]:
        sc = blk[:, 74:82]
        lo, hi = sc & 15, sc >> 4
        db = d[:, :, None] * (0.5 + np.stack([lo, hi], 2)) * 0.25  # (R, 8, 2) per 16
        return np.repeat(db.reshape(-1, 16), 16, axis=1).astype(np.float32)
    if tid == TYPES["IQ3_XXS"]:
        aux = blk[:, 66:98].copy().view(np.uint32).reshape(-1, 8)
        db = d * (0.5 + (aux >> 28)) * 0.5
        return np.repeat(db, 32, axis=1).astype(np.float32)
    raise ValueError(tid)


def encode(tid, blk, codes, signs):
    """write grid codes / sign bytes into blk (R, rb), keeping d and the sub-block scales.
    IQ2_XXS / IQ2_S: codes (R, 32) grid index per 8 columns, signs (R, 32) 8-bit pattern.
    IQ3_XXS: codes (R, 64) grid index per 4 columns, signs (R, 32) 8-bit pattern (even parity)."""
    blk = blk.copy()
    if tid == TYPES["IQ2_XXS"]:
        q = blk[:, 2:66].copy().view(np.uint32).reshape(-1, 8, 2)
        c = codes.reshape(-1, 8, 4).astype(np.uint32)
        q[:, :, 0] = c[..., 0] | c[..., 1] << 8 | c[..., 2] << 16 | c[..., 3] << 24
        s = (signs.reshape(-1, 8, 4) & 127).astype(np.uint32)
        q[:, :, 1] = (q[:, :, 1] & 0xF0000000) | s[..., 0] | s[..., 1] << 7 | s[..., 2] << 14 | s[..., 3] << 21
        blk[:, 2:66] = q.reshape(-1, 16).view(np.uint8)
    elif tid == TYPES["IQ2_S"]:
        c = codes.astype(np.int64)
        blk[:, 2:34] = (c & 255).astype(np.uint8)
        blk[:, 34:66] = signs.astype(np.uint8)
        hi = (c >> 8).reshape(-1, 8, 4)
        blk[:, 66:74] = (hi[..., 0] | hi[..., 1] << 2 | hi[..., 2] << 4 | hi[..., 3] << 6).astype(np.uint8)
    elif tid == TYPES["IQ3_XXS"]:
        blk[:, 2:66] = codes.astype(np.uint8)
        q = blk[:, 66:98].copy().view(np.uint32).reshape(-1, 8)
        s = (signs.reshape(-1, 8, 4) & 127).astype(np.uint32)
        q[:] = (q & 0xF0000000) | s[..., 0] | s[..., 1] << 7 | s[..., 2] << 14 | s[..., 3] << 21
        blk[:, 66:98] = q.view(np.uint8)
    return blk


# ---------------------------------------------------------------- core
def choose(grid, a, t, P, parity):
    """grid (G, k) magnitudes; a (R, k) = sign * s * db per column; t (R, k) scaled target; P (k, k).
    Returns code index (R,), chosen sign pattern (R, k) in +-1, quantized scaled values (R, k).
    With parity, rows with an odd number of negative signs try each single sign flip (9 variants)."""
    R, k = a.shape
    GG = (grid[:, :, None] * grid[:, None, :]).reshape(grid.shape[0], k * k)
    Pt = t @ P
    sg = torch.sign(a); sg[sg == 0] = 1
    if parity:
        odd = ((sg < 0).sum(1) % 2) == 1
        var = [torch.ones(k, device=a.device)] + [1 - 2 * torch.eye(k, device=a.device)[i] for i in range(k)]
    else:
        var = [torch.ones(k, device=a.device)]
    best = torch.full((R,), float("inf"), device=a.device); bi = torch.zeros(R, dtype=torch.long, device=a.device)
    bv = torch.zeros(R, dtype=torch.long, device=a.device)
    for vi, f in enumerate(var):
        av = a * f
        A = (av[:, :, None] * av[:, None, :] * P).reshape(R, k * k)
        cost = A @ GG.T - 2 * (av * Pt) @ grid.T
        c, i = cost.min(1)
        if parity:
            ok = odd if vi > 0 else ~odd
            c = torch.where(ok, c, torch.full_like(c, float("inf")))
        upd = c < best
        best = torch.where(upd, c, best); bi = torch.where(upd, i, bi); bv = torch.where(upd, torch.full_like(bv, vi), bv)
    fl = torch.stack(var)[bv]                       # (R, k)
    sgn = sg * fl
    return bi, sgn, a * fl * grid[bi]


def gptq_layer(tid, W, SQ, U, imat, dev, chunk):
    """W (E, n_out, n_in) float32 numpy, SQ (E, n_in) column scales, U upper Cholesky of H^-1 (n_in, n_in) torch.
    Returns quantized bytes (E, n_out, n_in/256*rb) and the scaled error matrix for the proxy loss."""
    E, n_out, n_in = W.shape
    rb = G.ggml_row_size(tid, BLK)
    out = np.empty((E, n_out, n_in // BLK, rb), np.uint8)
    if tid == TYPES["IQ3_XXS"]:
        grid, k, gpb = torch.tensor(GRID3, device=dev), 4, 64
    else:
        grid, k, gpb = torch.tensor(GRID2XXS if tid == TYPES["IQ2_XXS"] else GRID2S, device=dev), 8, 32
    parity = tid != TYPES["IQ2_S"]
    pool = ThreadPoolExecutor(6)
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk); ne = e1 - e0
        sq = torch.tensor(SQ[e0:e1], device=dev)                                   # (ne, n_in)
        Wt = torch.tensor(W[e0:e1], device=dev)
        T = (Wt * sq[:, None, :]).reshape(ne * n_out, n_in)                     # scaled targets
        rows_sq = sq.repeat_interleave(n_out, 0)                                # (R, n_in)
        for kb, st in enumerate(range(0, n_in, BLK)):
            en = st + BLK
            Tb = (T[:, st:en] / rows_sq[:, st:en]).reshape(ne, n_out, BLK).cpu().numpy().astype(np.float32)
            buf = np.empty((ne, n_out * rb), np.uint8)

            def q1(i):
                x = np.ascontiguousarray(Tb[i]); im = np.ascontiguousarray(imat[e0 + i, st:en], np.float32)
                G.ggml_quantize_chunk(tid, x.ctypes.data, buf[i].ctypes.data, 0, n_out, BLK, im.ctypes.data)
            list(pool.map(q1, range(ne)))
            blk = buf.reshape(ne * n_out, rb)
            db = torch.tensor(parse_db(tid, blk), device=dev)                    # (R, 256)
            if tid == TYPES["IQ3_XXS"]:
                # signs over 8 columns with even parity, chosen up front (stock rule: flip the smallest t^2)
                tb = T[:, st:en].reshape(-1, 32, 8)
                sg = torch.sign(tb); sg[sg == 0] = 1
                odd = ((sg < 0).sum(2) % 2) == 1
                imin = (tb * tb).argmin(2)
                flip = torch.nn.functional.one_hot(imin, 8).bool() & odd[..., None]
                sg = torch.where(flip, -sg, sg).reshape(-1, BLK)
            codes = torch.zeros(T.shape[0], gpb, dtype=torch.long, device=dev)
            sgn_all = torch.ones(T.shape[0], BLK, device=dev)
            Err = torch.zeros(T.shape[0], BLK, device=dev)
            for g in range(gpb):
                c0, c1 = st + g * k, st + (g + 1) * k
                Ug = U[c0:c1, c0:c1]
                P = torch.linalg.inv(Ug.T @ Ug)
                t = T[:, c0:c1]
                s_db = rows_sq[:, c0:c1] * db[:, c0 - st:c1 - st]
                if tid == TYPES["IQ3_XXS"]:
                    a = sg[:, c0 - st:c1 - st] * s_db
                    bi, sgn, q = choose(grid, a, t, P, False)
                else:
                    a = torch.where(t >= 0, 1.0, -1.0) * s_db
                    bi, sgn, q = choose(grid, a, t, P, parity)
                codes[:, g] = bi; sgn_all[:, c0 - st:c1 - st] = sgn
                for j in range(c0, c1):            # column-wise GPTQ update inside the block (q of this group fixed)
                    err = (T[:, j] - q[:, j - c0]) / U[j, j]
                    Err[:, j - st] = err
                    if j + 1 < en:
                        T[:, j + 1:en] -= err[:, None] * U[j, j + 1:en][None, :]
                T[:, c0:c1] = q
            if en < n_in:
                T[:, en:] -= Err @ U[st:en, en:]
            neg = (sgn_all < 0).reshape(-1, BLK // 8, 8).to(torch.uint8)
            pat = (neg * (2 ** torch.arange(8, device=dev, dtype=torch.uint8))).sum(2).to(torch.uint8)
            enc = encode(tid, blk, codes.cpu().numpy(), pat.cpu().numpy())
            # check: ggml's dequantizer must reproduce the values we optimized
            if kb == 0 and e0 == 0:
                deq = to_float(tid, enc[:n_out].reshape(-1), n_out * BLK).reshape(n_out, BLK)
                mine = (T[:n_out, st:en] / rows_sq[:n_out, st:en]).cpu().numpy()
                assert np.allclose(deq, mine, rtol=1e-3, atol=1e-6), np.abs(deq - mine).max()
            out[e0:e1, :, kb, :] = enc.reshape(ne, n_out, rb)
        print(f"  experts {e0}-{e1 - 1} done", flush=True)
    return out.reshape(E, n_out, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", required=True, choices=["IQ2_XXS", "IQ2_S", "IQ3_XXS"])
    ap.add_argument("--layers", required=True)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--out", default="gptq_vq")
    ap.add_argument("--experts", type=int, default=0, help="only the first N experts, report proxy loss (test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=16, help="experts per GPU batch")
    ap.add_argument("--cov", default="covg")
    ap.add_argument("--imatrix", default="imatrix_q8.gguf")
    a = ap.parse_args()
    tid = TYPES[a.type]; dev = torch.device(a.device)
    layers = []
    for part in a.layers.split(","):
        lo, _, hi = part.partition("-"); layers += list(range(int(lo), int(hi or lo) + 1))
    src = "../cmp/Inkling-2-Mini-UD-Q8_K_XL.gguf"
    rs = {t.name: t for t in GGUFReader(src).tensors}
    ims = {t.name: np.asarray(t.data) for t in GGUFReader(a.imatrix).tensors}
    os.makedirs(a.out, exist_ok=True)
    for l in layers:
        name = f"blk.{l}.ffn_gate_up_exps.weight"
        outf = f"{a.out}/gu_{a.type}_{l}.gguf"
        if os.path.exists(outf) and not a.experts: print(f"{outf} exists"); continue
        t0 = time.time()
        t = rs[name]; n_in, n_out, n_exp = (int(v) for v in t.shape)
        ne = a.experts or n_exp
        raw = np.asarray(t.data).reshape(n_exp, -1)
        W = np.stack([to_float(int(t.tensor_type), raw[e], n_in * n_out).reshape(n_out, n_in) for e in range(ne)])
        s2 = ims[name + ".in_sum2"].reshape(n_exp, -1).astype(np.float64)
        cnt = ims[name + ".counts"].reshape(n_exp).astype(np.float64)
        cov = np.fromfile(f"{a.cov}/{name}.cov", np.float32)
        nc = int(round(np.sqrt(cov.size))); cov = cov.reshape(nc, nc).astype(np.float64)
        gd = np.diag(cov).copy()
        d = np.where(cnt[:, None] > 0, s2 / np.maximum(cnt, 1)[:, None], np.pad(gd, (0, n_in - nc))[None, :])[:ne]
        imat = np.maximum(d, 1e-8 * np.maximum(d.max(1, keepdims=True), 1e-30)).astype(np.float32)
        SQ = np.sqrt(np.maximum(d, 1e-6 * d.max(1, keepdims=True))).astype(np.float32)
        dg = np.sqrt(np.maximum(gd, 1e-30))
        C = np.eye(n_in); C[:nc, :nc] = cov / np.outer(dg, dg)
        Cd = torch.tensor(C + a.damp * np.eye(n_in), dtype=torch.float64)
        U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Cd)), upper=True).float().to(dev)
        data = gptq_layer(tid, W, SQ, U, imat, dev, a.chunk)
        if a.experts:
            Cu = torch.tensor(C, dtype=torch.float32)
            tot = 0.0
            for e in range(ne):
                Q = to_float(tid, data[e].reshape(-1), n_in * n_out).reshape(n_out, n_in)
                Ee = torch.tensor((Q - W[e]) * SQ[e])
                tot += float(torch.einsum("ij,jk,ik->", Ee, Cu, Ee))
            print(f"{name} {a.type}: proxy loss over {ne} experts {tot:.6g}  ({time.time() - t0:.0f}s, test, not written)")
            continue
        w = gguf.GGUFWriter(outf + ".tmp", "inkling2")
        w.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType(tid))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        os.replace(outf + ".tmp", outf)
        print(f"{outf}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
