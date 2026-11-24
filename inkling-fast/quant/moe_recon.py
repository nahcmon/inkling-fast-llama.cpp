"""MoE-block output reconstruction: tune the fp16 super-block scales d of the quantized expert tensors (grid codes
frozen) so that each layer's whole MoE output (routing, gate_up, SITU, down, weighted sum) matches the reference model
on real layer inputs. Unlike the per-matmul GPTQ proxy this sees the activation nonlinearity, the routing weights and
how gate/up/down errors combine. One layer at a time: ~4 GB of GPU memory.

Inputs: dump/<blk.L.ffn_gate_up_exps.weight>.x16 (f16 MoE inputs of all loop passes) from llama-imatrix with
IMAT_DUMP_DIR, the reference expert weights (Q8_K_XL gate_up, down_f16.gguf down) and the candidate GGUF to tune.

python moe_recon.py --layers 0-23 --cand ../Inkling-2-Mini-fast-9.2-gptq.gguf --out recon [--check]
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
# types whose whole 256-value super-block is (fp16 d in bytes 0..1) x (values fixed by the codes)
D_SCALED = {TYPES[k] for k in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS") if k in TYPES}
ONE = np.frombuffer(np.float16(1.0).tobytes(), np.uint8)


def deq(t):
    n_in, n_out, n_exp = (int(v) for v in t.shape)
    raw = np.asarray(t.data).reshape(n_exp, -1)
    return np.stack([to_float(int(t.tensor_type), raw[e], n_in * n_out).reshape(n_out, n_in) for e in range(n_exp)])


def split_d(t):
    """quantized expert tensor -> B (E, rows, cols) f16 with every d set to 1, d (E, rows, nblk) f32"""
    tid = int(t.tensor_type); n_in, n_out, n_exp = (int(v) for v in t.shape)
    rb = G.ggml_row_size(tid, BLK); nb = n_in // BLK
    raw = np.asarray(t.data).reshape(n_exp, n_out, nb, rb)
    d = raw[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
    r1 = raw.copy(); r1[..., :2] = ONE
    B = np.stack([to_float(tid, r1[e].reshape(-1), n_in * n_out).reshape(n_out, n_in) for e in range(n_exp)])
    assert np.array_equal(B.astype(np.float16).astype(np.float32), B)
    return B.astype(np.float16), d


def put_d(t, d_new):
    tid = int(t.tensor_type); n_in, n_out, n_exp = (int(v) for v in t.shape)
    rb = G.ggml_row_size(tid, BLK)
    raw = np.asarray(t.data).reshape(n_exp, n_out, n_in // BLK, rb).copy()
    raw[..., :2] = d_new.astype(np.float16)[..., None].view(np.uint8)
    return raw.reshape(n_exp, n_out, -1)


class Layer:
    def __init__(self, hp, router, bias, dev):
        self.hp, self.dev = hp, dev
        self.Wr = torch.tensor(router, dtype=torch.float32, device=dev)            # (E, n_embd)
        self.b = torch.tensor(bias, dtype=torch.float32, device=dev) if bias is not None else None

    def route(self, X):
        p = torch.sigmoid(X @ self.Wr.T)
        s = p + self.b if self.b is not None else p
        ids = torch.topk(s, self.hp["k"], dim=1).indices
        w = p.gather(1, ids)
        w = w / (w.sum(1, keepdim=True) + 1e-20) * self.hp["wscale"]
        return ids, w

    def act(self, h):
        nf = h.shape[1] // 2
        g, u = h[:, :nf], h[:, nf:]
        a = self.hp["beta"] * torch.tanh(g / self.hp["beta"]) * torch.sigmoid(g)
        lb = self.hp["lbeta"]
        return a * (lb * torch.tanh(u / lb) if lb > 0 else u)

    def moe(self, X, ids, w, expert):
        """expert(e, xe) -> (c, n_embd) unweighted output of expert e"""
        y = torch.zeros_like(X)
        flat = ids.reshape(-1)
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=self.Wr.shape[0]).tolist()
        tok = order // ids.shape[1]; ws = w.reshape(-1)[order]
        st = 0
        for e, c in enumerate(counts):
            if c == 0: continue
            sel = tok[st:st + c]
            y = y.index_add(0, sel, expert(e, X[sel]) * ws[st:st + c, None])
            st += c
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--cand", default="../Inkling-2-Mini-fast-9.2-gptq.gguf")
    ap.add_argument("--dump", default="dump")
    ap.add_argument("--out", default="recon")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="only compare the reference MoE with llama.cpp's dumped expert inputs")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    layers = []
    for part in a.layers.split(","):
        lo, _, hi = part.partition("-"); layers += list(range(int(lo), int(hi or lo) + 1))
    rc = GGUFReader(a.cand)
    f = rc.fields
    kv = lambda k: float(f[k].parts[f[k].data[0]][0])
    hp = dict(k=int(kv("inkling2.expert_used_count")), wscale=kv("inkling2.expert_weights_scale"),
              beta=kv("inkling2.activation.situ_beta"), lbeta=kv("inkling2.activation.situ_linear_beta"))
    cand = {t.name: t for t in rc.tensors}
    ref = {t.name: t for t in GGUFReader("../cmp/Inkling-2-Mini-UD-Q8_K_XL.gguf").tensors}
    refd = {t.name: t for t in GGUFReader("down_f16.gguf").tensors}
    os.makedirs(a.out, exist_ok=True)
    for l in layers:
        t0 = time.time()
        gu, dn = f"blk.{l}.ffn_gate_up_exps.weight", f"blk.{l}.ffn_down_exps.weight"
        X = np.fromfile(f"{a.dump}/{gu}.x16", np.float16)
        n_embd = int(cand[gu].shape[0]); X = X.reshape(-1, n_embd)
        if a.max_tokens: X = X[:a.max_tokens]
        rt = cand[f"blk.{l}.ffn_gate_inp.weight"]
        router = to_float(int(rt.tensor_type), np.asarray(rt.data).reshape(-1), int(rt.shape[0]) * int(rt.shape[1])).reshape(int(rt.shape[1]), int(rt.shape[0]))
        bt = cand.get(f"blk.{l}.exp_probs_b.bias")
        lay = Layer(hp, router, np.asarray(bt.data).astype(np.float32) if bt is not None else None, dev)
        sc_t = ref.get(f"blk.{l}.ffn_gate_up_exps.scale")
        sc = torch.tensor(np.asarray(sc_t.data), dtype=torch.float32, device=dev) if sc_t is not None else None
        sc_c = cand.get(f"blk.{l}.ffn_gate_up_exps.scale")
        sc_s = torch.tensor(np.asarray(sc_c.data), dtype=torch.float32, device=dev) if sc_c is not None else None

        # ---- reference outputs
        Wgu = torch.tensor(deq(ref[gu]), device=dev)                               # (E, 2nf, n_embd) f32
        Wdn = torch.tensor(deq(refd[dn]).astype(np.float32), device=dev)          # (E, n_embd, n_down)
        if f"blk.{l}.ffn_down_exps.scale" in ref and refd[dn].tensor_type.name not in ("F16", "F32", "BF16"):
            raise SystemExit("down reference must be the scaled f16 export")
        n_down = Wdn.shape[2]

        def ref_expert(e, xe):
            h = xe @ Wgu[e].T
            if sc is not None: h = h * sc[e]
            return F.pad(lay.act(h), (0, n_down - h.shape[1] // 2)) @ Wdn[e].T

        if a.check:
            chk = f"{a.dump}/{dn}.chk"
            hdr = np.fromfile(chk, np.int32, 3); nt, nid, ne0 = (int(v) for v in hdr)
            ids_c = np.fromfile(chk, np.int32, nt * nid, offset=12).reshape(nt, nid)
            xs_c = np.fromfile(chk, np.float32, nt * nid * ne0, offset=12 + 4 * nt * nid).reshape(nt, nid, ne0)
            Xt = torch.tensor(X[:1].astype(np.float32), device=dev)
            ids, w = lay.route(Xt)
            same = sorted(ids[0].tolist()) == sorted(ids_c[0].tolist())
            errs = []
            for j, e in enumerate(ids_c[0]):
                h = Xt @ Wgu[e].T
                if sc is not None: h = h * sc[e]
                mine = F.pad(lay.act(h), (0, ne0 - h.shape[1] // 2))[0].cpu().numpy()
                errs.append(np.linalg.norm(mine - xs_c[0, j]) / max(np.linalg.norm(xs_c[0, j]), 1e-30))
            print(f"layer {l}: routing ids match {same} (llama {sorted(ids_c[0].tolist())} mine {sorted(ids[0].tolist())}), "
                  f"expert input rel err max {max(errs):.2e}")
            del Wgu, Wdn; torch.cuda.empty_cache(); continue

        N = X.shape[0]; nh = int(N * a.holdout)
        Xs = torch.tensor(X.astype(np.float32))                                     # CPU
        Y = torch.empty_like(Xs); IDS = torch.empty(N, hp["k"], dtype=torch.long); WS = torch.empty(N, hp["k"])
        with torch.no_grad():
            for s in range(0, N, a.batch):
                xb = Xs[s:s + a.batch].to(dev)
                ids, w = lay.route(xb)
                Y[s:s + a.batch] = lay.moe(xb, ids, w, ref_expert).cpu()
                IDS[s:s + a.batch] = ids.cpu(); WS[s:s + a.batch] = w.cpu()
        del Wgu, Wdn; torch.cuda.empty_cache()

        # ---- candidate: B (codes) frozen, log-multipliers on d trainable
        tgu, tdn = cand[gu], cand[dn]
        parts = {}
        for key, t in (("gu", tgu), ("dn", tdn)):
            if int(t.tensor_type) in D_SCALED:
                B, d = split_d(t)
                parts[key] = dict(B=torch.tensor(B, device=dev), d=torch.tensor(d, device=dev),
                                  m=torch.zeros(d.shape, device=dev, requires_grad=True))
            else:
                W = deq(t).astype(np.float32)
                parts[key] = dict(W=torch.tensor(W, device=dev))
                print(f"  {t.name}: {t.tensor_type.name} has no single block scale, kept frozen")
        n_down_c = int(tdn.shape[0])

        def wmat(p, e):
            if "W" in p: return p["W"][e]
            return p["B"][e].float() * (p["d"][e] * torch.exp(p["m"][e])).repeat_interleave(BLK, dim=1)

        def cand_expert_fn(e):
            def fn(xe, *ms):
                h = xe @ wmat(parts["gu"], e).T
                if sc_s is not None: h = h * sc_s[e]
                return F.pad(lay.act(h), (0, n_down_c - h.shape[1] // 2)) @ wmat(parts["dn"], e).T
            return fn

        def cand_expert(e, xe):
            ms = [p["m"] for p in parts.values() if "m" in p]
            return checkpoint(cand_expert_fn(e), xe, *ms, use_reentrant=False)

        params = [p["m"] for p in parts.values() if "m" in p]
        if not params: print(f"layer {l}: nothing to tune"); continue
        opt = torch.optim.Adam(params, lr=a.lr)

        def evaluate(lo, hi):
            num = den = 0.0
            with torch.no_grad():
                for s in range(lo, hi, a.batch):
                    e_ = min(hi, s + a.batch)
                    xb, yb = Xs[s:e_].to(dev), Y[s:e_].to(dev)
                    ys = lay.moe(xb, IDS[s:e_].to(dev), WS[s:e_].to(dev), cand_expert)
                    num += float(((ys - yb) ** 2).sum()); den += float((yb ** 2).sum())
            return num / den

        ntr = N - nh
        base_h = evaluate(ntr, N); base_t = evaluate(0, min(ntr, 4 * a.batch))
        best = (base_h, [p.detach().clone() for p in params], 0)
        print(f"layer {l}: {N} rows ({nh} held out), start rel err train {base_t:.4e} held-out {base_h:.4e}", flush=True)
        g = torch.Generator().manual_seed(l)
        for ep in range(1, a.epochs + 1):
            perm = torch.randperm(ntr, generator=g)
            for s in range(0, ntr, a.batch):
                idx = perm[s:s + a.batch]
                xb, yb = Xs[idx].to(dev), Y[idx].to(dev)
                ys = lay.moe(xb, IDS[idx].to(dev), WS[idx].to(dev), cand_expert)
                loss = ((ys - yb) ** 2).sum() / (yb ** 2).sum()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            h = evaluate(ntr, N)
            if h < best[0]: best = (h, [p.detach().clone() for p in params], ep)
            print(f"  epoch {ep}: held-out {h:.4e} ({100 * (h / base_h - 1):+.2f}%)", flush=True)
        h, ms, ep = best
        print(f"layer {l}: best epoch {ep}, held-out {base_h:.4e} -> {h:.4e} ({100 * (h / base_h - 1):+.2f}%), {time.time() - t0:.0f}s")
        if ep == 0: continue
        w = gguf.GGUFWriter(f"{a.out}/recon_{l}.gguf.tmp", "inkling2")
        i = 0
        for key, t in (("gu", tgu), ("dn", tdn)):
            p = parts[key]
            if "m" not in p: continue
            d_new = (p["d"] * torch.exp(ms[i])).cpu().numpy(); i += 1
            w.add_tensor(t.name, put_d(t, d_new), raw_dtype=gguf.GGMLQuantizationType(int(t.tensor_type)))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        os.replace(f"{a.out}/recon_{l}.gguf.tmp", f"{a.out}/recon_{l}.gguf")


if __name__ == "__main__":
    main()
