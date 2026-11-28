"""Bake the AtomicChat uncensored rank-1 LoRA into the Unsloth Q8_K_XL layout.

python merge_lora.py SRC LORA OUT
  attn_output, token_embd  -> F32 (base + B.A)
  ffn_down_exps            -> F16 padded to PAD rows, per-expert NVFP4 scale baked in, + B.A (verified against
                              AtomicChat's merged Q8_0: the delta applies to the scaled weight)
  ffn_down_exps.scale      -> dropped; inkling2.expert_down_pad key added (as down_pad.py prep)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import dequantize

PAD = 1280
KEY = "inkling2.expert_down_pad"
src, lora_p, dst = sys.argv[1:4]
r = GGUFReader(src)
lora = {t.name: np.asarray(t.data, np.float32) for t in GGUFReader(lora_p).tensors}
scales = {t.name: np.array(t.data, np.float32) for t in r.tensors if t.name.endswith("ffn_down_exps.scale")}
used = set()

def AB(name):
    a, b = lora.get(name + ".lora_a"), lora.get(name + ".lora_b")
    if a is None: return None
    used.add(name); return a, b

def deq(t):
    return dequantize(np.asarray(t.data), t.tensor_type).reshape([int(x) for x in reversed(t.shape)]).astype(np.float32)

w = GGUFWriter(dst, r.fields[gguf.Keys.General.ARCHITECTURE].contents())
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."): continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=st)
w.add_key_value(KEY, PAD, gguf.GGUFValueType.UINT32)
w.add_key_value("general.uncensored_lora", os.path.basename(lora_p), gguf.GGUFValueType.STRING)

keep = [t for t in r.tensors if not t.name.endswith("ffn_down_exps.scale")]
plan = {}
for t in keep:
    n = t.name
    if n.endswith("ffn_down_exps.weight"):
        shp = (int(t.shape[2]), int(t.shape[1]), PAD); plan[n] = "down"
        w.add_tensor_info(n, shp, np.dtype(np.float16), int(np.prod(shp)) * 2, Q.F16)
    elif n + ".lora_a" in lora:
        shp = tuple(int(x) for x in reversed(t.shape)); plan[n] = "f32"
        w.add_tensor_info(n, shp, np.dtype(np.float32), int(np.prod(shp)) * 4, Q.F32)
    else:
        w.add_tensor_info(n, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    n = t.name; p = plan.get(n)
    if p == "down":
        x = deq(t); s = scales.get(n.replace(".weight", ".scale"))   # (E, n_embd, n_ff)
        if s is not None: x *= s.reshape(-1, 1, 1)
        a, b = AB(n)
        E = x.shape[0]
        d = np.einsum("eo,ei->eoi", b.reshape(E, -1), a.reshape(E, -1))
        rel = float(np.linalg.norm(d) / np.linalg.norm(x)); x += d
        out = np.zeros((E, x.shape[1], PAD), np.float16); out[:, :, : x.shape[2]] = x
        print(n, "delta/W %.3e" % rel, "absmax", float(np.abs(out).max()), flush=True)
        w.write_tensor_data(out); del x, d, out
    elif p == "f32":
        x = deq(t); a, b = AB(n)
        if n == "token_embd.weight":     # get_rows: row t += b * a[t]
            d = a.reshape(-1, 1) * b.reshape(1, -1)
        else:
            d = b.reshape(-1, 1) @ a.reshape(1, -1)
        rel = float(np.linalg.norm(d) / np.linalg.norm(x)); x += d
        print(n, x.shape, "delta/W %.3e" % rel, flush=True)
        w.write_tensor_data(x); del x, d
    else:
        w.write_tensor_data(t.data)
w.close()
miss = {k.rsplit(".lora_", 1)[0] for k in lora} - used
print("LoRA sites used", len(used), "missing", sorted(miss))
