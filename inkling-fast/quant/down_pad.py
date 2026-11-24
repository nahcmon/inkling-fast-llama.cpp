"""Expert-down padding for Inkling-2-Mini (1152 -> PAD rows so 256-block K/IQ quants can be used).

prep   SRC OUT_F16           : copy SRC, but every blk.N.ffn_down_exps -> F16 [PAD, n_embd, n_expert] with the
                               per-expert NVFP4 scale baked in and zero padding; drop .scale tensors; add the
                               inkling2.expert_down_pad key. Feed OUT_F16 to llama-quantize.
imat   IMAT_IN IMAT_OUT      : pad the ffn_down_exps entries of an imatrix.gguf to PAD (tiny weight on the pad).
splice SRC QUANT OUT         : SRC with its down tensors replaced by QUANT's (byte copy), scales dropped, key added.
"""
import sys, os
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q
from gguf.quants import dequantize

PAD = int(os.environ.get("PAD", "1280"))
KEY = "inkling2.expert_down_pad"
is_down = lambda n: n.endswith("ffn_down_exps.weight")
is_scale = lambda n: n.endswith("ffn_down_exps.scale")


def copy_kv(r, w, extra):
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF.") or f.name in extra:
            continue
        vt = f.types[0]
        st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    for k, (v, t) in extra.items():
        w.add_key_value(k, v, t)


def arch(r):
    f = r.fields[gguf.Keys.General.ARCHITECTURE]
    return f.contents()


def padded_down(r, t, scales):
    x = dequantize(t.data, t.tensor_type).astype(np.float32)  # numpy shape [n_expert, n_embd, n_ff]
    x = x.reshape(int(t.shape[2]), int(t.shape[1]), int(t.shape[0]))
    s = scales.get(t.name.replace(".weight", ".scale"))
    if s is not None:
        x *= s.reshape(-1, 1, 1)
    out = np.zeros((x.shape[0], x.shape[1], PAD), dtype=np.float16)
    out[:, :, : x.shape[2]] = x
    return out


def prep(src, dst):
    r = GGUFReader(src)
    scales = {t.name: np.array(t.data, dtype=np.float32) for t in r.tensors if is_scale(t.name)}
    w = GGUFWriter(dst, arch(r))
    copy_kv(r, w, {KEY: (PAD, gguf.GGUFValueType.UINT32)})
    keep = [t for t in r.tensors if not is_scale(t.name)]
    for t in keep:
        if is_down(t.name):
            shp = (int(t.shape[2]), int(t.shape[1]), PAD)
            w.add_tensor_info(t.name, shp, np.dtype(np.float16), int(np.prod(shp)) * 2, Q.F16)
        else:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in keep:
        if is_down(t.name):
            d = padded_down(r, t, scales)
            print(t.name, t.tensor_type.name, "->", d.shape, "absmax", float(np.abs(d).max()), flush=True)
            w.write_tensor_data(d)
        else:
            w.write_tensor_data(t.data)
    w.close()


def imat(src, dst):
    r = GGUFReader(src)
    w = GGUFWriter(dst, arch(r)) if gguf.Keys.General.ARCHITECTURE in r.fields else GGUFWriter(dst, "")
    copy_kv(r, w, {})
    datas = []
    for t in r.tensors:
        d = np.array(t.data)
        if "ffn_down_exps" in t.name and t.name.endswith("in_sum") and d.shape[-1] < PAD:
            p = np.zeros(d.shape[:-1] + (PAD,), dtype=d.dtype); p[..., : d.shape[-1]] = d; d = p
        if "ffn_down_exps" in t.name and t.name.endswith("in_sum2") and d.shape[-1] < PAD:
            p = np.zeros(d.shape[:-1] + (PAD,), dtype=d.dtype)
            p[..., : d.shape[-1]] = d
            p[..., d.shape[-1]:] = np.maximum(d.mean(axis=-1, keepdims=True) * 1e-3, 1e-12)
            print(t.name, d.shape, "->", p.shape)
            d = p
        datas.append((t.name, d, t.tensor_type))
    for n, d, ty in datas:
        w.add_tensor(n, d, raw_dtype=ty)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
    w.close()


def splice(src, quant, dst):
    r = GGUFReader(src); rq = GGUFReader(quant)
    qt = {t.name: t for t in rq.tensors if is_down(t.name)}
    w = GGUFWriter(dst, arch(r))
    copy_kv(r, w, {KEY: (PAD, gguf.GGUFValueType.UINT32)})
    keep = [qt.get(t.name, t) for t in r.tensors if not is_scale(t.name)]
    for t in keep:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    tot = 0
    for t in keep:
        w.write_tensor_data(t.data); tot += t.data.nbytes
    w.close()
    print(f"{dst}: {tot / 2**30:.3f} GiB tensor data; down types: {sorted({t.tensor_type.name for t in qt.values()})}")


if __name__ == "__main__":
    {"prep": prep, "imat": imat, "splice": splice}[sys.argv[1]](*sys.argv[2:])
