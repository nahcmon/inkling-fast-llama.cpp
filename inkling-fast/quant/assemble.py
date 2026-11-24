"""Assemble a GGUF from a base file, taking chosen tensors from other GGUFs and converting exact-BF16 F32 tensors.

python assemble.py BASE OUT [--bf16 REGEX]... [--take REGEX=FILE]...
  --bf16 REGEX      F32 tensors matching REGEX are stored as BF16, but only if every value is exactly BF16 (lossless)
  --take REGEX=FILE tensors matching REGEX are copied byte-for-byte from FILE (first matching rule wins)
  --add REGEX=FILE  tensors of FILE matching REGEX that BASE lacks are appended (e.g. the .scale of NVFP4 experts)
  --drop REGEX      tensors matching REGEX are left out (e.g. a .scale whose weight was replaced)
"""
import sys, re, argparse
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as Q

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("out")
ap.add_argument("--bf16", action="append", default=[])
ap.add_argument("--take", action="append", default=[])
ap.add_argument("--add", action="append", default=[])
ap.add_argument("--drop", action="append", default=[])
ap.add_argument("--kv", action="append", default=[], help="PREFIX=FILE: copy metadata keys starting with PREFIX from FILE")
a = ap.parse_args()

r = GGUFReader(a.base)
srcs = {}
rules = []
for t in a.take:
    rx, f = t.split("=", 1)
    if f not in srcs:
        srcs[f] = {x.name: x for x in GGUFReader(f).tensors}
    rules.append((re.compile(rx), srcs[f]))

w = GGUFWriter(a.out, r.fields[gguf.Keys.General.ARCHITECTURE].contents())
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    w.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)

for kv in a.kv:
    pre, f = kv.split("=", 1)
    for fld in GGUFReader(f).fields.values():
        if fld.name.startswith(pre):
            vt = fld.types[0]
            w.add_key_value(fld.name, fld.contents(), vt, sub_type=fld.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)

plan = []  # (name, data, type, shape-for-info)
have = {t.name for t in r.tensors}
extra = []
for t in a.add:
    rx, f = t.split("=", 1)
    rx = re.compile(rx)
    extra += [x for x in GGUFReader(f).tensors if rx.search(x.name) and x.name not in have]
    have |= {x.name for x in extra}
for t in list(r.tensors) + extra:
    if any(re.search(x, t.name) for x in a.drop):
        continue
    src = t
    for rx, pool in rules:
        if rx.search(t.name):
            src = pool.get(t.name, t)
            break
    data, ty = src.data, src.tensor_type
    if ty == Q.F32 and any(re.search(x, t.name) for x in a.bf16):
        x = np.ascontiguousarray(np.array(src.data, dtype=np.float32))
        u = x.view(np.uint32)
        if np.all((u & 0xFFFF) == 0):
            data, ty = (u >> 16).astype(np.uint16), Q.BF16
    plan.append((t.name, data, ty))

for n, d, ty in plan:
    w.add_tensor_info(n, d.shape, d.dtype, d.nbytes, ty)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
tot = 0
summary = {}
for n, d, ty in plan:
    w.write_tensor_data(d); tot += d.nbytes
    summary[ty.name] = summary.get(ty.name, 0) + d.nbytes
w.close()
print(f"{a.out}: {tot / 2**30:.3f} GiB tensor data")
for k, v in sorted(summary.items(), key=lambda x: -x[1]):
    print(f"  {k:8s} {v / 2**20:9.1f} MiB")
