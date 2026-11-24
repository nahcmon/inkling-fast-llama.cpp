#!/bin/bash
# cand.sh NAME [assemble.py args: --take REGEX=POOL --add REGEX=POOL --drop REGEX]...
# builds BASE (default fast-small) + swaps, KLD vs Q8_K_XL on the held-out eval text (80 x 512), deletes the file
# unless KEEP=1. Results go to kld.tsv (name, PPL, KLD, KLD99, top-1, ...). Size is logged to sizes.tsv.
set -e
cd "$(dirname "$0")"
BIN=${LLAMA_BIN:-$(cd ../../build-cuda/bin 2>/dev/null && pwd)}
KLD=$(dirname "$0")/kld.py
BASE=${BASE:-../Inkling-2-Mini-fast-small.gguf}
N=$1; shift
grep -q "^$N	" kld.tsv && { grep "^$N	" kld.tsv; exit 0; }
python assemble.py $BASE c_$N.gguf "$@" | head -1 > /dev/null
echo -e "$N\t$(stat -c %s c_$N.gguf)" >> sizes.tsv
python "$KLD" compare --exe "$BIN/llama-perplexity.exe" --model c_$N.gguf --logits ref_q8.bin --corpus ../cmp/eval.txt \
  --chunks 80 --name $N -- ${XARGS:--ngl 99 --fit off -fa on}
[ -n "$KEEP" ] || rm -f c_$N.gguf
