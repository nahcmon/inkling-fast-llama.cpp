#!/bin/bash
# measure every line of a plan file (name<TAB>assemble args) with cand.sh
cd "$(dirname "$0")"
while IFS=$'\t' read -r n rest; do
  eval "args=($rest)"
  ./cand.sh "$n" "${args[@]}" > /dev/null 2>&1 || echo "$n failed"
  grep "^$n	" kld.tsv | cut -f1,3,5
done < "$1"
echo "round done $(date +%T)"
