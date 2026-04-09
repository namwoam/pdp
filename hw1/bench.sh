#!/usr/bin/env bash
# bench.sh — micro-benchmark for spmv_openmp against stored baseline.
# Read-only by default. Use `./bench.sh --save` to overwrite the baseline
# with the current run's medians.
set -euo pipefail
cd "$(dirname "$0")"

BIN=spmv_openmp
BASE=bench_baseline.txt
REPS=5
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}  # spec caps at 16/team/node

SAVE=0
[ "${1:-}" = "--save" ] && SAVE=1

if [ ! -x "./$BIN" ] || [ "${BIN}.c" -nt "./$BIN" ]; then
    echo "building: gcc -fopenmp -O3 ${BIN}.c -o $BIN"
    gcc -fopenmp -O3 "${BIN}.c" -o "$BIN"
fi

[ -f "$BASE" ] || : > "$BASE"
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

printf "\n%-30s %10s %10s %9s  %s\n" "testcase" "old(ms)" "new(ms)" "speedup" "verify"
printf -- '-%.0s' {1..73}; echo

for mtx in testcases/*.mtx; do
    tc=$(basename "$mtx" .mtx)
    vec=testcases/${tc}.vec

    samples=""
    verdict="?"
    for ((i=0; i<REPS; i++)); do
        out=$(./"$BIN" "$mtx" "$vec" 2>&1)
        t=$(printf '%s\n' "$out" | sed -n 's/.*time_ms=\([0-9.]*\).*/\1/p')
        v=$(printf '%s\n' "$out" | grep -oE 'OK|WRONG' | head -n1 || true)
        [ -n "${v:-}" ] && verdict=$v
        samples+="$t"$'\n'
    done
    new=$(printf '%s' "$samples" | sort -g | sed -n "$(( (REPS+1)/2 ))p")

    old=$(awk -v k="$tc" '$1==k {print $2; exit}' "$BASE")
    if [ -n "$old" ]; then
        sp=$(awk -v a="$old" -v b="$new" 'BEGIN{printf "%6.2fx", a/b}')
    else
        sp="     -"
    fi
    printf "%-30s %10s %10s %9s  %s\n" "$tc" "${old:--}" "$new" "$sp" "$verdict"

    echo "$tc $new" >> "$TMP"
done

if [ "$SAVE" = "1" ]; then
    sort -o "$BASE" "$TMP"
    echo
    echo "baseline overwritten -> $BASE"
else
    echo
    echo "(read-only; use './bench.sh --save' to overwrite $BASE)"
fi
