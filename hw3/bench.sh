#!/usr/bin/env bash
# bench.sh — run ./main for each kernel id, dump raw stdout for the
# figure generator (scripts/generate_report_assets.py) to parse.
# Read-only; just records what main.cu already prints per size.
set -euo pipefail
cd "$(dirname "$0")"

BIN=main
KERNELS=${KERNELS:-"0 1"}
OUT_DIR=data
SRUN=${SRUN:-""}   # set SRUN="srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 2" on the cluster

if [ ! -x "./$BIN" ] || [ kernels/student_kernel.cu -nt "./$BIN" ]; then
    echo "building via task build..."
    task build
fi

mkdir -p "$OUT_DIR"

printf "\n%-10s %6s %12s %12s\n" "kernel" "size" "time(s)" "GFLOPS"
printf -- '-%.0s' {1..46}; echo

for k in $KERNELS; do
    case $k in
        0) name=cublas ;;
        1) name=student ;;
        *) name="kernel_$k" ;;
    esac
    raw="$OUT_DIR/raw_${name}.txt"
    $SRUN ./"$BIN" "$k" | tee "$raw" >/dev/null

    # Pretty per-size summary on stdout.
    grep -E '^Running size:' "$raw" | while read -r line; do
        size=$(echo "$line" | sed -n 's/.*size: \([0-9]\+\).*/\1/p')
        t=$(echo "$line" | sed -n 's/.*avg time: \([0-9.]*\)s.*/\1/p')
        g=$(echo "$line" | sed -n 's/.*performance: *\([0-9.]*\) GFLOPS.*/\1/p')
        printf "%-10s %6s %12s %12s\n" "$name" "$size" "$t" "$g"
    done
done

echo
echo "raw outputs -> $OUT_DIR/raw_*.txt"
echo "run 'poetry run python scripts/generate_report_assets.py' to refresh figures"
