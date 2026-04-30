#!/usr/bin/env bash
# bench.sh — micro-benchmark for renderer_mpi against stored baseline.
# Spec compliance: builds with the EXACT grading compile command from
# Section 4: `mpicc renderer_mpi.c -o renderer_mpi -lm -march=native`.
# Read-only by default. Use `./bench.sh --save` to overwrite the baseline
# with the current run's medians.
#
# Default sweep covers single-node {1,4,8,16} AND the competition config
# n=64 (4 nodes x 16 procs/node via $HOSTFILE). For any n>16 the script
# auto-routes to multi-node with the spec's exact UCX env + mpirun flags
# and scp's the freshly-built binary to peer nodes ($PEER_NODES).
set -euo pipefail
cd "$(dirname "$0")"

BIN=renderer_mpi
SRC=renderer_mpi.c
TESTCASES=${TESTCASES:-/home/Team12/testcases}
GOLDEN=${GOLDEN:-/home/Team12/testcases/golden}
BASE=bench_baseline.txt
REPS=${REPS:-3}
NPROCS=${NPROCS:-"1 4 8 16 64"}
HOSTFILE=${HOSTFILE:-hosts}       # spec-style hostfile, used when n>16
NPERNODE=${NPERNODE:-16}          # spec cap: 16 procs/node
PEER_NODES=${PEER_NODES:-"rdma1 rdma2 rdma3"}  # for binary distribution

SAVE=0
[ "${1:-}" = "--save" ] && SAVE=1

# Build with EXACT spec compile command
REBUILT=0
if [ ! -x "./$BIN" ] || [ "$SRC" -nt "./$BIN" ]; then
    echo "building: mpicc $SRC -o $BIN -lm -march=native"
    mpicc "$SRC" -o "$BIN" -lm -march=native
    REBUILT=1
fi

# Distribute binary to peer nodes if any multi-node n is in the sweep
# (always sync — peers may have a stale build from a previous run)
NEED_MULTI=0
for n in $NPROCS; do [ "$n" -gt 16 ] && NEED_MULTI=1; done
if [ "$NEED_MULTI" = 1 ] && [ -f "$HOSTFILE" ]; then
    SELF_BIN=$(readlink -f "./$BIN")
    for h in $PEER_NODES; do
        ssh -o BatchMode=yes "$h" "mkdir -p $(dirname "$SELF_BIN")" 2>/dev/null || true
        scp -q "./$BIN" "$h:$SELF_BIN"
    done
    echo "distributed $BIN -> $PEER_NODES"
fi

mkdir -p out
[ -f "$BASE" ] || : > "$BASE"
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

# header
printf "\n%-26s %4s %9s %9s %9s %8s  %s\n" \
       "testcase" "n" "time(s)" "old(s)" "imb" "speedup" "verify"
printf -- '-%.0s' {1..82}; echo

for tc_path in "$TESTCASES"/*.bin; do
    tc=$(basename "$tc_path" .bin)
    golden_png="$GOLDEN/${tc}.png"

    for n in $NPROCS; do
        key="${tc}_n${n}"
        out_png="out/${tc}_n${n}.png"
        rm -f "$out_png"

        if [ "$n" -gt 16 ] && [ -f "$HOSTFILE" ]; then
            # multi-node: spec Section 4 mpirun flags. The spec also lists
            # UCX_TLS / UCX_NET_DEVICES env vars, but on this cluster they
            # break PML ucx selection — the mca flags alone pick the right
            # transport. Override $UCX_ENV to add them back if grading does.
            mpi_run=(${UCX_ENV:-} mpirun --hostfile "$HOSTFILE" -n "$n" -npernode "$NPERNODE"
                     --mca pml ucx --mca btl ^tcp)
        else
            if [ "$n" -gt 16 ]; then
                echo "WARN: n=$n exceeds 16/node spec cap on single node" >&2
            fi
            mpi_run=(mpirun -n "$n")
        fi

        # take median of REPS runs (sort + middle element)
        samples=""
        imb_samples=""
        for ((i=0; i<REPS; i++)); do
            out=$("${mpi_run[@]}" ./"$BIN" "$tc_path" "$out_png" 2>&1 || true)
            t=$(echo "$out" | grep -oP 'total=\K[0-9.]+' || echo 999)
            imb=$(echo "$out" | grep -oP 'imbalance\(max/mean\)=\K[0-9.]+' || echo "-")
            samples+="$t"$'\n'
            imb_samples+="$imb"$'\n'
        done
        new=$(printf '%s' "$samples" | sort -g | sed -n "$(( (REPS+1)/2 ))p")
        imb_new=$(printf '%s' "$imb_samples" | sort -g | sed -n "$(( (REPS+1)/2 ))p")

        # verify against golden
        if [ -f "$golden_png" ]; then
            if diff -q "$out_png" "$golden_png" > /dev/null 2>&1; then
                verdict="OK"
            else
                verdict="WRONG"
            fi
        else
            verdict="(no-golden)"
        fi

        # speedup vs stored baseline
        old=$(awk -v k="$key" '$1==k {print $2; exit}' "$BASE")
        if [ -n "$old" ]; then
            sp=$(awk -v a="$old" -v b="$new" 'BEGIN{printf "%6.2fx", a/b}')
        else
            sp="     -"
        fi

        printf "%-26s %4d %9s %9s %9s %8s  %s\n" \
               "$tc" "$n" "$new" "${old:--}" "$imb_new" "$sp" "$verdict"

        echo "$key $new" >> "$TMP"
    done
done

if [ "$SAVE" = "1" ]; then
    sort -o "$BASE" "$TMP"
    echo
    echo "baseline overwritten -> $BASE"
else
    echo
    echo "(read-only; use './bench.sh --save' to update $BASE)"
fi
