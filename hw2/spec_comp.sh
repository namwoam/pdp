#!/usr/bin/env bash
set -euo pipefail

# Compile (linking math library and enabling CPU-specific optimizations)
mpicc renderer_mpi.c -o renderer_mpi -lm -march=native
# Copy binary to other nodes (example for nodes rdma1..rdma3)
for i in 1 2 3; do scp renderer_mpi rdma$i:~/; done
# Run on 4 nodes with 16 processes per node

TESTCASES=${TESTCASES:-/home/Team12/testcases}
GOLDEN=${GOLDEN:-/home/Team12/testcases/golden}
UCX_TLS_VAL=${UCX_TLS_VAL:-rc,tcp,sm,self}
UCX_NET_DEVICES_VAL=${UCX_NET_DEVICES_VAL:-}
UCX_LOG_LEVEL_VAL=${UCX_LOG_LEVEL_VAL:-warn}

for tc_path in "$TESTCASES"/*.bin; do
    tc=$(basename "$tc_path" .bin)
    golden_png="$GOLDEN/${tc}.png"
    out_png="out/${tc}.png"

    mpi_cmd=(
        mpirun
        --hostfile hosts
        -npernode 16
        --mca pml ucx
        --mca btl ^openib,tcp
        --mca btl_base_warn_component_unused 0
        ./renderer_mpi "$tc_path" "$out_png"
    )

    ucx_env=(
        "UCX_TLS=$UCX_TLS_VAL"
        "UCX_LOG_LEVEL=$UCX_LOG_LEVEL_VAL"
    )
    if [ -n "$UCX_NET_DEVICES_VAL" ]; then
        ucx_env+=("UCX_NET_DEVICES=$UCX_NET_DEVICES_VAL")
    fi

    env "${ucx_env[@]}" "${mpi_cmd[@]}"

    if [ -f "$golden_png" ]; then
        if diff -q "$out_png" "$golden_png" > /dev/null 2>&1; then
            verdict="OK"
        else
            verdict="WRONG"
        fi
    else
        verdict="(no-golden)"
    fi
    printf "%s\n" "$verdict"
done