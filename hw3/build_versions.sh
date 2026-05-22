#!/usr/bin/env bash
# Build main1..main6, each compiled against a different student kernel version.
# kernels/student_kernel.cu is swapped per build and restored on exit.
set -euo pipefail

NVCC=${NVCC:-nvcc}
NVCC_FLAGS="-O3 -std=c++14"
ARCH=70
LDFLAGS="-lcublas"

VERSIONS=(
    kernels/v1_naive.cu
    kernels/v2_smem_tiling.cu
    kernels/v3_1d_blocktiling.cu
    kernels/v4_2d_blocktiling.cu
    kernels/v5_vectorized.cu
    kernels/v6_warp_tiling.cu
)

STUDENT=kernels/student_kernel.cu
BACKUP=${STUDENT}.orig

cp "$STUDENT" "$BACKUP"
trap 'cp "$BACKUP" "$STUDENT"; rm -f "$BACKUP"' EXIT

for i in "${!VERSIONS[@]}"; do
    src="${VERSIONS[$i]}"
    num=$((i + 1))
    target="main${num}"
    printf '\n[%d/6]  %-38s  →  %s\n' "$num" "$src" "$target"
    cp "$src" "$STUDENT"
    $NVCC $NVCC_FLAGS \
        -gencode=arch=compute_${ARCH},code=sm_${ARCH} \
        main.cu -o "$target" $LDFLAGS
    echo "       built $target"
done

echo -e "\nDone: main1 .. main6"
