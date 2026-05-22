#include "../math_utils.h"

// Shared-memory tiling: cooperative tile load into SMEM, then compute.
// One thread per output element; BM=BN=BK=32 fills 1024 threads (the max).

#define BM 32
#define BN 32
#define BK 32

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;

    float sum = 0.f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        // Each thread loads one element of each tile.
        As[threadIdx.y * BK + threadIdx.x] =
            (row < M && k0 + threadIdx.x < K) ? A[row * K + k0 + threadIdx.x] : 0.f;
        Bs[threadIdx.y * BN + threadIdx.x] =
            (k0 + threadIdx.y < K && col < N) ? B[(k0 + threadIdx.y) * N + col] : 0.f;
        __syncthreads();

        for (int k = 0; k < BK; ++k)
            sum += As[threadIdx.y * BK + k] * Bs[k * BN + threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(BN, BM);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
