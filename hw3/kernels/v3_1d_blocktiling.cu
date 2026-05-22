#include "../math_utils.h"

// 1-D block tiling: each thread computes a vertical strip of TM output
// elements in one column.  Increases register reuse along M.
//
// BM=64, BN=64, BK=8, TM=8 → NUM_THREADS=512.
// As/Bs are each 512 elements → one scalar load per thread per K-strip.

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define NUM_THREADS (BM * BN / TM)  // 512

__global__ __launch_bounds__(NUM_THREADS)
void StudentKernel(int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
    __shared__ float As[BM * BK];  // row-major: As[m][k]
    __shared__ float Bs[BK * BN];  // row-major: Bs[k][n]

    // Load indices (one element each)
    const int as_row = threadIdx.x / BK;   // 0..63
    const int as_col = threadIdx.x % BK;   // 0..7
    const int bs_row = threadIdx.x / BN;   // 0..7
    const int bs_col = threadIdx.x % BN;   // 0..63

    // Compute indices: each thread owns TM rows at a fixed column
    const int thread_row = threadIdx.x / BN;  // 0..7
    const int thread_col = threadIdx.x % BN;  // 0..63

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        As[as_row * BK + as_col] = A[as_row * K + as_col];
        Bs[bs_row * BN + bs_col] = B[bs_row * N + bs_col];
        A += BK;
        B += BK * N;
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            float b = Bs[k * BN + thread_col];
            for (int m = 0; m < TM; ++m)
                acc[m] += As[(thread_row * TM + m) * BK + k] * b;
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m)
        C[(thread_row * TM + m) * N + thread_col] =
            alpha * acc[m] + beta * C[(thread_row * TM + m) * N + thread_col];
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(NUM_THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
