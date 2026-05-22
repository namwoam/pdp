#include "../math_utils.h"

// 2-D block tiling: each thread computes a TM×TN register micro-tile.
// A is stored transposed in SMEM (As[k][m]) so the inner loop reads
// As with stride 1 along m — a broadcast across the warp with no conflicts.
//
// BM=BN=128, BK=8, TM=TN=8 → NUM_THREADS=256.
// Each thread issues 4 scalar loads for As and 2 for Bs per K-strip.

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM * BN) / (TM * TN))  // 256

__global__ __launch_bounds__(NUM_THREADS)
void StudentKernel(int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
    __shared__ float As[BK * BM];  // transposed: As[k][m]
    __shared__ float Bs[BK * BN];  // row-major:  Bs[k][n]

    // Compute indices
    const int thread_row = threadIdx.x / (BN / TN);  // 0..15
    const int thread_col = threadIdx.x % (BN / TN);  // 0..15

    // Load indices — each thread loads multiple elements
    constexpr int as_row_stride = NUM_THREADS / BK;   // 32
    constexpr int bs_row_stride = NUM_THREADS / BN;   // 2
    const int as_inner_row = threadIdx.x / BK;        // 0..31
    const int as_inner_col = threadIdx.x % BK;        // 0..7
    const int bs_inner_row = threadIdx.x / BN;        // 0..1
    const int bs_inner_col = threadIdx.x % BN;        // 0..127

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {};
    float reg_A[TM];
    float reg_B[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load As transposed so reading along m is contiguous
        for (int off = 0; off < BM; off += as_row_stride)
            As[as_inner_col * BM + as_inner_row + off] =
                A[(as_inner_row + off) * K + as_inner_col];
        for (int off = 0; off < BK; off += bs_row_stride)
            Bs[(bs_inner_row + off) * BN + bs_inner_col] =
                B[(bs_inner_row + off) * N + bs_inner_col];
        A += BK;
        B += BK * N;
        __syncthreads();

        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m)
                reg_A[m] = As[k * BM + thread_row * TM + m];
            for (int n = 0; n < TN; ++n)
                reg_B[n] = Bs[k * BN + thread_col * TN + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    acc[m * TN + n] += reg_A[m] * reg_B[n];
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
            C[(thread_row * TM + m) * N + thread_col * TN + n] =
                alpha * acc[m * TN + n] +
                beta  * C[(thread_row * TM + m) * N + thread_col * TN + n];
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(NUM_THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
