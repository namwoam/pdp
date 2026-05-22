#include "../math_utils.h"

// Vectorized float4: all global loads and the C write-back use 128-bit
// instructions, saturating the memory bus.  BK grows to 16 so each
// thread issues exactly one float4 load for A and one for B per K-strip.
// TN drops to 4 to match the float4 store width for C.
//
// BM=BN=128, BK=16, TM=8, TN=4 → NUM_THREADS=512.

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 4
#define NUM_THREADS ((BM * BN) / (TM * TN))  // 512

__global__ __launch_bounds__(NUM_THREADS)
void StudentKernel(int M, int N, int K, float alpha,
                   const float * __restrict__ A, const float * __restrict__ B,
                   float beta, float * __restrict__ C) {
    __shared__ float As[BK * BM];  // transposed: As[k][m]
    __shared__ float Bs[BK * BN];  // row-major:  Bs[k][n]

    // Compute indices
    const int thread_row = threadIdx.x / (BN / TN);  // 0..15
    const int thread_col = threadIdx.x % (BN / TN);  // 0..31

    // Load indices — each thread issues one float4 for A and one for B
    const int as_inner_row = threadIdx.x / (BK / 4);  // 0..127
    const int as_inner_col = threadIdx.x % (BK / 4);  // 0..3
    const int bs_inner_row = threadIdx.x / (BN / 4);  // 0..15
    const int bs_inner_col = threadIdx.x % (BN / 4);  // 0..31

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[TM * TN] = {};
    float reg_A[TM];
    float reg_B[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile with float4, store transposed into As
        float4 a4 = *reinterpret_cast<const float4 *>(
            &A[as_inner_row * K + as_inner_col * 4]);
        As[(as_inner_col * 4 + 0) * BM + as_inner_row] = a4.x;
        As[(as_inner_col * 4 + 1) * BM + as_inner_row] = a4.y;
        As[(as_inner_col * 4 + 2) * BM + as_inner_row] = a4.z;
        As[(as_inner_col * 4 + 3) * BM + as_inner_row] = a4.w;

        // Load B tile with float4 (row-major, no transpose)
        *reinterpret_cast<float4 *>(&Bs[bs_inner_row * BN + bs_inner_col * 4]) =
            *reinterpret_cast<const float4 *>(&B[bs_inner_row * N + bs_inner_col * 4]);

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

    // Write back with float4 (TN == 4 == one float4)
    for (int m = 0; m < TM; ++m) {
        float4 c4 = *reinterpret_cast<float4 *>(
            &C[(thread_row * TM + m) * N + thread_col * TN]);
        int base = m * TN;
        c4.x = alpha * acc[base + 0] + beta * c4.x;
        c4.y = alpha * acc[base + 1] + beta * c4.y;
        c4.z = alpha * acc[base + 2] + beta * c4.z;
        c4.w = alpha * acc[base + 3] + beta * c4.w;
        *reinterpret_cast<float4 *>(
            &C[(thread_row * TM + m) * N + thread_col * TN]) = c4;
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(NUM_THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
