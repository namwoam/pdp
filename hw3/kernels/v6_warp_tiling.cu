#include "../math_utils.h"

// Warp tiling: organise the 4 warps in a 2×2 grid, each owning
// WM×WN=64×64.  Within a warp, each thread sweeps WNITER=4 sub-columns
// (WSUBN=16 each) using TM×TN=8×4 accumulators — improving register
// reuse and hiding ALU latency without software pipelining.
//
// Same parameters as the final kernel; double buffering is the only
// difference between this version and student_kernel.cu.
//
// BM=BN=128, BK=16, WM=WN=64, WNITER=4, TM=8, TN=4 → NUM_THREADS=128.

#define BM 128
#define BN 128
#define BK 16
#define WM 64
#define WN 64
#define WNITER 4
#define TM 8
#define TN 4
#define NUM_THREADS 128

#define WSUBN  (WN / WNITER)                          // 16
#define WMITER ((WM * WN) / (32 * TM * TN * WNITER))  // 1
#define WSUBM  (WM / WMITER)                           // 64

#define A_LOAD_ITERS (BM * BK / (NUM_THREADS * 4))    // 4
#define B_LOAD_ITERS (BK * BN / (NUM_THREADS * 4))    // 4

__global__ __launch_bounds__(NUM_THREADS)
void StudentKernel(int M, int N, int K, float alpha,
                   const float * __restrict__ A, const float * __restrict__ B,
                   float beta, float * __restrict__ C) {
    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];

    const int warp_id  = threadIdx.x / 32;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    const int lane       = threadIdx.x % 32;
    const int thread_row = lane / (WSUBN / TN);
    const int thread_col = lane % (WSUBN / TN);

    constexpr int rowStrideA = NUM_THREADS / (BK / 4);  // 32
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);  // 4
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float acc[WMITER * TM * WNITER * TN] = {};
    float reg_A[WMITER * TM];
    float reg_B[WNITER * TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load tiles with float4
        #pragma unroll
        for (int i = 0; i < A_LOAD_ITERS; ++i) {
            float4 t = *reinterpret_cast<const float4 *>(
                &A[(innerRowA + i * rowStrideA) * K + innerColA * 4]);
            As[(innerColA * 4 + 0) * BM + innerRowA + i * rowStrideA] = t.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + i * rowStrideA] = t.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + i * rowStrideA] = t.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + i * rowStrideA] = t.w;
        }
        #pragma unroll
        for (int i = 0; i < B_LOAD_ITERS; ++i)
            *reinterpret_cast<float4 *>(
                &Bs[(innerRowB + i * rowStrideB) * BN + innerColB * 4]) =
            *reinterpret_cast<const float4 *>(
                &B[(innerRowB + i * rowStrideB) * N + innerColB * 4]);
        A += BK;
        B += BK * N;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WMITER; ++wm)
                #pragma unroll
                for (int m = 0; m < TM; ++m)
                    reg_A[wm * TM + m] =
                        As[k * BM + warp_row * WM + wm * WSUBM + thread_row * TM + m];
            #pragma unroll
            for (int wn = 0; wn < WNITER; ++wn)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    reg_B[wn * TN + n] =
                        Bs[k * BN + warp_col * WN + wn * WSUBN + thread_col * TN + n];
            #pragma unroll
            for (int wm = 0; wm < WMITER; ++wm)
                #pragma unroll
                for (int wn = 0; wn < WNITER; ++wn)
                    #pragma unroll
                    for (int m = 0; m < TM; ++m)
                        #pragma unroll
                        for (int n = 0; n < TN; ++n)
                            acc[(wm * TM + m) * (WNITER * TN) + wn * TN + n] +=
                                reg_A[wm * TM + m] * reg_B[wn * TN + n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wm = 0; wm < WMITER; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WNITER; ++wn) {
            float *C_sub = C + (warp_row * WM + wm * WSUBM) * N
                             + (warp_col * WN + wn * WSUBN);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int row = thread_row * TM + m;
                int col = thread_col * TN;
                float4 t = *reinterpret_cast<float4 *>(&C_sub[row * N + col]);
                int base = (wm * TM + m) * (WNITER * TN) + wn * TN;
                t.x = alpha * acc[base + 0] + beta * t.x;
                t.y = alpha * acc[base + 1] + beta * t.y;
                t.z = alpha * acc[base + 2] + beta * t.z;
                t.w = alpha * acc[base + 3] + beta * t.w;
                *reinterpret_cast<float4 *>(&C_sub[row * N + col]) = t;
            }
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(NUM_THREADS);
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
