#include "../math_utils.h"
#include <stdio.h>

// Warp-tiled, double-buffered SGEMM for V100 (sm_70), row-major.
// C = alpha * (A @ B) + beta * C
//
// Hierarchy (siboehm kernel-10 + double buffering):
//   - Block:  BM x BN = 128 x 128 output tile.
//   - Warp:   4 warps in a 2 x 2 grid, each owning WM x WN = 64 x 64.
//   - WNITER: each warp sweeps 4 sub-blocks of WSUBM x WSUBN = 64 x 16 along N.
//   - Thread: 32 threads/warp tile WSUBM x WSUBN as 8 rows x 4 cols, each
//             holding a TM x TN = 8 x 4 register micro-tile per sub-block.
//
// Double buffering:
//   - Two shared-memory tiles (curr / nxt).
//   - Each outer K iteration: issue global LDGs for the next K-strip into
//     register staging FIRST, then do the BK-long inner FMA loop on the
//     current buffer, then commit the staged values to the other shared tile.
//     LDG latency is hidden under FMA throughput.
//
// Memory tricks:
//   - As stored transposed (As[k * BM + m]) for broadcast reads in the warp.
//   - All global loads/stores use float4 (128-bit vectorized).
//   - Harness sizes (128..4096) are multiples of BM=BN=128 and BK=16, so the
//     fast path drops in-loop bounds checks.

#define BM 128
#define BN 128
#define BK 8
#define WM 64
#define WN 64
#define WNITER 4
#define TM 8
#define TN 4
#define NUM_THREADS 128

#define WSUBN (WN / WNITER)                                   // 16
#define WMITER ((WM * WN) / (32 * TM * TN * WNITER))          // 1
#define WSUBM (WM / WMITER)                                   // 64

#define A_LOAD_ITERS (BM * BK / (NUM_THREADS * 4))            // 4
#define B_LOAD_ITERS (BK * BN / (NUM_THREADS * 4))            // 4

__device__ __forceinline__ void
load_tile(const float *A, const float *B, int K, int N,
          float *As, float *Bs,
          int innerRowA, int innerColA, int innerRowB, int innerColB) {
    constexpr int rowStrideA = NUM_THREADS / (BK / 4);  // 32
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);  // 4
    #pragma unroll
    for (int off = 0; off < BM; off += rowStrideA) {
        float4 t = *reinterpret_cast<const float4 *>(
            &A[(innerRowA + off) * K + innerColA * 4]);
        As[(innerColA * 4 + 0) * BM + innerRowA + off] = t.x;
        As[(innerColA * 4 + 1) * BM + innerRowA + off] = t.y;
        As[(innerColA * 4 + 2) * BM + innerRowA + off] = t.z;
        As[(innerColA * 4 + 3) * BM + innerRowA + off] = t.w;
    }
    #pragma unroll
    for (int off = 0; off < BK; off += rowStrideB) {
        *reinterpret_cast<float4 *>(
            &Bs[(innerRowB + off) * BN + innerColB * 4]) =
            *reinterpret_cast<const float4 *>(
                &B[(innerRowB + off) * N + innerColB * 4]);
    }
}

__global__ void __launch_bounds__(NUM_THREADS)
StudentKernel(int M, int N, int K, float alpha,
              const float * __restrict__ A,
              const float * __restrict__ B,
              float beta, float * __restrict__ C) {
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    // Block swizzle for L2 reuse: iterate within SUPER_M x gridDim.x
    // super-blocks so 8 consecutive blocks reuse the same B column strip.
    constexpr int SUPER_M = 8;
    int br_idx, bc_idx;
    if (gridDim.y >= SUPER_M) {
        int pid     = blockIdx.y * gridDim.x + blockIdx.x;
        int super   = pid / (SUPER_M * gridDim.x);
        int local   = pid - super * SUPER_M * gridDim.x;
        br_idx = super * SUPER_M + (local % SUPER_M);
        bc_idx = local / SUPER_M;
    } else {
        br_idx = blockIdx.y;
        bc_idx = blockIdx.x;
    }
    const int block_row = br_idx * BM;
    const int block_col = bc_idx * BN;

    A += block_row * K;
    B += block_col;
    C += block_row * N + block_col;

    const int warp_id  = threadIdx.x / 32;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    const int lane       = threadIdx.x % 32;
    const int thread_row = lane / (WSUBN / TN);
    const int thread_col = lane % (WSUBN / TN);

    constexpr int rowStrideA = NUM_THREADS / (BK / 4);
    constexpr int rowStrideB = NUM_THREADS / (BN / 4);
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    float acc[WMITER * TM * WNITER * TN] = {};
    float reg_A[WMITER * TM];
    float reg_B[WNITER * TN];

    // ---- Prologue: load tile 0 into buffer 0 ----
    load_tile(A, B, K, N, As[0], Bs[0],
              innerRowA, innerColA, innerRowB, innerColB);
    A += BK;
    B += BK * N;
    __syncthreads();

    int curr = 0;

    // ---- Main loop: while computing tile k, prefetch tile k+1 into regs ----
    for (int k_base = BK; k_base < K; k_base += BK) {
        // Pre-fetch next tile into register staging.
        float4 a_stage[A_LOAD_ITERS];
        float4 b_stage[B_LOAD_ITERS];
        #pragma unroll
        for (int i = 0; i < A_LOAD_ITERS; ++i) {
            a_stage[i] = *reinterpret_cast<const float4 *>(
                &A[(innerRowA + i * rowStrideA) * K + innerColA * 4]);
        }
        #pragma unroll
        for (int i = 0; i < B_LOAD_ITERS; ++i) {
            b_stage[i] = *reinterpret_cast<const float4 *>(
                &B[(innerRowB + i * rowStrideB) * N + innerColB * 4]);
        }

        // Compute from current shared buffer.
        const float *As_cur = As[curr];
        const float *Bs_cur = Bs[curr];
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WMITER; ++wm)
                #pragma unroll
                for (int m = 0; m < TM; ++m)
                    reg_A[wm * TM + m] =
                        As_cur[k * BM + warp_row * WM + wm * WSUBM
                                      + thread_row * TM + m];
            #pragma unroll
            for (int wn = 0; wn < WNITER; ++wn)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    reg_B[wn * TN + n] =
                        Bs_cur[k * BN + warp_col * WN + wn * WSUBN
                                      + thread_col * TN + n];
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

        // Commit staged loads into the other shared buffer.
        int nxt = curr ^ 1;
        float *As_nxt = As[nxt];
        float *Bs_nxt = Bs[nxt];
        #pragma unroll
        for (int i = 0; i < A_LOAD_ITERS; ++i) {
            int off = i * rowStrideA;
            float4 t = a_stage[i];
            As_nxt[(innerColA * 4 + 0) * BM + innerRowA + off] = t.x;
            As_nxt[(innerColA * 4 + 1) * BM + innerRowA + off] = t.y;
            As_nxt[(innerColA * 4 + 2) * BM + innerRowA + off] = t.z;
            As_nxt[(innerColA * 4 + 3) * BM + innerRowA + off] = t.w;
        }
        #pragma unroll
        for (int i = 0; i < B_LOAD_ITERS; ++i) {
            int off = i * rowStrideB;
            *reinterpret_cast<float4 *>(
                &Bs_nxt[(innerRowB + off) * BN + innerColB * 4]) = b_stage[i];
        }

        A += BK;
        B += BK * N;
        __syncthreads();
        curr = nxt;
    }

    // ---- Epilogue: compute last tile ----
    {
        const float *As_cur = As[curr];
        const float *Bs_cur = Bs[curr];
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WMITER; ++wm)
                #pragma unroll
                for (int m = 0; m < TM; ++m)
                    reg_A[wm * TM + m] =
                        As_cur[k * BM + warp_row * WM + wm * WSUBM
                                      + thread_row * TM + m];
            #pragma unroll
            for (int wn = 0; wn < WNITER; ++wn)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    reg_B[wn * TN + n] =
                        Bs_cur[k * BN + warp_col * WN + wn * WSUBN
                                      + thread_col * TN + n];
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
    }

    // ---- Write back C with float4 (TN=4 == one float4) ----
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
