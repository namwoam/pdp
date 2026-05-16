#include "../math_utils.h"
#include <stdio.h>

// Tiled SGEMM with shared memory blocking.
// Each 32x32 thread block computes a 32x32 output tile of C.
// Iterates over K in strips of width BK=32.
// C = alpha * (A @ B) + beta * C   (row-major)

#define BM 128
#define BN 128
#define BK  8
#define TM  8
#define TN  8

// 256 threads per block (BM/TM * BN/TN = 16*16)
__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Which output tile this block owns
    const int row_block = blockIdx.y * BM;
    const int col_block = blockIdx.x * BN;

    // Thread's position within the block tile (16x16 arrangement)
    const int thread_row = threadIdx.x / (BN / TN); // 0..15
    const int thread_col = threadIdx.x % (BN / TN); // 0..15

    // Accumulators for TM x TN output sub-tile per thread
    float c[TM][TN] = {};

    // Number of threads per block
    const int num_threads = (BM / TM) * (BN / TN); // 256

    // Load strides for As and Bs
    // As: BM rows x BK cols -> each thread loads (BM*BK)/num_threads elements
    const int A_stride = num_threads / BK; // rows loaded per iter = 256/8 = 32
    const int B_stride = num_threads / BN; // rows loaded per iter = 256/128 = 2

    for (int k_base = 0; k_base < K; k_base += BK) {
        // Load As: BM x BK from A[row_block..row_block+BM][k_base..k_base+BK]
        for (int load_row = 0; load_row < BM; load_row += A_stride) {
            int r = load_row + threadIdx.x / BK;
            int c_idx = threadIdx.x % BK;
            int global_row = row_block + r;
            int global_col = k_base + c_idx;
            As[r][c_idx] = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col] : 0.0f;
        }

        // Load Bs: BK x BN from B[k_base..k_base+BK][col_block..col_block+BN]
        for (int load_row = 0; load_row < BK; load_row += B_stride) {
            int r = load_row + threadIdx.x / BN;
            int c_idx = threadIdx.x % BN;
            int global_row = k_base + r;
            int global_col = col_block + c_idx;
            Bs[r][c_idx] = (global_row < K && global_col < N)
                           ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        // Compute TM x TN partial products
        for (int k = 0; k < BK; k++) {
            float a[TM], b[TN];
            for (int m = 0; m < TM; m++)
                a[m] = As[thread_row * TM + m][k];
            for (int n = 0; n < TN; n++)
                b[n] = Bs[k][thread_col * TN + n];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    c[m][n] += a[m] * b[n];
        }

        __syncthreads();
    }

    // Write results back to C with alpha/beta scaling
    for (int m = 0; m < TM; m++) {
        int global_row = row_block + thread_row * TM + m;
        for (int n = 0; n < TN; n++) {
            int global_col = col_block + thread_col * TN + n;
            if (global_row < M && global_col < N) {
                int idx = global_row * N + global_col;
                C[idx] = alpha * c[m][n] + beta * C[idx];
            }
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block((BM / TM) * (BN / TN)); // 256 threads, 1-D
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
