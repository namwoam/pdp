#include "../math_utils.h"

// Naive: one thread computes one C[row][col].
// Each thread walks the entire K dimension from global memory.

#define BLOCK 16

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));
    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
