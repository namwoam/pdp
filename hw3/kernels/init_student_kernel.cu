#include "../math_utils.h"
#include <stdio.h>

// =============================================================================
// HW4: CUDA Matmul Optimization
//
// Implement your optimized single-precision GEMM here:
//     C = alpha * (A @ B) + beta * C      (row-major, M = N = K per test size)
//
// Hard rules (violation => 0 points for the performance / rank components):
//   1. DO NOT call cuBLAS, cuDNN, CUTLASS, or any vendor GEMM library.
//   2. DO NOT modify any file outside kernels/student_kernel.cu.
//   3. The signature of runStudent() MUST remain unchanged — TA's grading
//      harness calls it directly.
//
// Your kernel is verified against the cuBLAS reference (tolerance 1e-2) for
// all sizes in {128, 256, 512, 1024, 2048, 4096}. Failing any size zeroes out
// the performance and rank components of the grade.
// =============================================================================

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              float *A, float *B, float beta, float *C) {
    // TODO: implement your kernel.
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    // TODO: configure grid/block dimensions and launch StudentKernel.
}
