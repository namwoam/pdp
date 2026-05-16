#include <stdio.h>
#include <cublas_v2.h>

void runCublasFP32(int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C,
                   cublasHandle_t handle) {
    // cuBLAS uses column-major order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // This runs cuBLAS in full fp32 mode
    
    cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        N, M, K, 
        &alpha, 
        B, CUDA_R_32F, N, // B pointer, leading dim = N (cols in row-major)
        A, CUDA_R_32F, K, // A pointer, leading dim = K
        &beta, 
        C, CUDA_R_32F, N, // C pointer, leading dim = N
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
