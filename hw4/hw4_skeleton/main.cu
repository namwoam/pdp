#include <stdio.h>
#include <vector>
#include <random>
using namespace std;

#include "cuda_utils.h"
#include "matrix_utils.h"
#include "kernels/runner.cu"

int main(int argc, char* argv[]) {
    int selected_kernel = 1;
    if (argc > 1) {
        selected_kernel = atoi(argv[1]);
        if (selected_kernel < 0) {
            fprintf(stderr, "error: please no negative kernel id\n");
            return 1;
        }
    }
    // print device name prop
    CudaDeviceInfo();
    // CudaDeviceInfoDetailed();
    // create array on cpu
    vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

    long m, n, k, max_size;
    max_size = SIZE[SIZE.size() - 1]; // SIZE[-1]
    printf("Max size: %ld\n", max_size);
    printf("Selected kernel: %d\n", selected_kernel);

    // c = alpha (a @ b) + beta C
    float alpha = 0.5, beta = 3.0;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr;
    
    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    // randomize matrix
    randomize_matrix(A, max_size * max_size);
    // printf("ori: ");
    // for (int i = 0; i < 16; i++) {
    //     printf("%f ", A[i]);
    // } printf("\n");

    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(float) * max_size * max_size));
    CUDA_CHECK(cudaMalloc((void**)&dB, sizeof(float) * max_size * max_size));
    CUDA_CHECK(cudaMalloc((void**)&dC, sizeof(float) * max_size * max_size));
    CUDA_CHECK(cudaMalloc((void**)&dC_ref, sizeof(float) * max_size * max_size));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size, cudaMemcpyHostToDevice));

    int repeat_times = 50;
    float elapsed_time;
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    // launch kernel (also pass in ptr)
    // modify this block later
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);

    for (int size : SIZE) {
        printf("Running size: %d... \t", size);

        m = n = k = size;
        // run once for:
        // 1. verification
        // 2. warm up
        run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle);
        run_kernel(selected_kernel, m, n, k, alpha, dA, dB, beta, dC, handle);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // move result c gpu back to cpu (only m*n elements)
        CUDA_CHECK(cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
        
        if (!verify_matrix(C_ref, C, m * n)) {
            printf("verification failed\n");
            return 1;
        }

        // actual record time
        CUDA_CHECK(cudaEventRecord(beg));
        for (int j = 0; j < repeat_times; j++) {
            run_kernel(selected_kernel, m, n, k, alpha, dA, dB, beta, dC, handle);
        }
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(beg));
        CUDA_CHECK(cudaEventSynchronize(end));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, beg, end));
        elapsed_time /= 1000; // into seconds

        double flops = 2.0 * (double)m * n * k;
        double avg_time = elapsed_time / repeat_times;
        double gflops = (double)repeat_times * flops / elapsed_time / 1e9;
        printf("avg time: %7.6fs, performance: %7.3f GFLOPS\n",
               avg_time, gflops);
        fflush(stdout);
        // c = alpha AB + beta C, so we stabilize c again (only m*n elements)
        CUDA_CHECK(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
    }
    
    // if (stat != CUBLAS_STATUS_SUCCESS) { 
    //     /* handle error */ 
    //     printf("cublas failed\n");
    //     exit(1);
    // }
    cublasDestroy(handle);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dC_ref));

    return 0;
}