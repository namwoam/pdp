#pragma once
#include <stdio.h>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        return 1;                                               \
    }                                                           \
} while (0)

void CudaDeviceInfo() {
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("Using device: %s\n", props.name);
}

void CudaDeviceInfoDetailed() {
    int deviceId;
    cudaGetDevice(&deviceId);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    printf("=== CUDA Device Information ===\n\n");
    
    // Basic info
    printf("%-40s %s\n", "Name", props.name);
    printf("%-40s %d.%d\n", "Compute Capability", props.major, props.minor);
    
    // Thread/Warp limits
    printf("%-40s %d\n", "max threads per block", props.maxThreadsPerBlock);
    printf("%-40s %d\n", "max threads per multiprocessor", props.maxThreadsPerMultiProcessor);
    printf("%-40s %d\n", "threads per warp", props.warpSize);
    
    // Warp allocation (CC 8.x+ fields)
    #if CUDART_VERSION >= 11000
    if (props.major >= 8) {
        // printf("%-40s %d\n", "warp allocation granularity", props.warpAllocationGranularity);
    }
    #endif
    
    // Register limits
    printf("%-40s %d\n", "max regs per block", props.regsPerBlock);
    printf("%-40s %d\n", "max regs per multiprocessor", props.regsPerMultiprocessor);
    
    // Register allocation details (CC 8.x+)
    #if CUDART_VERSION >= 11000
    if (props.major >= 8) {
        // printf("%-40s %d\n", "reg allocation unit size", props.regAllocationUnitSize);
        // const char* granularityStr = (props.regAllocationGranularity == 0) ? "thread" : "warp";
        // printf("%-40s %s\n", "reg allocation granularity", granularityStr);
    }
    #endif
    
    // Memory info
    printf("%-40s %.0f MB\n", "total global mem", props.totalGlobalMem / (1024.0 * 1024.0));
    printf("%-40s %d KB\n", "max shared mem per block",(int)props.sharedMemPerBlock / 1024);
    
    // Shared mem overhead (runtime reserve - typically 1KB for CC 8.x)
    int smOverhead = (props.major >= 8) ? 1024 : 0;
    printf("%-40s %d B\n", "CUDA runtime shared mem overhead per block", smOverhead);
    
    printf("%-40s %d KB\n", "shared mem per multiprocessor", (int)props.sharedMemPerMultiprocessor / 1024);
    
    // Multiprocessor info
    printf("%-40s %d\n", "multiprocessor count", props.multiProcessorCount);
    // printf("%-40s %d\n", "max warps per multiprocessor", props.maxWarpsPerMultiprocessor);
    
    // Optional: occupancy helper info
    printf("\n=== Derived Metrics ===\n");
    int maxBlocksPerSM = props.maxThreadsPerMultiProcessor / 1024; // assuming 1024 threads/block
    printf("Max blocks per SM (est., 1024 threads/block): %d\n", maxBlocksPerSM);
    
    // Theoretical max occupancy for your kernel (example: 32 regs, 8KB smem)
    // Use cudaOccupancyMaxActiveBlocksPerMultiprocessor for accurate calc
}