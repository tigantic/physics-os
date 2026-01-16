/**
 * Minimal cuSOLVER gesvdaStridedBatched test for Blackwell debugging
 * 
 * Compile: nvcc -arch=sm_120 -o test_cusolver test_cusolver_minimal.cu -lcudart -lcusolver
 * Run: ./test_cusolver
 */

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        printf("cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("=== cuSOLVER gesvdaStridedBatched Minimal Test ===\n\n");
    
    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("CUDA Runtime: %d.%d\n\n", CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
    
    // Small test case
    const int batch = 2;
    const int m = 16;
    const int n = 16;
    const int k = 8;  // rank
    
    printf("Test parameters:\n");
    printf("  batch=%d, m=%d, n=%d, rank=%d\n\n", batch, m, n, k);
    
    // Column-major parameters
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const long long strideA = (long long)m * n;
    const long long strideU = (long long)m * k;
    const long long strideS = k;
    const long long strideV = (long long)n * k;
    
    // Create handle
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    printf("✓ cuSOLVER handle created\n");
    
    // Allocate device memory
    float *d_A, *d_U, *d_S, *d_V, *d_work;
    int *d_info;
    double *d_h_R_nrmF;
    
    size_t A_bytes = batch * m * n * sizeof(float);
    size_t U_bytes = batch * m * k * sizeof(float);
    size_t S_bytes = batch * k * sizeof(float);
    size_t V_bytes = batch * n * k * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, A_bytes));
    CHECK_CUDA(cudaMalloc(&d_U, U_bytes));
    CHECK_CUDA(cudaMalloc(&d_S, S_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, V_bytes));
    CHECK_CUDA(cudaMalloc(&d_info, batch * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_h_R_nrmF, batch * sizeof(double)));
    printf("✓ Device memory allocated\n");
    
    // Initialize with random data
    std::vector<float> h_A(batch * m * n);
    for (int i = 0; i < batch * m * n; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), A_bytes, cudaMemcpyHostToDevice));
    printf("✓ Input data copied to device\n");
    
    // Query workspace size
    int lwork = 0;
    printf("\nCalling cusolverDnSgesvdaStridedBatched_bufferSize...\n");
    CHECK_CUSOLVER(cusolverDnSgesvdaStridedBatched_bufferSize(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,
        k,
        m, n,
        d_A, lda, strideA,
        d_S, strideS,
        d_U, ldu, strideU,
        d_V, ldv, strideV,
        &lwork,
        batch
    ));
    printf("✓ Workspace size: %d floats\n", lwork);
    
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
    printf("✓ Workspace allocated\n");
    
    // Execute SVD
    printf("\nCalling cusolverDnSgesvdaStridedBatched...\n");
    fflush(stdout);
    
    cusolverStatus_t svd_status = cusolverDnSgesvdaStridedBatched(
        handle,
        CUSOLVER_EIG_MODE_VECTOR,
        k,
        m, n,
        d_A, lda, strideA,
        d_S, strideS,
        d_U, ldu, strideU,
        d_V, ldv, strideV,
        d_work, lwork,
        d_info,
        d_h_R_nrmF,
        batch
    );
    
    if (svd_status != CUSOLVER_STATUS_SUCCESS) {
        printf("✗ SVD failed with status: %d\n", svd_status);
    } else {
        printf("✓ SVD call returned successfully\n");
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("✓ Device synchronized\n");
    
    // Check info
    std::vector<int> h_info(batch);
    CHECK_CUDA(cudaMemcpy(h_info.data(), d_info, batch * sizeof(int), cudaMemcpyDeviceToHost));
    
    bool all_ok = true;
    for (int i = 0; i < batch; i++) {
        if (h_info[i] != 0) {
            printf("✗ Batch %d: info=%d\n", i, h_info[i]);
            all_ok = false;
        }
    }
    
    if (all_ok) {
        printf("✓ All batches completed successfully\n");
        
        // Read back singular values
        std::vector<float> h_S(batch * k);
        CHECK_CUDA(cudaMemcpy(h_S.data(), d_S, S_bytes, cudaMemcpyDeviceToHost));
        
        printf("\nSingular values (first 4 per batch):\n");
        for (int b = 0; b < batch; b++) {
            printf("  Batch %d: ", b);
            for (int i = 0; i < 4 && i < k; i++) {
                printf("%.4f ", h_S[b * k + i]);
            }
            printf("...\n");
        }
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_V);
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_h_R_nrmF);
    cusolverDnDestroy(handle);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
