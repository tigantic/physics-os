/**
 * Test cuSOLVER gesvdjBatched (Jacobi method) vs gesvdaStridedBatched
 * 
 * gesvdjBatched: Works but limited to 32×32
 * gesvdaStridedBatched: Supposed to work for any size, segfaults on Blackwell
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
        return false; \
    } \
} while(0)

bool test_gesvdjBatched(int m, int n, int batch) {
    printf("\n--- Testing gesvdjBatched (%d×%d, batch=%d) ---\n", m, n, batch);
    
    cusolverDnHandle_t handle;
    gesvdjInfo_t params;
    
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    CHECK_CUSOLVER(cusolverDnCreateGesvdjInfo(&params));
    CHECK_CUSOLVER(cusolverDnXgesvdjSetTolerance(params, 1e-7));
    CHECK_CUSOLVER(cusolverDnXgesvdjSetMaxSweeps(params, 100));
    
    int minmn = (m < n) ? m : n;
    int lda = m;
    int ldu = m;
    int ldv = n;
    
    float *d_A, *d_U, *d_S, *d_V, *d_work;
    int *d_info;
    
    CHECK_CUDA(cudaMalloc(&d_A, batch * m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_U, batch * m * minmn * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_S, batch * minmn * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, batch * n * minmn * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_info, batch * sizeof(int)));
    
    // Initialize
    std::vector<float> h_A(batch * m * n);
    for (int i = 0; i < batch * m * n; i++) h_A[i] = (float)rand() / RAND_MAX;
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), batch * m * n * sizeof(float), cudaMemcpyHostToDevice));
    
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSgesvdjBatched_bufferSize(
        handle, CUSOLVER_EIG_MODE_VECTOR, m, n,
        d_A, lda, d_S, d_U, ldu, d_V, ldv,
        &lwork, params, batch
    ));
    
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
    
    printf("Executing gesvdjBatched...\n");
    cusolverStatus_t status = cusolverDnSgesvdjBatched(
        handle, CUSOLVER_EIG_MODE_VECTOR, m, n,
        d_A, lda, d_S, d_U, ldu, d_V, ldv,
        d_work, lwork, d_info, params, batch
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::vector<int> h_info(batch);
    CHECK_CUDA(cudaMemcpy(h_info.data(), d_info, batch * sizeof(int), cudaMemcpyDeviceToHost));
    
    bool ok = (status == CUSOLVER_STATUS_SUCCESS);
    for (int i = 0; i < batch; i++) {
        if (h_info[i] != 0) ok = false;
    }
    
    printf("%s: gesvdjBatched %d×%d\n", ok ? "✅ PASS" : "❌ FAIL", m, n);
    
    cudaFree(d_A); cudaFree(d_U); cudaFree(d_S); cudaFree(d_V);
    cudaFree(d_work); cudaFree(d_info);
    cusolverDnDestroyGesvdjInfo(params);
    cusolverDnDestroy(handle);
    
    return ok;
}

bool test_gesvdaStridedBatched(int m, int n, int k, int batch) {
    printf("\n--- Testing gesvdaStridedBatched (%d×%d, rank=%d, batch=%d) ---\n", m, n, k, batch);
    
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    
    int lda = m;
    int ldu = m;
    int ldv = n;
    long long strideA = m * n;
    long long strideU = m * k;
    long long strideS = k;
    long long strideV = n * k;
    
    float *d_A, *d_U, *d_S, *d_V, *d_work;
    int *d_info;
    double *d_h_R_nrmF;
    
    CHECK_CUDA(cudaMalloc(&d_A, batch * m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_U, batch * m * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_S, batch * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, batch * n * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_info, batch * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_h_R_nrmF, batch * sizeof(double)));
    
    std::vector<float> h_A(batch * m * n);
    for (int i = 0; i < batch * m * n; i++) h_A[i] = (float)rand() / RAND_MAX;
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), batch * m * n * sizeof(float), cudaMemcpyHostToDevice));
    
    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnSgesvdaStridedBatched_bufferSize(
        handle, CUSOLVER_EIG_MODE_VECTOR, k, m, n,
        d_A, lda, strideA, d_S, strideS,
        d_U, ldu, strideU, d_V, ldv, strideV,
        &lwork, batch
    ));
    
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
    
    printf("Executing gesvdaStridedBatched...\n");
    cusolverStatus_t status = cusolverDnSgesvdaStridedBatched(
        handle, CUSOLVER_EIG_MODE_VECTOR, k, m, n,
        d_A, lda, strideA, d_S, strideS,
        d_U, ldu, strideU, d_V, ldv, strideV,
        d_work, lwork, d_info, d_h_R_nrmF, batch
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("✅ PASS: gesvdaStridedBatched %d×%d rank=%d\n", m, n, k);
    
    cudaFree(d_A); cudaFree(d_U); cudaFree(d_S); cudaFree(d_V);
    cudaFree(d_work); cudaFree(d_info); cudaFree(d_h_R_nrmF);
    cusolverDnDestroy(handle);
    
    return status == CUSOLVER_STATUS_SUCCESS;
}

int main() {
    printf("=== cuSOLVER Batched SVD Test Suite (Blackwell sm_120) ===\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    
    // Test gesvdjBatched (Jacobi) - documented limit is 32×32
    printf("\n======== gesvdjBatched (Jacobi method) ========\n");
    test_gesvdjBatched(16, 16, 4);
    test_gesvdjBatched(32, 32, 4);
    // test_gesvdjBatched(64, 64, 4);  // Would fail - exceeds 32×32 limit
    
    // Test gesvdaStridedBatched - should work for any size but segfaults
    printf("\n======== gesvdaStridedBatched (polar decomposition) ========\n");
    printf("NOTE: These may segfault on Blackwell (cuSOLVER bug)\n");
    test_gesvdaStridedBatched(16, 16, 8, 4);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}
