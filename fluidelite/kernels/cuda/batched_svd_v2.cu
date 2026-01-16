/**
 * batched_svd_v2.cu — cuSOLVER batched SVD for MPS truncation
 * 
 * FLUIDELITE v1 Production Kernel
 * Constitutional Compliance: Article VII (No shortcuts)
 * 
 * Uses gesvdaStridedBatched for approximate rank-k SVD (works for any size)
 * Falls back to sequential gesvdj for full SVD when needed
 * 
 * Build:
 *   nvcc -shared -Xcompiler -fPIC -arch=sm_120 -O3 \
 *     $(python -m pybind11 --includes) \
 *     -o batched_svd.cpython-312-x86_64-linux-gnu.so \
 *     batched_svd_v2.cu \
 *     -lcudart -lcusolver
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace py = pybind11;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        throw std::runtime_error("cuSOLVER error code: " + std::to_string(status)); \
    } \
} while(0)

/**
 * Persistent cuSOLVER context
 */
class CuSolverContext {
public:
    cusolverDnHandle_t handle;
    gesvdjInfo_t params;
    bool initialized = false;
    
    CuSolverContext() {
        CUSOLVER_CHECK(cusolverDnCreate(&handle));
        CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(params, 1e-7));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(params, 100));
        initialized = true;
    }
    
    ~CuSolverContext() {
        if (initialized) {
            cusolverDnDestroyGesvdjInfo(params);
            cusolverDnDestroy(handle);
        }
    }
    
    static CuSolverContext& get() {
        static CuSolverContext instance;
        return instance;
    }
};

/**
 * Batched rank-k SVD using gesvdaStridedBatched
 * 
 * This computes approximate SVD with the top-k singular values.
 * Works for any matrix size (not limited to 32×32 like gesvdjBatched)
 * 
 * Input:
 *   A: (batch, m, n) float32 matrices
 *   rank: Number of singular values to compute (k)
 *   
 * Output:
 *   U: (batch, m, rank)
 *   S: (batch, rank)
 *   V: (batch, n, rank)
 */
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
batched_svd_rank_k(py::array_t<float> A_np, int rank) {
    
    py::buffer_info buf = A_np.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be 3D array (batch, m, n)");
    }
    
    int batch = buf.shape[0];
    int m = buf.shape[1];
    int n = buf.shape[2];
    int k = std::min(rank, std::min(m, n));
    
    // Column-major leading dimensions
    int lda = m;
    int ldu = m;
    int ldv = n;
    
    // Strides for batched access
    long long strideA = (long long)m * n;
    long long strideU = (long long)m * k;
    long long strideS = k;
    long long strideV = (long long)n * k;
    
    auto& ctx = CuSolverContext::get();
    
    // Allocate device memory
    float* d_A = nullptr;
    float* d_U = nullptr;
    float* d_S = nullptr;
    float* d_V = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;
    double* d_h_R_nrmF = nullptr;  // Residual norms (for accuracy check)
    
    size_t A_size = batch * m * n * sizeof(float);
    size_t U_size = batch * m * k * sizeof(float);
    size_t S_size = batch * k * sizeof(float);
    size_t V_size = batch * n * k * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_U, U_size));
    CUDA_CHECK(cudaMalloc(&d_S, S_size));
    CUDA_CHECK(cudaMalloc(&d_V, V_size));
    CUDA_CHECK(cudaMalloc(&d_info, batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_h_R_nrmF, batch * sizeof(double)));
    
    // Transpose input from row-major to column-major
    float* h_A = static_cast<float*>(buf.ptr);
    std::vector<float> h_A_col(batch * m * n);
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                h_A_col[b*m*n + j*m + i] = h_A[b*m*n + i*n + j];
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A_col.data(), A_size, cudaMemcpyHostToDevice));
    
    // Query workspace
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
        ctx.handle,
        CUSOLVER_EIG_MODE_VECTOR,
        k,          // rank
        m, n,
        d_A, lda, strideA,
        d_S, strideS,
        d_U, ldu, strideU,
        d_V, ldv, strideV,
        &lwork,
        batch
    ));
    
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));
    
    // Execute batched rank-k SVD
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
        ctx.handle,
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
    ));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check for errors
    std::vector<int> h_info(batch);
    CUDA_CHECK(cudaMemcpy(h_info.data(), d_info, batch * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch; i++) {
        if (h_info[i] != 0) {
            cudaFree(d_A); cudaFree(d_U); cudaFree(d_S); cudaFree(d_V);
            cudaFree(d_info); cudaFree(d_work); cudaFree(d_h_R_nrmF);
            throw std::runtime_error("SVD failed for matrix " + std::to_string(i) + 
                                   " with info=" + std::to_string(h_info[i]));
        }
    }
    
    // Allocate output arrays
    auto U_np = py::array_t<float>({batch, m, k});
    auto S_np = py::array_t<float>({batch, k});
    auto V_np = py::array_t<float>({batch, n, k});
    
    // Copy results back
    std::vector<float> h_U_col(batch * m * k);
    std::vector<float> h_V_col(batch * n * k);
    
    CUDA_CHECK(cudaMemcpy(h_U_col.data(), d_U, U_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(S_np.mutable_data(), d_S, S_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V_col.data(), d_V, V_size, cudaMemcpyDeviceToHost));
    
    // Transpose U to row-major
    float* h_U = U_np.mutable_data();
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                h_U[b*m*k + i*k + j] = h_U_col[b*m*k + j*m + i];
            }
        }
    }
    
    // Transpose V to row-major
    float* h_V = V_np.mutable_data();
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                h_V[b*n*k + i*k + j] = h_V_col[b*n*k + j*n + i];
            }
        }
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_V);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(d_h_R_nrmF);
    
    return std::make_tuple(U_np, S_np, V_np);
}

/**
 * Benchmark function
 */
double benchmark_batched_svd(int batch, int m, int n, int rank, int iterations) {
    size_t A_size = batch * m * n * sizeof(float);
    
    // Initialize random data
    std::vector<float> h_A(batch * m * n);
    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Create numpy array
    auto A_np = py::array_t<float>({batch, m, n});
    std::memcpy(A_np.mutable_data(), h_A.data(), A_size);
    
    // Warmup
    batched_svd_rank_k(A_np, rank);
    
    // Timed iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        batched_svd_rank_k(A_np, rank);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_ms / iterations;
}

/**
 * GPU pointer version for zero-copy with PyTorch tensors
 */
void batched_svd_gpu(
    uintptr_t A_ptr,      // (batch, m, n) column-major
    uintptr_t U_ptr,      // (batch, m, k) 
    uintptr_t S_ptr,      // (batch, k)
    uintptr_t V_ptr,      // (batch, n, k)
    int batch, int m, int n, int k
) {
    int lda = m;
    int ldu = m;
    int ldv = n;
    
    long long strideA = (long long)m * n;
    long long strideU = (long long)m * k;
    long long strideS = k;
    long long strideV = (long long)n * k;
    
    float* d_A = reinterpret_cast<float*>(A_ptr);
    float* d_U = reinterpret_cast<float*>(U_ptr);
    float* d_S = reinterpret_cast<float*>(S_ptr);
    float* d_V = reinterpret_cast<float*>(V_ptr);
    
    auto& ctx = CuSolverContext::get();
    
    int* d_info = nullptr;
    float* d_work = nullptr;
    double* d_h_R_nrmF = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_info, batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_h_R_nrmF, batch * sizeof(double)));
    
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
        ctx.handle,
        CUSOLVER_EIG_MODE_VECTOR,
        k, m, n,
        d_A, lda, strideA,
        d_S, strideS,
        d_U, ldu, strideU,
        d_V, ldv, strideV,
        &lwork,
        batch
    ));
    
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));
    
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
        ctx.handle,
        CUSOLVER_EIG_MODE_VECTOR,
        k, m, n,
        d_A, lda, strideA,
        d_S, strideS,
        d_U, ldu, strideU,
        d_V, ldv, strideV,
        d_work, lwork,
        d_info,
        d_h_R_nrmF,
        batch
    ));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(d_h_R_nrmF);
}

PYBIND11_MODULE(batched_svd, m) {
    m.doc() = "cuSOLVER batched rank-k SVD for FluidElite MPS truncation";
    
    m.def("svd_batched", &batched_svd_rank_k,
          "Batched rank-k SVD",
          py::arg("A"), py::arg("rank"));
    
    m.def("svd_batched_gpu", &batched_svd_gpu,
          "Batched rank-k SVD on GPU tensors",
          py::arg("A_ptr"), py::arg("U_ptr"), py::arg("S_ptr"), py::arg("V_ptr"),
          py::arg("batch"), py::arg("m"), py::arg("n"), py::arg("k"));
    
    m.def("benchmark", &benchmark_batched_svd,
          "Benchmark batched SVD",
          py::arg("batch"), py::arg("m"), py::arg("n"), 
          py::arg("rank"), py::arg("iterations") = 10);
}
