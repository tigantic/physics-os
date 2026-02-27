/**
 * batched_svd.cu — cuSOLVER gesvdjBatched wrapper for MPS truncation
 * 
 * FLUIDELITE v1 Production Kernel
 * Constitutional Compliance: Article VII (No shortcuts)
 * 
 * Purpose: Compute batched SVD for all MPS sites in a single GPU call
 * Target: 15× (chi×chi) matrices per forward pass
 * 
 * Build:
 *   nvcc -shared -Xcompiler -fPIC -arch=sm_120 \
 *     $(python -m pybind11 --includes) \
 *     -o batched_svd.cpython-312-x86_64-linux-gnu.so \
 *     batched_svd.cu \
 *     -lcudart -lcusolver
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// CUDA error checking macro
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
 * Persistent cuSOLVER handle for performance
 * Avoids handle creation overhead on every call
 */
class CuSolverContext {
public:
    cusolverDnHandle_t handle;
    gesvdjInfo_t params;
    bool initialized = false;
    
    CuSolverContext() {
        CUSOLVER_CHECK(cusolverDnCreate(&handle));
        CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&params));
        
        // Configure Jacobi SVD parameters
        // tolerance: 1e-7, max_sweeps: 15 (default is good for our use case)
        CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(params, 1e-7));
        CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(params, 15));
        
        initialized = true;
    }
    
    ~CuSolverContext() {
        if (initialized) {
            cusolverDnDestroyGesvdjInfo(params);
            cusolverDnDestroy(handle);
        }
    }
    
    // Singleton access
    static CuSolverContext& get() {
        static CuSolverContext instance;
        return instance;
    }
};

/**
 * Batched SVD using cuSOLVER gesvdjBatched
 * 
 * Input:
 *   A: (batch, m, n) float32 matrices (row-major from numpy)
 *   
 * Output:
 *   U: (batch, m, min(m,n))
 *   S: (batch, min(m,n))
 *   V: (batch, n, min(m,n))  -- Note: V, not V^T
 *   
 * cuSOLVER uses column-major, so we transpose on copy
 */
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
batched_svd_float32(py::array_t<float> A_np) {
    
    // Get array info
    py::buffer_info buf = A_np.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Input must be 3D array (batch, m, n)");
    }
    
    int batch = buf.shape[0];
    int m = buf.shape[1];
    int n = buf.shape[2];
    int minmn = std::min(m, n);
    
    // cuSOLVER is column-major, leading dims are the first dimension of the 2D slice
    int lda = m;  // Column-major: leading dimension is m
    int ldu = m;
    int ldv = n;
    
    // Get cuSOLVER context
    auto& ctx = CuSolverContext::get();
    
    // Allocate device memory
    float* d_A = nullptr;
    float* d_U = nullptr;
    float* d_S = nullptr;
    float* d_V = nullptr;
    int* d_info = nullptr;
    float* d_work = nullptr;
    
    size_t A_size = batch * m * n * sizeof(float);
    size_t U_size = batch * m * minmn * sizeof(float);
    size_t S_size = batch * minmn * sizeof(float);
    size_t V_size = batch * n * minmn * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_U, U_size));
    CUDA_CHECK(cudaMalloc(&d_S, S_size));
    CUDA_CHECK(cudaMalloc(&d_V, V_size));
    CUDA_CHECK(cudaMalloc(&d_info, batch * sizeof(int)));
    
    // Transpose input from row-major to column-major on host
    float* h_A = static_cast<float*>(buf.ptr);
    std::vector<float> h_A_col(batch * m * n);
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // row-major: A[b,i,j] = h_A[b*m*n + i*n + j]
                // col-major: A[b,i,j] = h_A_col[b*m*n + j*m + i]
                h_A_col[b*m*n + j*m + i] = h_A[b*m*n + i*n + j];
            }
        }
    }
    
    // Copy transposed input to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A_col.data(), A_size, cudaMemcpyHostToDevice));
    
    // Query workspace size
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(
        ctx.handle,
        CUSOLVER_EIG_MODE_VECTOR,  // Compute U and V
        m, n,
        d_A, lda,
        d_S,
        d_U, ldu,
        d_V, ldv,
        &lwork,
        ctx.params,
        batch
    ));
    
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));
    
    // Execute batched SVD
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
        ctx.handle,
        CUSOLVER_EIG_MODE_VECTOR,
        m, n,
        d_A, lda,
        d_S,
        d_U, ldu,
        d_V, ldv,
        d_work, lwork,
        d_info,
        ctx.params,
        batch
    ));
    
    // Synchronize and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Check convergence info
    std::vector<int> h_info(batch);
    CUDA_CHECK(cudaMemcpy(h_info.data(), d_info, batch * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch; i++) {
        if (h_info[i] != 0) {
            cudaFree(d_A); cudaFree(d_U); cudaFree(d_S); cudaFree(d_V);
            cudaFree(d_info); cudaFree(d_work);
            throw std::runtime_error("SVD failed to converge for matrix " + std::to_string(i));
        }
    }
    
    // Allocate output arrays
    auto U_np = py::array_t<float>({batch, m, minmn});
    auto S_np = py::array_t<float>({batch, minmn});
    auto V_np = py::array_t<float>({batch, n, minmn});
    
    // Copy results back to host (column-major)
    std::vector<float> h_U_col(batch * m * minmn);
    std::vector<float> h_V_col(batch * n * minmn);
    
    CUDA_CHECK(cudaMemcpy(h_U_col.data(), d_U, U_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(S_np.mutable_data(), d_S, S_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V_col.data(), d_V, V_size, cudaMemcpyDeviceToHost));
    
    // Transpose U from column-major to row-major
    float* h_U = U_np.mutable_data();
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < minmn; j++) {
                // col-major: U[b,i,j] = h_U_col[b*m*minmn + j*m + i]
                // row-major: U[b,i,j] = h_U[b*m*minmn + i*minmn + j]
                h_U[b*m*minmn + i*minmn + j] = h_U_col[b*m*minmn + j*m + i];
            }
        }
    }
    
    // Transpose V from column-major to row-major
    float* h_V = V_np.mutable_data();
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < minmn; j++) {
                // col-major: V[b,i,j] = h_V_col[b*n*minmn + j*n + i]
                // row-major: V[b,i,j] = h_V[b*n*minmn + i*minmn + j]
                h_V[b*n*minmn + i*minmn + j] = h_V_col[b*n*minmn + j*n + i];
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
    
    return std::make_tuple(U_np, S_np, V_np);
}

/**
 * Batched SVD with GPU tensor input/output (zero-copy with PyTorch)
 * 
 * Input:
 *   A_ptr: Raw pointer to GPU memory (batch, m, n) float32
 *   batch, m, n: Dimensions
 *   
 * Output:
 *   Writes directly to pre-allocated GPU buffers
 */
void batched_svd_gpu(
    uintptr_t A_ptr,      // Input: (batch, m, n)
    uintptr_t U_ptr,      // Output: (batch, m, minmn)
    uintptr_t S_ptr,      // Output: (batch, minmn)
    uintptr_t V_ptr,      // Output: (batch, n, minmn)
    int batch, int m, int n
) {
    int minmn = std::min(m, n);
    int lda = n;
    int ldu = minmn;
    int ldv = minmn;
    
    float* d_A = reinterpret_cast<float*>(A_ptr);
    float* d_U = reinterpret_cast<float*>(U_ptr);
    float* d_S = reinterpret_cast<float*>(S_ptr);
    float* d_V = reinterpret_cast<float*>(V_ptr);
    
    auto& ctx = CuSolverContext::get();
    
    // Allocate workspace and info
    int* d_info = nullptr;
    float* d_work = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_info, batch * sizeof(int)));
    
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(
        ctx.handle,
        CUSOLVER_EIG_MODE_VECTOR,
        m, n,
        d_A, lda,
        d_S,
        d_U, ldu,
        d_V, ldv,
        &lwork,
        ctx.params,
        batch
    ));
    
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));
    
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(
        ctx.handle,
        CUSOLVER_EIG_MODE_VECTOR,
        m, n,
        d_A, lda,
        d_S,
        d_U, ldu,
        d_V, ldv,
        d_work, lwork,
        d_info,
        ctx.params,
        batch
    ));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    cudaFree(d_info);
    cudaFree(d_work);
}

/**
 * Get SVD statistics for debugging/monitoring
 */
py::dict get_svd_stats() {
    auto& ctx = CuSolverContext::get();
    
    double residual = 0.0;
    int executed_sweeps = 0;
    
    CUSOLVER_CHECK(cusolverDnXgesvdjGetResidual(ctx.handle, ctx.params, &residual));
    CUSOLVER_CHECK(cusolverDnXgesvdjGetSweeps(ctx.handle, ctx.params, &executed_sweeps));
    
    py::dict stats;
    stats["residual"] = residual;
    stats["sweeps"] = executed_sweeps;
    return stats;
}

/**
 * Benchmark function for performance testing
 */
double benchmark_batched_svd(int batch, int m, int n, int iterations) {
    // Allocate random matrices
    size_t A_size = batch * m * n * sizeof(float);
    size_t U_size = batch * m * std::min(m, n) * sizeof(float);
    size_t S_size = batch * std::min(m, n) * sizeof(float);
    size_t V_size = batch * n * std::min(m, n) * sizeof(float);
    
    float* d_A = nullptr;
    float* d_U = nullptr;
    float* d_S = nullptr;
    float* d_V = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_U, U_size));
    CUDA_CHECK(cudaMalloc(&d_S, S_size));
    CUDA_CHECK(cudaMalloc(&d_V, V_size));
    
    // Initialize with random data on host and copy
    std::vector<float> h_A(batch * m * n);
    for (size_t i = 0; i < h_A.size(); i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_size, cudaMemcpyHostToDevice));
    
    // Warmup
    batched_svd_gpu(
        reinterpret_cast<uintptr_t>(d_A),
        reinterpret_cast<uintptr_t>(d_U),
        reinterpret_cast<uintptr_t>(d_S),
        reinterpret_cast<uintptr_t>(d_V),
        batch, m, n
    );
    
    // Timed iterations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        batched_svd_gpu(
            reinterpret_cast<uintptr_t>(d_A),
            reinterpret_cast<uintptr_t>(d_U),
            reinterpret_cast<uintptr_t>(d_S),
            reinterpret_cast<uintptr_t>(d_V),
            batch, m, n
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_U);
    cudaFree(d_S);
    cudaFree(d_V);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed_ms / iterations;
}

PYBIND11_MODULE(batched_svd, m) {
    m.doc() = "cuSOLVER batched SVD for FluidElite MPS truncation";
    
    m.def("svd_batched", &batched_svd_float32,
          "Batched SVD on CPU numpy arrays",
          py::arg("A"));
    
    m.def("svd_batched_gpu", &batched_svd_gpu,
          "Batched SVD on GPU tensors (raw pointers)",
          py::arg("A_ptr"), py::arg("U_ptr"), py::arg("S_ptr"), py::arg("V_ptr"),
          py::arg("batch"), py::arg("m"), py::arg("n"));
    
    m.def("get_svd_stats", &get_svd_stats,
          "Get statistics from last SVD execution");
    
    m.def("benchmark", &benchmark_batched_svd,
          "Benchmark batched SVD performance",
          py::arg("batch"), py::arg("m"), py::arg("n"), py::arg("iterations") = 10);
}
