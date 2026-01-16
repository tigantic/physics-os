/**
 * fused_mps_ops.cu — Hybrid cuBLAS + Custom Fused Kernels
 * 
 * FLUIDELITE v1 Production Kernel
 * Constitutional Compliance: Article VII (No shortcuts)
 * 
 * Strategy:
 *   - cuBLAS gemmBatched → heavy matmul (MPO contraction)
 *   - cuSOLVER → SVD (via PyTorch, works reliably)
 *   - Custom kernels → fuse reshape/truncate/glue ops
 * 
 * Build:
 *   nvcc -shared -Xcompiler -fPIC -arch=sm_120 -O3 \
 *     $(python -m pybind11 --includes) \
 *     -o fused_mps_ops.cpython-312-x86_64-linux-gnu.so \
 *     fused_mps_ops.cu \
 *     -lcudart -lcublas
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error code: " + std::to_string(status)); \
    } \
} while(0)

/**
 * Persistent cuBLAS handle
 */
class CuBlasContext {
public:
    cublasHandle_t handle;
    bool initialized = false;
    
    CuBlasContext() {
        CUBLAS_CHECK(cublasCreate(&handle));
        // Use tensor cores when available
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        initialized = true;
    }
    
    ~CuBlasContext() {
        if (initialized) {
            cublasDestroy(handle);
        }
    }
    
    static CuBlasContext& get() {
        static CuBlasContext instance;
        return instance;
    }
};

/**
 * Fused kernel: Apply truncation mask and reshape
 * 
 * After SVD: U (m×k), S (k), V (n×k)
 * This kernel:
 *   1. Applies epsilon threshold to S
 *   2. Computes U @ diag(S) in-place
 *   3. Reshapes result for next MPS core
 * 
 * All in one kernel launch, data stays in L2 cache
 */
__global__ void fused_truncate_reshape_kernel(
    const float* __restrict__ U,      // (batch, m, k)
    const float* __restrict__ S,      // (batch, k)
    float* __restrict__ US,           // Output: U @ diag(S), (batch, m, new_k)
    const float epsilon,              // Truncation threshold
    const int batch,
    const int m,
    const int k,
    int* __restrict__ new_ranks       // Output: actual rank after truncation per batch
) {
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch || row >= m || col >= k) return;
    
    float s_val = S[b * k + col];
    float s_max = S[b * k];  // S is sorted descending, first element is max
    
    // Epsilon threshold: keep if s_val > epsilon * s_max
    if (s_val > epsilon * s_max) {
        US[b * m * k + row * k + col] = U[b * m * k + row * k + col] * s_val;
        
        // First thread of each batch computes the rank
        if (row == 0 && threadIdx.x == 0) {
            int rank = 0;
            for (int i = 0; i < k; i++) {
                if (S[b * k + i] > epsilon * s_max) rank++;
            }
            new_ranks[b] = rank;
        }
    } else {
        US[b * m * k + row * k + col] = 0.0f;
    }
}

/**
 * Fused kernel: Contract MPS core with MPO and accumulate
 * 
 * MPS core: (chi_l, d, chi_r)
 * MPO core: (D, d_out, d_in, D)  -- simplified for now
 * 
 * This fuses:
 *   1. Reshape MPS for contraction
 *   2. Contract with MPO
 *   3. Reshape result
 */
__global__ void fused_mpo_contract_kernel(
    const float* __restrict__ mps,    // (batch, chi_l, d, chi_r)
    const float* __restrict__ mpo,    // (D, d_out, d_in, D)
    float* __restrict__ out,          // (batch, chi_l, D, d_out, D, chi_r)
    const int batch,
    const int chi_l,
    const int d,
    const int chi_r,
    const int D
) {
    // Shared memory for MPS tile
    extern __shared__ float smem[];
    
    int b = blockIdx.z;
    int tid = threadIdx.x;
    
    // Load MPS core into shared memory
    int mps_size = chi_l * d * chi_r;
    for (int i = tid; i < mps_size; i += blockDim.x) {
        smem[i] = mps[b * mps_size + i];
    }
    __syncthreads();
    
    // Each thread computes one output element
    // Output shape: (chi_l, D, d_out, D, chi_r) = chi_l * D * d * D * chi_r
    int out_size = chi_l * D * d * D * chi_r;
    
    for (int idx = tid; idx < out_size; idx += blockDim.x) {
        // Decode output indices
        int r = idx % chi_r;
        int d2 = (idx / chi_r) % D;
        int d_out = (idx / (chi_r * D)) % d;
        int d1 = (idx / (chi_r * D * d)) % D;
        int l = idx / (chi_r * D * d * D);
        
        // Contract: sum over d_in
        float acc = 0.0f;
        for (int d_in = 0; d_in < d; d_in++) {
            // MPS[l, d_in, r]
            float mps_val = smem[l * d * chi_r + d_in * chi_r + r];
            // MPO[d1, d_out, d_in, d2]
            float mpo_val = mpo[d1 * d * d * D + d_out * d * D + d_in * D + d2];
            acc += mps_val * mpo_val;
        }
        
        out[b * out_size + idx] = acc;
    }
}

/**
 * Batched matrix multiply using cuBLAS
 * For heavy lifting - let NVIDIA's engineers do the work
 */
void batched_matmul_cublas(
    uintptr_t A_ptr,      // (batch, m, k)
    uintptr_t B_ptr,      // (batch, k, n)
    uintptr_t C_ptr,      // (batch, m, n)
    int batch, int m, int n, int k,
    bool transA, bool transB
) {
    float* d_A = reinterpret_cast<float*>(A_ptr);
    float* d_B = reinterpret_cast<float*>(B_ptr);
    float* d_C = reinterpret_cast<float*>(C_ptr);
    
    auto& ctx = CuBlasContext::get();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // cuBLAS uses column-major, so we compute C^T = B^T @ A^T
    // Which in row-major is C = A @ B
    
    long long strideA = (long long)m * k;
    long long strideB = (long long)k * n;
    long long strideC = (long long)m * n;
    
    cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    
    int lda = transA ? m : k;
    int ldb = transB ? k : n;
    int ldc = n;
    
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        ctx.handle,
        opB, opA,  // Swapped for row-major
        n, m, k,
        &alpha,
        d_B, CUDA_R_32F, ldb, strideB,
        d_A, CUDA_R_32F, lda, strideA,
        &beta,
        d_C, CUDA_R_32F, ldc, strideC,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

/**
 * Apply truncation with epsilon threshold
 * Returns: US = U @ diag(S) with zeros where S < eps*S_max
 */
void apply_truncation(
    uintptr_t U_ptr,
    uintptr_t S_ptr,
    uintptr_t US_ptr,
    uintptr_t ranks_ptr,
    int batch, int m, int k,
    float epsilon
) {
    float* d_U = reinterpret_cast<float*>(U_ptr);
    float* d_S = reinterpret_cast<float*>(S_ptr);
    float* d_US = reinterpret_cast<float*>(US_ptr);
    int* d_ranks = reinterpret_cast<int*>(ranks_ptr);
    
    dim3 block(16, 16);
    dim3 grid((k + 15) / 16, (m + 15) / 16, batch);
    
    fused_truncate_reshape_kernel<<<grid, block>>>(
        d_U, d_S, d_US, epsilon, batch, m, k, d_ranks
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Benchmark cuBLAS batched GEMM
 */
double benchmark_cublas_gemm(int batch, int m, int n, int k, int iterations) {
    size_t A_size = batch * m * k * sizeof(float);
    size_t B_size = batch * k * n * sizeof(float);
    size_t C_size = batch * m * n * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, A_size));
    CUDA_CHECK(cudaMalloc(&d_B, B_size));
    CUDA_CHECK(cudaMalloc(&d_C, C_size));
    
    // Initialize with random data
    std::vector<float> h_A(batch * m * k), h_B(batch * k * n);
    for (size_t i = 0; i < h_A.size(); i++) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < h_B.size(); i++) h_B[i] = (float)rand() / RAND_MAX;
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_size, cudaMemcpyHostToDevice));
    
    // Warmup
    batched_matmul_cublas(
        (uintptr_t)d_A, (uintptr_t)d_B, (uintptr_t)d_C,
        batch, m, n, k, false, false
    );
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        batched_matmul_cublas(
            (uintptr_t)d_A, (uintptr_t)d_B, (uintptr_t)d_C,
            batch, m, n, k, false, false
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return elapsed_ms / iterations;
}

PYBIND11_MODULE(fused_mps_ops, m) {
    m.doc() = "Hybrid cuBLAS + Custom fused kernels for FluidElite";
    
    m.def("batched_matmul", &batched_matmul_cublas,
          "Batched matrix multiply using cuBLAS",
          py::arg("A_ptr"), py::arg("B_ptr"), py::arg("C_ptr"),
          py::arg("batch"), py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("transA") = false, py::arg("transB") = false);
    
    m.def("apply_truncation", &apply_truncation,
          "Fused truncation with epsilon threshold",
          py::arg("U_ptr"), py::arg("S_ptr"), py::arg("US_ptr"), py::arg("ranks_ptr"),
          py::arg("batch"), py::arg("m"), py::arg("k"), py::arg("epsilon"));
    
    m.def("benchmark_gemm", &benchmark_cublas_gemm,
          "Benchmark cuBLAS batched GEMM",
          py::arg("batch"), py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("iterations") = 100);
}
