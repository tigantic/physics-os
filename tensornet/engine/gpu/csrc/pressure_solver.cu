/*
 * OPERATION VALHALLA - Phase 2.3 Optimization
 * cuSparse Pressure Solver for Incompressible Navier-Stokes
 * 
 * Solves: ∇²p = ∇·u using Preconditioned Conjugate Gradient
 * Target: <5ms per solve on RTX 5070
 * 
 * Author: The Architect
 * Date: 2025-12-28
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <vector>

// CUDA kernel: Compute divergence of velocity field
__global__ void compute_divergence_kernel(
    const float* __restrict__ u,
    const float* __restrict__ v, 
    const float* __restrict__ w,
    float* __restrict__ div,
    int nx, int ny, int nz,
    float inv_dx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + nx * (j + ny * k);
    
    // Periodic boundary conditions using modulo
    int ip = ((i + 1) % nx) + nx * (j + ny * k);
    int im = ((i - 1 + nx) % nx) + nx * (j + ny * k);
    int jp = i + nx * (((j + 1) % ny) + ny * k);
    int jm = i + nx * (((j - 1 + ny) % ny) + ny * k);
    int kp = i + nx * (j + ny * ((k + 1) % nz));
    int km = i + nx * (j + ny * ((k - 1 + nz) % nz));
    
    // Central differences: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
    float du_dx = (u[ip] - u[im]) * 0.5f * inv_dx;
    float dv_dy = (v[jp] - v[jm]) * 0.5f * inv_dx;
    float dw_dz = (w[kp] - w[km]) * 0.5f * inv_dx;
    
    div[idx] = du_dx + dv_dy + dw_dz;
}

// CUDA kernel: Compute pressure gradient and update velocity
__global__ void apply_pressure_gradient_kernel(
    float* __restrict__ u,
    float* __restrict__ v,
    float* __restrict__ w,
    const float* __restrict__ p,
    int nx, int ny, int nz,
    float inv_dx
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int idx = i + nx * (j + ny * k);
    
    // Periodic boundary conditions
    int ip = ((i + 1) % nx) + nx * (j + ny * k);
    int im = ((i - 1 + nx) % nx) + nx * (j + ny * k);
    int jp = i + nx * (((j + 1) % ny) + ny * k);
    int jm = i + nx * (((j - 1 + ny) % ny) + ny * k);
    int kp = i + nx * (j + ny * ((k + 1) % nz));
    int km = i + nx * (j + ny * ((k - 1 + nz) % nz));
    
    // Compute pressure gradient
    float dp_dx = (p[ip] - p[im]) * 0.5f * inv_dx;
    float dp_dy = (p[jp] - p[jm]) * 0.5f * inv_dx;
    float dp_dz = (p[kp] - p[km]) * 0.5f * inv_dx;
    
    // Update velocity: u_new = u - ∇p
    u[idx] -= dp_dx;
    v[idx] -= dp_dy;
    w[idx] -= dp_dz;
}

// Conjugate Gradient solver for Poisson equation: ∇²p = div
// Uses cuSparse for sparse matrix-vector product (Laplacian stencil)
class PressureSolverCG {
private:
    cusparseHandle_t cusparse_handle;
    cublasHandle_t cublas_handle;
    cusparseSpMatDescr_t mat_laplacian;
    cusparseDnVecDescr_t vec_x, vec_b, vec_r, vec_p, vec_Ap;
    
    int nx, ny, nz, n_total;
    float dx;
    
    // Device memory for CG algorithm
    float *d_p;      // Pressure (solution)
    float *d_r;      // Residual
    float *d_p_vec;  // Search direction
    float *d_Ap;     // Matrix-vector product
    
    // Sparse matrix storage (CSR format)
    int *d_csr_row_ptr;
    int *d_csr_col_idx;
    float *d_csr_values;
    int nnz;
    
public:
    PressureSolverCG(int nx, int ny, int nz, float dx) 
        : nx(nx), ny(ny), nz(nz), dx(dx) {
        
        n_total = nx * ny * nz;
        
        // Initialize cuSparse and cuBLAS
        cusparseCreate(&cusparse_handle);
        cublasCreate(&cublas_handle);
        
        // Allocate device memory
        cudaMalloc(&d_p, n_total * sizeof(float));
        cudaMalloc(&d_r, n_total * sizeof(float));
        cudaMalloc(&d_p_vec, n_total * sizeof(float));
        cudaMalloc(&d_Ap, n_total * sizeof(float));
        
        // Build sparse Laplacian matrix (7-point stencil)
        build_laplacian_matrix();
    }
    
    ~PressureSolverCG() {
        cudaFree(d_p);
        cudaFree(d_r);
        cudaFree(d_p_vec);
        cudaFree(d_Ap);
        cudaFree(d_csr_row_ptr);
        cudaFree(d_csr_col_idx);
        cudaFree(d_csr_values);
        
        cusparseDestroySpMat(mat_laplacian);
        cusparseDestroy(cusparse_handle);
        cublasDestroy(cublas_handle);
    }
    
    void build_laplacian_matrix() {
        // 7-point stencil: center + 6 neighbors
        // Maximum non-zeros per row = 7
        nnz = n_total * 7;
        
        // Allocate CSR arrays
        cudaMalloc(&d_csr_row_ptr, (n_total + 1) * sizeof(int));
        cudaMalloc(&d_csr_col_idx, nnz * sizeof(int));
        cudaMalloc(&d_csr_values, nnz * sizeof(float));
        
        // Build on CPU first (easier indexing), then copy to GPU
        std::vector<int> row_ptr(n_total + 1);
        std::vector<int> col_idx;
        std::vector<float> values;
        
        col_idx.reserve(nnz);
        values.reserve(nnz);
        
        float inv_dx2 = 1.0f / (dx * dx);
        int nnz_count = 0;
        
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int idx = i + nx * (j + ny * k);
                    row_ptr[idx] = nnz_count;
                    
                    // Center coefficient: -6/dx²
                    col_idx.push_back(idx);
                    values.push_back(-6.0f * inv_dx2);
                    nnz_count++;
                    
                    // X neighbors
                    int im = ((i - 1 + nx) % nx) + nx * (j + ny * k);
                    int ip = ((i + 1) % nx) + nx * (j + ny * k);
                    col_idx.push_back(im);
                    values.push_back(inv_dx2);
                    col_idx.push_back(ip);
                    values.push_back(inv_dx2);
                    nnz_count += 2;
                    
                    // Y neighbors
                    int jm = i + nx * (((j - 1 + ny) % ny) + ny * k);
                    int jp = i + nx * (((j + 1) % ny) + ny * k);
                    col_idx.push_back(jm);
                    values.push_back(inv_dx2);
                    col_idx.push_back(jp);
                    values.push_back(inv_dx2);
                    nnz_count += 2;
                    
                    // Z neighbors
                    int km = i + nx * (j + ny * ((k - 1 + nz) % nz));
                    int kp = i + nx * (j + ny * ((k + 1) % nz));
                    col_idx.push_back(km);
                    values.push_back(inv_dx2);
                    col_idx.push_back(kp);
                    values.push_back(inv_dx2);
                    nnz_count += 2;
                }
            }
        }
        row_ptr[n_total] = nnz_count;
        nnz = nnz_count;
        
        // Copy to device
        cudaMemcpy(d_csr_row_ptr, row_ptr.data(), (n_total + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_col_idx, col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        
        // Create sparse matrix descriptor
        cusparseCreateCsr(&mat_laplacian, n_total, n_total, nnz,
                         d_csr_row_ptr, d_csr_col_idx, d_csr_values,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    }
    
    void solve(float* d_div, float* d_pressure_out, float* d_pressure_guess, 
               int max_iter = 10, float tolerance = 1e-4) {
        
        // Initialize with warm start (previous frame's pressure)
        cudaMemcpy(d_p, d_pressure_guess, n_total * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // r = b - A*x (residual)
        float alpha = 1.0f, beta = 0.0f;
        size_t buffer_size = 0;
        void* d_buffer = nullptr;
        
        // Create dense vector descriptors
        cusparseCreateDnVec(&vec_x, n_total, d_p, CUDA_R_32F);
        cusparseCreateDnVec(&vec_b, n_total, d_div, CUDA_R_32F);
        cusparseCreateDnVec(&vec_r, n_total, d_r, CUDA_R_32F);
        cusparseCreateDnVec(&vec_p, n_total, d_p_vec, CUDA_R_32F);
        cusparseCreateDnVec(&vec_Ap, n_total, d_Ap, CUDA_R_32F);
        
        // Get buffer size for SpMV
        cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, mat_laplacian, vec_x, &beta, vec_r,
                               CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
        cudaMalloc(&d_buffer, buffer_size);
        
        // Compute initial residual: r = b - A*x
        alpha = -1.0f; beta = 0.0f;
        cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, mat_laplacian, vec_x, &beta, vec_r,
                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        
        // r = b + r (since we used -A*x)
        alpha = 1.0f;
        cublasSaxpy(cublas_handle, n_total, &alpha, d_div, 1, d_r, 1);
        
        // p = r (initial search direction)
        cudaMemcpy(d_p_vec, d_r, n_total * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // r_dot_r_old = r^T * r
        float r_dot_r_old, r_dot_r_new;
        cublasSdot(cublas_handle, n_total, d_r, 1, d_r, 1, &r_dot_r_old);
        
        // CG iteration
        for (int iter = 0; iter < max_iter; iter++) {
            // Ap = A * p
            alpha = 1.0f; beta = 0.0f;
            cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, mat_laplacian, vec_p, &beta, vec_Ap,
                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
            
            // alpha = r_dot_r_old / (p^T * Ap)
            float p_dot_Ap;
            cublasSdot(cublas_handle, n_total, d_p_vec, 1, d_Ap, 1, &p_dot_Ap);
            alpha = r_dot_r_old / p_dot_Ap;
            
            // x = x + alpha * p
            cublasSaxpy(cublas_handle, n_total, &alpha, d_p_vec, 1, d_p, 1);
            
            // r = r - alpha * Ap
            alpha = -alpha;
            cublasSaxpy(cublas_handle, n_total, &alpha, d_Ap, 1, d_r, 1);
            
            // r_dot_r_new = r^T * r
            cublasSdot(cublas_handle, n_total, d_r, 1, d_r, 1, &r_dot_r_new);
            
            // Check convergence
            if (sqrt(r_dot_r_new) < tolerance) {
                break;
            }
            
            // beta = r_dot_r_new / r_dot_r_old
            beta = r_dot_r_new / r_dot_r_old;
            
            // p = r + beta * p
            // First scale p by beta
            cublasSscal(cublas_handle, n_total, &beta, d_p_vec, 1);
            // Then add r
            alpha = 1.0f;
            cublasSaxpy(cublas_handle, n_total, &alpha, d_r, 1, d_p_vec, 1);
            
            r_dot_r_old = r_dot_r_new;
        }
        
        // Copy result to output
        cudaMemcpy(d_pressure_out, d_p, n_total * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Cleanup
        cudaFree(d_buffer);
        cusparseDestroyDnVec(vec_x);
        cusparseDestroyDnVec(vec_b);
        cusparseDestroyDnVec(vec_r);
        cusparseDestroyDnVec(vec_p);
        cusparseDestroyDnVec(vec_Ap);
    }
};

// Global solver instance (persistent across frames)
static PressureSolverCG* g_solver = nullptr;

// Python interface functions
torch::Tensor compute_divergence(
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor w,
    float dx
) {
    const int nx = u.size(0);
    const int ny = u.size(1);
    const int nz = u.size(2);
    
    auto div = torch::zeros_like(u);
    
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    
    compute_divergence_kernel<<<blocks, threads>>>(
        u.data_ptr<float>(),
        v.data_ptr<float>(),
        w.data_ptr<float>(),
        div.data_ptr<float>(),
        nx, ny, nz,
        1.0f / dx
    );
    
    cudaDeviceSynchronize();
    return div;
}

torch::Tensor solve_pressure_poisson(
    torch::Tensor divergence,
    torch::Tensor pressure_guess,
    float dx,
    int max_iter,
    float tolerance
) {
    const int nx = divergence.size(0);
    const int ny = divergence.size(1);
    const int nz = divergence.size(2);
    
    // Initialize solver on first call
    if (g_solver == nullptr) {
        g_solver = new PressureSolverCG(nx, ny, nz, dx);
    }
    
    auto pressure = torch::zeros_like(divergence);
    
    g_solver->solve(
        divergence.data_ptr<float>(),
        pressure.data_ptr<float>(),
        pressure_guess.data_ptr<float>(),
        max_iter,
        tolerance
    );
    
    return pressure;
}

void apply_pressure_gradient(
    torch::Tensor u,
    torch::Tensor v,
    torch::Tensor w,
    torch::Tensor pressure,
    float dx
) {
    const int nx = u.size(0);
    const int ny = u.size(1);
    const int nz = u.size(2);
    
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    
    apply_pressure_gradient_kernel<<<blocks, threads>>>(
        u.data_ptr<float>(),
        v.data_ptr<float>(),
        w.data_ptr<float>(),
        pressure.data_ptr<float>(),
        nx, ny, nz,
        1.0f / dx
    );
    
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_divergence", &compute_divergence, "Compute velocity divergence (CUDA)");
    m.def("solve_pressure_poisson", &solve_pressure_poisson, "Solve Poisson equation with CG (cuSparse)");
    m.def("apply_pressure_gradient", &apply_pressure_gradient, "Apply pressure gradient to velocity (CUDA)");
}
