/**
 * QTT Native CUDA Kernels - Fused Operations for CFD
 * ===================================================
 * 
 * High-performance CUDA kernels for QTT arithmetic:
 * 1. Batched core contraction (MPO × MPS)
 * 2. Fused Jacobi iteration (shift + add + scale)
 * 3. QTT inner product (parallel reduction)
 * 4. Batched truncation SVD
 * 
 * Design Principles:
 * - NO DENSE: All operations stay in O(L × r³) format
 * - Fused kernels eliminate Python loop overhead
 * - Shared memory for core caching (cores are small: ~r² × d)
 * - Warp-level primitives for reductions
 * 
 * Target: 100× speedup over CPU sequential contractions
 * 
 * Author: HyperTensor Team
 * Date: January 2026
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

// ============================================================================
// Constants
// ============================================================================

#define WARP_SIZE 32
#define MAX_RANK 256      // Maximum bond dimension
#define MAX_CORES 32      // Maximum L (qubits)
#define PHYS_DIM 2        // QTT physical dimension (qubit)
#define BLOCK_SIZE 256

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Warp-level sum reduction
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Block-level sum reduction using shared memory
 */
template<typename T>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    // Warp reduction
    val = warp_reduce_sum(val);
    
    // Write warp result to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // First warp reduces all warp results
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// ============================================================================
// QTT Inner Product Kernel
// ============================================================================

/**
 * Fused QTT inner product: <a|b> = sum_i a_i * b_i
 * 
 * Computes by contracting all L cores in sequence.
 * Each block handles one core, with threads doing parallel matrix multiply.
 * 
 * Algorithm:
 *   env = ones(1,1)
 *   for i in 0..L-1:
 *     env[ra_r, rb_r] = sum_{ra_l, rb_l, d} env[ra_l, rb_l] * a[ra_l, d, ra_r] * b[rb_l, d, rb_r]
 *   return env[0,0]
 * 
 * Parallelization: threads parallelize over (ra_r, rb_r) output elements.
 */
template<typename scalar_t>
__global__ void qtt_inner_product_kernel(
    const scalar_t* __restrict__ a_cores_flat,  // All cores concatenated
    const scalar_t* __restrict__ b_cores_flat,
    const int* __restrict__ core_offsets,       // [L+1] start of each core
    const int* __restrict__ core_shapes,        // [L, 3] (r_left, d, r_right)
    scalar_t* __restrict__ env,                 // Working buffer [MAX_RANK, MAX_RANK]
    scalar_t* __restrict__ output,              // [1] scalar result
    int num_cores
) {
    extern __shared__ char shared_mem[];
    scalar_t* shared_env = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Initialize environment to identity
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        env[0] = scalar_t(1.0);
    }
    __syncthreads();
    
    // Sequential over cores (each block does one core's contribution)
    for (int core_idx = 0; core_idx < num_cores; ++core_idx) {
        int offset_a = core_offsets[core_idx];
        int offset_b = core_offsets[core_idx];
        
        int r_left = core_shapes[core_idx * 3 + 0];
        int d = core_shapes[core_idx * 3 + 1];
        int r_right = core_shapes[core_idx * 3 + 2];
        
        // Parallelize over (ra_r, rb_r) output pairs
        int num_outputs = r_right * r_right;
        
        for (int out_idx = threadIdx.x; out_idx < num_outputs; out_idx += blockDim.x) {
            int ra_r = out_idx / r_right;
            int rb_r = out_idx % r_right;
            
            scalar_t sum = scalar_t(0.0);
            
            // Contract over (ra_l, rb_l, d)
            for (int ra_l = 0; ra_l < r_left; ++ra_l) {
                for (int rb_l = 0; rb_l < r_left; ++rb_l) {
                    scalar_t env_val = (core_idx == 0) ? 
                        ((ra_l == 0 && rb_l == 0) ? scalar_t(1.0) : scalar_t(0.0)) :
                        env[ra_l * r_left + rb_l];  // Previous env
                    
                    for (int di = 0; di < d; ++di) {
                        // a[ra_l, d, ra_r], b[rb_l, d, rb_r]
                        int idx_a = offset_a + ra_l * d * r_right + di * r_right + ra_r;
                        int idx_b = offset_b + rb_l * d * r_right + di * r_right + rb_r;
                        
                        sum += env_val * a_cores_flat[idx_a] * b_cores_flat[idx_b];
                    }
                }
            }
            
            shared_env[out_idx] = sum;
        }
        
        __syncthreads();
        
        // Copy shared to global env for next iteration
        for (int i = threadIdx.x; i < num_outputs; i += blockDim.x) {
            env[i] = shared_env[i];
        }
        
        __syncthreads();
    }
    
    // Final result is env[0]
    if (threadIdx.x == 0) {
        output[0] = env[0];
    }
}

// ============================================================================
// Batched MPO Application Kernel
// ============================================================================

/**
 * Apply MPO to QTT: result = MPO @ QTT
 * 
 * Each core contraction: O[rLo, d_out, d_in, rRo] × P[rLp, d_in, rRp]
 *                      → R[rLo*rLp, d_out, rRo*rRp]
 * 
 * This kernel processes all L cores in one launch.
 * Each block handles one core.
 */
template<typename scalar_t>
__global__ void apply_mpo_batched_kernel(
    const scalar_t* __restrict__ mpo_flat,      // All MPO cores
    const scalar_t* __restrict__ qtt_flat,      // All QTT cores
    scalar_t* __restrict__ result_flat,         // Output cores
    const int* __restrict__ mpo_offsets,        // [L+1]
    const int* __restrict__ qtt_offsets,        // [L+1]
    const int* __restrict__ result_offsets,     // [L+1]
    const int* __restrict__ mpo_shapes,         // [L, 4]: (rLo, d_out, d_in, rRo)
    const int* __restrict__ qtt_shapes,         // [L, 3]: (rLp, d_in, rRp)
    int num_cores
) {
    int core_idx = blockIdx.x;
    if (core_idx >= num_cores) return;
    
    // Get shapes
    int rLo = mpo_shapes[core_idx * 4 + 0];
    int d_out = mpo_shapes[core_idx * 4 + 1];
    int d_in = mpo_shapes[core_idx * 4 + 2];
    int rRo = mpo_shapes[core_idx * 4 + 3];
    
    int rLp = qtt_shapes[core_idx * 3 + 0];
    int rRp = qtt_shapes[core_idx * 3 + 2];
    
    // Output shape: (rLo*rLp, d_out, rRo*rRp)
    int rL_out = rLo * rLp;
    int rR_out = rRo * rRp;
    int output_size = rL_out * d_out * rR_out;
    
    // Get data pointers
    const scalar_t* mpo_core = mpo_flat + mpo_offsets[core_idx];
    const scalar_t* qtt_core = qtt_flat + qtt_offsets[core_idx];
    scalar_t* result_core = result_flat + result_offsets[core_idx];
    
    // Parallelize over output elements
    for (int out_idx = threadIdx.x; out_idx < output_size; out_idx += blockDim.x) {
        // Decode output index: (rL_out, d_out, rR_out)
        int rR = out_idx % rR_out;
        int tmp = out_idx / rR_out;
        int d = tmp % d_out;
        int rL = tmp / d_out;
        
        // Decode combined indices
        int o = rL / rLp;  // MPO left
        int p = rL % rLp;  // QTT left
        int r = rR / rRp;  // MPO right
        int q = rR % rRp;  // QTT right
        
        // Contract over d_in (b index)
        scalar_t sum = scalar_t(0.0);
        for (int b = 0; b < d_in; ++b) {
            // mpo[o, d, b, r], qtt[p, b, q]
            int mpo_idx = o * d_out * d_in * rRo + d * d_in * rRo + b * rRo + r;
            int qtt_idx = p * d_in * rRp + b * rRp + q;
            
            sum += mpo_core[mpo_idx] * qtt_core[qtt_idx];
        }
        
        result_core[out_idx] = sum;
    }
}

// ============================================================================
// QTT Add Kernel (Block-Diagonal Concatenation)
// ============================================================================

/**
 * Add two QTT states by block-diagonal core concatenation.
 * 
 * For middle cores: new_core = [[c1, 0], [0, c2]]
 * First core: cat along right
 * Last core: cat along left
 */
template<typename scalar_t>
__global__ void qtt_add_kernel(
    const scalar_t* __restrict__ qtt1_flat,
    const scalar_t* __restrict__ qtt2_flat,
    scalar_t* __restrict__ result_flat,
    const int* __restrict__ qtt1_offsets,
    const int* __restrict__ qtt2_offsets,
    const int* __restrict__ result_offsets,
    const int* __restrict__ qtt1_shapes,  // [L, 3]
    const int* __restrict__ qtt2_shapes,  // [L, 3]
    int num_cores
) {
    int core_idx = blockIdx.x;
    if (core_idx >= num_cores) return;
    
    // Get shapes
    int r1L = qtt1_shapes[core_idx * 3 + 0];
    int d = qtt1_shapes[core_idx * 3 + 1];
    int r1R = qtt1_shapes[core_idx * 3 + 2];
    
    int r2L = qtt2_shapes[core_idx * 3 + 0];
    int r2R = qtt2_shapes[core_idx * 3 + 2];
    
    const scalar_t* c1 = qtt1_flat + qtt1_offsets[core_idx];
    const scalar_t* c2 = qtt2_flat + qtt2_offsets[core_idx];
    scalar_t* result = result_flat + result_offsets[core_idx];
    
    bool is_first = (core_idx == 0);
    bool is_last = (core_idx == num_cores - 1);
    
    if (is_first) {
        // Cat along right: (r1L, d, r1R + r2R)
        int rL = r1L;  // Same
        int rR = r1R + r2R;
        int out_size = rL * d * rR;
        
        for (int i = threadIdx.x; i < out_size; i += blockDim.x) {
            int r = i % rR;
            int tmp = i / rR;
            int di = tmp % d;
            int l = tmp / d;
            
            if (r < r1R) {
                // From c1
                result[i] = c1[l * d * r1R + di * r1R + r];
            } else {
                // From c2
                result[i] = c2[l * d * r2R + di * r2R + (r - r1R)];
            }
        }
    } else if (is_last) {
        // Cat along left: (r1L + r2L, d, r1R)
        int rL = r1L + r2L;
        int rR = r1R;  // Same
        int out_size = rL * d * rR;
        
        for (int i = threadIdx.x; i < out_size; i += blockDim.x) {
            int r = i % rR;
            int tmp = i / rR;
            int di = tmp % d;
            int l = tmp / d;
            
            if (l < r1L) {
                result[i] = c1[l * d * r1R + di * r1R + r];
            } else {
                result[i] = c2[(l - r1L) * d * r2R + di * r2R + r];
            }
        }
    } else {
        // Block diagonal: (r1L + r2L, d, r1R + r2R)
        int rL = r1L + r2L;
        int rR = r1R + r2R;
        int out_size = rL * d * rR;
        
        for (int i = threadIdx.x; i < out_size; i += blockDim.x) {
            int r = i % rR;
            int tmp = i / rR;
            int di = tmp % d;
            int l = tmp / d;
            
            scalar_t val = scalar_t(0.0);
            
            // Check if in c1 block or c2 block
            if (l < r1L && r < r1R) {
                val = c1[l * d * r1R + di * r1R + r];
            } else if (l >= r1L && r >= r1R) {
                val = c2[(l - r1L) * d * r2R + di * r2R + (r - r1R)];
            }
            // Else: off-diagonal zeros
            
            result[i] = val;
        }
    }
}

// ============================================================================
// QTT Hadamard (Element-wise Product) Kernel
// ============================================================================

/**
 * Hadamard product: result[i] = a[i] * b[i]
 * 
 * In QTT: Kronecker product of cores at each site.
 * new_core[rL1*rL2, d, rR1*rR2] = kron(c1[rL1,d,rR1], c2[rL2,d,rR2])
 */
template<typename scalar_t>
__global__ void qtt_hadamard_kernel(
    const scalar_t* __restrict__ qtt1_flat,
    const scalar_t* __restrict__ qtt2_flat,
    scalar_t* __restrict__ result_flat,
    const int* __restrict__ qtt1_offsets,
    const int* __restrict__ qtt2_offsets,
    const int* __restrict__ result_offsets,
    const int* __restrict__ qtt1_shapes,
    const int* __restrict__ qtt2_shapes,
    int num_cores
) {
    int core_idx = blockIdx.x;
    if (core_idx >= num_cores) return;
    
    int r1L = qtt1_shapes[core_idx * 3 + 0];
    int d = qtt1_shapes[core_idx * 3 + 1];
    int r1R = qtt1_shapes[core_idx * 3 + 2];
    
    int r2L = qtt2_shapes[core_idx * 3 + 0];
    int r2R = qtt2_shapes[core_idx * 3 + 2];
    
    const scalar_t* c1 = qtt1_flat + qtt1_offsets[core_idx];
    const scalar_t* c2 = qtt2_flat + qtt2_offsets[core_idx];
    scalar_t* result = result_flat + result_offsets[core_idx];
    
    // Output shape: (r1L*r2L, d, r1R*r2R)
    int rL_out = r1L * r2L;
    int rR_out = r1R * r2R;
    int out_size = rL_out * d * rR_out;
    
    for (int i = threadIdx.x; i < out_size; i += blockDim.x) {
        int rR = i % rR_out;
        int tmp = i / rR_out;
        int di = tmp % d;
        int rL = tmp / d;
        
        // Decode Kronecker indices
        int l1 = rL / r2L;
        int l2 = rL % r2L;
        int r1 = rR / r2R;
        int r2 = rR % r2R;
        
        // c1[l1, di, r1] * c2[l2, di, r2]
        scalar_t v1 = c1[l1 * d * r1R + di * r1R + r1];
        scalar_t v2 = c2[l2 * d * r2R + di * r2R + r2];
        
        result[i] = v1 * v2;
    }
}

// ============================================================================
// Fused Jacobi Iteration Kernel
// ============================================================================

/**
 * Single Jacobi iteration for Poisson: ψ_new = (1/D)(rhs + neighbor_sum)
 * 
 * Fuses: 4 shifts + 4 scales + 3 adds + 1 final scale
 * All in QTT format via batched core operations.
 * 
 * This is the HOT LOOP - where 90% of solve time is spent.
 * 
 * NOTE: This kernel operates on pre-shifted cores (MPO already applied).
 * The shift MPOs should be cached and applied outside this kernel.
 * This kernel fuses the arithmetic: (psi_xp + psi_xm)/dx² + (psi_yp + psi_ym)/dy²
 */
template<typename scalar_t>
__global__ void jacobi_combine_kernel(
    const scalar_t* __restrict__ psi_xp_flat,   // Shifted +x
    const scalar_t* __restrict__ psi_xm_flat,   // Shifted -x
    const scalar_t* __restrict__ psi_yp_flat,   // Shifted +y
    const scalar_t* __restrict__ psi_ym_flat,   // Shifted -y
    const scalar_t* __restrict__ rhs_flat,      // RHS
    scalar_t* __restrict__ result_flat,         // Output ψ_new
    const int* __restrict__ offsets,            // Core offsets (same for all)
    const int* __restrict__ shapes,             // Core shapes
    scalar_t inv_dx2,                           // 1/dx²
    scalar_t inv_dy2,                           // 1/dy²
    scalar_t inv_D,                             // 1/(2/dx² + 2/dy²)
    int num_cores
) {
    int core_idx = blockIdx.x;
    if (core_idx >= num_cores) return;
    
    int rL = shapes[core_idx * 3 + 0];
    int d = shapes[core_idx * 3 + 1];
    int rR = shapes[core_idx * 3 + 2];
    int core_size = rL * d * rR;
    
    int offset = offsets[core_idx];
    
    // Linear combination for this core (first core gets the scalars)
    scalar_t scale_x = (core_idx == 0) ? inv_dx2 : scalar_t(1.0);
    scalar_t scale_y = (core_idx == 0) ? inv_dy2 : scalar_t(1.0);
    scalar_t scale_rhs = (core_idx == 0) ? scalar_t(1.0) : scalar_t(1.0);
    scalar_t scale_final = (core_idx == 0) ? inv_D : scalar_t(1.0);
    
    // Note: Full QTT arithmetic would require proper block-diagonal structure
    // This is a simplified version for when all inputs have SAME rank
    // For variable ranks, use qtt_add_kernel multiple times
    
    for (int i = threadIdx.x; i < core_size; i += blockDim.x) {
        // For same-rank case: element-wise sum after proper alignment
        // This is NOT correct for general QTT add - just a placeholder
        scalar_t xp = psi_xp_flat[offset + i];
        scalar_t xm = psi_xm_flat[offset + i];
        scalar_t yp = psi_yp_flat[offset + i];
        scalar_t ym = psi_ym_flat[offset + i];
        scalar_t r = rhs_flat[offset + i];
        
        // Combine: (xp + xm)*inv_dx2 + (yp + ym)*inv_dy2 + rhs, then * inv_D
        scalar_t val = (xp + xm) * scale_x + (yp + ym) * scale_y + r * scale_rhs;
        result_flat[offset + i] = val * scale_final;
    }
}

// ============================================================================
// Truncation via Randomized SVD (Batched)
// ============================================================================

/**
 * QR decomposition for truncation sweep (left-to-right)
 * 
 * For core[i]: reshape (rL*d, rR), compute Q,R, update core[i]=Q, core[i+1]=R@core[i+1]
 */
template<typename scalar_t>
__global__ void truncation_qr_kernel(
    scalar_t* __restrict__ cores_flat,
    const int* __restrict__ offsets,
    const int* __restrict__ shapes,
    int* __restrict__ new_shapes,           // Output: updated ranks
    int core_idx,                           // Which core to process
    int max_rank
) {
    // NOTE: Full QR in CUDA requires cuSOLVER
    // This is a placeholder - actual implementation uses torch.linalg.qr from Python
    // The key optimization is to do the sweep in a single Python call with batched ops
}

// ============================================================================
// Python/PyTorch Binding Wrappers
// ============================================================================

/**
 * QTT inner product - PyTorch wrapper
 */
torch::Tensor qtt_inner_product_cuda(
    torch::Tensor a_cores_flat,
    torch::Tensor b_cores_flat,
    torch::Tensor core_offsets,
    torch::Tensor core_shapes
) {
    int num_cores = core_shapes.size(0);
    auto options = a_cores_flat.options();
    
    // Allocate environment and output
    auto env = torch::zeros({MAX_RANK * MAX_RANK}, options);
    auto output = torch::zeros({1}, options);
    
    int shared_size = MAX_RANK * MAX_RANK * sizeof(float);
    if (a_cores_flat.dtype() == torch::kFloat64) {
        shared_size *= 2;
    }
    
    AT_DISPATCH_FLOATING_TYPES(a_cores_flat.scalar_type(), "qtt_inner_product_cuda", ([&] {
        qtt_inner_product_kernel<scalar_t><<<1, BLOCK_SIZE, shared_size>>>(
            a_cores_flat.data_ptr<scalar_t>(),
            b_cores_flat.data_ptr<scalar_t>(),
            core_offsets.data_ptr<int>(),
            core_shapes.data_ptr<int>(),
            env.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_cores
        );
    }));
    
    return output;
}

/**
 * Batched MPO application - PyTorch wrapper
 */
torch::Tensor apply_mpo_cuda(
    torch::Tensor mpo_flat,
    torch::Tensor qtt_flat,
    torch::Tensor mpo_offsets,
    torch::Tensor qtt_offsets,
    torch::Tensor result_offsets,
    torch::Tensor mpo_shapes,
    torch::Tensor qtt_shapes,
    int total_result_size
) {
    int num_cores = mpo_shapes.size(0);
    auto options = qtt_flat.options();
    
    auto result = torch::zeros({total_result_size}, options);
    
    AT_DISPATCH_FLOATING_TYPES(qtt_flat.scalar_type(), "apply_mpo_cuda", ([&] {
        apply_mpo_batched_kernel<scalar_t><<<num_cores, BLOCK_SIZE>>>(
            mpo_flat.data_ptr<scalar_t>(),
            qtt_flat.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            mpo_offsets.data_ptr<int>(),
            qtt_offsets.data_ptr<int>(),
            result_offsets.data_ptr<int>(),
            mpo_shapes.data_ptr<int>(),
            qtt_shapes.data_ptr<int>(),
            num_cores
        );
    }));
    
    return result;
}

/**
 * QTT add - PyTorch wrapper
 */
torch::Tensor qtt_add_cuda(
    torch::Tensor qtt1_flat,
    torch::Tensor qtt2_flat,
    torch::Tensor qtt1_offsets,
    torch::Tensor qtt2_offsets,
    torch::Tensor result_offsets,
    torch::Tensor qtt1_shapes,
    torch::Tensor qtt2_shapes,
    int total_result_size
) {
    int num_cores = qtt1_shapes.size(0);
    auto options = qtt1_flat.options();
    
    auto result = torch::zeros({total_result_size}, options);
    
    AT_DISPATCH_FLOATING_TYPES(qtt1_flat.scalar_type(), "qtt_add_cuda", ([&] {
        qtt_add_kernel<scalar_t><<<num_cores, BLOCK_SIZE>>>(
            qtt1_flat.data_ptr<scalar_t>(),
            qtt2_flat.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            qtt1_offsets.data_ptr<int>(),
            qtt2_offsets.data_ptr<int>(),
            result_offsets.data_ptr<int>(),
            qtt1_shapes.data_ptr<int>(),
            qtt2_shapes.data_ptr<int>(),
            num_cores
        );
    }));
    
    return result;
}

/**
 * QTT Hadamard - PyTorch wrapper
 */
torch::Tensor qtt_hadamard_cuda(
    torch::Tensor qtt1_flat,
    torch::Tensor qtt2_flat,
    torch::Tensor qtt1_offsets,
    torch::Tensor qtt2_offsets,
    torch::Tensor result_offsets,
    torch::Tensor qtt1_shapes,
    torch::Tensor qtt2_shapes,
    int total_result_size
) {
    int num_cores = qtt1_shapes.size(0);
    auto options = qtt1_flat.options();
    
    auto result = torch::zeros({total_result_size}, options);
    
    AT_DISPATCH_FLOATING_TYPES(qtt1_flat.scalar_type(), "qtt_hadamard_cuda", ([&] {
        qtt_hadamard_kernel<scalar_t><<<num_cores, BLOCK_SIZE>>>(
            qtt1_flat.data_ptr<scalar_t>(),
            qtt2_flat.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            qtt1_offsets.data_ptr<int>(),
            qtt2_offsets.data_ptr<int>(),
            result_offsets.data_ptr<int>(),
            qtt1_shapes.data_ptr<int>(),
            qtt2_shapes.data_ptr<int>(),
            num_cores
        );
    }));
    
    return result;
}

// ============================================================================
// Module Registration
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        QTT Native CUDA Kernels
        =======================
        
        Fused CUDA kernels for QTT operations in CFD:
        - qtt_inner_product_cuda: O(L*r²) inner product
        - apply_mpo_cuda: Batched MPO × QTT contraction
        - qtt_add_cuda: Block-diagonal core concatenation
        - qtt_hadamard_cuda: Kronecker product for element-wise multiply
        
        All operations stay in O(L × r³) - NO DENSE!
    )pbdoc";
    
    m.def("qtt_inner_product", &qtt_inner_product_cuda,
        "QTT inner product <a|b> via core contraction",
        py::arg("a_cores_flat"),
        py::arg("b_cores_flat"),
        py::arg("core_offsets"),
        py::arg("core_shapes"));
    
    m.def("apply_mpo", &apply_mpo_cuda,
        "Apply MPO to QTT (batched over all cores)",
        py::arg("mpo_flat"),
        py::arg("qtt_flat"),
        py::arg("mpo_offsets"),
        py::arg("qtt_offsets"),
        py::arg("result_offsets"),
        py::arg("mpo_shapes"),
        py::arg("qtt_shapes"),
        py::arg("total_result_size"));
    
    m.def("qtt_add", &qtt_add_cuda,
        "Add two QTT states via block-diagonal concatenation",
        py::arg("qtt1_flat"),
        py::arg("qtt2_flat"),
        py::arg("qtt1_offsets"),
        py::arg("qtt2_offsets"),
        py::arg("result_offsets"),
        py::arg("qtt1_shapes"),
        py::arg("qtt2_shapes"),
        py::arg("total_result_size"));
    
    m.def("qtt_hadamard", &qtt_hadamard_cuda,
        "Element-wise product via Kronecker core product",
        py::arg("qtt1_flat"),
        py::arg("qtt2_flat"),
        py::arg("qtt1_offsets"),
        py::arg("qtt2_offsets"),
        py::arg("result_offsets"),
        py::arg("qtt1_shapes"),
        py::arg("qtt2_shapes"),
        py::arg("total_result_size"));
}
