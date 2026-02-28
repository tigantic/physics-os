/*
 * Phase 2B-2: CUDA Advection Kernel
 * ==================================
 * 
 * Parallel Semi-Lagrangian advection for 2D scalar fields.
 * Runs one thread per grid cell for massive parallelism.
 * 
 * Target: RTX 5070 (2048 CUDA cores)
 * Expected: <1ms for 512x512 grid
 * 
 * Algorithm:
 * 1. Backtrace: Find where particle came from (x - u*dt, y - v*dt)
 * 2. Clamp: Keep within grid boundaries
 * 3. Interpolate: Bilinear interpolation at source position
 * 4. Write: Store result at current position
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
} while(0)

/*
 * 2D Advection Kernel
 * 
 * Each thread handles one grid cell.
 * Uses Semi-Lagrangian method: trace back along velocity, interpolate.
 */
__global__ void advect_2d_kernel(
    const float* __restrict__ density_in,
    const float* __restrict__ u_vel,
    const float* __restrict__ v_vel,
    float* __restrict__ density_out,
    int width,
    int height,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // 1. Backtrace: Where did this particle come from?
    float src_x = (float)x - u_vel[idx] * dt;
    float src_y = (float)y - v_vel[idx] * dt;
    
    // 2. Clamp to grid boundaries
    src_x = fmaxf(0.0f, fminf(src_x, (float)(width - 1) - 0.001f));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(height - 1) - 0.001f));
    
    // 3. Bilinear interpolation
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    
    // Sample four corners
    float v00 = density_in[y0 * width + x0];
    float v10 = density_in[y0 * width + x1];
    float v01 = density_in[y1 * width + x0];
    float v11 = density_in[y1 * width + x1];
    
    // Interpolate horizontally, then vertically
    float top = v00 * (1.0f - dx) + v10 * dx;
    float bot = v01 * (1.0f - dx) + v11 * dx;
    
    // 4. Write result
    density_out[idx] = top * (1.0f - dy) + bot * dy;
}


/*
 * 3D Advection Kernel
 * 
 * Extension to 3D grids for volumetric simulations.
 */
__global__ void advect_3d_kernel(
    const float* __restrict__ density_in,
    const float* __restrict__ u_vel,
    const float* __restrict__ v_vel,
    const float* __restrict__ w_vel,
    float* __restrict__ density_out,
    int width,
    int height,
    int depth,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int idx = z * width * height + y * width + x;
    
    // Backtrace
    float src_x = (float)x - u_vel[idx] * dt;
    float src_y = (float)y - v_vel[idx] * dt;
    float src_z = (float)z - w_vel[idx] * dt;
    
    // Clamp
    src_x = fmaxf(0.0f, fminf(src_x, (float)(width - 1) - 0.001f));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(height - 1) - 0.001f));
    src_z = fmaxf(0.0f, fminf(src_z, (float)(depth - 1) - 0.001f));
    
    // Trilinear interpolation
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int z0 = (int)src_z;
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    int z1 = min(z0 + 1, depth - 1);
    
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float dz = src_z - (float)z0;
    
    // Sample 8 corners
    float v000 = density_in[z0 * width * height + y0 * width + x0];
    float v100 = density_in[z0 * width * height + y0 * width + x1];
    float v010 = density_in[z0 * width * height + y1 * width + x0];
    float v110 = density_in[z0 * width * height + y1 * width + x1];
    float v001 = density_in[z1 * width * height + y0 * width + x0];
    float v101 = density_in[z1 * width * height + y0 * width + x1];
    float v011 = density_in[z1 * width * height + y1 * width + x0];
    float v111 = density_in[z1 * width * height + y1 * width + x1];
    
    // Interpolate
    float c00 = v000 * (1.0f - dx) + v100 * dx;
    float c10 = v010 * (1.0f - dx) + v110 * dx;
    float c01 = v001 * (1.0f - dx) + v101 * dx;
    float c11 = v011 * (1.0f - dx) + v111 * dx;
    
    float c0 = c00 * (1.0f - dy) + c10 * dy;
    float c1 = c01 * (1.0f - dy) + c11 * dy;
    
    density_out[idx] = c0 * (1.0f - dz) + c1 * dz;
}


/*
 * Vector Advection Kernel (for velocity self-advection)
 * 
 * Advects a 2D vector field (u, v) by itself.
 * Used for: u^{n+1} = u^n - dt * (u · ∇)u
 */
__global__ void advect_velocity_2d_kernel(
    const float* __restrict__ u_in,
    const float* __restrict__ v_in,
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    int width,
    int height,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Backtrace using current velocity
    float src_x = (float)x - u_in[idx] * dt;
    float src_y = (float)y - v_in[idx] * dt;
    
    // Clamp
    src_x = fmaxf(0.0f, fminf(src_x, (float)(width - 1) - 0.001f));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(height - 1) - 0.001f));
    
    // Bilinear interpolation setup
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);
    
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    
    // Interpolate u component
    float u00 = u_in[y0 * width + x0];
    float u10 = u_in[y0 * width + x1];
    float u01 = u_in[y1 * width + x0];
    float u11 = u_in[y1 * width + x1];
    
    float u_top = u00 * (1.0f - dx) + u10 * dx;
    float u_bot = u01 * (1.0f - dx) + u11 * dx;
    u_out[idx] = u_top * (1.0f - dy) + u_bot * dy;
    
    // Interpolate v component
    float v00 = v_in[y0 * width + x0];
    float v10 = v_in[y0 * width + x1];
    float v01 = v_in[y1 * width + x0];
    float v11 = v_in[y1 * width + x1];
    
    float v_top = v00 * (1.0f - dx) + v10 * dx;
    float v_bot = v01 * (1.0f - dx) + v11 * dx;
    v_out[idx] = v_top * (1.0f - dy) + v_bot * dy;
}


// ═══════════════════════════════════════════════════════════════════════════
// C++ Launch Functions (called from Python bindings)
// ═══════════════════════════════════════════════════════════════════════════

/*
 * Launch 2D advection kernel
 * 
 * Args:
 *   density: Input scalar field [H, W]
 *   velocity: Velocity field [2, H, W] where [0] = u, [1] = v
 *   dt: Time step
 * 
 * Returns:
 *   Advected density field [H, W]
 */
torch::Tensor launch_advect_2d(
    torch::Tensor density,
    torch::Tensor velocity,
    float dt
) {
    // Validate inputs
    TORCH_CHECK(density.device().is_cuda(), "density must be on CUDA");
    TORCH_CHECK(velocity.device().is_cuda(), "velocity must be on CUDA");
    TORCH_CHECK(density.dtype() == torch::kFloat32, "density must be float32");
    TORCH_CHECK(velocity.dtype() == torch::kFloat32, "velocity must be float32");
    TORCH_CHECK(density.dim() == 2, "density must be 2D");
    TORCH_CHECK(velocity.dim() == 3, "velocity must be 3D [2, H, W]");
    TORCH_CHECK(velocity.size(0) == 2, "velocity must have 2 components");
    
    int height = density.size(0);
    int width = density.size(1);
    
    // Allocate output
    auto output = torch::empty_like(density);
    
    // Configure grid/blocks
    const dim3 threads(16, 16);
    const dim3 blocks((width + 15) / 16, (height + 15) / 16);
    
    // Get contiguous tensors
    auto density_c = density.contiguous();
    auto velocity_c = velocity.contiguous();
    
    // Launch kernel
    advect_2d_kernel<<<blocks, threads>>>(
        density_c.data_ptr<float>(),
        velocity_c.select(0, 0).data_ptr<float>(),  // u component
        velocity_c.select(0, 1).data_ptr<float>(),  // v component
        output.data_ptr<float>(),
        width,
        height,
        dt
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}


/*
 * Launch velocity self-advection kernel
 * 
 * Args:
 *   velocity: Velocity field [2, H, W]
 *   dt: Time step
 * 
 * Returns:
 *   Advected velocity field [2, H, W]
 */
torch::Tensor launch_advect_velocity_2d(
    torch::Tensor velocity,
    float dt
) {
    TORCH_CHECK(velocity.device().is_cuda(), "velocity must be on CUDA");
    TORCH_CHECK(velocity.dtype() == torch::kFloat32, "velocity must be float32");
    TORCH_CHECK(velocity.dim() == 3 && velocity.size(0) == 2, "velocity must be [2, H, W]");
    
    int height = velocity.size(1);
    int width = velocity.size(2);
    
    auto output = torch::empty_like(velocity);
    auto velocity_c = velocity.contiguous();
    
    const dim3 threads(16, 16);
    const dim3 blocks((width + 15) / 16, (height + 15) / 16);
    
    advect_velocity_2d_kernel<<<blocks, threads>>>(
        velocity_c.select(0, 0).data_ptr<float>(),
        velocity_c.select(0, 1).data_ptr<float>(),
        output.select(0, 0).data_ptr<float>(),
        output.select(0, 1).data_ptr<float>(),
        width,
        height,
        dt
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}


/*
 * Launch 3D advection kernel
 */
torch::Tensor launch_advect_3d(
    torch::Tensor density,
    torch::Tensor velocity,
    float dt
) {
    TORCH_CHECK(density.device().is_cuda(), "density must be on CUDA");
    TORCH_CHECK(velocity.device().is_cuda(), "velocity must be on CUDA");
    TORCH_CHECK(density.dtype() == torch::kFloat32, "density must be float32");
    TORCH_CHECK(velocity.dim() == 4 && velocity.size(0) == 3, "velocity must be [3, D, H, W]");
    
    int depth = density.size(0);
    int height = density.size(1);
    int width = density.size(2);
    
    auto output = torch::empty_like(density);
    auto density_c = density.contiguous();
    auto velocity_c = velocity.contiguous();
    
    const dim3 threads(8, 8, 8);
    const dim3 blocks(
        (width + 7) / 8,
        (height + 7) / 8,
        (depth + 7) / 8
    );
    
    advect_3d_kernel<<<blocks, threads>>>(
        density_c.data_ptr<float>(),
        velocity_c.select(0, 0).data_ptr<float>(),
        velocity_c.select(0, 1).data_ptr<float>(),
        velocity_c.select(0, 2).data_ptr<float>(),
        output.data_ptr<float>(),
        width,
        height,
        depth,
        dt
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    return output;
}
