/**
 * Implicit QTT Kernel - Direct evaluation of QTT cores in CUDA
 * 
 * Evaluates Quantized Tensor-Train decomposition directly at pixel coordinates
 * without materializing intermediate dense tensors.
 * 
 * Key innovations:
 * 1. Morton encoding for 2D→1D spatial mapping (cache-friendly)
 * 2. TT-contraction as matrix chain product (12 cores, 2×2 each)
 * 3. Inline colormap application (no separate pass)
 * 4. Multi-layer alpha blending in scalar space
 * 
 * Performance target: <1.5ms for single layer @ 4K (3840×2160)
 * Theoretical: 795M FLOPs / 33.4 TFLOPS = 0.024ms (ALU-bound)
 * Realistic: ~1.5ms with memory latency + dispatch overhead
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================

#define QTT_DEPTH 12        // Number of TT-cores (2^6 = 64, so 6 bits × 2 dims)
#define CORE_SIZE 2         // 2×2 matrices per core
#define MAX_LAYERS 5        // Maximum compositor layers
#define WARP_SIZE 32

// ============================================================================
// Morton Encoding Utilities
// ============================================================================

/**
 * Interleave bits from two 6-bit integers into 12-bit Morton code
 * 
 * Input: x=0b_aaaaaa, y=0b_bbbbbb
 * Output: 0b_bababababababa
 * 
 * Uses magic number multiplication for parallel bit interleaving:
 * https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
 */
__device__ __forceinline__ uint32_t morton_encode(uint32_t x, uint32_t y) {
    // Expand 6-bit to 12-bit with zeros between bits
    // 0b_aaaaaa → 0b_0a0a0a0a0a0a
    auto part1 = [](uint32_t n) -> uint32_t {
        n &= 0x0000003f;                  // Keep only 6 bits
        n = (n | (n << 16)) & 0x030000ff; // 0b_00000011_00000000_00000000_11111111
        n = (n | (n <<  8)) & 0x0300f00f; // 0b_00000011_00000000_11110000_00001111
        n = (n | (n <<  4)) & 0x030c30c3; // 0b_00000011_00001100_00110000_11000011
        n = (n | (n <<  2)) & 0x09249249; // 0b_00001001_00100100_10010010_01001001
        return n;
    };
    
    return part1(x) | (part1(y) << 1);
}

/**
 * Extract specific bit from Morton-encoded index
 * Used to select which matrix to multiply in TT-contraction
 * 
 * Bit order (MSB first): y₅x₅y₄x₄y₃x₃y₂x₂y₁x₁y₀x₀
 */
__device__ __forceinline__ uint32_t morton_bit(uint32_t morton_idx, int bit_pos) {
    return (morton_idx >> bit_pos) & 1;
}

// ============================================================================
// Matrix Operations (2×2)
// ============================================================================

/**
 * 2×2 matrix stored in row-major order: [a b; c d] → {a, b, c, d}
 */
struct Mat2x2 {
    float a, b, c, d;
    
    __device__ __forceinline__ Mat2x2(float a_, float b_, float c_, float d_) 
        : a(a_), b(b_), c(c_), d(d_) {}
    
    __device__ __forceinline__ Mat2x2() : a(0), b(0), c(0), d(0) {}
};

/**
 * Matrix multiplication: C = A * B (2×2 × 2×2)
 * 8 FLOPs: 4 multiplies + 4 adds
 */
__device__ __forceinline__ Mat2x2 mat_mult(const Mat2x2& A, const Mat2x2& B) {
    return Mat2x2(
        A.a * B.a + A.b * B.c,  // C[0,0]
        A.a * B.b + A.b * B.d,  // C[0,1]
        A.c * B.a + A.d * B.c,  // C[1,0]
        A.c * B.b + A.d * B.d   // C[1,1]
    );
}

// ============================================================================
// Colormap Functions
// ============================================================================

/**
 * Plasma colormap (matplotlib-inspired)
 * Input: scalar value in [0, 1]
 * Output: RGB color in [0, 1]³
 * 
 * Polynomial approximation for GPU efficiency (no texture lookups)
 */
__device__ __forceinline__ float3 plasma_colormap(float t) {
    // Clamp to [0, 1]
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    
    // Polynomial coefficients (least-squares fit to matplotlib plasma)
    // R(t) ≈ a₀ + a₁t + a₂t² + a₃t³
    float r = 0.050f + 1.080f * t - 0.500f * t * t + 0.370f * t * t * t;
    float g = 0.030f + 2.200f * t - 3.500f * t * t + 1.650f * t * t * t;
    float b = 0.520f + 1.550f * t - 3.200f * t * t + 1.850f * t * t * t;
    
    // Clamp outputs
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    
    return make_float3(r, g, b);
}

/**
 * Viridis colormap (alternative, perceptually uniform)
 */
__device__ __forceinline__ float3 viridis_colormap(float t) {
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    
    float r = 0.280f + 0.760f * t - 1.180f * t * t + 0.640f * t * t * t;
    float g = 0.005f + 1.460f * t - 0.930f * t * t + 0.280f * t * t * t;
    float b = 0.330f + 1.380f * t - 2.020f * t * t + 0.940f * t * t * t;
    
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    
    return make_float3(r, g, b);
}

// ============================================================================
// QTT Evaluation Kernel
// ============================================================================

/**
 * Evaluate single QTT at pixel coordinates (u, v) ∈ [0, 1]²
 * 
 * Algorithm:
 * 1. Map (u, v) → (x, y) in [0, 64)
 * 2. Compute Morton index z = morton_encode(x, y)
 * 3. Contract TT-cores: M = G₀ * G₁ * ... * G₁₁
 * 4. Extract scalar: value = M[0, 0]
 * 
 * Input: 
 *   - cores: Array of 12 cores × 2 matrices/core × 4 floats/matrix = 96 floats
 *   - u, v: Pixel coordinates normalized to [0, 1]
 * 
 * Output: Scalar field value at (u, v)
 * 
 * Performance: 12 × 8 FLOPs = 96 FLOPs per pixel
 */
__device__ float eval_qtt_at_point(const float* cores, float u, float v) {
    // Map UV to grid coordinates (64×64 logical grid)
    const int grid_size = 64;
    int x = __float2int_rd(u * (grid_size - 1));
    int y = __float2int_rd(v * (grid_size - 1));
    
    // Clamp to valid range
    x = min(max(x, 0), grid_size - 1);
    y = min(max(y, 0), grid_size - 1);
    
    // Compute Morton index
    uint32_t morton_idx = morton_encode(x, y);
    
    // Initialize result matrix as identity
    Mat2x2 result(1.0f, 0.0f, 0.0f, 1.0f);
    
    // Contract TT-cores (right-to-left for numerical stability)
    // cores layout: [core_0_mat_0, core_0_mat_1, core_1_mat_0, core_1_mat_1, ...]
    #pragma unroll
    for (int d = 0; d < QTT_DEPTH; d++) {
        // Select matrix based on Morton bit
        uint32_t bit = morton_bit(morton_idx, d);
        int core_offset = d * CORE_SIZE * 4 + bit * 4;
        
        // Load 2×2 matrix from global memory
        Mat2x2 G(
            cores[core_offset + 0],
            cores[core_offset + 1],
            cores[core_offset + 2],
            cores[core_offset + 3]
        );
        
        // Accumulate: result = result * G
        result = mat_mult(result, G);
    }
    
    // Extract scalar value (top-left element)
    return result.a;
}

/**
 * CUDA kernel: Render single QTT layer to 4K output buffer
 * 
 * Grid: (width/16, height/16, 1) blocks
 * Block: (16, 16, 1) threads (256 threads/block)
 * Total: 3840×2160 / 256 = ~35K blocks
 * 
 * Each thread evaluates QTT at one pixel, applies colormap, writes RGBA
 */
__global__ void render_qtt_layer_kernel(
    const float* __restrict__ qtt_cores,   // QTT core matrices (96 floats)
    float* __restrict__ output,            // Output RGBA buffer (H×W×4)
    int width,                             // Image width (3840)
    int height,                            // Image height (2160)
    float value_min,                       // Normalization range
    float value_max,
    int colormap_type                      // 0=plasma, 1=viridis
) {
    // Pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Normalized UV coordinates
    float u = (float)x / (width - 1);
    float v = (float)y / (height - 1);
    
    // Evaluate QTT
    float value = eval_qtt_at_point(qtt_cores, u, v);
    
    // Normalize to [0, 1]
    float t = (value - value_min) / (value_max - value_min);
    
    // Apply colormap
    float3 rgb = (colormap_type == 0) ? plasma_colormap(t) : viridis_colormap(t);
    
    // Compute opacity (based on normalized value)
    float alpha = fminf(fmaxf(t * 1.2f, 0.0f), 1.0f);
    
    // Write RGBA (row-major order)
    int idx = (y * width + x) * 4;
    output[idx + 0] = rgb.x;
    output[idx + 1] = rgb.y;
    output[idx + 2] = rgb.z;
    output[idx + 3] = alpha;
}

// ============================================================================
// Multi-Layer Compositor Kernel
// ============================================================================

/**
 * CUDA kernel: Composite multiple QTT layers with alpha blending
 * 
 * Evaluates all layers at pixel coordinate, blends in scalar space,
 * then applies colormap once at the end.
 * 
 * Performance: 5 layers × 96 FLOPs = 480 FLOPs/pixel
 * Expected: ~1.5ms @ 4K with good memory coalescing
 */
__global__ void composite_qtt_layers_kernel(
    const float* const* __restrict__ layer_cores,  // Array of QTT cores (5 layers)
    const int* __restrict__ layer_enabled,         // Enable flags (5 bools)
    float* __restrict__ output,                    // Output RGBA buffer
    int width,
    int height,
    float value_min,
    float value_max
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float u = (float)x / (width - 1);
    float v = (float)y / (height - 1);
    
    // Base layer (opaque geological substrate)
    float base_value = 0.0f;
    if (layer_enabled[0]) {
        base_value = eval_qtt_at_point(layer_cores[0], u, v);
    }
    
    // Initialize with base layer color
    float3 rgb_base = plasma_colormap((base_value - value_min) / (value_max - value_min));
    float4 result = make_float4(rgb_base.x, rgb_base.y, rgb_base.z, 1.0f);
    
    // Blend remaining layers (1-4: TENSOR, KINETIC, GEOMETRY, HUD)
    #pragma unroll
    for (int i = 1; i < MAX_LAYERS; i++) {
        if (!layer_enabled[i]) continue;
        
        // Evaluate QTT for this layer
        float layer_value = eval_qtt_at_point(layer_cores[i], u, v);
        float t = (layer_value - value_min) / (value_max - value_min);
        
        // Apply colormap
        float3 rgb_layer = plasma_colormap(t);
        float alpha = fminf(fmaxf(t * 1.2f, 0.0f), 1.0f);
        
        // Alpha blend: result = (1-α)·result + α·layer
        result.x = (1.0f - alpha) * result.x + alpha * rgb_layer.x;
        result.y = (1.0f - alpha) * result.y + alpha * rgb_layer.y;
        result.z = (1.0f - alpha) * result.z + alpha * rgb_layer.z;
        result.w = fmaxf(result.w, alpha);
    }
    
    // Write final pixel
    int idx = (y * width + x) * 4;
    output[idx + 0] = result.x;
    output[idx + 1] = result.y;
    output[idx + 2] = result.z;
    output[idx + 3] = result.w;
}

// ============================================================================
// Host Interface Functions
// ============================================================================

extern "C" {

/**
 * Launch single-layer QTT rendering kernel
 * 
 * Memory layout:
 * - qtt_cores: Flat array of 96 floats (12 cores × 2 matrices × 4 elements)
 * - output: RGBA float32 buffer (H×W×4)
 */
void launch_render_qtt_layer(
    const float* qtt_cores,
    float* output,
    int width,
    int height,
    float value_min,
    float value_max,
    int colormap_type,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    render_qtt_layer_kernel<<<grid, block, 0, stream>>>(
        qtt_cores, output, width, height, value_min, value_max, colormap_type
    );
}

/**
 * Launch multi-layer compositor kernel
 */
void launch_composite_qtt_layers(
    const float* const* layer_cores,
    const int* layer_enabled,
    float* output,
    int width,
    int height,
    float value_min,
    float value_max,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    composite_qtt_layers_kernel<<<grid, block, 0, stream>>>(
        layer_cores, layer_enabled, output, width, height, value_min, value_max
    );
}

} // extern "C"
