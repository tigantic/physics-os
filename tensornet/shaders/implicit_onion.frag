// implicit_onion.frag
// QTT-Native Implicit Synthesis Fragment Shader
// 
// This shader performs Tensor Train contraction at each fragment,
// synthesizing pixel values from compressed QTT cores instead of
// fetching from materialized buffers.
//
// Bandwidth: O(d × r²) core data vs O(n^d) pixel data
// Compute: Matrix contractions using 33.4 TFLOPS
//
// The Architect: "The GPU is a mathematician, not a forklift."

#version 430 core

// Outputs
out vec4 FragColor;

// QTT Core Data (uploaded via Shader Storage Buffer)
layout(std430, binding = 0) buffer CoreData {
    float core_data[];  // Float16 cores flattened
};

layout(std430, binding = 1) buffer CoreOffsets {
    int core_offsets[];  // Starting index for each core
};

layout(std430, binding = 2) buffer RankData {
    int ranks[];  // Bond dimensions [r_0, r_1, ..., r_n]
};

// QTT Parameters
uniform int u_n_cores;
uniform int u_nx_bits;
uniform int u_ny_bits;
uniform ivec2 u_resolution;  // (width, height)

// Color mapping
uniform sampler1D u_colormap;  // 1D LUT for value → RGB


// Morton Z-curve bit interleaving
// Interleaves x and y bits: (x0y0 x1y1 x2y2 ...)
int morton_encode(int x, int y) {
    int morton = 0;
    int max_bits = max(u_nx_bits, u_ny_bits);
    
    for (int i = 0; i < max_bits; i++) {
        // Alternate y-bit (LSB) then x-bit
        if (i < u_ny_bits) {
            int y_bit = (y >> i) & 1;
            morton |= (y_bit << (2 * i));
        }
        if (i < u_nx_bits) {
            int x_bit = (x >> i) & 1;
            morton |= (x_bit << (2 * i + 1));
        }
    }
    
    return morton;
}


// Extract bit array from Morton index
// Returns 1 if bit is set, 0 otherwise
int extract_bit(int morton_idx, int bit_pos) {
    return (morton_idx >> bit_pos) & 1;
}


// QTT Core Contraction
// 
// Algorithm:
// 1. result = cores[0][0, bit_0, :]  (size r_1)
// 2. For k = 1 to n_cores-1:
//       core_k = cores[k][:, bit_k, :]  (size r_k × r_{k+1})
//       result = result · core_k       (matrix-vector multiply)
// 3. Return result[0] (scalar)
//
// GLSL Translation:
// - Cores stored flattened: [r_left, 2, r_right]
// - Use core_offsets[k] to locate core k in core_data[]
// - Manual matrix multiplication with loop
float qtt_contract(int morton_idx) {
    // Working buffer for intermediate result
    float result[32];  // Max rank 32 (sufficient for most cases)
    int result_size;
    
    // Extract all bits from Morton index
    int bits[64];  // Max 64 cores (32 x-bits + 32 y-bits)
    for (int k = 0; k < u_n_cores; k++) {
        bits[k] = extract_bit(morton_idx, u_n_cores - 1 - k);  // MSB first
    }
    
    // Initialize: result = cores[0][0, bits[0], :]
    int offset_0 = core_offsets[0];
    int r_right_0 = ranks[1];
    int bit_0 = bits[0];
    
    // Core 0 layout: [1, 2, r_right]
    // Index: [0, bit_0, j] = offset_0 + bit_0 * r_right_0 + j
    for (int j = 0; j < r_right_0; j++) {
        int idx = offset_0 + bit_0 * r_right_0 + j;
        result[j] = core_data[idx];
    }
    result_size = r_right_0;
    
    // Contract through remaining cores
    for (int k = 1; k < u_n_cores; k++) {
        int offset_k = core_offsets[k];
        int r_left = ranks[k];
        int r_right = ranks[k + 1];
        int bit_k = bits[k];
        
        // Core k layout: [r_left, 2, r_right]
        // Index: [i, bit_k, j] = offset_k + (i * 2 + bit_k) * r_right + j
        
        // Temporary buffer for new result
        float new_result[32];
        
        // Matrix-vector multiply: new_result[j] = sum_i result[i] * core[i, bit_k, j]
        for (int j = 0; j < r_right; j++) {
            float sum = 0.0;
            for (int i = 0; i < r_left; i++) {
                int core_idx = offset_k + (i * 2 + bit_k) * r_right + j;
                sum += result[i] * core_data[core_idx];
            }
            new_result[j] = sum;
        }
        
        // Copy back
        for (int j = 0; j < r_right; j++) {
            result[j] = new_result[j];
        }
        result_size = r_right;
    }
    
    // Final result is scalar (r_final = 1)
    return result[0];
}


void main() {
    // Pixel coordinates
    ivec2 pixel_coord = ivec2(gl_FragCoord.xy);
    int x = pixel_coord.x;
    int y = pixel_coord.y;
    
    // Bounds check
    if (x >= u_resolution.x || y >= u_resolution.y) {
        discard;
    }
    
    // Morton encode
    int morton_idx = morton_encode(x, y);
    
    // Perform QTT contraction
    float value = qtt_contract(morton_idx);
    
    // Map value to color via 1D LUT
    // Normalize value to [0, 1] (assume value in [-1, 1])
    float normalized = (value + 1.0) * 0.5;
    normalized = clamp(normalized, 0.0, 1.0);
    
    vec3 color = texture(u_colormap, normalized).rgb;
    
    // Output with full opacity
    FragColor = vec4(color, 1.0);
}
