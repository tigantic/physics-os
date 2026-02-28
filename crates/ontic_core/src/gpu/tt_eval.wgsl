// Tensor Train Evaluation Compute Shader
//
// QTT Doctrine: NO DECOMPRESSION - Evaluate TT directly on GPU
//
// This shader evaluates a Tensor Train at specific multi-indices WITHOUT
// reconstructing the full dense tensor. This enables O(L·χ²) evaluation
// instead of O(d^L) memory/compute.
//
// Algorithm:
// For each query point x = (x₁, x₂, ..., x_L):
//   1. Initialize accumulator as 1×1 identity
//   2. For each site i = 0 to L-1:
//      a. Select slice A[i][:, x_i, :] from core (shape: χ_left × χ_right)
//      b. Multiply accumulator by this slice: acc = acc @ slice
//   3. Final scalar value is acc[0,0]
//
// Memory Layout:
// - cores: Flattened array of all TT-cores
// - core_offsets: Byte offset to start of each core
// - bond_dims: Bond dimension array [χ₀, χ₁, ..., χ_{L-1}]
// - indices: Query points, each is L consecutive u32 values
// - output: Scalar results, one per query point

struct TTParams {
    num_sites: u32,
    physical_dim: u32,
    max_bond_dim: u32,
    num_queries: u32,
}

@group(0) @binding(0) var<uniform> params: TTParams;
@group(0) @binding(1) var<storage, read> bond_dims: array<u32>;
@group(0) @binding(2) var<storage, read> core_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> cores: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

// Maximum bond dimension we support (must match shader compilation)
const MAX_CHI: u32 = 64u;

// Workgroup shared memory for intermediate results
var<workgroup> shared_acc: array<f32, 64>;

// Get left bond dimension for site i
fn chi_left(site: u32) -> u32 {
    if (site == 0u) {
        return 1u;
    }
    return bond_dims[site - 1u];
}

// Get right bond dimension for site i  
fn chi_right(site: u32) -> u32 {
    if (site == params.num_sites - 1u) {
        return 1u;
    }
    return bond_dims[site];
}

// Get element from core tensor at position (alpha_left, x, alpha_right)
// Core[i] has shape (chi_left, d, chi_right) stored row-major
fn get_core_element(site: u32, alpha_left: u32, x: u32, alpha_right: u32) -> f32 {
    let chi_l = chi_left(site);
    let chi_r = chi_right(site);
    let d = params.physical_dim;
    
    // Row-major index: alpha_left * (d * chi_r) + x * chi_r + alpha_right
    let local_idx = alpha_left * (d * chi_r) + x * chi_r + alpha_right;
    
    // Global index in cores array (divide by 4 since core_offsets is in bytes)
    let global_idx = core_offsets[site] / 4u + local_idx;
    
    return cores[global_idx];
}

// Evaluate TT at a single query point
// query_start: index into indices array where this query's L indices start
fn evaluate_tt_single(query_start: u32) -> f32 {
    // Working vectors for matrix-vector multiplication
    // We alternate between acc_a and acc_b to avoid copies
    var acc_a: array<f32, 64>;
    var acc_b: array<f32, 64>;
    
    // Initialize: acc_a = [1.0] (1×1 identity)
    acc_a[0] = 1.0;
    var current_dim: u32 = 1u;
    var use_a: bool = true;
    
    // Contract through all sites
    for (var site = 0u; site < params.num_sites; site++) {
        let x_i = indices[query_start + site];
        let chi_l = chi_left(site);
        let chi_r = chi_right(site);
        
        // Safety check
        if (x_i >= params.physical_dim) {
            return 0.0; // Invalid index
        }
        
        // Matrix-vector multiply: new_acc = current_acc @ core_slice
        // core_slice has shape (chi_l, chi_r) for fixed x_i
        
        if (use_a) {
            // acc_b = acc_a @ core_slice
            for (var j = 0u; j < chi_r; j++) {
                var sum: f32 = 0.0;
                for (var i = 0u; i < chi_l; i++) {
                    sum += acc_a[i] * get_core_element(site, i, x_i, j);
                }
                acc_b[j] = sum;
            }
        } else {
            // acc_a = acc_b @ core_slice
            for (var j = 0u; j < chi_r; j++) {
                var sum: f32 = 0.0;
                for (var i = 0u; i < chi_l; i++) {
                    sum += acc_b[i] * get_core_element(site, i, x_i, j);
                }
                acc_a[j] = sum;
            }
        }
        
        current_dim = chi_r;
        use_a = !use_a;
    }
    
    // Final result is the single element
    if (use_a) {
        return acc_a[0];
    } else {
        return acc_b[0];
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_idx = gid.x;
    
    // Bounds check
    if (query_idx >= params.num_queries) {
        return;
    }
    
    // Each query has num_sites consecutive indices
    let query_start = query_idx * params.num_sites;
    
    // Evaluate TT at this query point
    let result = evaluate_tt_single(query_start);
    
    // Store result
    output[query_idx] = result;
}

// Batch evaluation with shared memory optimization (for small χ)
@compute @workgroup_size(64)
fn main_shared(@builtin(global_invocation_id) gid: vec3<u32>,
               @builtin(local_invocation_id) lid: vec3<u32>) {
    let query_idx = gid.x;
    
    if (query_idx >= params.num_queries) {
        return;
    }
    
    // For small bond dimensions, we can use shared memory
    // This variant is optimized for χ ≤ 64
    let query_start = query_idx * params.num_sites;
    let result = evaluate_tt_single(query_start);
    output[query_idx] = result;
}
