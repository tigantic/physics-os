//! Test curve point serialization

use icicle_bn254::curve::{G1Projective, G1Affine, ScalarField};
use icicle_core::traits::GenerateRandom;
use fluidelite_zk::gpu::GpuAccelerator;

fn main() {
    println!("Initializing GPU...");
    let _gpu = GpuAccelerator::new().expect("GPU init failed");
    
    println!("\n=== Struct Sizes ===");
    println!("G1Projective: {} bytes", std::mem::size_of::<G1Projective>());
    println!("G1Affine: {} bytes", std::mem::size_of::<G1Affine>());
    println!("ScalarField: {} bytes", std::mem::size_of::<ScalarField>());
    
    // Generate random points
    println!("\n=== Random G1 Point ===");
    let points = G1Affine::generate_random(1);
    let p = &points[0];
    println!("G1Affine: {:?}", p);
    
    // Access x, y coordinates directly
    println!("\n=== Coordinate Access ===");
    println!("p.x: {:?}", p.x);
    println!("p.y: {:?}", p.y);
    
    // Convert to raw bytes via transmute (these are just field elements)
    println!("\n=== Raw Bytes (Big Endian for Solidity) ===");
    let x_ptr = &p.x as *const _ as *const [u8; 32];
    let y_ptr = &p.y as *const _ as *const [u8; 32];
    
    let x_bytes: [u8; 32] = unsafe { *x_ptr };
    let y_bytes: [u8; 32] = unsafe { *y_ptr };
    
    // Reverse for big-endian (Solidity)
    let mut x_be = x_bytes;
    let mut y_be = y_bytes;
    x_be.reverse();
    y_be.reverse();
    
    println!("x (BE): 0x{}", hex::encode(&x_be));
    println!("y (BE): 0x{}", hex::encode(&y_be));
    
    // Full 64-byte encoding for ecAdd/ecMul
    println!("\n=== Full 64-byte Encoding ===");
    let mut full = [0u8; 64];
    full[0..32].copy_from_slice(&x_be);
    full[32..64].copy_from_slice(&y_be);
    println!("Full point: 0x{}", hex::encode(&full));
    
    println!("\n=== Done ===");
}
