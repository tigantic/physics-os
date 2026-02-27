//! FluidElite Block-Independent QTT Engine
//!
//! Architecture: RAID Array of Tensors
//! - Each 64MB block is INDEPENDENT (no global state, no merging)
//! - Format: [Header 128B][Index 24B×N][Block_0][Block_1]...
//! - Random access: Seek to Block_N, decode ONLY that block
//! - Infinite scale: 1 PB = 15,625 blocks × 64MB RAM = constant memory
//!
//! "You don't move the mountain. Each block IS the mountain."
//!
//! FluidElite-ZK v2.0.0 | Block-Independent Protocol

use std::fs::File;
use std::io::{self, Read, Write, Seek, SeekFrom, BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};
use sha2::{Sha256, Digest};
use rayon::prelude::*;

#[cfg(feature = "s3")]
use aws_config::BehaviorVersion;
#[cfg(feature = "s3")]
use aws_sdk_s3::Client as S3Client;
#[cfg(feature = "s3")]
use aws_sdk_s3::config::Region;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Block size: 64MB (optimal for L3 cache, ~15,625 blocks per PB)
const BLOCK_SIZE: usize = 64 * 1024 * 1024;

/// Magic bytes for .fluid format
const MAGIC: [u8; 8] = *b"FLUIDBLK";

/// Header size (fixed)
const HEADER_SIZE: usize = 128;

/// Index entry size: offset(8) + core_size(8) + residual_size(8) = 24 bytes
const INDEX_ENTRY_SIZE: usize = 24;

/// QTT reshape: 64MB = 2^26 bytes = 2^26 sites of 1 byte each
/// We use 26 sites (log2(64MB)) with bond dimension up to max_rank
const N_SITES: usize = 26;

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser)]
#[command(name = "fluid_block")]
#[command(version = "2.0.0")]
#[command(about = "Block-Independent QTT Compression - RAID Array of Tensors")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress file using Block-Independent QTT
    Compress {
        /// Input file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output .fluid file path
        #[arg(short, long)]
        output: PathBuf,

        /// Maximum QTT rank (default 64)
        #[arg(short, long, default_value = "64")]
        max_rank: usize,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Decompress .fluid file back to original
    Decompress {
        /// Input .fluid file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Extract single block (random access)
    Extract {
        /// Input .fluid file
        #[arg(short, long)]
        input: PathBuf,

        /// Block index to extract
        #[arg(short, long)]
        block: usize,

        /// Output file for extracted block
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Stream compress from S3
    #[cfg(feature = "s3")]
    Cloud {
        /// S3 URI (s3://bucket/key)
        #[arg(short, long)]
        input: String,

        /// Output .fluid file (local)
        #[arg(short, long)]
        output: PathBuf,

        /// AWS region
        #[arg(short, long, default_value = "us-east-1")]
        region: String,

        /// Maximum QTT rank
        #[arg(short, long, default_value = "64")]
        max_rank: usize,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Benchmark: Calculate theoretical compression for N bytes
    Benchmark {
        /// Input size in bytes (supports K, M, G, T, P suffixes)
        #[arg(short, long)]
        size: String,

        /// Maximum QTT rank
        #[arg(short, long, default_value = "64")]
        max_rank: usize,
    },
}

// ============================================================================
// QTT CORE - Block-Level Tensor Train
// ============================================================================

/// A single QTT core tensor: shape (r_left, d, r_right)
/// For binary QTT: d=2 always (one qubit per site)
#[derive(Clone)]
struct QttCore {
    /// Flattened data: [r_left × 2 × r_right]
    data: Vec<f64>,
    r_left: usize,
    r_right: usize,
}

impl QttCore {
    fn new(r_left: usize, r_right: usize) -> Self {
        Self {
            data: vec![0.0; r_left * 2 * r_right],
            r_left,
            r_right,
        }
    }

    #[inline]
    fn get(&self, i: usize, d: usize, j: usize) -> f64 {
        self.data[i * 2 * self.r_right + d * self.r_right + j]
    }

    #[inline]
    fn set(&mut self, i: usize, d: usize, j: usize, val: f64) {
        self.data[i * 2 * self.r_right + d * self.r_right + j] = val;
    }

    fn bytes(&self) -> usize {
        self.data.len() * 8
    }

    fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.data.len() * 8);
        out.extend_from_slice(&(self.r_left as u64).to_le_bytes());
        out.extend_from_slice(&(self.r_right as u64).to_le_bytes());
        for &v in &self.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn deserialize(data: &[u8]) -> io::Result<(Self, usize)> {
        if data.len() < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Core too short"));
        }
        let r_left = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let r_right = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let n_floats = r_left * 2 * r_right;
        let expected_len = 16 + n_floats * 8;
        if data.len() < expected_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Core data truncated"));
        }
        let mut core = Self::new(r_left, r_right);
        for i in 0..n_floats {
            let offset = 16 + i * 8;
            core.data[i] = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        }
        Ok((core, expected_len))
    }
}

/// Block-level QTT: Independent decomposition of one 64MB block
struct BlockQtt {
    cores: Vec<QttCore>,
    n_sites: usize,
    original_size: usize,
}

impl BlockQtt {
    /// Decompose a block using TT-SVD (Oseledets 2011)
    fn decompose(block: &[u8], max_rank: usize) -> Self {
        let n = block.len();
        let n_sites = (n as f64).log2().ceil() as usize;
        let padded_size = 1usize << n_sites;

        // Pad to power of 2
        let mut padded = vec![0u8; padded_size];
        padded[..n].copy_from_slice(block);

        // Convert to f64 for numerical stability
        let values: Vec<f64> = padded.iter().map(|&b| b as f64).collect();

        // TT-SVD decomposition
        let cores = Self::tt_svd(&values, n_sites, max_rank);

        Self {
            cores,
            n_sites,
            original_size: n,
        }
    }

    /// TT-SVD: Oseledets 2011 Algorithm
    /// Decomposes N-dimensional tensor into train of 3D cores
    fn tt_svd(values: &[f64], n_sites: usize, max_rank: usize) -> Vec<QttCore> {
        let mut cores = Vec::with_capacity(n_sites);
        let mut current = values.to_vec();
        let mut r_left = 1usize;

        for site in 0..n_sites {
            let remaining_sites = n_sites - site;
            let cols = 1usize << (remaining_sites - 1); // 2^(remaining-1)
            let rows = current.len() / cols;

            // Reshape current into matrix [rows × cols]
            // Then do truncated SVD

            // For binary QTT: split rows into (r_left, 2)
            // Matrix shape: (r_left * 2) × cols
            let m = r_left * 2;
            let n_cols = cols;

            // Power iteration for top-r singular vectors (rSVD)
            let r_new = max_rank.min(m).min(n_cols);
            let (u, s, _vt, actual_rank) = Self::truncated_svd(&current, m, n_cols, r_new);

            // Build core from U: shape (r_left, 2, r_new)
            let mut core = QttCore::new(r_left, actual_rank);
            for i in 0..r_left {
                for d in 0..2 {
                    for j in 0..actual_rank {
                        let row = i * 2 + d;
                        let val = u[row * actual_rank + j] * s[j].sqrt();
                        core.set(i, d, j, val);
                    }
                }
            }
            cores.push(core);

            // Contract: new current = S^(1/2) * Vt
            if site < n_sites - 1 {
                let new_size = actual_rank * n_cols;
                let mut new_current = vec![0.0; new_size];
                for j in 0..actual_rank {
                    let scale = s[j].sqrt();
                    for k in 0..n_cols {
                        // Vt is stored column-major in our SVD
                        // We computed U, S, Vt where A = U * S * Vt
                        // new_current[j, k] = sqrt(s[j]) * vt[j, k]
                        // But we need to recompute from residual
                        // Actually: C = U * S * Vt, residual = A, next iteration works on S*Vt
                        // For simplicity, contract directly
                        let mut sum = 0.0;
                        for i in 0..m {
                            sum += u[i * actual_rank + j] * current[i * n_cols + k];
                        }
                        new_current[j * n_cols + k] = sum * scale / s[j].max(1e-15);
                    }
                }
                current = new_current;
            }

            r_left = actual_rank;
        }

        // Fix the last core to have r_right = 1
        if let Some(last) = cores.last_mut() {
            // Already should be shape (r, 2, 1) but let's verify
            // For the last site, cols=1, so r_right should be 1
        }

        cores
    }

    /// Truncated SVD using power iteration (rSVD)
    fn truncated_svd(data: &[f64], m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let k = k.min(m).min(n);
        if k == 0 {
            return (vec![], vec![], vec![], 0);
        }

        // Power iteration to find top-k singular vectors
        let n_iter = 5;

        // Random initialization for Q
        let mut q: Vec<f64> = (0..n * k)
            .map(|i| ((i * 7919 + 104729) % 10007) as f64 / 10007.0 - 0.5)
            .collect();

        // Normalize columns of Q
        for j in 0..k {
            let mut norm = 0.0;
            for i in 0..n {
                norm += q[i * k + j].powi(2);
            }
            norm = norm.sqrt().max(1e-15);
            for i in 0..n {
                q[i * k + j] /= norm;
            }
        }

        // Power iteration: Q = A^T * A * Q
        for _ in 0..n_iter {
            // Y = A * Q: (m × n) * (n × k) = (m × k)
            let mut y = vec![0.0; m * k];
            for i in 0..m {
                for j in 0..k {
                    let mut sum = 0.0;
                    for l in 0..n {
                        sum += data[i * n + l] * q[l * k + j];
                    }
                    y[i * k + j] = sum;
                }
            }

            // Q_new = A^T * Y: (n × m) * (m × k) = (n × k)
            let mut q_new = vec![0.0; n * k];
            for i in 0..n {
                for j in 0..k {
                    let mut sum = 0.0;
                    for l in 0..m {
                        sum += data[l * n + i] * y[l * k + j];
                    }
                    q_new[i * k + j] = sum;
                }
            }

            // Orthonormalize Q (modified Gram-Schmidt)
            for j in 0..k {
                // Subtract projections onto previous columns
                for jj in 0..j {
                    let mut dot = 0.0;
                    for i in 0..n {
                        dot += q_new[i * k + j] * q_new[i * k + jj];
                    }
                    for i in 0..n {
                        q_new[i * k + j] -= dot * q_new[i * k + jj];
                    }
                }
                // Normalize
                let mut norm = 0.0;
                for i in 0..n {
                    norm += q_new[i * k + j].powi(2);
                }
                norm = norm.sqrt().max(1e-15);
                for i in 0..n {
                    q_new[i * k + j] /= norm;
                }
            }
            q = q_new;
        }

        // B = A * Q: (m × n) * (n × k) = (m × k)
        let mut b = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0;
                for l in 0..n {
                    sum += data[i * n + l] * q[l * k + j];
                }
                b[i * k + j] = sum;
            }
        }

        // QR of B to get U (m × k)
        let mut u = b.clone();
        let mut r_diag = vec![0.0; k];
        for j in 0..k {
            for jj in 0..j {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += u[i * k + j] * u[i * k + jj];
                }
                for i in 0..m {
                    u[i * k + j] -= dot * u[i * k + jj];
                }
            }
            let mut norm = 0.0;
            for i in 0..m {
                norm += u[i * k + j].powi(2);
            }
            norm = norm.sqrt();
            r_diag[j] = norm;
            if norm > 1e-15 {
                for i in 0..m {
                    u[i * k + j] /= norm;
                }
            }
        }

        // Singular values are the R diagonal (approximately)
        // Filter zero singular values
        let tol = 1e-10;
        let actual_rank = r_diag.iter().filter(|&&s| s > tol).count().max(1);

        (u, r_diag, q, actual_rank)
    }

    /// Reconstruct the original block from cores
    fn reconstruct(&self) -> Vec<u8> {
        let total_size = 1usize << self.n_sites;
        let mut result = vec![0.0; total_size];

        // For each output index, contract the tensor train
        for idx in 0..total_size {
            let mut val = 1.0;
            let mut prev_vec: Vec<f64> = vec![1.0]; // Start with scalar

            for (site, core) in self.cores.iter().enumerate() {
                let bit = (idx >> (self.n_sites - 1 - site)) & 1;

                // Contract: new_vec[j] = sum_i prev_vec[i] * core[i, bit, j]
                let mut new_vec = vec![0.0; core.r_right];
                for i in 0..core.r_left {
                    for j in 0..core.r_right {
                        new_vec[j] += prev_vec[i] * core.get(i, bit, j);
                    }
                }
                prev_vec = new_vec;
            }

            result[idx] = prev_vec[0];
        }

        // Convert back to bytes, clamping to [0, 255]
        let mut bytes = Vec::with_capacity(self.original_size);
        for i in 0..self.original_size {
            let v = result[i].round().clamp(0.0, 255.0) as u8;
            bytes.push(v);
        }
        bytes
    }

    fn total_core_bytes(&self) -> usize {
        self.cores.iter().map(|c| c.bytes()).sum()
    }

    fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.n_sites as u32).to_le_bytes());
        out.extend_from_slice(&(self.original_size as u64).to_le_bytes());
        out.extend_from_slice(&(self.cores.len() as u32).to_le_bytes());

        for core in &self.cores {
            let core_bytes = core.serialize();
            out.extend_from_slice(&(core_bytes.len() as u32).to_le_bytes());
            out.extend(core_bytes);
        }
        out
    }

    fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.len() < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "BlockQtt too short"));
        }
        let n_sites = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let original_size = u64::from_le_bytes(data[4..12].try_into().unwrap()) as usize;
        let n_cores = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;

        let mut offset = 16;
        let mut cores = Vec::with_capacity(n_cores);

        for _ in 0..n_cores {
            if offset + 4 > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Core length truncated"));
            }
            let core_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            if offset + core_len > data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Core data truncated"));
            }
            let (core, _) = QttCore::deserialize(&data[offset..offset + core_len])?;
            cores.push(core);
            offset += core_len;
        }

        Ok(Self { cores, n_sites, original_size })
    }
}

// ============================================================================
// COMPRESSED BLOCK - Cores + zstd(Residual)
// ============================================================================

struct CompressedBlock {
    qtt: BlockQtt,
    residual_compressed: Vec<u8>,
}

impl CompressedBlock {
    fn compress(block: &[u8], max_rank: usize) -> Self {
        // Step 1: QTT decomposition
        let qtt = BlockQtt::decompose(block, max_rank);

        // Step 2: Reconstruct approximation
        let approx = qtt.reconstruct();

        // Step 3: Compute residual = original XOR approximation (lossless)
        let residual: Vec<u8> = block.iter()
            .zip(approx.iter())
            .map(|(&o, &a)| o ^ a)
            .collect();

        // Step 4: zstd compress the residual
        let residual_compressed = zstd::encode_all(residual.as_slice(), 19)
            .unwrap_or_else(|_| residual.clone());

        Self { qtt, residual_compressed }
    }

    fn decompress(&self) -> Vec<u8> {
        // Step 1: Reconstruct QTT approximation
        let approx = self.qtt.reconstruct();

        // Step 2: Decompress residual
        let residual = zstd::decode_all(self.residual_compressed.as_slice())
            .unwrap_or_else(|_| self.residual_compressed.clone());

        // Step 3: XOR to get original
        approx.iter()
            .zip(residual.iter())
            .map(|(&a, &r)| a ^ r)
            .collect()
    }

    fn serialize(&self) -> Vec<u8> {
        let qtt_bytes = self.qtt.serialize();
        let mut out = Vec::with_capacity(8 + qtt_bytes.len() + self.residual_compressed.len());

        out.extend_from_slice(&(qtt_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.residual_compressed.len() as u32).to_le_bytes());
        out.extend(qtt_bytes);
        out.extend(&self.residual_compressed);
        out
    }

    fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "CompressedBlock too short"));
        }
        let qtt_len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let res_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        if data.len() < 8 + qtt_len + res_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "CompressedBlock truncated"));
        }

        let qtt = BlockQtt::deserialize(&data[8..8 + qtt_len])?;
        let residual_compressed = data[8 + qtt_len..8 + qtt_len + res_len].to_vec();

        Ok(Self { qtt, residual_compressed })
    }

    fn total_bytes(&self) -> usize {
        8 + self.qtt.serialize().len() + self.residual_compressed.len()
    }
}

// ============================================================================
// FLUID FILE FORMAT
// ============================================================================
// [Header 128 bytes]
//   - Magic: 8 bytes "FLUIDBLK"
//   - Version: 4 bytes
//   - Block count: 8 bytes
//   - Original size: 8 bytes
//   - Block size: 8 bytes
//   - Max rank: 4 bytes
//   - SHA256 of original: 32 bytes
//   - Reserved: 56 bytes
//
// [Index: 24 bytes × N blocks]
//   - Block offset: 8 bytes
//   - Core size: 8 bytes
//   - Residual size: 8 bytes
//
// [Block data...]

struct FluidFile {
    version: u32,
    block_count: u64,
    original_size: u64,
    block_size: u64,
    max_rank: u32,
    original_hash: [u8; 32],
    index: Vec<(u64, u64, u64)>, // (offset, core_size, residual_size)
}

impl FluidFile {
    fn write_header<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let mut header = [0u8; HEADER_SIZE];

        header[0..8].copy_from_slice(&MAGIC);
        header[8..12].copy_from_slice(&self.version.to_le_bytes());
        header[12..20].copy_from_slice(&self.block_count.to_le_bytes());
        header[20..28].copy_from_slice(&self.original_size.to_le_bytes());
        header[28..36].copy_from_slice(&self.block_size.to_le_bytes());
        header[36..40].copy_from_slice(&self.max_rank.to_le_bytes());
        header[40..72].copy_from_slice(&self.original_hash);

        writer.write_all(&header)?;
        Ok(())
    }

    fn read_header<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut header = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header)?;

        if &header[0..8] != &MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic"));
        }

        let version = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let block_count = u64::from_le_bytes(header[12..20].try_into().unwrap());
        let original_size = u64::from_le_bytes(header[20..28].try_into().unwrap());
        let block_size = u64::from_le_bytes(header[28..36].try_into().unwrap());
        let max_rank = u32::from_le_bytes(header[36..40].try_into().unwrap());
        let mut original_hash = [0u8; 32];
        original_hash.copy_from_slice(&header[40..72]);

        Ok(Self {
            version,
            block_count,
            original_size,
            block_size,
            max_rank,
            original_hash,
            index: Vec::new(),
        })
    }

    fn write_index<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        for &(offset, core_size, residual_size) in &self.index {
            writer.write_all(&offset.to_le_bytes())?;
            writer.write_all(&core_size.to_le_bytes())?;
            writer.write_all(&residual_size.to_le_bytes())?;
        }
        Ok(())
    }

    fn read_index<R: Read>(&mut self, reader: &mut R) -> io::Result<()> {
        self.index.clear();
        for _ in 0..self.block_count {
            let mut buf = [0u8; INDEX_ENTRY_SIZE];
            reader.read_exact(&mut buf)?;
            let offset = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let core_size = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            let residual_size = u64::from_le_bytes(buf[16..24].try_into().unwrap());
            self.index.push((offset, core_size, residual_size));
        }
        Ok(())
    }
}

// ============================================================================
// COMPRESSION ENGINE
// ============================================================================

fn compress_file(input: &Path, output: &Path, max_rank: usize, verbose: bool) -> io::Result<()> {
    let start = Instant::now();

    // Open input
    let input_file = File::open(input)?;
    let input_size = input_file.metadata()?.len();
    let mut reader = BufReader::with_capacity(BLOCK_SIZE, input_file);

    // Calculate SHA256 of original
    let hash = {
        let mut hasher = Sha256::new();
        let mut file = File::open(input)?;
        io::copy(&mut file, &mut hasher)?;
        let result: [u8; 32] = hasher.finalize().into();
        result
    };

    // Calculate block count
    let block_count = (input_size as usize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if verbose {
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║         FluidElite Block-Independent QTT Compression         ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║ Input:       {:>48} ║", format!("{}", input.display()));
        eprintln!("║ Size:        {:>45} B ║", input_size);
        eprintln!("║ Blocks:      {:>48} ║", block_count);
        eprintln!("║ Block Size:  {:>45} B ║", BLOCK_SIZE);
        eprintln!("║ Max Rank:    {:>48} ║", max_rank);
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }

    // Create output file
    let output_file = File::create(output)?;
    let mut writer = BufWriter::new(output_file);

    // Write placeholder header (will update later)
    let mut fluid = FluidFile {
        version: 2,
        block_count: block_count as u64,
        original_size: input_size,
        block_size: BLOCK_SIZE as u64,
        max_rank: max_rank as u32,
        original_hash: hash,
        index: Vec::with_capacity(block_count),
    };
    fluid.write_header(&mut writer)?;

    // Reserve space for index
    let index_size = block_count * INDEX_ENTRY_SIZE;
    writer.write_all(&vec![0u8; index_size])?;

    // Process blocks
    let data_start = HEADER_SIZE + index_size;
    let mut current_offset = data_start as u64;
    let mut total_core_bytes = 0usize;
    let mut total_residual_bytes = 0usize;
    let mut block_buf = vec![0u8; BLOCK_SIZE];

    for block_idx in 0..block_count {
        let bytes_remaining = input_size as usize - block_idx * BLOCK_SIZE;
        let block_len = bytes_remaining.min(BLOCK_SIZE);

        reader.read_exact(&mut block_buf[..block_len])?;

        // Compress this block
        let compressed = CompressedBlock::compress(&block_buf[..block_len], max_rank);

        // Serialize
        let block_data = compressed.serialize();
        let qtt_bytes = compressed.qtt.serialize().len();
        let res_bytes = compressed.residual_compressed.len();

        total_core_bytes += qtt_bytes;
        total_residual_bytes += res_bytes;

        // Write block data
        writer.write_all(&block_data)?;

        // Record index entry
        fluid.index.push((current_offset, qtt_bytes as u64, res_bytes as u64));
        current_offset += block_data.len() as u64;

        if verbose && (block_idx + 1) % 10 == 0 {
            eprintln!("  Block {}/{} complete", block_idx + 1, block_count);
        }
    }

    // Flush and rewind to write index
    writer.flush()?;
    drop(writer);

    // Rewrite header and index
    let mut file = File::options().write(true).open(output)?;
    file.seek(SeekFrom::Start(0))?;
    fluid.write_header(&mut file)?;
    fluid.write_index(&mut file)?;

    let elapsed = start.elapsed();
    let output_size = std::fs::metadata(output)?.len();

    if verbose {
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║                      COMPRESSION COMPLETE                    ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║ Input:       {:>45} B ║", input_size);
        eprintln!("║ Output:      {:>45} B ║", output_size);
        eprintln!("║ QTT Cores:   {:>45} B ║", total_core_bytes);
        eprintln!("║ Residuals:   {:>45} B ║", total_residual_bytes);
        eprintln!("║ Ratio:       {:>47.2}x ║", input_size as f64 / output_size as f64);
        eprintln!("║ Time:        {:>44.2}s ║", elapsed.as_secs_f64());
        eprintln!("║ Throughput:  {:>40.2} MB/s ║", (input_size as f64 / 1e6) / elapsed.as_secs_f64());
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }

    Ok(())
}

fn decompress_file(input: &Path, output: &Path, verbose: bool) -> io::Result<()> {
    let start = Instant::now();

    let mut file = File::open(input)?;

    // Read header
    let mut fluid = FluidFile::read_header(&mut file)?;

    // Read index
    fluid.read_index(&mut file)?;

    if verbose {
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║        FluidElite Block-Independent QTT Decompression        ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║ Blocks:      {:>48} ║", fluid.block_count);
        eprintln!("║ Original:    {:>45} B ║", fluid.original_size);
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }

    // Create output file
    let output_file = File::create(output)?;
    let mut writer = BufWriter::new(output_file);
    let mut hasher = Sha256::new();

    // Decompress each block
    for (block_idx, &(offset, core_size, res_size)) in fluid.index.iter().enumerate() {
        let block_size = core_size + res_size + 8; // 8 bytes for lengths

        file.seek(SeekFrom::Start(offset))?;
        let mut block_data = vec![0u8; block_size as usize];
        file.read_exact(&mut block_data)?;

        let compressed = CompressedBlock::deserialize(&block_data)?;
        let decompressed = compressed.decompress();

        writer.write_all(&decompressed)?;
        hasher.update(&decompressed);

        if verbose && (block_idx + 1) % 10 == 0 {
            eprintln!("  Block {}/{} decompressed", block_idx + 1, fluid.block_count as usize);
        }
    }

    writer.flush()?;

    // Verify hash
    let computed_hash: [u8; 32] = hasher.finalize().into();
    let hash_match = computed_hash == fluid.original_hash;

    let elapsed = start.elapsed();

    if verbose {
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║                    DECOMPRESSION COMPLETE                    ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║ Output:      {:>45} B ║", fluid.original_size);
        eprintln!("║ SHA256:      {:>48} ║", if hash_match { "✓ VERIFIED" } else { "✗ MISMATCH" });
        eprintln!("║ Time:        {:>44.2}s ║", elapsed.as_secs_f64());
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }

    if !hash_match {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Hash mismatch"));
    }

    Ok(())
}

fn extract_block(input: &Path, block_idx: usize, output: &Path) -> io::Result<()> {
    let mut file = File::open(input)?;

    // Read header
    let mut fluid = FluidFile::read_header(&mut file)?;
    fluid.read_index(&mut file)?;

    if block_idx >= fluid.index.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Block {} out of range (max {})", block_idx, fluid.block_count - 1),
        ));
    }

    let (offset, core_size, res_size) = fluid.index[block_idx];
    let block_size = core_size + res_size + 8;

    // Seek directly to block (RANDOM ACCESS!)
    file.seek(SeekFrom::Start(offset))?;
    let mut block_data = vec![0u8; block_size as usize];
    file.read_exact(&mut block_data)?;

    let compressed = CompressedBlock::deserialize(&block_data)?;
    let decompressed = compressed.decompress();

    let mut out = File::create(output)?;
    out.write_all(&decompressed)?;

    eprintln!("Extracted block {} ({} bytes) to {}", block_idx, decompressed.len(), output.display());

    Ok(())
}

fn parse_size(s: &str) -> u64 {
    let s = s.trim().to_uppercase();
    let (num_str, mult) = if s.ends_with("PB") || s.ends_with('P') {
        (s.trim_end_matches("PB").trim_end_matches('P'), 1_000_000_000_000_000u64)
    } else if s.ends_with("TB") || s.ends_with('T') {
        (s.trim_end_matches("TB").trim_end_matches('T'), 1_000_000_000_000u64)
    } else if s.ends_with("GB") || s.ends_with('G') {
        (s.trim_end_matches("GB").trim_end_matches('G'), 1_000_000_000u64)
    } else if s.ends_with("MB") || s.ends_with('M') {
        (s.trim_end_matches("MB").trim_end_matches('M'), 1_000_000u64)
    } else if s.ends_with("KB") || s.ends_with('K') {
        (s.trim_end_matches("KB").trim_end_matches('K'), 1_000u64)
    } else {
        (s.as_str(), 1u64)
    };
    num_str.trim().parse::<u64>().unwrap_or(0) * mult
}

fn format_bytes(b: u64) -> String {
    if b >= 1_000_000_000_000_000 {
        format!("{:.2} PB", b as f64 / 1e15)
    } else if b >= 1_000_000_000_000 {
        format!("{:.2} TB", b as f64 / 1e12)
    } else if b >= 1_000_000_000 {
        format!("{:.2} GB", b as f64 / 1e9)
    } else if b >= 1_000_000 {
        format!("{:.2} MB", b as f64 / 1e6)
    } else if b >= 1_000 {
        format!("{:.2} KB", b as f64 / 1e3)
    } else {
        format!("{} B", b)
    }
}

fn benchmark(size_str: &str, max_rank: usize) {
    let input_bytes = parse_size(size_str);
    let block_size = BLOCK_SIZE as u64;
    let n_blocks = (input_bytes + block_size - 1) / block_size;

    // QTT core size per block: O(n_sites × rank²)
    // n_sites = log2(64MB) = 26
    // Each core: (r_left × 2 × r_right) floats × 8 bytes
    // Conservative estimate: sum of all cores ≈ n_sites × rank² × 8 × 2
    let n_sites = 26usize;
    let core_bytes_per_block = n_sites * max_rank * max_rank * 8 * 2;

    // For highly compressible data: residual compresses to ~0.01% of block
    // For random data: residual = 100% of block (zstd can't help)
    // For satellite data (structured): estimate 0.1% - 10% residual

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║              Block-Independent QTT Theoretical Analysis                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║ Input Size:         {:>54} ║", format_bytes(input_bytes));
    println!("║ Input Bytes:        {:>54} ║", input_bytes);
    println!("║ Block Size:         {:>54} ║", format_bytes(block_size));
    println!("║ Block Count:        {:>54} ║", n_blocks);
    println!("║ Max Rank:           {:>54} ║", max_rank);
    println!("║ Sites per Block:    {:>54} ║", n_sites);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                         Per-Block Analysis                               ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║ QTT Cores/Block:    {:>54} ║", format_bytes(core_bytes_per_block as u64));
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                         Projection Scenarios                             ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    // Scenario 1: Low-rank (highly structured, like repeating patterns)
    let residual_low = 0.0001; // 0.01%
    let total_cores_low = n_blocks * core_bytes_per_block as u64;
    let total_residual_low = (input_bytes as f64 * residual_low) as u64;
    let total_low = total_cores_low + total_residual_low;
    let ratio_low = input_bytes as f64 / total_low as f64;

    println!("║ LOW-RANK (0.01% residual):                                               ║");
    println!("║   Cores:            {:>54} ║", format_bytes(total_cores_low));
    println!("║   Residuals:        {:>54} ║", format_bytes(total_residual_low));
    println!("║   Total:            {:>54} ║", format_bytes(total_low));
    println!("║   Ratio:            {:>53.2}x ║", ratio_low);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    // Scenario 2: Satellite data (structured but not trivial)
    let residual_sat = 0.001; // 0.1%
    let total_residual_sat = (input_bytes as f64 * residual_sat) as u64;
    let total_sat = total_cores_low + total_residual_sat;
    let ratio_sat = input_bytes as f64 / total_sat as f64;

    println!("║ SATELLITE (0.1% residual):                                               ║");
    println!("║   Cores:            {:>54} ║", format_bytes(total_cores_low));
    println!("║   Residuals:        {:>54} ║", format_bytes(total_residual_sat));
    println!("║   Total:            {:>54} ║", format_bytes(total_sat));
    println!("║   Ratio:            {:>53.2}x ║", ratio_sat);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    // Scenario 3: Random data (worst case)
    let residual_rand = 1.0; // 100%
    let total_residual_rand = input_bytes;
    let total_rand = total_cores_low + total_residual_rand;
    let ratio_rand = input_bytes as f64 / total_rand as f64;

    println!("║ RANDOM (100% residual - worst case):                                     ║");
    println!("║   Cores:            {:>54} ║", format_bytes(total_cores_low));
    println!("║   Residuals:        {:>54} ║", format_bytes(total_residual_rand));
    println!("║   Total:            {:>54} ║", format_bytes(total_rand));
    println!("║   Ratio:            {:>53.2}x ║", ratio_rand);
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║ RAM Usage:          {:>51} (constant) ║", format_bytes(block_size));
    println!("║ Parallelism:        {:>54} ║", "Each block independent");
    println!("║ Random Access:      {:>54} ║", "Seek to any block instantly");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

// ============================================================================
// S3 STREAMING (Feature-gated)
// ============================================================================

#[cfg(feature = "s3")]
async fn compress_from_s3(
    uri: &str,
    output: &Path,
    region: &str,
    max_rank: usize,
    verbose: bool,
) -> io::Result<()> {
    use aws_sdk_s3::primitives::ByteStreamError;

    let start = Instant::now();

    // Parse S3 URI
    let (bucket, key) = parse_s3_uri(uri)?;

    // Create S3 client
    let config = aws_config::defaults(BehaviorVersion::latest())
        .region(Region::new(region.to_string()))
        .no_credentials() // Public bucket
        .load()
        .await;

    let client = S3Client::new(&config);

    // Get object size via HEAD
    let head = client
        .head_object()
        .bucket(&bucket)
        .key(&key)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let object_size = head.content_length().unwrap_or(0) as u64;
    let block_count = (object_size as usize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if verbose {
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║      FluidElite S3 Block-Independent QTT Compression         ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║ Bucket:      {:>48} ║", bucket);
        eprintln!("║ Key:         {:>48} ║", &key[..key.len().min(48)]);
        eprintln!("║ Size:        {:>45} B ║", object_size);
        eprintln!("║ Blocks:      {:>48} ║", block_count);
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }

    // Create output file
    let output_file = File::create(output)?;
    let mut writer = BufWriter::new(output_file);

    // Placeholder hash (can't compute without full download)
    let hash = [0u8; 32];

    let mut fluid = FluidFile {
        version: 2,
        block_count: block_count as u64,
        original_size: object_size,
        block_size: BLOCK_SIZE as u64,
        max_rank: max_rank as u32,
        original_hash: hash,
        index: Vec::with_capacity(block_count),
    };
    fluid.write_header(&mut writer)?;

    // Reserve space for index
    let index_size = block_count * INDEX_ENTRY_SIZE;
    writer.write_all(&vec![0u8; index_size])?;

    let data_start = HEADER_SIZE + index_size;
    let mut current_offset = data_start as u64;
    let mut total_core_bytes = 0usize;
    let mut total_residual_bytes = 0usize;
    let mut hasher = Sha256::new();

    // Stream blocks via byte-range GET
    for block_idx in 0..block_count {
        let start_byte = block_idx * BLOCK_SIZE;
        let end_byte = ((block_idx + 1) * BLOCK_SIZE).min(object_size as usize) - 1;

        let range = format!("bytes={}-{}", start_byte, end_byte);

        let resp = client
            .get_object()
            .bucket(&bucket)
            .key(&key)
            .range(&range)
            .send()
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

        let block_data = resp
            .body
            .collect()
            .await
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
            .into_bytes()
            .to_vec();

        hasher.update(&block_data);

        // Compress this block
        let compressed = CompressedBlock::compress(&block_data, max_rank);

        let block_bytes = compressed.serialize();
        let qtt_bytes = compressed.qtt.serialize().len();
        let res_bytes = compressed.residual_compressed.len();

        total_core_bytes += qtt_bytes;
        total_residual_bytes += res_bytes;

        writer.write_all(&block_bytes)?;

        fluid.index.push((current_offset, qtt_bytes as u64, res_bytes as u64));
        current_offset += block_bytes.len() as u64;

        if verbose {
            eprintln!(
                "  Block {}/{}: {} → {} (cores: {}, residual: {})",
                block_idx + 1,
                block_count,
                format_bytes(block_data.len() as u64),
                format_bytes(block_bytes.len() as u64),
                format_bytes(qtt_bytes as u64),
                format_bytes(res_bytes as u64)
            );
        }
    }

    writer.flush()?;
    drop(writer);

    // Update header with hash
    let final_hash: [u8; 32] = hasher.finalize().into();
    fluid.original_hash = final_hash;

    let mut file = File::options().write(true).open(output)?;
    file.seek(SeekFrom::Start(0))?;
    fluid.write_header(&mut file)?;
    fluid.write_index(&mut file)?;

    let elapsed = start.elapsed();
    let output_size = std::fs::metadata(output)?.len();

    if verbose {
        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════════╗");
        eprintln!("║                   S3 COMPRESSION COMPLETE                    ║");
        eprintln!("╠══════════════════════════════════════════════════════════════╣");
        eprintln!("║ Input:       {:>45} B ║", object_size);
        eprintln!("║ Output:      {:>45} B ║", output_size);
        eprintln!("║ QTT Cores:   {:>45} B ║", total_core_bytes);
        eprintln!("║ Residuals:   {:>45} B ║", total_residual_bytes);
        eprintln!("║ Ratio:       {:>47.2}x ║", object_size as f64 / output_size as f64);
        eprintln!("║ Time:        {:>44.2}s ║", elapsed.as_secs_f64());
        eprintln!("╚══════════════════════════════════════════════════════════════╝");
    }

    Ok(())
}

#[cfg(feature = "s3")]
fn parse_s3_uri(uri: &str) -> io::Result<(String, String)> {
    let path = uri
        .strip_prefix("s3://")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid S3 URI"))?;

    let mut parts = path.splitn(2, '/');
    let bucket = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing bucket"))?
        .to_string();
    let key = parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Missing key"))?
        .to_string();

    Ok((bucket, key))
}

// ============================================================================
// MAIN
// ============================================================================

#[cfg(feature = "s3")]
#[tokio::main]
async fn main() -> io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output, max_rank, verbose } => {
            compress_file(&input, &output, max_rank, verbose)?;
        }
        Commands::Decompress { input, output, verbose } => {
            decompress_file(&input, &output, verbose)?;
        }
        Commands::Extract { input, block, output } => {
            extract_block(&input, block, &output)?;
        }
        Commands::Cloud { input, output, region, max_rank, verbose } => {
            compress_from_s3(&input, &output, &region, max_rank, verbose).await?;
        }
        Commands::Benchmark { size, max_rank } => {
            benchmark(&size, max_rank);
        }
    }

    Ok(())
}

#[cfg(not(feature = "s3"))]
fn main() -> io::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Compress { input, output, max_rank, verbose } => {
            compress_file(&input, &output, max_rank, verbose)?;
        }
        Commands::Decompress { input, output, verbose } => {
            decompress_file(&input, &output, verbose)?;
        }
        Commands::Extract { input, block, output } => {
            extract_block(&input, block, &output)?;
        }
        Commands::Benchmark { size, max_rank } => {
            benchmark(&size, max_rank);
        }
    }

    Ok(())
}
