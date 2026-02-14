//! Trusted setup parameter management
//!
//! Provides deterministic KZG parameter generation with filesystem caching.
//! Parameters are expensive to generate (~seconds for k=17) but identical
//! across runs, so we cache them to disk and verify integrity via SHA-256.
//!
//! # Cache Layout
//!
//! ```text
//! $FLUIDELITE_PARAMS_DIR/
//!   kzg_bn254_k10.params   (+ .sha256)
//!   kzg_bn254_k17.params   (+ .sha256)
//!   ...
//! ```
//!
//! The directory defaults to `$HOME/.fluidelite/params/` if `FLUIDELITE_PARAMS_DIR`
//! is not set.

use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use halo2_axiom::{
    halo2curves::bn256::Bn256,
    poly::kzg::commitment::ParamsKZG,
    SerdeFormat,
};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};

use crate::circuit::config::{MAX_K, MIN_K};

/// Environment variable for overriding the parameter cache directory.
const PARAMS_DIR_ENV: &str = "FLUIDELITE_PARAMS_DIR";

/// Default subdirectory under `$HOME` for parameter storage.
const DEFAULT_PARAMS_SUBDIR: &str = ".fluidelite/params";

/// File extension for serialized KZG parameters.
const PARAMS_EXT: &str = "params";

/// File extension for SHA-256 integrity digests.
const DIGEST_EXT: &str = "sha256";

/// Errors that can occur during parameter management.
#[derive(Debug)]
pub enum ParamsError {
    /// The requested k value is outside the valid range.
    InvalidK { k: u32, min: u32, max: u32 },
    /// Filesystem I/O error during read/write.
    Io(io::Error),
    /// The cached file failed SHA-256 integrity verification.
    IntegrityCheckFailed {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    /// Could not determine a cache directory (no $HOME, no env override).
    NoCacheDir,
}

impl std::fmt::Display for ParamsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamsError::InvalidK { k, min, max } => {
                write!(f, "k={k} is outside valid range [{min}, {max}]")
            }
            ParamsError::Io(e) => write!(f, "I/O error: {e}"),
            ParamsError::IntegrityCheckFailed {
                path,
                expected,
                actual,
            } => write!(
                f,
                "integrity check failed for {}: expected {expected}, got {actual}",
                path.display()
            ),
            ParamsError::NoCacheDir => write!(
                f,
                "could not determine params cache directory — \
                 set ${PARAMS_DIR_ENV} or ensure $HOME is defined"
            ),
        }
    }
}

impl std::error::Error for ParamsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParamsError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ParamsError {
    fn from(e: io::Error) -> Self {
        ParamsError::Io(e)
    }
}

/// Resolve the parameter cache directory.
///
/// Priority:
/// 1. `$FLUIDELITE_PARAMS_DIR` environment variable
/// 2. `$HOME/.fluidelite/params/`
fn params_dir() -> Result<PathBuf, ParamsError> {
    if let Ok(dir) = std::env::var(PARAMS_DIR_ENV) {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(home) = std::env::var("HOME") {
        return Ok(PathBuf::from(home).join(DEFAULT_PARAMS_SUBDIR));
    }
    // Windows fallback
    if let Ok(profile) = std::env::var("USERPROFILE") {
        return Ok(PathBuf::from(profile).join(DEFAULT_PARAMS_SUBDIR));
    }
    Err(ParamsError::NoCacheDir)
}

/// Canonical filename for parameters at a given k.
fn params_filename(k: u32) -> String {
    format!("kzg_bn254_k{k}.{PARAMS_EXT}")
}

/// Canonical filename for the SHA-256 digest accompanying a params file.
fn digest_filename(k: u32) -> String {
    format!("kzg_bn254_k{k}.{DIGEST_EXT}")
}

/// Compute the SHA-256 hex digest of a file.
fn sha256_file(path: &Path) -> Result<String, ParamsError> {
    let f = fs::File::open(path)?;
    let mut reader = BufReader::with_capacity(1 << 16, f);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Write a SHA-256 digest file alongside a params file.
fn write_digest(params_path: &Path, digest: &str) -> Result<(), ParamsError> {
    let digest_path = params_path.with_extension(DIGEST_EXT);
    fs::write(&digest_path, digest)?;
    Ok(())
}

/// Read the expected SHA-256 digest from the companion file.
fn read_digest(params_path: &Path) -> Result<String, ParamsError> {
    let digest_path = params_path.with_extension(DIGEST_EXT);
    let content = fs::read_to_string(&digest_path)?;
    Ok(content.trim().to_string())
}

/// Load KZG parameters from a cached file, verifying SHA-256 integrity.
fn load_cached(path: &Path) -> Result<ParamsKZG<Bn256>, ParamsError> {
    // Verify integrity first
    let expected = read_digest(path)?;
    let actual = sha256_file(path)?;
    if expected != actual {
        return Err(ParamsError::IntegrityCheckFailed {
            path: path.to_path_buf(),
            expected,
            actual,
        });
    }

    let f = fs::File::open(path)?;
    let mut reader = BufReader::new(f);
    let params = ParamsKZG::<Bn256>::read_custom(&mut reader, SerdeFormat::RawBytes)
        .map_err(|e| ParamsError::Io(io::Error::new(io::ErrorKind::InvalidData, e.to_string())))?;
    Ok(params)
}

/// Serialize KZG parameters to a file and write the companion SHA-256 digest.
fn save_cached(path: &Path, params: &ParamsKZG<Bn256>) -> Result<(), ParamsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let f = fs::File::create(path)?;
    let mut writer = BufWriter::new(f);
    params
        .write_custom(&mut writer, SerdeFormat::RawBytes)
        .map_err(|e| ParamsError::Io(io::Error::new(io::ErrorKind::Other, e.to_string())))?;
    writer.flush()?;

    let digest = sha256_file(path)?;
    write_digest(path, &digest)?;

    Ok(())
}

/// Load or generate KZG parameters for the given circuit size.
///
/// This is the primary API for parameter management. It:
/// 1. Checks the filesystem cache for pre-computed parameters.
/// 2. If found, verifies SHA-256 integrity and returns them.
/// 3. If not found (or corrupt), generates fresh parameters via
///    `ParamsKZG::setup(k, OsRng)`, caches them, and returns them.
///
/// # Arguments
///
/// * `k` — log₂ of the number of rows. Must be in [`MIN_K`, `MAX_K`].
///
/// # Errors
///
/// Returns `ParamsError` if k is out of range, filesystem operations fail,
/// or a cached file fails integrity verification (in which case the corrupt
/// file is deleted and regeneration is attempted once).
pub fn load_or_generate_params(k: u32) -> Result<ParamsKZG<Bn256>, ParamsError> {
    let dir = params_dir()?;
    load_or_generate_params_in(k, &dir)
}

/// Like [`load_or_generate_params`] but with an explicit cache directory.
///
/// This is useful for tests that need isolation and for deployment scenarios
/// where the cache location is known at call time.
pub fn load_or_generate_params_in(
    k: u32,
    dir: &Path,
) -> Result<ParamsKZG<Bn256>, ParamsError> {
    if k < MIN_K || k > MAX_K {
        return Err(ParamsError::InvalidK {
            k,
            min: MIN_K,
            max: MAX_K,
        });
    }

    let path = dir.join(params_filename(k));

    // Attempt cache hit
    if path.exists() {
        match load_cached(&path) {
            Ok(params) => {
                eprintln!(
                    "[params] loaded cached KZG params k={k} from {}",
                    path.display()
                );
                return Ok(params);
            }
            Err(ParamsError::IntegrityCheckFailed { .. }) => {
                eprintln!(
                    "[params] WARNING: corrupt cache for k={k} at {} — regenerating",
                    path.display()
                );
                // Delete corrupt files and fall through to regeneration
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(path.with_extension(DIGEST_EXT));
            }
            Err(e) => return Err(e),
        }
    }

    // Generate fresh parameters
    eprintln!("[params] generating KZG params k={k} (2^{k} = {} rows)...", 1u64 << k);
    let start = Instant::now();
    let params = ParamsKZG::<Bn256>::setup(k, OsRng);
    let elapsed = start.elapsed();
    eprintln!(
        "[params] KZG params k={k} generated in {:.2?}",
        elapsed
    );

    // Cache to disk
    match save_cached(&path, &params) {
        Ok(()) => {
            eprintln!("[params] cached to {}", path.display());
        }
        Err(e) => {
            eprintln!(
                "[params] WARNING: failed to cache k={k} params: {e} — continuing without cache"
            );
        }
    }

    Ok(params)
}

/// Delete all cached parameters for a given k.
///
/// Useful for forcing regeneration after a trusted setup ceremony update.
pub fn clear_cached_params(k: u32) -> Result<(), ParamsError> {
    let dir = params_dir()?;
    clear_cached_params_in(k, &dir)
}

/// Like [`clear_cached_params`] but with an explicit cache directory.
pub fn clear_cached_params_in(k: u32, dir: &Path) -> Result<(), ParamsError> {
    let path = dir.join(params_filename(k));
    let digest_path = dir.join(digest_filename(k));

    if path.exists() {
        fs::remove_file(&path)?;
    }
    if digest_path.exists() {
        fs::remove_file(&digest_path)?;
    }
    Ok(())
}

/// Check whether cached parameters exist and are valid for the given k.
pub fn params_cached(k: u32) -> bool {
    let dir = match params_dir() {
        Ok(d) => d,
        Err(_) => return false,
    };
    params_cached_in(k, &dir)
}

/// Like [`params_cached`] but with an explicit cache directory.
pub fn params_cached_in(k: u32, dir: &Path) -> bool {
    let path = dir.join(params_filename(k));
    if !path.exists() {
        return false;
    }
    // Quick check: digest file must also exist
    path.with_extension(DIGEST_EXT).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_axiom::poly::commitment::Params;

    /// Create a unique temp dir per test to avoid parallel env var races.
    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("fluidelite_params_test")
            .join(name);
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_load_or_generate_small_k() {
        let tmp = test_dir("load_gen");
        let k = 8;

        // First call: generate
        let params1 = load_or_generate_params_in(k, &tmp).expect("generate failed");
        assert_eq!(params1.k() as u32, k);

        // Verify cache was written
        let path = tmp.join(params_filename(k));
        assert!(path.exists(), "params file should exist after generation");
        assert!(
            path.with_extension(DIGEST_EXT).exists(),
            "digest file should exist"
        );
        assert!(params_cached_in(k, &tmp));

        // Second call: load from cache
        let params2 = load_or_generate_params_in(k, &tmp).expect("load failed");
        assert_eq!(params2.k() as u32, k);

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_invalid_k_rejected() {
        let tmp = test_dir("invalid_k");
        let err = load_or_generate_params_in(3, &tmp);
        assert!(err.is_err());
        let err = load_or_generate_params_in(25, &tmp);
        assert!(err.is_err());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_integrity_failure_triggers_regeneration() {
        let tmp = test_dir("integrity");
        let k = 8;

        // Generate first
        let params1 = load_or_generate_params_in(k, &tmp).expect("generate");
        assert_eq!(params1.k() as u32, k);

        // Corrupt the digest file
        let digest_path = tmp.join(params_filename(k)).with_extension(DIGEST_EXT);
        fs::write(
            &digest_path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        )
        .unwrap();

        // Next load should detect corruption, regenerate, and succeed
        let params2 =
            load_or_generate_params_in(k, &tmp).expect("should regenerate after corruption");
        assert_eq!(params2.k() as u32, k);

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_clear_cached_params() {
        let tmp = test_dir("clear");
        let k = 8;

        let _ = load_or_generate_params_in(k, &tmp).expect("generate");
        assert!(params_cached_in(k, &tmp));

        clear_cached_params_in(k, &tmp).expect("clear should succeed");
        assert!(!params_cached_in(k, &tmp));

        // Cleanup
        let _ = fs::remove_dir_all(&tmp);
    }
}
