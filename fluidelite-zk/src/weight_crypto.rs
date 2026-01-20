//! Weight Encryption Module for Enterprise IP Protection
//!
//! This module provides AES-256-GCM encryption for model weights,
//! enabling secure deployment on decentralized proving networks
//! without exposing proprietary model parameters.
//!
//! # Enterprise Use Case
//!
//! Regulated financial institutions (Canton Network, etc.) require:
//! 1. Model weights encrypted at rest
//! 2. Runtime decryption using TEE-derived keys
//! 3. Audit trail for key usage
//!
//! # Key Derivation
//!
//! Keys are derived using HKDF-SHA256 from:
//! - Hardware attestation (TEE measurement)
//! - Model version hash
//! - Customer-specific salt

use std::fs;
use std::io::{self, Read};
use std::path::Path;

/// AES-256-GCM nonce size
const NONCE_SIZE: usize = 12;

/// AES-256-GCM tag size
const TAG_SIZE: usize = 16;

/// Magic bytes for encrypted weight files
const MAGIC: &[u8; 8] = b"FLENC001";

/// Encrypted weight file header
#[derive(Debug)]
pub struct EncryptedHeader {
    /// Magic bytes ("FLENC001")
    pub magic: [u8; 8],
    /// Version of encryption format
    pub version: u32,
    /// Original unencrypted size
    pub original_size: u64,
    /// SHA256 of original (for integrity check)
    pub original_hash: [u8; 32],
    /// Nonce for AES-GCM
    pub nonce: [u8; NONCE_SIZE],
    /// Key ID (for key rotation support)
    pub key_id: [u8; 16],
}

impl EncryptedHeader {
    /// Header size in bytes
    pub const SIZE: usize = 8 + 4 + 8 + 32 + NONCE_SIZE + 16;

    /// Parse header from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<Self> {
        if data.len() < Self::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header too short",
            ));
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&data[0..8]);
        
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic bytes - not an encrypted weight file",
            ));
        }

        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let original_size = u64::from_le_bytes(data[12..20].try_into().unwrap());
        
        let mut original_hash = [0u8; 32];
        original_hash.copy_from_slice(&data[20..52]);
        
        let mut nonce = [0u8; NONCE_SIZE];
        nonce.copy_from_slice(&data[52..64]);
        
        let mut key_id = [0u8; 16];
        key_id.copy_from_slice(&data[64..80]);

        Ok(Self {
            magic,
            version,
            original_size,
            original_hash,
            nonce,
            key_id,
        })
    }

    /// Serialize header to bytes
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.version.to_le_bytes());
        buf[12..20].copy_from_slice(&self.original_size.to_le_bytes());
        buf[20..52].copy_from_slice(&self.original_hash);
        buf[52..64].copy_from_slice(&self.nonce);
        buf[64..80].copy_from_slice(&self.key_id);
        buf
    }
}

/// Weight encryption/decryption using AES-256-GCM
/// 
/// This is a software implementation. For production TEE deployment,
/// the key derivation would use hardware attestation measurements.
#[cfg(feature = "encryption")]
pub mod crypto {
    use super::*;
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use sha2::{Digest, Sha256};

    /// Encryption key (32 bytes for AES-256)
    pub struct WeightKey([u8; 32]);

    impl WeightKey {
        /// Derive key from password and salt using HKDF
        pub fn derive(password: &[u8], salt: &[u8]) -> Self {
            use hkdf::Hkdf;
            
            let hk = Hkdf::<Sha256>::new(Some(salt), password);
            let mut key = [0u8; 32];
            hk.expand(b"fluidelite-weight-encryption-v1", &mut key)
                .expect("HKDF expand failed");
            Self(key)
        }

        /// Create key from raw bytes (for TEE-derived keys)
        pub fn from_bytes(bytes: [u8; 32]) -> Self {
            Self(bytes)
        }

        /// Encrypt weight file
        pub fn encrypt_weights(&self, plaintext: &[u8]) -> io::Result<Vec<u8>> {
            let cipher = Aes256Gcm::new_from_slice(&self.0)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            
            // Generate random nonce
            let mut nonce_bytes = [0u8; NONCE_SIZE];
            getrandom::getrandom(&mut nonce_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            let nonce = Nonce::from_slice(&nonce_bytes);
            
            // Hash original for integrity check
            let mut hasher = Sha256::new();
            hasher.update(plaintext);
            let original_hash: [u8; 32] = hasher.finalize().into();
            
            // Encrypt
            let ciphertext = cipher
                .encrypt(nonce, plaintext)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
            
            // Build header
            let header = EncryptedHeader {
                magic: *MAGIC,
                version: 1,
                original_size: plaintext.len() as u64,
                original_hash,
                nonce: nonce_bytes,
                key_id: [0u8; 16], // No key rotation for now
            };
            
            // Combine header + ciphertext
            let mut output = Vec::with_capacity(EncryptedHeader::SIZE + ciphertext.len());
            output.extend_from_slice(&header.to_bytes());
            output.extend_from_slice(&ciphertext);
            
            Ok(output)
        }

        /// Decrypt weight file
        pub fn decrypt_weights(&self, encrypted: &[u8]) -> io::Result<Vec<u8>> {
            let header = EncryptedHeader::from_bytes(encrypted)?;
            
            let cipher = Aes256Gcm::new_from_slice(&self.0)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            
            let nonce = Nonce::from_slice(&header.nonce);
            let ciphertext = &encrypted[EncryptedHeader::SIZE..];
            
            let plaintext = cipher
                .decrypt(nonce, ciphertext)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Decryption failed - wrong key or corrupted data"))?;
            
            // Verify hash
            let mut hasher = Sha256::new();
            hasher.update(&plaintext);
            let computed_hash: [u8; 32] = hasher.finalize().into();
            
            if computed_hash != header.original_hash {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Hash mismatch - data corrupted after decryption",
                ));
            }
            
            if plaintext.len() as u64 != header.original_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Size mismatch - data corrupted",
                ));
            }
            
            Ok(plaintext)
        }
    }

    /// Encrypt a weight file on disk
    pub fn encrypt_file(input_path: &Path, output_path: &Path, key: &WeightKey) -> io::Result<()> {
        let plaintext = fs::read(input_path)?;
        let encrypted = key.encrypt_weights(&plaintext)?;
        fs::write(output_path, encrypted)?;
        Ok(())
    }

    /// Decrypt a weight file on disk
    pub fn decrypt_file(input_path: &Path, output_path: &Path, key: &WeightKey) -> io::Result<()> {
        let encrypted = fs::read(input_path)?;
        let plaintext = key.decrypt_weights(&encrypted)?;
        fs::write(output_path, plaintext)?;
        Ok(())
    }
}

/// Check if a file is an encrypted weight file
pub fn is_encrypted(path: &Path) -> io::Result<bool> {
    let mut file = fs::File::open(path)?;
    let mut magic = [0u8; 8];
    
    if file.read_exact(&mut magic).is_err() {
        return Ok(false);
    }
    
    Ok(&magic == MAGIC)
}

/// Get header info from encrypted file (for debugging)
pub fn get_header_info(path: &Path) -> io::Result<EncryptedHeader> {
    let data = fs::read(path)?;
    EncryptedHeader::from_bytes(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = EncryptedHeader {
            magic: *MAGIC,
            version: 1,
            original_size: 1024 * 1024,
            original_hash: [42u8; 32],
            nonce: [1u8; NONCE_SIZE],
            key_id: [0u8; 16],
        };
        
        let bytes = header.to_bytes();
        let parsed = EncryptedHeader::from_bytes(&bytes).unwrap();
        
        assert_eq!(parsed.magic, header.magic);
        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.original_size, header.original_size);
        assert_eq!(parsed.original_hash, header.original_hash);
        assert_eq!(parsed.nonce, header.nonce);
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        use crypto::WeightKey;
        
        let key = WeightKey::derive(b"test-password", b"test-salt");
        let original = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        
        let encrypted = key.encrypt_weights(&original).unwrap();
        let decrypted = key.decrypt_weights(&encrypted).unwrap();
        
        assert_eq!(decrypted, original);
    }
}
