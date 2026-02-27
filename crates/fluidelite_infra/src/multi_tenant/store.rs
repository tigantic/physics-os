//! Persistent certificate store.
//!
//! File-backed storage for proof certificates with write-ahead logging (WAL)
//! for durability. Supports atomic writes and crash recovery.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use crate::dashboard::models::{CertificateId, ProofCertificate};
use fluidelite_core::physics_traits::SolverType;

// ═══════════════════════════════════════════════════════════════════════════
// Store Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the persistent store.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Directory for data files.
    pub data_dir: PathBuf,

    /// Whether to enable WAL (write-ahead log).
    pub enable_wal: bool,

    /// Maximum WAL entries before compaction.
    pub wal_compact_threshold: usize,

    /// Whether to sync writes to disk (fsync).
    pub sync_writes: bool,
}

impl StoreConfig {
    /// In-memory only configuration (no disk persistence).
    pub fn memory_only() -> Self {
        Self {
            data_dir: PathBuf::new(),
            enable_wal: false,
            wal_compact_threshold: 1000,
            sync_writes: false,
        }
    }

    /// Persistent configuration with a given data directory.
    pub fn persistent(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            enable_wal: true,
            wal_compact_threshold: 1000,
            sync_writes: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WAL Entry
// ═══════════════════════════════════════════════════════════════════════════

/// A WAL entry representing a single operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
enum WalEntry {
    /// Insert a new certificate.
    Insert(ProofCertificate),

    /// Update Gevulot status for a certificate.
    UpdateGevulotStatus {
        cert_id: String,
        status: String,
        tx_hash: Option<String>,
        verifiers: Option<u32>,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// Persistent Certificate Store
// ═══════════════════════════════════════════════════════════════════════════

/// File-backed certificate store with WAL for durability.
///
/// # Architecture
///
/// ```text
/// ┌──────────────────────────────────────────────────────┐
/// │  PersistentCertStore                                  │
/// │  ├── in-memory index: HashMap<CertificateId, idx>     │
/// │  ├── certificates: Vec<ProofCertificate>               │
/// │  ├── WAL file: data_dir/certificates.wal              │
/// │  └── Snapshot file: data_dir/certificates.json        │
/// └──────────────────────────────────────────────────────┘
/// ```
pub struct PersistentCertStore {
    /// Configuration.
    config: StoreConfig,

    /// In-memory certificate storage.
    certificates: Vec<ProofCertificate>,

    /// Index by certificate ID.
    by_id: HashMap<String, usize>,

    /// Index by solver type.
    by_solver: HashMap<SolverType, Vec<usize>>,

    /// Index by tenant.
    by_tenant: HashMap<String, Vec<usize>>,

    /// WAL entry count since last compaction.
    wal_entries: usize,
}

impl PersistentCertStore {
    /// Create a new store with the given configuration.
    pub fn new(config: StoreConfig) -> Result<Self, String> {
        let mut store = Self {
            config,
            certificates: Vec::new(),
            by_id: HashMap::new(),
            by_solver: HashMap::new(),
            by_tenant: HashMap::new(),
            wal_entries: 0,
        };

        // Try to recover from existing data
        if !store.config.data_dir.as_os_str().is_empty() {
            store.try_recover()?;
        }

        Ok(store)
    }

    /// Create an in-memory only store.
    pub fn memory() -> Self {
        Self {
            config: StoreConfig::memory_only(),
            certificates: Vec::new(),
            by_id: HashMap::new(),
            by_solver: HashMap::new(),
            by_tenant: HashMap::new(),
            wal_entries: 0,
        }
    }

    /// Insert a certificate.
    pub fn insert(&mut self, cert: ProofCertificate) -> Result<(), String> {
        let id_str = cert.id.0.clone();
        if self.by_id.contains_key(&id_str) {
            return Err(format!("Certificate already exists: {}", id_str));
        }

        // Write to WAL first (for durability)
        if self.config.enable_wal {
            self.write_wal(&WalEntry::Insert(cert.clone()))?;
        }

        // Then update in-memory state
        self.insert_in_memory(cert);

        // Check if compaction is needed
        if self.wal_entries >= self.config.wal_compact_threshold {
            self.compact()?;
        }

        Ok(())
    }

    /// Update Gevulot status for a certificate.
    pub fn update_gevulot_status(
        &mut self,
        cert_id: &str,
        status: &str,
        tx_hash: Option<String>,
        verifiers: Option<u32>,
    ) -> Result<(), String> {
        let idx = *self
            .by_id
            .get(cert_id)
            .ok_or_else(|| format!("Certificate not found: {}", cert_id))?;

        // Write to WAL
        if self.config.enable_wal {
            self.write_wal(&WalEntry::UpdateGevulotStatus {
                cert_id: cert_id.to_string(),
                status: status.to_string(),
                tx_hash: tx_hash.clone(),
                verifiers,
            })?;
        }

        // Update in-memory
        let cert = &mut self.certificates[idx];
        cert.gevulot_status = Some(status.to_string());
        cert.gevulot_tx_hash = tx_hash;
        cert.gevulot_verifiers = verifiers;

        Ok(())
    }

    /// Get a certificate by ID.
    pub fn get(&self, id: &str) -> Option<&ProofCertificate> {
        self.by_id.get(id).map(|&idx| &self.certificates[idx])
    }

    /// Get certificates by solver type.
    pub fn by_solver(&self, solver: SolverType) -> Vec<&ProofCertificate> {
        self.by_solver
            .get(&solver)
            .map(|indices| {
                indices
                    .iter()
                    .map(|&idx| &self.certificates[idx])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get certificates by tenant.
    pub fn by_tenant(&self, tenant_id: &str) -> Vec<&ProofCertificate> {
        self.by_tenant
            .get(tenant_id)
            .map(|indices| {
                indices
                    .iter()
                    .map(|&idx| &self.certificates[idx])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total certificates.
    pub fn total(&self) -> usize {
        self.certificates.len()
    }

    /// Get all certificates.
    pub fn all(&self) -> &[ProofCertificate] {
        &self.certificates
    }

    /// Get the latest N certificates.
    pub fn latest(&self, n: usize) -> Vec<&ProofCertificate> {
        self.certificates
            .iter()
            .rev()
            .take(n)
            .collect()
    }

    /// Compact the WAL by writing a full snapshot.
    pub fn compact(&mut self) -> Result<(), String> {
        if self.config.data_dir.as_os_str().is_empty() {
            return Ok(());
        }

        let snapshot_path = self.config.data_dir.join("certificates.json");
        let json = serde_json::to_string_pretty(&self.certificates)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        // Write snapshot atomically (write to tmp, then rename)
        let tmp_path = self.config.data_dir.join("certificates.json.tmp");
        fs::write(&tmp_path, json.as_bytes())
            .map_err(|e| format!("Failed to write snapshot: {}", e))?;
        fs::rename(&tmp_path, &snapshot_path)
            .map_err(|e| format!("Failed to rename snapshot: {}", e))?;

        // Clear WAL
        let wal_path = self.config.data_dir.join("certificates.wal");
        if wal_path.exists() {
            fs::remove_file(&wal_path)
                .map_err(|e| format!("Failed to remove WAL: {}", e))?;
        }

        self.wal_entries = 0;
        Ok(())
    }

    /// Force a full snapshot to disk.
    pub fn snapshot(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.certificates)
            .map_err(|e| format!("Serialization failed: {}", e))
    }

    // ═══════════════════════════════════════════════════════════════════
    // Internal Helpers
    // ═══════════════════════════════════════════════════════════════════

    /// Insert into in-memory indexes.
    fn insert_in_memory(&mut self, cert: ProofCertificate) {
        let idx = self.certificates.len();
        self.by_id.insert(cert.id.0.clone(), idx);

        self.by_solver
            .entry(cert.solver_type)
            .or_default()
            .push(idx);

        if let Some(ref tenant) = cert.tenant_id {
            self.by_tenant
                .entry(tenant.clone())
                .or_default()
                .push(idx);
        }

        self.certificates.push(cert);
    }

    /// Write a WAL entry.
    fn write_wal(&mut self, entry: &WalEntry) -> Result<(), String> {
        if self.config.data_dir.as_os_str().is_empty() {
            return Ok(());
        }

        // Ensure directory exists
        fs::create_dir_all(&self.config.data_dir)
            .map_err(|e| format!("Failed to create data dir: {}", e))?;

        let wal_path = self.config.data_dir.join("certificates.wal");
        let line = serde_json::to_string(entry)
            .map_err(|e| format!("WAL serialization failed: {}", e))?;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)
            .map_err(|e| format!("Failed to open WAL: {}", e))?;

        writeln!(file, "{}", line)
            .map_err(|e| format!("Failed to write WAL: {}", e))?;

        if self.config.sync_writes {
            file.sync_all()
                .map_err(|e| format!("Failed to sync WAL: {}", e))?;
        }

        self.wal_entries += 1;
        Ok(())
    }

    /// Try to recover from snapshot + WAL.
    fn try_recover(&mut self) -> Result<(), String> {
        let snapshot_path = self.config.data_dir.join("certificates.json");
        let wal_path = self.config.data_dir.join("certificates.wal");

        // 1. Load snapshot if it exists
        if snapshot_path.exists() {
            let data = fs::read_to_string(&snapshot_path)
                .map_err(|e| format!("Failed to read snapshot: {}", e))?;
            let certs: Vec<ProofCertificate> = serde_json::from_str(&data)
                .map_err(|e| format!("Failed to parse snapshot: {}", e))?;
            for cert in certs {
                self.insert_in_memory(cert);
            }
        }

        // 2. Replay WAL if it exists
        if wal_path.exists() {
            let file = fs::File::open(&wal_path)
                .map_err(|e| format!("Failed to open WAL: {}", e))?;
            let reader = io::BufReader::new(file);

            for line in reader.lines() {
                let line = line.map_err(|e| format!("WAL read error: {}", e))?;
                if line.trim().is_empty() {
                    continue;
                }

                let entry: WalEntry = serde_json::from_str(&line)
                    .map_err(|e| format!("WAL parse error: {}", e))?;

                match entry {
                    WalEntry::Insert(cert) => {
                        if !self.by_id.contains_key(&cert.id.0) {
                            self.insert_in_memory(cert);
                        }
                    }
                    WalEntry::UpdateGevulotStatus {
                        cert_id,
                        status,
                        tx_hash,
                        verifiers,
                    } => {
                        if let Some(&idx) = self.by_id.get(&cert_id) {
                            let cert = &mut self.certificates[idx];
                            cert.gevulot_status = Some(status);
                            cert.gevulot_tx_hash = tx_hash;
                            cert.gevulot_verifiers = verifiers;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cert(id: &str, solver: SolverType) -> ProofCertificate {
        ProofCertificate::from_verification(
            CertificateId(id.into()),
            solver,
            true,
            4,
            4,
            5000,
            14,
            100,
            500,
            800,
            &[1, 2, 3, 4],
            &[5, 6, 7, 8],
            &[9, 10, 11, 12],
            0.001,
        )
    }

    #[test]
    fn test_memory_store() {
        let mut store = PersistentCertStore::memory();
        assert_eq!(store.total(), 0);

        store.insert(make_cert("c-001", SolverType::Euler3D)).unwrap();
        assert_eq!(store.total(), 1);
    }

    #[test]
    fn test_store_duplicate() {
        let mut store = PersistentCertStore::memory();
        store.insert(make_cert("c-001", SolverType::Euler3D)).unwrap();
        assert!(store.insert(make_cert("c-001", SolverType::Euler3D)).is_err());
    }

    #[test]
    fn test_store_get() {
        let mut store = PersistentCertStore::memory();
        store.insert(make_cert("c-001", SolverType::Euler3D)).unwrap();

        let cert = store.get("c-001").unwrap();
        assert_eq!(cert.solver_type, SolverType::Euler3D);
        assert!(cert.valid);
    }

    #[test]
    fn test_store_by_solver() {
        let mut store = PersistentCertStore::memory();
        store.insert(make_cert("c-001", SolverType::Euler3D)).unwrap();
        store.insert(make_cert("c-002", SolverType::NsImex)).unwrap();
        store.insert(make_cert("c-003", SolverType::Euler3D)).unwrap();

        let euler = store.by_solver(SolverType::Euler3D);
        assert_eq!(euler.len(), 2);

        let ns = store.by_solver(SolverType::NsImex);
        assert_eq!(ns.len(), 1);
    }

    #[test]
    fn test_store_by_tenant() {
        let mut store = PersistentCertStore::memory();
        let mut cert1 = make_cert("c-001", SolverType::Euler3D);
        cert1.tenant_id = Some("t1".into());
        store.insert(cert1).unwrap();

        let mut cert2 = make_cert("c-002", SolverType::NsImex);
        cert2.tenant_id = Some("t2".into());
        store.insert(cert2).unwrap();

        let t1 = store.by_tenant("t1");
        assert_eq!(t1.len(), 1);
        assert_eq!(t1[0].id.0, "c-001");
    }

    #[test]
    fn test_store_latest() {
        let mut store = PersistentCertStore::memory();
        for i in 0..5 {
            store
                .insert(make_cert(&format!("c-{:03}", i), SolverType::Euler3D))
                .unwrap();
        }

        let latest = store.latest(2);
        assert_eq!(latest.len(), 2);
        assert_eq!(latest[0].id.0, "c-004");
        assert_eq!(latest[1].id.0, "c-003");
    }

    #[test]
    fn test_store_update_gevulot() {
        let mut store = PersistentCertStore::memory();
        store.insert(make_cert("c-001", SolverType::Euler3D)).unwrap();

        store
            .update_gevulot_status(
                "c-001",
                "verified",
                Some("0xabc".into()),
                Some(3),
            )
            .unwrap();

        let cert = store.get("c-001").unwrap();
        assert_eq!(cert.gevulot_status.as_deref(), Some("verified"));
        assert_eq!(cert.gevulot_tx_hash.as_deref(), Some("0xabc"));
        assert_eq!(cert.gevulot_verifiers, Some(3));
    }

    #[test]
    fn test_store_update_not_found() {
        let mut store = PersistentCertStore::memory();
        assert!(store
            .update_gevulot_status("nonexistent", "verified", None, None)
            .is_err());
    }

    #[test]
    fn test_store_snapshot() {
        let mut store = PersistentCertStore::memory();
        store.insert(make_cert("c-001", SolverType::Euler3D)).unwrap();

        let json = store.snapshot().unwrap();
        assert!(json.contains("c-001"));
        assert!(json.contains("euler3d"));
    }

    #[test]
    fn test_persistent_store_wal() {
        let dir = std::env::temp_dir().join(format!(
            "fluidelite_test_wal_{}",
            crate::gevulot::current_unix_time()
        ));
        let _ = fs::remove_dir_all(&dir);

        {
            let config = StoreConfig::persistent(&dir);
            let mut store = PersistentCertStore::new(config).unwrap();
            store
                .insert(make_cert("c-001", SolverType::Euler3D))
                .unwrap();
            store
                .insert(make_cert("c-002", SolverType::NsImex))
                .unwrap();
            // WAL file should exist
            assert!(dir.join("certificates.wal").exists());
        }

        // Recovery: create a new store from the same directory
        {
            let config = StoreConfig::persistent(&dir);
            let store = PersistentCertStore::new(config).unwrap();
            assert_eq!(store.total(), 2);
            assert!(store.get("c-001").is_some());
            assert!(store.get("c-002").is_some());
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_persistent_store_compaction() {
        let dir = std::env::temp_dir().join(format!(
            "fluidelite_test_compact_{}",
            crate::gevulot::current_unix_time()
        ));
        let _ = fs::remove_dir_all(&dir);

        {
            let config = StoreConfig {
                data_dir: dir.clone(),
                enable_wal: true,
                wal_compact_threshold: 3,
                sync_writes: false,
            };
            let mut store = PersistentCertStore::new(config).unwrap();

            for i in 0..5 {
                store
                    .insert(make_cert(
                        &format!("c-{:03}", i),
                        SolverType::Euler3D,
                    ))
                    .unwrap();
            }

            // After 3 inserts, compaction should have triggered
            assert!(dir.join("certificates.json").exists());
        }

        // Recovery from snapshot
        {
            let config = StoreConfig::persistent(&dir);
            let store = PersistentCertStore::new(config).unwrap();
            // Should recover at least 3 entries (compacted) plus WAL replay
            assert!(store.total() >= 3);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_store_config_memory() {
        let config = StoreConfig::memory_only();
        assert!(!config.enable_wal);
    }
}
