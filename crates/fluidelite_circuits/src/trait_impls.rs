//! Trait implementations connecting physics circuits to the `PhysicsProof`,
//! `PhysicsProver`, and `PhysicsVerifier` traits defined in `fluidelite_core`.
//!
//! These impl blocks bridge the concrete Euler3D, NS-IMEX, and Thermal types
//! to the unified trait interface used by batch provers, Gevulot, and dashboards.
//!
//! © 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.

use fluidelite_core::mpo::MPO;
use fluidelite_core::mps::MPS;
use fluidelite_core::physics_traits::{
    PhysicsProof, PhysicsProver, PhysicsVerifier, ProverFactory, SolverType,
    UnifiedVerificationResult,
};

// ═══════════════════════════════════════════════════════════════════════════
// Implementations for Euler 3D
// ═══════════════════════════════════════════════════════════════════════════

impl PhysicsProof for crate::euler3d::Euler3DProof {
    fn solver_type(&self) -> SolverType {
        SolverType::Euler3D
    }

    fn proof_bytes(&self) -> &[u8] {
        &self.proof_bytes
    }

    fn to_serialized_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }

    fn generation_time_ms(&self) -> u64 {
        self.generation_time_ms
    }

    fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    fn k(&self) -> u32 {
        self.k
    }

    fn proof_size(&self) -> usize {
        self.size()
    }

    fn input_hash_limbs(&self) -> &[u64; 4] {
        &self.input_state_hash_limbs
    }

    fn output_hash_limbs(&self) -> &[u64; 4] {
        &self.output_state_hash_limbs
    }

    fn params_hash_limbs(&self) -> &[u64; 4] {
        &self.params_hash_limbs
    }

    fn grid_bits(&self) -> usize {
        self.params.grid_bits
    }

    fn chi_max(&self) -> usize {
        self.params.chi_max
    }
}

impl PhysicsProver for crate::euler3d::Euler3DProver {
    type Proof = crate::euler3d::Euler3DProof;

    fn solver_type(&self) -> SolverType {
        SolverType::Euler3D
    }

    fn prove(
        &mut self,
        input_states: &[MPS],
        shift_mpos: &[MPO],
    ) -> Result<Self::Proof, String> {
        self.prove(input_states, shift_mpos)
    }

    fn total_proofs(&self) -> usize {
        self.stats().total_proofs
    }

    fn total_time_ms(&self) -> u64 {
        self.stats().total_time_ms
    }
}

impl PhysicsVerifier for crate::euler3d::Euler3DVerifier {
    type Proof = crate::euler3d::Euler3DProof;

    fn verify(
        &self,
        proof: &Self::Proof,
    ) -> Result<UnifiedVerificationResult, String> {
        let result = self.verify(proof)?;
        let max_residual = result
            .conservation_residuals
            .iter()
            .map(|r| r.to_f64().abs())
            .fold(0.0f64, f64::max);

        Ok(UnifiedVerificationResult {
            valid: result.valid,
            verification_time_us: result.verification_time_us,
            num_constraints: result.num_constraints,
            grid_bits: result.grid_bits,
            chi_max: result.chi_max,
            solver_type: SolverType::Euler3D,
            max_residual,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Implementations for NS-IMEX
// ═══════════════════════════════════════════════════════════════════════════

impl PhysicsProof for crate::ns_imex::NSIMEXProof {
    fn solver_type(&self) -> SolverType {
        SolverType::NsImex
    }

    fn proof_bytes(&self) -> &[u8] {
        &self.proof_bytes
    }

    fn to_serialized_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }

    fn generation_time_ms(&self) -> u64 {
        self.generation_time_ms
    }

    fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    fn k(&self) -> u32 {
        self.k
    }

    fn proof_size(&self) -> usize {
        self.size()
    }

    fn input_hash_limbs(&self) -> &[u64; 4] {
        &self.input_state_hash_limbs
    }

    fn output_hash_limbs(&self) -> &[u64; 4] {
        &self.output_state_hash_limbs
    }

    fn params_hash_limbs(&self) -> &[u64; 4] {
        &self.params_hash_limbs
    }

    fn grid_bits(&self) -> usize {
        self.params.grid_bits
    }

    fn chi_max(&self) -> usize {
        self.params.chi_max
    }
}

impl PhysicsProver for crate::ns_imex::NSIMEXProver {
    type Proof = crate::ns_imex::NSIMEXProof;

    fn solver_type(&self) -> SolverType {
        SolverType::NsImex
    }

    fn prove(
        &mut self,
        input_states: &[MPS],
        shift_mpos: &[MPO],
    ) -> Result<Self::Proof, String> {
        self.prove(input_states, shift_mpos)
    }

    fn total_proofs(&self) -> usize {
        self.stats().total_proofs
    }

    fn total_time_ms(&self) -> u64 {
        self.stats().total_time_ms
    }
}

impl PhysicsVerifier for crate::ns_imex::NSIMEXVerifier {
    type Proof = crate::ns_imex::NSIMEXProof;

    fn verify(
        &self,
        proof: &Self::Proof,
    ) -> Result<UnifiedVerificationResult, String> {
        let result = self.verify(proof)?;
        let max_residual = [
            result.ke_residual.to_f64().abs(),
            result.enstrophy_residual.to_f64().abs(),
            result.divergence_residual.to_f64().abs(),
        ]
        .into_iter()
        .fold(0.0f64, f64::max);

        Ok(UnifiedVerificationResult {
            valid: result.valid,
            verification_time_us: result.verification_time_us,
            num_constraints: result.num_constraints,
            grid_bits: result.grid_bits,
            chi_max: result.chi_max,
            solver_type: SolverType::NsImex,
            max_residual,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Implementations for Thermal
// ═══════════════════════════════════════════════════════════════════════════

impl PhysicsProof for crate::thermal::ThermalProof {
    fn solver_type(&self) -> SolverType {
        SolverType::Thermal
    }

    fn proof_bytes(&self) -> &[u8] {
        &self.proof_bytes
    }

    fn to_serialized_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }

    fn generation_time_ms(&self) -> u64 {
        self.generation_time_ms
    }

    fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    fn k(&self) -> u32 {
        self.k
    }

    fn proof_size(&self) -> usize {
        self.size()
    }

    fn input_hash_limbs(&self) -> &[u64; 4] {
        &self.input_state_hash_limbs
    }

    fn output_hash_limbs(&self) -> &[u64; 4] {
        &self.output_state_hash_limbs
    }

    fn params_hash_limbs(&self) -> &[u64; 4] {
        &self.params_hash_limbs
    }

    fn grid_bits(&self) -> usize {
        self.params.grid_bits
    }

    fn chi_max(&self) -> usize {
        self.params.chi_max
    }
}

impl PhysicsProver for crate::thermal::ThermalProver {
    type Proof = crate::thermal::ThermalProof;

    fn solver_type(&self) -> SolverType {
        SolverType::Thermal
    }

    fn prove(
        &mut self,
        input_states: &[MPS],
        shift_mpos: &[MPO],
    ) -> Result<Self::Proof, String> {
        self.prove(input_states, shift_mpos)
    }

    fn total_proofs(&self) -> usize {
        self.stats().total_proofs
    }

    fn total_time_ms(&self) -> u64 {
        self.stats().total_time_ms
    }
}

impl PhysicsVerifier for crate::thermal::ThermalVerifier {
    type Proof = crate::thermal::ThermalProof;

    fn verify(
        &self,
        proof: &Self::Proof,
    ) -> Result<UnifiedVerificationResult, String> {
        let result = self.verify(proof)?;
        let max_residual = result.conservation_residual.to_f64().abs();

        Ok(UnifiedVerificationResult {
            valid: result.valid,
            verification_time_us: result.verification_time_us,
            num_constraints: result.num_constraints,
            grid_bits: result.grid_bits,
            chi_max: result.chi_max,
            solver_type: SolverType::Thermal,
            max_residual,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Prover Factory Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Create a factory for Euler 3D provers.
pub fn euler3d_factory(
    params: crate::euler3d::Euler3DParams,
) -> ProverFactory<crate::euler3d::Euler3DProver> {
    Box::new(move || crate::euler3d::Euler3DProver::new(params.clone()))
}

/// Create a factory for NS-IMEX provers.
pub fn ns_imex_factory(
    params: crate::ns_imex::NSIMEXParams,
) -> ProverFactory<crate::ns_imex::NSIMEXProver> {
    Box::new(move || crate::ns_imex::NSIMEXProver::new(params.clone()))
}

/// Create a factory for Thermal provers.
pub fn thermal_factory(
    params: crate::thermal::ThermalParams,
) -> ProverFactory<crate::thermal::ThermalProver> {
    Box::new(move || crate::thermal::ThermalProver::new(params.clone()))
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════
