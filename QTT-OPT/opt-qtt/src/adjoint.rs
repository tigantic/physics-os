//! Adjoint Sensitivity Analysis
//!
//! For minimum compliance topology optimization:
//!   J = Fᵀu = uᵀKu  (compliance = strain energy × 2)
//!
//! Adjoint equation: Kλ = -dJ/du = -F → λ = -u
//!   (self-adjoint for compliance problems!)
//!
//! Sensitivity:
//!   dJ/dρₑ = -λᵀ (∂K/∂ρₑ) u = uₑᵀ (∂Kₑ/∂ρₑ) uₑ
//!
//! With SIMP: E(ρ) = E_min + ρᵖ(E₀ - E_min)
//!   ∂E/∂ρ = p·ρᵖ⁻¹·(E₀ - E_min)
//!   dJ/dρₑ = -p·ρₑᵖ⁻¹·(E₀ - E_min)·uₑᵀKe₀uₑ
//!
//! For general objectives (non-self-adjoint):
//!   Kᵀλ = -∂J/∂u (adjoint equation)
//!   dJ/dρₑ = ∂J/∂ρₑ + λᵀ (∂R/∂ρₑ) (total derivative)
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::forward::{Mesh2D, element_compliance};

/// Compute compliance sensitivities for SIMP topology optimization.
/// dC/dρₑ = -p · ρₑ^(p-1) · (E₀ - E_min) · uₑᵀKe₀uₑ
///
/// Returns sensitivity per element.
pub fn compliance_sensitivity(
    mesh: &Mesh2D,
    ke0: &[Q16; 64],
    u: &[Q16],
    densities: &[Q16],
    penal: u32,
    e_min: Q16,
) -> Vec<Q16> {
    let ne = mesh.num_elements();
    let mut dc = vec![Q16::ZERO; ne];
    let e_range = Q16::ONE - e_min;
    let p_q = Q16::from_int(penal as i32);

    for ey in 0..mesh.ny {
        for ex in 0..mesh.nx {
            let eid = ey * mesh.nx + ex;
            let dofs = mesh.element_dofs(ex, ey);
            let ce = element_compliance(ke0, u, &dofs);

            // dE/dρ = p · ρ^(p-1) · (E₀ - E_min)
            let rho_pm1 = if penal > 1 {
                densities[eid].powi(penal - 1)
            } else {
                Q16::ONE
            };

            dc[eid] = -(p_q * rho_pm1 * e_range * ce);
        }
    }

    dc
}

/// Compute volume sensitivities.
/// dV/dρₑ = Vₑ (element volume / total volume)
pub fn volume_sensitivity(mesh: &Mesh2D) -> Vec<Q16> {
    let ne = mesh.num_elements();
    let ve = Q16::ONE.div(Q16::from_int(ne as i32));
    vec![ve; ne]
}

/// Adjoint state for general (non-self-adjoint) objectives.
/// Solves Kᵀλ = -∂J/∂u.
/// For compliance: λ = -u (self-adjoint), so this is a no-op.
/// For other objectives, pass dj_du and solve.
pub struct AdjointState {
    pub lambda: Vec<Q16>,
    pub is_self_adjoint: bool,
}

impl AdjointState {
    /// Self-adjoint case: λ = -u.
    pub fn self_adjoint(u: &[Q16]) -> Self {
        AdjointState {
            lambda: u.iter().map(|&v| -v).collect(),
            is_self_adjoint: true,
        }
    }

    /// General sensitivity: dJ/dρₑ = ∂J/∂ρₑ|_explicit + λᵀ(∂R/∂ρₑ)
    /// R = Ku - F = 0 is the residual.
    /// ∂R/∂ρₑ = (∂K/∂ρₑ)u
    pub fn total_sensitivity(
        &self,
        mesh: &Mesh2D,
        ke0: &[Q16; 64],
        u: &[Q16],
        densities: &[Q16],
        penal: u32,
        e_min: Q16,
        explicit_dj_drho: &[Q16],
    ) -> Vec<Q16> {
        let ne = mesh.num_elements();
        let mut dj = vec![Q16::ZERO; ne];
        let e_range = Q16::ONE - e_min;

        for ey in 0..mesh.ny {
            for ex in 0..mesh.nx {
                let eid = ey * mesh.nx + ex;
                let dofs = mesh.element_dofs(ex, ey);

                // ∂E/∂ρ
                let rho_pm1 = if penal > 1 {
                    densities[eid].powi(penal - 1)
                } else {
                    Q16::ONE
                };
                let de_drho = Q16::from_int(penal as i32) * rho_pm1 * e_range;

                // λᵀ (∂K/∂ρ) u = dE/dρ · λₑᵀ Ke₀ uₑ
                let mut lku = Q16::ZERO;
                for i in 0..8 {
                    for j in 0..8 {
                        lku = lku + self.lambda[dofs[i]] * ke0[i*8+j] * u[dofs[j]];
                    }
                }

                dj[eid] = explicit_dj_drho[eid] + de_drho * lku;
            }
        }

        dj
    }
}
