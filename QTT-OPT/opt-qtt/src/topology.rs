//! Topology Optimization — SIMP with Optimality Criteria
//!
//! Minimum compliance topology optimization:
//!   min  C(ρ) = Fᵀu(ρ)
//!   s.t. K(ρ)u = F
//!        V(ρ) = Σρₑvₑ ≤ V*
//!        0 < ρ_min ≤ ρₑ ≤ 1
//!
//! SIMP interpolation: E(ρ) = E_min + ρᵖ(E₀ - E_min)
//! Optimality Criteria update: ρₑ ← ρₑ · Bₑ^η  where  Bₑ = -(dC/dρₑ)/(Λ·dV/dρₑ)
//!
//! Based on: Sigmund (2001), "A 99 line topology optimization code"
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::forward::*;
use crate::adjoint::*;
use crate::filter::*;

/// Configuration for topology optimization.
#[derive(Clone, Debug)]
pub struct TopOptConfig {
    pub nx: usize,
    pub ny: usize,
    pub volume_fraction: f64,
    pub penal: u32,
    pub rmin: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub e_min: f64,
    pub move_limit: f64,
    pub eta: f64,
}

impl Default for TopOptConfig {
    fn default() -> Self {
        TopOptConfig {
            nx: 6, ny: 4,
            volume_fraction: 0.5,
            penal: 3,
            rmin: 1.5,
            max_iter: 50,
            tol: 0.01,
            e_min: 0.001,
            move_limit: 0.2,
            eta: 0.5,
        }
    }
}

/// Result of topology optimization.
#[derive(Clone, Debug)]
pub struct TopOptResult {
    pub densities: Vec<Q16>,
    pub compliance: Vec<Q16>,
    pub iterations: usize,
    pub converged: bool,
    pub final_volume: Q16,
    pub final_compliance: Q16,
}

/// Run SIMP topology optimization loop.
pub fn optimize(config: &TopOptConfig) -> TopOptResult {
    let mesh = Mesh2D::new(config.nx, config.ny, config.nx as f64, config.ny as f64);
    let ne = mesh.num_elements();
    let ndof = mesh.num_dofs();

    let nu = Q16::from_f64(0.3);
    let ke0 = unit_element_stiffness(mesh.dx, mesh.dy, nu);
    let e_min = Q16::from_f64(config.e_min);
    let penal = config.penal;
    let vf = Q16::from_f64(config.volume_fraction);
    let move_lim = Q16::from_f64(config.move_limit);
    let eta = config.eta;

    // Initialize uniform density
    let mut densities = vec![vf; ne];
    let mut compliance_history = Vec::new();

    // Build filter
    let filter = SensitivityFilter::new(&mesh, config.rmin);

    // Boundary conditions: cantilever — left edge fixed, point load bottom-right
    let left_nodes = mesh.boundary_nodes("left");
    let mut fixed_dofs = Vec::new();
    for &nid in &left_nodes {
        fixed_dofs.push(nid * 2);
        fixed_dofs.push(nid * 2 + 1);
    }

    // Point load at bottom-right corner, downward
    let mut f = vec![Q16::ZERO; ndof];
    let load_node = mesh.node_id(config.nx, 0);
    f[load_node * 2 + 1] = Q16::NEG_ONE; // unit downward force

    // Assembly buffer
    let mut k_global = SparseMat::new(ndof);

    let mut converged = false;
    let mut iter_count = 0;

    for iter in 0..config.max_iter {
        // 1. Assemble K(ρ)
        assemble_stiffness(&mesh, &ke0, &densities, penal, e_min, &mut k_global);

        // Apply BCs
        let mut f_bc = f.clone();
        apply_bcs(&mut k_global, &mut f_bc, &fixed_dofs);

        // 2. Solve Ku = F
        let (u, _cg_iters) = solve_cg(&k_global, &f_bc, 500, 100);

        // 3. Compute compliance
        let compliance = dot_q16(&f, &u);
        compliance_history.push(compliance);

        // 4. Compute adjoint sensitivities
        let dc = compliance_sensitivity(&mesh, &ke0, &u, &densities, penal, e_min);

        // 5. Filter
        let dc_filt = filter.apply(&densities, &dc);

        // 6. OC update
        let dv = volume_sensitivity(&mesh);
        densities = oc_update(&densities, &dc_filt, &dv, vf, move_lim, eta);

        // 7. Convergence check
        iter_count = iter + 1;
        if iter > 0 {
            let prev = compliance_history[iter - 1].to_f64();
            let curr = compliance.to_f64();
            let change = if prev.abs() > 1e-10 { ((curr - prev) / prev).abs() } else { 1.0 };
            if change < config.tol {
                converged = true;
                break;
            }
        }
    }

    let final_vol = volume_fraction(&densities);
    let final_c = *compliance_history.last().unwrap_or(&Q16::ZERO);

    TopOptResult {
        densities,
        compliance: compliance_history,
        iterations: iter_count,
        converged,
        final_volume: final_vol,
        final_compliance: final_c,
    }
}

/// Optimality Criteria update.
/// Bisection on Lagrange multiplier Λ to satisfy volume constraint.
fn oc_update(
    rho: &[Q16],
    dc: &[Q16],
    dv: &[Q16],
    vf: Q16,
    move_lim: Q16,
    eta: f64,
) -> Vec<Q16> {
    let ne = rho.len();
    let rho_min = Q16::from_f64(0.001);

    // Bisection on Lagrange multiplier
    let mut lam_lo = 0.0f64;
    let mut lam_hi = 1e6;

    let mut rho_new = vec![Q16::ZERO; ne];

    for _ in 0..50 {
        let lam_mid = 0.5 * (lam_lo + lam_hi);

        for e in 0..ne {
            // Bₑ = -dc[e] / (Λ · dv[e])
            let dc_f = dc[e].to_f64();
            let dv_f = dv[e].to_f64().max(1e-12);
            let b_e = (-dc_f / (lam_mid * dv_f)).max(0.0);

            // ρ_new = ρ · B^η, clamped by move limit and [rho_min, 1]
            let rho_f = rho[e].to_f64();
            let rho_cand = rho_f * b_e.powf(eta);

            let lo = (rho_f - move_lim.to_f64()).max(rho_min.to_f64());
            let hi = (rho_f + move_lim.to_f64()).min(1.0);
            let rho_clamped = rho_cand.max(lo).min(hi);

            rho_new[e] = Q16::from_f64(rho_clamped);
        }

        // Check volume constraint
        let vol: f64 = rho_new.iter().map(|r| r.to_f64()).sum::<f64>() / ne as f64;
        if vol > vf.to_f64() {
            lam_lo = lam_mid;
        } else {
            lam_hi = lam_mid;
        }

        if lam_hi - lam_lo < 1e-6 { break; }
    }

    rho_new
}

/// Compute volume fraction.
pub fn volume_fraction(densities: &[Q16]) -> Q16 {
    let ne = densities.len();
    if ne == 0 { return Q16::ZERO; }
    let sum: i64 = densities.iter().map(|r| r.0 as i64).sum();
    Q16((sum / ne as i64) as i32)
}

/// Q16 dot product.
fn dot_q16(a: &[Q16], b: &[Q16]) -> Q16 {
    let mut s: i64 = 0;
    for i in 0..a.len() { s += (a[i].0 as i64 * b[i].0 as i64) >> 16; }
    Q16(s as i32)
}
