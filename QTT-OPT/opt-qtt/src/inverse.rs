//! Inverse Problem Framework
//!
//! Given observed data u_obs, find parameters θ such that:
//!   min J(θ) = ½ ‖u(θ) - u_obs‖²  (least squares)
//!   s.t. R(u, θ) = 0               (PDE constraint)
//!
//! Adjoint-based gradient:
//!   dJ/dθ = ∂J/∂θ + λᵀ ∂R/∂θ
//!   where λ solves: (∂R/∂u)ᵀ λ = -∂J/∂u
//!
//! Algorithms:
//!   - Gradient descent with backtracking line search
//!   - L2 Tikhonov regularization
//!   - Projected gradient for box constraints
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;

/// Inverse problem configuration.
#[derive(Clone, Debug)]
pub struct InverseConfig {
    pub max_iter: usize,
    pub learning_rate: f64,
    pub regularization: f64,  // Tikhonov λ
    pub tol: f64,
    pub param_min: f64,
    pub param_max: f64,
    pub line_search: bool,
}

impl Default for InverseConfig {
    fn default() -> Self {
        InverseConfig {
            max_iter: 100,
            learning_rate: 0.01,
            regularization: 0.0,
            tol: 1e-6,
            param_min: 0.001,
            param_max: 10.0,
            line_search: true,
        }
    }
}

/// Result of inverse solve.
#[derive(Clone, Debug)]
pub struct InverseResult {
    pub parameters: Vec<Q16>,
    pub objective_history: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub final_misfit: f64,
}

/// Generic forward model trait.
/// User implements this to define the forward problem.
pub trait ForwardModel {
    /// Solve forward problem for given parameters, return solution.
    fn solve(&self, params: &[Q16]) -> Vec<Q16>;

    /// Compute misfit: ½‖u - u_obs‖².
    fn misfit(&self, u: &[Q16], u_obs: &[Q16]) -> f64 {
        let mut s = 0.0;
        for i in 0..u.len().min(u_obs.len()) {
            let diff = u[i].to_f64() - u_obs[i].to_f64();
            s += diff * diff;
        }
        0.5 * s
    }

    /// Compute gradient dJ/dθ via finite differences (fallback).
    fn gradient_fd(&self, params: &[Q16], u_obs: &[Q16], eps: f64) -> Vec<f64> {
        let u0 = self.solve(params);
        let j0 = self.misfit(&u0, u_obs);

        let mut grad = vec![0.0; params.len()];
        for i in 0..params.len() {
            let mut p_pert = params.to_vec();
            p_pert[i] = Q16::from_f64(params[i].to_f64() + eps);
            let u_pert = self.solve(&p_pert);
            let j_pert = self.misfit(&u_pert, u_obs);
            grad[i] = (j_pert - j0) / eps;
        }
        grad
    }
}

/// Gradient descent with optional line search and Tikhonov regularization.
pub fn solve_inverse<M: ForwardModel>(
    model: &M,
    initial_params: &[Q16],
    u_obs: &[Q16],
    config: &InverseConfig,
) -> InverseResult {
    let np = initial_params.len();
    let mut params: Vec<f64> = initial_params.iter().map(|p| p.to_f64()).collect();
    let mut obj_history = Vec::new();

    let eps = 1e-4;

    for iter in 0..config.max_iter {
        let params_q: Vec<Q16> = params.iter().map(|&v| Q16::from_f64(v)).collect();
        let u = model.solve(&params_q);
        let misfit = model.misfit(&u, u_obs);

        // Tikhonov regularization: J_total = misfit + λ/2 ‖θ‖²
        let reg = config.regularization * 0.5
            * params.iter().map(|p| p * p).sum::<f64>();
        let total_obj = misfit + reg;
        obj_history.push(total_obj);

        // Convergence check
        if iter > 0 {
            let change = (obj_history[iter] - obj_history[iter - 1]).abs()
                / obj_history[iter - 1].abs().max(1e-12);
            if change < config.tol {
                return InverseResult {
                    parameters: params.iter().map(|&v| Q16::from_f64(v)).collect(),
                    objective_history: obj_history,
                    iterations: iter + 1,
                    converged: true,
                    final_misfit: misfit,
                };
            }
        }

        // Gradient via finite differences
        let grad_misfit = model.gradient_fd(&params_q, u_obs, eps);

        // Add regularization gradient: dR/dθ = λ·θ
        let grad: Vec<f64> = grad_misfit.iter().enumerate().map(|(i, &gm)| {
            gm + config.regularization * params[i]
        }).collect();

        // Step
        let mut lr = config.learning_rate;

        if config.line_search {
            // Armijo backtracking
            let c1 = 1e-4;
            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();

            for _ in 0..20 {
                let trial: Vec<f64> = params.iter().zip(grad.iter())
                    .map(|(p, g)| (p - lr * g).max(config.param_min).min(config.param_max))
                    .collect();
                let trial_q: Vec<Q16> = trial.iter().map(|&v| Q16::from_f64(v)).collect();
                let u_trial = model.solve(&trial_q);
                let misfit_trial = model.misfit(&u_trial, u_obs);
                let reg_trial = config.regularization * 0.5
                    * trial.iter().map(|p| p * p).sum::<f64>();
                let obj_trial = misfit_trial + reg_trial;

                if obj_trial <= total_obj - c1 * lr * grad_norm_sq {
                    break;
                }
                lr *= 0.5;
            }
        }

        // Projected gradient update
        for i in 0..np {
            params[i] = (params[i] - lr * grad[i])
                .max(config.param_min)
                .min(config.param_max);
        }
    }

    let params_q: Vec<Q16> = params.iter().map(|&v| Q16::from_f64(v)).collect();
    let u_final = model.solve(&params_q);
    let final_misfit = model.misfit(&u_final, u_obs);

    InverseResult {
        parameters: params_q,
        objective_history: obj_history,
        iterations: config.max_iter,
        converged: false,
        final_misfit,
    }
}

/// 1D Poisson inverse problem: -d/dx(κ du/dx) = f.
/// Estimate κ from observed u.
pub struct Poisson1DModel {
    pub nx: usize,
    pub source: Vec<Q16>,
    pub bc_left: Q16,
    pub bc_right: Q16,
}

impl Poisson1DModel {
    pub fn new(nx: usize, source_val: f64) -> Self {
        Poisson1DModel {
            nx,
            source: vec![Q16::from_f64(source_val); nx],
            bc_left: Q16::ZERO,
            bc_right: Q16::ZERO,
        }
    }
}

impl ForwardModel for Poisson1DModel {
    /// Solve -d/dx(κ du/dx) = f on [0,1] with Dirichlet BCs.
    /// κ is element-wise (params[e] for each element).
    fn solve(&self, params: &[Q16]) -> Vec<Q16> {
        let n = self.nx;
        let h = 1.0 / n as f64;
        let _h_q = Q16::from_f64(h);

        // Assemble tridiagonal system in f64 for stability
        let nn = n + 1; // nodes
        let mut a_diag = vec![0.0f64; nn];
        let mut a_upper = vec![0.0f64; nn];
        let mut a_lower = vec![0.0f64; nn];
        let mut rhs = vec![0.0f64; nn];

        // Interior: -κ_{e-1}/h² u_{i-1} + (κ_{e-1}+κ_e)/h² u_i - κ_e/h² u_{i+1} = f_i
        for i in 1..n {
            let kl = if i > 0 && (i-1) < params.len() { params[i-1].to_f64() } else { 1.0 };
            let kr = if i < params.len() { params[i].to_f64() } else { 1.0 };
            a_lower[i] = -kl / (h * h);
            a_diag[i] = (kl + kr) / (h * h);
            a_upper[i] = -kr / (h * h);
            rhs[i] = self.source[i.min(n-1)].to_f64();
        }

        // BCs
        a_diag[0] = 1.0; rhs[0] = self.bc_left.to_f64();
        a_diag[n] = 1.0; rhs[n] = self.bc_right.to_f64();

        // Thomas algorithm
        let mut c_prime = vec![0.0; nn];
        let mut d_prime = vec![0.0; nn];
        c_prime[0] = a_upper[0] / a_diag[0];
        d_prime[0] = rhs[0] / a_diag[0];
        for i in 1..nn {
            let m = a_diag[i] - a_lower[i] * c_prime[i-1];
            c_prime[i] = if m.abs() > 1e-15 { a_upper[i] / m } else { 0.0 };
            d_prime[i] = if m.abs() > 1e-15 { (rhs[i] - a_lower[i] * d_prime[i-1]) / m } else { 0.0 };
        }

        let mut u = vec![0.0; nn];
        u[nn-1] = d_prime[nn-1];
        for i in (0..nn-1).rev() {
            u[i] = d_prime[i] - c_prime[i] * u[i+1];
        }

        u.iter().map(|&v| Q16::from_f64(v)).collect()
    }
}
