//! Yee Lattice FDTD Solver in QTT Format
//!
//! Implements the Finite-Difference Time-Domain method for Maxwell's equations
//! on a staggered Yee grid with all field components represented as MPS
//! in Quantized Tensor Train (QTT) format.
//!
//! Governing equations:
//!   ∂B/∂t = −∇ × E
//!   ∂E/∂t = (1/ε)(∇ × B/μ − J − σE)
//!
//! Time integration: explicit leapfrog (E at half-steps, H at full steps).
//!
//! © 2026 Brad McAllister. All rights reserved. PROPRIETARY.

use crate::q16::Q16;
use crate::mps::Mps;
use crate::material::{MaterialMap, Constants};
use crate::pml::PmlParams;

/// Boundary condition type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    /// Periodic boundaries (wrap-around).
    Periodic,
    /// Perfect Electric Conductor (E_tangential = 0).
    PEC,
    /// Perfectly Matched Layer (absorbing).
    PML,
}

/// Source type for field excitation.
#[derive(Clone, Debug)]
pub enum Source {
    /// Gaussian pulse: A * exp(-((t - t0) / τ)^2)
    GaussianPulse {
        amplitude: Q16,
        t0: Q16,
        tau: Q16,
        /// Grid location (i, j, k).
        location: (usize, usize, usize),
        /// Which E-field component (0=x, 1=y, 2=z).
        component: usize,
    },
    /// Sinusoidal source: A * sin(2π*f*t)
    Sinusoidal {
        amplitude: Q16,
        /// Frequency encoded as ω*Δt in Q16.16.
        omega_dt: Q16,
        location: (usize, usize, usize),
        component: usize,
    },
    /// Plane wave (soft source across a plane).
    PlaneWave {
        amplitude: Q16,
        omega_dt: Q16,
        /// Normal direction (0=x, 1=y, 2=z).
        direction: usize,
        /// Plane index along normal direction.
        plane_index: usize,
        component: usize,
    },
}

/// Configuration for the FDTD simulation.
#[derive(Clone, Debug)]
pub struct FdtdConfig {
    /// Grid dimensions (must be powers of 2 for QTT).
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// Grid spacing.
    pub dx: Q16,
    pub dy: Q16,
    pub dz: Q16,
    /// Time step (must satisfy CFL: dt <= dx / (c * sqrt(3))).
    pub dt: Q16,
    /// Number of QTT sites per dimension.
    pub grid_bits: usize,
    /// Maximum bond dimension.
    pub chi_max: usize,
    /// Boundary condition.
    pub boundary: BoundaryCondition,
    /// PML parameters (if PML boundary).
    pub pml: Option<PmlParams>,
    /// Sources.
    pub sources: Vec<Source>,
    /// Material map.
    pub materials: MaterialMap,
}

impl FdtdConfig {
    /// Create a default vacuum configuration.
    pub fn vacuum_cube(grid_bits: usize, chi_max: usize) -> Self {
        let n = 1 << grid_bits;
        let dx = Q16::from_f64(1.0 / n as f64);
        // CFL condition: dt <= dx / (c * sqrt(3))
        // With c=1, sqrt(3) ≈ 1.732, use Courant number 0.5
        let dt = Q16::from_f64(0.5 / (n as f64 * 1.732));

        FdtdConfig {
            nx: n, ny: n, nz: n,
            dx, dy: dx, dz: dx,
            dt,
            grid_bits,
            chi_max,
            boundary: BoundaryCondition::Periodic,
            pml: None,
            sources: Vec::new(),
            materials: MaterialMap::vacuum(n, n, n),
        }
    }

    /// Verify CFL stability condition.
    pub fn check_cfl(&self) -> bool {
        // dt <= 1/c * 1/sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)
        let inv_dx2 = Q16::ONE.div(self.dx * self.dx);
        let inv_dy2 = Q16::ONE.div(self.dy * self.dy);
        let inv_dz2 = Q16::ONE.div(self.dz * self.dz);
        let sum = inv_dx2 + inv_dy2 + inv_dz2;
        let dt_max = Q16::ONE.div(sum.sqrt() * Constants::C);
        self.dt.raw() <= dt_max.raw()
    }

    /// Add a Gaussian pulse source.
    pub fn add_gaussian_source(&mut self, loc: (usize, usize, usize), component: usize,
                                amplitude: f64, t0: f64, tau: f64) {
        self.sources.push(Source::GaussianPulse {
            amplitude: Q16::from_f64(amplitude),
            t0: Q16::from_f64(t0),
            tau: Q16::from_f64(tau),
            location: loc,
            component,
        });
    }

    /// Add a sinusoidal source.
    pub fn add_sinusoidal_source(&mut self, loc: (usize, usize, usize), component: usize,
                                  amplitude: f64, frequency: f64) {
        self.sources.push(Source::Sinusoidal {
            amplitude: Q16::from_f64(amplitude),
            omega_dt: Q16::from_f64(2.0 * std::f64::consts::PI * frequency * self.dt.to_f64()),
            location: loc,
            component,
        });
    }
}

/// Electromagnetic field state on a Yee lattice.
/// All six field components stored as flat Q16.16 arrays.
/// QTT compression applied via snapshot/compress methods.
#[derive(Clone, Debug)]
pub struct YeeFields {
    /// E-field components: Ex, Ey, Ez.
    pub ex: Vec<Q16>,
    pub ey: Vec<Q16>,
    pub ez: Vec<Q16>,
    /// H-field components: Hx, Hy, Hz.
    pub hx: Vec<Q16>,
    pub hy: Vec<Q16>,
    pub hz: Vec<Q16>,
    /// Grid dimensions.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl YeeFields {
    /// Initialize zero fields.
    pub fn zeros(nx: usize, ny: usize, nz: usize) -> Self {
        let n = nx * ny * nz;
        YeeFields {
            ex: vec![Q16::ZERO; n],
            ey: vec![Q16::ZERO; n],
            ez: vec![Q16::ZERO; n],
            hx: vec![Q16::ZERO; n],
            hy: vec![Q16::ZERO; n],
            hz: vec![Q16::ZERO; n],
            nx, ny, nz,
        }
    }

    /// Linear index from (i, j, k).
    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.ny * self.nz + j * self.nz + k
    }

    /// Get field component by index at (i,j,k). Component: 0-2 = Ex,Ey,Ez, 3-5 = Hx,Hy,Hz.
    pub fn get_component(&self, comp: usize, i: usize, j: usize, k: usize) -> Q16 {
        let idx = self.idx(i, j, k);
        match comp {
            0 => self.ex[idx],
            1 => self.ey[idx],
            2 => self.ez[idx],
            3 => self.hx[idx],
            4 => self.hy[idx],
            5 => self.hz[idx],
            _ => Q16::ZERO,
        }
    }

    /// Set field component.
    pub fn set_component(&mut self, comp: usize, i: usize, j: usize, k: usize, val: Q16) {
        let idx = self.idx(i, j, k);
        match comp {
            0 => self.ex[idx] = val,
            1 => self.ey[idx] = val,
            2 => self.ez[idx] = val,
            3 => self.hx[idx] = val,
            4 => self.hy[idx] = val,
            5 => self.hz[idx] = val,
            _ => {}
        }
    }

    /// Total electromagnetic energy: u = 0.5 * (ε|E|² + μ|H|²).
    pub fn total_energy(&self, materials: &MaterialMap) -> Q16 {
        let mut energy = Q16::ZERO;
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let idx = self.idx(i, j, k);
                    let mat = materials.get(i, j, k);

                    let e_sq = self.ex[idx] * self.ex[idx]
                        + self.ey[idx] * self.ey[idx]
                        + self.ez[idx] * self.ez[idx];

                    let h_sq = self.hx[idx] * self.hx[idx]
                        + self.hy[idx] * self.hy[idx]
                        + self.hz[idx] * self.hz[idx];

                    energy = energy + Q16::HALF * (mat.epsilon_r * e_sq + mat.mu_r * h_sq);
                }
            }
        }
        energy
    }

    /// Poynting vector magnitude integrated over domain.
    /// S = E × H, returns |∫S dV|.
    pub fn poynting_flux(&self) -> Q16 {
        let mut sx = Q16::ZERO;
        let mut sy = Q16::ZERO;
        let mut sz = Q16::ZERO;

        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    let idx = self.idx(i, j, k);
                    sx = sx + self.ey[idx] * self.hz[idx] - self.ez[idx] * self.hy[idx];
                    sy = sy + self.ez[idx] * self.hx[idx] - self.ex[idx] * self.hz[idx];
                    sz = sz + self.ex[idx] * self.hy[idx] - self.ey[idx] * self.hx[idx];
                }
            }
        }

        (sx * sx + sy * sy + sz * sz).sqrt()
    }
}

/// FDTD simulation engine.
pub struct FdtdSolver {
    pub config: FdtdConfig,
    pub fields: YeeFields,
    pub timestep: usize,
    pub time: Q16,
    /// Update coefficients cached per grid point.
    ca: Vec<Q16>,
    cb: Vec<Q16>,
    da: Vec<Q16>,
    db: Vec<Q16>,
}

impl FdtdSolver {
    /// Initialize solver from configuration.
    pub fn new(config: FdtdConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;
        let nz = config.nz;
        let n = nx * ny * nz;

        // Precompute update coefficients
        let mut ca = vec![Q16::ZERO; n];
        let mut cb = vec![Q16::ZERO; n];
        let mut da = vec![Q16::ZERO; n];
        let mut db = vec![Q16::ZERO; n];

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = i * ny * nz + j * nz + k;
                    let mat = config.materials.get(i, j, k);
                    let (ca_v, cb_v) = mat.e_update_coefficients(config.dt, config.dx);
                    let (da_v, db_v) = mat.h_update_coefficients(config.dt, config.dx);
                    ca[idx] = ca_v;
                    cb[idx] = cb_v;
                    da[idx] = da_v;
                    db[idx] = db_v;
                }
            }
        }

        FdtdSolver {
            fields: YeeFields::zeros(nx, ny, nz),
            config,
            timestep: 0,
            time: Q16::ZERO,
            ca, cb, da, db,
        }
    }

    /// Wrap index for periodic boundary.
    #[inline]
    fn wrap(&self, idx: i64, max: usize) -> usize {
        ((idx % max as i64 + max as i64) % max as i64) as usize
    }

    /// Perform one complete FDTD timestep (leapfrog).
    pub fn step(&mut self) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;
        let periodic = self.config.boundary == BoundaryCondition::Periodic;

        // ── Update H-field: H^{n+1/2} = da*H^{n-1/2} - db*(∇ × E^n) ──
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = i * ny * nz + j * nz + k;

                    // Neighbor indices for curl computation
                    let ip = if periodic { self.wrap(i as i64 + 1, nx) } else { (i + 1).min(nx - 1) };
                    let jp = if periodic { self.wrap(j as i64 + 1, ny) } else { (j + 1).min(ny - 1) };
                    let kp = if periodic { self.wrap(k as i64 + 1, nz) } else { (k + 1).min(nz - 1) };

                    let idx_ip = ip * ny * nz + j * nz + k;
                    let idx_jp = i * ny * nz + jp * nz + k;
                    let idx_kp = i * ny * nz + j * nz + kp;

                    // Curl uses raw differences — db already includes Δt/(μ·Δx)
                    let curl_x = (self.fields.ez[idx_jp] - self.fields.ez[idx])
                        - (self.fields.ey[idx_kp] - self.fields.ey[idx]);

                    let curl_y = (self.fields.ex[idx_kp] - self.fields.ex[idx])
                        - (self.fields.ez[idx_ip] - self.fields.ez[idx]);

                    let curl_z = (self.fields.ey[idx_ip] - self.fields.ey[idx])
                        - (self.fields.ex[idx_jp] - self.fields.ex[idx]);

                    self.fields.hx[idx] = self.da[idx] * self.fields.hx[idx] - self.db[idx] * curl_x;
                    self.fields.hy[idx] = self.da[idx] * self.fields.hy[idx] - self.db[idx] * curl_y;
                    self.fields.hz[idx] = self.da[idx] * self.fields.hz[idx] - self.db[idx] * curl_z;
                }
            }
        }

        // ── Update E-field: E^{n+1} = ca*E^n + cb*(∇ × H^{n+1/2}) ────
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = i * ny * nz + j * nz + k;

                    let im = if periodic { self.wrap(i as i64 - 1, nx) } else { if i > 0 { i - 1 } else { 0 } };
                    let jm = if periodic { self.wrap(j as i64 - 1, ny) } else { if j > 0 { j - 1 } else { 0 } };
                    let km = if periodic { self.wrap(k as i64 - 1, nz) } else { if k > 0 { k - 1 } else { 0 } };

                    let idx_im = im * ny * nz + j * nz + k;
                    let idx_jm = i * ny * nz + jm * nz + k;
                    let idx_km = i * ny * nz + j * nz + km;

                    // Curl uses raw differences — cb already includes Δt/(ε·Δx)
                    let curl_x = (self.fields.hz[idx] - self.fields.hz[idx_jm])
                        - (self.fields.hy[idx] - self.fields.hy[idx_km]);

                    let curl_y = (self.fields.hx[idx] - self.fields.hx[idx_km])
                        - (self.fields.hz[idx] - self.fields.hz[idx_im]);

                    let curl_z = (self.fields.hy[idx] - self.fields.hy[idx_im])
                        - (self.fields.hx[idx] - self.fields.hx[idx_jm]);

                    self.fields.ex[idx] = self.ca[idx] * self.fields.ex[idx] + self.cb[idx] * curl_x;
                    self.fields.ey[idx] = self.ca[idx] * self.fields.ey[idx] + self.cb[idx] * curl_y;
                    self.fields.ez[idx] = self.ca[idx] * self.fields.ez[idx] + self.cb[idx] * curl_z;
                }
            }
        }

        // ── Apply sources ──────────────────────────────────────────────
        self.apply_sources();

        // ── Apply PEC boundaries if needed ──────────────────────────────
        if self.config.boundary == BoundaryCondition::PEC {
            self.apply_pec();
        }

        self.timestep += 1;
        self.time = self.time + self.config.dt;
    }

    /// Apply field sources at current timestep.
    fn apply_sources(&mut self) {
        for source in &self.config.sources {
            match source {
                Source::GaussianPulse { amplitude, t0, tau, location, component } => {
                    let t_diff = self.time - *t0;
                    // Gaussian: exp(-(t-t0)²/τ²) approximated in Q16.16
                    let arg = t_diff.div(*tau);
                    let arg_sq = arg * arg;
                    // Approximate exp(-x²) via polynomial: 1 - x² + x⁴/2 - x⁶/6
                    let exp_approx = Q16::ONE - arg_sq
                        + Q16::HALF * arg_sq * arg_sq
                        - Q16::from_ratio(1, 6) * arg_sq * arg_sq * arg_sq;
                    let exp_val = exp_approx.max(Q16::ZERO);
                    let val = *amplitude * exp_val;

                    let (i, j, k) = *location;
                    let idx = i * self.config.ny * self.config.nz + j * self.config.nz + k;
                    match component {
                        0 => self.fields.ex[idx] = self.fields.ex[idx] + val,
                        1 => self.fields.ey[idx] = self.fields.ey[idx] + val,
                        2 => self.fields.ez[idx] = self.fields.ez[idx] + val,
                        _ => {}
                    }
                }
                Source::Sinusoidal { amplitude, omega_dt, location, component } => {
                    // sin(ω*n*Δt) approximated via Taylor: x - x³/6 + x⁵/120
                    let phase = *omega_dt * Q16::from_int(self.timestep as i32);
                    let x = phase;
                    let x3 = x * x * x;
                    let x5 = x3 * x * x;
                    let sin_approx = x - Q16::from_ratio(1, 6) * x3
                        + Q16::from_ratio(1, 120) * x5;
                    let val = *amplitude * sin_approx;

                    let (i, j, k) = *location;
                    let idx = i * self.config.ny * self.config.nz + j * self.config.nz + k;
                    match component {
                        0 => self.fields.ex[idx] = self.fields.ex[idx] + val,
                        1 => self.fields.ey[idx] = self.fields.ey[idx] + val,
                        2 => self.fields.ez[idx] = self.fields.ez[idx] + val,
                        _ => {}
                    }
                }
                Source::PlaneWave { amplitude, omega_dt, direction, plane_index, component } => {
                    let phase = *omega_dt * Q16::from_int(self.timestep as i32);
                    let x = phase;
                    let sin_approx = x - Q16::from_ratio(1, 6) * x * x * x;
                    let val = *amplitude * sin_approx;

                    match direction {
                        0 => {
                            for j in 0..self.config.ny {
                                for k in 0..self.config.nz {
                                    self.fields.set_component(*component, *plane_index, j, k,
                                        self.fields.get_component(*component, *plane_index, j, k) + val);
                                }
                            }
                        }
                        1 => {
                            for i in 0..self.config.nx {
                                for k in 0..self.config.nz {
                                    self.fields.set_component(*component, i, *plane_index, k,
                                        self.fields.get_component(*component, i, *plane_index, k) + val);
                                }
                            }
                        }
                        2 => {
                            for i in 0..self.config.nx {
                                for j in 0..self.config.ny {
                                    self.fields.set_component(*component, i, j, *plane_index,
                                        self.fields.get_component(*component, i, j, *plane_index) + val);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Apply PEC boundary: zero tangential E-field on all faces.
    fn apply_pec(&mut self) {
        let nx = self.config.nx;
        let ny = self.config.ny;
        let nz = self.config.nz;

        for j in 0..ny {
            for k in 0..nz {
                let idx0 = j * nz + k;
                let idx1 = (nx - 1) * ny * nz + j * nz + k;
                self.fields.ey[idx0] = Q16::ZERO;
                self.fields.ez[idx0] = Q16::ZERO;
                self.fields.ey[idx1] = Q16::ZERO;
                self.fields.ez[idx1] = Q16::ZERO;
            }
        }
        for i in 0..nx {
            for k in 0..nz {
                let idx0 = i * ny * nz + k;
                let idx1 = i * ny * nz + (ny - 1) * nz + k;
                self.fields.ex[idx0] = Q16::ZERO;
                self.fields.ez[idx0] = Q16::ZERO;
                self.fields.ex[idx1] = Q16::ZERO;
                self.fields.ez[idx1] = Q16::ZERO;
            }
        }
        for i in 0..nx {
            for j in 0..ny {
                let idx0 = i * ny * nz + j * nz;
                let idx1 = i * ny * nz + j * nz + nz - 1;
                self.fields.ex[idx0] = Q16::ZERO;
                self.fields.ey[idx0] = Q16::ZERO;
                self.fields.ex[idx1] = Q16::ZERO;
                self.fields.ey[idx1] = Q16::ZERO;
            }
        }
    }

    /// Run simulation for `num_steps` timesteps.
    /// Returns energy at each timestep for conservation tracking.
    pub fn run(&mut self, num_steps: usize) -> Vec<Q16> {
        let mut energies = Vec::with_capacity(num_steps + 1);
        energies.push(self.fields.total_energy(&self.config.materials));

        for _ in 0..num_steps {
            self.step();
            energies.push(self.fields.total_energy(&self.config.materials));
        }

        energies
    }

    /// Snapshot current fields into QTT-compressed MPS format.
    /// Returns (Ex, Ey, Ez, Hx, Hy, Hz) as MPS.
    pub fn compress(&self, chi_max: usize) -> FieldSnapshot {
        let n = self.config.grid_bits;
        let num_sites = n * 3; // 3D QTT: x,y,z interleaved

        FieldSnapshot {
            ex: Mps::from_values(&self.fields.ex, num_sites, 2, chi_max),
            ey: Mps::from_values(&self.fields.ey, num_sites, 2, chi_max),
            ez: Mps::from_values(&self.fields.ez, num_sites, 2, chi_max),
            hx: Mps::from_values(&self.fields.hx, num_sites, 2, chi_max),
            hy: Mps::from_values(&self.fields.hy, num_sites, 2, chi_max),
            hz: Mps::from_values(&self.fields.hz, num_sites, 2, chi_max),
            timestep: self.timestep,
            time: self.time,
        }
    }

    /// Get current simulation time.
    pub fn current_time(&self) -> Q16 {
        self.time
    }
}

/// QTT-compressed snapshot of all six field components.
#[derive(Clone, Debug)]
pub struct FieldSnapshot {
    pub ex: Mps,
    pub ey: Mps,
    pub ez: Mps,
    pub hx: Mps,
    pub hy: Mps,
    pub hz: Mps,
    pub timestep: usize,
    pub time: Q16,
}

impl FieldSnapshot {
    /// Total tensor elements across all six components.
    pub fn total_elements(&self) -> usize {
        self.ex.total_elements() + self.ey.total_elements() + self.ez.total_elements()
            + self.hx.total_elements() + self.hy.total_elements() + self.hz.total_elements()
    }

    /// Compression ratio vs full grid.
    pub fn compression_ratio(&self, full_grid_size: usize) -> f64 {
        (6 * full_grid_size) as f64 / self.total_elements().max(1) as f64
    }

    /// Maximum bond dimension across all components.
    pub fn max_bond_dim(&self) -> usize {
        let all = [&self.ex, &self.ey, &self.ez, &self.hx, &self.hy, &self.hz];
        all.iter()
            .flat_map(|m| m.bond_dimensions())
            .max()
            .unwrap_or(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_config() {
        let config = FdtdConfig::vacuum_cube(3, 4);
        assert_eq!(config.nx, 8);
        assert!(config.check_cfl());
    }

    #[test]
    fn test_zero_field_energy() {
        let config = FdtdConfig::vacuum_cube(3, 4);
        let solver = FdtdSolver::new(config);
        let energy = solver.fields.total_energy(&solver.config.materials);
        assert_eq!(energy.raw(), 0);
    }

    #[test]
    fn test_step_no_source() {
        let config = FdtdConfig::vacuum_cube(3, 4);
        let mut solver = FdtdSolver::new(config);
        solver.step();
        // Zero fields remain zero
        let energy = solver.fields.total_energy(&solver.config.materials);
        assert_eq!(energy.raw(), 0);
    }

    #[test]
    fn test_gaussian_source() {
        let mut config = FdtdConfig::vacuum_cube(3, 4);
        config.add_gaussian_source((4, 4, 4), 2, 1.0, 0.0, 0.1);
        let mut solver = FdtdSolver::new(config);
        solver.step();
        // Should have nonzero energy after source injection
        let energy = solver.fields.total_energy(&solver.config.materials);
        assert!(energy.raw() > 0);
    }

    #[test]
    fn test_pec_boundary() {
        let mut config = FdtdConfig::vacuum_cube(3, 4);
        config.boundary = BoundaryCondition::PEC;
        config.add_gaussian_source((4, 4, 4), 2, 1.0, 0.0, 0.1);
        let mut solver = FdtdSolver::new(config);
        for _ in 0..10 {
            solver.step();
        }
        // PEC: tangential E should be zero at boundaries
        let n = solver.config.nx;
        for j in 0..n {
            for k in 0..n {
                assert_eq!(solver.fields.ey[j * n + k].raw(), 0);
            }
        }
    }

    #[test]
    fn test_energy_conservation_vacuum() {
        let mut config = FdtdConfig::vacuum_cube(3, 4);
        config.boundary = BoundaryCondition::Periodic;
        // Inject initial energy manually
        let mut solver = FdtdSolver::new(config);
        let center = 4 * 8 * 8 + 4 * 8 + 4;
        solver.fields.ez[center] = Q16::from_f64(0.5);

        let energies = solver.run(20);
        let e0 = energies[0];

        // In vacuum with periodic BC and no source, energy should be approximately conserved
        for e in &energies[1..] {
            let diff = (*e - e0).abs();
            // Allow some fixed-point drift
            assert!(diff.to_f64() < 0.1,
                "Energy conservation violated: e0={}, e={}, diff={}",
                e0.to_f64(), e.to_f64(), diff.to_f64());
        }
    }

    #[test]
    fn test_cfl_violation_detected() {
        let mut config = FdtdConfig::vacuum_cube(3, 4);
        config.dt = Q16::from_f64(1.0); // Way too large
        assert!(!config.check_cfl());
    }
}
