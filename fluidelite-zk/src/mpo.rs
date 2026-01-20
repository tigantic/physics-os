//! Matrix Product Operator (MPO) implementation
//!
//! An MPO represents a 1D tensor network operator with bond dimension D.
//! In FluidElite, MPOs encode the weight matrices (W_hidden, W_input).
//!
//! Structure:
//! ```text
//!    d_out  d_out  d_out        d_out
//!     │      │      │            │
//!   ┌─┴─┐  ┌─┴─┐  ┌─┴─┐        ┌─┴─┐
//!   │ W │──│ W │──│ W │── ··· ─│ W │
//!   └─┬─┘  └─┬─┘  └─┬─┘        └─┬─┘
//!     │      │      │            │
//!    d_in   d_in   d_in         d_in
//!
//!   1──D──D──D── ··· ──D──1
//! ```
//!
//! Each core W[i] has shape (D_left, d_out, d_in, D_right) where:
//! - D_left, D_right = MPO bond dimensions (1 at boundaries)
//! - d_out, d_in = physical dimensions (typically both 2)

use crate::config::{D, L, PHYS_DIM};
use crate::field::Q16;
use rand::Rng;

/// A single MPO core tensor
#[derive(Clone, Debug)]
pub struct MPOCore {
    /// Core tensor data in row-major order: [D_left, d_out, d_in, D_right]
    pub data: Vec<Q16>,
    /// Left bond dimension
    pub d_left: usize,
    /// Output physical dimension
    pub d_out: usize,
    /// Input physical dimension
    pub d_in: usize,
    /// Right bond dimension
    pub d_right: usize,
}

impl MPOCore {
    /// Create a new MPO core with given dimensions
    pub fn new(d_left: usize, d_out: usize, d_in: usize, d_right: usize) -> Self {
        Self {
            data: vec![Q16::zero(); d_left * d_out * d_in * d_right],
            d_left,
            d_out,
            d_in,
            d_right,
        }
    }

    /// Create from raw data
    pub fn from_data(
        data: Vec<Q16>,
        d_left: usize,
        d_out: usize,
        d_in: usize,
        d_right: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            d_left * d_out * d_in * d_right,
            "Data size mismatch"
        );
        Self {
            data,
            d_left,
            d_out,
            d_in,
            d_right,
        }
    }

    /// Get element at (left, out, in, right)
    #[inline]
    pub fn get(&self, left: usize, out: usize, inp: usize, right: usize) -> Q16 {
        debug_assert!(left < self.d_left);
        debug_assert!(out < self.d_out);
        debug_assert!(inp < self.d_in);
        debug_assert!(right < self.d_right);
        self.data[((left * self.d_out + out) * self.d_in + inp) * self.d_right + right]
    }

    /// Set element at (left, out, in, right)
    #[inline]
    pub fn set(&mut self, left: usize, out: usize, inp: usize, right: usize, val: Q16) {
        debug_assert!(left < self.d_left);
        debug_assert!(out < self.d_out);
        debug_assert!(inp < self.d_in);
        debug_assert!(right < self.d_right);
        self.data[((left * self.d_out + out) * self.d_in + inp) * self.d_right + right] = val;
    }

    /// Total number of elements
    pub fn size(&self) -> usize {
        self.d_left * self.d_out * self.d_in * self.d_right
    }
}

/// Matrix Product Operator
#[derive(Clone, Debug)]
pub struct MPO {
    /// Core tensors for each site
    pub cores: Vec<MPOCore>,
    /// Number of sites
    pub num_sites: usize,
}

impl MPO {
    /// Create an empty MPO with given dimensions
    pub fn new(num_sites: usize, d: usize, d_out: usize, d_in: usize) -> Self {
        let mut cores = Vec::with_capacity(num_sites);

        for i in 0..num_sites {
            let d_left = if i == 0 { 1 } else { d };
            let d_right = if i == num_sites - 1 { 1 } else { d };
            cores.push(MPOCore::new(d_left, d_out, d_in, d_right));
        }

        Self { cores, num_sites }
    }

    /// Create an identity MPO (output = input)
    pub fn identity(num_sites: usize, d_phys: usize) -> Self {
        let mut mpo = Self::new(num_sites, 1, d_phys, d_phys);

        for i in 0..num_sites {
            // Identity: W[0, j, j, 0] = 1 for all j
            for j in 0..d_phys {
                mpo.cores[i].set(0, j, j, 0, Q16::one());
            }
        }

        mpo
    }

    /// Create a random MPO (for testing/initialization)
    pub fn random<R: Rng>(num_sites: usize, d: usize, d_phys: usize, rng: &mut R) -> Self {
        let mut mpo = Self::new(num_sites, d, d_phys, d_phys);

        // Small random initialization
        let scale = 0.02;
        for core in &mut mpo.cores {
            for val in &mut core.data {
                let r: f64 = rng.gen_range(-scale..scale);
                *val = Q16::from_f64(r);
            }
        }

        // Add identity bias for stability
        for core in &mut mpo.cores {
            let d_left = core.d_left;
            let d_right = core.d_right;
            for l in 0..d_left.min(d_right) {
                for j in 0..core.d_out.min(core.d_in) {
                    let old = core.get(l, j, j, l);
                    core.set(l, j, j, l, old + Q16::one());
                }
            }
        }

        mpo
    }

    /// Total number of parameters
    pub fn num_params(&self) -> usize {
        self.cores.iter().map(|c| c.size()).sum()
    }

    /// Maximum bond dimension
    pub fn max_bond(&self) -> usize {
        self.cores
            .iter()
            .map(|c| c.d_left.max(c.d_right))
            .max()
            .unwrap_or(1)
    }

    /// Serialize MPO weights to bytes (for loading into circuit)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: num_sites, then each core's dimensions
        bytes.extend(&(self.num_sites as u32).to_le_bytes());

        for core in &self.cores {
            bytes.extend(&(core.d_left as u32).to_le_bytes());
            bytes.extend(&(core.d_out as u32).to_le_bytes());
            bytes.extend(&(core.d_in as u32).to_le_bytes());
            bytes.extend(&(core.d_right as u32).to_le_bytes());

            // Data as i64 (fixed-point raw values)
            for val in &core.data {
                bytes.extend(&val.raw.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize MPO weights from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 4 {
            return Err("Buffer too small");
        }

        let mut pos = 0;
        let num_sites = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut cores = Vec::with_capacity(num_sites);

        for _ in 0..num_sites {
            if pos + 16 > bytes.len() {
                return Err("Buffer underrun at core header");
            }

            let d_left = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let d_out = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let d_in = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let d_right = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            let size = d_left * d_out * d_in * d_right;
            let data_bytes = size * 8;

            if pos + data_bytes > bytes.len() {
                return Err("Buffer underrun at core data");
            }

            let mut data = Vec::with_capacity(size);
            for _ in 0..size {
                let raw = i64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
                data.push(Q16::from_raw(raw));
                pos += 8;
            }

            cores.push(MPOCore::from_data(data, d_left, d_out, d_in, d_right));
        }

        Ok(Self { cores, num_sites })
    }
}

/// Default configuration MPO (D=1, making it essentially a per-site linear transform)
impl Default for MPO {
    fn default() -> Self {
        Self::new(L, D, PHYS_DIM, PHYS_DIM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpo_creation() {
        let mpo = MPO::new(4, 2, 2, 2);
        assert_eq!(mpo.num_sites, 4);
        assert_eq!(mpo.cores[0].d_left, 1);
        assert_eq!(mpo.cores[0].d_right, 2);
        assert_eq!(mpo.cores[3].d_left, 2);
        assert_eq!(mpo.cores[3].d_right, 1);
    }

    #[test]
    fn test_identity_mpo() {
        let mpo = MPO::identity(4, 2);

        for i in 0..4 {
            // Check diagonal elements are 1
            assert_eq!(mpo.cores[i].get(0, 0, 0, 0), Q16::one());
            assert_eq!(mpo.cores[i].get(0, 1, 1, 0), Q16::one());
            // Check off-diagonal are 0
            assert_eq!(mpo.cores[i].get(0, 0, 1, 0), Q16::zero());
            assert_eq!(mpo.cores[i].get(0, 1, 0, 0), Q16::zero());
        }
    }

    #[test]
    fn test_serialization() {
        let mut rng = rand::thread_rng();
        let mpo = MPO::random(4, 2, 2, &mut rng);

        let bytes = mpo.to_bytes();
        let mpo2 = MPO::from_bytes(&bytes).unwrap();

        assert_eq!(mpo.num_sites, mpo2.num_sites);
        for i in 0..mpo.num_sites {
            assert_eq!(mpo.cores[i].data.len(), mpo2.cores[i].data.len());
            for j in 0..mpo.cores[i].data.len() {
                assert_eq!(mpo.cores[i].data[j], mpo2.cores[i].data[j]);
            }
        }
    }
}
