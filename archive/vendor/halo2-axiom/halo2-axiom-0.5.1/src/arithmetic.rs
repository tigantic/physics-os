//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

use super::multicore;
pub use ff::Field;
use group::{
    ff::{BatchInvert, PrimeField},
    prime::PrimeCurveAffine,
    Curve, GroupOpsOwned, ScalarMulOwned,
};
use rayon::prelude::*;

use halo2curves::msm::msm_best;
pub use halo2curves::{CurveAffine, CurveExt};

/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

// [JPW] Keep this adapter to halo2curves to minimize code changes.
// [HyperTensor] GPU MSM dispatch added: when the `gpu-msm` feature is enabled
// and the curve is BN254, this routes through ICICLE's CUDA MSM kernel instead
// of the CPU Pippenger implementation. Falls back to CPU transparently.
/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
/// With `gpu-msm` feature: dispatches BN254 MSM to ICICLE CUDA backend
/// for vectors >= 4096 elements, falling back to CPU for smaller inputs
/// and non-BN254 curves.
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    #[cfg(feature = "gpu-msm")]
    {
        if let Some(result) = gpu_msm::try_gpu_multiexp::<C>(coeffs, bases) {
            return result;
        }
    }
    msm_best(coeffs, bases)
}

/// GPU-accelerated MSM for BN254 via ICICLE CUDA backend.
///
/// Only dispatches when:
///   1. The curve is BN254 (verified via TypeId)
///   2. The ICICLE GPU backend has been initialised
///   3. The input vector is large enough to amortise transfer overhead (>= 4096)
///
/// Returns `None` to fall back to CPU on any failure.
#[cfg(feature = "gpu-msm")]
mod gpu_msm {
    use super::CurveAffine;
    use halo2curves::bn256;
    use std::sync::OnceLock;

    /// Minimum size for GPU dispatch. Below this, CPU is faster due to
    /// PCIe transfer overhead for host → device copies.
    const GPU_MSM_THRESHOLD: usize = 4096;

    /// Global GPU initialisation state. Initialised once on first MSM call.
    static GPU_READY: OnceLock<bool> = OnceLock::new();

    fn ensure_gpu_ready() -> bool {
        *GPU_READY.get_or_init(|| {
            match icicle_runtime::runtime::load_backend_from_env_or_default() {
                Ok(()) => {
                    let gpu_device = icicle_runtime::Device::new("CUDA", 0);
                    icicle_runtime::set_device(&gpu_device).is_ok()
                }
                Err(_) => false,
            }
        })
    }

    pub fn try_gpu_multiexp<C: CurveAffine>(
        coeffs: &[C::Scalar],
        bases: &[C],
    ) -> Option<C::Curve> {
        // Only intercept BN254 G1 operations
        if std::any::TypeId::of::<C>() != std::any::TypeId::of::<bn256::G1Affine>() {
            return None;
        }
        // Skip small MSMs — CPU is faster due to transfer overhead
        if coeffs.len() < GPU_MSM_THRESHOLD {
            return None;
        }
        // Ensure GPU is ready
        if !ensure_gpu_ready() {
            return None;
        }

        // Compile-time size validation (BN254 scalar = 32 bytes, G1Affine = 64 bytes)
        debug_assert_eq!(std::mem::size_of::<C::Scalar>(), 32);
        debug_assert_eq!(std::mem::size_of::<C>(), 64);

        use icicle_bn254::curve::{
            G1Affine as IcicleG1Affine, G1Projective as IcicleG1Proj, ScalarField,
        };
        use icicle_core::msm::{msm, MSMConfig};
        use icicle_core::traits::ArkConvertible;
        use icicle_runtime::memory::HostSlice;

        // Safety: TypeId check guarantees C is bn256::G1Affine.
        // Both ScalarField and bn256::Fr are 32-byte BN254 Montgomery field elements.
        // Both IcicleG1Affine and bn256::G1Affine are 64-byte {x: Fp, y: Fp} pairs.
        let scalars: &[ScalarField] = unsafe {
            std::slice::from_raw_parts(
                coeffs.as_ptr() as *const ScalarField,
                coeffs.len(),
            )
        };
        let icicle_bases: &[IcicleG1Affine] = unsafe {
            std::slice::from_raw_parts(
                bases.as_ptr() as *const IcicleG1Affine,
                bases.len(),
            )
        };

        let cfg = MSMConfig::default();
        let mut results = vec![IcicleG1Proj::zero(); 1];

        let ok = msm(
            HostSlice::from_slice(scalars),
            HostSlice::from_slice(icicle_bases),
            &cfg,
            HostSlice::from_mut_slice(&mut results),
        );

        if ok.is_err() {
            return None;
        }

        // Convert result: ICICLE Projective → Affine (universal representation)
        // → halo2curves Affine → halo2curves Projective.
        // Going through affine avoids any coordinate-system ambiguity between
        // ICICLE (standard projective) and halo2curves (potentially Jacobian).
        let icicle_affine: IcicleG1Affine = results[0].to_affine();

        // Transmute ICICLE affine → halo2curves affine (same 64-byte layout)
        let halo2_affine: bn256::G1Affine =
            unsafe { std::ptr::read(&icicle_affine as *const IcicleG1Affine as *const bn256::G1Affine) };

        // Convert to projective (halo2curves' native projective type)
        let halo2_proj: bn256::G1 = halo2_affine.into();

        // Cast back to the generic C::Curve (which we know is bn256::G1)
        Some(unsafe { std::ptr::read(&halo2_proj as *const bn256::G1 as *const C::Curve) })
    }
}

/// Dispatcher
pub fn best_fft<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    omega: Scalar,
    log_n: u32,
    data: &FFTData<Scalar>,
    inverse: bool,
) {
    fft::fft(a, omega, log_n, data, inverse);
}

/// Convert coefficient bases group elements to lagrange basis by inverse FFT.
pub fn g_to_lagrange<C: PrimeCurveAffine>(g_projective: Vec<C::Curve>, k: u32) -> Vec<C> {
    let n_inv = C::Scalar::TWO_INV.pow_vartime([k as u64, 0, 0, 0]);
    let omega = C::Scalar::ROOT_OF_UNITY;
    let mut omega_inv = C::Scalar::ROOT_OF_UNITY_INV;
    for _ in k..C::Scalar::S {
        omega_inv = omega_inv.square();
    }

    let mut g_lagrange_projective = g_projective;
    let n = g_lagrange_projective.len();
    let fft_data = FFTData::new(n, omega, omega_inv);

    best_fft(&mut g_lagrange_projective, omega_inv, k, &fft_data, true);
    parallelize(&mut g_lagrange_projective, |g, _| {
        for g in g.iter_mut() {
            *g *= n_inv;
        }
    });

    let mut g_lagrange = vec![C::identity(); 1 << k];
    parallelize(&mut g_lagrange, |g_lagrange, starts| {
        C::Curve::batch_normalize(
            &g_lagrange_projective[starts..(starts + g_lagrange.len())],
            g_lagrange,
        );
    });

    g_lagrange
}

/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    fn evaluate<F: Field>(poly: &[F], point: F) -> F {
        poly.iter()
            .rev()
            .fold(F::ZERO, |acc, coeff| acc * point + coeff)
    }
    let n = poly.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(poly, point)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ZERO; num_threads];
        multicore::scope(|scope| {
            for (chunk_idx, (out, poly)) in
                parts.chunks_mut(1).zip(poly.chunks(chunk_size)).enumerate()
            {
                scope.spawn(move |_| {
                    let start = chunk_idx * chunk_size;
                    out[0] = evaluate(poly, point) * point.pow_vartime([start as u64, 0, 0, 0]);
                });
            }
        });
        parts.iter().fold(F::ZERO, |acc, coeff| acc + coeff)
    }
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
/// For vectors smaller than 32 elements, it uses sequential computation for better performance.
/// For larger vectors, it switches to parallel computation.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(a.len(), b.len());

    if a.len() < 32 {
        // Use sequential computation for small vectors
        let mut acc = F::ZERO;
        for (a, b) in a.iter().zip(b.iter()) {
            acc += (*a) * (*b);
        }
        return acc;
    }

    // Use parallel computation
    a.par_iter().zip(b.par_iter()).map(|(a, b)| (*a) * b).sum()
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, mut b: F) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    b = -b;
    let a = a.into_iter();

    let mut q = vec![F::ZERO; a.len() - 1];

    let mut tmp = F::ZERO;
    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;
        lead_coeff.sub_assign(&tmp);
        *q = lead_coeff;
        tmp = lead_coeff;
        tmp.mul_assign(&b);
    }

    q
}

/// This utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    // Algorithm rationale:
    //
    // Using the stdlib `chunks_mut` will lead to severe load imbalance.
    // From https://github.com/rust-lang/rust/blob/e94bda3/library/core/src/slice/iter.rs#L1607-L1637
    // if the division is not exact, the last chunk will be the remainder.
    //
    // Dividing 40 items on 12 threads will lead to a chunk size of 40/12 = 3,
    // There will be a 13 chunks of size 3 and 1 of size 1 distributed on 12 threads.
    // This leads to 1 thread working on 6 iterations, 1 on 4 iterations and 10 on 3 iterations,
    // a load imbalance of 2x.
    //
    // Instead we can divide work into chunks of size
    // 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3 = 4*4 + 3*8 = 40
    //
    // This would lead to a 6/4 = 1.5x speedup compared to naive chunks_mut
    //
    // See also OpenMP spec (page 60)
    // http://www.openmp.org/mp-documents/openmp-4.5.pdf
    // "When no chunk_size is specified, the iteration space is divided into chunks
    // that are approximately equal in size, and at most one chunk is distributed to
    // each thread. The size of the chunks is unspecified in this case."
    // This implies chunks are the same size ±1

    let f = &f;
    let total_iters = v.len();
    let num_threads = multicore::current_num_threads();
    let base_chunk_size = total_iters / num_threads;
    let cutoff_chunk_id = total_iters % num_threads;
    let split_pos = cutoff_chunk_id * (base_chunk_size + 1);
    let (v_hi, v_lo) = v.split_at_mut(split_pos);

    multicore::scope(|scope| {
        // Skip special-case: number of iterations is cleanly divided by number of threads.
        if cutoff_chunk_id != 0 {
            for (chunk_id, chunk) in v_hi.chunks_exact_mut(base_chunk_size + 1).enumerate() {
                let offset = chunk_id * (base_chunk_size + 1);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
        // Skip special-case: less iterations than number of threads.
        if base_chunk_size != 0 {
            for (chunk_id, chunk) in v_lo.chunks_exact_mut(base_chunk_size).enumerate() {
                let offset = split_pos + (chunk_id * base_chunk_size);
                scope.spawn(move |_| f(chunk, offset));
            }
        }
    });
}

pub fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: Field>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms)
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

pub(crate) fn evaluate_vanishing_polynomial<F: Field>(roots: &[F], z: F) -> F {
    fn evaluate<F: Field>(roots: &[F], z: F) -> F {
        roots.iter().fold(F::ONE, |acc, point| (z - point) * acc)
    }
    let n = roots.len();
    let num_threads = multicore::current_num_threads();
    if n * 2 < num_threads {
        evaluate(roots, z)
    } else {
        let chunk_size = (n + num_threads - 1) / num_threads;
        let mut parts = vec![F::ONE; num_threads];
        multicore::scope(|scope| {
            for (out, roots) in parts.chunks_mut(1).zip(roots.chunks(chunk_size)) {
                scope.spawn(move |_| out[0] = evaluate(roots, z));
            }
        });
        parts.iter().fold(F::ONE, |acc, part| acc * part)
    }
}

pub(crate) fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
    std::iter::successors(Some(F::ONE), move |power| Some(base * power))
}

/// Reverse `l` LSBs of bitvector `n`
pub fn bitreverse(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

#[cfg(test)]
use rand_core::OsRng;

use crate::fft::{self, recursive::FFTData};
#[cfg(test)]
use crate::halo2curves::pasta::Fp;
// use crate::plonk::{get_duration, get_time, start_measure, stop_measure};

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;

    let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
    let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

    for coeffs in 0..5 {
        let points = &points[0..coeffs];
        let evals = &evals[0..coeffs];

        let poly = lagrange_interpolate(points, evals);
        assert_eq!(poly.len(), points.len());

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), *eval);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::OsRng;

    #[test]
    fn test_compute_inner_product() {
        let rng = OsRng;

        // Test small vectors (sequential)
        let a_small: Vec<Fp> = (0..16).map(|_| Fp::random(rng)).collect();
        let b_small: Vec<Fp> = (0..16).map(|_| Fp::random(rng)).collect();
        let result_small = compute_inner_product(&a_small, &b_small);
        let expected_small = a_small
            .iter()
            .zip(b_small.iter())
            .fold(Fp::ZERO, |acc, (a, b)| acc + (*a) * (*b));
        assert_eq!(result_small, expected_small);

        // Test large vectors (parallel)
        let a_large: Vec<Fp> = (0..64).map(|_| Fp::random(rng)).collect();
        let b_large: Vec<Fp> = (0..64).map(|_| Fp::random(rng)).collect();
        let result_large = compute_inner_product(&a_large, &b_large);
        let expected_large = a_large
            .iter()
            .zip(b_large.iter())
            .fold(Fp::ZERO, |acc, (a, b)| acc + (*a) * (*b));
        assert_eq!(result_large, expected_large);
    }
}
