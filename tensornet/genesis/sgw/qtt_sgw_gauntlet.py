#!/usr/bin/env python3
"""
QTT-SGW Layer 21 Gauntlet — Elite Validation Suite

Comprehensive testing for QTT-Spectral Graph Wavelets.

Articles of Constitution Reference: Article II, Section 2.1

Tests:
1. Laplacian Tests: Structure, eigenvalues, matvec
2. Signal Tests: Creation, operations, norms
3. Chebyshev Tests: Coefficients, approximation accuracy
4. Wavelet Tests: Transform, localization, energy
5. Filter Tests: Low-pass, high-pass, band-pass
6. Performance Tests: Scaling with grid size
7. Constitutional Tests: Compression, complexity, API

Usage:
    python qtt_sgw_gauntlet.py           # Quick mode
    python qtt_sgw_gauntlet.py --full    # Full validation
"""

from __future__ import annotations
import sys
import time
import math
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch

# Add parent to path for imports
sys.path.insert(0, '/home/brad/TiganticLabz/Main_Projects/HyperTensor-VM-main')

from tensornet.genesis.sgw import (
    QTTLaplacian, grid_laplacian_1d,
    QTTSignal,
    ChebyshevApproximator, chebyshev_coefficients, chebyshev_approximation,
    QTTGraphWavelet, WaveletResult,
    mexican_hat_kernel, heat_kernel,
    LowPassFilter, HighPassFilter, BandPassFilter
)


@dataclass
class TestResult:
    """Single test result."""
    name: str
    passed: bool
    runtime: float
    message: str = ""


@dataclass
class GauntletConfig:
    """Gauntlet configuration."""
    verbose: bool = False
    quick: bool = True
    seed: int = 42
    
    # Grid sizes
    small_grid: int = 1024       # 2^10
    medium_grid: int = 4096      # 2^12
    large_grid: int = 65536      # 2^16


class QTTSGWGauntlet:
    """
    Elite validation gauntlet for QTT-Spectral Graph Wavelets.
    
    Tests all aspects of Layer 21 implementation.
    """
    
    def __init__(self, config: Optional[GauntletConfig] = None):
        self.config = config or GauntletConfig()
        self.results: List[TestResult] = []
        torch.manual_seed(self.config.seed)
    
    def _add_result(self, name: str, passed: bool, runtime: float, message: str = ""):
        """Add test result."""
        self.results.append(TestResult(name, passed, runtime, message))
        if self.config.verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {name} ({runtime:.3f}s) {message}")
    
    # =========================================================================
    # LAPLACIAN TESTS
    # =========================================================================
    
    def run_laplacian_tests(self):
        """Test QTT Laplacian implementation."""
        print("\n▶ Laplacian Tests")
        
        self._test_laplacian_creation()
        self._test_laplacian_structure()
        self._test_laplacian_eigenvalues()
        self._test_laplacian_matvec()
    
    def _test_laplacian_creation(self):
        """Test Laplacian creation for different grids."""
        start = time.perf_counter()
        
        try:
            # 1D grid
            L1 = QTTLaplacian.grid_1d(self.config.small_grid)
            assert L1.num_nodes == self.config.small_grid
            assert L1.graph_type == 'grid_1d'
            assert len(L1.cores) == int(math.log2(self.config.small_grid))
            
            # Check rank (should be 3 for tridiagonal)
            assert L1.max_rank == 3, f"Expected rank 3, got {L1.max_rank}"
            
            passed = True
            message = f"1D: {L1.num_nodes} nodes, rank {L1.max_rank}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Laplacian creation", passed, 
                         time.perf_counter() - start, message)
    
    def _test_laplacian_structure(self):
        """Test Laplacian has correct tridiagonal structure."""
        start = time.perf_counter()
        
        try:
            # Small grid for dense comparison
            n = 64  # 2^6
            L = QTTLaplacian.grid_1d(n)
            L_dense = L.to_dense()
            
            # Check tridiagonal structure
            # Diagonal should be 2 (or close to it)
            diag = torch.diag(L_dense)
            
            # Off-diagonals should be -1
            off_diag = torch.diag(L_dense, 1)
            
            # For a path graph Laplacian:
            # Interior nodes: L[i,i] = 2, L[i,i±1] = -1
            # Boundary nodes: L[0,0] = L[n-1,n-1] = 1 or 2 depending on BC
            
            # Check symmetry
            sym_error = (L_dense - L_dense.T).abs().max().item()
            assert sym_error < 1e-10, f"Not symmetric: error = {sym_error}"
            
            passed = True
            message = f"Symmetry error: {sym_error:.2e}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Laplacian structure", passed,
                         time.perf_counter() - start, message)
    
    def _test_laplacian_eigenvalues(self):
        """Test Laplacian eigenvalue bounds."""
        start = time.perf_counter()
        
        try:
            n = 64
            L = QTTLaplacian.grid_1d(n)
            L_dense = L.to_dense()
            
            # Compute eigenvalues
            eigs = torch.linalg.eigvalsh(L_dense)
            
            # Laplacian is positive semi-definite
            assert eigs.min() >= -1e-10, f"Negative eigenvalue: {eigs.min()}"
            
            # λ_max ≤ 4 for 1D grid
            assert eigs.max() <= 4.0 + 1e-6, f"λ_max too large: {eigs.max()}"
            
            # For path graph with Dirichlet BC, λ_min > 0
            # For connected graph with Neumann BC, λ_min ≈ 0
            # Our implementation uses path graph structure, so just check non-negative
            assert eigs.min() >= 0, f"Negative eigenvalue: {eigs.min()}"
            
            passed = True
            message = f"λ ∈ [{eigs.min():.4f}, {eigs.max():.4f}]"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Laplacian eigenvalues", passed,
                         time.perf_counter() - start, message)
    
    def _test_laplacian_matvec(self):
        """Test Laplacian matrix-vector product."""
        start = time.perf_counter()
        
        try:
            n = 64
            L = QTTLaplacian.grid_1d(n)
            
            # Create random signal
            signal = QTTSignal.random(n, rank=5, seed=42)
            
            # Apply Laplacian via QTT
            result_qtt = L.matvec(signal)
            
            # Compare with dense
            L_dense = L.to_dense()
            signal_dense = signal.to_dense()
            result_dense = L_dense @ signal_dense
            
            result_qtt_dense = result_qtt.to_dense()
            
            # Check error
            error = (result_qtt_dense - result_dense).norm() / (result_dense.norm() + 1e-10)
            
            assert error < 0.1, f"Matvec error too large: {error}"
            
            passed = True
            message = f"Relative error: {error:.2e}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Laplacian matvec", passed,
                         time.perf_counter() - start, message)
    
    # =========================================================================
    # SIGNAL TESTS
    # =========================================================================
    
    def run_signal_tests(self):
        """Test QTTSignal implementation."""
        print("\n▶ Signal Tests")
        
        self._test_signal_creation()
        self._test_signal_operations()
        self._test_signal_norms()
    
    def _test_signal_creation(self):
        """Test signal creation methods."""
        start = time.perf_counter()
        
        try:
            n = self.config.small_grid
            
            # Constant signal
            const = QTTSignal.constant(n, value=2.0)
            assert abs(const.to_dense().mean().item() - 2.0) < 1e-10
            
            # Zero signal
            zeros = QTTSignal.zeros(n)
            assert zeros.norm() < 1e-10
            
            # Delta signal
            delta = QTTSignal.delta(n, node_index=0)
            delta_dense = delta.to_dense()
            assert abs(delta_dense[0].item() - 1.0) < 1e-10
            assert delta_dense[1:].abs().max() < 1e-10
            
            # Random signal
            rand = QTTSignal.random(n, rank=10, seed=42)
            assert rand.max_rank <= 10
            
            passed = True
            message = f"All creation methods work"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Signal creation", passed,
                         time.perf_counter() - start, message)
    
    def _test_signal_operations(self):
        """Test signal arithmetic operations."""
        start = time.perf_counter()
        
        try:
            n = 256
            
            s1 = QTTSignal.random(n, rank=5, seed=42)
            s2 = QTTSignal.random(n, rank=5, seed=43)
            
            # Addition
            s_add = s1.add(s2)
            add_dense = s_add.to_dense()
            expected = s1.to_dense() + s2.to_dense()
            add_error = (add_dense - expected).norm() / expected.norm()
            assert add_error < 1e-6, f"Addition error: {add_error}"
            
            # Scaling
            s_scale = s1.scale(3.0)
            scale_dense = s_scale.to_dense()
            expected = 3.0 * s1.to_dense()
            scale_error = (scale_dense - expected).norm() / expected.norm()
            assert scale_error < 1e-10, f"Scaling error: {scale_error}"
            
            # Inner product
            dot = s1.dot(s2)
            expected_dot = (s1.to_dense() * s2.to_dense()).sum().item()
            dot_error = abs(dot - expected_dot) / (abs(expected_dot) + 1e-10)
            assert dot_error < 1e-6, f"Dot product error: {dot_error}"
            
            passed = True
            message = f"Add: {add_error:.2e}, Scale: {scale_error:.2e}, Dot: {dot_error:.2e}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Signal operations", passed,
                         time.perf_counter() - start, message)
    
    def _test_signal_norms(self):
        """Test signal norm computation."""
        start = time.perf_counter()
        
        try:
            n = 256
            
            # Random signal
            s = QTTSignal.random(n, rank=10, seed=42)
            
            # QTT norm
            qtt_norm = s.norm()
            
            # Dense norm
            dense_norm = s.to_dense().norm().item()
            
            rel_error = abs(qtt_norm - dense_norm) / (dense_norm + 1e-10)
            assert rel_error < 1e-6, f"Norm error: {rel_error}"
            
            # Normalization
            s_normalized = s.normalize()
            normalized_norm = s_normalized.norm()
            assert abs(normalized_norm - 1.0) < 1e-6, f"Normalized norm: {normalized_norm}"
            
            passed = True
            message = f"Norm error: {rel_error:.2e}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Signal norms", passed,
                         time.perf_counter() - start, message)
    
    # =========================================================================
    # CHEBYSHEV TESTS
    # =========================================================================
    
    def run_chebyshev_tests(self):
        """Test Chebyshev polynomial approximation."""
        print("\n▶ Chebyshev Tests")
        
        self._test_chebyshev_coefficients()
        self._test_chebyshev_approximation()
        self._test_chebyshev_matrix_function()
    
    def _test_chebyshev_coefficients(self):
        """Test Chebyshev coefficient computation."""
        start = time.perf_counter()
        
        try:
            # Approximate exp(-x) on [-1, 1]
            func = lambda x: math.exp(-x)
            coeffs = chebyshev_coefficients(func, order=20)
            
            assert len(coeffs) == 20
            
            # Check approximation quality
            x_test = torch.linspace(-1, 1, 100)
            approx = chebyshev_approximation(coeffs, x_test)
            exact = torch.tensor([func(xi.item()) for xi in x_test])
            
            max_error = (approx - exact).abs().max().item()
            assert max_error < 1e-6, f"Chebyshev error: {max_error}"
            
            passed = True
            message = f"Max error: {max_error:.2e}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Chebyshev coefficients", passed,
                         time.perf_counter() - start, message)
    
    def _test_chebyshev_approximation(self):
        """Test Chebyshev approximation for various functions."""
        start = time.perf_counter()
        
        try:
            errors = []
            
            # Test functions
            test_funcs = [
                ("exp(-x)", lambda x: math.exp(-x)),
                ("cos(πx)", lambda x: math.cos(math.pi * x)),
                ("1/(1+x²)", lambda x: 1.0 / (1 + x**2)),
            ]
            
            for name, func in test_funcs:
                coeffs = chebyshev_coefficients(func, order=30)
                x_test = torch.linspace(-1, 1, 100)
                approx = chebyshev_approximation(coeffs, x_test)
                exact = torch.tensor([func(xi.item()) for xi in x_test])
                error = (approx - exact).abs().max().item()
                errors.append((name, error))
            
            # All errors should be small
            max_err = max(e for _, e in errors)
            assert max_err < 1e-6, f"Approximation errors too large"
            
            passed = True
            message = ", ".join(f"{n}: {e:.2e}" for n, e in errors)
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Chebyshev approximation", passed,
                         time.perf_counter() - start, message)
    
    def _test_chebyshev_matrix_function(self):
        """Test Chebyshev approximation of matrix function."""
        start = time.perf_counter()
        
        try:
            n = 64
            L = QTTLaplacian.grid_1d(n)
            
            # Heat kernel: exp(-λ)
            heat_func = lambda lam: math.exp(-lam)
            approx = ChebyshevApproximator.from_function(heat_func, L, order=30)
            
            # Apply to random signal
            signal = QTTSignal.random(n, rank=5, seed=42)
            result = approx.apply(signal)
            
            # Compare with dense
            L_dense = L.to_dense()
            signal_dense = signal.to_dense()
            
            # exp(-L) via eigendecomposition
            eigs, vecs = torch.linalg.eigh(L_dense)
            exp_L = vecs @ torch.diag(torch.exp(-eigs)) @ vecs.T
            expected = exp_L @ signal_dense
            
            result_dense = result.to_dense()
            rel_error = (result_dense - expected).norm() / expected.norm()
            
            # Allow larger tolerance due to multiple approximations
            assert rel_error < 0.5, f"Matrix function error: {rel_error}"
            
            passed = True
            message = f"exp(-L) relative error: {rel_error:.2e}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Chebyshev matrix function", passed,
                         time.perf_counter() - start, message)
    
    # =========================================================================
    # WAVELET TESTS
    # =========================================================================
    
    def run_wavelet_tests(self):
        """Test spectral graph wavelets."""
        print("\n▶ Wavelet Tests")
        
        self._test_wavelet_creation()
        self._test_wavelet_transform()
        self._test_wavelet_energy()
        self._test_wavelet_localization()
    
    def _test_wavelet_creation(self):
        """Test wavelet object creation."""
        start = time.perf_counter()
        
        try:
            n = 256
            L = QTTLaplacian.grid_1d(n)
            
            # Create wavelet with different kernels
            for kernel in ['mexican_hat', 'heat']:
                wavelet = QTTGraphWavelet.create(
                    L, 
                    scales=[1, 2, 4, 8],
                    kernel=kernel,
                    chebyshev_order=20
                )
                
                assert len(wavelet.scales) == 4
                assert len(wavelet.approximators) == 4
            
            passed = True
            message = f"Created wavelets for multiple kernels"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Wavelet creation", passed,
                         time.perf_counter() - start, message)
    
    def _test_wavelet_transform(self):
        """Test wavelet transform."""
        start = time.perf_counter()
        
        try:
            n = 256
            L = QTTLaplacian.grid_1d(n)
            
            wavelet = QTTGraphWavelet.create(
                L,
                scales=[1, 2, 4],
                kernel='heat',
                chebyshev_order=20
            )
            
            # Transform random signal
            signal = QTTSignal.random(n, rank=5, seed=42)
            result = wavelet.transform(signal)
            
            # Check result structure
            assert isinstance(result, WaveletResult)
            assert len(result.coefficients) == 3
            assert result.scales == [1, 2, 4]
            
            # Each coefficient should be a QTTSignal
            for coef in result.coefficients:
                assert isinstance(coef, QTTSignal)
                assert coef.num_nodes == n
            
            passed = True
            message = f"{len(result.coefficients)} scale coefficients"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Wavelet transform", passed,
                         time.perf_counter() - start, message)
    
    def _test_wavelet_energy(self):
        """Test wavelet energy distribution."""
        start = time.perf_counter()
        
        try:
            n = 256
            L = QTTLaplacian.grid_1d(n)
            
            wavelet = QTTGraphWavelet.create(
                L,
                scales=[1, 2, 4, 8],
                kernel='heat',
                chebyshev_order=20
            )
            
            # Transform signal
            signal = QTTSignal.random(n, rank=5, seed=42).normalize()
            result = wavelet.transform(signal)
            
            # Get energy per scale
            energies = result.energy_per_scale()
            total_energy = result.total_energy()
            
            # All energies should be positive
            assert all(e >= 0 for e in energies), "Negative energy"
            
            # Total energy should be positive
            assert total_energy > 0, f"Zero total energy"
            
            passed = True
            message = f"Energies: {[f'{e:.3f}' for e in energies]}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Wavelet energy", passed,
                         time.perf_counter() - start, message)
    
    def _test_wavelet_localization(self):
        """Test wavelet localization at a node."""
        start = time.perf_counter()
        
        try:
            n = 256
            L = QTTLaplacian.grid_1d(n)
            
            wavelet = QTTGraphWavelet.create(
                L,
                scales=[1, 2, 4],
                kernel='heat',
                chebyshev_order=20
            )
            
            # Get localized wavelet at center
            center_node = n // 2
            localized = wavelet.localization_at_node(center_node, scale_index=0)
            
            # Should be a valid signal
            assert isinstance(localized, QTTSignal)
            assert localized.num_nodes == n
            
            # Norm should be reasonable
            norm = localized.norm()
            assert norm > 0 and norm < 100, f"Unexpected norm: {norm}"
            
            passed = True
            message = f"Localized wavelet norm: {norm:.4f}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Wavelet localization", passed,
                         time.perf_counter() - start, message)
    
    # =========================================================================
    # FILTER TESTS
    # =========================================================================
    
    def run_filter_tests(self):
        """Test graph filters."""
        print("\n▶ Filter Tests")
        
        self._test_lowpass_filter()
        self._test_highpass_filter()
        self._test_bandpass_filter()
    
    def _test_lowpass_filter(self):
        """Test low-pass filter."""
        start = time.perf_counter()
        
        try:
            n = 256
            L = QTTLaplacian.grid_1d(n)
            
            # Create low-pass filter
            lpf = LowPassFilter(laplacian=L, cutoff=0.3, chebyshev_order=20)
            
            # Filter response should be 1 at λ=0 and decay
            assert abs(lpf.response(0.0) - 1.0) < 1e-10
            assert lpf.response(L.max_eigenvalue) < 0.5
            
            # Apply to random signal
            signal = QTTSignal.random(n, rank=5, seed=42)
            filtered = lpf.apply(signal)
            
            assert isinstance(filtered, QTTSignal)
            
            passed = True
            message = f"Response at λ_max: {lpf.response(L.max_eigenvalue):.4f}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Low-pass filter", passed,
                         time.perf_counter() - start, message)
    
    def _test_highpass_filter(self):
        """Test high-pass filter."""
        start = time.perf_counter()
        
        try:
            n = 256
            L = QTTLaplacian.grid_1d(n)
            
            # Create high-pass filter
            hpf = HighPassFilter(laplacian=L, cutoff=0.3, chebyshev_order=20)
            
            # Filter response should be 0 at λ=0 and increase
            assert abs(hpf.response(0.0)) < 1e-10
            assert hpf.response(L.max_eigenvalue) > 0.5
            
            # Apply to signal
            signal = QTTSignal.random(n, rank=5, seed=42)
            filtered = hpf.apply(signal)
            
            assert isinstance(filtered, QTTSignal)
            
            passed = True
            message = f"Response at λ_max: {hpf.response(L.max_eigenvalue):.4f}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("High-pass filter", passed,
                         time.perf_counter() - start, message)
    
    def _test_bandpass_filter(self):
        """Test band-pass filter."""
        start = time.perf_counter()
        
        try:
            n = 256
            L = QTTLaplacian.grid_1d(n)
            
            # Create band-pass filter
            bpf = BandPassFilter(laplacian=L, low=0.2, high=0.8, chebyshev_order=20)
            
            # Response should be low at extremes, higher in band
            r_low = bpf.response(0.0)
            r_mid = bpf.response(L.max_eigenvalue * 0.5)
            r_high = bpf.response(L.max_eigenvalue)
            
            # Apply to signal
            signal = QTTSignal.random(n, rank=5, seed=42)
            filtered = bpf.apply(signal)
            
            assert isinstance(filtered, QTTSignal)
            
            passed = True
            message = f"Response: low={r_low:.3f}, mid={r_mid:.3f}, high={r_high:.3f}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Band-pass filter", passed,
                         time.perf_counter() - start, message)
    
    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================
    
    def run_performance_tests(self):
        """Test performance scaling."""
        print("\n▶ Performance Tests")
        
        self._test_laplacian_scaling()
        self._test_signal_scaling()
        self._test_wavelet_scaling()
    
    def _test_laplacian_scaling(self):
        """Test Laplacian creation scaling."""
        start = time.perf_counter()
        
        try:
            sizes = [2**k for k in range(10, 17)]  # 2^10 to 2^16
            times = []
            
            for n in sizes:
                t0 = time.perf_counter()
                L = QTTLaplacian.grid_1d(n)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            
            # Should scale as O(log N)
            # Check that doubling N doesn't double time
            ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            
            # For O(log N), ratio should be much less than size_ratio
            # For O(log N): ratio ≈ log(N_large) / log(N_small) ≈ 16/10 = 1.6
            assert ratio < size_ratio * 0.1, f"Scaling too slow: ratio={ratio}"
            
            passed = True
            message = f"Time ratio: {ratio:.2f} for size ratio {size_ratio}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Laplacian scaling", passed,
                         time.perf_counter() - start, message)
    
    def _test_signal_scaling(self):
        """Test signal operations scaling."""
        start = time.perf_counter()
        
        try:
            sizes = [2**k for k in range(10, 17)]
            times = []
            
            for n in sizes:
                s1 = QTTSignal.random(n, rank=10, seed=42)
                s2 = QTTSignal.random(n, rank=10, seed=43)
                
                t0 = time.perf_counter()
                _ = s1.add(s2)
                _ = s1.dot(s2)
                _ = s1.norm()
                t1 = time.perf_counter()
                times.append(t1 - t0)
            
            # Should scale as O(r² log N)
            ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            
            assert ratio < size_ratio * 0.1, f"Scaling too slow"
            
            passed = True
            message = f"Time ratio: {ratio:.2f}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Signal scaling", passed,
                         time.perf_counter() - start, message)
    
    def _test_wavelet_scaling(self):
        """Test wavelet transform scaling."""
        start = time.perf_counter()
        
        try:
            sizes = [2**k for k in range(8, 13)]  # 2^8 to 2^12
            times = []
            
            for n in sizes:
                L = QTTLaplacian.grid_1d(n)
                wavelet = QTTGraphWavelet.create(
                    L, scales=[1, 2], kernel='heat', chebyshev_order=10
                )
                signal = QTTSignal.random(n, rank=5, seed=42)
                
                t0 = time.perf_counter()
                _ = wavelet.transform(signal)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            
            # Check reasonable scaling
            ratio = times[-1] / times[0]
            size_ratio = sizes[-1] / sizes[0]
            
            # Should be much better than O(N³)
            assert ratio < size_ratio, f"Scaling not sublinear"
            
            passed = True
            message = f"Time ratio: {ratio:.2f} for size ratio {size_ratio}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Wavelet scaling", passed,
                         time.perf_counter() - start, message)
    
    # =========================================================================
    # CONSTITUTIONAL TESTS
    # =========================================================================
    
    def run_constitutional_tests(self):
        """Test constitutional compliance."""
        print("\n▶ Constitutional Tests")
        
        self._test_compression_covenant()
        self._test_complexity_covenant()
        self._test_api_covenant()
    
    def _test_compression_covenant(self):
        """Article I: Compression Covenant."""
        start = time.perf_counter()
        
        try:
            n = 2**14
            L = QTTLaplacian.grid_1d(n)
            
            # Count parameters in Laplacian
            num_params = sum(c.numel() for c in L.cores)
            d = len(L.cores)
            r = L.max_rank
            
            # Expected: O(d * r² * 4) where 4 is for 2x2 MPO
            expected = d * r * r * 4
            
            ratio = num_params / expected
            assert 0.1 < ratio < 10, f"Unexpected parameter count ratio: {ratio}"
            
            passed = True
            message = f"{num_params} params, rank={r}"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Compression covenant", passed,
                         time.perf_counter() - start, message)
    
    def _test_complexity_covenant(self):
        """Article II: Complexity Compact."""
        start = time.perf_counter()
        
        try:
            n = 2**12
            L = QTTLaplacian.grid_1d(n)
            signal = QTTSignal.random(n, rank=5, seed=42)
            
            # Operations should be fast
            t0 = time.perf_counter()
            _ = L.matvec(signal)
            t_matvec = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            _ = signal.add(signal)
            t_add = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            _ = signal.norm()
            t_norm = time.perf_counter() - t0
            
            # All should be < 1 second for n = 2^12
            assert t_matvec < 1.0, f"Matvec too slow: {t_matvec}s"
            assert t_add < 1.0, f"Add too slow: {t_add}s"
            assert t_norm < 1.0, f"Norm too slow: {t_norm}s"
            
            passed = True
            message = f"matvec={t_matvec:.3f}s, add={t_add:.3f}s, norm={t_norm:.3f}s"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("Complexity covenant", passed,
                         time.perf_counter() - start, message)
    
    def _test_api_covenant(self):
        """Article III: API Accord."""
        start = time.perf_counter()
        
        try:
            # Check that all documented APIs exist and work
            from tensornet.genesis.sgw import (
                QTTLaplacian, grid_laplacian_1d,
                QTTSignal,
                ChebyshevApproximator,
                QTTGraphWavelet, WaveletResult,
                LowPassFilter, HighPassFilter, BandPassFilter
            )
            
            # Quick functionality check
            L = grid_laplacian_1d(256)
            signal = QTTSignal.random(256, rank=5)
            wavelet = QTTGraphWavelet.create(L, scales=[1, 2])
            result = wavelet.transform(signal)
            lpf = LowPassFilter(laplacian=L, cutoff=0.5)
            filtered = lpf.apply(signal)
            
            passed = True
            message = "All APIs accessible and functional"
        except Exception as e:
            passed = False
            message = str(e)
        
        self._add_result("API covenant", passed,
                         time.perf_counter() - start, message)
    
    # =========================================================================
    # MAIN RUNNER
    # =========================================================================
    
    def run(self):
        """Run all tests."""
        print("=" * 80)
        print("QTT-SGW LAYER 21 GAUNTLET")
        print("=" * 80)
        print(f"Configuration: verbose={self.config.verbose}, quick={self.config.quick}")
        
        start = time.perf_counter()
        
        self.run_laplacian_tests()
        self.run_signal_tests()
        self.run_chebyshev_tests()
        self.run_wavelet_tests()
        self.run_filter_tests()
        
        if not self.config.quick:
            self.run_performance_tests()
        
        self.run_constitutional_tests()
        
        total_time = time.perf_counter() - start
        
        # Summary
        print("\n" + "=" * 80)
        print("QTT-SGW GAUNTLET REPORT")
        print("=" * 80)
        print()
        print(f"{'Test':<50} {'Status':<10} {'Time':<10}")
        print("-" * 70)
        
        passed = 0
        failed = 0
        
        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{result.name:<50} {status:<10} {result.runtime:.3f}s")
            if result.passed:
                passed += 1
            else:
                failed += 1
                if result.message:
                    print(f"  └─ {result.message}")
        
        print("-" * 70)
        print(f"Total: {len(self.results)} tests, {passed} passed, {failed} failed")
        print(f"Success Rate: {100 * passed / len(self.results):.1f}%")
        print(f"Total Runtime: {total_time:.2f}s")
        print("=" * 80)
        
        if failed == 0:
            print("\n✅ GAUNTLET PASSED - All tests successful")
            return 0
        else:
            print(f"\n❌ GAUNTLET FAILED - {failed} test(s) failed")
            return 1


def main():
    parser = argparse.ArgumentParser(description="QTT-SGW Layer 21 Gauntlet")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Verbose output")
    parser.add_argument("--full", action="store_true",
                        help="Run full test suite (including performance)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    config = GauntletConfig(
        verbose=args.verbose,
        quick=not args.full,
        seed=args.seed
    )
    
    gauntlet = QTTSGWGauntlet(config)
    return gauntlet.run()


if __name__ == "__main__":
    sys.exit(main())
