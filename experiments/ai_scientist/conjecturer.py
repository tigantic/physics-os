#!/usr/bin/env python3
"""
Phase 1: The Conjecturer
========================

Neural Symbolic Regression Engine

The Problem: QTT tensors are huge arrays of numbers.
The Goal:    Find the compact function f(i₁,i₂,...) = Tensor[i₁,i₂,...]

This is the difference between:
    - HAVING data: [0.5, 0.25, 0.125, 0.0625, ...]
    - UNDERSTANDING data: f(k) = 0.5 × exp(-ln(2) × k) = 0.5 × 2^(-k)

The Tool: PySR (Parallelized Symbolic Regression)
    - Genetic algorithms to evolve equations
    - Pareto frontier of complexity vs accuracy
    - Outputs human-readable formulas

If this works, we don't just have numbers.
We have the ANALYTIC SOLUTION to the vacuum state.

Example Output:
    Input: σ = [0.8, 0.4, 0.2, 0.1, 0.05]
    Output: σ(k) = 0.8 × exp(-0.693 × k)
    Meaning: Exponential decay with rate γ = ln(2)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable
import json
from datetime import datetime

# Try to import PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Warning: PySR not available. Install with: pip install pysr")


@dataclass
class DiscoveredFormula:
    """
    A formula discovered by symbolic regression.
    
    This is the output of the Conjecturer:
        - The symbolic expression (human-readable)
        - The numerical coefficients
        - The error bounds
        - The complexity score
    """
    expression: str           # e.g., "C * exp(-gamma * k)"
    latex: str                # e.g., "C e^{-\\gamma k}"
    coefficients: Dict[str, float]  # e.g., {"C": 0.8, "gamma": 0.693}
    complexity: int           # Pareto complexity score
    mse: float               # Mean squared error
    r_squared: float         # R² score
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the formula at given points."""
        # Build evaluation context
        ctx = self.coefficients.copy()
        ctx['x'] = x
        ctx['exp'] = np.exp
        ctx['sin'] = np.sin
        ctx['cos'] = np.cos
        ctx['log'] = np.log
        ctx['sqrt'] = np.sqrt
        return eval(self.expression.replace('^', '**'), ctx)
    
    def to_dict(self) -> Dict:
        return {
            'expression': self.expression,
            'latex': self.latex,
            'coefficients': self.coefficients,
            'complexity': self.complexity,
            'mse': self.mse,
            'r_squared': self.r_squared
        }
    
    def __repr__(self):
        return f"Formula({self.latex}, R²={self.r_squared:.4f})"


class Conjecturer:
    """
    The Conjecturer: Discovers analytic formulas from numerical data.
    
    This is Phase 1 of the AI Scientist:
        Input:  Raw numerical data (QTT tensors, singular values, etc.)
        Output: Symbolic formulas that explain the data
    
    The key insight: We're not fitting curves, we're discovering LAWS.
    
    Usage:
        conjecturer = Conjecturer()
        
        # Discover decay law from singular values
        formula = conjecturer.discover_decay_law(singular_values)
        print(formula)  # "σ(k) = 0.8 × exp(-0.693 × k)"
        
        # Discover scaling law from gap vs L
        formula = conjecturer.discover_scaling_law(L_values, gap_values)
        print(formula)  # "Δ(L) = 1.5 - 0.3/L²"
    """
    
    def __init__(self, 
                 niterations: int = 100,
                 populations: int = 15,
                 maxsize: int = 20,
                 binary_operators: List[str] = None,
                 unary_operators: List[str] = None,
                 verbose: bool = True):
        """
        Initialize the Conjecturer.
        
        Args:
            niterations: Number of evolution iterations
            populations: Number of parallel populations
            maxsize: Maximum complexity of formulas
            binary_operators: Allowed binary ops (default: +,-,*,/,^)
            unary_operators: Allowed unary ops (default: exp,log,sqrt,sin,cos)
            verbose: Print progress
        """
        self.niterations = niterations
        self.populations = populations
        self.maxsize = maxsize
        self.verbose = verbose
        
        self.binary_operators = binary_operators or ["+", "-", "*", "/", "^"]
        self.unary_operators = unary_operators or ["exp", "log", "sqrt", "sin", "cos"]
        
        self.discovered_formulas: List[DiscoveredFormula] = []
    
    def _create_regressor(self, **kwargs) -> 'PySRRegressor':
        """Create a PySR regressor with our settings."""
        if not PYSR_AVAILABLE:
            raise ImportError("PySR not available")
        
        # Build params dict, avoiding duplicates
        params = {
            'niterations': kwargs.pop('niterations', self.niterations),
            'populations': kwargs.pop('populations', self.populations),
            'maxsize': kwargs.pop('maxsize', self.maxsize),
            'binary_operators': self.binary_operators,
            'unary_operators': self.unary_operators,
            'verbosity': 1 if self.verbose else 0,
            'progress': self.verbose,
            # Physics-informed settings
            'nested_constraints': {
                "exp": {"exp": 0, "log": 1},  # No exp(exp(x))
                "log": {"exp": 1, "log": 0},  # No log(log(x))
                "sqrt": {"sqrt": 0},          # No sqrt(sqrt(x))
            },
            # Complexity weights (prefer simpler)
            'complexity_of_operators': {
                "+": 1, "-": 1, "*": 1, "/": 2,
                "^": 2, "exp": 2, "log": 2, "sqrt": 2,
                "sin": 3, "cos": 3
            },
        }
        
        # Add remaining kwargs (but don't override our settings)
        for key, val in kwargs.items():
            if key not in params:
                params[key] = val
        
        return PySRRegressor(**params)
    
    def discover(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 feature_names: List[str] = None) -> DiscoveredFormula:
        """
        Discover a formula from data.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: Names for the features
        
        Returns:
            The best discovered formula
        """
        if not PYSR_AVAILABLE:
            return self._fallback_regression(X, y, feature_names)
        
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        model = self._create_regressor()
        model.fit(X, y, variable_names=feature_names)
        
        # Get best formula
        best = model.get_best()
        
        formula = DiscoveredFormula(
            expression=str(best['equation']),
            latex=model.latex(),
            coefficients={},  # TODO: extract from sympy
            complexity=int(best['complexity']),
            mse=float(best['loss']),
            r_squared=float(model.score(X, y))
        )
        
        self.discovered_formulas.append(formula)
        return formula
    
    def discover_decay_law(self, 
                           singular_values: np.ndarray,
                           expect_exponential: bool = True) -> DiscoveredFormula:
        """
        Discover the decay law for singular values.
        
        This is crucial for QTT: if σ_k ~ C exp(-γk), then:
            - The QTT representation is efficient
            - The correlation length ξ = 1/γ is finite
            - The mass gap Δ ~ γ exists
        
        Args:
            singular_values: Array of singular values σ_k
            expect_exponential: If True, prioritize exponential forms
        
        Returns:
            Formula for σ(k)
        """
        k = np.arange(len(singular_values))
        sigma = singular_values
        
        # Filter out zeros/negatives for log fitting
        valid = sigma > 1e-15
        k_valid = k[valid]
        sigma_valid = sigma[valid]
        
        if not PYSR_AVAILABLE:
            # Fallback: fit exponential directly
            return self._fit_exponential(k_valid, sigma_valid)
        
        # Use PySR with physics-informed operators
        model = self._create_regressor(
            unary_operators=["exp", "log"] if expect_exponential else self.unary_operators,
            maxsize=15,  # Keep it simple
            niterations=50  # Faster for this simple case
        )
        
        X = k_valid.reshape(-1, 1)
        y = sigma_valid
        
        model.fit(X, y, variable_names=["k"])
        
        best = model.get_best()
        
        formula = DiscoveredFormula(
            expression=str(best['equation']),
            latex=model.latex(),
            coefficients=self._extract_exponential_params(str(best['equation'])),
            complexity=int(best['complexity']),
            mse=float(best['loss']),
            r_squared=float(model.score(X, y))
        )
        
        self.discovered_formulas.append(formula)
        return formula
    
    def discover_scaling_law(self,
                              L_values: np.ndarray,
                              observables: np.ndarray,
                              name: str = "Δ") -> DiscoveredFormula:
        """
        Discover the scaling law: Observable(L) → Observable(∞)
        
        This is THE KEY to the undecidability problem!
        If we find: Δ(L) = Δ_∞ + a/L^α
        Then: lim_{L→∞} Δ(L) = Δ_∞ (analytically!)
        
        Args:
            L_values: Lattice sizes
            observables: Observable values at each L
            name: Name of the observable
        
        Returns:
            Formula for Observable(L) with extrapolation to L=∞
        """
        L = np.array(L_values, dtype=float)
        obs = np.array(observables, dtype=float)
        
        if not PYSR_AVAILABLE:
            return self._fit_power_law(L, obs, name)
        
        # For scaling laws, we expect forms like:
        # Δ(L) = a + b/L + c/L²
        # or Δ(L) = a + b*exp(-c*L)
        
        model = self._create_regressor()
        # Override specific settings for scaling law discovery
        model.niterations = 100
        model.maxsize = 20
        
        X = L.reshape(-1, 1)
        y = obs
        
        model.fit(X, y, variable_names=["L"])
        
        best = model.get_best()
        expr = str(best['equation'])
        
        # Try to extract the infinite limit
        inf_limit = self._extract_infinite_limit(expr, "L")
        
        formula = DiscoveredFormula(
            expression=expr,
            latex=model.latex(),
            coefficients={"infinite_limit": inf_limit} if inf_limit else {},
            complexity=int(best['complexity']),
            mse=float(best['loss']),
            r_squared=float(model.score(X, y))
        )
        
        self.discovered_formulas.append(formula)
        return formula
    
    def _fit_exponential(self, k: np.ndarray, sigma: np.ndarray) -> DiscoveredFormula:
        """Fallback: fit exponential decay directly."""
        # log(σ) = log(C) - γk
        log_sigma = np.log(sigma)
        coeffs = np.polyfit(k, log_sigma, 1)
        gamma = -coeffs[0]
        C = np.exp(coeffs[1])
        
        # Compute fit quality
        sigma_pred = C * np.exp(-gamma * k)
        mse = np.mean((sigma - sigma_pred)**2)
        ss_res = np.sum((sigma - sigma_pred)**2)
        ss_tot = np.sum((sigma - np.mean(sigma))**2)
        r_squared = 1 - ss_res / ss_tot
        
        return DiscoveredFormula(
            expression=f"{C:.4f} * exp(-{gamma:.4f} * k)",
            latex=f"{C:.3f} e^{{-{gamma:.3f} k}}",
            coefficients={"C": C, "gamma": gamma},
            complexity=4,
            mse=mse,
            r_squared=r_squared
        )
    
    def _fit_power_law(self, L: np.ndarray, obs: np.ndarray, name: str) -> DiscoveredFormula:
        """Fallback: fit power law scaling."""
        # Try: obs(L) = a + b/L^α
        # First estimate α by looking at log(obs - obs_inf) vs log(L)
        
        # Assume obs_inf ≈ obs[-1] (largest L)
        obs_inf_guess = obs[-1]
        
        # Try different power laws
        best_r2 = -np.inf
        best_params = None
        
        for alpha in [1, 2, 3]:
            # Fit: obs = a + b/L^α
            X = np.column_stack([np.ones_like(L), 1/L**alpha])
            coeffs, residuals, _, _ = np.linalg.lstsq(X, obs, rcond=None)
            a, b = coeffs
            
            obs_pred = a + b / L**alpha
            ss_res = np.sum((obs - obs_pred)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2 = 1 - ss_res / ss_tot
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = (a, b, alpha)
        
        a, b, alpha = best_params
        mse = np.mean((obs - (a + b/L**alpha))**2)
        
        return DiscoveredFormula(
            expression=f"{a:.4f} + {b:.4f} / L^{alpha}",
            latex=f"{a:.3f} + \\frac{{{b:.3f}}}{{L^{alpha}}}",
            coefficients={"a": a, "b": b, "alpha": alpha, "infinite_limit": a},
            complexity=5,
            mse=mse,
            r_squared=best_r2
        )
    
    def _extract_exponential_params(self, expr: str) -> Dict[str, float]:
        """Try to extract C and γ from an exponential expression."""
        # This is a simple heuristic - could be improved with sympy
        import re
        
        params = {}
        
        # Look for patterns like "0.5 * exp(-0.3 * k)"
        exp_match = re.search(r'([\d.]+)\s*\*\s*exp\(([-\d.]+)\s*\*', expr)
        if exp_match:
            params['C'] = float(exp_match.group(1))
            params['gamma'] = -float(exp_match.group(2))
        
        return params
    
    def _extract_infinite_limit(self, expr: str, var: str) -> Optional[float]:
        """Try to extract the L→∞ limit from a scaling expression."""
        try:
            import sympy as sp
            x = sp.Symbol(var)
            parsed = sp.sympify(expr)
            limit = sp.limit(parsed, x, sp.oo)
            return float(limit)
        except:
            return None
    
    def _fallback_regression(self, X, y, feature_names) -> DiscoveredFormula:
        """Fallback when PySR is not available."""
        X = np.atleast_2d(X)
        if X.shape[0] == 1:
            X = X.T
        
        # Simple polynomial fit
        if X.shape[1] == 1:
            coeffs = np.polyfit(X.flatten(), y, min(3, len(y)-1))
            expr = " + ".join([f"{c:.4f}*x^{i}" for i, c in enumerate(reversed(coeffs))])
            
            y_pred = np.polyval(coeffs, X.flatten())
            mse = np.mean((y - y_pred)**2)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ss_res / ss_tot
            
            return DiscoveredFormula(
                expression=expr,
                latex=expr,
                coefficients={"poly_coeffs": coeffs.tolist()},
                complexity=len(coeffs),
                mse=mse,
                r_squared=r_squared
            )
        else:
            raise NotImplementedError("Multi-feature fallback not implemented")


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: THE CONJECTURER")
    print("=" * 60)
    print()
    
    conjecturer = Conjecturer(niterations=20, verbose=True)
    
    # Test 1: Exponential decay (simulating singular value decay)
    print("Test 1: Discovering Exponential Decay Law")
    print("-" * 40)
    
    k = np.arange(20)
    C_true, gamma_true = 0.8, 0.5
    sigma_true = C_true * np.exp(-gamma_true * k)
    sigma_noisy = sigma_true * (1 + 0.01 * np.random.randn(len(k)))
    
    print(f"  True formula: σ(k) = {C_true} × exp(-{gamma_true} × k)")
    print(f"  Data: {sigma_noisy[:5]}...")
    
    formula = conjecturer.discover_decay_law(sigma_noisy)
    print(f"  Discovered: {formula}")
    print(f"  Coefficients: {formula.coefficients}")
    print()
    
    # Test 2: Scaling law (simulating gap vs L)
    print("Test 2: Discovering Scaling Law")
    print("-" * 40)
    
    L = np.array([4, 8, 16, 32, 64, 128])
    gap_inf, b_true = 1.5, 0.5
    gap = gap_inf + b_true / L**2
    gap_noisy = gap * (1 + 0.005 * np.random.randn(len(L)))
    
    print(f"  True formula: Δ(L) = {gap_inf} + {b_true}/L²")
    print(f"  Data: {gap_noisy}")
    
    formula = conjecturer.discover_scaling_law(L, gap_noisy, name="Gap")
    print(f"  Discovered: {formula}")
    print(f"  Infinite limit: {formula.coefficients.get('infinite_limit', 'N/A')}")
    print()
    
    print("=" * 60)
    print("CONJECTURER READY")
    print("=" * 60)
