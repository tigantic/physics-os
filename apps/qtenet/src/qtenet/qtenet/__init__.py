"""QTeneT — Break the Curse of Dimensionality

Enterprise-grade QTT (Quantized Tensor Train) library for physics-first computing.

QTeneT enables O(log N) complexity for problems that are IMPOSSIBLE with 
traditional methods. The curse of dimensionality is broken.

Primary Exports (Curse-Breaking Primitives):
    operators: N-dimensional shift, Laplacian, gradient operators
    tci: Build QTT from black-box functions with O(r² × log N) samples
    genesis: 7 meta-primitives (OT, SGW, RMT, Tropical, RKHS, Topology, GA)
    solvers: Vlasov 5D/6D, Euler N-D at O(log N) per timestep
    benchmarks: Quantify the curse-breaking advantage
    demos: Runnable demonstrations (Holy Grail 6D)

Quick Start:
    >>> from qtenet.operators import shift_nd
    >>> from qtenet.tci import from_function
    >>> from qtenet.solvers import Vlasov6D
    >>> from qtenet.demos import holy_grail_6d
    >>> 
    >>> # Run the Holy Grail: 6D Vlasov-Maxwell with 1 billion points
    >>> results = holy_grail_6d(n_steps=100)
    >>> print(f"Compression: {results.compression_ratio:,.0f}×")

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

__version__ = "0.1.0"
__author__ = "Tigantic Holdings LLC"

# ============================================================================
# CURSE-BREAKING PRIMITIVES (Primary API)
# These are the exports that matter — they break the curse of dimensionality
# ============================================================================

# 1. N-Dimensional Operators (The Master Key)
from qtenet import operators
from qtenet.operators import shift_nd, apply_shift, laplacian_nd, gradient_nd

# 2. TCI: Black-Box Function → QTT with O(r² × log N) samples
from qtenet import tci
from qtenet.tci import from_function, from_function_nd

# 3. Genesis: 7 Meta-Primitives for New Mathematical Domains
from qtenet import genesis

# 4. Solvers: N-D Physics at O(log N)
from qtenet import solvers
from qtenet.solvers import Vlasov5D, Vlasov6D, EulerND

# 5. Benchmarks: Prove the Curse is Broken
from qtenet import benchmarks
from qtenet.benchmarks import curse_of_dimensionality

# 6. Demos: Runnable Demonstrations
from qtenet import demos
from qtenet.demos import holy_grail_6d, holy_grail_5d

# Legacy: SDK for enterprise integration
from qtenet import sdk

__all__ = [
    # Version
    "__version__",
    # Submodules
    "operators",
    "tci",
    "genesis",
    "solvers",
    "benchmarks",
    "demos",
    "sdk",
    # Operators (The Master Key)
    "shift_nd",
    "apply_shift",
    "laplacian_nd",
    "gradient_nd",
    # TCI (Function → QTT)
    "from_function",
    "from_function_nd",
    # Solvers
    "Vlasov5D",
    "Vlasov6D",
    "EulerND",
    # Benchmarks
    "curse_of_dimensionality",
    # Demos
    "holy_grail_6d",
    "holy_grail_5d",
]
