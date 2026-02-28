"""TCI Core - High-performance TT-Cross Interpolation

This package provides Rust-accelerated TCI algorithms for The Ontic Engine.
"""

__version__ = "0.1.0"

# Import from compiled Rust extension
try:
    from tci_core._tci_core import (
        TCISampler,
        IndexBatch,
        MaxVolConfig,
        TruncationPolicy,
        TCIConfig,
    )
    
    RUST_AVAILABLE = True
except ImportError as e:
    RUST_AVAILABLE = False
    _import_error = str(e)
    
    # Provide stub classes for type hints when Rust not built
    class TCISampler:
        """Stub - build Rust extension with `maturin develop`"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"TCI Core Rust extension not available: {_import_error}")
    
    class IndexBatch:
        """Stub - build Rust extension with `maturin develop`"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"TCI Core Rust extension not available: {_import_error}")
    
    class MaxVolConfig:
        """Stub - build Rust extension with `maturin develop`"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"TCI Core Rust extension not available: {_import_error}")
    
    class TruncationPolicy:
        """Stub - build Rust extension with `maturin develop`"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"TCI Core Rust extension not available: {_import_error}")
    
    class TCIConfig:
        """Stub - build Rust extension with `maturin develop`"""
        def __init__(self, *args, **kwargs):
            raise ImportError(f"TCI Core Rust extension not available: {_import_error}")


__all__ = [
    "TCISampler",
    "IndexBatch", 
    "MaxVolConfig",
    "TruncationPolicy",
    "TCIConfig",
    "RUST_AVAILABLE",
]
