"""
QTT: Quantum Tensor Train Compression & Universal File System
==============================================================

A production-grade library for:
- TT-SVD compression of N-dimensional spatial data
- Product Quantization for semantic embeddings
- Unified .qtt container format for random-access slicing

Installation:
    pip install qtt

Quick Start:
    >>> from qtt import QTTContainer
    
    # Spatial data (physics simulations)
    >>> container = QTTContainer.from_spatial_data(temperature_field)
    >>> container.save("physics.qtt")
    >>> with QTTContainer.open("physics.qtt") as f:
    ...     value = f.slice(coords=(64, 64, 64))
    
    # Semantic data (text corpus)
    >>> container = QTTContainer.from_text_corpus(sentences)
    >>> container.save("corpus.qtt")
    >>> with QTTContainer.open("corpus.qtt") as f:
    ...     results = f.slice(query="quantum physics")
    ...     for match in results.matches:
    ...         print(f.read_text(match))

Author: HyperTensor Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "HyperTensor Team"

from qtt.container import (
    QTTContainer,
    QTTHeader,
    QTTFooter,
    SliceResult,
    SemanticMatch,
    MODE_SPATIAL,
    MODE_SEMANTIC,
)

from qtt.spatial import (
    SpatialCompressor,
    tt_svd,
    tt_reconstruct,
    tt_reconstruct_element,
)

from qtt.semantic import (
    SemanticIndex,
    ProductQuantizer,
)

from qtt.slicer import (
    QTTSlicer,
    SpatialIndex,
)

__all__ = [
    # Version
    "__version__",
    
    # Container
    "QTTContainer",
    "QTTHeader", 
    "QTTFooter",
    "SliceResult",
    "SemanticMatch",
    "MODE_SPATIAL",
    "MODE_SEMANTIC",
    
    # Spatial
    "SpatialCompressor",
    "tt_svd",
    "tt_reconstruct",
    "tt_reconstruct_element",
    
    # Semantic
    "SemanticIndex",
    "ProductQuantizer",
    
    # Slicer
    "QTTSlicer",
    "SpatialIndex",
]
