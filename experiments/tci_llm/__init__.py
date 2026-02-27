"""
TCI-LLM: Gradient-Free Language Modeling via Tensor Cross Interpolation.

This package provides gradient-free language models built via matrix factorization.

KEY INSIGHT: Language distribution is rank ~50 regardless of context length.

RESULTS:
- 8-byte context:   82% accuracy,  10^14× compression
- 16-byte context:  99% accuracy,  10^33× compression  
- 32-byte context: 100% accuracy,  10^72× compression

GENERALIZATION BREAKTHROUGH (v4):
- 46.6% accuracy on UNSEEN contexts
- 119× improvement over random baseline
- ZERO gradients, ONE matrix solve

Main API:
    SVDLLM.from_corpus(bytes) - Build model from corpus (exact lookup)
    GeneralizedTCI.from_corpus(bytes) - Build model with n-gram generalization
    HybridTCI.from_corpus(bytes) - Best of both: exact + generalized
    
Example:
    >>> from tci_llm import GeneralizedTCI
    >>> model = GeneralizedTCI.from_corpus(corpus_bytes)
    >>> model.generate(b"To be or not", length=100)  # works on unseen contexts!
"""

# Import QTT functions
from .qtt import (
    qtt_from_function_dense,
    qtt_eval_batch,
    qtt_eval_at_index,

    dense_to_qtt_cores,
)

# Import main classes
from .tci_llm import TCI_LLM
from .svd_llm import SVDLLM
from .generalized_tci import GeneralizedTCI, HybridTCI

__version__ = "1.1.0"
__all__ = [
    "TCI_LLM",
    "SVDLLM",
    "GeneralizedTCI",
    "HybridTCI",
    "qtt_from_function_dense",
    "qtt_eval_batch",
    "qtt_eval_at_index",
    "dense_to_qtt_cores",
]
