"""
Sovereign Architecture: End-to-End Tensor Sparsity

Implicit QTT evaluation without materialization.
"""

from .implicit_qtt_renderer import ImplicitQTTRenderer, test_implicit_renderer

__all__ = ['ImplicitQTTRenderer', 'test_implicit_renderer']
