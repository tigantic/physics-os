"""HyperTensor Physics API — Production inference server.

Exposes the QTT Physics VM as a stateless REST API.  All tensor-train
internals (bond dimensions, singular-value spectra, compression
ratios, TT cores) are stripped before responses leave the server.
Clients receive physical observables and conservation diagnostics only.
"""

__version__ = "1.0.0"
