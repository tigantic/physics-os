"""QTT Physics VM — Domain compilers.

Each compiler transforms a physics domain's governing equations into
the same operator IR, proving that the backend is the product — not
the domain.
"""

from .base import BaseCompiler
from .diffusion import DiffusionCompiler
from .maxwell import MaxwellCompiler
from .navier_stokes import BurgersCompiler
from .schrodinger import SchrodingerCompiler
from .vlasov_poisson import VlasovPoissonCompiler

ALL_COMPILERS: list[type[BaseCompiler]] = [
    BurgersCompiler,
    MaxwellCompiler,
    SchrodingerCompiler,
    DiffusionCompiler,
    VlasovPoissonCompiler,
]

__all__ = [
    "BaseCompiler",
    "BurgersCompiler",
    "MaxwellCompiler",
    "SchrodingerCompiler",
    "DiffusionCompiler",
    "VlasovPoissonCompiler",
    "ALL_COMPILERS",
]
