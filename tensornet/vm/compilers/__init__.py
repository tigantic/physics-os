"""QTT Physics VM — Domain compilers.

Each compiler transforms a physics domain's governing equations into
the same operator IR, proving that the backend is the product — not
the domain.
"""

from .base import BaseCompiler
from .diffusion import DiffusionCompiler
from .maxwell import MaxwellCompiler
from .maxwell_3d import Maxwell3DCompiler
from .navier_stokes import BurgersCompiler
from .navier_stokes_2d import NavierStokes2DCompiler
from .schrodinger import SchrodingerCompiler
from .vlasov_poisson import VlasovPoissonCompiler

ALL_COMPILERS: list[type[BaseCompiler]] = [
    BurgersCompiler,
    MaxwellCompiler,
    SchrodingerCompiler,
    DiffusionCompiler,
    VlasovPoissonCompiler,
]

MULTI_DIM_COMPILERS: list[type[BaseCompiler]] = [
    NavierStokes2DCompiler,
    Maxwell3DCompiler,
]

__all__ = [
    "BaseCompiler",
    "BurgersCompiler",
    "MaxwellCompiler",
    "Maxwell3DCompiler",
    "NavierStokes2DCompiler",
    "SchrodingerCompiler",
    "DiffusionCompiler",
    "VlasovPoissonCompiler",
    "ALL_COMPILERS",
    "MULTI_DIM_COMPILERS",
]
