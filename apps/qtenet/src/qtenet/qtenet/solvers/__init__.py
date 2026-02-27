"""
QTeneT Solvers — Physics Solvers that Break the Curse

Native N-dimensional PDE solvers operating entirely in QTT format,
achieving O(log N) complexity for problems that are impossible with
traditional methods.

Available Solvers:
    NS3D: 3D Navier-Stokes DNS solver (O(log N) turbulence)
    vlasov_5d: 5D Vlasov-Poisson for plasma physics (x,y,z,vx,vy)
    vlasov_6d: 6D Vlasov-Maxwell — THE Holy Grail
    euler_nd: N-dimensional compressible Euler equations

Example:
    >>> from qtenet.solvers import NS3D
    >>> 
    >>> # 1024^3 = 1 billion cells, ~9MB memory
    >>> solver = NS3D(n_bits=10, max_rank=64)
    >>> state = solver.taylor_green()  # 49ms, no dense allocation
    >>> 
    >>> for _ in range(1000):
    ...     state = solver.step(state, dt=0.001)  # 8ms per step

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from qtenet.solvers.vlasov import (
    Vlasov5D,
    Vlasov5DConfig,
    Vlasov6D,
    Vlasov6DConfig,
    VlasovState,
)
from qtenet.solvers.vlasov6d_genuine import (
    Vlasov6DGenuine,
    Vlasov6DGenuineConfig,
    Vlasov6DGenuineState,
)
from qtenet.solvers.vlasov_genuine import (
    Vlasov1D1V,
    Vlasov1D1VConfig,
    Vlasov1D1VState,
)
from qtenet.solvers.euler import (
    EulerND,
    EulerNDConfig,
    EulerState,
)
from qtenet.solvers.ns3d import (
    NS3D,
    NS3DConfig,
    NS3DState,
    NS3DDiagnostics,
    QTT3DField,
    QTT3DVectorField,
)

__all__ = [
    # Navier-Stokes 3D (Turbulence)
    "NS3D",
    "NS3DConfig",
    "NS3DState",
    "NS3DDiagnostics",
    "QTT3DField",
    "QTT3DVectorField",
    # Vlasov (Phase Space)
    "Vlasov5D",
    "Vlasov5DConfig",
    "Vlasov6D",
    "Vlasov6DConfig",
    "VlasovState",
    # Vlasov 6D Genuine (Velocity-Dependent Transport)
    "Vlasov6DGenuine",
    "Vlasov6DGenuineConfig",
    "Vlasov6DGenuineState",
    # Vlasov 1D+1V Genuine (Validated Reference)
    "Vlasov1D1V",
    "Vlasov1D1VConfig",
    "Vlasov1D1VState",
    # Euler (Compressible Flow)
    "EulerND",
    "EulerNDConfig",
    "EulerState",
]
