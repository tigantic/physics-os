"""HyperTensor API — Domain catalog endpoint.

``GET /v1/domains``      → list all available physics domains
``GET /v1/domains/{id}`` → detailed info for a single domain
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..models import (
    DomainInfo,
    DomainListResponse,
    DomainParameter,
    PhysicsDomain,
)

router = APIRouter(prefix="/v1", tags=["domains"])


# ═══════════════════════════════════════════════════════════════════
# Domain catalog (static metadata — no IP-sensitive content)
# ═══════════════════════════════════════════════════════════════════

_CATALOG: dict[str, DomainInfo] = {
    PhysicsDomain.BURGERS: DomainInfo(
        domain="burgers",
        domain_label="Viscous Burgers Equation (1D Navier–Stokes)",
        equation="∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²",
        spatial_dimensions=1,
        fields=["u"],
        conserved_quantity="total_mass",
        parameters=[
            DomainParameter(
                name="viscosity", default=0.01, min=1e-6, max=1.0,
                description="Kinematic viscosity ν.",
            ),
        ],
        example_request={
            "domain": "burgers",
            "resolution": {"n_bits": 8, "n_steps": 100},
            "parameters": {"viscosity": 0.01},
        },
    ),
    PhysicsDomain.MAXWELL_1D: DomainInfo(
        domain="maxwell",
        domain_label="Maxwell Equations (1D TE mode)",
        equation="∂E/∂t = c·∂B/∂x,  ∂B/∂t = c·∂E/∂x",
        spatial_dimensions=1,
        fields=["E", "B"],
        conserved_quantity="em_energy",
        parameters=[
            DomainParameter(
                name="c", default=1.0, min=0.01, max=1e8,
                description="Speed of light / wave speed.",
            ),
        ],
        example_request={
            "domain": "maxwell",
            "resolution": {"n_bits": 8, "n_steps": 100},
            "parameters": {"c": 1.0},
        },
    ),
    PhysicsDomain.MAXWELL_3D: DomainInfo(
        domain="maxwell_3d",
        domain_label="3D Maxwell Equations (full curl)",
        equation="∂E/∂t = c·∇×B,  ∂B/∂t = −c·∇×E",
        spatial_dimensions=3,
        fields=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
        conserved_quantity="em_energy",
        parameters=[
            DomainParameter(
                name="c", default=1.0, min=0.01, max=1e8,
                description="Speed of light / wave speed.",
            ),
        ],
        example_request={
            "domain": "maxwell_3d",
            "resolution": {"n_bits": 4, "n_steps": 50},
            "parameters": {"c": 1.0},
        },
    ),
    PhysicsDomain.SCHRODINGER: DomainInfo(
        domain="schrodinger",
        domain_label="Schrödinger Equation (1D harmonic oscillator)",
        equation="i·∂ψ/∂t = −½·∂²ψ/∂x² + V(x)·ψ",
        spatial_dimensions=1,
        fields=["psi_re", "psi_im"],
        conserved_quantity="total_probability",
        parameters=[
            DomainParameter(
                name="omega", default=4.0, min=0.1, max=100.0,
                description="Harmonic oscillator frequency.",
            ),
            DomainParameter(
                name="x0", default=0.5, min=0.0, max=1.0,
                description="Potential center position.",
            ),
            DomainParameter(
                name="k0", default=10.0, min=0.0, max=100.0,
                description="Initial wave-packet momentum.",
            ),
        ],
        example_request={
            "domain": "schrodinger",
            "resolution": {"n_bits": 8, "n_steps": 200},
            "parameters": {"omega": 4.0, "x0": 0.5, "k0": 10.0},
        },
    ),
    PhysicsDomain.DIFFUSION: DomainInfo(
        domain="advection_diffusion",
        domain_label="Advection–Diffusion (scalar transport)",
        equation="∂u/∂t + v·∂u/∂x = κ·∂²u/∂x²",
        spatial_dimensions=1,
        fields=["u"],
        conserved_quantity="total_mass",
        parameters=[
            DomainParameter(
                name="velocity", default=1.0, min=-100.0, max=100.0,
                description="Constant advection velocity v.",
            ),
            DomainParameter(
                name="diffusivity", default=0.01, min=1e-6, max=10.0,
                description="Diffusion coefficient κ.",
            ),
        ],
        example_request={
            "domain": "advection_diffusion",
            "resolution": {"n_bits": 8, "n_steps": 100},
            "parameters": {"velocity": 1.0, "diffusivity": 0.01},
        },
    ),
    PhysicsDomain.VLASOV_POISSON: DomainInfo(
        domain="vlasov_poisson",
        domain_label="Vlasov–Poisson (1D1V electrostatic plasma)",
        equation="∂f/∂t + v·∂f/∂x + E·∂f/∂v = 0,  ∂²φ/∂x² = −ρ",
        spatial_dimensions=2,
        fields=["f"],
        conserved_quantity="particle_number",
        parameters=[
            DomainParameter(
                name="bits_x", type="int", default=6, min=4, max=10,
                description="Spatial grid bits (grid = 2^bits_x).",
            ),
            DomainParameter(
                name="bits_v", type="int", default=6, min=4, max=10,
                description="Velocity grid bits (grid = 2^bits_v).",
            ),
            DomainParameter(
                name="v_max", default=6.0, min=1.0, max=20.0,
                description="Velocity domain half-width [−v_max, v_max].",
            ),
            DomainParameter(
                name="perturbation", default=0.01, min=1e-6, max=1.0,
                description="Landau damping perturbation amplitude.",
            ),
        ],
        example_request={
            "domain": "vlasov_poisson",
            "resolution": {"n_bits": 6, "n_steps": 50},
            "parameters": {"bits_x": 6, "bits_v": 6, "v_max": 6.0},
        },
    ),
    PhysicsDomain.NAVIER_STOKES_2D: DomainInfo(
        domain="navier_stokes_2d",
        domain_label="2D Navier–Stokes (vorticity–stream function)",
        equation="∂ω/∂t + (u·∇)ω = ν·∇²ω,  ∇²ψ = −ω",
        spatial_dimensions=2,
        fields=["omega", "psi"],
        conserved_quantity="total_vorticity",
        parameters=[
            DomainParameter(
                name="viscosity", default=0.01, min=1e-6, max=1.0,
                description="Kinematic viscosity ν.",
            ),
        ],
        example_request={
            "domain": "navier_stokes_2d",
            "resolution": {"n_bits": 6, "n_steps": 50},
            "parameters": {"viscosity": 0.01},
        },
    ),
}


# ═══════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════


@router.get(
    "/domains",
    response_model=DomainListResponse,
    summary="List available physics domains",
    description="Returns metadata for every physics domain the API can simulate.",
)
async def list_domains() -> DomainListResponse:
    infos = list(_CATALOG.values())
    return DomainListResponse(domains=infos, count=len(infos))


@router.get(
    "/domains/{domain_id}",
    response_model=DomainInfo,
    summary="Get domain details",
    description=(
        "Returns detailed information about a single physics domain, "
        "including available parameters and an example request body."
    ),
)
async def get_domain(domain_id: PhysicsDomain) -> DomainInfo:
    info = _CATALOG.get(domain_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown domain: {domain_id}",
        )
    return info
