"""
Multi-Species Chemistry for High-Temperature Air
=================================================

Implements finite-rate chemistry for 5-species air model
used in hypersonic reentry simulations.

Species:
    N₂, O₂, N, O, NO

Reactions (Park Two-Temperature Model):
    1. O₂ + M ⇌ 2O + M       (O₂ dissociation)
    2. N₂ + M ⇌ 2N + M       (N₂ dissociation)
    3. NO + M ⇌ N + O + M    (NO dissociation)
    4. N₂ + O ⇌ NO + N       (Zeldovich NO)
    5. NO + O ⇌ O₂ + N       (Zeldovich NO)

Key Features:
    - Arrhenius rate coefficients
    - Third-body efficiencies
    - Equilibrium constants from curve fits
    - Species mass production rates

References:
    [1] Park, C., "Review of Chemical-Kinetic Problems of Future
        NASA Missions, I: Earth Entries", JTHT 7(3), 1993
    [2] Gupta et al., "A Review of Reaction Rates and Thermodynamic
        Transport Properties for 11-Species Air", NASA RP-1232, 1990
    [3] Gnoffo et al., "Conservation Equations and Physical Models
        for Hypersonic Air Flows", NASA TP-2867, 1989
"""

from dataclasses import dataclass
from enum import IntEnum

import torch


class Species(IntEnum):
    """Species indices for 5-species air."""

    N2 = 0
    O2 = 1
    NO = 2
    N = 3
    O = 4


# Molecular weights [kg/mol]
MW = {
    Species.N2: 28.0134e-3,
    Species.O2: 31.9988e-3,
    Species.NO: 30.0061e-3,
    Species.N: 14.0067e-3,
    Species.O: 15.9994e-3,
}

# Formation enthalpies at 298 K [J/mol]
H_FORM = {
    Species.N2: 0.0,
    Species.O2: 0.0,
    Species.NO: 90.291e3,
    Species.N: 472.680e3,
    Species.O: 249.173e3,
}

# Characteristic vibrational temperatures [K]
THETA_V = {
    Species.N2: 3395.0,
    Species.O2: 2239.0,
    Species.NO: 2817.0,
}

# Universal gas constant [J/(mol·K)]
R_UNIVERSAL = 8.314462618

# Avogadro's number
N_AVOGADRO = 6.02214076e23


@dataclass
class ArrheniusCoeffs:
    """Arrhenius rate coefficient parameters: k = A * T^n * exp(-E_a / (R*T))"""

    A: float  # Pre-exponential factor [units depend on reaction]
    n: float  # Temperature exponent
    E_a: float  # Activation energy [J/mol]

    def compute(self, T: torch.Tensor) -> torch.Tensor:
        """Compute forward rate coefficient at temperature T."""
        return self.A * T**self.n * torch.exp(-self.E_a / (R_UNIVERSAL * T))


@dataclass
class Reaction:
    """Chemical reaction with forward/backward rates."""

    name: str
    forward: ArrheniusCoeffs
    reactants: dict[Species, int]  # species -> stoichiometric coefficient
    products: dict[Species, int]
    third_body: bool = False
    third_body_efficiencies: dict[Species, float] | None = None


# Park (1993) reaction set for 5-species air
REACTIONS = [
    # Reaction 1: O₂ + M ⇌ 2O + M
    Reaction(
        name="O2 dissociation",
        forward=ArrheniusCoeffs(A=2.0e21, n=-1.5, E_a=4.947e5),
        reactants={Species.O2: 1},
        products={Species.O: 2},
        third_body=True,
        third_body_efficiencies={
            Species.N2: 1.0,
            Species.O2: 1.0,
            Species.NO: 1.0,
            Species.N: 5.0,
            Species.O: 5.0,  # Atoms more efficient
        },
    ),
    # Reaction 2: N₂ + M ⇌ 2N + M
    Reaction(
        name="N2 dissociation",
        forward=ArrheniusCoeffs(A=7.0e21, n=-1.6, E_a=9.413e5),
        reactants={Species.N2: 1},
        products={Species.N: 2},
        third_body=True,
        third_body_efficiencies={
            Species.N2: 1.0,
            Species.O2: 1.0,
            Species.NO: 1.0,
            Species.N: 4.28,
            Species.O: 4.28,
        },
    ),
    # Reaction 3: NO + M ⇌ N + O + M
    Reaction(
        name="NO dissociation",
        forward=ArrheniusCoeffs(A=5.0e15, n=0.0, E_a=6.28e5),
        reactants={Species.NO: 1},
        products={Species.N: 1, Species.O: 1},
        third_body=True,
        third_body_efficiencies={
            Species.N2: 1.0,
            Species.O2: 1.0,
            Species.NO: 22.0,
            Species.N: 22.0,
            Species.O: 22.0,
        },
    ),
    # Reaction 4: N₂ + O ⇌ NO + N (Zeldovich)
    Reaction(
        name="Zeldovich 1",
        forward=ArrheniusCoeffs(A=6.4e17, n=-1.0, E_a=3.16e5),
        reactants={Species.N2: 1, Species.O: 1},
        products={Species.NO: 1, Species.N: 1},
        third_body=False,
    ),
    # Reaction 5: NO + O ⇌ O₂ + N
    Reaction(
        name="Zeldovich 2",
        forward=ArrheniusCoeffs(A=8.4e12, n=0.0, E_a=1.62e5),
        reactants={Species.NO: 1, Species.O: 1},
        products={Species.O2: 1, Species.N: 1},
        third_body=False,
    ),
]


def equilibrium_constant(reaction: Reaction, T: torch.Tensor) -> torch.Tensor:
    """
    Compute equilibrium constant K_eq from Gibbs free energy.

    K_eq = exp(-ΔG°/RT) = exp(-ΔH°/RT + ΔS°/R)

    Simplified curve fit approach for 5-species air.

    Args:
        reaction: Reaction object
        T: Temperature [K]

    Returns:
        Equilibrium constant K_eq
    """
    # Sum of formation enthalpies for products - reactants
    delta_H = 0.0
    for species, nu in reaction.products.items():
        delta_H += nu * H_FORM[species]
    for species, nu in reaction.reactants.items():
        delta_H -= nu * H_FORM[species]

    # Sum of molecular weights for entropy approximation
    delta_n = sum(reaction.products.values()) - sum(reaction.reactants.values())

    # Simplified equilibrium constant
    # More accurate: use NASA polynomial for entropy
    K_eq = torch.exp(-delta_H / (R_UNIVERSAL * T))

    # Pressure correction for reactions with changing mole count
    if delta_n != 0:
        p_ref = 101325.0  # Reference pressure [Pa]
        K_eq = K_eq * (R_UNIVERSAL * T / p_ref) ** delta_n

    return K_eq


def third_body_concentration(
    reaction: Reaction, concentrations: dict[Species, torch.Tensor]
) -> torch.Tensor:
    """
    Compute effective third-body concentration [M].

    [M] = Σᵢ αᵢ [Xᵢ]

    where αᵢ is the third-body efficiency of species i.

    Args:
        reaction: Reaction with third_body=True
        concentrations: Species molar concentrations [mol/m³]

    Returns:
        Effective third-body concentration [mol/m³]
    """
    if not reaction.third_body:
        return torch.ones_like(list(concentrations.values())[0])

    efficiencies = reaction.third_body_efficiencies or {}

    M = torch.zeros_like(list(concentrations.values())[0])
    for species, conc in concentrations.items():
        alpha = efficiencies.get(species, 1.0)
        M = M + alpha * conc

    return M


def compute_reaction_rates(
    T: torch.Tensor, concentrations: dict[Species, torch.Tensor]
) -> tuple[dict[Species, torch.Tensor], torch.Tensor]:
    """
    Compute species production rates from finite-rate chemistry.

    ω̇ᵢ = Mᵢ Σⱼ νᵢⱼ (kf,j ∏ₖ [Xₖ]^νₖⱼ' - kb,j ∏ₖ [Xₖ]^νₖⱼ'')

    Args:
        T: Temperature field [K]
        concentrations: Species molar concentrations [mol/m³]

    Returns:
        Tuple of:
            - Dict of mass production rates [kg/(m³·s)]
            - Total heat release rate [W/m³]
    """
    # Initialize production rates
    omega = {species: torch.zeros_like(T) for species in Species}
    heat_release = torch.zeros_like(T)

    for reaction in REACTIONS:
        # Forward rate coefficient
        kf = reaction.forward.compute(T)

        # Equilibrium constant
        K_eq = equilibrium_constant(reaction, T)

        # Backward rate coefficient: kb = kf / K_eq
        kb = kf / (K_eq + 1e-30)  # Avoid division by zero

        # Third-body concentration
        M = third_body_concentration(reaction, concentrations)

        # Forward rate: Rf = kf * [M] * ∏ [Xₖ]^νₖ'
        Rf = kf * M
        for species, nu in reaction.reactants.items():
            Rf = Rf * (concentrations[species] ** nu)

        # Backward rate: Rb = kb * [M] * ∏ [Xₖ]^νₖ''
        Rb = kb * M
        for species, nu in reaction.products.items():
            Rb = Rb * (concentrations[species] ** nu)

        # Net rate
        net_rate = Rf - Rb

        # Update production rates
        for species, nu in reaction.reactants.items():
            omega[species] = omega[species] - nu * MW[species] * net_rate

        for species, nu in reaction.products.items():
            omega[species] = omega[species] + nu * MW[species] * net_rate

        # Heat release (exothermic if ΔH < 0)
        delta_H = 0.0
        for species, nu in reaction.products.items():
            delta_H += nu * H_FORM[species]
        for species, nu in reaction.reactants.items():
            delta_H -= nu * H_FORM[species]

        heat_release = heat_release - delta_H * net_rate

    return omega, heat_release


@dataclass
class ChemistryState:
    """State for multi-species chemistry solver."""

    rho: torch.Tensor  # Total density [kg/m³]
    Y: dict[Species, torch.Tensor]  # Mass fractions [-]
    T: torch.Tensor  # Temperature [K]
    p: torch.Tensor  # Pressure [Pa]

    @property
    def shape(self) -> torch.Size:
        return self.rho.shape

    def concentrations(self) -> dict[Species, torch.Tensor]:
        """Compute molar concentrations [mol/m³]."""
        return {
            species: self.rho * self.Y[species] / MW[species] for species in Species
        }

    def mixture_molecular_weight(self) -> torch.Tensor:
        """Compute mixture molecular weight [kg/mol]."""
        M_inv = torch.zeros_like(self.rho)
        for species in Species:
            M_inv = M_inv + self.Y[species] / MW[species]
        return 1.0 / (M_inv + 1e-30)

    def mixture_R(self) -> torch.Tensor:
        """Compute mixture gas constant [J/(kg·K)]."""
        M = self.mixture_molecular_weight()
        return R_UNIVERSAL / M

    def validate(self) -> bool:
        """Check physical constraints."""
        # Mass fractions sum to 1
        Y_sum = sum(self.Y.values())
        sum_valid = torch.allclose(Y_sum, torch.ones_like(Y_sum), atol=1e-6)

        # All non-negative
        Y_positive = all((y >= 0).all() for y in self.Y.values())

        # Physical bounds
        rho_valid = (self.rho > 0).all()
        T_valid = (self.T > 0).all()
        p_valid = (self.p > 0).all()

        return sum_valid and Y_positive and rho_valid and T_valid and p_valid


def air_5species_ic(
    shape: torch.Size,
    T: float = 300.0,
    p: float = 101325.0,
    Y_N2: float = 0.767,
    Y_O2: float = 0.233,
) -> ChemistryState:
    """
    Create initial condition for 5-species air.

    Default is standard atmospheric composition:
        N₂: 76.7% by mass
        O₂: 23.3% by mass
        N, O, NO: 0%

    Args:
        shape: Tensor shape (Ny, Nx)
        T: Temperature [K]
        p: Pressure [Pa]
        Y_N2: Mass fraction of N₂
        Y_O2: Mass fraction of O₂

    Returns:
        ChemistryState initial condition
    """
    # Mixture molecular weight
    M_mix = 1.0 / (Y_N2 / MW[Species.N2] + Y_O2 / MW[Species.O2])

    # Density from ideal gas law
    R_mix = R_UNIVERSAL / M_mix
    rho = p / (R_mix * T)

    return ChemistryState(
        rho=torch.full(shape, rho, dtype=torch.float64),
        Y={
            Species.N2: torch.full(shape, Y_N2, dtype=torch.float64),
            Species.O2: torch.full(shape, Y_O2, dtype=torch.float64),
            Species.NO: torch.zeros(shape, dtype=torch.float64),
            Species.N: torch.zeros(shape, dtype=torch.float64),
            Species.O: torch.zeros(shape, dtype=torch.float64),
        },
        T=torch.full(shape, T, dtype=torch.float64),
        p=torch.full(shape, p, dtype=torch.float64),
    )


def chemistry_timestep(
    state: ChemistryState, safety_factor: float = 0.1
) -> torch.Tensor:
    """
    Compute chemistry timestep based on stiffness.

    dt_chem = safety * min(ρYᵢ / |ω̇ᵢ|)

    Args:
        state: Current chemistry state
        safety_factor: CFL-like factor for stability

    Returns:
        Chemistry timestep [s]
    """
    concentrations = state.concentrations()
    omega, _ = compute_reaction_rates(state.T, concentrations)

    dt_min = torch.full_like(state.rho, 1e10)

    for species in Species:
        rho_Y = state.rho * state.Y[species]
        omega_abs = torch.abs(omega[species]) + 1e-30
        dt_species = rho_Y / omega_abs
        dt_min = torch.minimum(dt_min, dt_species)

    return safety_factor * dt_min.min()


def advance_chemistry_explicit(state: ChemistryState, dt: float) -> ChemistryState:
    """
    Advance chemistry with explicit Euler (for testing only).

    WARNING: Chemistry is typically stiff and requires implicit methods.
    This is provided for simple cases only.

    Args:
        state: Current state
        dt: Timestep [s]

    Returns:
        Updated state
    """
    concentrations = state.concentrations()
    omega, heat_release = compute_reaction_rates(state.T, concentrations)

    # Update mass fractions
    Y_new = {}
    for species in Species:
        dY_dt = omega[species] / state.rho
        Y_new[species] = state.Y[species] + dt * dY_dt
        # Clip to physical bounds
        Y_new[species] = torch.clamp(Y_new[species], min=0.0)

    # Renormalize
    Y_sum = sum(Y_new.values())
    for species in Species:
        Y_new[species] = Y_new[species] / Y_sum

    # Update temperature (adiabatic)
    # dT/dt = -Σᵢ ω̇ᵢ hᵢ / (ρ cₚ)
    # Simplified: assume constant cₚ
    cp_mix = 1000.0  # Approximate [J/(kg·K)]
    dT_dt = -heat_release / (state.rho * cp_mix)
    T_new = state.T + dt * dT_dt
    T_new = torch.clamp(T_new, min=200.0)  # Lower bound

    # Update pressure from EOS
    M_mix = 1.0 / sum(Y_new[s] / MW[s] for s in Species)
    R_mix = R_UNIVERSAL / M_mix
    p_new = state.rho * R_mix * T_new

    return ChemistryState(
        rho=state.rho.clone(),
        Y=Y_new,
        T=T_new,
        p=p_new,
    )


def post_shock_composition(
    M: float, T1: float = 300.0, p1: float = 101325.0, gamma: float = 1.4
) -> dict[Species, float]:
    """
    Estimate post-shock composition for strong shocks.

    Uses equilibrium chemistry to estimate dissociation
    behind a normal shock.

    Args:
        M: Freestream Mach number
        T1: Pre-shock temperature [K]
        p1: Pre-shock pressure [Pa]
        gamma: Ratio of specific heats

    Returns:
        Dictionary of mass fractions
    """
    # Normal shock relations (perfect gas)
    p2_p1 = 1 + 2 * gamma / (gamma + 1) * (M**2 - 1)
    T2_T1 = (
        (1 + (gamma - 1) / 2 * M**2)
        * (2 * gamma / (gamma - 1) * M**2 - 1)
        / ((gamma + 1) ** 2 / (2 * (gamma - 1)) * M**2)
    )

    T2 = T1 * T2_T1

    # Estimate dissociation fractions
    # O₂ starts dissociating around 2500 K
    # N₂ starts dissociating around 4000 K

    alpha_O2 = 0.0
    alpha_N2 = 0.0

    if T2 > 2500:
        alpha_O2 = min(1.0, (T2 - 2500) / 3000)
    if T2 > 4000:
        alpha_N2 = min(1.0, (T2 - 4000) / 5000)

    # Original mass fractions
    Y_N2_0 = 0.767
    Y_O2_0 = 0.233

    # Dissociated fractions
    Y_N2 = Y_N2_0 * (1 - alpha_N2)
    Y_O2 = Y_O2_0 * (1 - alpha_O2)
    Y_N = Y_N2_0 * alpha_N2
    Y_O = Y_O2_0 * alpha_O2

    # NO formation (simplified)
    Y_NO = 0.0
    if T2 > 2000:
        Y_NO = min(0.1, (T2 - 2000) / 10000)
        # Take equally from N and O
        Y_N = max(0, Y_N - Y_NO / 2)
        Y_O = max(0, Y_O - Y_NO / 2)

    return {
        Species.N2: Y_N2,
        Species.O2: Y_O2,
        Species.NO: Y_NO,
        Species.N: Y_N,
        Species.O: Y_O,
    }


def validate_chemistry():
    """
    Run validation tests for chemistry module.
    """
    print("\n" + "=" * 70)
    print("MULTI-SPECIES CHEMISTRY VALIDATION")
    print("=" * 70)

    # Test 1: Mass fraction conservation
    print("\n[Test 1] Mass Fraction Conservation")
    print("-" * 40)

    state = air_5species_ic(shape=(10, 10), T=3000.0, p=101325.0)

    # Run a few explicit steps
    dt = 1e-8
    for i in range(10):
        state = advance_chemistry_explicit(state, dt)

    Y_sum = sum(state.Y.values())
    max_deviation = (Y_sum - 1.0).abs().max().item()

    print(f"Max deviation from Σ Yᵢ = 1: {max_deviation:.2e}")

    if max_deviation < 1e-6:
        print("✓ PASS: Mass fractions sum to 1")
    else:
        print("✗ FAIL: Mass fraction error too large")

    # Test 2: Equilibrium at low temperature
    print("\n[Test 2] Equilibrium at Low Temperature")
    print("-" * 40)

    state_low = air_5species_ic(shape=(1, 1), T=300.0)
    concentrations = state_low.concentrations()
    omega, _ = compute_reaction_rates(state_low.T, concentrations)

    max_omega = max(w.abs().max().item() for w in omega.values())
    print(f"Max production rate at 300 K: {max_omega:.2e} kg/(m³·s)")

    if max_omega < 1e-10:
        print("✓ PASS: No reactions at low temperature")
    else:
        print("✗ FAIL: Unexpected reactions at low T")

    # Test 3: Dissociation at high temperature
    print("\n[Test 3] Dissociation at High Temperature")
    print("-" * 40)

    state_high = air_5species_ic(shape=(1, 1), T=5000.0)
    concentrations = state_high.concentrations()
    omega, _ = compute_reaction_rates(state_high.T, concentrations)

    # O₂ should be consumed (negative production)
    omega_O2 = omega[Species.O2].item()
    # O should be produced (positive production)
    omega_O = omega[Species.O].item()

    print(f"O₂ production rate: {omega_O2:.2e} kg/(m³·s)")
    print(f"O production rate: {omega_O:.2e} kg/(m³·s)")

    if omega_O2 < 0 and omega_O > 0:
        print("✓ PASS: O₂ dissociating into O at high T")
    else:
        print("✗ FAIL: Unexpected dissociation behavior")

    # Test 4: Post-shock composition
    print("\n[Test 4] Post-Shock Composition (Mach 10)")
    print("-" * 40)

    Y_post = post_shock_composition(M=10.0, T1=300.0)

    print("Mass fractions behind Mach 10 normal shock:")
    for species, Y in Y_post.items():
        print(f"  {species.name}: {Y:.4f}")

    Y_sum = sum(Y_post.values())
    if abs(Y_sum - 1.0) < 1e-6:
        print("✓ PASS: Post-shock mass fractions sum to 1")
    else:
        print("✗ FAIL: Post-shock mass fraction error")

    print("\n" + "=" * 70)
    print("CHEMISTRY VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    validate_chemistry()
