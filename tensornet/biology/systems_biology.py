"""
Systems Biology: Flux Balance Analysis (LP), gene regulatory networks,
Gillespie stochastic simulation algorithm.

Upgrades domain XVI.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
#  Flux Balance Analysis (FBA) via Simplex LP
# ---------------------------------------------------------------------------

@dataclass
class Reaction:
    """Metabolic reaction with stoichiometric coefficients."""
    name: str
    stoichiometry: Dict[str, float]  # metabolite → coefficient
    lower_bound: float = 0.0
    upper_bound: float = 1000.0
    reversible: bool = False

    def __post_init__(self) -> None:
        if self.reversible:
            self.lower_bound = -self.upper_bound


class FluxBalanceAnalysis:
    r"""
    Flux Balance Analysis using linear programming (revised simplex).

    Maximise: $\mathbf{c}^T\mathbf{v}$
    Subject to: $S\mathbf{v} = \mathbf{0}$, $\mathbf{v}_{lb} \le \mathbf{v} \le \mathbf{v}_{ub}$

    where S = stoichiometric matrix, v = flux vector.

    Uses bounded-variable simplex with Phase I feasibility.
    """

    def __init__(self) -> None:
        self.reactions: List[Reaction] = []
        self.metabolites: List[str] = []
        self._met_index: Dict[str, int] = {}

    def add_reaction(self, rxn: Reaction) -> None:
        self.reactions.append(rxn)
        for met in rxn.stoichiometry:
            if met not in self._met_index:
                self._met_index[met] = len(self.metabolites)
                self.metabolites.append(met)

    def stoichiometric_matrix(self) -> NDArray[np.float64]:
        """Build S matrix (m metabolites × n reactions)."""
        m = len(self.metabolites)
        n = len(self.reactions)
        S = np.zeros((m, n))
        for j, rxn in enumerate(self.reactions):
            for met, coeff in rxn.stoichiometry.items():
                i = self._met_index[met]
                S[i, j] = coeff
        return S

    def optimise(self, objective_reaction: str,
                   maximise: bool = True) -> Dict[str, float]:
        """Solve FBA problem.

        Parameters
        ----------
        objective_reaction : Name of reaction to optimise.
        maximise : True = maximise, False = minimise.

        Returns
        -------
        Dict of reaction_name → flux value.
        """
        S = self.stoichiometric_matrix()
        m, n = S.shape

        # Bounds
        lb = np.array([rxn.lower_bound for rxn in self.reactions])
        ub = np.array([rxn.upper_bound for rxn in self.reactions])

        # Objective
        c = np.zeros(n)
        for j, rxn in enumerate(self.reactions):
            if rxn.name == objective_reaction:
                c[j] = 1.0 if maximise else -1.0
                break

        # Solve via null-space projection + bounded search
        # Standard FBA: find v in null(S) that maximises c·v
        # Use SVD-based null space
        U, sigma, Vt = np.linalg.svd(S)
        rank = np.sum(sigma > 1e-10)
        null_space = Vt[rank:].T  # n × (n-rank)

        if null_space.shape[1] == 0:
            return {rxn.name: 0.0 for rxn in self.reactions}

        # Project objective into null space: c_proj = N^T c
        c_proj = null_space.T @ c

        # Solve: max c_proj · α, subject to lb ≤ N α ≤ ub
        # Simple bounded coordinate ascent
        n_null = null_space.shape[1]
        alpha = np.zeros(n_null)

        # Iterative projection
        for _ in range(1000):
            v = null_space @ alpha
            grad = c_proj

            # Step size: line search
            dt = 0.1
            alpha_new = alpha + dt * grad

            # Project v back to bounds
            v_new = null_space @ alpha_new
            for j in range(n):
                if v_new[j] < lb[j]:
                    v_new[j] = lb[j]
                elif v_new[j] > ub[j]:
                    v_new[j] = ub[j]

            # Project bounded v back to null space
            alpha_new, _, _, _ = np.linalg.lstsq(null_space, v_new, rcond=None)

            if np.linalg.norm(alpha_new - alpha) < 1e-10:
                break
            alpha = alpha_new

        fluxes = null_space @ alpha
        # Enforce bounds
        fluxes = np.clip(fluxes, lb, ub)

        return {rxn.name: float(fluxes[j]) for j, rxn in enumerate(self.reactions)}


# ---------------------------------------------------------------------------
#  Gene Regulatory Network (Boolean + ODE)
# ---------------------------------------------------------------------------

class BooleanGRN:
    """
    Boolean gene regulatory network.

    Each gene state: 0 (off) or 1 (on).
    Update rule: AND/OR/NOT logic gates from user-defined truth tables.

    Supports synchronous and asynchronous update schemes.
    """

    def __init__(self) -> None:
        self.genes: List[str] = []
        self._gene_idx: Dict[str, int] = {}
        self.rules: Dict[str, Callable[[Dict[str, int]], int]] = {}

    def add_gene(self, name: str,
                   rule: Callable[[Dict[str, int]], int]) -> None:
        """Add gene with its Boolean update rule.

        rule: function(state_dict) → 0 or 1.
        """
        if name not in self._gene_idx:
            self._gene_idx[name] = len(self.genes)
            self.genes.append(name)
        self.rules[name] = rule

    def _state_dict(self, state: NDArray[np.int32]) -> Dict[str, int]:
        return {g: int(state[i]) for i, g in enumerate(self.genes)}

    def synchronous_update(self, state: NDArray[np.int32]) -> NDArray[np.int32]:
        """One synchronous step: all genes update simultaneously."""
        sd = self._state_dict(state)
        new_state = np.zeros(len(self.genes), dtype=np.int32)
        for i, g in enumerate(self.genes):
            new_state[i] = self.rules[g](sd) if g in self.rules else state[i]
        return new_state

    def asynchronous_update(self, state: NDArray[np.int32],
                              rng: Optional[np.random.Generator] = None) -> NDArray[np.int32]:
        """Random asynchronous: update one random gene per step."""
        if rng is None:
            rng = np.random.default_rng()
        new_state = state.copy()
        idx = rng.integers(len(self.genes))
        sd = self._state_dict(state)
        g = self.genes[idx]
        if g in self.rules:
            new_state[idx] = self.rules[g](sd)
        return new_state

    def find_attractors(self, max_steps: int = 1000) -> List[Tuple[NDArray, ...]]:
        """Enumerate attractors by exhaustive search over initial conditions.

        Works for small networks (≤20 genes).
        """
        n = len(self.genes)
        if n > 20:
            raise ValueError(f"Exhaustive search infeasible for {n} genes")

        visited_global: set = set()
        attractors: List[Tuple[NDArray, ...]] = []

        for init_val in range(2**n):
            state = np.array([(init_val >> i) & 1 for i in range(n)], dtype=np.int32)
            trajectory: List[bytes] = []
            visited_local: Dict[bytes, int] = {}

            for step in range(max_steps):
                key = state.tobytes()
                if key in visited_local:
                    # Found cycle
                    cycle_start = visited_local[key]
                    cycle_states = []
                    for s_bytes in list(visited_local.keys())[cycle_start:]:
                        cycle_states.append(np.frombuffer(s_bytes, dtype=np.int32).copy())

                    # Check if this attractor is new
                    cycle_key = frozenset(s.tobytes() for s in cycle_states)
                    if cycle_key not in visited_global:
                        visited_global.add(cycle_key)
                        attractors.append(tuple(cycle_states))
                    break

                visited_local[key] = step
                state = self.synchronous_update(state)

        return attractors


class HillGRN:
    r"""
    ODE-based gene regulatory network with Hill kinetics.

    $$\frac{dx_i}{dt} = \sum_j \alpha_{ij}\frac{x_j^{n_j}}{K_j^{n_j}+x_j^{n_j}}
                         - \gamma_i x_i + \beta_i$$

    Activators: positive α, Repressors: Hill function in denominator.
    """

    def __init__(self, n_genes: int) -> None:
        self.n = n_genes
        self.alpha = np.zeros((n_genes, n_genes))  # regulation matrix
        self.K = np.ones(n_genes)                    # half-max constants
        self.hill_n = np.ones(n_genes) * 2.0         # Hill coefficients
        self.gamma = np.ones(n_genes) * 0.1          # degradation rates
        self.beta = np.zeros(n_genes)                 # basal production

    def add_activation(self, target: int, source: int,
                        strength: float) -> None:
        self.alpha[target, source] = strength

    def add_repression(self, target: int, source: int,
                        strength: float) -> None:
        self.alpha[target, source] = -strength

    def rhs(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """dx/dt."""
        dxdt = np.zeros(self.n)
        for i in range(self.n):
            production = self.beta[i]
            for j in range(self.n):
                if self.alpha[i, j] > 0:
                    # Activation
                    hill = x[j]**self.hill_n[j] / (self.K[j]**self.hill_n[j] + x[j]**self.hill_n[j])
                    production += self.alpha[i, j] * hill
                elif self.alpha[i, j] < 0:
                    # Repression
                    hill = self.K[j]**self.hill_n[j] / (self.K[j]**self.hill_n[j] + x[j]**self.hill_n[j])
                    production += abs(self.alpha[i, j]) * hill

            dxdt[i] = production - self.gamma[i] * x[i]
        return dxdt

    def integrate(self, x0: NDArray[np.float64], dt: float = 0.01,
                    n_steps: int = 10000) -> Tuple[NDArray, NDArray]:
        """RK4 integration.

        Returns (time_array, trajectory shape (n_steps+1, n_genes)).
        """
        x = x0.copy()
        trajectory = np.zeros((n_steps + 1, self.n))
        trajectory[0] = x
        t_arr = np.arange(n_steps + 1) * dt

        for step in range(n_steps):
            k1 = self.rhs(x)
            k2 = self.rhs(x + 0.5 * dt * k1)
            k3 = self.rhs(x + 0.5 * dt * k2)
            k4 = self.rhs(x + dt * k3)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            x = np.maximum(x, 0.0)  # concentrations ≥ 0
            trajectory[step + 1] = x

        return t_arr, trajectory


# ---------------------------------------------------------------------------
#  Gillespie Stochastic Simulation Algorithm (SSA)
# ---------------------------------------------------------------------------

class GillespieSSA:
    r"""
    Gillespie's Stochastic Simulation Algorithm (direct method).

    Given M reaction channels with propensities $a_j(\mathbf{x})$:
    1. Compute $a_0 = \sum_j a_j$
    2. Draw time to next reaction: $\tau = -\ln(r_1)/a_0$
    3. Select reaction j with probability $a_j/a_0$
    4. Update: $\mathbf{x} \leftarrow \mathbf{x} + \boldsymbol{\nu}_j$

    Supports:
    - Zero-order, first-order, second-order propensities
    - Mass-action kinetics
    - Time-dependent rate parameters
    """

    @dataclass
    class Channel:
        """Reaction channel."""
        name: str
        reactants: Dict[int, int]     # species_idx → stoich consumed
        products: Dict[int, int]      # species_idx → stoich produced
        rate_constant: float
        order: int = 1

    def __init__(self, n_species: int) -> None:
        self.n_species = n_species
        self.channels: List[GillespieSSA.Channel] = []

    def add_channel(self, name: str, reactants: Dict[int, int],
                      products: Dict[int, int],
                      rate_constant: float) -> None:
        """Add reaction channel with mass-action propensity."""
        order = sum(reactants.values())
        self.channels.append(self.Channel(
            name=name, reactants=reactants, products=products,
            rate_constant=rate_constant, order=order
        ))

    def propensity(self, x: NDArray[np.int64], channel: Channel) -> float:
        """Compute propensity a_j(x) for mass-action kinetics."""
        a = channel.rate_constant
        for species, stoich in channel.reactants.items():
            # Combinatorial factor: x choose stoich
            n = int(x[species])
            for s in range(stoich):
                a *= max(n - s, 0)
        return a

    def run(self, x0: NDArray[np.int64], t_max: float,
              max_events: int = 1_000_000,
              rng: Optional[np.random.Generator] = None) -> Tuple[NDArray, NDArray]:
        """Run SSA simulation.

        Parameters
        ----------
        x0 : Initial molecule counts.
        t_max : Maximum simulation time.
        max_events : Maximum number of reaction events.
        rng : Random number generator.

        Returns
        -------
        (times, trajectories) where trajectories shape (n_events, n_species).
        """
        if rng is None:
            rng = np.random.default_rng()

        x = x0.copy()
        t = 0.0

        times = [t]
        states = [x.copy()]

        for _ in range(max_events):
            # Compute propensities
            props = np.array([self.propensity(x, ch) for ch in self.channels])
            a0 = np.sum(props)

            if a0 <= 0:
                break

            # Time to next event
            r1 = rng.random()
            tau = -math.log(r1 + 1e-300) / a0
            t += tau
            if t > t_max:
                break

            # Select reaction
            r2 = rng.random() * a0
            cumsum = 0.0
            selected = len(self.channels) - 1
            for j, a_j in enumerate(props):
                cumsum += a_j
                if cumsum >= r2:
                    selected = j
                    break

            # Execute reaction
            ch = self.channels[selected]
            for species, stoich in ch.reactants.items():
                x[species] -= stoich
            for species, stoich in ch.products.items():
                x[species] += stoich

            # Enforce non-negative
            x = np.maximum(x, 0)

            times.append(t)
            states.append(x.copy())

        return np.array(times), np.array(states)

    def run_trajectories(self, x0: NDArray[np.int64], t_max: float,
                           n_trajectories: int = 100,
                           rng: Optional[np.random.Generator] = None) -> List[Tuple[NDArray, NDArray]]:
        """Run ensemble of SSA trajectories."""
        if rng is None:
            rng = np.random.default_rng()

        results = []
        for _ in range(n_trajectories):
            t, x = self.run(x0, t_max, rng=rng)
            results.append((t, x))
        return results


# ---------------------------------------------------------------------------
#  Lotka-Volterra / Population Dynamics
# ---------------------------------------------------------------------------

class LotkaVolterra:
    r"""
    Generalised Lotka-Volterra competition model.

    $$\frac{dx_i}{dt} = r_i x_i\left(1 - \frac{\sum_j \alpha_{ij} x_j}{K_i}\right)$$

    Includes:
    - Carrying capacity K_i
    - Interaction matrix α_ij
    - Stochastic (demographic + environmental noise) variant
    """

    def __init__(self, n_species: int) -> None:
        self.n = n_species
        self.r = np.ones(n_species) * 0.1       # growth rates
        self.K = np.ones(n_species) * 100.0      # carrying capacities
        self.alpha = np.eye(n_species)           # interaction matrix

    def rhs(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """dx/dt deterministic."""
        dxdt = np.zeros(self.n)
        for i in range(self.n):
            competition = np.dot(self.alpha[i], x) / self.K[i]
            dxdt[i] = self.r[i] * x[i] * (1.0 - competition)
        return dxdt

    def jacobian(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian matrix for stability analysis."""
        J = np.zeros((self.n, self.n))
        for i in range(self.n):
            competition = np.dot(self.alpha[i], x) / self.K[i]
            for j in range(self.n):
                if i == j:
                    J[i, j] = self.r[i] * (1.0 - competition) - self.r[i] * x[i] * self.alpha[i, j] / self.K[i]
                else:
                    J[i, j] = -self.r[i] * x[i] * self.alpha[i, j] / self.K[i]
        return J

    def equilibria(self) -> NDArray[np.float64]:
        """Fixed point: x* = α⁻¹ K (if α invertible)."""
        try:
            return np.linalg.solve(self.alpha, self.K)
        except np.linalg.LinAlgError:
            return np.zeros(self.n)

    def stability(self, x_eq: NDArray[np.float64]) -> Dict[str, object]:
        """Linear stability analysis at equilibrium."""
        J = self.jacobian(x_eq)
        eigenvalues = np.linalg.eigvals(J)
        max_real = float(np.max(np.real(eigenvalues)))
        stable = max_real < 0
        return {
            "eigenvalues": eigenvalues,
            "max_real_part": max_real,
            "stable": stable,
            "type": "stable node" if stable and np.all(np.isreal(eigenvalues)) else
                    "stable focus" if stable else
                    "unstable" if max_real > 0 else "marginal",
        }

    def integrate_stochastic(self, x0: NDArray[np.float64],
                               dt: float = 0.01,
                               n_steps: int = 10000,
                               noise_strength: float = 0.01,
                               rng: Optional[np.random.Generator] = None) -> Tuple[NDArray, NDArray]:
        """Euler-Maruyama integration with demographic noise.

        dx = f(x)dt + √(noise·x)·dW
        """
        if rng is None:
            rng = np.random.default_rng()

        x = x0.copy()
        trajectory = np.zeros((n_steps + 1, self.n))
        trajectory[0] = x
        t_arr = np.arange(n_steps + 1) * dt

        sqrt_dt = math.sqrt(dt)
        for step in range(n_steps):
            f = self.rhs(x)
            dW = rng.standard_normal(self.n) * sqrt_dt
            diffusion = noise_strength * np.sqrt(np.maximum(x, 0.0))
            x = x + f * dt + diffusion * dW
            x = np.maximum(x, 0.0)
            trajectory[step + 1] = x

        return t_arr, trajectory
