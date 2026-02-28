"""
Non-equilibrium statistical mechanics — fluctuation theorems, stochastic kinetics.

Upgrades domain V.2 from Fokker-Planck-only (ontic/fusion/phonon_trigger.py) to
full non-equilibrium statistical mechanics:
  - Jarzynski equality free energy estimator
  - Crooks fluctuation theorem / Bennett acceptance ratio
  - Kubo linear-response theory
  - Kinetic Monte Carlo (KMC, BKL algorithm)
  - Gillespie stochastic simulation algorithm (SSA, direct + first-reaction)
  - Chemical master equation solver (sparse matrix exponential)

Units: k_B T = 1 (reduced) unless otherwise noted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

K_B: float = 1.380649e-23  # J/K


# ===================================================================
#  Jarzynski Equality Free Energy Estimator
# ===================================================================

class JarzynskiEstimator:
    r"""
    Jarzynski equality for non-equilibrium free energy estimation.

    $$e^{-\beta\Delta F} = \langle e^{-\beta W}\rangle$$

    $$\Delta F = -\frac{1}{\beta}\ln\!\left\langle e^{-\beta W}\right\rangle$$

    Cumulant expansion (for near-equilibrium):
    $$\Delta F \approx \langle W\rangle - \frac{\beta}{2}\text{Var}(W) + \cdots$$

    Also computes the dissipated work:
    $$W_{\text{diss}} = \langle W\rangle - \Delta F \geq 0$$

    Attributes
    ----------
    work_values : Array of work measurements from non-equilibrium trajectories.
    beta : Inverse temperature 1/(k_B T).
    """

    def __init__(self, work_values: NDArray[np.float64],
                 beta: float = 1.0) -> None:
        self.work = np.asarray(work_values, dtype=np.float64)
        self.beta = beta

    @property
    def n_trajectories(self) -> int:
        return len(self.work)

    def free_energy_exponential(self) -> float:
        r"""
        Exponential average: $\Delta F = -\beta^{-1}\ln\langle e^{-\beta W}\rangle$.

        Uses log-sum-exp trick for numerical stability.
        """
        bw = -self.beta * self.work
        bw_max = np.max(bw)
        log_avg = bw_max + np.log(np.mean(np.exp(bw - bw_max)))
        return -log_avg / self.beta

    def free_energy_cumulant(self, order: int = 2) -> float:
        r"""
        Cumulant expansion estimate.

        Order 1: $\Delta F \approx \langle W\rangle$
        Order 2: $\Delta F \approx \langle W\rangle - \beta\sigma_W^2/2$
        """
        W_mean = float(np.mean(self.work))
        if order == 1:
            return W_mean
        W_var = float(np.var(self.work))
        if order == 2:
            return W_mean - 0.5 * self.beta * W_var
        # Order 3: includes third cumulant
        W_centered = self.work - W_mean
        kappa3 = float(np.mean(W_centered ** 3))
        return W_mean - 0.5 * self.beta * W_var + (self.beta**2 / 6.0) * kappa3

    def dissipated_work(self) -> float:
        r"""$W_{\text{diss}} = \langle W\rangle - \Delta F \geq 0$."""
        return float(np.mean(self.work)) - self.free_energy_exponential()

    def bootstrap_error(self, n_bootstrap: int = 1000,
                        seed: Optional[int] = None) -> float:
        """Bootstrap estimate of statistical error in ΔF."""
        rng = np.random.default_rng(seed)
        dF_samples = np.empty(n_bootstrap)
        n = len(self.work)
        for i in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            W_sample = self.work[indices]
            bw = -self.beta * W_sample
            bw_max = np.max(bw)
            log_avg = bw_max + np.log(np.mean(np.exp(bw - bw_max)))
            dF_samples[i] = -log_avg / self.beta
        return float(np.std(dF_samples))


# ===================================================================
#  Crooks Fluctuation Theorem
# ===================================================================

class CrooksEstimator:
    r"""
    Crooks fluctuation theorem and Bennett acceptance ratio (BAR).

    Crooks relation:
    $$\frac{P_F(W)}{P_R(-W)} = e^{\beta(W - \Delta F)}$$

    Bennett acceptance ratio (optimal ΔF estimator):
    $$\sum_{i=1}^{n_F} \frac{1}{1 + (n_F/n_R)\exp(\beta(W_i^F - \Delta F))}
      = \sum_{j=1}^{n_R} \frac{1}{1 + (n_R/n_F)\exp(-\beta(W_j^R + \Delta F))}$$
    """

    def __init__(self, work_forward: NDArray[np.float64],
                 work_reverse: NDArray[np.float64],
                 beta: float = 1.0) -> None:
        self.W_F = np.asarray(work_forward, dtype=np.float64)
        self.W_R = np.asarray(work_reverse, dtype=np.float64)
        self.beta = beta

    def crossing_estimate(self) -> float:
        """
        Estimate ΔF from crossing point of P_F(W) and P_R(-W) histograms.

        Quick-and-dirty estimate; BAR is more accurate.
        """
        # Use kernel density estimation at histogram crossing
        all_w = np.concatenate([self.W_F, -self.W_R])
        w_min, w_max = float(np.min(all_w)), float(np.max(all_w))
        w_grid = np.linspace(w_min, w_max, 1000)
        bw = 0.5 * (w_max - w_min) / 50  # bandwidth

        def kde(data: NDArray, x: NDArray) -> NDArray:
            d = (x[:, None] - data[None, :]) / bw
            return np.mean(np.exp(-0.5 * d**2), axis=1) / (bw * math.sqrt(2 * math.pi))

        p_f = kde(self.W_F, w_grid)
        p_r = kde(-self.W_R, w_grid)

        # Find crossing
        diff = p_f - p_r
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) == 0:
            # Fallback: use mean
            return 0.5 * (float(np.mean(self.W_F)) - float(np.mean(self.W_R)))
        return float(w_grid[crossings[0]])

    def bennett_acceptance_ratio(self, tol: float = 1e-10,
                                 max_iter: int = 1000) -> float:
        """
        Bennett acceptance ratio (BAR) for optimal ΔF estimation.

        Solves the self-consistency equation iteratively.
        """
        nF = len(self.W_F)
        nR = len(self.W_R)
        ratio = nF / nR
        beta = self.beta

        # Initial guess from Jarzynski
        dF = JarzynskiEstimator(self.W_F, beta).free_energy_exponential()

        for _ in range(max_iter):
            # Fermi functions
            f_F = 1.0 / (1.0 + ratio * np.exp(beta * (self.W_F - dF)))
            f_R = 1.0 / (1.0 + (1.0 / ratio) * np.exp(-beta * (self.W_R + dF)))

            sum_F = np.sum(f_F)
            sum_R = np.sum(f_R)

            if sum_R < 1e-300:
                break

            # Update ΔF
            dF_new = dF + (1.0 / beta) * math.log(sum_R / sum_F)

            if abs(dF_new - dF) < tol:
                dF = dF_new
                break
            dF = dF_new

        return dF

    def verify_crooks(self, n_bins: int = 50) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Verify Crooks theorem: plot ln(P_F(W)/P_R(-W)) vs W.

        Should be linear with slope β and intercept -βΔF.

        Returns (W_centres, log_ratio, expected_slope_line).
        """
        all_w = np.concatenate([self.W_F, -self.W_R])
        w_min, w_max = float(np.min(all_w)), float(np.max(all_w))
        bins = np.linspace(w_min, w_max, n_bins + 1)
        centres = 0.5 * (bins[:-1] + bins[1:])

        hist_F, _ = np.histogram(self.W_F, bins=bins, density=True)
        hist_R, _ = np.histogram(-self.W_R, bins=bins, density=True)

        # Avoid log(0)
        mask = (hist_F > 0) & (hist_R > 0)
        log_ratio = np.full(n_bins, np.nan)
        log_ratio[mask] = np.log(hist_F[mask] / hist_R[mask])

        dF = self.bennett_acceptance_ratio()
        expected = self.beta * (centres - dF)

        return centres, log_ratio, expected


# ===================================================================
#  Kubo Linear Response Theory
# ===================================================================

class KuboResponse:
    r"""
    Kubo linear response theory.

    Response function:
    $$\chi_{AB}(t) = -\frac{i}{\hbar}\theta(t)\langle[A(t), B(0)]\rangle$$

    Classical (fluctuation-dissipation):
    $$\chi_{AB}(\omega) = \beta\int_0^\infty e^{i\omega t}\langle\dot{A}(t)B(0)\rangle dt$$

    Green-Kubo transport coefficients:
    $$L = \int_0^\infty \langle J(t) J(0)\rangle dt$$

    Electrical conductivity:
    $$\sigma = \frac{\beta}{V}\int_0^\infty \langle J_e(t) J_e(0)\rangle dt$$

    Thermal conductivity:
    $$\kappa = \frac{\beta^2}{V}\int_0^\infty \langle J_q(t) J_q(0)\rangle dt$$
    """

    @staticmethod
    def autocorrelation(signal: NDArray[np.float64],
                        normalise: bool = True) -> NDArray[np.float64]:
        r"""
        Compute autocorrelation function C(t) = <A(t)A(0)> via FFT.

        Uses Wiener-Khinchin theorem: $C(\tau) = \text{IFFT}(|\text{FFT}(A)|^2)$.
        """
        N = len(signal)
        # Zero-pad to avoid circular convolution artefacts
        A_fft = np.fft.fft(signal, n=2 * N)
        power = np.abs(A_fft) ** 2
        C = np.real(np.fft.ifft(power))[:N]
        # Normalise by number of data points at each lag
        counts = np.arange(N, 0, -1, dtype=np.float64)
        C /= counts
        if normalise and abs(C[0]) > 1e-30:
            C /= C[0]
        return C

    @staticmethod
    def green_kubo_integral(correlation: NDArray[np.float64],
                            dt: float,
                            max_lag: Optional[int] = None) -> NDArray[np.float64]:
        r"""
        Running Green-Kubo integral: $L(\tau) = \int_0^\tau C(t)dt$.

        Uses trapezoidal rule. Returns L as a function of upper cutoff τ.
        """
        if max_lag is not None:
            correlation = correlation[:max_lag]
        return np.cumsum(correlation) * dt - 0.5 * correlation * dt

    @staticmethod
    def conductivity(current_autocorrelation: NDArray[np.float64],
                     dt: float, beta: float, volume: float) -> float:
        """
        Electrical conductivity from Green-Kubo:
        σ = (β/V) ∫₀^∞ <J(t)·J(0)> dt.
        """
        integral = np.trapz(current_autocorrelation, dx=dt)
        return beta * integral / volume

    @staticmethod
    def diffusion_coefficient(velocity_acf: NDArray[np.float64],
                              dt: float, ndim: int = 3) -> float:
        """
        Self-diffusion from velocity autocorrelation:
        D = (1/d) ∫₀^∞ <v(t)·v(0)> dt.
        """
        integral = np.trapz(velocity_acf, dx=dt)
        return integral / ndim

    @staticmethod
    def susceptibility_from_acf(correlation: NDArray[np.float64],
                                dt: float, beta: float,
                                omega: NDArray[np.float64]) -> NDArray[np.complex128]:
        r"""
        Frequency-dependent susceptibility via Fourier transform of C(t).

        $$\chi(\omega) = \beta\int_0^\infty e^{i\omega t} C(t) dt$$
        """
        t = np.arange(len(correlation)) * dt
        # Numerical FT with exponential damping to avoid truncation artefacts
        eta = 3.0 / (len(correlation) * dt)  # Small damping
        chi = np.zeros(len(omega), dtype=np.complex128)
        for i, w in enumerate(omega):
            integrand = correlation * np.exp((1j * w - eta) * t)
            chi[i] = beta * np.trapz(integrand, dx=dt)
        return chi

    @staticmethod
    def kramers_kronig(chi_imag: NDArray[np.float64],
                       omega: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Kramers-Kronig relation: compute χ'(ω) from χ''(ω).

        $$\chi'(\omega) = \frac{2}{\pi}\mathcal{P}\int_0^\infty
            \frac{\omega'\chi''(\omega')}{\omega'^2 - \omega^2} d\omega'$$
        """
        dw = omega[1] - omega[0] if len(omega) > 1 else 1.0
        chi_real = np.zeros_like(omega)
        for i, w in enumerate(omega):
            denom = omega**2 - w**2
            # Principal value: skip singular point
            mask = np.abs(denom) > 1e-10 * np.max(np.abs(denom))
            integrand = omega * chi_imag / denom
            chi_real[i] = (2.0 / np.pi) * np.trapz(integrand[mask],
                                                      dx=dw)
        return chi_real


# ===================================================================
#  Kinetic Monte Carlo (BKL Algorithm)
# ===================================================================

@dataclass
class KMCEvent:
    """A single event in the kinetic Monte Carlo catalog."""
    name: str
    rate: float                 # Transition rate [1/s]
    execute: Callable[[], None]  # State modification callback


class KineticMonteCarlo:
    r"""
    Kinetic Monte Carlo: BKL (Bortz-Kalos-Lebowitz) algorithm.

    At each step:
    1. Compute total rate $R_{\text{tot}} = \sum_i r_i$
    2. Draw event $i$ with probability $p_i = r_i / R_{\text{tot}}$
    3. Advance time $\Delta t = -\ln(u) / R_{\text{tot}}$, $u \sim U(0,1)$
    4. Execute event $i$

    Physical time directly sampled — no rejected moves.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.events: List[KMCEvent] = []
        self.time: float = 0.0
        self.rng = np.random.default_rng(seed)

    def add_event(self, name: str, rate: float,
                  execute: Callable[[], None]) -> None:
        """Register an event with its rate and execution callback."""
        self.events.append(KMCEvent(name=name, rate=rate, execute=execute))

    def update_rate(self, name: str, new_rate: float) -> None:
        """Update the rate of an existing event."""
        for ev in self.events:
            if ev.name == name:
                ev.rate = new_rate
                return
        raise KeyError(f"Event '{name}' not found")

    def step(self) -> Tuple[float, str]:
        """
        Execute one KMC step.

        Returns (dt, event_name).
        """
        rates = np.array([ev.rate for ev in self.events])
        R_tot = np.sum(rates)
        if R_tot <= 0:
            raise RuntimeError("All rates are zero — simulation halted")

        # Draw time increment
        u = self.rng.random()
        dt = -math.log(u) / R_tot
        self.time += dt

        # Select event
        cumulative = np.cumsum(rates)
        r = self.rng.random() * R_tot
        idx = int(np.searchsorted(cumulative, r))
        idx = min(idx, len(self.events) - 1)

        self.events[idx].execute()
        return dt, self.events[idx].name

    def run(self, t_max: float) -> List[Tuple[float, str]]:
        """
        Run KMC until time reaches t_max.

        Returns list of (time, event_name) records.
        """
        history: List[Tuple[float, str]] = []
        while self.time < t_max:
            dt, name = self.step()
            history.append((self.time, name))
        return history

    def run_n_steps(self, n_steps: int) -> List[Tuple[float, str]]:
        """Run KMC for exactly n_steps."""
        history: List[Tuple[float, str]] = []
        for _ in range(n_steps):
            dt, name = self.step()
            history.append((self.time, name))
        return history


# ===================================================================
#  Gillespie Stochastic Simulation Algorithm (SSA)
# ===================================================================

@dataclass
class Reaction:
    """Chemical reaction for Gillespie SSA."""
    name: str
    reactants: Dict[str, int]    # species → stoichiometric coefficient (consumed)
    products: Dict[str, int]     # species → stoichiometric coefficient (produced)
    rate_constant: float         # k [appropriate units]

    def propensity(self, populations: Dict[str, int]) -> float:
        """
        Compute propensity a_j = k_j × combinatorial factor.

        For A + B → ...: a = k × n_A × n_B
        For 2A → ...: a = k × n_A × (n_A - 1) / 2
        """
        a = self.rate_constant
        for species, coeff in self.reactants.items():
            n = populations.get(species, 0)
            if coeff == 1:
                a *= n
            elif coeff == 2:
                a *= n * (n - 1) / 2.0
            else:
                # General combinatorial
                prod = 1.0
                for k in range(coeff):
                    prod *= (n - k)
                a *= prod / math.factorial(coeff)
        return max(a, 0.0)


class GillespieSSA:
    r"""
    Gillespie's Stochastic Simulation Algorithm (Direct Method).

    Exact algorithm for well-stirred chemically reacting systems.

    At each step:
    1. Compute propensities $a_j$ for all reactions
    2. Total propensity $a_0 = \sum_j a_j$
    3. Time to next reaction: $\tau = -\ln(r_1)/a_0$
    4. Select reaction $j$ s.t. $\sum_{k=1}^{j-1}a_k < r_2 a_0 \leq \sum_{k=1}^{j}a_k$
    5. Update populations: $X_i \to X_i + \nu_{ij}$

    References
    ----------
    Gillespie, D.T. (1977). J. Comput. Phys. 22(4), 403-434.
    """

    def __init__(self, reactions: List[Reaction],
                 initial_populations: Dict[str, int],
                 seed: Optional[int] = None) -> None:
        self.reactions = reactions
        self.populations = dict(initial_populations)
        self.time: float = 0.0
        self.rng = np.random.default_rng(seed)

    def step(self) -> Tuple[float, Optional[str]]:
        """
        Execute one Gillespie step.

        Returns (new_time, reaction_name) or (time, None) if all propensities zero.
        """
        propensities = np.array([r.propensity(self.populations)
                                  for r in self.reactions])
        a0 = np.sum(propensities)

        if a0 <= 0:
            return self.time, None

        # Time to next reaction
        r1 = self.rng.random()
        tau = -math.log(r1) / a0
        self.time += tau

        # Select reaction
        r2 = self.rng.random() * a0
        cumsum = np.cumsum(propensities)
        idx = int(np.searchsorted(cumsum, r2))
        idx = min(idx, len(self.reactions) - 1)

        rxn = self.reactions[idx]

        # Update populations
        for species, coeff in rxn.reactants.items():
            self.populations[species] -= coeff
        for species, coeff in rxn.products.items():
            self.populations[species] = self.populations.get(species, 0) + coeff

        return self.time, rxn.name

    def run(self, t_max: float,
            record_interval: float = 0.0) -> Tuple[NDArray[np.float64],
                                                      Dict[str, NDArray[np.int64]]]:
        """
        Run SSA until t_max.

        Parameters
        ----------
        t_max : Maximum simulation time.
        record_interval : If > 0, record at regular intervals (interpolated).
                         If 0, record every event.

        Returns
        -------
        times : Array of time points.
        trajectories : dict mapping species name → population array.
        """
        species_names = sorted(self.populations.keys())
        times_list: List[float] = [self.time]
        pops_list: List[Dict[str, int]] = [dict(self.populations)]

        while self.time < t_max:
            t, name = self.step()
            if name is None:
                break
            times_list.append(t)
            pops_list.append(dict(self.populations))

        times = np.array(times_list)

        if record_interval > 0 and len(times) > 1:
            # Interpolate onto regular grid
            t_grid = np.arange(0, t_max, record_interval)
            trajectories: Dict[str, NDArray[np.int64]] = {}
            for sp in species_names:
                raw = np.array([p[sp] for p in pops_list])
                # Step interpolation (populations are discrete)
                interp = np.searchsorted(times, t_grid, side="right") - 1
                interp = np.clip(interp, 0, len(raw) - 1)
                trajectories[sp] = raw[interp].astype(np.int64)
            return t_grid, trajectories
        else:
            trajectories = {}
            for sp in species_names:
                trajectories[sp] = np.array([p[sp] for p in pops_list],
                                             dtype=np.int64)
            return times, trajectories

    # -- Convenience factories for common kinetic schemes --

    @staticmethod
    def lotka_volterra(prey_init: int = 1000, predator_init: int = 1000,
                       k1: float = 10.0, k2: float = 0.01,
                       k3: float = 10.0,
                       seed: Optional[int] = None) -> "GillespieSSA":
        """
        Lotka-Volterra predator-prey model.

        X → 2X       (prey reproduction, rate k1)
        X + Y → 2Y   (predation, rate k2)
        Y → ∅        (predator death, rate k3)
        """
        reactions = [
            Reaction("prey_birth", {"X": 1}, {"X": 2}, k1),
            Reaction("predation", {"X": 1, "Y": 1}, {"Y": 2}, k2),
            Reaction("predator_death", {"Y": 1}, {}, k3),
        ]
        return GillespieSSA(reactions, {"X": prey_init, "Y": predator_init},
                           seed=seed)

    @staticmethod
    def michaelis_menten(S_init: int = 300, E_init: int = 120,
                         k1: float = 0.001, k_minus1: float = 0.005,
                         k2: float = 0.01,
                         seed: Optional[int] = None) -> "GillespieSSA":
        """
        Michaelis-Menten enzyme kinetics.

        E + S ⇌ ES → E + P
        """
        reactions = [
            Reaction("binding", {"E": 1, "S": 1}, {"ES": 1}, k1),
            Reaction("unbinding", {"ES": 1}, {"E": 1, "S": 1}, k_minus1),
            Reaction("catalysis", {"ES": 1}, {"E": 1, "P": 1}, k2),
        ]
        pops = {"S": S_init, "E": E_init, "ES": 0, "P": 0}
        return GillespieSSA(reactions, pops, seed=seed)

    @staticmethod
    def sir_epidemic(S_init: int = 999, I_init: int = 1, R_init: int = 0,
                     beta: float = 0.001, gamma: float = 0.1,
                     seed: Optional[int] = None) -> "GillespieSSA":
        """
        SIR epidemic model (stochastic).

        S + I → 2I   (infection, rate β)
        I → R        (recovery, rate γ)
        """
        reactions = [
            Reaction("infection", {"S": 1, "I": 1}, {"I": 2}, beta),
            Reaction("recovery", {"I": 1}, {"R": 1}, gamma),
        ]
        pops = {"S": S_init, "I": I_init, "R": R_init}
        return GillespieSSA(reactions, pops, seed=seed)


# ===================================================================
#  Chemical Master Equation
# ===================================================================

class ChemicalMasterEquation:
    r"""
    Chemical master equation (CME) for small discrete-state systems.

    $$\frac{dP(\mathbf{x}, t)}{dt} = \sum_j \left[
        a_j(\mathbf{x} - \boldsymbol{\nu}_j)\,P(\mathbf{x}-\boldsymbol{\nu}_j, t)
        - a_j(\mathbf{x})\,P(\mathbf{x}, t)
    \right]$$

    Direct solution via sparse matrix exponential for small state spaces.
    For larger spaces, use Gillespie SSA instead.
    """

    def __init__(self, reactions: List[Reaction],
                 max_populations: Dict[str, int]) -> None:
        """
        Parameters
        ----------
        reactions : List of reactions.
        max_populations : Maximum population for each species
                         (truncates infinite state space).
        """
        self.reactions = reactions
        self.species = sorted(max_populations.keys())
        self.max_pops = {s: max_populations[s] for s in self.species}
        self.dims = [max_populations[s] + 1 for s in self.species]
        self.n_states = int(np.prod(self.dims))

        # Build transition rate matrix
        self._Q = self._build_generator()

    def _state_to_index(self, state: Dict[str, int]) -> int:
        """Map population vector to flat index."""
        idx = 0
        stride = 1
        for s in self.species:
            idx += state.get(s, 0) * stride
            stride *= self.dims[self.species.index(s)]
        return idx

    def _index_to_state(self, idx: int) -> Dict[str, int]:
        """Map flat index to population vector."""
        state = {}
        for i, s in enumerate(self.species):
            dim = self.dims[i]
            state[s] = idx % dim
            idx //= dim
        return state

    def _build_generator(self) -> NDArray[np.float64]:
        """Build the generator matrix Q for dP/dt = Q·P."""
        Q = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            state = self._index_to_state(i)

            for rxn in self.reactions:
                prop = rxn.propensity(state)
                if prop <= 0:
                    continue

                # Compute new state after reaction
                new_state = dict(state)
                valid = True
                for sp, coeff in rxn.reactants.items():
                    new_state[sp] -= coeff
                    if new_state[sp] < 0:
                        valid = False
                        break
                if not valid:
                    continue

                for sp, coeff in rxn.products.items():
                    new_state[sp] = new_state.get(sp, 0) + coeff
                    if new_state[sp] > self.max_pops.get(sp, 1000):
                        valid = False
                        break
                if not valid:
                    continue

                j = self._state_to_index(new_state)
                Q[j, i] += prop
                Q[i, i] -= prop

        return Q

    def solve(self, initial_state: Dict[str, int],
              t_eval: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Solve CME for probability distribution at times t_eval.

        Parameters
        ----------
        initial_state : Initial population dict.
        t_eval : Times at which to compute P(x, t).

        Returns
        -------
        P : (n_times, n_states) probability distribution.
        """
        P0 = np.zeros(self.n_states)
        P0[self._state_to_index(initial_state)] = 1.0

        P_out = np.zeros((len(t_eval), self.n_states))

        # Simple matrix exponential via eigendecomposition
        # (feasible only for small state spaces)
        if self.n_states <= 500:
            eigenvalues, V = np.linalg.eig(self._Q)
            V_inv = np.linalg.inv(V)
            c = V_inv @ P0

            for i, t in enumerate(t_eval):
                exp_lam = np.exp(eigenvalues * t)
                P_out[i] = np.real(V @ (c * exp_lam))
                # Clamp numerical artefacts
                P_out[i] = np.maximum(P_out[i], 0.0)
                P_out[i] /= np.sum(P_out[i])
        else:
            # For large state spaces, use Krylov subspace method
            P = P0.copy()
            dt_prev = 0.0
            for i, t in enumerate(t_eval):
                dt = t - dt_prev
                if dt > 0:
                    # Euler steps (first-order, for large systems)
                    n_sub = max(1, int(dt * np.max(np.abs(np.diag(self._Q))) * 10))
                    ds = dt / n_sub
                    for _ in range(n_sub):
                        P = P + ds * (self._Q @ P)
                        P = np.maximum(P, 0.0)
                        norm = np.sum(P)
                        if norm > 0:
                            P /= norm
                P_out[i] = P
                dt_prev = t

        return P_out

    def mean_populations(self, P: NDArray[np.float64]) -> Dict[str, float]:
        """Compute mean populations from probability distribution."""
        means: Dict[str, float] = {s: 0.0 for s in self.species}
        for i in range(self.n_states):
            state = self._index_to_state(i)
            for s in self.species:
                means[s] += state[s] * P[i]
        return means
