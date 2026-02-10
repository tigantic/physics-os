"""
Agent-Based Modelling (ABM) for Crowd / Granular / Biological Systems
======================================================================

Production-grade agent-based modelling framework supporting:
    * Social-force model (Helbing & Molnár 1995)
    * Boids flocking (Reynolds 1987)
    * Susceptible-Infected-Recovered (SIR) epidemiological diffusion
    * Custom agent behaviours via composition

Social-force model:
    The total force on pedestrian *i* is:

    .. math::
        \\mathbf{F}_i = \\mathbf{F}_i^{\\text{des}}
            + \\sum_{j \\neq i} \\mathbf{F}_{ij}^{\\text{soc}}
            + \\sum_w \\mathbf{F}_{iw}^{\\text{wall}}

    with desired-velocity driving:

    .. math::
        \\mathbf{F}_i^{\\text{des}} = \\frac{1}{\\tau}
            (v_0 \\hat{\\mathbf{e}}_i - \\mathbf{v}_i)

    and exponential social repulsion:

    .. math::
        \\mathbf{F}_{ij}^{\\text{soc}} = A \\exp\\left(
            \\frac{r_{ij} - d_{ij}}{B}\\right) \\hat{\\mathbf{n}}_{ij}

References:
    [1] Helbing & Molnár, "Social force model for pedestrian dynamics",
        PRE 51, 4282 (1995).
    [2] Reynolds, "Flocks, herds and schools: a distributed behavioral
        model", Comp. Graphics 21, 25 (1987).
    [3] Kermack & McKendrick, "A contribution to the mathematical theory
        of epidemics", Proc. R. Soc. London A 115, 700 (1927).

Domain IX.20 — Biology / Agent-Based Modelling.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# Agent definitions
# ===================================================================

class AgentState(enum.Enum):
    """Health state for epidemiological models."""
    SUSCEPTIBLE = 0
    INFECTED = 1
    RECOVERED = 2
    DEAD = 3


@dataclass
class Agent:
    """
    Single agent with position, velocity, and optional state metadata.

    Attributes:
        pos: Position vector (2D or 3D).
        vel: Velocity vector.
        desired_speed: Preferred cruising speed [m/s].
        radius: Body radius for collision [m].
        mass: Agent mass [kg].
        state: Discrete state for epidemiological models.
        infection_timer: Ticks remaining before recovery.
    """
    pos: NDArray
    vel: NDArray
    desired_speed: float = 1.34
    radius: float = 0.3
    mass: float = 80.0
    state: AgentState = AgentState.SUSCEPTIBLE
    infection_timer: int = 0


@dataclass
class Wall:
    """
    Line-segment wall for social-force boundaries.

    Attributes:
        p0: Start point.
        p1: End point.
    """
    p0: NDArray
    p1: NDArray


# ===================================================================
# Social-force model
# ===================================================================

@dataclass
class SocialForceParams:
    """
    Parameters of the Helbing-Molnár social-force model.

    Attributes:
        tau: Relaxation time [s].
        A: Social-force strength [N].
        B: Social-force range [m].
        A_wall: Wall-force strength [N].
        B_wall: Wall-force range [m].
        k_body: Body-compression spring constant [N/m].
        kappa_friction: Sliding-friction coefficient [kg/(m·s)].
    """
    tau: float = 0.5
    A: float = 2000.0
    B: float = 0.08
    A_wall: float = 2000.0
    B_wall: float = 0.08
    k_body: float = 1.2e5
    kappa_friction: float = 2.4e5


def _closest_point_on_segment(p: NDArray, a: NDArray, b: NDArray) -> NDArray:
    """Project p onto segment [a, b], clamp to endpoints."""
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-30)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


class SocialForceModel:
    """
    Social-force crowd simulation.

    Example::

        agents = [Agent(pos=np.array([0., 0.]), vel=np.array([1., 0.]))]
        walls = [Wall(p0=np.array([-5., 2.]), p1=np.array([5., 2.]))]
        goals = [np.array([10., 0.])]
        sfm = SocialForceModel(agents, walls, goals, SocialForceParams())
        for _ in range(1000):
            sfm.step(dt=0.01)
    """

    def __init__(
        self,
        agents: List[Agent],
        walls: List[Wall],
        goals: List[NDArray],
        params: SocialForceParams,
    ) -> None:
        self.agents = agents
        self.walls = walls
        self.goals = goals
        self.params = params

    def _desired_force(self, i: int) -> NDArray:
        """Driving force towards goal."""
        a = self.agents[i]
        goal = self.goals[i % len(self.goals)]
        direction = goal - a.pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.zeros_like(a.pos)
        e_i = direction / dist
        return a.mass * (a.desired_speed * e_i - a.vel) / self.params.tau

    def _social_force(self, i: int, j: int) -> NDArray:
        """Pairwise social + body + friction force between agents i and j."""
        ai, aj = self.agents[i], self.agents[j]
        d_ij = ai.pos - aj.pos
        dist = np.linalg.norm(d_ij)
        if dist < 1e-10:
            return np.zeros_like(ai.pos)
        n_ij = d_ij / dist
        r_ij = ai.radius + aj.radius

        # Social repulsion
        f_soc = self.params.A * np.exp((r_ij - dist) / self.params.B) * n_ij

        # Body compression (if overlapping)
        overlap = max(0.0, r_ij - dist)
        f_body = self.params.k_body * overlap * n_ij

        # Tangential friction
        t_ij = np.array([-n_ij[1], n_ij[0]]) if len(n_ij) == 2 else np.cross(n_ij, np.array([0, 0, 1]))[:len(n_ij)]
        delta_v = np.dot(aj.vel - ai.vel, t_ij)
        f_friction = self.params.kappa_friction * overlap * delta_v * t_ij

        return f_soc + f_body + f_friction

    def _wall_force(self, i: int) -> NDArray:
        """Sum of wall repulsion forces on agent i."""
        ai = self.agents[i]
        f_total = np.zeros_like(ai.pos)

        for wall in self.walls:
            closest = _closest_point_on_segment(ai.pos, wall.p0, wall.p1)
            d = ai.pos - closest
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                continue
            n_iw = d / dist

            f_soc = self.params.A_wall * np.exp((ai.radius - dist) / self.params.B_wall) * n_iw
            overlap = max(0.0, ai.radius - dist)
            f_body = self.params.k_body * overlap * n_iw
            f_total += f_soc + f_body

        return f_total

    def step(self, dt: float = 0.01) -> None:
        """Advance one time step using Euler integration."""
        n = len(self.agents)
        forces = [np.zeros_like(self.agents[0].pos) for _ in range(n)]

        for i in range(n):
            forces[i] += self._desired_force(i)
            forces[i] += self._wall_force(i)
            for j in range(n):
                if j != i:
                    forces[i] += self._social_force(i, j)

        for i in range(n):
            a = self.agents[i]
            a.vel = a.vel + (forces[i] / a.mass) * dt
            speed = np.linalg.norm(a.vel)
            max_speed = 2.0 * a.desired_speed
            if speed > max_speed:
                a.vel = a.vel * (max_speed / speed)
            a.pos = a.pos + a.vel * dt

    def positions(self) -> NDArray:
        """Return ``(n_agents, ndim)`` position array."""
        return np.array([a.pos for a in self.agents])

    def velocities(self) -> NDArray:
        """Return ``(n_agents, ndim)`` velocity array."""
        return np.array([a.vel for a in self.agents])


# ===================================================================
# Boids flocking
# ===================================================================

@dataclass
class BoidsParams:
    """
    Boids flocking parameters (Reynolds 1987).

    Attributes:
        separation_radius: Distance within which agents separate.
        alignment_radius: Distance for velocity alignment.
        cohesion_radius: Distance for cohesion attraction.
        w_sep: Separation weight.
        w_ali: Alignment weight.
        w_coh: Cohesion weight.
        max_speed: Maximum agent speed.
        max_force: Maximum steering force.
    """
    separation_radius: float = 1.0
    alignment_radius: float = 3.0
    cohesion_radius: float = 5.0
    w_sep: float = 1.5
    w_ali: float = 1.0
    w_coh: float = 1.0
    max_speed: float = 2.0
    max_force: float = 0.5


class BoidsSimulation:
    """
    Boids flocking simulation.

    Example::

        boids = BoidsSimulation(n_agents=100, ndim=2)
        for _ in range(500):
            boids.step(dt=0.05)
    """

    def __init__(
        self,
        n_agents: int = 100,
        ndim: int = 2,
        params: Optional[BoidsParams] = None,
        domain_size: float = 50.0,
    ) -> None:
        self.params = params or BoidsParams()
        self.ndim = ndim
        self.domain_size = domain_size
        self.positions = np.random.uniform(0, domain_size, (n_agents, ndim))
        self.velocities = np.random.uniform(-1, 1, (n_agents, ndim))
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities *= self.params.max_speed / (speeds + 1e-10)

    @property
    def n_agents(self) -> int:
        return self.positions.shape[0]

    def _limit_vector(self, v: NDArray, max_mag: float) -> NDArray:
        mag = np.linalg.norm(v)
        if mag > max_mag:
            return v * (max_mag / mag)
        return v

    def step(self, dt: float = 0.05) -> None:
        """Advance one time step."""
        p = self.params
        n = self.n_agents

        for i in range(n):
            sep = np.zeros(self.ndim)
            ali = np.zeros(self.ndim)
            coh = np.zeros(self.ndim)
            n_sep = 0
            n_ali = 0
            n_coh = 0

            for j in range(n):
                if j == i:
                    continue
                diff = self.positions[i] - self.positions[j]
                # Periodic wrapping
                diff = diff - self.domain_size * np.round(diff / self.domain_size)
                dist = np.linalg.norm(diff)

                if dist < p.separation_radius and dist > 1e-10:
                    sep += diff / (dist * dist)
                    n_sep += 1
                if dist < p.alignment_radius:
                    ali += self.velocities[j]
                    n_ali += 1
                if dist < p.cohesion_radius:
                    coh += self.positions[j]
                    n_coh += 1

            steer = np.zeros(self.ndim)

            if n_sep > 0:
                sep /= n_sep
                steer += p.w_sep * self._limit_vector(sep, p.max_force)
            if n_ali > 0:
                ali /= n_ali
                ali_steer = ali - self.velocities[i]
                steer += p.w_ali * self._limit_vector(ali_steer, p.max_force)
            if n_coh > 0:
                coh /= n_coh
                coh_steer = coh - self.positions[i]
                steer += p.w_coh * self._limit_vector(coh_steer, p.max_force)

            steer = self._limit_vector(steer, p.max_force)
            self.velocities[i] += steer * dt
            speed = np.linalg.norm(self.velocities[i])
            if speed > p.max_speed:
                self.velocities[i] *= p.max_speed / speed

        self.positions += self.velocities * dt
        # Periodic boundary
        self.positions %= self.domain_size


# ===================================================================
# SIR epidemiological model on agents
# ===================================================================

@dataclass
class SIRParams:
    """
    SIR epidemic parameters for spatial ABM.

    Attributes:
        infection_radius: Distance for transmission [m].
        infection_prob: Probability of transmission per contact per tick.
        recovery_ticks: Ticks until recovery.
        mortality_rate: Probability of death upon recovery.
    """
    infection_radius: float = 1.5
    infection_prob: float = 0.05
    recovery_ticks: int = 140
    mortality_rate: float = 0.02


class SIRAgentModel:
    """
    Spatial SIR epidemic on mobile agents.

    Agents perform a random walk; infected agents transmit within
    *infection_radius* with *infection_prob* per tick.

    Example::

        sir = SIRAgentModel(n_agents=1000, n_initial_infected=5)
        for _ in range(2000):
            sir.step(dt=0.1)
        S, I, R, D = sir.counts()
    """

    def __init__(
        self,
        n_agents: int = 500,
        n_initial_infected: int = 5,
        params: Optional[SIRParams] = None,
        domain_size: float = 50.0,
        ndim: int = 2,
        speed: float = 1.0,
    ) -> None:
        self.params = params or SIRParams()
        self.domain_size = domain_size
        self.speed = speed
        self.ndim = ndim

        self.agents: List[Agent] = []
        for k in range(n_agents):
            pos = np.random.uniform(0, domain_size, ndim)
            angle = np.random.uniform(0, 2 * np.pi)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            state = AgentState.INFECTED if k < n_initial_infected else AgentState.SUSCEPTIBLE
            timer = self.params.recovery_ticks if state == AgentState.INFECTED else 0
            self.agents.append(Agent(
                pos=pos, vel=vel, desired_speed=speed,
                state=state, infection_timer=timer,
            ))

    def step(self, dt: float = 0.1) -> None:
        """One tick: move, infect, recover."""
        p = self.params

        # Random-walk direction perturbation
        for a in self.agents:
            if a.state == AgentState.DEAD:
                continue
            angle_pert = np.random.normal(0, 0.3)
            cos_a, sin_a = np.cos(angle_pert), np.sin(angle_pert)
            vx, vy = a.vel[0], a.vel[1]
            a.vel = np.array([cos_a * vx - sin_a * vy, sin_a * vx + cos_a * vy])
            a.pos += a.vel * dt
            a.pos %= self.domain_size

        # Infection transmission
        for i, ai in enumerate(self.agents):
            if ai.state != AgentState.SUSCEPTIBLE:
                continue
            for j, aj in enumerate(self.agents):
                if aj.state != AgentState.INFECTED:
                    continue
                diff = ai.pos - aj.pos
                diff -= self.domain_size * np.round(diff / self.domain_size)
                dist = np.linalg.norm(diff)
                if dist < p.infection_radius:
                    if np.random.random() < p.infection_prob:
                        ai.state = AgentState.INFECTED
                        ai.infection_timer = p.recovery_ticks
                        break

        # Recovery / mortality
        for a in self.agents:
            if a.state == AgentState.INFECTED:
                a.infection_timer -= 1
                if a.infection_timer <= 0:
                    if np.random.random() < p.mortality_rate:
                        a.state = AgentState.DEAD
                        a.vel = np.zeros(self.ndim)
                    else:
                        a.state = AgentState.RECOVERED

    def counts(self) -> Tuple[int, int, int, int]:
        """Return (S, I, R, D) counts."""
        S = sum(1 for a in self.agents if a.state == AgentState.SUSCEPTIBLE)
        I = sum(1 for a in self.agents if a.state == AgentState.INFECTED)
        R = sum(1 for a in self.agents if a.state == AgentState.RECOVERED)
        D = sum(1 for a in self.agents if a.state == AgentState.DEAD)
        return S, I, R, D

    def positions_by_state(self) -> dict:
        """Return dict[AgentState] -> NDArray of positions."""
        result = {}
        for st in AgentState:
            pts = [a.pos for a in self.agents if a.state == st]
            result[st] = np.array(pts) if pts else np.empty((0, self.ndim))
        return result
