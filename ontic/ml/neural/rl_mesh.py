"""
5.8 — Reinforcement Learning for Mesh Adaptation
==================================================

RL agent that learns h/p-refinement policies on PDE meshes.

* **State**: local element error indicators, element size, polynomial
  degree, neighbour statistics.
* **Action**: refine (h), coarsen, increase p-order, decrease p-order, noop.
* **Reward**: error reduction per DOF added (efficiency).

Implements PPO-style policy gradient in pure NumPy for small meshes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Action space ──────────────────────────────────────────────────

class MeshAction(Enum):
    """Available adaptation actions per element."""
    NOOP = 0
    H_REFINE = 1
    H_COARSEN = 2
    P_INCREASE = 3
    P_DECREASE = 4


# ── State / observation ──────────────────────────────────────────

@dataclass
class ElementState:
    """Observable state for a single mesh element."""
    error_indicator: float
    element_size: float
    p_order: int
    n_neighbours: int
    mean_neighbour_error: float
    aspect_ratio: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.error_indicator,
            self.element_size,
            float(self.p_order),
            float(self.n_neighbours),
            self.mean_neighbour_error,
            self.aspect_ratio,
        ], dtype=np.float32)


STATE_DIM = 6
N_ACTIONS = len(MeshAction)


# ── Simple mesh environment ──────────────────────────────────────

@dataclass
class MeshElement:
    """Lightweight mesh element."""
    idx: int
    size: float
    p_order: int
    error: float
    neighbours: List[int] = field(default_factory=list)


class MeshEnvironment:
    """Simplified mesh environment for RL training.

    Wraps a 1-D / 2-D mesh of elements with error indicators,
    provides state observations and reward after actions.
    """

    def __init__(
        self,
        n_elements: int = 100,
        target_error: float = 1e-4,
        max_dofs: int = 10000,
        seed: int = 0,
    ) -> None:
        self.target_error = target_error
        self.max_dofs = max_dofs
        self.rng = np.random.default_rng(seed)

        # Initialize mesh
        self.elements: List[MeshElement] = []
        for i in range(n_elements):
            size = 1.0 / n_elements
            nbs = []
            if i > 0:
                nbs.append(i - 1)
            if i < n_elements - 1:
                nbs.append(i + 1)
            self.elements.append(MeshElement(
                idx=i,
                size=size,
                p_order=1,
                error=self.rng.exponential(0.1),
                neighbours=nbs,
            ))

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def total_dofs(self) -> int:
        return sum(e.p_order + 1 for e in self.elements)

    @property
    def max_error(self) -> float:
        return max(e.error for e in self.elements) if self.elements else 0.0

    @property
    def mean_error(self) -> float:
        return float(np.mean([e.error for e in self.elements])) if self.elements else 0.0

    def get_state(self, elem_idx: int) -> ElementState:
        e = self.elements[elem_idx]
        nb_errors = [self.elements[n].error for n in e.neighbours] or [0.0]
        return ElementState(
            error_indicator=e.error,
            element_size=e.size,
            p_order=e.p_order,
            n_neighbours=len(e.neighbours),
            mean_neighbour_error=float(np.mean(nb_errors)),
        )

    def apply_action(self, elem_idx: int, action: MeshAction) -> float:
        """Apply action to element, return reward."""
        e = self.elements[elem_idx]
        old_error = e.error
        old_dofs = e.p_order + 1

        if action == MeshAction.H_REFINE:
            e.size *= 0.5
            e.error *= 0.25   # h-refinement ~ O(h²)
        elif action == MeshAction.H_COARSEN:
            e.size *= 2.0
            e.error *= 4.0
        elif action == MeshAction.P_INCREASE:
            e.p_order = min(e.p_order + 1, 10)
            e.error *= 0.5 ** (e.p_order - 1)  # exponential convergence
        elif action == MeshAction.P_DECREASE:
            if e.p_order > 1:
                e.p_order -= 1
                e.error *= 2.0
        # NOOP: no change

        new_dofs = e.p_order + 1
        error_reduction = max(old_error - e.error, 0.0)
        dof_cost = max(new_dofs - old_dofs, 0) + 1
        reward = error_reduction / dof_cost

        # Penalty for exceeding DOF budget
        if self.total_dofs > self.max_dofs:
            reward -= 1.0

        return float(reward)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-randomize errors."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for e in self.elements:
            e.error = self.rng.exponential(0.1)
            e.size = 1.0 / self.n_elements
            e.p_order = 1


# ── Policy network ────────────────────────────────────────────────

class PolicyNetwork:
    """Small MLP policy π(a|s) for mesh adaptation."""

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        hidden_dims = hidden_dims or [64, 64]
        rng = np.random.default_rng(seed)
        layers = [STATE_DIM] + hidden_dims + [N_ACTIONS]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            std = math.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.weights.append(
                rng.normal(0, std, (layers[i], layers[i + 1])).astype(np.float32)
            )
            self.biases.append(np.zeros(layers[i + 1], dtype=np.float32))

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Compute action logits. state: (..., STATE_DIM) → (..., N_ACTIONS)."""
        h = state
        for i in range(len(self.weights) - 1):
            h = np.tanh(h @ self.weights[i] + self.biases[i])
        logits = h @ self.weights[-1] + self.biases[-1]
        return logits

    def action_probs(self, state: np.ndarray) -> np.ndarray:
        logits = self.forward(state)
        exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp_l / exp_l.sum(axis=-1, keepdims=True)

    def sample_action(self, state: np.ndarray, rng: np.random.Generator) -> int:
        probs = self.action_probs(state)
        return int(rng.choice(N_ACTIONS, p=probs.ravel()))

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


# ── Value network ─────────────────────────────────────────────────

class ValueNetwork:
    """Critic V(s) for advantage estimation."""

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        seed: int = 7,
    ) -> None:
        hidden_dims = hidden_dims or [64, 64]
        rng = np.random.default_rng(seed)
        layers = [STATE_DIM] + hidden_dims + [1]

        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            std = math.sqrt(2.0 / (layers[i] + layers[i + 1]))
            self.weights.append(
                rng.normal(0, std, (layers[i], layers[i + 1])).astype(np.float32)
            )
            self.biases.append(np.zeros(layers[i + 1], dtype=np.float32))

    def forward(self, state: np.ndarray) -> float:
        h = state
        for i in range(len(self.weights) - 1):
            h = np.tanh(h @ self.weights[i] + self.biases[i])
        return float((h @ self.weights[-1] + self.biases[-1]).item())

    def parameters(self) -> List[np.ndarray]:
        return self.weights + self.biases


# ── PPO Agent ─────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    """PPO hyper-parameters for mesh adaptation RL."""
    gamma: float = 0.99
    lam: float = 0.95         # GAE λ
    clip_eps: float = 0.2
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    n_epochs: int = 4
    batch_size: int = 64
    max_steps_per_episode: int = 200


class RLMeshAgent:
    """PPO agent for adaptive mesh refinement.

    Workflow:
        1. Observe per-element states.
        2. Select actions (h/p refine/coarsen).
        3. Collect trajectories.
        4. Update policy via PPO.
    """

    def __init__(
        self,
        env: Optional[MeshEnvironment] = None,
        cfg: Optional[PPOConfig] = None,
        seed: int = 42,
    ) -> None:
        self.env = env or MeshEnvironment()
        self.cfg = cfg or PPOConfig()
        self.policy = PolicyNetwork(seed=seed)
        self.value = ValueNetwork(seed=seed + 1)
        self.rng = np.random.default_rng(seed)

        # Trajectory buffers
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._log_probs: List[float] = []
        self._values: List[float] = []

    def collect_episode(self) -> Dict[str, float]:
        """Run one episode of mesh adaptation and collect trajectory."""
        self.env.reset(seed=int(self.rng.integers(2**31)))
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()

        total_reward = 0.0

        for step in range(self.cfg.max_steps_per_episode):
            # Pick a random element to adapt
            elem_idx = int(self.rng.integers(self.env.n_elements))
            state = self.env.get_state(elem_idx).to_array()

            probs = self.policy.action_probs(state)
            action = int(self.rng.choice(N_ACTIONS, p=probs.ravel()))
            log_prob = float(np.log(probs.ravel()[action] + 1e-10))
            value = self.value.forward(state)

            reward = self.env.apply_action(elem_idx, MeshAction(action))

            self._states.append(state)
            self._actions.append(action)
            self._rewards.append(reward)
            self._log_probs.append(log_prob)
            self._values.append(value)
            total_reward += reward

            if self.env.max_error < self.env.target_error:
                break

        return {
            "total_reward": total_reward,
            "steps": len(self._states),
            "final_max_error": self.env.max_error,
            "final_mean_error": self.env.mean_error,
            "total_dofs": self.env.total_dofs,
        }

    def compute_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        T = len(self._rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            next_val = self._values[t + 1] if t + 1 < T else 0.0
            delta = self._rewards[t] + self.cfg.gamma * next_val - self._values[t]
            gae = delta + self.cfg.gamma * self.cfg.lam * gae
            advantages[t] = gae
            returns[t] = gae + self._values[t]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """PPO policy update using collected trajectory.

        Uses numerical gradients for small networks.
        """
        if len(self._states) < 2:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = np.array(self._states)
        actions = np.array(self._actions)
        old_log_probs = np.array(self._log_probs)

        policy_losses: List[float] = []
        value_losses: List[float] = []

        for _ in range(self.cfg.n_epochs):
            # Policy loss (PPO clipped)
            probs = self.policy.action_probs(states)
            new_log_probs = np.log(
                np.array([probs[i, a] for i, a in enumerate(actions)]) + 1e-10
            )
            ratio = np.exp(new_log_probs - old_log_probs)
            clipped = np.clip(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
            p_loss = -float(np.mean(np.minimum(ratio * advantages, clipped * advantages)))
            policy_losses.append(p_loss)

            # Value loss
            values = np.array([self.value.forward(s) for s in states])
            v_loss = float(np.mean((values - returns) ** 2))
            value_losses.append(v_loss)

            # Simple gradient step (numerical for small nets)
            def p_loss_fn() -> float:
                pr = self.policy.action_probs(states)
                nlp = np.log(np.array([pr[i, a] for i, a in enumerate(actions)]) + 1e-10)
                r = np.exp(nlp - old_log_probs)
                c = np.clip(r, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                return -float(np.mean(np.minimum(r * advantages, c * advantages)))

            eps = 1e-5
            for param in self.policy.parameters():
                grad = np.zeros_like(param)
                flat = param.ravel()
                for j in range(min(len(flat), 100)):  # cap gradient evals
                    old_val = flat[j]
                    flat[j] = old_val + eps
                    l_p = p_loss_fn()
                    flat[j] = old_val - eps
                    l_m = p_loss_fn()
                    flat[j] = old_val
                    grad.ravel()[j] = (l_p - l_m) / (2 * eps)
                param -= self.cfg.lr_policy * grad

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
        }

    def adapt_mesh(self, n_episodes: int = 10) -> List[Dict[str, float]]:
        """Train the agent for multiple episodes and return metrics."""
        history: List[Dict[str, float]] = []
        for _ in range(n_episodes):
            ep_info = self.collect_episode()
            update_info = self.update()
            ep_info.update(update_info)
            history.append(ep_info)
        return history


__all__ = [
    "MeshAction",
    "ElementState",
    "MeshElement",
    "MeshEnvironment",
    "PolicyNetwork",
    "ValueNetwork",
    "PPOConfig",
    "RLMeshAgent",
]
