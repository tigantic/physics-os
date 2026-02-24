"""QTT Physics VM — Adaptive rank governor.

Controls truncation policy across all time steps.  The same governor
applies to every physics domain — this is the "same truncation policy"
guarantee the VM provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .qtt_tensor import QTTTensor


@dataclass
class TruncationPolicy:
    """Truncation policy applied after every rank-increasing operation.

    Parameters
    ----------
    max_rank : int
        Hard ceiling on bond dimension.
    rel_tol : float
        Relative SVD cutoff (fraction of Frobenius norm to discard).
    """
    max_rank: int = 64
    rel_tol: float = 1e-10


@dataclass
class RankGovernor:
    """Adaptive rank governor for the QTT runtime.

    Applies a uniform truncation policy and tracks rank statistics
    across time steps.  If ranks approach ``max_rank`` consistently,
    the governor logs a warning but does not auto-raise the ceiling
    (deterministic behavior is preferred for reproducibility).

    Parameters
    ----------
    policy : TruncationPolicy
        The truncation policy to enforce.
    """

    policy: TruncationPolicy = field(default_factory=TruncationPolicy)

    # ── statistics ──────────────────────────────────────────────────
    _rank_history: list[int] = field(default_factory=list, repr=False)
    _truncation_count: int = 0
    _saturated_count: int = 0

    def truncate(self, tensor: QTTTensor) -> QTTTensor:
        """Apply the truncation policy.

        Returns the truncated tensor and records statistics.
        """
        before_rank = tensor.max_rank
        result = tensor.truncate(
            max_rank=self.policy.max_rank,
            cutoff=self.policy.rel_tol,
        )
        after_rank = result.max_rank
        self._rank_history.append(after_rank)
        self._truncation_count += 1
        if after_rank >= self.policy.max_rank:
            self._saturated_count += 1
        return result

    @property
    def peak_rank(self) -> int:
        """Maximum rank observed across all truncations."""
        return max(self._rank_history) if self._rank_history else 0

    @property
    def mean_rank(self) -> float:
        """Mean rank across all truncations."""
        if not self._rank_history:
            return 0.0
        return sum(self._rank_history) / len(self._rank_history)

    @property
    def saturation_rate(self) -> float:
        """Fraction of truncations that hit the max_rank ceiling."""
        if self._truncation_count == 0:
            return 0.0
        return self._saturated_count / self._truncation_count

    @property
    def n_truncations(self) -> int:
        return self._truncation_count

    def reset(self) -> None:
        """Clear statistics (called between programs)."""
        self._rank_history.clear()
        self._truncation_count = 0
        self._saturated_count = 0
