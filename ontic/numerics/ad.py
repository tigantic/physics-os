"""
Automatic Differentiation (AD)
================================

Forward- and reverse-mode automatic differentiation via operator
overloading (dual numbers / tape-based).

Implements:

1. **Dual** — dual number :math:`a + b\\varepsilon` for forward-mode AD
   (tangent propagation).
2. **TapeNode** / **Tape** — Wengert tape for reverse-mode AD
   (adjoint back-propagation).
3. **Variable** — tracked computational graph node supporting
   standard arithmetic and common elementary functions.
4. **grad** / **jacobian** — convenience entry points.

These are self-contained, pure-Python implementations suitable for
verifying derivatives of ontic physics kernels without external
AD frameworks.

Forward mode (dual):
    :math:`f(a + \\varepsilon) = f(a) + f'(a)\\varepsilon`

Reverse mode (tape):
    Build forward pass → backward sweep accumulates adjoints
    :math:`\\bar{x}_i = \\sum_j \\bar{x}_j \\frac{\\partial x_j}{\\partial x_i}`

References:
    [1] Griewank & Walther, *Evaluating Derivatives*, SIAM 2008.
    [2] Baydin et al., "Automatic differentiation in machine learning:
        a survey", JMLR 2018.

Domain I.3.11 — Numerics / Solvers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# Part 1: Forward-mode AD via Dual Numbers
# ===================================================================

class Dual:
    """
    Dual number for forward-mode automatic differentiation.

    Represents :math:`a + b\\varepsilon` where :math:`\\varepsilon^2 = 0`.

    Attributes:
        val: Primal value.
        der: Tangent (derivative) value.

    Example::

        x = Dual(2.0, 1.0)  # seed derivative = 1
        y = x * x + Dual.sin(x)
        print(y.val, y.der)  # f(2), f'(2)
    """

    __slots__ = ("val", "der")

    def __init__(self, val: float, der: float = 0.0) -> None:
        self.val = float(val)
        self.der = float(der)

    def __repr__(self) -> str:
        return f"Dual({self.val}, {self.der})"

    # -- Arithmetic --

    def __add__(self, other: Union[Dual, float, int]) -> Dual:
        if isinstance(other, Dual):
            return Dual(self.val + other.val, self.der + other.der)
        return Dual(self.val + float(other), self.der)

    def __radd__(self, other: Union[float, int]) -> Dual:
        return Dual(float(other) + self.val, self.der)

    def __sub__(self, other: Union[Dual, float, int]) -> Dual:
        if isinstance(other, Dual):
            return Dual(self.val - other.val, self.der - other.der)
        return Dual(self.val - float(other), self.der)

    def __rsub__(self, other: Union[float, int]) -> Dual:
        return Dual(float(other) - self.val, -self.der)

    def __mul__(self, other: Union[Dual, float, int]) -> Dual:
        if isinstance(other, Dual):
            return Dual(
                self.val * other.val,
                self.val * other.der + self.der * other.val,
            )
        c = float(other)
        return Dual(self.val * c, self.der * c)

    def __rmul__(self, other: Union[float, int]) -> Dual:
        c = float(other)
        return Dual(c * self.val, c * self.der)

    def __truediv__(self, other: Union[Dual, float, int]) -> Dual:
        if isinstance(other, Dual):
            return Dual(
                self.val / other.val,
                (self.der * other.val - self.val * other.der) / (other.val**2),
            )
        c = float(other)
        return Dual(self.val / c, self.der / c)

    def __rtruediv__(self, other: Union[float, int]) -> Dual:
        c = float(other)
        return Dual(c / self.val, -c * self.der / (self.val**2))

    def __neg__(self) -> Dual:
        return Dual(-self.val, -self.der)

    def __pow__(self, n: Union[Dual, float, int]) -> Dual:
        if isinstance(n, Dual):
            # General: a^b = exp(b ln a)
            ln_a = math.log(abs(self.val) + 1e-30)
            val = self.val ** n.val
            der = val * (n.der * ln_a + n.val * self.der / (self.val + 1e-30))
            return Dual(val, der)
        p = float(n)
        return Dual(self.val**p, p * self.val ** (p - 1.0) * self.der)

    # -- Elementary functions --

    @staticmethod
    def sin(x: Dual) -> Dual:
        return Dual(math.sin(x.val), math.cos(x.val) * x.der)

    @staticmethod
    def cos(x: Dual) -> Dual:
        return Dual(math.cos(x.val), -math.sin(x.val) * x.der)

    @staticmethod
    def exp(x: Dual) -> Dual:
        e = math.exp(x.val)
        return Dual(e, e * x.der)

    @staticmethod
    def log(x: Dual) -> Dual:
        return Dual(math.log(x.val), x.der / x.val)

    @staticmethod
    def sqrt(x: Dual) -> Dual:
        s = math.sqrt(x.val)
        return Dual(s, x.der / (2.0 * s + 1e-30))

    @staticmethod
    def abs(x: Dual) -> Dual:
        sign = 1.0 if x.val >= 0 else -1.0
        return Dual(abs(x.val), sign * x.der)

    @staticmethod
    def tanh(x: Dual) -> Dual:
        t = math.tanh(x.val)
        return Dual(t, (1.0 - t * t) * x.der)


def forward_derivative(f: Callable[[Dual], Dual], x: float) -> float:
    """
    Compute f'(x) via forward-mode AD.

    Parameters:
        f: Function mapping Dual → Dual.
        x: Evaluation point.

    Returns:
        Derivative f'(x).
    """
    result = f(Dual(x, 1.0))
    return result.der


def forward_gradient(
    f: Callable[..., Dual], x: NDArray,
) -> NDArray:
    """
    Compute gradient ∇f(x) via forward-mode AD (one pass per dimension).

    Parameters:
        f: Function mapping (Dual, Dual, ...) → Dual.
        x: (n,) evaluation point.

    Returns:
        (n,) gradient vector.
    """
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        args = [Dual(x[j], 1.0 if j == i else 0.0) for j in range(n)]
        result = f(*args)
        grad[i] = result.der

    return grad


# ===================================================================
# Part 2: Reverse-mode AD via Tape
# ===================================================================

@dataclass
class TapeEntry:
    """One operation on the Wengert tape."""
    parents: List[int]        # indices of parent variables
    local_grads: List[float]  # ∂(this) / ∂(parent) at forward value


class Tape:
    """
    Global Wengert tape for reverse-mode AD.

    Thread-local singleton pattern; create a fresh tape per differentiation.
    """

    def __init__(self) -> None:
        self.entries: List[TapeEntry] = []

    def push(self, parents: List[int], local_grads: List[float]) -> int:
        idx = len(self.entries)
        self.entries.append(TapeEntry(parents=parents, local_grads=local_grads))
        return idx

    def backward(self, seed_idx: int) -> NDArray:
        """
        Reverse sweep: compute all adjoints from seed node.

        Returns:
            (len(entries),) adjoint array.
        """
        n = len(self.entries)
        adjoints = np.zeros(n)
        adjoints[seed_idx] = 1.0

        for i in range(n - 1, -1, -1):
            if adjoints[i] == 0.0:
                continue
            entry = self.entries[i]
            for parent, grad in zip(entry.parents, entry.local_grads):
                adjoints[parent] += adjoints[i] * grad

        return adjoints


class Variable:
    """
    Tracked variable for reverse-mode AD.

    All standard arithmetic operations record entries on the tape.

    Example::

        tape = Tape()
        x = Variable(2.0, tape)
        y = Variable(3.0, tape)
        z = x * y + Variable.sin(x)
        adjoints = tape.backward(z.idx)
        dz_dx = adjoints[x.idx]
        dz_dy = adjoints[y.idx]
    """

    __slots__ = ("val", "idx", "tape")

    def __init__(self, val: float, tape: Tape) -> None:
        self.val = float(val)
        self.tape = tape
        self.idx = tape.push(parents=[], local_grads=[])

    def __repr__(self) -> str:
        return f"Variable({self.val}, idx={self.idx})"

    def _from_op(self, val: float, parents: List[int], grads: List[float]) -> Variable:
        v = object.__new__(Variable)
        v.val = val
        v.tape = self.tape
        v.idx = self.tape.push(parents=parents, local_grads=grads)
        return v

    # -- Arithmetic --

    def __add__(self, other: Union[Variable, float, int]) -> Variable:
        if isinstance(other, Variable):
            return self._from_op(self.val + other.val, [self.idx, other.idx], [1.0, 1.0])
        return self._from_op(self.val + float(other), [self.idx], [1.0])

    def __radd__(self, other: Union[float, int]) -> Variable:
        return self._from_op(float(other) + self.val, [self.idx], [1.0])

    def __sub__(self, other: Union[Variable, float, int]) -> Variable:
        if isinstance(other, Variable):
            return self._from_op(self.val - other.val, [self.idx, other.idx], [1.0, -1.0])
        return self._from_op(self.val - float(other), [self.idx], [1.0])

    def __rsub__(self, other: Union[float, int]) -> Variable:
        return self._from_op(float(other) - self.val, [self.idx], [-1.0])

    def __mul__(self, other: Union[Variable, float, int]) -> Variable:
        if isinstance(other, Variable):
            return self._from_op(
                self.val * other.val,
                [self.idx, other.idx],
                [other.val, self.val],
            )
        c = float(other)
        return self._from_op(self.val * c, [self.idx], [c])

    def __rmul__(self, other: Union[float, int]) -> Variable:
        c = float(other)
        return self._from_op(c * self.val, [self.idx], [c])

    def __truediv__(self, other: Union[Variable, float, int]) -> Variable:
        if isinstance(other, Variable):
            return self._from_op(
                self.val / other.val,
                [self.idx, other.idx],
                [1.0 / other.val, -self.val / (other.val**2)],
            )
        c = float(other)
        return self._from_op(self.val / c, [self.idx], [1.0 / c])

    def __neg__(self) -> Variable:
        return self._from_op(-self.val, [self.idx], [-1.0])

    def __pow__(self, n: Union[Variable, float, int]) -> Variable:
        if isinstance(n, Variable):
            val = self.val ** n.val
            ln_a = math.log(abs(self.val) + 1e-30)
            return self._from_op(
                val,
                [self.idx, n.idx],
                [n.val * self.val ** (n.val - 1.0), val * ln_a],
            )
        p = float(n)
        return self._from_op(
            self.val**p,
            [self.idx],
            [p * self.val ** (p - 1.0)],
        )

    # -- Elementary functions --

    @staticmethod
    def sin(x: Variable) -> Variable:
        return x._from_op(math.sin(x.val), [x.idx], [math.cos(x.val)])

    @staticmethod
    def cos(x: Variable) -> Variable:
        return x._from_op(math.cos(x.val), [x.idx], [-math.sin(x.val)])

    @staticmethod
    def exp(x: Variable) -> Variable:
        e = math.exp(x.val)
        return x._from_op(e, [x.idx], [e])

    @staticmethod
    def log(x: Variable) -> Variable:
        return x._from_op(math.log(x.val), [x.idx], [1.0 / x.val])

    @staticmethod
    def sqrt(x: Variable) -> Variable:
        s = math.sqrt(x.val)
        return x._from_op(s, [x.idx], [0.5 / (s + 1e-30)])

    @staticmethod
    def tanh(x: Variable) -> Variable:
        t = math.tanh(x.val)
        return x._from_op(t, [x.idx], [1.0 - t * t])


def grad(
    f: Callable[..., Variable],
    x: NDArray,
) -> NDArray:
    """
    Compute gradient ∇f(x) via reverse-mode AD (single backward pass).

    Parameters:
        f: Function mapping (Variable, Variable, …) → Variable.
        x: (n,) evaluation point.

    Returns:
        (n,) gradient vector.
    """
    tape = Tape()
    variables = [Variable(x[i], tape) for i in range(len(x))]
    result = f(*variables)
    adjoints = tape.backward(result.idx)
    return np.array([adjoints[v.idx] for v in variables])


def jacobian(
    f: Callable[..., List[Variable]],
    x: NDArray,
) -> NDArray:
    """
    Compute Jacobian J[i,j] = ∂f_i/∂x_j via reverse-mode AD.

    Parameters:
        f: Function mapping (Variable, …) → list of Variable outputs.
        x: (n,) evaluation point.

    Returns:
        (m, n) Jacobian matrix.
    """
    tape = Tape()
    variables = [Variable(x[i], tape) for i in range(len(x))]
    outputs = f(*variables)

    m = len(outputs)
    n = len(x)
    J = np.zeros((m, n))

    for i, out in enumerate(outputs):
        adjoints = tape.backward(out.idx)
        for j, v in enumerate(variables):
            J[i, j] = adjoints[v.idx]

    return J
