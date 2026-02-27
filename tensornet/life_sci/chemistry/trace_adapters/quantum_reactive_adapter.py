"""
Quantum Reactive Scattering Trace Adapter (XV.5)
===================================================

Wraps tensornet.chemistry.quantum_reactive.TransitionStateTheory for STARK tracing.
Conservation: detailed balance, Arrhenius consistency.

© 2026 Tigantic Holdings LLC. All rights reserved. PROPRIETARY.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tensornet.core.trace import TraceSession


def _hash_scalar(v: float) -> str:
    return hashlib.sha256(np.array(v).tobytes()).hexdigest()


def _hash_array(arr: NDArray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@dataclass
class QuantumReactiveConservation:
    rate_300K: float
    rate_500K: float
    wigner_correction: float
    arrhenius_A: float
    arrhenius_Ea_eV: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rate_300K": self.rate_300K,
            "rate_500K": self.rate_500K,
            "wigner_correction": self.wigner_correction,
            "arrhenius_A": self.arrhenius_A,
            "arrhenius_Ea_eV": self.arrhenius_Ea_eV,
        }


class QuantumReactiveTraceAdapter:
    """
    Transition-state theory adapter with trace logging.

    Parameters
    ----------
    Ea : float
        Activation energy (eV).
    nu_imag : float
        Imaginary frequency at saddle (Hz).
    Q_ratio : float
        Partition function ratio Q‡/Q_R.
    """

    def __init__(
        self,
        Ea: float = 0.5,
        nu_imag: float = 1e13,
        Q_ratio: float = 1.0,
    ) -> None:
        from tensornet.life_sci.chemistry.quantum_reactive import TransitionStateTheory

        self.solver = TransitionStateTheory(Ea=Ea, nu_imag=nu_imag, Q_ratio=Q_ratio)
        self.Ea = Ea

    def evaluate(
        self,
        temperatures: NDArray | None = None,
    ) -> tuple[NDArray, NDArray, QuantumReactiveConservation, TraceSession]:
        """
        Evaluate TST rate constants over a range of temperatures.

        Parameters
        ----------
        temperatures : 1-D Array of temperatures (K).  Defaults to 200–1000 K.

        Returns
        -------
        temperatures, rates, conservation, session
        """
        session = TraceSession()

        session.log_custom(

            name="input_state",

            input_hashes=[],

            output_hashes=[],

            params={},

            metrics={},

        )

        if temperatures is None:
            temperatures = np.linspace(200, 1000, 50)

        rates = np.array([self.solver.rate_constant(float(T)) for T in temperatures])

        wigner = self.solver.wigner_correction(300.0)
        arr = self.solver.arrhenius_parameters(T1=300.0, T2=500.0)

        k300 = self.solver.rate_constant(300.0)
        k500 = self.solver.rate_constant(500.0)

        cons = QuantumReactiveConservation(
            rate_300K=k300,
            rate_500K=k500,
            wigner_correction=wigner,
            arrhenius_A=arr["A"],
            arrhenius_Ea_eV=arr["Ea_eV"],
        )

        session.log_custom(
            name="tst_evaluate",
            input_hashes=[_hash_array(temperatures)],
            output_hashes=[_hash_array(rates)],
            metrics=cons.to_dict(),
        )

        return temperatures, rates, cons, session
