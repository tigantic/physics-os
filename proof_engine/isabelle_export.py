"""
Isabelle/HOL Proof Backend
============================

Export HyperTensor certificates to Isabelle/HOL format (.thy files).

Isabelle/HOL uses Higher-Order Logic with a powerful simplifier
(simp), Isar structured proofs, and extensive standard library
(HOL-Analysis, HOL-Algebra).

Provides:
- IsabelleTheory: structured theory representation
- IsabelleExporter: Certificate → .thy file generation
- Witness encoding in HOL integer/rational arithmetic
- Structured Isar proof blocks
- Session ROOT file generation for project build

Compatible with Isabelle 2024+ and AFP (Archive of Formal Proofs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Isabelle theory representation
# ---------------------------------------------------------------------------

@dataclass
class IsabelleLemma:
    """A lemma in Isabelle/HOL Isar format."""

    name: str
    statement: str
    proof: str = "by simp"

    def to_isar(self) -> str:
        return f'lemma {self.name}: "{self.statement}"\n  {self.proof}'


@dataclass
class IsabelleDefinition:
    """An Isabelle definition."""

    name: str
    type_sig: str
    body: str

    def to_isar(self) -> str:
        return (
            f'definition {self.name} :: "{self.type_sig}" where\n'
            f'  "{self.name} = {self.body}"'
        )


@dataclass
class IsabelleTheory:
    """A complete Isabelle .thy file."""

    name: str
    imports: List[str] = field(default_factory=lambda: ["Main"])
    definitions: List[IsabelleDefinition] = field(default_factory=list)
    lemmas: List[IsabelleLemma] = field(default_factory=list)
    raw_blocks: List[str] = field(default_factory=list)

    def to_thy(self) -> str:
        """Generate Isabelle/HOL theory source."""
        parts: List[str] = []
        parts.append(f"theory {self.name}")

        imports_str = " ".join(f'"{i}"' if "/" in i or "-" in i else i for i in self.imports)
        parts.append(f"  imports {imports_str}")
        parts.append("begin")
        parts.append("")

        for d in self.definitions:
            parts.append(d.to_isar())
            parts.append("")

        for block in self.raw_blocks:
            parts.append(block)
            parts.append("")

        for lem in self.lemmas:
            parts.append(lem.to_isar())
            parts.append("")

        parts.append("end")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Witness encoding
# ---------------------------------------------------------------------------

def float_to_nat(x: float, precision: int = 32) -> int:
    """Encode non-negative float as scaled natural number."""
    scale = 2 ** precision
    return max(0, int(round(x * scale)))


def float_to_int(x: float, precision: int = 32) -> int:
    """Encode float as scaled integer."""
    scale = 2 ** precision
    return int(round(x * scale))


def encode_list_nat(values: Sequence[float], precision: int = 32) -> str:
    """Encode a float sequence as an Isabelle nat list."""
    ints = [float_to_nat(v, precision) for v in values]
    return "[" + ", ".join(str(i) for i in ints) + "]"


# ---------------------------------------------------------------------------
# Certificate → Isabelle exporter
# ---------------------------------------------------------------------------

class IsabelleExporter:
    """Export HyperTensor certificates to Isabelle/HOL theories.

    Generates .thy files with decidable witness proofs
    (simp, arith, eval).
    """

    def __init__(self, precision: int = 32) -> None:
        self._precision = precision

    def export_interval_bound(
        self,
        theory_name: str,
        name: str,
        value: float,
        lower: float,
        upper: float,
        description: str = "",
    ) -> IsabelleTheory:
        """Generate a theory proving lower ≤ value ≤ upper."""
        scale = 2 ** self._precision
        val_int = float_to_int(value, self._precision)
        lo_int = int(np.floor(lower * scale))
        hi_int = int(np.ceil(upper * scale))

        theory = IsabelleTheory(
            name=theory_name,
            imports=["Main"],
        )

        if description:
            theory.raw_blocks.append(f"(* {description} *)")

        theory.definitions.extend([
            IsabelleDefinition(f"{name}_value", "int", str(val_int)),
            IsabelleDefinition(f"{name}_lower", "int", str(lo_int)),
            IsabelleDefinition(f"{name}_upper", "int", str(hi_int)),
        ])

        theory.lemmas.extend([
            IsabelleLemma(
                f"{name}_lower_bound",
                f"{name}_lower \\<le> {name}_value",
                f"by (simp add: {name}_lower_def {name}_value_def)",
            ),
            IsabelleLemma(
                f"{name}_upper_bound",
                f"{name}_value \\<le> {name}_upper",
                f"by (simp add: {name}_value_def {name}_upper_def)",
            ),
        ])

        return theory

    def export_monotone_sequence(
        self,
        theory_name: str,
        name: str,
        values: Sequence[float],
        description: str = "",
    ) -> IsabelleTheory:
        """Prove a sequence is monotonically decreasing."""
        int_vals = [float_to_nat(v, self._precision) for v in values]

        theory = IsabelleTheory(
            name=theory_name,
            imports=["Main"],
        )

        if description:
            theory.raw_blocks.append(f"(* {description} *)")

        list_str = "[" + ", ".join(str(v) for v in int_vals) + "] :: nat list"
        theory.definitions.append(
            IsabelleDefinition(f"{name}_seq", "nat list", list_str)
        )

        for i in range(len(int_vals) - 1):
            theory.lemmas.append(IsabelleLemma(
                f"{name}_decrease_{i}",
                f"{name}_seq ! {i + 1} \\<le> {name}_seq ! {i}",
                f"by (simp add: {name}_seq_def)",
            ))

        return theory

    def export_conservation(
        self,
        theory_name: str,
        name: str,
        initial: float,
        final: float,
        tolerance: float,
        description: str = "",
    ) -> IsabelleTheory:
        """Prove a conserved quantity stays within tolerance."""
        scale = 2 ** self._precision
        init_int = float_to_int(initial, self._precision)
        final_int = float_to_int(final, self._precision)
        tol_int = int(np.ceil(tolerance * scale))

        theory = IsabelleTheory(
            name=theory_name,
            imports=["Main"],
        )

        if description:
            theory.raw_blocks.append(f"(* {description} *)")

        theory.definitions.extend([
            IsabelleDefinition(f"{name}_initial", "int", str(init_int)),
            IsabelleDefinition(f"{name}_final", "int", str(final_int)),
            IsabelleDefinition(f"{name}_tolerance", "int", str(tol_int)),
        ])

        theory.lemmas.append(IsabelleLemma(
            f"{name}_conserved",
            f"\\<bar>{name}_final - {name}_initial\\<bar> \\<le> {name}_tolerance",
            f"by (simp add: {name}_initial_def {name}_final_def {name}_tolerance_def)",
        ))

        return theory

    def export_energy_bound(
        self,
        theory_name: str,
        name: str,
        energies: Sequence[float],
        bound: float,
        description: str = "",
    ) -> IsabelleTheory:
        """Prove all energy values stay below a bound."""
        scale = 2 ** self._precision
        bound_int = int(np.ceil(bound * scale))
        energy_ints = [float_to_nat(e, self._precision) for e in energies]

        theory = IsabelleTheory(
            name=theory_name,
            imports=["Main"],
        )

        if description:
            theory.raw_blocks.append(f"(* {description} *)")

        theory.definitions.append(
            IsabelleDefinition(f"{name}_bound", "nat", str(bound_int))
        )

        list_str = "[" + ", ".join(str(e) for e in energy_ints) + "] :: nat list"
        theory.definitions.append(
            IsabelleDefinition(f"{name}_energies", "nat list", list_str)
        )

        # Global bound
        theory.lemmas.append(IsabelleLemma(
            f"{name}_bounded",
            f"\\<forall>e \\<in> set {name}_energies. e \\<le> {name}_bound",
            f"by (simp add: {name}_energies_def {name}_bound_def)",
        ))

        return theory

    def export_certificate(self, cert: Dict[str, Any]) -> str:
        """Export a generic certificate dict to Isabelle/HOL."""
        theory = IsabelleTheory(
            name=cert.get("name", "Certificate"),
            imports=["Main", '"HOL-Analysis.Analysis"'],
        )

        theory.raw_blocks.append("(* HyperTensor Certificate Export *)")

        for claim in cert.get("claims", []):
            sub = self.export_interval_bound(
                theory_name="_",
                name=claim["name"],
                value=claim["value"],
                lower=claim["lower"],
                upper=claim["upper"],
                description=claim.get("description", ""),
            )
            theory.definitions.extend(sub.definitions)
            theory.lemmas.extend(sub.lemmas)

        return theory.to_thy()

    def generate_session(
        self,
        output_dir: Union[str, Path],
        theories: Dict[str, IsabelleTheory],
        session_name: str = "HyperTensor",
    ) -> Path:
        """Generate a full Isabelle session with ROOT file.

        Parameters
        ----------
        output_dir : directory for output files
        theories : dict of theory_name → IsabelleTheory
        session_name : session name

        Returns
        -------
        Path to output directory
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ROOT file
        theory_names = sorted(theories.keys())
        root_lines = [
            f"session {session_name} = HOL +",
            f'  options [document = false]',
            f"  theories",
        ]
        for t in theory_names:
            root_lines.append(f"    {t}")
        (out / "ROOT").write_text("\n".join(root_lines) + "\n")

        # .thy files
        for t_name, theory in theories.items():
            (out / f"{t_name}.thy").write_text(theory.to_thy())

        logger.info("Generated Isabelle session in %s with %d theories", out, len(theories))
        return out


__all__ = [
    "IsabelleLemma",
    "IsabelleDefinition",
    "IsabelleTheory",
    "IsabelleExporter",
    "float_to_nat",
    "float_to_int",
    "encode_list_nat",
]
