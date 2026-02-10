"""Blepharoplasty operator library — upper and lower eyelid surgery primitives.

Operators model real blepharoplasty maneuvers:
  - Upper lid skin/muscle excision
  - Upper lid fat pad removal (medial, central, lateral)
  - Lower lid skin excision (transcutaneous / skin-muscle flap)
  - Lower lid fat transposition (transconjunctival repositioning)
  - Canthopexy (lateral canthal tendon tightening)
  - Orbicularis oculi tightening
  - Skin pinch technique

References:
  - Codner & McCord 2008 (eyelid surgery);
  - Rohrich & Pessa 2009 (periorbital fat anatomy);
  - Goldberg 2014 (transconjunctival lower blepharoplasty).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ...core.types import ProcedureType, StructureType
from ..dsl import (
    OpCategory,
    OperatorParam,
    ParamType,
    SurgicalOp,
    SurgicalPlan,
)


# ── Operator definitions ──────────────────────────────────────────


def upper_lid_skin_excision(
    *,
    width_mm: float = 12.0,
    include_orbicularis: bool = True,
    crease_height_mm: float = 9.0,
    side: str = "bilateral",
) -> SurgicalOp:
    """Upper eyelid skin (and optional orbicularis) excision.

    Parameters
    ----------
    width_mm : float
        Maximum skin strip width at widest point.
    include_orbicularis : bool
        Include a strip of orbicularis oculi muscle.
    crease_height_mm : float
        Desired lid crease height (measured from lash line).
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="upper_lid_skin_excision",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.BLEPHAROPLASTY_UPPER,
        param_defs={
            "width_mm": OperatorParam(
                "width_mm", ParamType.FLOAT, "mm",
                "Maximum skin strip width",
                default=12.0, min_value=4.0, max_value=22.0,
            ),
            "include_orbicularis": OperatorParam(
                "include_orbicularis", ParamType.BOOL, "",
                "Include orbicularis muscle strip",
                default=True,
            ),
            "crease_height_mm": OperatorParam(
                "crease_height_mm", ParamType.FLOAT, "mm",
                "Desired lid crease height from lash line",
                default=9.0, min_value=6.0, max_value=14.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "width_mm": width_mm,
            "include_orbicularis": include_orbicularis,
            "crease_height_mm": crease_height_mm,
            "side": side,
        },
        affected_structures=[
            StructureType.SKIN_THIN,
            StructureType.MUSCLE_ORBICULARIS,
            StructureType.FAT_ORBITAL,
        ],
        description="Upper eyelid skin excision for blepharoplasty",
    )
    op.validate()
    return op


def upper_lid_fat_removal(
    *,
    medial_pad_cc: float = 0.3,
    central_pad_cc: float = 0.2,
    lateral_pad: bool = False,
    lateral_pad_cc: float = 0.0,
    side: str = "bilateral",
) -> SurgicalOp:
    """Upper eyelid fat pad debulking.

    Parameters
    ----------
    medial_pad_cc : float
        Volume removed from medial (nasal) fat pad in cc.
    central_pad_cc : float
        Volume removed from central (pre-aponeurotic) fat pad.
    lateral_pad : bool
        Address the lacrimal gland / lateral fat pad.
    lateral_pad_cc : float
        Volume removed from lateral pad (if lateral_pad is True).
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="upper_lid_fat_removal",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.BLEPHAROPLASTY_UPPER,
        param_defs={
            "medial_pad_cc": OperatorParam(
                "medial_pad_cc", ParamType.FLOAT, "cc",
                "Volume removed from medial fat pad",
                default=0.3, min_value=0.0, max_value=1.0,
            ),
            "central_pad_cc": OperatorParam(
                "central_pad_cc", ParamType.FLOAT, "cc",
                "Volume from central pre-aponeurotic pad",
                default=0.2, min_value=0.0, max_value=1.0,
            ),
            "lateral_pad": OperatorParam(
                "lateral_pad", ParamType.BOOL, "",
                "Address lateral fat pad",
                default=False,
            ),
            "lateral_pad_cc": OperatorParam(
                "lateral_pad_cc", ParamType.FLOAT, "cc",
                "Volume from lateral pad",
                default=0.0, min_value=0.0, max_value=0.5,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "medial_pad_cc": medial_pad_cc,
            "central_pad_cc": central_pad_cc,
            "lateral_pad": lateral_pad,
            "lateral_pad_cc": lateral_pad_cc,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_ORBITAL,
            StructureType.FAT_PREAPONEUROTIC,
        ],
        description="Upper eyelid fat pad debulking",
    )
    op.validate()
    return op


def lower_lid_skin_excision(
    *,
    approach: str = "subciliary",
    width_mm: float = 4.0,
    include_orbicularis: bool = True,
    side: str = "bilateral",
) -> SurgicalOp:
    """Lower eyelid skin excision — transcutaneous approach.

    Parameters
    ----------
    approach : str
        "subciliary" (skin-muscle flap) or "mid_lid" (mid-lid crease).
    width_mm : float
        Maximum skin strip width.
    include_orbicularis : bool
        Include orbicularis muscle in the flap.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="lower_lid_skin_excision",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.BLEPHAROPLASTY_LOWER,
        param_defs={
            "approach": OperatorParam(
                "approach", ParamType.ENUM, "",
                "Transcutaneous approach",
                default="subciliary",
                enum_values=("subciliary", "mid_lid"),
            ),
            "width_mm": OperatorParam(
                "width_mm", ParamType.FLOAT, "mm",
                "Maximum skin strip width",
                default=4.0, min_value=1.0, max_value=10.0,
            ),
            "include_orbicularis": OperatorParam(
                "include_orbicularis", ParamType.BOOL, "",
                "Include orbicularis in flap",
                default=True,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "approach": approach,
            "width_mm": width_mm,
            "include_orbicularis": include_orbicularis,
            "side": side,
        },
        affected_structures=[
            StructureType.SKIN_THIN,
            StructureType.MUSCLE_ORBICULARIS,
        ],
        description="Lower eyelid skin excision (transcutaneous approach)",
    )
    op.validate()
    return op


def lower_lid_fat_transposition(
    *,
    approach: str = "transconjunctival",
    medial_pad: bool = True,
    central_pad: bool = True,
    lateral_pad: bool = False,
    side: str = "bilateral",
) -> SurgicalOp:
    """Lower lid fat transposition — reposition pseudoherniated fat.

    Parameters
    ----------
    approach : str
        "transconjunctival" or "transcutaneous".
    medial_pad : bool
        Transpose medial fat pad over infraorbital rim.
    central_pad : bool
        Transpose central fat pad.
    lateral_pad : bool
        Transpose lateral fat pad.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="lower_lid_fat_transposition",
        category=OpCategory.REPOSITIONING,
        procedure=ProcedureType.BLEPHAROPLASTY_LOWER,
        param_defs={
            "approach": OperatorParam(
                "approach", ParamType.ENUM, "",
                "Surgical approach",
                default="transconjunctival",
                enum_values=("transconjunctival", "transcutaneous"),
            ),
            "medial_pad": OperatorParam(
                "medial_pad", ParamType.BOOL, "",
                "Transpose medial fat pad",
                default=True,
            ),
            "central_pad": OperatorParam(
                "central_pad", ParamType.BOOL, "",
                "Transpose central fat pad",
                default=True,
            ),
            "lateral_pad": OperatorParam(
                "lateral_pad", ParamType.BOOL, "",
                "Transpose lateral fat pad",
                default=False,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "approach": approach,
            "medial_pad": medial_pad,
            "central_pad": central_pad,
            "lateral_pad": lateral_pad,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_ORBITAL,
            StructureType.FAT_PREAPONEUROTIC,
            StructureType.BONE_ORBIT,
            StructureType.PERIOSTEUM,
        ],
        description="Lower lid orbital fat transposition over infraorbital rim",
    )
    op.validate()
    return op


def canthopexy(
    *,
    technique: str = "lateral_retinacular",
    vector: str = "superolateral",
    side: str = "bilateral",
) -> SurgicalOp:
    """Canthopexy — tighten lateral canthal tendon.

    Prevents lower lid retraction / scleral show after blepharoplasty.

    Parameters
    ----------
    technique : str
        "lateral_retinacular" (minimal), "tarsal_strip" (canthopexy/plasty),
        or "drill_hole" (canthoplasty).
    vector : str
        "superolateral" or "lateral".
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="canthopexy",
        category=OpCategory.SUTURE,
        procedure=ProcedureType.BLEPHAROPLASTY_LOWER,
        param_defs={
            "technique": OperatorParam(
                "technique", ParamType.ENUM, "",
                "Canthopexy technique",
                default="lateral_retinacular",
                enum_values=("lateral_retinacular", "tarsal_strip", "drill_hole"),
            ),
            "vector": OperatorParam(
                "vector", ParamType.ENUM, "",
                "Fixation vector",
                default="superolateral",
                enum_values=("superolateral", "lateral"),
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "technique": technique,
            "vector": vector,
            "side": side,
        },
        affected_structures=[
            StructureType.MUSCLE_ORBICULARIS,
            StructureType.BONE_ORBIT,
            StructureType.PERIOSTEUM,
        ],
        description="Lateral canthal tendon tightening (canthopexy)",
    )
    op.validate()
    return op


def orbicularis_tightening(
    *,
    technique: str = "muscle_flap",
    side: str = "bilateral",
) -> SurgicalOp:
    """Orbicularis oculi muscle tightening.

    Parameters
    ----------
    technique : str
        "muscle_flap" (orbicularis flap suspension) or
        "plication" (direct suture plication).
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="orbicularis_tightening",
        category=OpCategory.SUTURE,
        procedure=ProcedureType.BLEPHAROPLASTY_LOWER,
        param_defs={
            "technique": OperatorParam(
                "technique", ParamType.ENUM, "",
                "Muscle tightening technique",
                default="muscle_flap",
                enum_values=("muscle_flap", "plication"),
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "technique": technique,
            "side": side,
        },
        affected_structures=[
            StructureType.MUSCLE_ORBICULARIS,
            StructureType.SKIN_THIN,
        ],
        description="Orbicularis oculi muscle tightening",
    )
    op.validate()
    return op


def skin_pinch(
    *,
    width_mm: float = 2.0,
    side: str = "bilateral",
) -> SurgicalOp:
    """Lower eyelid skin pinch — conservative skin excision.

    Parameters
    ----------
    width_mm : float
        Width of pinched skin strip in mm. Conservative (1-4 mm).
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="skin_pinch",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.BLEPHAROPLASTY_LOWER,
        param_defs={
            "width_mm": OperatorParam(
                "width_mm", ParamType.FLOAT, "mm",
                "Skin pinch strip width",
                default=2.0, min_value=0.5, max_value=5.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "width_mm": width_mm,
            "side": side,
        },
        affected_structures=[
            StructureType.SKIN_THIN,
        ],
        description="Conservative lower lid skin pinch excision",
    )
    op.validate()
    return op


# ── Operator registry ─────────────────────────────────────────────

BLEPHAROPLASTY_OPERATORS = {
    "upper_lid_skin_excision": upper_lid_skin_excision,
    "upper_lid_fat_removal": upper_lid_fat_removal,
    "lower_lid_skin_excision": lower_lid_skin_excision,
    "lower_lid_fat_transposition": lower_lid_fat_transposition,
    "canthopexy": canthopexy,
    "orbicularis_tightening": orbicularis_tightening,
    "skin_pinch": skin_pinch,
}


# ── Plan builder templates ────────────────────────────────────────

class BlepharoplastyPlanBuilder:
    """Build pre-configured blepharoplasty plans."""

    @staticmethod
    def upper_blepharoplasty(
        *,
        width_mm: float = 12.0,
        remove_fat: bool = True,
        crease_height_mm: float = 9.0,
    ) -> SurgicalPlan:
        """Build an upper blepharoplasty plan."""
        plan = SurgicalPlan(
            name="upper_blepharoplasty",
            procedure=ProcedureType.BLEPHAROPLASTY_UPPER,
            description="Upper blepharoplasty with skin excision and optional fat removal",
        )

        plan.add_step(upper_lid_skin_excision(
            width_mm=width_mm,
            crease_height_mm=crease_height_mm,
        ))

        if remove_fat:
            plan.add_step(upper_lid_fat_removal())

        return plan

    @staticmethod
    def transconjunctival_lower_blepharoplasty(
        *,
        medial_pad: bool = True,
        central_pad: bool = True,
        lateral_pad: bool = False,
        skin_pinch_width_mm: float = 2.0,
        canthopexy_needed: bool = False,
    ) -> SurgicalPlan:
        """Build a transconjunctival lower blepharoplasty plan."""
        plan = SurgicalPlan(
            name="transconjunctival_lower_blepharoplasty",
            procedure=ProcedureType.BLEPHAROPLASTY_LOWER,
            description="Transconjunctival lower blepharoplasty with fat transposition",
        )

        plan.add_step(lower_lid_fat_transposition(
            approach="transconjunctival",
            medial_pad=medial_pad,
            central_pad=central_pad,
            lateral_pad=lateral_pad,
        ))

        if canthopexy_needed:
            plan.add_step(canthopexy())

        plan.add_step(skin_pinch(width_mm=skin_pinch_width_mm))

        return plan

    @staticmethod
    def transcutaneous_lower_blepharoplasty(
        *,
        width_mm: float = 4.0,
        fat_transposition: bool = True,
        canthopexy_needed: bool = True,
    ) -> SurgicalPlan:
        """Build a transcutaneous lower blepharoplasty plan."""
        plan = SurgicalPlan(
            name="transcutaneous_lower_blepharoplasty",
            procedure=ProcedureType.BLEPHAROPLASTY_LOWER,
            description="Transcutaneous lower blepharoplasty with skin-muscle flap",
        )

        plan.add_step(lower_lid_skin_excision(
            approach="subciliary",
            width_mm=width_mm,
        ))

        if fat_transposition:
            plan.add_step(lower_lid_fat_transposition(approach="transcutaneous"))

        plan.add_step(orbicularis_tightening())

        if canthopexy_needed:
            plan.add_step(canthopexy(technique="tarsal_strip"))

        return plan

    @staticmethod
    def four_lid_blepharoplasty(
        *,
        upper_width_mm: float = 12.0,
        lower_approach: str = "transconjunctival",
        canthopexy_needed: bool = True,
    ) -> SurgicalPlan:
        """Build a combined upper + lower blepharoplasty plan."""
        plan = SurgicalPlan(
            name="four_lid_blepharoplasty",
            procedure=ProcedureType.BLEPHAROPLASTY_UPPER,
            description="Four-lid blepharoplasty (upper and lower bilateral)",
        )

        # Upper lids
        plan.add_step(upper_lid_skin_excision(width_mm=upper_width_mm))
        plan.add_step(upper_lid_fat_removal())

        # Lower lids
        plan.add_step(lower_lid_fat_transposition(approach=lower_approach))

        if canthopexy_needed:
            plan.add_step(canthopexy())

        plan.add_step(skin_pinch())

        return plan
