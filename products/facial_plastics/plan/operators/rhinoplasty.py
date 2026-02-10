"""Rhinoplasty operator library — typed surgical primitives.

Each operator is a factory function that returns a fully-typed,
validated SurgicalOp with parameter definitions.

Operators map to real rhinoplasty maneuvers:
  - Dorsal reduction (hump removal)
  - Lateral/medial osteotomies
  - Septoplasty
  - Turbinate reduction
  - Tip support (columellar strut, shield graft)
  - Tip refinement (cephalic trim, suture techniques)
  - Alar base modification
  - Spreader grafts / valve support
  - Graft placement (costal, septal, ear cartilage)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.types import ProcedureType, StructureType, Vec3
from ..dsl import (
    CompositeOp,
    OpCategory,
    OperatorParam,
    ParamType,
    SequenceNode,
    SurgicalOp,
    SurgicalPlan,
)


# ── Operator definitions ──────────────────────────────────────────

def dorsal_reduction(
    amount_mm: float = 2.0,
    *,
    start_fraction: float = 0.0,
    end_fraction: float = 1.0,
    taper: bool = True,
) -> SurgicalOp:
    """Dorsal hump reduction — remove bone and/or cartilage from the dorsum.

    Parameters
    ----------
    amount_mm : float
        Maximum reduction depth in mm.
    start_fraction : float
        Fraction along dorsum where reduction starts (0=nasion, 1=tip).
    end_fraction : float
        Fraction along dorsum where reduction ends.
    taper : bool
        If True, taper smoothly at endpoints.
    """
    op = SurgicalOp(
        name="dorsal_reduction",
        category=OpCategory.REDUCTION,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "amount_mm": OperatorParam(
                "amount_mm", ParamType.FLOAT, "mm",
                "Maximum depth of dorsal reduction",
                default=2.0, min_value=0.5, max_value=8.0,
            ),
            "start_fraction": OperatorParam(
                "start_fraction", ParamType.FLOAT, "",
                "Start position along dorsum (0=nasion, 1=tip)",
                default=0.0, min_value=0.0, max_value=0.9,
            ),
            "end_fraction": OperatorParam(
                "end_fraction", ParamType.FLOAT, "",
                "End position along dorsum",
                default=1.0, min_value=0.1, max_value=1.0,
            ),
            "taper": OperatorParam(
                "taper", ParamType.BOOL, "",
                "Taper reduction at endpoints",
                default=True,
            ),
        },
        params={
            "amount_mm": amount_mm,
            "start_fraction": start_fraction,
            "end_fraction": end_fraction,
            "taper": taper,
        },
        affected_structures=[
            StructureType.BONE_NASAL,
            StructureType.CARTILAGE_UPPER_LATERAL,
            StructureType.SKIN_THICK,
        ],
        description="Remove dorsal hump by resecting bone and cartilage",
    )
    op.validate()
    return op


def lateral_osteotomy(
    side: str = "bilateral",
    *,
    angle_deg: float = 30.0,
    low_to_low: bool = True,
    continuous: bool = True,
) -> SurgicalOp:
    """Lateral osteotomy — controlled fracture of nasal sidewall bone.

    Parameters
    ----------
    side : str
        "left", "right", or "bilateral".
    angle_deg : float
        Osteotomy angle relative to piriform aperture.
    low_to_low : bool
        Low-to-low path (vs low-to-high).
    continuous : bool
        Continuous cut (vs perforating).
    """
    op = SurgicalOp(
        name="lateral_osteotomy",
        category=OpCategory.OSTEOTOMY,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side of osteotomy",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
            "angle_deg": OperatorParam(
                "angle_deg", ParamType.FLOAT, "deg",
                "Osteotomy angle",
                default=30.0, min_value=10.0, max_value=60.0,
            ),
            "low_to_low": OperatorParam(
                "low_to_low", ParamType.BOOL, "",
                "Low-to-low path",
                default=True,
            ),
            "continuous": OperatorParam(
                "continuous", ParamType.BOOL, "",
                "Continuous vs perforating cut",
                default=True,
            ),
        },
        params={
            "side": side,
            "angle_deg": angle_deg,
            "low_to_low": low_to_low,
            "continuous": continuous,
        },
        affected_structures=[StructureType.BONE_NASAL, StructureType.BONE_MAXILLA],
        description="Controlled fracture of nasal sidewall for narrowing",
    )
    op.validate()
    return op


def medial_osteotomy(
    side: str = "bilateral",
    *,
    height_mm: float = 10.0,
) -> SurgicalOp:
    """Medial osteotomy — cut along dorsal bone edge."""
    op = SurgicalOp(
        name="medial_osteotomy",
        category=OpCategory.OSTEOTOMY,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side", default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
            "height_mm": OperatorParam(
                "height_mm", ParamType.FLOAT, "mm",
                "Cut height from dorsum",
                default=10.0, min_value=3.0, max_value=20.0,
            ),
        },
        params={"side": side, "height_mm": height_mm},
        affected_structures=[StructureType.BONE_NASAL],
        description="Medial osteotomy along dorsal border",
    )
    op.validate()
    return op


def septoplasty(
    deviation_side: str = "left",
    *,
    resection_extent: str = "partial",
) -> SurgicalOp:
    """Septoplasty — correct deviated nasal septum."""
    op = SurgicalOp(
        name="septoplasty",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "deviation_side": OperatorParam(
                "deviation_side", ParamType.ENUM, "",
                "Side of septal deviation",
                default="left",
                enum_values=("left", "right", "s_shaped"),
            ),
            "resection_extent": OperatorParam(
                "resection_extent", ParamType.ENUM, "",
                "Extent of cartilage resection",
                default="partial",
                enum_values=("partial", "submucous", "extracorporeal"),
            ),
        },
        params={
            "deviation_side": deviation_side,
            "resection_extent": resection_extent,
        },
        affected_structures=[
            StructureType.CARTILAGE_SEPTUM,
            StructureType.AIRWAY_NASAL,
            StructureType.MUCOSA_NASAL,
        ],
        description="Correct deviated nasal septum",
    )
    op.validate()
    return op


def turbinate_reduction(
    side: str = "bilateral",
    *,
    method: str = "submucosal",
    reduction_pct: float = 30.0,
) -> SurgicalOp:
    """Inferior turbinate reduction."""
    op = SurgicalOp(
        name="turbinate_reduction",
        category=OpCategory.REDUCTION,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side", default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
            "method": OperatorParam(
                "method", ParamType.ENUM, "",
                "Surgical method",
                default="submucosal",
                enum_values=("submucosal", "cautery", "outfracture", "partial_resection"),
            ),
            "reduction_pct": OperatorParam(
                "reduction_pct", ParamType.FLOAT, "%",
                "Volume reduction percentage",
                default=30.0, min_value=10.0, max_value=70.0,
            ),
        },
        params={"side": side, "method": method, "reduction_pct": reduction_pct},
        affected_structures=[StructureType.TURBINATE_INFERIOR, StructureType.MUCOSA_NASAL],
        description="Reduce inferior turbinate volume",
    )
    op.validate()
    return op


def spreader_graft(
    side: str = "bilateral",
    *,
    length_mm: float = 18.0,
    width_mm: float = 3.0,
    thickness_mm: float = 1.5,
    source: str = "septal",
) -> SurgicalOp:
    """Spreader graft — widen internal nasal valve."""
    op = SurgicalOp(
        name="spreader_graft",
        category=OpCategory.GRAFT,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "side": OperatorParam(
                "side", ParamType.ENUM, "", "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
            "length_mm": OperatorParam(
                "length_mm", ParamType.FLOAT, "mm",
                "Graft length", default=18.0, min_value=8.0, max_value=30.0,
            ),
            "width_mm": OperatorParam(
                "width_mm", ParamType.FLOAT, "mm",
                "Graft width", default=3.0, min_value=1.0, max_value=6.0,
            ),
            "thickness_mm": OperatorParam(
                "thickness_mm", ParamType.FLOAT, "mm",
                "Graft thickness", default=1.5, min_value=0.5, max_value=3.0,
            ),
            "source": OperatorParam(
                "source", ParamType.ENUM, "",
                "Cartilage source", default="septal",
                enum_values=("septal", "ear", "costal"),
            ),
        },
        params={
            "side": side, "length_mm": length_mm,
            "width_mm": width_mm, "thickness_mm": thickness_mm,
            "source": source,
        },
        affected_structures=[
            StructureType.CARTILAGE_UPPER_LATERAL,
            StructureType.CARTILAGE_SEPTUM,
            StructureType.AIRWAY_NASAL,
        ],
        description="Insert spreader graft to support internal nasal valve",
    )
    op.validate()
    return op


def columellar_strut(
    *,
    length_mm: float = 20.0,
    width_mm: float = 3.0,
    thickness_mm: float = 2.0,
    source: str = "septal",
    floating: bool = False,
) -> SurgicalOp:
    """Columellar strut graft — tip support."""
    op = SurgicalOp(
        name="columellar_strut",
        category=OpCategory.GRAFT,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "length_mm": OperatorParam(
                "length_mm", ParamType.FLOAT, "mm",
                "Strut length", default=20.0, min_value=10.0, max_value=30.0,
            ),
            "width_mm": OperatorParam(
                "width_mm", ParamType.FLOAT, "mm",
                "Strut width", default=3.0, min_value=1.5, max_value=5.0,
            ),
            "thickness_mm": OperatorParam(
                "thickness_mm", ParamType.FLOAT, "mm",
                "Strut thickness", default=2.0, min_value=1.0, max_value=4.0,
            ),
            "source": OperatorParam(
                "source", ParamType.ENUM, "", "Source",
                default="septal",
                enum_values=("septal", "ear", "costal"),
            ),
            "floating": OperatorParam(
                "floating", ParamType.BOOL, "",
                "Floating (not fixed to ANS)", default=False,
            ),
        },
        params={
            "length_mm": length_mm, "width_mm": width_mm,
            "thickness_mm": thickness_mm, "source": source,
            "floating": floating,
        },
        affected_structures=[
            StructureType.CARTILAGE_LOWER_LATERAL,
            StructureType.CARTILAGE_SEPTUM,
        ],
        description="Place columellar strut for tip support",
    )
    op.validate()
    return op


def cephalic_trim(
    side: str = "bilateral",
    *,
    residual_strip_mm: float = 6.0,
) -> SurgicalOp:
    """Cephalic trim of lower lateral cartilage — tip refinement."""
    op = SurgicalOp(
        name="cephalic_trim",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "side": OperatorParam(
                "side", ParamType.ENUM, "", "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
            "residual_strip_mm": OperatorParam(
                "residual_strip_mm", ParamType.FLOAT, "mm",
                "Width of remaining cartilage strip",
                default=6.0, min_value=4.0, max_value=10.0,
            ),
        },
        params={"side": side, "residual_strip_mm": residual_strip_mm},
        affected_structures=[StructureType.CARTILAGE_LOWER_LATERAL],
        description="Remove cephalic portion of lower lateral cartilage",
    )
    op.validate()
    return op


def tip_suture(
    technique: str = "transdomal",
    *,
    tension: float = 0.5,
) -> SurgicalOp:
    """Tip suture technique — reshape tip cartilages."""
    op = SurgicalOp(
        name="tip_suture",
        category=OpCategory.SUTURE,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "technique": OperatorParam(
                "technique", ParamType.ENUM, "",
                "Suture technique",
                default="transdomal",
                enum_values=(
                    "transdomal", "interdomal", "lateral_crural_spanning",
                    "medial_crural_fixation", "tongue_in_groove",
                ),
            ),
            "tension": OperatorParam(
                "tension", ParamType.FLOAT, "",
                "Normalized suture tension (0=loose, 1=tight)",
                default=0.5, min_value=0.0, max_value=1.0,
            ),
        },
        params={"technique": technique, "tension": tension},
        affected_structures=[StructureType.CARTILAGE_LOWER_LATERAL],
        description="Suture technique for tip reshaping",
    )
    op.validate()
    return op


def shield_graft(
    *,
    height_mm: float = 8.0,
    width_mm: float = 6.0,
    thickness_mm: float = 1.5,
    source: str = "septal",
) -> SurgicalOp:
    """Shield graft at nasal tip — projection and definition."""
    op = SurgicalOp(
        name="shield_graft",
        category=OpCategory.GRAFT,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "height_mm": OperatorParam(
                "height_mm", ParamType.FLOAT, "mm",
                "Graft height", default=8.0, min_value=4.0, max_value=15.0,
            ),
            "width_mm": OperatorParam(
                "width_mm", ParamType.FLOAT, "mm",
                "Graft width", default=6.0, min_value=3.0, max_value=10.0,
            ),
            "thickness_mm": OperatorParam(
                "thickness_mm", ParamType.FLOAT, "mm",
                "Graft thickness", default=1.5, min_value=0.5, max_value=3.0,
            ),
            "source": OperatorParam(
                "source", ParamType.ENUM, "", "Source",
                default="septal",
                enum_values=("septal", "ear", "costal"),
            ),
        },
        params={
            "height_mm": height_mm, "width_mm": width_mm,
            "thickness_mm": thickness_mm, "source": source,
        },
        affected_structures=[StructureType.CARTILAGE_LOWER_LATERAL],
        description="Shield graft for tip projection and definition",
    )
    op.validate()
    return op


def alar_base_reduction(
    technique: str = "weir",
    *,
    amount_mm: float = 2.0,
    side: str = "bilateral",
) -> SurgicalOp:
    """Alar base reduction — narrow nostril base."""
    op = SurgicalOp(
        name="alar_base_reduction",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "technique": OperatorParam(
                "technique", ParamType.ENUM, "",
                "Excision technique",
                default="weir",
                enum_values=("weir", "sill", "combined"),
            ),
            "amount_mm": OperatorParam(
                "amount_mm", ParamType.FLOAT, "mm",
                "Excision width", default=2.0, min_value=0.5, max_value=5.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "", "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={"technique": technique, "amount_mm": amount_mm, "side": side},
        affected_structures=[StructureType.SKIN_THICK, StructureType.CARTILAGE_ALAR],
        description="Reduce alar base width",
    )
    op.validate()
    return op


def bone_infracture(
    side: str = "bilateral",
    *,
    displacement_mm: float = 2.0,
) -> SurgicalOp:
    """Controlled infracture of nasal bones after osteotomy."""
    op = SurgicalOp(
        name="bone_infracture",
        category=OpCategory.REPOSITIONING,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "side": OperatorParam(
                "side", ParamType.ENUM, "", "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
            "displacement_mm": OperatorParam(
                "displacement_mm", ParamType.FLOAT, "mm",
                "Medial displacement of bone segment",
                default=2.0, min_value=0.5, max_value=5.0,
            ),
        },
        params={"side": side, "displacement_mm": displacement_mm},
        affected_structures=[StructureType.BONE_NASAL],
        description="Infracture nasal sidewall bones medially",
    )
    op.validate()
    return op


def cartilage_scoring(
    structure: str = "lower_lateral_cartilage",
    *,
    depth_fraction: float = 0.5,
    n_scores: int = 3,
) -> SurgicalOp:
    """Score cartilage to allow controlled bending."""
    op = SurgicalOp(
        name="cartilage_scoring",
        category=OpCategory.SCORING,
        procedure=ProcedureType.RHINOPLASTY,
        param_defs={
            "structure": OperatorParam(
                "structure", ParamType.ENUM, "",
                "Cartilage to score",
                default="lower_lateral_cartilage",
                enum_values=(
                    "lower_lateral_cartilage", "upper_lateral_cartilage",
                    "septal_cartilage",
                ),
            ),
            "depth_fraction": OperatorParam(
                "depth_fraction", ParamType.FLOAT, "",
                "Scoring depth as fraction of thickness",
                default=0.5, min_value=0.2, max_value=0.8,
            ),
            "n_scores": OperatorParam(
                "n_scores", ParamType.INT, "",
                "Number of parallel score lines",
                default=3, min_value=1, max_value=8,
            ),
        },
        params={
            "structure": structure,
            "depth_fraction": depth_fraction,
            "n_scores": n_scores,
        },
        affected_structures=[StructureType.CARTILAGE_LOWER_LATERAL],
        description="Score cartilage for controlled bending",
    )
    op.validate()
    return op


# ── Operator registry ─────────────────────────────────────────────

RHINOPLASTY_OPERATORS = {
    "dorsal_reduction": dorsal_reduction,
    "lateral_osteotomy": lateral_osteotomy,
    "medial_osteotomy": medial_osteotomy,
    "septoplasty": septoplasty,
    "turbinate_reduction": turbinate_reduction,
    "spreader_graft": spreader_graft,
    "columellar_strut": columellar_strut,
    "cephalic_trim": cephalic_trim,
    "tip_suture": tip_suture,
    "shield_graft": shield_graft,
    "alar_base_reduction": alar_base_reduction,
    "bone_infracture": bone_infracture,
    "cartilage_scoring": cartilage_scoring,
}


# ── Common rhinoplasty plan builder ──────────────────────────────

class RhinoplastyPlanBuilder:
    """Convenience builder for common rhinoplasty plan templates.

    Provides high-level plan construction from clinical parameters.
    """

    @staticmethod
    def reduction_rhinoplasty(
        *,
        dorsal_reduction_mm: float = 2.0,
        osteotomy_angle: float = 30.0,
        spreader_grafts: bool = True,
        septoplasty_needed: bool = False,
        tip_work: str = "minimal",  # minimal, moderate, extensive
        alar_base_reduction_mm: float = 0.0,
    ) -> SurgicalPlan:
        """Build a reduction rhinoplasty plan.

        This is the most common rhinoplasty variant:
        dorsal reduction + osteotomies + optional tip work.
        """
        plan = SurgicalPlan(
            name="reduction_rhinoplasty",
            procedure=ProcedureType.RHINOPLASTY,
            description="Reduction rhinoplasty with osteotomies",
        )

        # Step 1: Dorsal reduction
        plan.add_step(dorsal_reduction(dorsal_reduction_mm))

        # Step 2: Septoplasty if needed
        if septoplasty_needed:
            plan.add_step(septoplasty())

        # Step 3: Osteotomies
        plan.add_step(medial_osteotomy())
        plan.add_step(lateral_osteotomy(angle_deg=osteotomy_angle))
        plan.add_step(bone_infracture())

        # Step 4: Spreader grafts
        if spreader_grafts:
            plan.add_step(spreader_graft())

        # Step 5: Tip work
        if tip_work in ("moderate", "extensive"):
            plan.add_step(cephalic_trim())
            plan.add_step(tip_suture(technique="transdomal"))
        if tip_work == "extensive":
            plan.add_step(columellar_strut())
            plan.add_step(shield_graft())

        # Step 6: Alar base
        if alar_base_reduction_mm > 0:
            plan.add_step(alar_base_reduction(amount_mm=alar_base_reduction_mm))

        return plan

    @staticmethod
    def functional_rhinoplasty(
        *,
        septoplasty_extent: str = "partial",
        turbinate_method: str = "submucosal",
        spreader_grafts: bool = True,
    ) -> SurgicalPlan:
        """Build a functional rhinoplasty plan (airway-focused)."""
        plan = SurgicalPlan(
            name="functional_rhinoplasty",
            procedure=ProcedureType.RHINOPLASTY,
            description="Functional rhinoplasty for airway improvement",
        )

        plan.add_step(septoplasty(resection_extent=septoplasty_extent))
        plan.add_step(turbinate_reduction(method=turbinate_method))

        if spreader_grafts:
            plan.add_step(spreader_graft())

        return plan

    @staticmethod
    def tip_rhinoplasty(
        *,
        cephalic_trim_width_mm: float = 6.0,
        suture_technique: str = "transdomal",
        strut: bool = True,
        shield: bool = False,
    ) -> SurgicalPlan:
        """Build a tip-only rhinoplasty plan."""
        plan = SurgicalPlan(
            name="tip_rhinoplasty",
            procedure=ProcedureType.RHINOPLASTY,
            description="Tip rhinoplasty for nasal tip refinement",
        )

        plan.add_step(cephalic_trim(residual_strip_mm=cephalic_trim_width_mm))
        plan.add_step(tip_suture(technique=suture_technique))

        if strut:
            plan.add_step(columellar_strut())
        if shield:
            plan.add_step(shield_graft())

        return plan
