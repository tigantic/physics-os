"""Facelift and necklift operator library — typed surgical primitives.

Operators map to real rhytidectomy maneuvers:
  - SMAS plication (tightening without flap elevation)
  - SMAS flap / SMASectomy
  - Deep plane dissection (sub-SMAS with retaining ligament release)
  - Skin excision and redraping
  - Fat compartment repositioning (malar fat pad descent correction)
  - Platysma plication (midline and lateral band correction)
  - Submentoplasty (liposuction + platysma + submental skin)
  - Malar fat pad suspension

References:
  - Hamra 1992 (deep plane); Baker 1997 (lateral SMASectomy);
  - Mendelson & Wong 2012 (facial spaces); Rohrich & Ghavami 2015
    (dual-plane facelift component approach).
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


def smas_plication(
    vector_deg: float = 60.0,
    *,
    plication_width_mm: float = 10.0,
    side: str = "bilateral",
    suture_type: str = "permanent",
) -> SurgicalOp:
    """SMAS plication — tighten SMAS layer without flap elevation.

    Parameters
    ----------
    vector_deg : float
        Plication vector angle (0=lateral, 90=superior). Typical 45-75.
    plication_width_mm : float
        Width of tissue folded over in mm.
    side : str
        "left", "right", or "bilateral".
    suture_type : str
        "permanent" or "absorbable".
    """
    op = SurgicalOp(
        name="smas_plication",
        category=OpCategory.SUTURE,
        procedure=ProcedureType.FACELIFT,
        param_defs={
            "vector_deg": OperatorParam(
                "vector_deg", ParamType.FLOAT, "deg",
                "Superolateral plication vector angle",
                default=60.0, min_value=20.0, max_value=90.0,
            ),
            "plication_width_mm": OperatorParam(
                "plication_width_mm", ParamType.FLOAT, "mm",
                "Width of SMAS folded and sutured",
                default=10.0, min_value=3.0, max_value=25.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side of face",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
            "suture_type": OperatorParam(
                "suture_type", ParamType.ENUM, "",
                "Suture material permanence",
                default="permanent",
                enum_values=("permanent", "absorbable"),
            ),
        },
        params={
            "vector_deg": vector_deg,
            "plication_width_mm": plication_width_mm,
            "side": side,
            "suture_type": suture_type,
        },
        affected_structures=[
            StructureType.SMAS,
            StructureType.SKIN_ENVELOPE,
            StructureType.FAT_SUBCUTANEOUS,
        ],
        description="SMAS plication along superolateral vector",
    )
    op.validate()
    return op


def smas_flap(
    flap_type: str = "lateral_smasectomy",
    *,
    elevation_extent_mm: float = 40.0,
    resection_width_mm: float = 15.0,
    side: str = "bilateral",
) -> SurgicalOp:
    """SMAS flap elevation and SMASectomy.

    Parameters
    ----------
    flap_type : str
        "lateral_smasectomy" (Baker), "extended_smas" (Stuzin),
        or "high_smas".
    elevation_extent_mm : float
        Extent of SMAS flap undermining in mm.
    resection_width_mm : float
        Width of SMAS strip excised (SMASectomy) in mm.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="smas_flap",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.FACELIFT,
        param_defs={
            "flap_type": OperatorParam(
                "flap_type", ParamType.ENUM, "",
                "SMAS flap technique",
                default="lateral_smasectomy",
                enum_values=("lateral_smasectomy", "extended_smas", "high_smas"),
            ),
            "elevation_extent_mm": OperatorParam(
                "elevation_extent_mm", ParamType.FLOAT, "mm",
                "Extent of SMAS undermining",
                default=40.0, min_value=15.0, max_value=80.0,
            ),
            "resection_width_mm": OperatorParam(
                "resection_width_mm", ParamType.FLOAT, "mm",
                "Width of SMAS strip excised",
                default=15.0, min_value=5.0, max_value=30.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side of face",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "flap_type": flap_type,
            "elevation_extent_mm": elevation_extent_mm,
            "resection_width_mm": resection_width_mm,
            "side": side,
        },
        affected_structures=[
            StructureType.SMAS,
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.FAT_MALAR,
            StructureType.MUSCLE_MIMETIC,
        ],
        description="SMAS flap elevation with controlled resection",
    )
    op.validate()
    return op


def deep_plane_dissection(
    *,
    release_zygomatic: bool = True,
    release_mandibular: bool = True,
    release_cervical: bool = False,
    side: str = "bilateral",
) -> SurgicalOp:
    """Deep plane facelift — sub-SMAS dissection with retaining ligament release.

    Enters the sub-SMAS plane and releases zygomatic, masseteric,
    and/or cervical retaining ligaments for composite flap mobilization.

    Parameters
    ----------
    release_zygomatic : bool
        Release zygomatic retaining ligament.
    release_mandibular : bool
        Release mandibular retaining ligament.
    release_cervical : bool
        Release cervical retaining ligament (for extended deep plane).
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="deep_plane_dissection",
        category=OpCategory.RELEASE,
        procedure=ProcedureType.FACELIFT,
        param_defs={
            "release_zygomatic": OperatorParam(
                "release_zygomatic", ParamType.BOOL, "",
                "Release zygomatic retaining ligament",
                default=True,
            ),
            "release_mandibular": OperatorParam(
                "release_mandibular", ParamType.BOOL, "",
                "Release mandibular retaining ligament",
                default=True,
            ),
            "release_cervical": OperatorParam(
                "release_cervical", ParamType.BOOL, "",
                "Release cervical retaining ligament",
                default=False,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side of face",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "release_zygomatic": release_zygomatic,
            "release_mandibular": release_mandibular,
            "release_cervical": release_cervical,
            "side": side,
        },
        affected_structures=[
            StructureType.SMAS,
            StructureType.FAT_MALAR,
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.MUSCLE_MIMETIC,
            StructureType.PERIOSTEUM,
        ],
        description="Sub-SMAS deep plane dissection with retaining ligament release",
    )
    op.validate()
    return op


def skin_excision(
    excision_pattern: str = "pre_auricular",
    *,
    width_mm: float = 20.0,
    include_post_auricular: bool = True,
    side: str = "bilateral",
) -> SurgicalOp:
    """Skin excision and redraping — remove redundant skin.

    Parameters
    ----------
    excision_pattern : str
        "pre_auricular", "post_auricular", or "short_scar".
    width_mm : float
        Maximum skin excision width in mm.
    include_post_auricular : bool
        Include post-auricular excision.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="skin_excision_facelift",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.FACELIFT,
        param_defs={
            "excision_pattern": OperatorParam(
                "excision_pattern", ParamType.ENUM, "",
                "Incision and excision pattern",
                default="pre_auricular",
                enum_values=("pre_auricular", "post_auricular", "short_scar"),
            ),
            "width_mm": OperatorParam(
                "width_mm", ParamType.FLOAT, "mm",
                "Maximum skin excision width",
                default=20.0, min_value=5.0, max_value=40.0,
            ),
            "include_post_auricular": OperatorParam(
                "include_post_auricular", ParamType.BOOL, "",
                "Include post-auricular limb",
                default=True,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side of face",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "excision_pattern": excision_pattern,
            "width_mm": width_mm,
            "include_post_auricular": include_post_auricular,
            "side": side,
        },
        affected_structures=[
            StructureType.SKIN_ENVELOPE,
            StructureType.SKIN_THICK,
            StructureType.FAT_SUBCUTANEOUS,
        ],
        description="Skin excision and redraping for facial rejuvenation",
    )
    op.validate()
    return op


def fat_repositioning(
    compartment: str = "malar",
    *,
    vector_deg: float = 70.0,
    displacement_mm: float = 8.0,
    side: str = "bilateral",
) -> SurgicalOp:
    """Facial fat compartment repositioning — correct ptotic fat descent.

    Parameters
    ----------
    compartment : str
        "malar", "nasolabial", "buccal", or "jowl".
    vector_deg : float
        Repositioning vector (0=lateral, 90=superior).
    displacement_mm : float
        Target displacement in mm.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="fat_repositioning",
        category=OpCategory.REPOSITIONING,
        procedure=ProcedureType.FACELIFT,
        param_defs={
            "compartment": OperatorParam(
                "compartment", ParamType.ENUM, "",
                "Fat compartment to reposition",
                default="malar",
                enum_values=("malar", "nasolabial", "buccal", "jowl"),
            ),
            "vector_deg": OperatorParam(
                "vector_deg", ParamType.FLOAT, "deg",
                "Repositioning vector angle",
                default=70.0, min_value=30.0, max_value=90.0,
            ),
            "displacement_mm": OperatorParam(
                "displacement_mm", ParamType.FLOAT, "mm",
                "Target displacement",
                default=8.0, min_value=2.0, max_value=20.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side of face",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "compartment": compartment,
            "vector_deg": vector_deg,
            "displacement_mm": displacement_mm,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_MALAR,
            StructureType.FAT_NASOLABIAL,
            StructureType.FAT_BUCCAL,
            StructureType.SMAS,
        ],
        description=f"Reposition {compartment} fat compartment superolaterally",
    )
    op.validate()
    return op


def platysma_plication(
    technique: str = "corset",
    *,
    midline_plication: bool = True,
    band_transection: bool = False,
) -> SurgicalOp:
    """Platysma plication — tighten the platysma muscle layer.

    Parameters
    ----------
    technique : str
        "corset" (midline suturing), "lateral_pull",
        or "full" (midline + lateral).
    midline_plication : bool
        Plicate at midline (for medial bands).
    band_transection : bool
        Transect prominent platysma bands.
    """
    op = SurgicalOp(
        name="platysma_plication",
        category=OpCategory.SUTURE,
        procedure=ProcedureType.NECKLIFT,
        param_defs={
            "technique": OperatorParam(
                "technique", ParamType.ENUM, "",
                "Platysma plication technique",
                default="corset",
                enum_values=("corset", "lateral_pull", "full"),
            ),
            "midline_plication": OperatorParam(
                "midline_plication", ParamType.BOOL, "",
                "Plicate at midline",
                default=True,
            ),
            "band_transection": OperatorParam(
                "band_transection", ParamType.BOOL, "",
                "Transect prominent bands",
                default=False,
            ),
        },
        params={
            "technique": technique,
            "midline_plication": midline_plication,
            "band_transection": band_transection,
        },
        affected_structures=[
            StructureType.MUSCLE_PLATYSMA,
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.SKIN_ENVELOPE,
        ],
        description="Platysma muscle plication for neck contour improvement",
    )
    op.validate()
    return op


def submentoplasty(
    *,
    liposuction: bool = True,
    liposuction_volume_cc: float = 15.0,
    direct_excision: bool = False,
    platysma_work: bool = True,
) -> SurgicalOp:
    """Submentoplasty — submental fat reduction and neck contouring.

    Parameters
    ----------
    liposuction : bool
        Perform submental liposuction.
    liposuction_volume_cc : float
        Target liposuction volume in cc.
    direct_excision : bool
        Direct fat excision (subplatysmal or pre-platysmal).
    platysma_work : bool
        Include platysma plication in the submental approach.
    """
    op = SurgicalOp(
        name="submentoplasty",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.NECKLIFT,
        param_defs={
            "liposuction": OperatorParam(
                "liposuction", ParamType.BOOL, "",
                "Perform submental liposuction",
                default=True,
            ),
            "liposuction_volume_cc": OperatorParam(
                "liposuction_volume_cc", ParamType.FLOAT, "cc",
                "Target liposuction volume",
                default=15.0, min_value=5.0, max_value=50.0,
            ),
            "direct_excision": OperatorParam(
                "direct_excision", ParamType.BOOL, "",
                "Direct subplatysmal fat excision",
                default=False,
            ),
            "platysma_work": OperatorParam(
                "platysma_work", ParamType.BOOL, "",
                "Include platysma plication",
                default=True,
            ),
        },
        params={
            "liposuction": liposuction,
            "liposuction_volume_cc": liposuction_volume_cc,
            "direct_excision": direct_excision,
            "platysma_work": platysma_work,
        },
        affected_structures=[
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.MUSCLE_PLATYSMA,
            StructureType.SKIN_ENVELOPE,
        ],
        description="Submentoplasty with liposuction and platysma work",
    )
    op.validate()
    return op


def malar_fat_suspension(
    *,
    vector_deg: float = 80.0,
    fixation_point: str = "deep_temporal_fascia",
    side: str = "bilateral",
) -> SurgicalOp:
    """Malar fat pad suspension — elevate descended malar fat.

    Parameters
    ----------
    vector_deg : float
        Suspension vector (0=lateral, 90=superior).
    fixation_point : str
        Anchor point for suspension suture.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="malar_fat_suspension",
        category=OpCategory.REPOSITIONING,
        procedure=ProcedureType.FACELIFT,
        param_defs={
            "vector_deg": OperatorParam(
                "vector_deg", ParamType.FLOAT, "deg",
                "Suspension vector angle",
                default=80.0, min_value=45.0, max_value=90.0,
            ),
            "fixation_point": OperatorParam(
                "fixation_point", ParamType.ENUM, "",
                "Suture anchor point",
                default="deep_temporal_fascia",
                enum_values=(
                    "deep_temporal_fascia",
                    "periosteum",
                    "smas",
                ),
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side of face",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "vector_deg": vector_deg,
            "fixation_point": fixation_point,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_MALAR,
            StructureType.SMAS,
            StructureType.PERIOSTEUM,
        ],
        description="Malar fat pad suspension to restore youthful contour",
    )
    op.validate()
    return op


# ── Operator registry ─────────────────────────────────────────────

FACELIFT_OPERATORS = {
    "smas_plication": smas_plication,
    "smas_flap": smas_flap,
    "deep_plane_dissection": deep_plane_dissection,
    "skin_excision_facelift": skin_excision,
    "fat_repositioning": fat_repositioning,
    "platysma_plication": platysma_plication,
    "submentoplasty": submentoplasty,
    "malar_fat_suspension": malar_fat_suspension,
}


# ── Plan builder templates ────────────────────────────────────────

class FaceliftPlanBuilder:
    """Build pre-configured facelift/necklift plans."""

    @staticmethod
    def smas_plication_facelift(
        *,
        plication_vector_deg: float = 60.0,
        plication_width_mm: float = 10.0,
        skin_excision_width_mm: float = 20.0,
        malar_repositioning: bool = True,
    ) -> SurgicalPlan:
        """Build a SMAS plication facelift plan."""
        plan = SurgicalPlan(
            name="smas_plication_facelift",
            procedure=ProcedureType.FACELIFT,
            description="SMAS plication facelift with skin redraping",
        )

        plan.add_step(smas_plication(
            vector_deg=plication_vector_deg,
            plication_width_mm=plication_width_mm,
        ))

        if malar_repositioning:
            plan.add_step(fat_repositioning(compartment="malar"))

        plan.add_step(skin_excision(width_mm=skin_excision_width_mm))

        return plan

    @staticmethod
    def deep_plane_facelift(
        *,
        release_cervical: bool = False,
        skin_excision_width_mm: float = 15.0,
        necklift: bool = True,
    ) -> SurgicalPlan:
        """Build a deep plane facelift plan (Hamra-style)."""
        plan = SurgicalPlan(
            name="deep_plane_facelift",
            procedure=ProcedureType.FACELIFT,
            description="Deep plane facelift with composite flap mobilization",
        )

        plan.add_step(deep_plane_dissection(release_cervical=release_cervical))
        plan.add_step(malar_fat_suspension())
        plan.add_step(fat_repositioning(compartment="nasolabial"))

        if necklift:
            plan.add_step(platysma_plication(technique="full"))
            plan.add_step(submentoplasty(liposuction=True))

        plan.add_step(skin_excision(width_mm=skin_excision_width_mm))

        return plan

    @staticmethod
    def necklift(
        *,
        liposuction_volume_cc: float = 15.0,
        platysma_technique: str = "corset",
        band_transection: bool = False,
    ) -> SurgicalPlan:
        """Build an isolated necklift plan."""
        plan = SurgicalPlan(
            name="necklift",
            procedure=ProcedureType.NECKLIFT,
            description="Necklift with submentoplasty and platysma plication",
        )

        plan.add_step(submentoplasty(
            liposuction_volume_cc=liposuction_volume_cc,
        ))
        plan.add_step(platysma_plication(
            technique=platysma_technique,
            band_transection=band_transection,
        ))

        return plan

    @staticmethod
    def lateral_smasectomy_facelift(
        *,
        resection_width_mm: float = 15.0,
        malar_repositioning: bool = True,
        necklift: bool = False,
    ) -> SurgicalPlan:
        """Build a lateral SMASectomy facelift plan (Baker-style)."""
        plan = SurgicalPlan(
            name="lateral_smasectomy_facelift",
            procedure=ProcedureType.FACELIFT,
            description="Lateral SMASectomy facelift",
        )

        plan.add_step(smas_flap(
            flap_type="lateral_smasectomy",
            resection_width_mm=resection_width_mm,
        ))

        if malar_repositioning:
            plan.add_step(malar_fat_suspension())

        if necklift:
            plan.add_step(platysma_plication(technique="corset"))
            plan.add_step(submentoplasty())

        plan.add_step(skin_excision())

        return plan
