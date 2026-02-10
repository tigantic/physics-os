"""Filler injection and fat grafting operator library.

Operators model real injectable/augmentation procedures:
  - HA (hyaluronic acid) filler injection
  - Fat harvest (lipoaspiration from donor site)
  - Fat graft injection (Coleman technique)
  - Bio-stimulatory filler (CaHA, PLLA)
  - Thread lift (barbed suture suspension)
  - Alloplastic implant placement (chin, malar)

References:
  - Coleman 2006 (structural fat grafting);
  - Rohrich et al. 2012 (facial volumization);
  - de Maio 2018 (MD Codes, anatomical injection mapping);
  - Sundaram & Fagien 2015 (filler rheology and tissue interactions).
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


def ha_filler_injection(
    *,
    zone: str = "nasolabial_fold",
    volume_cc: float = 0.5,
    depth: str = "deep_dermal",
    product_viscosity: str = "medium",
    technique: str = "linear_threading",
    side: str = "bilateral",
) -> SurgicalOp:
    """Hyaluronic acid filler injection.

    Parameters
    ----------
    zone : str
        Anatomical injection zone.
    volume_cc : float
        Volume per side in cc.
    depth : str
        Injection depth plane.
    product_viscosity : str
        Filler rheology class.
    technique : str
        Injection technique.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="ha_filler_injection",
        category=OpCategory.AUGMENTATION,
        procedure=ProcedureType.FILLER_INJECTION,
        param_defs={
            "zone": OperatorParam(
                "zone", ParamType.ENUM, "",
                "Anatomical injection zone",
                default="nasolabial_fold",
                enum_values=(
                    "nasolabial_fold", "cheek", "temple", "jawline",
                    "chin", "lip_body", "lip_border", "marionette",
                    "tear_trough", "nose", "perioral",
                ),
            ),
            "volume_cc": OperatorParam(
                "volume_cc", ParamType.FLOAT, "cc",
                "Volume per side",
                default=0.5, min_value=0.05, max_value=3.0,
            ),
            "depth": OperatorParam(
                "depth", ParamType.ENUM, "",
                "Injection depth plane",
                default="deep_dermal",
                enum_values=(
                    "intradermal", "deep_dermal", "subcutaneous",
                    "supraperiosteal", "intramuscular",
                ),
            ),
            "product_viscosity": OperatorParam(
                "product_viscosity", ParamType.ENUM, "",
                "Filler viscosity class",
                default="medium",
                enum_values=("thin", "medium", "thick", "very_thick"),
            ),
            "technique": OperatorParam(
                "technique", ParamType.ENUM, "",
                "Injection technique",
                default="linear_threading",
                enum_values=(
                    "serial_puncture", "linear_threading", "fanning",
                    "cross_hatching", "bolus", "microdroplet",
                ),
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral", "midline"),
            ),
        },
        params={
            "zone": zone,
            "volume_cc": volume_cc,
            "depth": depth,
            "product_viscosity": product_viscosity,
            "technique": technique,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.SKIN_ENVELOPE,
        ],
        description=f"HA filler injection — {zone} ({volume_cc} cc/side)",
    )
    op.validate()
    return op


def fat_harvest(
    *,
    donor_site: str = "abdomen",
    volume_cc: float = 30.0,
    technique: str = "coleman",
    cannula_diameter_mm: float = 3.0,
) -> SurgicalOp:
    """Fat harvest — lipoaspiration for autologous fat grafting.

    Parameters
    ----------
    donor_site : str
        Donor site for fat aspiration.
    volume_cc : float
        Total aspirated volume in cc (before processing).
    technique : str
        Harvest technique.
    cannula_diameter_mm : float
        Cannula diameter.
    """
    op = SurgicalOp(
        name="fat_harvest",
        category=OpCategory.RESECTION,
        procedure=ProcedureType.FAT_GRAFTING,
        param_defs={
            "donor_site": OperatorParam(
                "donor_site", ParamType.ENUM, "",
                "Donor site for fat aspiration",
                default="abdomen",
                enum_values=(
                    "abdomen", "inner_thigh", "outer_thigh", "flank",
                    "knee", "trochanteric",
                ),
            ),
            "volume_cc": OperatorParam(
                "volume_cc", ParamType.FLOAT, "cc",
                "Total aspirated volume (pre-processing)",
                default=30.0, min_value=10.0, max_value=200.0,
            ),
            "technique": OperatorParam(
                "technique", ParamType.ENUM, "",
                "Fat harvest technique",
                default="coleman",
                enum_values=("coleman", "lipokit", "telfa_decant", "nanofat"),
            ),
            "cannula_diameter_mm": OperatorParam(
                "cannula_diameter_mm", ParamType.FLOAT, "mm",
                "Aspiration cannula diameter",
                default=3.0, min_value=1.0, max_value=6.0,
            ),
        },
        params={
            "donor_site": donor_site,
            "volume_cc": volume_cc,
            "technique": technique,
            "cannula_diameter_mm": cannula_diameter_mm,
        },
        affected_structures=[
            StructureType.FAT_SUBCUTANEOUS,
        ],
        description=f"Fat harvest from {donor_site} — {volume_cc} cc",
    )
    op.validate()
    return op


def fat_graft_injection(
    *,
    zone: str = "cheek",
    volume_cc: float = 5.0,
    depth: str = "multi_plane",
    cannula_diameter_mm: float = 1.2,
    side: str = "bilateral",
) -> SurgicalOp:
    """Fat graft injection — structural fat grafting (Coleman technique).

    Parameters
    ----------
    zone : str
        Target facial zone.
    volume_cc : float
        Volume per side in cc.
    depth : str
        Injection depth strategy.
    cannula_diameter_mm : float
        Injection cannula diameter.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="fat_graft_injection",
        category=OpCategory.AUGMENTATION,
        procedure=ProcedureType.FAT_GRAFTING,
        param_defs={
            "zone": OperatorParam(
                "zone", ParamType.ENUM, "",
                "Target facial zone for fat injection",
                default="cheek",
                enum_values=(
                    "cheek", "temple", "nasolabial_fold", "jawline",
                    "chin", "lip", "periorbital", "forehead",
                    "tear_trough", "buccal_hollow",
                ),
            ),
            "volume_cc": OperatorParam(
                "volume_cc", ParamType.FLOAT, "cc",
                "Volume per side (processed fat)",
                default=5.0, min_value=0.5, max_value=20.0,
            ),
            "depth": OperatorParam(
                "depth", ParamType.ENUM, "",
                "Injection depth strategy",
                default="multi_plane",
                enum_values=(
                    "subcutaneous", "deep_fat", "supraperiosteal",
                    "multi_plane",
                ),
            ),
            "cannula_diameter_mm": OperatorParam(
                "cannula_diameter_mm", ParamType.FLOAT, "mm",
                "Injection cannula diameter",
                default=1.2, min_value=0.7, max_value=3.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral", "midline"),
            ),
        },
        params={
            "zone": zone,
            "volume_cc": volume_cc,
            "depth": depth,
            "cannula_diameter_mm": cannula_diameter_mm,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.FAT_DEEP,
            StructureType.SKIN_ENVELOPE,
        ],
        description=f"Fat graft injection — {zone} ({volume_cc} cc/side)",
    )
    op.validate()
    return op


def biostimulatory_filler(
    *,
    agent: str = "calcium_hydroxylapatite",
    zone: str = "cheek",
    volume_cc: float = 1.5,
    dilution_ratio: float = 1.0,
    side: str = "bilateral",
) -> SurgicalOp:
    """Bio-stimulatory filler injection (CaHA, PLLA, PCL).

    Parameters
    ----------
    agent : str
        Bio-stimulatory agent type.
    zone : str
        Target facial zone.
    volume_cc : float
        Volume per side in cc (after dilution).
    dilution_ratio : float
        Product:diluent ratio (1.0 = pure, 0.5 = 1:1 dilution, etc.).
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="biostimulatory_filler",
        category=OpCategory.AUGMENTATION,
        procedure=ProcedureType.FILLER_INJECTION,
        param_defs={
            "agent": OperatorParam(
                "agent", ParamType.ENUM, "",
                "Bio-stimulatory agent",
                default="calcium_hydroxylapatite",
                enum_values=(
                    "calcium_hydroxylapatite",  # Radiesse
                    "poly_l_lactic_acid",        # Sculptra
                    "polycaprolactone",           # Ellansé
                ),
            ),
            "zone": OperatorParam(
                "zone", ParamType.ENUM, "",
                "Target injection zone",
                default="cheek",
                enum_values=(
                    "cheek", "temple", "jawline", "chin",
                    "hands", "perioral",
                ),
            ),
            "volume_cc": OperatorParam(
                "volume_cc", ParamType.FLOAT, "cc",
                "Volume per side (post-dilution)",
                default=1.5, min_value=0.3, max_value=5.0,
            ),
            "dilution_ratio": OperatorParam(
                "dilution_ratio", ParamType.FLOAT, "",
                "Product fraction (1.0 = pure, 0.5 = 1:1 diluted)",
                default=1.0, min_value=0.1, max_value=1.0,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral", "midline"),
            ),
        },
        params={
            "agent": agent,
            "zone": zone,
            "volume_cc": volume_cc,
            "dilution_ratio": dilution_ratio,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.SKIN_ENVELOPE,
            StructureType.FAT_DEEP,
        ],
        description=f"Bio-stimulatory filler — {agent} to {zone}",
    )
    op.validate()
    return op


def thread_lift(
    *,
    thread_type: str = "pdo_barbed",
    zone: str = "midface",
    thread_count: int = 4,
    side: str = "bilateral",
) -> SurgicalOp:
    """Thread lift — barbed suture tissue suspension.

    Parameters
    ----------
    thread_type : str
        Thread material and barb configuration.
    zone : str
        Target zone for suspension.
    thread_count : int
        Number of threads per side.
    side : str
        "left", "right", or "bilateral".
    """
    op = SurgicalOp(
        name="thread_lift",
        category=OpCategory.REPOSITIONING,
        procedure=ProcedureType.FILLER_INJECTION,
        param_defs={
            "thread_type": OperatorParam(
                "thread_type", ParamType.ENUM, "",
                "Thread material and configuration",
                default="pdo_barbed",
                enum_values=(
                    "pdo_barbed", "pdo_smooth", "plla_barbed",
                    "pca_barbed",
                ),
            ),
            "zone": OperatorParam(
                "zone", ParamType.ENUM, "",
                "Target zone for suspension",
                default="midface",
                enum_values=(
                    "midface", "jawline", "brow", "neck", "nasolabial",
                ),
            ),
            "thread_count": OperatorParam(
                "thread_count", ParamType.INT, "",
                "Number of threads per side",
                default=4, min_value=1, max_value=12,
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="bilateral",
                enum_values=("left", "right", "bilateral"),
            ),
        },
        params={
            "thread_type": thread_type,
            "zone": zone,
            "thread_count": thread_count,
            "side": side,
        },
        affected_structures=[
            StructureType.FAT_SUBCUTANEOUS,
            StructureType.SMAS,
            StructureType.SKIN_ENVELOPE,
        ],
        description=f"Thread lift — {thread_count}x {thread_type} to {zone}",
    )
    op.validate()
    return op


def implant_placement(
    *,
    implant_type: str = "silicone",
    zone: str = "chin",
    size: str = "medium",
    approach: str = "intraoral",
    side: str = "midline",
) -> SurgicalOp:
    """Alloplastic facial implant placement.

    Parameters
    ----------
    implant_type : str
        Implant material.
    zone : str
        Anatomical zone for implant.
    size : str
        Implant size (small/medium/large/custom).
    approach : str
        Surgical approach.
    side : str
        "midline" (chin), "left", "right", or "bilateral" (malar).
    """
    op = SurgicalOp(
        name="implant_placement",
        category=OpCategory.AUGMENTATION,
        procedure=ProcedureType.CHIN_AUGMENTATION,
        param_defs={
            "implant_type": OperatorParam(
                "implant_type", ParamType.ENUM, "",
                "Implant material",
                default="silicone",
                enum_values=(
                    "silicone", "medpor", "gore_tex", "peek",
                ),
            ),
            "zone": OperatorParam(
                "zone", ParamType.ENUM, "",
                "Anatomical zone for implant placement",
                default="chin",
                enum_values=("chin", "malar", "submalar", "mandible_angle", "paranasal"),
            ),
            "size": OperatorParam(
                "size", ParamType.ENUM, "",
                "Implant size",
                default="medium",
                enum_values=("small", "medium", "large", "custom"),
            ),
            "approach": OperatorParam(
                "approach", ParamType.ENUM, "",
                "Surgical approach",
                default="intraoral",
                enum_values=("intraoral", "submental", "subciliary"),
            ),
            "side": OperatorParam(
                "side", ParamType.ENUM, "",
                "Side",
                default="midline",
                enum_values=("left", "right", "bilateral", "midline"),
            ),
        },
        params={
            "implant_type": implant_type,
            "zone": zone,
            "size": size,
            "approach": approach,
            "side": side,
        },
        affected_structures=[
            StructureType.BONE_MANDIBLE,
            StructureType.BONE_ZYGOMATIC,
            StructureType.PERIOSTEUM,
            StructureType.FAT_SUBCUTANEOUS,
        ],
        description=f"{implant_type} implant — {zone} ({size})",
    )
    op.validate()
    return op


# ── Operator registry ─────────────────────────────────────────────

FILLER_OPERATORS = {
    "ha_filler_injection": ha_filler_injection,
    "fat_harvest": fat_harvest,
    "fat_graft_injection": fat_graft_injection,
    "biostimulatory_filler": biostimulatory_filler,
    "thread_lift": thread_lift,
    "implant_placement": implant_placement,
}


# ── Plan builder templates ────────────────────────────────────────

class FillerPlanBuilder:
    """Build pre-configured filler / fat grafting plans."""

    @staticmethod
    def liquid_facelift(
        *,
        cheek_volume_cc: float = 1.0,
        nasolabial_volume_cc: float = 0.5,
        marionette_volume_cc: float = 0.3,
        jawline_volume_cc: float = 0.8,
        temple_volume_cc: float = 0.5,
    ) -> SurgicalPlan:
        """Build a non-surgical liquid facelift plan (multi-zone HA)."""
        plan = SurgicalPlan(
            name="liquid_facelift",
            procedure=ProcedureType.FILLER_INJECTION,
            description="Multi-zone HA filler liquid facelift",
        )

        plan.add_step(ha_filler_injection(
            zone="cheek", volume_cc=cheek_volume_cc,
            depth="supraperiosteal", product_viscosity="very_thick",
            technique="bolus",
        ))
        plan.add_step(ha_filler_injection(
            zone="temple", volume_cc=temple_volume_cc,
            depth="supraperiosteal", product_viscosity="thick",
            technique="linear_threading",
        ))
        plan.add_step(ha_filler_injection(
            zone="nasolabial_fold", volume_cc=nasolabial_volume_cc,
            depth="deep_dermal", product_viscosity="medium",
            technique="linear_threading",
        ))
        plan.add_step(ha_filler_injection(
            zone="marionette", volume_cc=marionette_volume_cc,
            depth="deep_dermal", product_viscosity="medium",
            technique="linear_threading",
        ))
        plan.add_step(ha_filler_injection(
            zone="jawline", volume_cc=jawline_volume_cc,
            depth="supraperiosteal", product_viscosity="very_thick",
            technique="linear_threading",
        ))

        return plan

    @staticmethod
    def structural_fat_grafting(
        *,
        donor_site: str = "abdomen",
        harvest_volume_cc: float = 60.0,
        cheek_cc: float = 8.0,
        temple_cc: float = 4.0,
        nasolabial_cc: float = 3.0,
        jawline_cc: float = 5.0,
    ) -> SurgicalPlan:
        """Build a structural fat grafting plan (Coleman technique)."""
        plan = SurgicalPlan(
            name="structural_fat_grafting",
            procedure=ProcedureType.FAT_GRAFTING,
            description="Multi-zone structural fat grafting (Coleman technique)",
        )

        plan.add_step(fat_harvest(
            donor_site=donor_site,
            volume_cc=harvest_volume_cc,
            technique="coleman",
        ))

        for zone, volume in [
            ("cheek", cheek_cc),
            ("temple", temple_cc),
            ("nasolabial_fold", nasolabial_cc),
            ("jawline", jawline_cc),
        ]:
            if volume > 0:
                plan.add_step(fat_graft_injection(
                    zone=zone, volume_cc=volume,
                    depth="multi_plane",
                ))

        return plan

    @staticmethod
    def chin_augmentation(
        *,
        implant_type: str = "silicone",
        size: str = "medium",
        approach: str = "intraoral",
    ) -> SurgicalPlan:
        """Build a chin augmentation plan with alloplastic implant."""
        plan = SurgicalPlan(
            name="chin_augmentation",
            procedure=ProcedureType.CHIN_AUGMENTATION,
            description=f"Chin augmentation with {implant_type} implant ({size})",
        )

        plan.add_step(implant_placement(
            implant_type=implant_type,
            zone="chin",
            size=size,
            approach=approach,
            side="midline",
        ))

        return plan

    @staticmethod
    def thread_lift_midface(
        *,
        thread_type: str = "pdo_barbed",
        thread_count: int = 4,
    ) -> SurgicalPlan:
        """Build a midface thread lift plan."""
        plan = SurgicalPlan(
            name="thread_lift_midface",
            procedure=ProcedureType.FILLER_INJECTION,
            description=f"Midface thread lift ({thread_count}x {thread_type} per side)",
        )

        plan.add_step(thread_lift(
            thread_type=thread_type,
            zone="midface",
            thread_count=thread_count,
        ))

        return plan
