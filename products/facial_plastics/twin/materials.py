"""Material property assignment for the digital twin.

Maps segmented anatomical structures to constitutive models and
parameters from the tissue property library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..core.config import DEFAULT_TISSUE_LIBRARY
from ..core.types import (
    MaterialModel,
    StructureType,
    TissueProperties,
    VolumeMesh,
)
from .segmentation import LABEL_MAP

logger = logging.getLogger(__name__)


# ── Default structure → material model mapping ────────────────────

STRUCTURE_MODEL_MAP: Dict[StructureType, Tuple[MaterialModel, str]] = {
    # Bone — linear elastic
    StructureType.BONE_NASAL: (MaterialModel.LINEAR_ELASTIC, "bone_nasal"),
    StructureType.BONE_MAXILLA: (MaterialModel.LINEAR_ELASTIC, "bone_maxilla"),
    StructureType.BONE_MANDIBLE: (MaterialModel.LINEAR_ELASTIC, "bone_maxilla"),
    StructureType.BONE_FRONTAL: (MaterialModel.LINEAR_ELASTIC, "bone_maxilla"),
    # Cartilage — Mooney-Rivlin
    StructureType.CARTILAGE_SEPTUM: (MaterialModel.MOONEY_RIVLIN, "cartilage_septal"),
    StructureType.CARTILAGE_UPPER_LATERAL: (MaterialModel.MOONEY_RIVLIN, "cartilage_upper_lateral"),
    StructureType.CARTILAGE_LOWER_LATERAL: (MaterialModel.MOONEY_RIVLIN, "cartilage_lower_lateral"),
    StructureType.CARTILAGE_EAR: (MaterialModel.MOONEY_RIVLIN, "cartilage_ear"),
    # Skin — Neo-Hookean (varies by region)
    StructureType.SKIN_THICK: (MaterialModel.NEO_HOOKEAN, "skin_nasal_tip"),
    StructureType.SKIN_THIN: (MaterialModel.NEO_HOOKEAN, "skin_eyelid"),
    StructureType.SKIN_SEBACEOUS: (MaterialModel.NEO_HOOKEAN, "skin_nasal_dorsum"),
    # Fat — Neo-Hookean
    StructureType.FAT_SUBCUTANEOUS: (MaterialModel.NEO_HOOKEAN, "subcutaneous_fat"),
    StructureType.FAT_DEEP: (MaterialModel.NEO_HOOKEAN, "subcutaneous_fat"),
    StructureType.FAT_BUCCAL: (MaterialModel.NEO_HOOKEAN, "subcutaneous_fat"),
    # Connective tissue
    StructureType.SMAS: (MaterialModel.NEO_HOOKEAN, "smas"),
    StructureType.PERIOSTEUM: (MaterialModel.NEO_HOOKEAN, "periosteum"),
    # Muscle
    StructureType.MUSCLE_MIMETIC: (MaterialModel.NEO_HOOKEAN, "muscle_mimetic"),
    # Mucosa
    StructureType.MUCOSA_NASAL: (MaterialModel.NEO_HOOKEAN, "mucosa_nasal"),
}


@dataclass
class MaterialAssignment:
    """Complete material assignment for one structure."""
    structure: StructureType
    model: MaterialModel
    library_key: str
    properties: TissueProperties
    element_indices: np.ndarray  # indices into volume mesh elements

    def param_dict(self) -> Dict[str, float]:
        """Return material parameters as a flat dict."""
        return dict(self.properties.parameters)


class MaterialAssigner:
    """Assign material properties to a meshed digital twin.

    Uses the structure-to-model mapping and the tissue property
    library.  Supports patient-specific overrides, age/sex
    adjustments, and custom parameter maps.
    """

    def __init__(
        self,
        library: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self._library = library or dict(DEFAULT_TISSUE_LIBRARY)

    def assign(
        self,
        mesh: VolumeMesh,
        element_labels: np.ndarray,
        *,
        overrides: Optional[Dict[StructureType, Dict[str, float]]] = None,
        age_years: Optional[int] = None,
        skin_fitzpatrick: Optional[int] = None,
    ) -> List[MaterialAssignment]:
        """Assign materials to all elements based on segmentation labels.

        Parameters
        ----------
        mesh : VolumeMesh
            FEM mesh.
        element_labels : ndarray (n_elements,) int
            Segmentation label per element.
        overrides : dict, optional
            {StructureType: {param: value}} patient-specific overrides.
        age_years : int, optional
            Patient age for age-dependent property adjustment.
        skin_fitzpatrick : int, optional
            Fitzpatrick skin type (I–VI) for skin property adjustment.

        Returns
        -------
        List of MaterialAssignment, one per structure present.
        """
        overrides = overrides or {}
        inv_label = {v: k for k, v in LABEL_MAP.items()}

        # Group elements by structure
        structure_elements: Dict[StructureType, List[int]] = {}
        for ei, label_val in enumerate(element_labels):
            if label_val in inv_label:
                st = inv_label[label_val]
                if st not in structure_elements:
                    structure_elements[st] = []
                structure_elements[st].append(ei)

        assignments = []
        for st, elem_indices in structure_elements.items():
            if st not in STRUCTURE_MODEL_MAP:
                logger.warning("No material model mapping for %s", st.value)
                continue

            model, lib_key = STRUCTURE_MODEL_MAP[st]

            # Get base parameters from library
            if lib_key not in self._library:
                logger.warning("Missing library entry: %s", lib_key)
                continue

            params = dict(self._library[lib_key])

            # Apply age-dependent adjustments
            if age_years is not None:
                params = self._adjust_for_age(params, model, st, age_years)

            # Apply skin type adjustments
            if skin_fitzpatrick is not None and "skin" in st.value.lower():
                params = self._adjust_for_skin_type(params, skin_fitzpatrick)

            # Apply patient-specific overrides
            if st in overrides:
                params.update(overrides[st])

            # Build TissueProperties
            tissue_props = TissueProperties(
                structure=st,
                material_model=model,
                parameters=params,
            )

            assignments.append(MaterialAssignment(
                structure=st,
                model=model,
                library_key=lib_key,
                properties=tissue_props,
                element_indices=np.array(elem_indices, dtype=np.int32),
            ))

        logger.info("Assigned materials to %d structures (%d total elements)",
                     len(assignments), sum(len(a.element_indices) for a in assignments))
        return assignments

    def export_for_solver(
        self,
        assignments: List[MaterialAssignment],
    ) -> Dict[str, Any]:
        """Export material assignments in solver-ready format.

        Returns a dict suitable for JSON serialization and
        ingestion by the FEM solver.
        """
        materials = {}
        for a in assignments:
            materials[a.structure.value] = {
                "model": a.model.value,
                "library_key": a.library_key,
                "parameters": a.param_dict(),
                "n_elements": len(a.element_indices),
                "element_indices": a.element_indices.tolist(),
            }
        return materials

    # ── Age-dependent adjustments ─────────────────────────────

    @staticmethod
    def _adjust_for_age(
        params: Dict[str, float],
        model: MaterialModel,
        structure: StructureType,
        age: int,
    ) -> Dict[str, float]:
        """Adjust material parameters for patient age.

        Literature-based adjustments:
          - Cartilage stiffens ~2% per decade after 30
          - Skin loses elasticity ~1.5% per decade after 25
          - Bone density decreases ~0.5% per decade after 40
          - Fat compliance changes minimally with age
        """
        params = dict(params)
        ref_age = 35  # reference age for library values

        if "cartilage" in structure.value.lower():
            factor = 1.0 + 0.002 * (age - ref_age)
            factor = max(0.8, min(1.4, factor))
            for key in ("C1", "C2", "kappa"):
                if key in params:
                    params[key] *= factor

        elif "skin" in structure.value.lower():
            factor = 1.0 - 0.0015 * (age - ref_age)
            factor = max(0.6, min(1.2, factor))
            for key in ("mu", "kappa"):
                if key in params:
                    params[key] *= factor

        elif "bone" in structure.value.lower():
            if age > 40:
                factor = 1.0 - 0.0005 * (age - 40)
                factor = max(0.8, min(1.0, factor))
                for key in ("E", "density"):
                    if key in params:
                        params[key] *= factor

        return params

    @staticmethod
    def _adjust_for_skin_type(
        params: Dict[str, float],
        fitzpatrick: int,
    ) -> Dict[str, float]:
        """Adjust skin parameters for Fitzpatrick type.

        Higher Fitzpatrick types tend to have:
          - Thicker dermis
          - Higher collagen density
          - Slightly stiffer mechanical response
        """
        params = dict(params)
        # Types I-II: baseline
        # Types III-IV: ~10% stiffer, ~15% thicker
        # Types V-VI: ~20% stiffer, ~25% thicker
        if fitzpatrick in (3, 4):
            for key in ("mu", "kappa"):
                if key in params:
                    params[key] *= 1.10
            if "thickness" in params:
                params["thickness"] *= 1.15
        elif fitzpatrick in (5, 6):
            for key in ("mu", "kappa"):
                if key in params:
                    params[key] *= 1.20
            if "thickness" in params:
                params["thickness"] *= 1.25

        return params
