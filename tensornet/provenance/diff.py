"""
Field Diff Engine
==================

Compute and analyze differences between field states.

Features:
- Structural diff (shape, dtype changes)
- Value diff (absolute, relative)
- Semantic diff (physical quantities)
- Diff visualization
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum


# =============================================================================
# DIFF TYPES
# =============================================================================

class DiffType(Enum):
    """Type of difference detected."""
    IDENTICAL = "identical"           # No difference
    VALUE_CHANGE = "value_change"     # Values changed
    SHAPE_CHANGE = "shape_change"     # Shape changed
    DTYPE_CHANGE = "dtype_change"     # Data type changed
    INCOMPATIBLE = "incompatible"     # Cannot compare


# =============================================================================
# DIFF SUMMARY
# =============================================================================

@dataclass
class DiffSummary:
    """
    Summary statistics of field differences.
    """
    diff_type: DiffType = DiffType.IDENTICAL
    
    # Shape info
    shape_before: Tuple[int, ...] = ()
    shape_after: Tuple[int, ...] = ()
    
    # Value statistics
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    l2_diff: float = 0.0
    
    # Relative statistics
    max_rel_diff: float = 0.0
    mean_rel_diff: float = 0.0
    
    # Change extent
    changed_elements: int = 0
    total_elements: int = 0
    
    @property
    def changed_fraction(self) -> float:
        """Fraction of elements that changed."""
        if self.total_elements == 0:
            return 0.0
        return self.changed_elements / self.total_elements
    
    @property
    def is_identical(self) -> bool:
        return self.diff_type == DiffType.IDENTICAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "diff_type": self.diff_type.value,
            "shape_before": list(self.shape_before),
            "shape_after": list(self.shape_after),
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "l2_diff": self.l2_diff,
            "max_rel_diff": self.max_rel_diff,
            "mean_rel_diff": self.mean_rel_diff,
            "changed_elements": self.changed_elements,
            "total_elements": self.total_elements,
            "changed_fraction": self.changed_fraction,
        }


# =============================================================================
# FIELD DIFF
# =============================================================================

@dataclass
class FieldDiff:
    """
    Detailed difference between two field states.
    
    Contains:
    - Summary statistics
    - Difference array (if compatible)
    - Change mask
    - Hotspot locations
    """
    summary: DiffSummary
    
    # Difference data
    diff_data: Optional[np.ndarray] = None
    change_mask: Optional[np.ndarray] = None  # Boolean mask of changed elements
    
    # Hotspots (regions of largest change)
    hotspots: List[Tuple[Tuple[int, ...], float]] = field(default_factory=list)
    
    # Metadata
    commit1_hash: Optional[str] = None
    commit2_hash: Optional[str] = None
    
    def get_hotspot_regions(
        self,
        n: int = 5,
        radius: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get regions around hotspots.
        
        Args:
            n: Number of hotspots to return
            radius: Radius around hotspot to include
            
        Returns:
            List of hotspot info dictionaries
        """
        if self.diff_data is None:
            return []
        
        regions = []
        for idx, value in self.hotspots[:n]:
            # Build slice for region
            slices = []
            for i, dim_size in enumerate(self.diff_data.shape):
                start = max(0, idx[i] - radius)
                end = min(dim_size, idx[i] + radius + 1)
                slices.append(slice(start, end))
            
            region_data = self.diff_data[tuple(slices)]
            
            regions.append({
                "center": idx,
                "value": value,
                "region_mean": float(np.mean(np.abs(region_data))),
                "region_max": float(np.max(np.abs(region_data))),
            })
        
        return regions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "hotspots": [
                {"index": list(idx), "value": val}
                for idx, val in self.hotspots
            ],
            "commit1_hash": self.commit1_hash,
            "commit2_hash": self.commit2_hash,
        }


# =============================================================================
# DIFF ENGINE
# =============================================================================

class DiffEngine:
    """
    Engine for computing field differences.
    
    Example:
        engine = DiffEngine(tolerance=1e-6)
        
        diff = engine.compute(field_before, field_after)
        
        print(f"Max change: {diff.summary.max_abs_diff}")
        print(f"Changed: {diff.summary.changed_fraction:.1%}")
        
        for hotspot in diff.get_hotspot_regions():
            print(f"Hotspot at {hotspot['center']}: {hotspot['value']:.2e}")
    """
    
    def __init__(
        self,
        tolerance: float = 1e-10,
        relative_tolerance: float = 1e-6,
        n_hotspots: int = 10,
    ):
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.n_hotspots = n_hotspots
    
    def compute(
        self,
        field1: Union[np.ndarray, torch.Tensor],
        field2: Union[np.ndarray, torch.Tensor],
        commit1_hash: Optional[str] = None,
        commit2_hash: Optional[str] = None,
    ) -> FieldDiff:
        """
        Compute difference between two fields.
        
        Args:
            field1: First field (before)
            field2: Second field (after)
            commit1_hash: Optional commit hash for field1
            commit2_hash: Optional commit hash for field2
            
        Returns:
            FieldDiff with detailed difference info
        """
        # Convert to numpy
        if isinstance(field1, torch.Tensor):
            arr1 = field1.detach().cpu().numpy()
        else:
            arr1 = np.asarray(field1)
        
        if isinstance(field2, torch.Tensor):
            arr2 = field2.detach().cpu().numpy()
        else:
            arr2 = np.asarray(field2)
        
        # Check compatibility
        if arr1.shape != arr2.shape:
            return FieldDiff(
                summary=DiffSummary(
                    diff_type=DiffType.SHAPE_CHANGE,
                    shape_before=arr1.shape,
                    shape_after=arr2.shape,
                ),
                commit1_hash=commit1_hash,
                commit2_hash=commit2_hash,
            )
        
        if arr1.dtype != arr2.dtype:
            # Convert to common dtype
            common_dtype = np.promote_types(arr1.dtype, arr2.dtype)
            arr1 = arr1.astype(common_dtype)
            arr2 = arr2.astype(common_dtype)
        
        # Compute difference
        diff = arr2 - arr1
        abs_diff = np.abs(diff)
        
        # Statistics
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))
        l2_diff = float(np.linalg.norm(diff))
        
        # Relative difference (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.abs(diff) / (np.abs(arr1) + self.tolerance)
            rel_diff = np.nan_to_num(rel_diff, nan=0.0, posinf=0.0, neginf=0.0)
        
        max_rel_diff = float(np.max(rel_diff))
        mean_rel_diff = float(np.mean(rel_diff))
        
        # Change mask
        change_mask = abs_diff > self.tolerance
        changed_elements = int(np.sum(change_mask))
        total_elements = arr1.size
        
        # Determine diff type
        if changed_elements == 0:
            diff_type = DiffType.IDENTICAL
        else:
            diff_type = DiffType.VALUE_CHANGE
        
        # Find hotspots
        hotspots = self._find_hotspots(abs_diff)
        
        # Create summary
        summary = DiffSummary(
            diff_type=diff_type,
            shape_before=arr1.shape,
            shape_after=arr2.shape,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            l2_diff=l2_diff,
            max_rel_diff=max_rel_diff,
            mean_rel_diff=mean_rel_diff,
            changed_elements=changed_elements,
            total_elements=total_elements,
        )
        
        return FieldDiff(
            summary=summary,
            diff_data=diff,
            change_mask=change_mask,
            hotspots=hotspots,
            commit1_hash=commit1_hash,
            commit2_hash=commit2_hash,
        )
    
    def _find_hotspots(
        self,
        abs_diff: np.ndarray,
    ) -> List[Tuple[Tuple[int, ...], float]]:
        """Find locations of largest differences."""
        # Flatten to find top indices
        flat = abs_diff.flatten()
        top_indices = np.argsort(flat)[-self.n_hotspots:][::-1]
        
        hotspots = []
        for flat_idx in top_indices:
            idx = np.unravel_index(flat_idx, abs_diff.shape)
            value = float(abs_diff[idx])
            if value > self.tolerance:
                hotspots.append((tuple(idx), value))
        
        return hotspots
    
    def compute_patch(
        self,
        field1: Union[np.ndarray, torch.Tensor],
        field2: Union[np.ndarray, torch.Tensor],
        sparse_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Compute a patch that can transform field1 to field2.
        
        For sparse changes, stores only changed elements.
        
        Args:
            field1: Original field
            field2: Target field
            sparse_threshold: Fraction below which to use sparse format
            
        Returns:
            Patch dictionary
        """
        diff = self.compute(field1, field2)
        
        if diff.summary.is_identical:
            return {"type": "identity"}
        
        if diff.summary.diff_type == DiffType.SHAPE_CHANGE:
            return {
                "type": "reshape",
                "new_shape": diff.summary.shape_after,
            }
        
        # Check if sparse is worthwhile
        if diff.summary.changed_fraction < sparse_threshold:
            # Sparse format
            indices = np.argwhere(diff.change_mask)
            values = diff.diff_data[diff.change_mask]
            
            return {
                "type": "sparse",
                "indices": indices.tolist(),
                "values": values.tolist(),
                "shape": diff.diff_data.shape,
            }
        else:
            # Dense format
            return {
                "type": "dense",
                "diff": diff.diff_data.tolist(),
            }
    
    def apply_patch(
        self,
        field: Union[np.ndarray, torch.Tensor],
        patch: Dict[str, Any],
    ) -> np.ndarray:
        """
        Apply a patch to a field.
        
        Args:
            field: Original field
            patch: Patch from compute_patch
            
        Returns:
            Patched field
        """
        if isinstance(field, torch.Tensor):
            arr = field.detach().cpu().numpy()
        else:
            arr = np.asarray(field).copy()
        
        patch_type = patch.get("type", "identity")
        
        if patch_type == "identity":
            return arr
        
        if patch_type == "reshape":
            # Can't really apply this without the actual data
            return arr
        
        if patch_type == "sparse":
            indices = patch["indices"]
            values = patch["values"]
            for idx, val in zip(indices, values):
                arr[tuple(idx)] += val
            return arr
        
        if patch_type == "dense":
            diff = np.array(patch["diff"])
            return arr + diff
        
        return arr


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_diff(
    field1: Union[np.ndarray, torch.Tensor],
    field2: Union[np.ndarray, torch.Tensor],
    **kwargs,
) -> FieldDiff:
    """
    Convenience function to compute field difference.
    
    Args:
        field1: First field
        field2: Second field
        **kwargs: DiffEngine parameters
        
    Returns:
        FieldDiff
    """
    engine = DiffEngine(**kwargs)
    return engine.compute(field1, field2)


def fields_equal(
    field1: Union[np.ndarray, torch.Tensor],
    field2: Union[np.ndarray, torch.Tensor],
    tolerance: float = 1e-10,
) -> bool:
    """
    Check if two fields are equal within tolerance.
    """
    diff = compute_diff(field1, field2, tolerance=tolerance)
    return diff.summary.is_identical
