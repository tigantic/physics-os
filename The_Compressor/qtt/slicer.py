"""
QTT Slicer - Universal Random-Access Interface
===============================================

The Slicer is the high-level API that abstracts away whether you're
accessing spatial (physics) or semantic (text) data.

You don't "open" a file. You Slice it.

Usage:
    >>> from qtt import QTTSlicer
    
    # From spatial data
    >>> slicer = QTTSlicer.from_spatial(temperature_field)
    >>> value = slicer.slice((64, 64, 64))  # 10µs
    
    # From text corpus
    >>> slicer = QTTSlicer.from_corpus(sentences)
    >>> results = slicer.slice("quantum physics")  # 6ms
    >>> for match in results:
    ...     print(match.text)
    
    # From file
    >>> slicer = QTTSlicer.open("data.qtt")
    >>> result = slicer.slice(...)  # Auto-detects mode
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union, Any, overload
from dataclasses import dataclass
from pathlib import Path

from qtt.container import QTTContainer, SliceResult, SemanticMatch
from qtt.spatial import SpatialCompressor, tt_svd, tt_reconstruct_element
from qtt.semantic import SemanticIndex


@dataclass
class SpatialSliceResult:
    """Result from spatial slice."""
    coords: Tuple[int, ...]
    value: float
    access_time_us: float
    
    def __repr__(self):
        return f"SpatialSliceResult(coords={self.coords}, value={self.value:.4f}, time={self.access_time_us:.1f}µs)"


class SpatialIndex:
    """
    In-memory spatial index using TT-SVD.
    
    For physics/simulation data where the TT cores ARE the data.
    """
    
    def __init__(self, cores: List[np.ndarray], shape: Tuple[int, ...]):
        self.cores = cores
        self.shape = shape
    
    @classmethod
    def from_data(cls, data: np.ndarray, max_rank: int = 64) -> 'SpatialIndex':
        """Create index from N-dimensional array."""
        cores = tt_svd(data, max_rank)
        return cls(cores, data.shape)
    
    def slice(self, coords: Tuple[int, ...]) -> float:
        """Get value at coordinates via TT contraction."""
        return tt_reconstruct_element(self.cores, coords)
    
    @property
    def compressed_size(self) -> int:
        """Size of TT cores in bytes."""
        return sum(c.nbytes for c in self.cores)


class QTTSlicer:
    """
    Universal Slicer - The file system driver for .qtt format.
    
    Provides a single API for both spatial and semantic data:
    
        slicer.slice(coords=(x,y,z))  # Spatial: TT contraction
        slicer.slice(query="...")      # Semantic: PQ lookup
    
    The underlying math (SVD vs PQ) is abstracted away.
    The user only interacts with the Coordinate System.
    """
    
    def __init__(self):
        self._container: Optional[QTTContainer] = None
        self._spatial_index: Optional[SpatialIndex] = None
        self._semantic_index: Optional[SemanticIndex] = None
        self._mode: Optional[str] = None
        self._texts: Optional[List[str]] = None  # In-memory text for corpus mode
    
    @classmethod
    def from_spatial(cls,
                     data: np.ndarray,
                     max_rank: int = 64) -> 'QTTSlicer':
        """
        Create slicer from spatial data (physics/simulation).
        
        Args:
            data: N-dimensional numpy array
            max_rank: Maximum TT rank
            
        Returns:
            QTTSlicer in spatial mode
        """
        slicer = cls()
        slicer._spatial_index = SpatialIndex.from_data(data, max_rank)
        slicer._mode = 'spatial'
        return slicer
    
    @classmethod
    def from_corpus(cls,
                    texts: List[str],
                    n_subvectors: int = 12,
                    n_centroids: int = 256,
                    show_progress: bool = True) -> 'QTTSlicer':
        """
        Create slicer from text corpus.
        
        Args:
            texts: List of text documents
            n_subvectors: PQ subvectors
            n_centroids: Centroids per subvector
            show_progress: Show progress bars
            
        Returns:
            QTTSlicer in semantic mode
        """
        slicer = cls()
        slicer._semantic_index = SemanticIndex.from_texts(
            texts, n_subvectors, n_centroids,
            show_progress=show_progress
        )
        slicer._texts = texts
        slicer._mode = 'semantic'
        return slicer
    
    @classmethod
    def open(cls, path: Union[str, Path]) -> 'QTTSlicer':
        """
        Open a .qtt file for slicing.
        
        Args:
            path: Path to .qtt file
            
        Returns:
            QTTSlicer ready for queries
        """
        slicer = cls()
        slicer._container = QTTContainer.open(path)
        slicer._mode = 'spatial' if slicer._container.is_spatial else 'semantic'
        return slicer
    
    def slice(self,
              coords: Optional[Tuple[int, ...]] = None,
              query: Optional[str] = None,
              top_k: int = 10) -> Union[float, SpatialSliceResult, List[SemanticMatch]]:
        """
        Universal slice interface.
        
        Args:
            coords: Spatial coordinates (for physics data)
            query: Text query (for semantic data)
            top_k: Number of results for semantic search
            
        Returns:
            For spatial: float value or SpatialSliceResult
            For semantic: List of SemanticMatch with text populated
        """
        if self._mode == 'spatial':
            if coords is None:
                raise ValueError("Spatial mode requires coords")
            return self._slice_spatial(coords)
        elif self._mode == 'semantic':
            if query is None:
                raise ValueError("Semantic mode requires query")
            return self._slice_semantic(query, top_k)
        else:
            raise ValueError("Slicer not initialized")
    
    def _slice_spatial(self, coords: Tuple[int, ...]) -> float:
        """Slice spatial data."""
        if self._container:
            return self._container.slice(coords=coords)
        elif self._spatial_index:
            return self._spatial_index.slice(coords)
        else:
            raise ValueError("No spatial index")
    
    def _slice_semantic(self, query: str, top_k: int) -> List[SemanticMatch]:
        """Slice semantic data and populate text."""
        if self._container:
            result = self._container.slice(query=query, top_k=top_k)
            # Populate text for each match
            for match in result.matches:
                match.text = self._container.read_text(match)
            return result.matches
        elif self._semantic_index:
            result = self._semantic_index.search(query, top_k)
            # Populate text from in-memory list
            matches = []
            for m in result.matches:
                matches.append(SemanticMatch(
                    score=m.score,
                    sentence_id=m.document_id,
                    offset=m.offset,
                    length=m.length,
                    text=self._texts[m.document_id] if self._texts else None
                ))
            return matches
        else:
            raise ValueError("No semantic index")
    
    def save(self, path: Union[str, Path]):
        """
        Save slicer to .qtt file.
        
        Args:
            path: Output path
        """
        if self._container:
            raise ValueError("Cannot save a slicer opened from file")
        
        if self._mode == 'spatial':
            container = QTTContainer()
            container.header.mode = 0x01
            # Would need to pack spatial index into container
            raise NotImplementedError("Use QTTContainer.from_spatial_data() directly")
        elif self._mode == 'semantic':
            if self._texts is None:
                raise ValueError("No texts to save")
            container = QTTContainer.from_text_corpus(self._texts)
            container.save(path)
        else:
            raise ValueError("Slicer not initialized")
    
    def close(self):
        """Release resources."""
        if self._container:
            self._container.close()
            self._container = None
        self._spatial_index = None
        self._semantic_index = None
        self._texts = None
    
    def __enter__(self) -> 'QTTSlicer':
        return self
    
    def __exit__(self, *args):
        self.close()
    
    @property
    def mode(self) -> Optional[str]:
        """Current mode: 'spatial' or 'semantic'."""
        return self._mode
    
    def info(self) -> str:
        """Get info about the slicer."""
        if self._container:
            return self._container.info()
        elif self._spatial_index:
            return f"SpatialIndex: shape={self._spatial_index.shape}, size={self._spatial_index.compressed_size:,} bytes"
        elif self._semantic_index:
            stats = self._semantic_index.get_stats()
            return f"SemanticIndex: {stats['n_documents']:,} docs, {stats['total_size_bytes']:,} bytes"
        else:
            return "Slicer not initialized"
