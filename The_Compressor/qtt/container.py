"""
QTT Container Format
====================

The Universal File Format for Random-Access Data.

A single .qtt file contains:
- HEADER: Magic bytes, version, mode (SPATIAL|SEMANTIC)
- METADATA: JSON with shape, dtype, configuration
- INDEX: The "Brain" (TT cores or PQ codebooks)
- PAYLOAD: The "Body" (None for spatial, GZIP stream for semantic)
- FOOTER: Checksums and segment offsets

File Layout:
    ┌─────────────────────────────────────────────────────────┐
    │ HEADER (64 bytes)                                       │
    │   Magic: QTT\x01, Version: 1, Mode: SPATIAL|SEMANTIC   │
    ├─────────────────────────────────────────────────────────┤
    │ METADATA (variable, JSON)                               │
    │   n_documents, embedding_dim, compression settings      │
    ├─────────────────────────────────────────────────────────┤
    │ INDEX - The "Brain"                                     │
    │   • SPATIAL: TT-SVD cores (pickle)                     │
    │   • SEMANTIC: PQ centroids + codes + offsets (pickle)  │
    ├─────────────────────────────────────────────────────────┤
    │ PAYLOAD - The "Body"                                    │
    │   • SPATIAL: None (cores ARE the data)                 │
    │   • SEMANTIC: GZIP-compressed raw text                 │
    ├─────────────────────────────────────────────────────────┤
    │ FOOTER (96 bytes)                                       │
    │   SHA-256 checksums for integrity                       │
    └─────────────────────────────────────────────────────────┘

Usage:
    # Create from text corpus
    >>> container = QTTContainer.from_text_corpus(sentences)
    >>> container.save("library.qtt")
    
    # Open and slice
    >>> with QTTContainer.open("library.qtt") as f:
    ...     results = f.slice(query="quantum mechanics")
    ...     for match in results.matches:
    ...         text = f.read_text(match)
    ...         print(text)

    # Create from spatial data
    >>> container = QTTContainer.from_spatial_data(temperature_field)
    >>> container.save("physics.qtt")
    
    >>> with QTTContainer.open("physics.qtt") as f:
    ...     value = f.slice(coords=(64, 64, 64))
"""

from __future__ import annotations

import struct
import json
import gzip
import hashlib
import mmap
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple, Any, BinaryIO, Dict
import time


# =============================================================================
# Constants
# =============================================================================

QTT_MAGIC = b'QTT\x01'  # Magic bytes identifying .qtt files
QTT_VERSION = 1

MODE_SPATIAL = 0x01
MODE_SEMANTIC = 0x02

# Segment type markers (for future extensibility)
SEGMENT_HEADER = 0x10
SEGMENT_METADATA = 0x20
SEGMENT_INDEX = 0x30
SEGMENT_PAYLOAD = 0x40
SEGMENT_FOOTER = 0xFF


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QTTHeader:
    """
    Fixed-size header at start of .qtt file.
    
    Layout (64 bytes):
        - magic: 4 bytes (QTT\x01)
        - version: 2 bytes (uint16)
        - mode: 1 byte (SPATIAL=0x01, SEMANTIC=0x02)
        - padding: 1 byte
        - metadata_offset: 8 bytes (uint64)
        - metadata_size: 8 bytes (uint64)
        - index_offset: 8 bytes (uint64)
        - index_size: 8 bytes (uint64)
        - payload_offset: 8 bytes (uint64)
        - payload_size: 8 bytes (uint64)
        - footer_offset: 8 bytes (uint64)
    """
    magic: bytes = QTT_MAGIC
    version: int = QTT_VERSION
    mode: int = MODE_SPATIAL
    metadata_offset: int = 0
    metadata_size: int = 0
    index_offset: int = 0
    index_size: int = 0
    payload_offset: int = 0
    payload_size: int = 0
    footer_offset: int = 0
    
    # Struct format: little-endian, 4-byte magic, uint16 version, uint8 mode, 
    # 1-byte pad, then 7 uint64 offsets/sizes
    STRUCT_FORMAT = '<4sHBxQQQQQQQ'
    SIZE = struct.calcsize(STRUCT_FORMAT)  # 64 bytes
    
    def pack(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            self.STRUCT_FORMAT,
            self.magic,
            self.version,
            self.mode,
            self.metadata_offset,
            self.metadata_size,
            self.index_offset,
            self.index_size,
            self.payload_offset,
            self.payload_size,
            self.footer_offset
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'QTTHeader':
        """Deserialize header from bytes."""
        values = struct.unpack(cls.STRUCT_FORMAT, data[:cls.SIZE])
        return cls(
            magic=values[0],
            version=values[1],
            mode=values[2],
            metadata_offset=values[3],
            metadata_size=values[4],
            index_offset=values[5],
            index_size=values[6],
            payload_offset=values[7],
            payload_size=values[8],
            footer_offset=values[9]
        )
    
    def validate(self):
        """Validate header fields."""
        if self.magic != QTT_MAGIC:
            raise ValueError(f"Invalid magic bytes: {self.magic}")
        if self.version > QTT_VERSION:
            raise ValueError(f"Unsupported version: {self.version}")
        if self.mode not in (MODE_SPATIAL, MODE_SEMANTIC):
            raise ValueError(f"Unknown mode: {self.mode}")


@dataclass
class QTTFooter:
    """
    Footer with checksums for integrity verification.
    
    Layout (96 bytes):
        - index_checksum: 32 bytes (SHA-256)
        - payload_checksum: 32 bytes (SHA-256)
        - total_checksum: 32 bytes (SHA-256 of entire file)
    """
    index_checksum: bytes = b'\x00' * 32
    payload_checksum: bytes = b'\x00' * 32
    total_checksum: bytes = b'\x00' * 32
    
    STRUCT_FORMAT = '<32s32s32s'
    SIZE = struct.calcsize(STRUCT_FORMAT)  # 96 bytes
    
    def pack(self) -> bytes:
        """Serialize footer to bytes."""
        return struct.pack(
            self.STRUCT_FORMAT,
            self.index_checksum,
            self.payload_checksum,
            self.total_checksum
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> 'QTTFooter':
        """Deserialize footer from bytes."""
        values = struct.unpack(cls.STRUCT_FORMAT, data[:cls.SIZE])
        return cls(
            index_checksum=values[0],
            payload_checksum=values[1],
            total_checksum=values[2]
        )


@dataclass
class SemanticMatch:
    """Result from semantic slice operation."""
    score: float          # Similarity score (0-1)
    sentence_id: int      # Document index
    offset: int           # Byte offset in payload
    length: int           # Byte length in payload
    text: Optional[str] = None  # Optional: retrieved text
    
    def __repr__(self):
        return f"SemanticMatch(score={self.score:.3f}, id={self.sentence_id}, offset={self.offset})"


@dataclass
class SliceResult:
    """Universal result from slice operation."""
    query: str
    matches: List[SemanticMatch]
    access_time_ms: float
    
    def __repr__(self):
        return f"SliceResult(query='{self.query[:30]}', matches={len(self.matches)}, time={self.access_time_ms:.1f}ms)"


# =============================================================================
# QTT Container
# =============================================================================

class QTTContainer:
    """
    Universal Container for .qtt format.
    
    Bundles Index (brain) + Payload (body) into a single portable file.
    Supports zero-copy index mapping and byte-range payload access.
    
    Example:
        # Create and save
        >>> container = QTTContainer.from_text_corpus(sentences)
        >>> container.save("corpus.qtt")
        
        # Open and slice
        >>> with QTTContainer.open("corpus.qtt") as f:
        ...     result = f.slice(query="quantum physics")
        ...     for match in result.matches:
        ...         print(f.read_text(match))
    """
    
    def __init__(self):
        self.header = QTTHeader()
        self.metadata: Dict[str, Any] = {}
        self.index_data: bytes = b''
        self.payload_data: bytes = b''
        self.footer = QTTFooter()
        
        # Runtime state (for open containers)
        self._file: Optional[BinaryIO] = None
        self._mmap: Optional[mmap.mmap] = None
        self._index_cache: Optional[Dict[str, Any]] = None
        self._embedding_model: Optional[Any] = None
        self._decompressed_payload: Optional[bytes] = None
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def from_text_corpus(cls,
                         sentences: List[str],
                         n_subvectors: int = 12,
                         n_centroids: int = 256,
                         compress_payload: bool = True,
                         model_name: str = 'all-MiniLM-L6-v2',
                         show_progress: bool = True) -> 'QTTContainer':
        """
        Create a .qtt container from a text corpus.
        
        Args:
            sentences: List of text sentences/documents
            n_subvectors: PQ subvectors (default 12 for 384-dim embeddings)
            n_centroids: Centroids per subvector (default 256 = 1 byte codes)
            compress_payload: GZIP compress the raw text
            model_name: Sentence-transformer model name
            show_progress: Show progress bars
            
        Returns:
            QTTContainer ready to save
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import MiniBatchKMeans
        
        container = cls()
        container.header.mode = MODE_SEMANTIC
        
        # =====================================================================
        # Step 1: Compute embeddings
        # =====================================================================
        if show_progress:
            print("Step 1/4: Computing embeddings...")
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            sentences, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        ).astype(np.float32)
        
        n_docs, dim = embeddings.shape
        subvector_dim = dim // n_subvectors
        
        # =====================================================================
        # Step 2: Train Product Quantization
        # =====================================================================
        if show_progress:
            print("Step 2/4: Training Product Quantization...")
        
        # Auto-adjust centroids for small corpora
        effective_centroids = min(n_centroids, n_docs)
        if effective_centroids < n_centroids and show_progress:
            print(f"  Note: Reduced centroids from {n_centroids} to {effective_centroids} (small corpus)")
        
        centroids = np.zeros((n_subvectors, effective_centroids, subvector_dim), dtype=np.float32)
        codes = np.zeros((n_docs, n_subvectors), dtype=np.uint8)
        
        for sv in range(n_subvectors):
            start = sv * subvector_dim
            end = start + subvector_dim
            subvectors = embeddings[:, start:end]
            
            kmeans = MiniBatchKMeans(
                n_clusters=effective_centroids, 
                random_state=42,
                batch_size=min(1024, n_docs),
                n_init=3
            )
            codes[:, sv] = kmeans.fit_predict(subvectors)
            centroids[sv] = kmeans.cluster_centers_
        
        # =====================================================================
        # Step 3: Build payload with offset table
        # =====================================================================
        if show_progress:
            print("Step 3/4: Building payload...")
        
        # Encode all sentences to UTF-8 bytes
        encoded_sentences = [s.encode('utf-8') for s in sentences]
        
        # Compute offsets within the raw (decompressed) payload
        offsets = np.zeros(n_docs, dtype=np.int64)
        lengths = np.zeros(n_docs, dtype=np.int32)
        
        current_offset = 0
        for i, enc in enumerate(encoded_sentences):
            offsets[i] = current_offset
            lengths[i] = len(enc)
            current_offset += len(enc)
        
        # Concatenate all raw text
        raw_payload = b''.join(encoded_sentences)
        
        # Optionally compress
        if compress_payload:
            payload_bytes = gzip.compress(raw_payload, compresslevel=6)
            payload_compressed = True
        else:
            payload_bytes = raw_payload
            payload_compressed = False
        
        # =====================================================================
        # Step 4: Pack index
        # =====================================================================
        if show_progress:
            print("Step 4/4: Packing index...")
        
        index_dict = {
            'centroids': centroids,
            'codes': codes,
            'offsets': offsets,
            'lengths': lengths,
            'n_subvectors': n_subvectors,
            'n_centroids': effective_centroids,
            'subvector_dim': subvector_dim,
            'embedding_dim': dim
        }
        
        index_bytes = pickle.dumps(index_dict, protocol=pickle.HIGHEST_PROTOCOL)
        
        # =====================================================================
        # Populate container
        # =====================================================================
        container.metadata = {
            'mode': 'semantic',
            'n_documents': n_docs,
            'embedding_dim': dim,
            'n_subvectors': n_subvectors,
            'n_centroids': effective_centroids,
            'model_name': model_name,
            'payload_compressed': payload_compressed,
            'payload_raw_size': len(raw_payload),
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'qtt_version': QTT_VERSION
        }
        
        container.index_data = index_bytes
        container.payload_data = payload_bytes
        
        if show_progress:
            print(f"\nContainer built:")
            print(f"  Documents: {n_docs:,}")
            print(f"  Index size: {len(index_bytes):,} bytes")
            print(f"  Payload size: {len(payload_bytes):,} bytes")
            print(f"  Total: {len(index_bytes) + len(payload_bytes):,} bytes")
            
            # Compression stats
            raw_embeddings = n_docs * dim * 4
            print(f"\nCompression:")
            print(f"  Raw embeddings: {raw_embeddings:,} bytes")
            print(f"  PQ index: {len(index_bytes):,} bytes")
            print(f"  Ratio: {raw_embeddings / len(index_bytes):.1f}x")
        
        return container
    
    @classmethod
    def from_spatial_data(cls,
                          data: np.ndarray,
                          max_rank: int = 64,
                          rel_eps: float = 1e-10,
                          show_progress: bool = True) -> 'QTTContainer':
        """
        Create a .qtt container from spatial/physics data.
        
        Args:
            data: N-dimensional numpy array
            max_rank: Maximum TT rank
            rel_eps: Relative epsilon for rank truncation
            show_progress: Show progress
            
        Returns:
            QTTContainer ready to save
        """
        from qtt.spatial import tt_svd
        
        container = cls()
        container.header.mode = MODE_SPATIAL
        
        if show_progress:
            print(f"Compressing {data.shape} tensor with max_rank={max_rank}...")
        
        # TT-SVD decomposition
        cores = tt_svd(data, max_rank, rel_eps)
        
        # Pack index
        index_dict = {
            'cores': cores,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'max_rank': max_rank,
            'ranks': [c.shape[2] for c in cores]
        }
        
        index_bytes = pickle.dumps(index_dict, protocol=pickle.HIGHEST_PROTOCOL)
        
        container.metadata = {
            'mode': 'spatial',
            'shape': list(data.shape),
            'dtype': str(data.dtype),
            'max_rank': max_rank,
            'ranks': [c.shape[2] for c in cores],
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'qtt_version': QTT_VERSION
        }
        
        container.index_data = index_bytes
        container.payload_data = b''  # Spatial has no separate payload
        
        if show_progress:
            compressed_size = len(index_bytes)
            original_size = data.nbytes
            print(f"  Original: {original_size:,} bytes")
            print(f"  Compressed: {compressed_size:,} bytes")
            print(f"  Ratio: {original_size / compressed_size:.1f}x")
        
        return container
    
    # =========================================================================
    # Save/Load
    # =========================================================================
    
    def save(self, path: Union[str, Path]) -> int:
        """
        Save container to .qtt file.
        
        Args:
            path: Output file path
            
        Returns:
            Total file size in bytes
        """
        path = Path(path)
        
        # Serialize metadata as JSON
        metadata_bytes = json.dumps(self.metadata, indent=None).encode('utf-8')
        
        # Calculate segment offsets
        header_size = QTTHeader.SIZE
        metadata_offset = header_size
        metadata_size = len(metadata_bytes)
        
        index_offset = metadata_offset + metadata_size
        index_size = len(self.index_data)
        
        payload_offset = index_offset + index_size
        payload_size = len(self.payload_data)
        
        footer_offset = payload_offset + payload_size
        
        # Update header
        self.header.metadata_offset = metadata_offset
        self.header.metadata_size = metadata_size
        self.header.index_offset = index_offset
        self.header.index_size = index_size
        self.header.payload_offset = payload_offset
        self.header.payload_size = payload_size
        self.header.footer_offset = footer_offset
        
        # Compute checksums
        self.footer.index_checksum = hashlib.sha256(self.index_data).digest()
        self.footer.payload_checksum = hashlib.sha256(self.payload_data).digest()
        
        # Write file
        with open(path, 'wb') as f:
            f.write(self.header.pack())
            f.write(metadata_bytes)
            f.write(self.index_data)
            f.write(self.payload_data)
            f.write(self.footer.pack())
        
        # Compute and update total checksum
        with open(path, 'rb') as f:
            total_hash = hashlib.sha256(f.read()).digest()
        
        self.footer.total_checksum = total_hash
        with open(path, 'r+b') as f:
            f.seek(footer_offset)
            f.write(self.footer.pack())
        
        total_size = footer_offset + QTTFooter.SIZE
        print(f"Saved: {path} ({total_size:,} bytes)")
        
        return total_size
    
    @classmethod
    def open(cls, 
             path: Union[str, Path], 
             mmap_index: bool = True,
             validate_checksums: bool = False) -> 'QTTContainer':
        """
        Open a .qtt file for reading.
        
        Uses memory-mapping for zero-copy index access.
        Payload is read on-demand via byte-range fetching.
        
        Args:
            path: Path to .qtt file
            mmap_index: Use mmap for index (recommended for large files)
            validate_checksums: Verify SHA-256 checksums on open
            
        Returns:
            QTTContainer ready for slicing
        """
        path = Path(path)
        container = cls()
        
        # Open file handle
        container._file = open(path, 'rb')
        
        # Read and validate header
        header_bytes = container._file.read(QTTHeader.SIZE)
        container.header = QTTHeader.unpack(header_bytes)
        container.header.validate()
        
        # Read metadata
        container._file.seek(container.header.metadata_offset)
        metadata_bytes = container._file.read(container.header.metadata_size)
        container.metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # Memory-map or read index
        if mmap_index and container.header.index_size > 0:
            container._mmap = mmap.mmap(
                container._file.fileno(), 
                0, 
                access=mmap.ACCESS_READ
            )
            index_view = container._mmap[
                container.header.index_offset:
                container.header.index_offset + container.header.index_size
            ]
            container._index_cache = pickle.loads(bytes(index_view))
        else:
            container._file.seek(container.header.index_offset)
            container.index_data = container._file.read(container.header.index_size)
            container._index_cache = pickle.loads(container.index_data)
        
        # Validate checksums if requested
        if validate_checksums:
            container._validate_checksums()
        
        return container
    
    def _validate_checksums(self):
        """Validate file integrity via checksums."""
        # Read footer
        self._file.seek(self.header.footer_offset)
        footer_bytes = self._file.read(QTTFooter.SIZE)
        self.footer = QTTFooter.unpack(footer_bytes)
        
        # Validate index checksum
        self._file.seek(self.header.index_offset)
        index_data = self._file.read(self.header.index_size)
        actual_index_hash = hashlib.sha256(index_data).digest()
        if actual_index_hash != self.footer.index_checksum:
            raise ValueError("Index checksum mismatch - file may be corrupted")
        
        # Validate payload checksum
        if self.header.payload_size > 0:
            self._file.seek(self.header.payload_offset)
            payload_data = self._file.read(self.header.payload_size)
            actual_payload_hash = hashlib.sha256(payload_data).digest()
            if actual_payload_hash != self.footer.payload_checksum:
                raise ValueError("Payload checksum mismatch - file may be corrupted")
    
    def close(self):
        """Close file handles and release resources."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
        self._index_cache = None
        self._decompressed_payload = None
        self._embedding_model = None
    
    def __enter__(self) -> 'QTTContainer':
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # =========================================================================
    # Slicing API
    # =========================================================================
    
    def slice(self,
              coords: Optional[Tuple[int, ...]] = None,
              query: Optional[str] = None,
              embedding: Optional[np.ndarray] = None,
              top_k: int = 10) -> Union[float, SliceResult]:
        """
        Universal slice interface.
        
        For spatial data:
            >>> value = container.slice(coords=(x, y, z))
        
        For semantic data:
            >>> result = container.slice(query="quantum physics")
            >>> for match in result.matches:
            ...     text = container.read_text(match)
        
        Args:
            coords: Spatial coordinates (for physics data)
            query: Text query (for semantic data)
            embedding: Pre-computed embedding vector (for semantic data)
            top_k: Number of results for semantic search
            
        Returns:
            float (spatial) or SliceResult (semantic)
        """
        if self.header.mode == MODE_SPATIAL:
            if coords is None:
                raise ValueError("Spatial mode requires coords parameter")
            return self._slice_spatial(coords)
        elif self.header.mode == MODE_SEMANTIC:
            return self._slice_semantic(query, embedding, top_k)
        else:
            raise ValueError(f"Unknown mode: {self.header.mode}")
    
    def _slice_spatial(self, coords: Tuple[int, ...]) -> float:
        """Slice spatial data via TT contraction."""
        cores = self._index_cache['cores']
        
        if len(coords) != len(cores):
            raise ValueError(f"Expected {len(cores)} coordinates, got {len(coords)}")
        
        # TT contraction: O(d·r²)
        result = cores[0][:, coords[0], :]
        for i in range(1, len(cores)):
            result = result @ cores[i][:, coords[i], :]
        
        return float(result.flatten()[0])
    
    def _slice_semantic(self, 
                        query: Optional[str],
                        embedding: Optional[np.ndarray],
                        top_k: int) -> SliceResult:
        """Slice semantic data via PQ lookup."""
        start = time.perf_counter()
        
        # Get query embedding
        if embedding is None:
            if query is None:
                raise ValueError("Must provide query or embedding")
            embedding = self._embed_query(query)
        
        index = self._index_cache
        centroids = index['centroids']
        codes = index['codes']
        offsets = index['offsets']
        lengths = index['lengths']
        
        n_subvectors = index['n_subvectors']
        n_centroids = index['n_centroids']
        subvector_dim = index['subvector_dim']
        n_docs = len(codes)
        
        # Precompute distance tables: O(M·K·D/M) = O(K·D)
        distance_tables = np.zeros((n_subvectors, n_centroids), dtype=np.float32)
        for sv in range(n_subvectors):
            start_dim = sv * subvector_dim
            end_dim = start_dim + subvector_dim
            query_sub = embedding[start_dim:end_dim]
            
            # Squared distance from query subvector to each centroid
            diffs = query_sub - centroids[sv]
            distance_tables[sv] = np.sum(diffs ** 2, axis=1)
        
        # Compute approximate distances via lookup: O(N·M)
        distances = np.zeros(n_docs, dtype=np.float32)
        for sv in range(n_subvectors):
            distances += distance_tables[sv, codes[:, sv]]
        
        # Get top-k
        if top_k < n_docs:
            top_indices = np.argpartition(distances, top_k)[:top_k]
            top_indices = top_indices[np.argsort(distances[top_indices])]
        else:
            top_indices = np.argsort(distances)[:top_k]
        
        # Build matches
        matches = []
        for idx in top_indices:
            matches.append(SemanticMatch(
                score=1.0 / (1.0 + distances[idx]),
                sentence_id=int(idx),
                offset=int(offsets[idx]),
                length=int(lengths[idx])
            ))
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return SliceResult(
            query=query or "<embedding>",
            matches=matches,
            access_time_ms=elapsed_ms
        )
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query string using sentence-transformer."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            model_name = self.metadata.get('model_name', 'all-MiniLM-L6-v2')
            self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model.encode(query, convert_to_numpy=True)
    
    # =========================================================================
    # Payload Access
    # =========================================================================
    
    def read_payload(self, offset: int, length: int) -> bytes:
        """
        Read raw bytes from payload segment.
        
        For semantic data, this retrieves the original text.
        Uses byte-range fetching within decompressed payload.
        
        Args:
            offset: Byte offset within the (decompressed) payload
            length: Number of bytes to read
            
        Returns:
            Raw bytes (UTF-8 encoded text for semantic data)
        """
        if self.header.payload_size == 0:
            raise ValueError("No payload in this container (spatial mode?)")
        
        is_compressed = self.metadata.get('payload_compressed', False)
        
        if is_compressed:
            # Decompress full payload on first access (cached)
            # Note: For production with large payloads, use block compression
            if self._decompressed_payload is None:
                self._file.seek(self.header.payload_offset)
                compressed = self._file.read(self.header.payload_size)
                self._decompressed_payload = gzip.decompress(compressed)
            
            return self._decompressed_payload[offset:offset + length]
        else:
            # Direct byte-range read
            absolute_offset = self.header.payload_offset + offset
            self._file.seek(absolute_offset)
            return self._file.read(length)
    
    def read_text(self, match: SemanticMatch) -> str:
        """
        Convenience method to read text for a search match.
        
        Args:
            match: SemanticMatch from slice() result
            
        Returns:
            Decoded UTF-8 text string
        """
        raw = self.read_payload(match.offset, match.length)
        return raw.decode('utf-8')
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def info(self) -> str:
        """Return human-readable info string about the container."""
        mode_str = 'SPATIAL' if self.header.mode == MODE_SPATIAL else 'SEMANTIC'
        
        lines = [
            "QTT Container",
            "=" * 50,
            f"Mode: {mode_str}",
            f"Version: {self.header.version}",
            "",
            "Segments:",
            f"  Header: {QTTHeader.SIZE} bytes",
            f"  Metadata: {self.header.metadata_size:,} bytes",
            f"  Index: {self.header.index_size:,} bytes",
            f"  Payload: {self.header.payload_size:,} bytes",
            f"  Footer: {QTTFooter.SIZE} bytes",
            "",
        ]
        
        if self.metadata:
            lines.append("Metadata:")
            for k, v in self.metadata.items():
                if isinstance(v, list) and len(v) > 5:
                    v = f"[{v[0]}, {v[1]}, ..., {v[-1]}] ({len(v)} items)"
                lines.append(f"  {k}: {v}")
        
        return "\n".join(lines)
    
    @property
    def is_spatial(self) -> bool:
        """Check if container is spatial mode."""
        return self.header.mode == MODE_SPATIAL
    
    @property
    def is_semantic(self) -> bool:
        """Check if container is semantic mode."""
        return self.header.mode == MODE_SEMANTIC
    
    @property
    def total_size(self) -> int:
        """Total file size in bytes."""
        return (
            QTTHeader.SIZE + 
            self.header.metadata_size + 
            self.header.index_size + 
            self.header.payload_size + 
            QTTFooter.SIZE
        )
