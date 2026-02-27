"""
QTT Semantic Compression Module
===============================

Product Quantization for semantic embeddings with offset-based retrieval.

Features:
- O(M·K) storage for codebook vs O(N·D) for embeddings
- O(1) per-document distance computation via lookup tables
- Byte-addressable document offsets for payload retrieval
- Support for multiple embedding models

Mathematical Foundation:
    A D-dimensional vector is split into M subvectors of dimension D/M.
    Each subvector is quantized to one of K centroids (typically K=256).
    
    Storage: N documents × M bytes (for K=256)
    vs Raw:  N documents × D × 4 bytes
    
    Compression: 4D/M × (typical: 32x for D=384, M=12)

Usage:
    >>> from qtt.semantic import SemanticIndex
    
    # Build index from text
    >>> index = SemanticIndex.from_texts(sentences)
    
    # Search
    >>> results = index.search("quantum physics", top_k=10)
    >>> for match in results:
    ...     print(match.score, match.offset, match.length)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class SearchMatch:
    """Result from semantic search."""
    score: float
    document_id: int
    offset: int
    length: int
    
    def __repr__(self):
        return f"SearchMatch(score={self.score:.3f}, doc={self.document_id}, offset={self.offset})"


@dataclass
class SearchResult:
    """Container for search results."""
    query: str
    matches: List[SearchMatch]
    search_time_ms: float
    
    def __repr__(self):
        return f"SearchResult(query='{self.query[:30]}...', matches={len(self.matches)}, time={self.search_time_ms:.1f}ms)"


class ProductQuantizer:
    """
    Product Quantization for vector compression.
    
    Splits D-dimensional vectors into M subvectors and quantizes
    each to one of K centroids.
    
    Example:
        >>> pq = ProductQuantizer(n_subvectors=12, n_centroids=256)
        >>> pq.train(embeddings)
        >>> codes = pq.encode(embeddings)
        >>> distances = pq.compute_distances(query_embedding)
    """
    
    def __init__(self,
                 n_subvectors: int = 12,
                 n_centroids: int = 256,
                 random_state: int = 42):
        """
        Initialize Product Quantizer.
        
        Args:
            n_subvectors: Number of subvectors (M)
            n_centroids: Centroids per subvector (K, typically 256)
            random_state: Random seed for k-means
        """
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.random_state = random_state
        
        self._centroids: Optional[np.ndarray] = None  # (M, K, D/M)
        self._embedding_dim: Optional[int] = None
        self._subvector_dim: Optional[int] = None
    
    @property
    def centroids(self) -> np.ndarray:
        """Get trained centroids."""
        if self._centroids is None:
            raise ValueError("PQ not trained. Call train() first.")
        return self._centroids
    
    @property
    def is_trained(self) -> bool:
        """Check if PQ has been trained."""
        return self._centroids is not None
    
    def train(self, 
              embeddings: np.ndarray,
              verbose: bool = False) -> 'ProductQuantizer':
        """
        Train PQ centroids using k-means.
        
        Args:
            embeddings: (N, D) array of training vectors
            verbose: Print progress
            
        Returns:
            self for chaining
        """
        from sklearn.cluster import MiniBatchKMeans
        
        n_docs, dim = embeddings.shape
        
        if dim % self.n_subvectors != 0:
            raise ValueError(
                f"Embedding dimension {dim} not divisible by n_subvectors {self.n_subvectors}"
            )
        
        self._embedding_dim = dim
        self._subvector_dim = dim // self.n_subvectors
        
        # Auto-adjust centroids if corpus is too small
        effective_centroids = min(self.n_centroids, n_docs)
        if effective_centroids < self.n_centroids:
            if verbose:
                print(f"  Note: Reduced centroids from {self.n_centroids} to {effective_centroids} (small corpus)")
        
        self._centroids = np.zeros(
            (self.n_subvectors, effective_centroids, self._subvector_dim),
            dtype=np.float32
        )
        self._effective_centroids = effective_centroids
        
        for sv in range(self.n_subvectors):
            if verbose:
                print(f"  Training subvector {sv+1}/{self.n_subvectors}")
            
            start = sv * self._subvector_dim
            end = start + self._subvector_dim
            subvectors = embeddings[:, start:end].astype(np.float32)
            
            kmeans = MiniBatchKMeans(
                n_clusters=effective_centroids,
                random_state=self.random_state,
                batch_size=min(1024, n_docs),
                n_init=3
            )
            kmeans.fit(subvectors)
            self._centroids[sv] = kmeans.cluster_centers_
        
        return self
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode embeddings to PQ codes.
        
        Args:
            embeddings: (N, D) array of vectors
            
        Returns:
            (N, M) array of uint8 codes
        """
        if not self.is_trained:
            raise ValueError("PQ not trained. Call train() first.")
        
        n_docs = len(embeddings)
        effective_centroids = self._centroids.shape[1]
        
        # Use uint8 if possible, otherwise uint16
        if effective_centroids <= 256:
            codes = np.zeros((n_docs, self.n_subvectors), dtype=np.uint8)
        else:
            codes = np.zeros((n_docs, self.n_subvectors), dtype=np.uint16)
        
        for sv in range(self.n_subvectors):
            start = sv * self._subvector_dim
            end = start + self._subvector_dim
            subvectors = embeddings[:, start:end].astype(np.float32)
            
            # Find nearest centroid for each subvector
            # (N, 1, D/M) - (1, K, D/M) -> (N, K)
            diffs = subvectors[:, np.newaxis, :] - self._centroids[sv][np.newaxis, :, :]
            distances = np.sum(diffs ** 2, axis=2)
            codes[:, sv] = np.argmin(distances, axis=1)
        
        return codes
    
    def compute_distance_tables(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute distance tables for a query.
        
        Args:
            query: (D,) query vector
            
        Returns:
            (M, K) distance table
        """
        if not self.is_trained:
            raise ValueError("PQ not trained. Call train() first.")
        
        effective_centroids = self._centroids.shape[1]
        tables = np.zeros((self.n_subvectors, effective_centroids), dtype=np.float32)
        
        for sv in range(self.n_subvectors):
            start = sv * self._subvector_dim
            end = start + self._subvector_dim
            query_sub = query[start:end].astype(np.float32)
            
            # Distance from query subvector to each centroid
            diffs = query_sub - self._centroids[sv]
            tables[sv] = np.sum(diffs ** 2, axis=1)
        
        return tables
    
    def compute_distances(self, 
                          query: np.ndarray,
                          codes: np.ndarray) -> np.ndarray:
        """
        Compute approximate distances from query to encoded documents.
        
        Args:
            query: (D,) query vector
            codes: (N, M) PQ codes
            
        Returns:
            (N,) approximate squared distances
        """
        tables = self.compute_distance_tables(query)
        
        n_docs = len(codes)
        distances = np.zeros(n_docs, dtype=np.float32)
        
        for sv in range(self.n_subvectors):
            distances += tables[sv, codes[:, sv].astype(np.int32)]
        
        return distances
    
    def save(self, path: str):
        """Save PQ to file."""
        np.savez_compressed(
            path,
            centroids=self._centroids,
            n_subvectors=self.n_subvectors,
            n_centroids=self.n_centroids,
            embedding_dim=self._embedding_dim,
            subvector_dim=self._subvector_dim
        )
    
    @classmethod
    def load(cls, path: str) -> 'ProductQuantizer':
        """Load PQ from file."""
        data = np.load(path)
        
        pq = cls(
            n_subvectors=int(data['n_subvectors']),
            n_centroids=int(data['n_centroids'])
        )
        pq._centroids = data['centroids']
        pq._embedding_dim = int(data['embedding_dim'])
        pq._subvector_dim = int(data['subvector_dim'])
        
        return pq


class SemanticIndex:
    """
    Complete semantic search index with offset-based retrieval.
    
    Combines:
    - Embedding model for text encoding
    - Product Quantization for compression
    - Offset table for byte-addressable retrieval
    
    Example:
        >>> index = SemanticIndex.from_texts(sentences)
        >>> results = index.search("quantum physics")
        >>> for match in results.matches:
        ...     print(match.offset, match.length)
    """
    
    def __init__(self,
                 n_subvectors: int = 12,
                 n_centroids: int = 256,
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic index.
        
        Args:
            n_subvectors: PQ subvectors
            n_centroids: Centroids per subvector
            model_name: Sentence-transformer model name
        """
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids
        self.model_name = model_name
        
        self._pq = ProductQuantizer(n_subvectors, n_centroids)
        self._codes: Optional[np.ndarray] = None
        self._offsets: Optional[np.ndarray] = None
        self._lengths: Optional[np.ndarray] = None
        self._embedding_model: Optional[Any] = None
        self._n_documents: int = 0
    
    @property
    def n_documents(self) -> int:
        """Number of indexed documents."""
        return self._n_documents
    
    @classmethod
    def from_texts(cls,
                   texts: List[str],
                   n_subvectors: int = 12,
                   n_centroids: int = 256,
                   model_name: str = 'all-MiniLM-L6-v2',
                   show_progress: bool = True) -> 'SemanticIndex':
        """
        Build index from text corpus.
        
        Args:
            texts: List of text documents
            n_subvectors: PQ subvectors
            n_centroids: Centroids per subvector
            model_name: Embedding model name
            show_progress: Show progress bars
            
        Returns:
            Trained SemanticIndex
        """
        from sentence_transformers import SentenceTransformer
        
        index = cls(n_subvectors, n_centroids, model_name)
        
        # Compute embeddings
        if show_progress:
            print("Computing embeddings...")
        
        index._embedding_model = SentenceTransformer(model_name)
        embeddings = index._embedding_model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # Train PQ
        if show_progress:
            print("Training Product Quantization...")
        
        index._pq.train(embeddings, verbose=show_progress)
        index._codes = index._pq.encode(embeddings)
        
        # Build offset table
        encoded_texts = [t.encode('utf-8') for t in texts]
        index._offsets = np.zeros(len(texts), dtype=np.int64)
        index._lengths = np.zeros(len(texts), dtype=np.int32)
        
        current_offset = 0
        for i, enc in enumerate(encoded_texts):
            index._offsets[i] = current_offset
            index._lengths[i] = len(enc)
            current_offset += len(enc)
        
        index._n_documents = len(texts)
        
        return index
    
    @classmethod
    def from_embeddings(cls,
                        embeddings: np.ndarray,
                        offsets: np.ndarray,
                        lengths: np.ndarray,
                        n_subvectors: int = 12,
                        n_centroids: int = 256) -> 'SemanticIndex':
        """
        Build index from pre-computed embeddings.
        
        Args:
            embeddings: (N, D) embedding matrix
            offsets: (N,) byte offsets
            lengths: (N,) byte lengths
            n_subvectors: PQ subvectors
            n_centroids: Centroids per subvector
            
        Returns:
            Trained SemanticIndex
        """
        index = cls(n_subvectors, n_centroids)
        
        index._pq.train(embeddings)
        index._codes = index._pq.encode(embeddings)
        index._offsets = offsets.astype(np.int64)
        index._lengths = lengths.astype(np.int32)
        index._n_documents = len(embeddings)
        
        return index
    
    def search(self,
               query: str,
               top_k: int = 10,
               embedding: Optional[np.ndarray] = None) -> SearchResult:
        """
        Search index for query.
        
        Args:
            query: Text query
            top_k: Number of results
            embedding: Pre-computed query embedding (optional)
            
        Returns:
            SearchResult with matches
        """
        start = time.perf_counter()
        
        # Get query embedding
        if embedding is None:
            embedding = self._embed_query(query)
        
        # Compute distances via PQ
        distances = self._pq.compute_distances(embedding, self._codes)
        
        # Get top-k
        if top_k < self._n_documents:
            top_indices = np.argpartition(distances, top_k)[:top_k]
            top_indices = top_indices[np.argsort(distances[top_indices])]
        else:
            top_indices = np.argsort(distances)[:top_k]
        
        # Build matches
        matches = []
        for idx in top_indices:
            matches.append(SearchMatch(
                score=1.0 / (1.0 + distances[idx]),
                document_id=int(idx),
                offset=int(self._offsets[idx]),
                length=int(self._lengths[idx])
            ))
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        return SearchResult(
            query=query,
            matches=matches,
            search_time_ms=elapsed_ms
        )
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query string."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model.encode(query, convert_to_numpy=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        pq_size = self._pq.centroids.nbytes if self._pq.is_trained else 0
        codes_size = self._codes.nbytes if self._codes is not None else 0
        offset_size = self._offsets.nbytes if self._offsets is not None else 0
        
        return {
            'n_documents': self._n_documents,
            'n_subvectors': self.n_subvectors,
            'n_centroids': self.n_centroids,
            'pq_size_bytes': pq_size,
            'codes_size_bytes': codes_size,
            'offset_size_bytes': offset_size,
            'total_size_bytes': pq_size + codes_size + offset_size
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Export index to dictionary (for serialization)."""
        return {
            'centroids': self._pq.centroids,
            'codes': self._codes,
            'offsets': self._offsets,
            'lengths': self._lengths,
            'n_subvectors': self.n_subvectors,
            'n_centroids': self.n_centroids,
            'subvector_dim': self._pq._subvector_dim,
            'embedding_dim': self._pq._embedding_dim
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticIndex':
        """Load index from dictionary."""
        index = cls(
            n_subvectors=data['n_subvectors'],
            n_centroids=data['n_centroids']
        )
        
        index._pq._centroids = data['centroids']
        index._pq._subvector_dim = data['subvector_dim']
        index._pq._embedding_dim = data['embedding_dim']
        index._codes = data['codes']
        index._offsets = data['offsets']
        index._lengths = data['lengths']
        index._n_documents = len(data['codes'])
        
        return index
