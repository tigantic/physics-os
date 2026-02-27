"""
5.17 — Retrieval-Augmented Generation for Physics
===================================================

RAG system that indexes physics knowledge (equations, papers,
simulation results) and retrieves relevant context to ground
LLM outputs in verified physics.

Components:
    * PhysicsDocument — a chunk of physics knowledge
    * VectorStore — in-memory embedding index (cosine similarity)
    * PhysicsRetriever — search interface with re-ranking
    * RAGPipeline — retrieve → augment prompt → generate answer
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Document types ────────────────────────────────────────────────

class DocType(Enum):
    """Type of physics knowledge document."""
    EQUATION = auto()
    PAPER_ABSTRACT = auto()
    SIMULATION_RESULT = auto()
    BOUNDARY_CONDITION = auto()
    MATERIAL_PROPERTY = auto()
    SOLVER_CONFIG = auto()
    TUTORIAL = auto()
    ERROR_ANALYSIS = auto()


@dataclass
class PhysicsDocument:
    """A chunk of indexed physics knowledge."""
    text: str
    doc_type: DocType
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    embedding: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.sha256(
                self.text.encode()
            ).hexdigest()[:16]


# ── Lightweight text embedder ─────────────────────────────────────

class BagOfWordsEmbedder:
    """Simple TF-IDF–style embedder for offline/testing use.

    For production, swap with sentence-transformers or OpenAI embeddings.
    """

    def __init__(self, dim: int = 256, seed: int = 42) -> None:
        self.dim = dim
        self._vocab: Dict[str, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)

    def _tokenise(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _get_word_vec(self, word: str) -> np.ndarray:
        if word not in self._vocab:
            # Deterministic random vector from word hash
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(h)
            self._vocab[word] = rng.standard_normal(self.dim).astype(np.float32)
        return self._vocab[word]

    def embed(self, text: str) -> np.ndarray:
        """Embed text into a fixed-dim vector."""
        tokens = self._tokenise(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = np.array([self._get_word_vec(t) for t in tokens])
        avg = vecs.mean(axis=0)
        norm = np.linalg.norm(avg)
        return (avg / max(norm, 1e-10)).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


# ── Vector store ──────────────────────────────────────────────────

class VectorStore:
    """In-memory vector index with cosine similarity search."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self._ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, dim)
        self._documents: List[PhysicsDocument] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, doc: PhysicsDocument, embedding: np.ndarray) -> None:
        """Add a document with its embedding."""
        doc.embedding = embedding
        self._ids.append(doc.doc_id)
        self._documents.append(doc)
        if self._embeddings is None:
            self._embeddings = embedding.reshape(1, -1)
        else:
            self._embeddings = np.vstack([
                self._embeddings, embedding.reshape(1, -1)
            ])

    def add_batch(self, docs: List[PhysicsDocument],
                  embeddings: np.ndarray) -> None:
        for doc, emb in zip(docs, embeddings):
            self.add(doc, emb)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[PhysicsDocument, float]]:
        """Cosine similarity search."""
        if self._embeddings is None or len(self._ids) == 0:
            return []
        q = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True).clip(1e-10)
        normed = self._embeddings / norms
        sims = normed @ q
        top_idx = np.argsort(-sims)[:top_k]
        return [(self._documents[i], float(sims[i])) for i in top_idx]

    def remove(self, doc_id: str) -> bool:
        """Remove a document by ID."""
        if doc_id not in self._ids:
            return False
        idx = self._ids.index(doc_id)
        self._ids.pop(idx)
        self._documents.pop(idx)
        self._embeddings = np.delete(self._embeddings, idx, axis=0)
        if len(self._ids) == 0:
            self._embeddings = None
        return True


# ── Retriever ─────────────────────────────────────────────────────

class PhysicsRetriever:
    """Search interface with optional re-ranking and filtering."""

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[BagOfWordsEmbedder] = None,
    ) -> None:
        self.store = store or VectorStore()
        self.embedder = embedder or BagOfWordsEmbedder()

    def index_documents(self, docs: List[PhysicsDocument]) -> int:
        """Index a batch of documents."""
        texts = [d.text for d in docs]
        embeddings = self.embedder.embed_batch(texts)
        self.store.add_batch(docs, embeddings)
        return len(docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_type: Optional[DocType] = None,
        min_score: float = 0.0,
    ) -> List[Tuple[PhysicsDocument, float]]:
        """Retrieve relevant documents for a query."""
        q_emb = self.embedder.embed(query)
        results = self.store.search(q_emb, top_k=top_k * 3)  # over-fetch for filtering

        # Filter by type
        if doc_type is not None:
            results = [(d, s) for d, s in results if d.doc_type == doc_type]

        # Filter by score
        results = [(d, s) for d, s in results if s >= min_score]

        return results[:top_k]


# ── RAG Pipeline ──────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    retrieved_docs: List[Tuple[str, float]]  # (doc_id, score)
    context_used: str
    confidence: float


class RAGPipeline:
    """Retrieval-Augmented Generation for physics questions.

    1. Retrieve relevant documents
    2. Build augmented context
    3. Generate answer using LLM backend

    Works with MockLLMBackend for offline operation.
    """

    def __init__(
        self,
        retriever: Optional[PhysicsRetriever] = None,
        llm_complete: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.retriever = retriever or PhysicsRetriever()
        self._complete = llm_complete or self._default_complete

    @staticmethod
    def _default_complete(prompt: str) -> str:
        """Extractive fallback when no LLM is available."""
        lines = prompt.split("\n")
        context_lines = [l for l in lines if l.strip() and not l.startswith("Question")]
        if context_lines:
            return context_lines[0].strip()
        return "No relevant information found."

    def _build_prompt(self, query: str, docs: List[Tuple[PhysicsDocument, float]]) -> str:
        parts = ["Answer the following physics question using the provided context.\n"]
        parts.append("Context:")
        for i, (doc, score) in enumerate(docs):
            parts.append(f"  [{i+1}] ({doc.doc_type.name}) {doc.text}")
        parts.append(f"\nQuestion: {query}")
        parts.append("Answer:")
        return "\n".join(parts)

    def query(
        self,
        question: str,
        top_k: int = 3,
        doc_type: Optional[DocType] = None,
    ) -> RAGResponse:
        """Full RAG query."""
        docs = self.retriever.search(question, top_k=top_k, doc_type=doc_type)
        prompt = self._build_prompt(question, docs)
        answer = self._complete(prompt)

        avg_score = float(np.mean([s for _, s in docs])) if docs else 0.0

        return RAGResponse(
            answer=answer,
            retrieved_docs=[(d.doc_id, s) for d, s in docs],
            context_used=prompt,
            confidence=avg_score,
        )

    def index(self, docs: List[PhysicsDocument]) -> int:
        """Convenience: index documents."""
        return self.retriever.index_documents(docs)


__all__ = [
    "DocType",
    "PhysicsDocument",
    "BagOfWordsEmbedder",
    "VectorStore",
    "PhysicsRetriever",
    "RAGResponse",
    "RAGPipeline",
]
