"""
Search index generation and querying for documentation site.

This module provides full-text search capabilities for the
HyperTensor documentation site.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Single search result."""

    document_id: str
    title: str
    content_preview: str
    score: float
    highlights: list[str] = field(default_factory=list)
    url: str = ""
    doc_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.document_id,
            "title": self.title,
            "preview": self.content_preview,
            "score": self.score,
            "highlights": self.highlights,
            "url": self.url,
            "type": self.doc_type,
        }


@dataclass
class IndexedDocument:
    """Document in search index."""

    id: str
    title: str
    content: str
    url: str
    doc_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed fields
    term_frequencies: dict[str, int] = field(default_factory=dict)
    word_count: int = 0


class Tokenizer:
    """Text tokenizer for search indexing."""

    # Common stop words to exclude
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "can",
        "you",
        "your",
        "we",
        "our",
        "but",
        "not",
        "have",
        "been",
        "which",
        "when",
        "there",
        "their",
        "what",
        "so",
        "if",
        "out",
        "up",
        "into",
        "just",
    }

    def __init__(
        self,
        min_length: int = 2,
        max_length: int = 50,
        lowercase: bool = True,
        remove_stop_words: bool = True,
        stem: bool = True,
    ):
        """
        Initialize tokenizer.

        Args:
            min_length: Minimum token length
            max_length: Maximum token length
            lowercase: Convert to lowercase
            remove_stop_words: Remove common stop words
            stem: Apply basic stemming
        """
        self.min_length = min_length
        self.max_length = max_length
        self.lowercase = lowercase
        self.remove_stop_words = remove_stop_words
        self.stem = stem

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into terms.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Replace non-alphanumeric with spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Split into words
        words = text.split()

        # Filter and process
        tokens = []
        for word in words:
            # Length filter
            if len(word) < self.min_length or len(word) > self.max_length:
                continue

            # Stop word filter
            if self.remove_stop_words and word in self.STOP_WORDS:
                continue

            # Basic stemming (simple suffix removal)
            if self.stem:
                word = self._simple_stem(word)

            tokens.append(word)

        return tokens

    def _simple_stem(self, word: str) -> str:
        """Apply simple stemming rules."""
        # Remove common suffixes
        suffixes = ["ing", "ed", "er", "est", "ly", "tion", "ment", "ness"]
        for suffix in sorted(suffixes, key=len, reverse=True):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]
        return word


class SearchIndex:
    """
    Full-text search index using TF-IDF scoring.

    Provides efficient document indexing and retrieval with
    relevance-based ranking.
    """

    def __init__(self, tokenizer: Tokenizer | None = None):
        """
        Initialize search index.

        Args:
            tokenizer: Text tokenizer
        """
        self.tokenizer = tokenizer or Tokenizer()
        self.documents: dict[str, IndexedDocument] = {}

        # Inverted index: term -> set of document IDs
        self.inverted_index: dict[str, set[str]] = defaultdict(set)

        # Document frequency: term -> number of documents containing term
        self.document_frequency: dict[str, int] = defaultdict(int)

        # Total documents
        self.total_documents: int = 0

    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        url: str = "",
        doc_type: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """
        Add document to index.

        Args:
            doc_id: Unique document ID
            title: Document title
            content: Document content
            url: Document URL
            doc_type: Document type (api, guide, etc.)
            metadata: Additional metadata
        """
        # Create indexed document
        doc = IndexedDocument(
            id=doc_id,
            title=title,
            content=content,
            url=url,
            doc_type=doc_type,
            metadata=metadata or {},
        )

        # Tokenize content (include title with boost)
        text = f"{title} {title} {title} {content}"  # Title boost (3x)
        tokens = self.tokenizer.tokenize(text)

        # Calculate term frequencies
        term_freq: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1

        doc.term_frequencies = dict(term_freq)
        doc.word_count = len(tokens)

        # Update inverted index
        for term in term_freq:
            if doc_id not in self.inverted_index[term]:
                self.inverted_index[term].add(doc_id)
                self.document_frequency[term] += 1

        # Store document
        self.documents[doc_id] = doc
        self.total_documents = len(self.documents)

    def remove_document(self, doc_id: str):
        """
        Remove document from index.

        Args:
            doc_id: Document ID to remove
        """
        if doc_id not in self.documents:
            return

        doc = self.documents[doc_id]

        # Update inverted index
        for term in doc.term_frequencies:
            if doc_id in self.inverted_index[term]:
                self.inverted_index[term].remove(doc_id)
                self.document_frequency[term] -= 1

                if self.document_frequency[term] == 0:
                    del self.document_frequency[term]
                    del self.inverted_index[term]

        del self.documents[doc_id]
        self.total_documents = len(self.documents)

    def search(
        self,
        query: str,
        max_results: int = 10,
        doc_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for documents matching query.

        Args:
            query: Search query
            max_results: Maximum results to return
            doc_type_filter: Filter by document type

        Returns:
            List of search results ranked by relevance
        """
        if not query.strip():
            return []

        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Find candidate documents
        candidates: set[str] = set()
        for token in query_tokens:
            if token in self.inverted_index:
                candidates.update(self.inverted_index[token])

        if not candidates:
            return []

        # Score candidates using TF-IDF
        results = []
        for doc_id in candidates:
            doc = self.documents[doc_id]

            # Apply type filter
            if doc_type_filter and doc.doc_type != doc_type_filter:
                continue

            score = self._calculate_score(query_tokens, doc)

            if score > 0:
                # Generate preview with highlights
                preview, highlights = self._generate_preview(doc.content, query_tokens)

                results.append(
                    SearchResult(
                        document_id=doc_id,
                        title=doc.title,
                        content_preview=preview,
                        score=score,
                        highlights=highlights,
                        url=doc.url,
                        doc_type=doc.doc_type,
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:max_results]

    def _calculate_score(
        self,
        query_tokens: list[str],
        doc: IndexedDocument,
    ) -> float:
        """Calculate TF-IDF score for document."""
        score = 0.0

        for token in query_tokens:
            if token not in doc.term_frequencies:
                continue

            # Term frequency (TF)
            tf = doc.term_frequencies[token] / max(doc.word_count, 1)

            # Inverse document frequency (IDF)
            df = self.document_frequency.get(token, 1)
            idf = math.log(self.total_documents / df) + 1

            score += tf * idf

        return score

    def _generate_preview(
        self,
        content: str,
        query_tokens: list[str],
        preview_length: int = 200,
    ) -> tuple[str, list[str]]:
        """Generate preview with query highlights."""
        # Find best matching section
        sentences = re.split(r"[.!?]+", content)
        best_sentence = ""
        best_score = 0

        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for t in query_tokens if t in sentence_lower)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if not best_sentence:
            best_sentence = content[:preview_length]

        # Truncate preview
        preview = best_sentence.strip()
        if len(preview) > preview_length:
            preview = preview[:preview_length] + "..."

        # Find highlights
        highlights = []
        content_lower = content.lower()
        for token in query_tokens:
            idx = content_lower.find(token)
            if idx >= 0:
                # Get surrounding context
                start = max(0, idx - 20)
                end = min(len(content), idx + len(token) + 20)
                highlight = content[start:end].strip()
                if start > 0:
                    highlight = "..." + highlight
                if end < len(content):
                    highlight = highlight + "..."
                highlights.append(highlight)

        return preview, highlights[:3]  # Max 3 highlights

    def to_json(self) -> str:
        """Serialize index to JSON."""
        data = {
            "version": "1.0",
            "total_documents": self.total_documents,
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "content": doc.content[:500],  # Truncate for size
                    "url": doc.url,
                    "type": doc.doc_type,
                }
                for doc in self.documents.values()
            ],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SearchIndex":
        """Deserialize index from JSON."""
        data = json.loads(json_str)
        index = cls()

        for doc in data.get("documents", []):
            index.add_document(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                url=doc.get("url", ""),
                doc_type=doc.get("type", ""),
            )

        return index


class Indexer:
    """
    Document indexer for building search indices.

    Provides batch indexing and incremental updates.
    """

    def __init__(self):
        """Initialize indexer."""
        self.index = SearchIndex()
        self._checksums: dict[str, str] = {}

    def index_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        url: str = "",
        doc_type: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Index a document, updating if changed.

        Args:
            doc_id: Document ID
            title: Document title
            content: Document content
            url: Document URL
            doc_type: Document type
            metadata: Additional metadata

        Returns:
            True if document was indexed/updated
        """
        # Calculate checksum
        checksum = hashlib.md5(f"{title}{content}".encode()).hexdigest()

        # Check if changed
        if doc_id in self._checksums:
            if self._checksums[doc_id] == checksum:
                return False  # No change
            # Remove old version
            self.index.remove_document(doc_id)

        # Add new version
        self.index.add_document(
            doc_id=doc_id,
            title=title,
            content=content,
            url=url,
            doc_type=doc_type,
            metadata=metadata,
        )

        self._checksums[doc_id] = checksum
        return True

    def index_pages(self, pages: list[Any]) -> int:
        """
        Index multiple pages.

        Args:
            pages: List of Page objects

        Returns:
            Number of pages indexed
        """
        count = 0
        for page in pages:
            indexed = self.index_document(
                doc_id=page.path,
                title=page.title,
                content=page.content,
                url=page.path,
                doc_type=(
                    page.page_type.name
                    if hasattr(page.page_type, "name")
                    else str(page.page_type)
                ),
                metadata=page.metadata if hasattr(page, "metadata") else {},
            )
            if indexed:
                count += 1
        return count

    def get_index(self) -> SearchIndex:
        """Get the search index."""
        return self.index

    def save(self, path: str):
        """Save index to file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.index.to_json())

    def load(self, path: str):
        """Load index from file."""
        with open(path, encoding="utf-8") as f:
            self.index = SearchIndex.from_json(f.read())


def build_search_index(pages: list[Any]) -> SearchIndex:
    """
    Build search index from pages.

    Args:
        pages: List of Page objects

    Returns:
        SearchIndex instance
    """
    indexer = Indexer()
    indexer.index_pages(pages)
    return indexer.get_index()


def search(
    index: SearchIndex,
    query: str,
    max_results: int = 10,
) -> list[SearchResult]:
    """
    Search the index.

    Args:
        index: Search index
        query: Search query
        max_results: Maximum results

    Returns:
        List of search results
    """
    return index.search(query, max_results)
