# QTT: Quantum Tensor Train Universal File System

**The Random-Access File Format for the AI Age**

QTT is a production-grade library that compresses and indexes data for instant random-access queries. It supports both **spatial data** (physics simulations, images) and **semantic data** (text, documents).

```
You don't "open" a file. You Slice it.
```

## Features

- **🚀 TT-SVD Compression**: 10-100× compression for N-dimensional arrays
- **🔍 Product Quantization**: Semantic search over millions of documents
- **📦 Universal Container**: Single `.qtt` file holds index + payload
- **⚡ Zero-Copy Access**: Memory-mapped index, byte-range payload
- **🔒 Integrity**: SHA-256 checksums for data validation

## Installation

```bash
pip install qtt

# With semantic search support
pip install qtt[semantic]

# With GPU acceleration
pip install qtt[gpu]

# Everything
pip install qtt[all]
```

## Quick Start

### Spatial Data (Physics/Simulation)

```python
from qtt import QTTContainer
import numpy as np

# Create 3D temperature field
temp = np.random.randn(128, 128, 128).astype(np.float32)

# Compress and save
container = QTTContainer.from_spatial_data(temp, max_rank=32)
container.save("simulation.qtt")

# Random access (10µs per query)
with QTTContainer.open("simulation.qtt") as f:
    value = f.slice(coords=(64, 64, 64))
    print(f"Temperature at center: {value:.2f}")
```

### Semantic Data (Text/Documents)

```python
from qtt import QTTContainer

sentences = [
    "Quantum mechanics describes particle behavior",
    "Machine learning is transforming AI",
    "The Roman Empire was ancient civilization",
    # ... millions more
]

# Create searchable archive
container = QTTContainer.from_text_corpus(sentences)
container.save("knowledge.qtt")

# Semantic search (6ms per query)
with QTTContainer.open("knowledge.qtt") as f:
    results = f.slice(query="quantum physics", top_k=10)
    
    for match in results.matches:
        text = f.read_text(match)
        print(f"[{match.score:.3f}] {text}")
```

## CLI

```bash
# Pack text file into .qtt
qtt pack corpus.txt -o library.qtt

# Query
qtt query library.qtt "quantum mechanics"

# Inspect
qtt info library.qtt

# Benchmark
qtt bench library.qtt
```

## File Format

```
┌─────────────────────────────────────────────┐
│ HEADER (64 bytes)                           │
│   Magic: QTT\x01, Version, Mode             │
├─────────────────────────────────────────────┤
│ METADATA (JSON)                             │
│   Configuration, document count, etc.       │
├─────────────────────────────────────────────┤
│ INDEX - The "Brain"                         │
│   • SPATIAL: TT-SVD cores                  │
│   • SEMANTIC: PQ centroids + codes         │
├─────────────────────────────────────────────┤
│ PAYLOAD - The "Body"                        │
│   • SPATIAL: None (cores ARE the data)     │
│   • SEMANTIC: GZIP-compressed raw text     │
├─────────────────────────────────────────────┤
│ FOOTER (96 bytes)                           │
│   SHA-256 checksums                         │
└─────────────────────────────────────────────┘
```

## Performance

| Mode | Data Size | Container | Compression | Query Time |
|------|-----------|-----------|-------------|------------|
| Spatial | 8.4 MB | 557 KB | **15×** | **10µs** |
| Semantic | 50K docs | 3.9 MB | **48×** | **6ms** |

Throughput:
- **Spatial**: 476,000 slices/sec
- **Semantic**: 160 QPS (with embedding)

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  QTTSlicer  │  ← Universal API
└──────┬──────┘
       │
       ├── coords=(x,y,z) ──→ TT Contraction (10µs)
       │                        │
       │                        ▼
       │                    float value
       │
       └── query="..." ──→ PQ Lookup (6ms)
                             │
                             ▼
                         matches[] with offsets
                             │
                             ▼
                         byte-range read
                             │
                             ▼
                         text content
```

## API Reference

### QTTContainer

```python
# Factory methods
QTTContainer.from_text_corpus(texts, n_subvectors=12, n_centroids=256)
QTTContainer.from_spatial_data(array, max_rank=64)

# I/O
container.save("file.qtt")
QTTContainer.open("file.qtt")

# Slicing
container.slice(coords=(x, y, z))  # Spatial
container.slice(query="...", top_k=10)  # Semantic

# Payload
container.read_text(match)  # Semantic only
container.read_payload(offset, length)  # Raw bytes
```

### QTTSlicer

High-level API that auto-detects mode:

```python
from qtt import QTTSlicer

# From data
slicer = QTTSlicer.from_spatial(array)
slicer = QTTSlicer.from_corpus(texts)

# From file
slicer = QTTSlicer.open("file.qtt")

# Unified slice
result = slicer.slice(coords=(x,y,z))  # or
result = slicer.slice(query="...")
```

## License

MIT License

## Citation

```bibtex
@software{qtt2026,
  title = {QTT: Quantum Tensor Train Universal File System},
  author = {Tigantic Holdings LLC},
  year = {2026},
  url = {https://github.com/hypertensor/qtt}
}
```
