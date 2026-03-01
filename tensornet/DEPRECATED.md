# `tensornet` — Deprecated Compatibility Shim

> **Status:** DEPRECATED as of v40.1.0

This directory exists **only** as a backward-compatibility shim.
All code has moved to the [`ontic`](../ontic/) package.

## Migration

```python
# Old (deprecated — emits DeprecationWarning)
from tensornet.core import TensorNetwork

# New (canonical)
from ontic.core import TensorNetwork
```

## Removal timeline

This shim will be removed in the next major release. Update your
imports to `ontic` / `ontic-engine` at your earliest convenience.

## What this directory contains

| File | Purpose |
|---|---|
| `__init__.py` | Meta-path finder that redirects `tensornet.*` → `ontic.*` |
| `DEPRECATED.md` | This notice |
