# `hypertensor` — Deprecated Compatibility Shim

> **Status:** DEPRECATED as of v40.1.0

This directory exists **only** as a backward-compatibility shim.
All code has moved to the [`physics_os`](../physics_os/) package.

## Migration

```python
# Old (deprecated — emits DeprecationWarning)
from hypertensor.core.registry import DomainRegistry

# New (canonical)
from physics_os.core.registry import DomainRegistry
```

## Removal timeline

This shim will be removed in the next major release. Update your
imports to `physics_os` at your earliest convenience.

## What this directory contains

| File | Purpose |
|---|---|
| `__init__.py` | Meta-path finder that redirects `hypertensor.*` → `physics_os.*` |
| `DEPRECATED.md` | This notice |
