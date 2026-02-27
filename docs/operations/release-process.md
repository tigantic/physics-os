# Release Process

## Overview

HyperTensor uses tag-based release automation via GitHub Actions.

## Steps

1. **Version bump** — Update `VERSION` file at repository root
2. **Sync versions** — `make version-sync` propagates to all manifests
3. **Quality gates** — `make check` runs Python + Rust quality gates
4. **Tag** — `git tag -a v4.0.1 -m "Release 4.0.1"`
5. **Push** — `git push origin v4.0.1` triggers the release workflow
6. **Automated** — `.github/workflows/release.yml` builds, validates, and
   creates a GitHub Release with artifacts

## Version Files

The canonical version source is `VERSION`:

```
RELEASE=4.0.0
PLATFORM=3.0.0
SUBSTRATE_API=2.0.0
PACKAGE=40.0.1
RUNTIME=1.0.0
API_CONTRACT=1
```

`tools/sync_versions.py` propagates these to:

- `pyproject.toml` (PACKAGE)
- `CITATION.cff` (RELEASE)
- `tensornet/__init__.py` (PACKAGE)
- `hypertensor/__init__.py` (PACKAGE, RUNTIME)
- `Cargo.toml` header comment (RELEASE)
