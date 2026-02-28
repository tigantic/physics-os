# The Physics OS — Release Process

This document describes the release process for The Physics OS.

---

## Quick Reference

```bash
# Standard release
make release VERSION=1.2.3

# Pre-release (beta)
make release VERSION=1.2.3b1 PRERELEASE=true
```

---

## Pre-Release Checklist

Before starting a release, verify:

- [ ] All CI checks pass on `main`
- [ ] No critical security vulnerabilities
- [ ] All P0 issues resolved
- [ ] Integration tests pass locally

---

## Release Steps

### 1. Version Bump

Update version in `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

Update version in `tensornet/__init__.py`:

```python
__version__ = "X.Y.Z"
```

### 2. Update CHANGELOG

Add release notes to `CHANGELOG.md`:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature 1
- New feature 2

### Changed
- Changed behavior 1

### Fixed
- Bug fix 1
- Bug fix 2

### Deprecated
- Deprecated feature 1

### Removed
- Removed feature 1

### Security
- Security fix 1
```

### 3. Run Release Validation

```bash
# Full validation suite
make release-validate

# Or manually:
make proofs      # Run formal proofs
make lint        # Check code style
make typecheck   # Type checking
make test        # Run all tests
```

### 4. Build Distribution

```bash
# Build wheel and sdist
python -m build

# Verify packages
twine check dist/*
```

### 5. Generate SBOM

```bash
# Generate Software Bill of Materials
python tools/scripts/generate_sbom.py --version X.Y.Z
```

### 6. Create Git Tag

```bash
git add -A
git commit -m "Release vX.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main --tags
```

### 7. Publish to PyPI

```bash
# Upload to PyPI (requires credentials)
twine upload dist/*

# Or use TestPyPI first
twine upload --repository testpypi dist/*
```

### 8. Create GitHub Release

1. Go to GitHub Releases
2. Click "Create a new release"
3. Select the tag `vX.Y.Z`
4. Title: `HyperTensor vX.Y.Z`
5. Body: Copy from CHANGELOG
6. Attach:
   - Wheel file (`*.whl`)
   - Source distribution (`*.tar.gz`)
   - SBOM file (`sbom-X.Y.Z.json`)
7. Click "Publish release"

---

## Post-Release

### 1. Verify Installation

```bash
pip install hypertensor==X.Y.Z
python -c "import tensornet; print(tensornet.__version__)"
```

### 2. Announce Release

- [ ] Update documentation site
- [ ] Post to relevant channels
- [ ] Update project roadmap

### 3. Prepare Next Version

Bump to next development version in `main`:

```bash
# In pyproject.toml
version = "X.Y.Z+1.dev0"
```

---

## Hotfix Process

For critical bug fixes:

1. Create hotfix branch from tag:
   ```bash
   git checkout -b hotfix/X.Y.Z+1 vX.Y.Z
   ```

2. Apply fix and commit

3. Follow normal release process with version `X.Y.Z+1`

4. Merge back to `main`:
   ```bash
   git checkout main
   git merge hotfix/X.Y.Z+1
   ```

---

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR (X)**: Breaking API changes
- **MINOR (Y)**: New features, backward compatible
- **PATCH (Z)**: Bug fixes, backward compatible

Pre-release versions:
- `X.Y.ZaN` - Alpha
- `X.Y.ZbN` - Beta
- `X.Y.ZrcN` - Release candidate

Development versions:
- `X.Y.Z.devN` - Development builds

---

## Troubleshooting

### Build Fails

```bash
# Clean build artifacts
make clean

# Rebuild
python -m build
```

### PyPI Upload Fails

```bash
# Check package validity
twine check dist/*

# Try uploading verbose
twine upload --verbose dist/*
```

### Tests Fail on Release Tag

```bash
# Checkout tag
git checkout vX.Y.Z

# Run tests
make test
```

---

## Automation

The CI pipeline automates:

- Version validation on tags
- Build verification
- Test execution
- Artifact creation

Manual steps remain:
- PyPI upload (requires credentials)
- GitHub release creation
- Announcement

---

## See Also

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [CHANGELOG.md](../CHANGELOG.md) - Version history
- [Makefile](../Makefile) - Build targets
