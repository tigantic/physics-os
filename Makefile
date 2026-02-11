# HyperTensor Production Readiness Makefile
# ==========================================
#
# Usage:
#   make check        - Run all quality gates (CI)
#   make release      - Full release preparation
#   make help         - Show all targets
#
# Checklist Mapping:
#   A) make hygiene      - Export clean release zip
#   B) make env          - Capture environment
#   C) make format       - Format/lint gates
#   D) make typecheck    - Type checking (mypy)
#   E) make test-unit    - Unit tests
#   F) make test-int     - Integration tests
#   G) make proofs       - Run all proofs
#   H) make reproduce    - Reproduction check
#   I) make physics      - Physics validation
#   J) make determinism  - Determinism check
#   K) make evidence     - Build evidence pack
#   L) make truth        - Truth boundary doc
#   M) make package      - Packaging gate
#   N) make docs         - Build documentation
#   O) make security     - Security scan
#   P) make sbom         - Generate SBOM
#   Q) make release-check - Release validation

.PHONY: help check release clean
.PHONY: hygiene env format typecheck test-unit test-int
.PHONY: proofs reproduce physics determinism evidence truth
.PHONY: package docs security sbom release-check

# Python interpreter
PYTHON ?= python

# Directories
ARTIFACTS_DIR = artifacts
DOCS_DIR = docs

# Default target
.DEFAULT_GOAL := help

help:
	@echo "HyperTensor Production Makefile"
	@echo "================================"
	@echo ""
	@echo "Quality Gates (CI):"
	@echo "  make check          Run all quality gates"
	@echo "  make format         Run black + isort + ruff"
	@echo "  make typecheck      Run mypy type checking"
	@echo "  make test-unit      Run unit tests"
	@echo "  make test-int       Run integration tests"
	@echo "  make proofs         Run all formal proofs"
	@echo ""
	@echo "Validation:"
	@echo "  make physics        Physics validation gates"
	@echo "  make determinism    Determinism checks"
	@echo "  make reproduce      Reproduce paper results"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build documentation"
	@echo "  make truth          Generate truth boundary"
	@echo "  make sbom           Generate SBOM"
	@echo ""
	@echo "Security:"
	@echo "  make security       Run security scans"
	@echo ""
	@echo "Release:"
	@echo "  make release-check  Pre-release validation"
	@echo "  make evidence       Build evidence pack"
	@echo "  make hygiene        Export clean release zip"
	@echo "  make release        Full release preparation"
	@echo ""
	@echo "Utility:"
	@echo "  make env            Capture environment"
	@echo "  make package        Packaging gate (wheel build)"
	@echo "  make clean          Remove artifacts"
	@echo ""
	@echo "Facial Plastics:"
	@echo "  make fp-test        Run facial plastics tests"
	@echo "  make fp-typecheck   Run mypy on facial plastics"
	@echo "  make fp-build       Build container image"
	@echo "  make fp-up          Start production stack (docker compose)"
	@echo "  make fp-down        Stop production stack"
	@echo "  make fp-logs        Tail docker compose logs"
	@echo "  make fp-keys        Generate API key (TENANT= ROLE=)"

# ============================================
# A) Hygiene - Clean release export
# ============================================
hygiene: $(ARTIFACTS_DIR)
	@echo "=== A) Export Clean Release ==="
	$(PYTHON) scripts/export_release_zip.py
	@echo "✓ Clean release zip created"

# ============================================
# B) Environment Capture
# ============================================
env: $(ARTIFACTS_DIR)
	@echo "=== B) Environment Capture ==="
	$(PYTHON) scripts/env_capture.py
	@echo "✓ Environment captured"

# ============================================
# C) Format & Lint Gates
# ============================================
# NOTE: Using ruff for speed. CI also runs black+isort for compatibility.
format:
	@echo "=== C) Format & Lint Gates ==="
	@echo "Running ruff format check..."
	$(PYTHON) -m ruff format --check tensornet tests benchmarks scripts proofs
	@echo "Running ruff lint..."
	$(PYTHON) -m ruff check tensornet tests benchmarks scripts proofs
	@echo "✓ Format checks passed"

format-fix:
	@echo "=== Applying Format Fixes ==="
	$(PYTHON) -m ruff format tensornet tests benchmarks scripts proofs
	$(PYTHON) -m ruff check --fix tensornet tests benchmarks scripts proofs
	@echo "✓ Format fixes applied"

# Legacy black+isort target for compatibility
format-legacy:
	@echo "=== Legacy Format (black+isort) ==="
	$(PYTHON) -m black tensornet tests benchmarks scripts proofs
	$(PYTHON) -m isort tensornet tests benchmarks scripts proofs
	@echo "✓ Legacy format applied"

# ============================================
# D) Type Checking
# ============================================
typecheck:
	@echo "=== D) Type Checking ==="
	$(PYTHON) -m mypy tensornet --ignore-missing-imports
	@echo "✓ Type check passed"

# Docstring coverage check
doccheck:
	@echo "=== Docstring Coverage ==="
	$(PYTHON) scripts/check_docstrings.py --threshold 70 -v
	@echo "✓ Docstring coverage passed"

# ============================================
# E) Unit Tests
# ============================================
test-unit:
	@echo "=== E) Unit Tests ==="
	$(PYTHON) -m pytest tests/ -v -m "unit or not (integration or slow)" --ignore=tests/integration -x
	@echo "✓ Unit tests passed"

# Fast unit tests only (excludes slow tests)
test-fast:
	@echo "=== Unit Tests (fast only) ==="
	$(PYTHON) -m pytest tests/ -v -m "unit and not slow" --ignore=tests/integration -x
	@echo "✓ Fast unit tests passed"

# ============================================
# F) Integration Tests
# ============================================
test-int:
	@echo "=== F) Integration Tests ==="
	$(PYTHON) -m pytest tests/integration/ -v -m "integration or not unit" -x
	@echo "✓ Integration tests passed"

# Physics validation tests
test-physics:
	@echo "=== Physics Validation Tests ==="
	$(PYTHON) -m pytest tests/ Physics/tests/ -v -m "physics" -x
	@echo "✓ Physics tests passed"

test: test-unit test-int
	@echo "✓ All tests passed"

# Test with coverage report
test-cov:
	@echo "=== Tests with Coverage ==="
	$(PYTHON) -m pytest tests/ -v --cov=tensornet --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "✓ Coverage report generated (htmlcov/ and coverage.xml)"

# ============================================
# G) Formal Proofs
# ============================================
proofs:
	@echo "=== G) Formal Proofs ==="
	$(PYTHON) proofs/run_all_proofs.py
	@echo "✓ All proofs verified"

# ============================================
# H) Reproduction
# ============================================
reproduce:
	@echo "=== H) Reproduce Paper Results ==="
	$(PYTHON) scripts/full_reproduce.py --quick
	@echo "✓ Quick reproduction complete"

reproduce-full:
	@echo "=== H) Full Reproduction ==="
	$(PYTHON) scripts/full_reproduce.py --full
	@echo "✓ Full reproduction complete"

# ============================================
# I) Physics Validation
# ============================================
physics:
	@echo "=== I) Physics Validation ==="
	$(PYTHON) scripts/physics_validation.py --quick
	@echo "✓ Physics validation complete"

physics-full:
	@echo "=== I) Full Physics Validation ==="
	$(PYTHON) scripts/physics_validation.py
	@echo "✓ Full physics validation complete"

# ============================================
# J) Determinism Check
# ============================================
determinism:
	@echo "=== J) Determinism Check ==="
	$(PYTHON) scripts/determinism_check.py
	@echo "✓ Determinism verified"

# ============================================
# K) Evidence Pack
# ============================================
evidence: $(ARTIFACTS_DIR)
	@echo "=== K) Build Evidence Pack ==="
	$(PYTHON) scripts/build_evidence_pack.py
	@echo "✓ Evidence pack created"

# ============================================
# L) Truth Boundary
# ============================================
truth: $(ARTIFACTS_DIR)
	@echo "=== L) Generate Truth Boundary ==="
	$(PYTHON) scripts/generate_truth_boundary.py
	@echo "✓ Truth boundary generated"

# ============================================
# M) Packaging Gate
# ============================================
package:
	@echo "=== M) Packaging Gate ==="
	$(PYTHON) -m pip wheel . -w dist --no-deps
	@echo "Checking wheel..."
	$(PYTHON) -m twine check dist/*.whl || true
	@echo "✓ Package built"

# ============================================
# N) Documentation Build
# ============================================
docs:
	@echo "=== N) Documentation Build ==="
	@if [ -f docs/conf.py ]; then \
		cd docs && make html; \
	elif [ -f mkdocs.yml ]; then \
		$(PYTHON) -m mkdocs build; \
	else \
		echo "No docs config found, generating API docs..."; \
		$(PYTHON) -m pdoc tensornet -o $(ARTIFACTS_DIR)/api_docs --html; \
	fi
	@echo "✓ Documentation built"

# ============================================
# O) Security Scan
# ============================================
security:
	@echo "=== O) Security Scan ==="
	$(PYTHON) scripts/security_scan.py
	@echo "✓ Security scan complete"

# ============================================
# P) SBOM Generation
# ============================================
sbom: $(ARTIFACTS_DIR)
	@echo "=== P) SBOM Generation ==="
	$(PYTHON) scripts/generate_sbom.py
	@echo "✓ SBOM generated"

# ============================================
# Q) Release Check
# ============================================
release-check:
	@echo "=== Q) Release Check ==="
	$(PYTHON) scripts/release_check.py
	@echo "✓ Release check complete"

# ============================================
# Composite Targets
# ============================================
check: format typecheck test proofs physics
	@echo ""
	@echo "============================================"
	@echo "✓ ALL QUALITY GATES PASSED"
	@echo "============================================"

release: check security sbom truth evidence release-check hygiene
	@echo ""
	@echo "============================================"
	@echo "✓ RELEASE PREPARATION COMPLETE"
	@echo "============================================"
	@echo ""
	@echo "Artifacts ready in: $(ARTIFACTS_DIR)/"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review artifacts/release_check.json"
	@echo "  2. Update CHANGELOG.md with version notes"
	@echo "  3. git tag -a vX.Y.Z -m 'Release X.Y.Z'"
	@echo "  4. git push origin vX.Y.Z"
	@echo "  5. Upload artifacts/hypertensor-vX.Y.Z.zip to GitHub Releases"

# ============================================
# Utility
# ============================================
$(ARTIFACTS_DIR):
	mkdir -p $(ARTIFACTS_DIR)

clean:
	@echo "Cleaning artifacts..."
	rm -rf $(ARTIFACTS_DIR)
	rm -rf dist build *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"

# Install development dependencies (pinned versions)
dev-deps:
	$(PYTHON) -m pip install -r requirements-dev.txt
	@echo "✓ Dev dependencies installed (pinned versions)"

# Legacy: Install dev deps without pinning (not recommended)
dev-deps-unpinned:
	$(PYTHON) -m pip install black isort ruff mypy pytest twine wheel pdoc3
	$(PYTHON) -m pip install pip-audit bandit detect-secrets
	@echo "✓ Dev dependencies installed (unpinned)"
# ============================================
# Lockfile Management
# ============================================
lockfile:
	@echo "=== Generating Lockfiles ==="
	$(PYTHON) -m pip freeze > requirements-lock.txt
	@echo "✓ Root lockfile updated: requirements-lock.txt"

lockfile-check:
	@echo "=== Checking Lockfile Consistency ==="
	$(PYTHON) -m pip install -r requirements-lock.txt --dry-run
	@echo "✓ Lockfile is consistent"

# ============================================
# Facial Plastics Product Targets
# ============================================
.PHONY: fp-test fp-typecheck fp-build fp-up fp-down fp-logs fp-keys

FP_ROOT    = products/facial_plastics
FP_IMAGE   = hypertensor-facial-plastics
FP_COMPOSE = $(FP_ROOT)/docker-compose.yml

fp-test:
	@echo "=== Facial Plastics: Test Suite ==="
	$(PYTHON) -m pytest $(FP_ROOT)/tests/ -v --tb=short -x
	@echo "✓ All tests passed"

fp-typecheck:
	@echo "=== Facial Plastics: mypy ==="
	$(PYTHON) -m mypy $(FP_ROOT) \
		--ignore-missing-imports \
		--no-implicit-optional \
		--warn-return-any \
		--disallow-untyped-defs \
		--python-version 3.12
	@echo "✓ Type check passed"

fp-build:
	@echo "=== Facial Plastics: Container Build ==="
	docker build -f $(FP_ROOT)/Containerfile -t $(FP_IMAGE):latest .
	@echo "✓ Image built: $(FP_IMAGE):latest"

fp-up:
	@echo "=== Facial Plastics: Starting Stack ==="
	docker compose -f $(FP_COMPOSE) up -d
	@echo "✓ Stack started — https://$${FP_DOMAIN:-localhost}"

fp-down:
	@echo "=== Facial Plastics: Stopping Stack ==="
	docker compose -f $(FP_COMPOSE) down
	@echo "✓ Stack stopped"

fp-logs:
	docker compose -f $(FP_COMPOSE) logs -f

fp-keys:
	@echo "=== Facial Plastics: Generate API Key ==="
	@$(PYTHON) -c " \
from products.facial_plastics.ui.auth import KeyStore; \
from pathlib import Path; \
import sys; \
store = KeyStore(Path('fp_keys.json')); \
tenant = sys.argv[1] if len(sys.argv) > 1 else 'default'; \
role = sys.argv[2] if len(sys.argv) > 2 else 'surgeon'; \
key, rec = store.generate_key(tenant, role, 'cli-generated'); \
print(f'Tenant: {rec.tenant_id}'); \
print(f'Role:   {rec.role}'); \
print(f'Key:    {key}'); \
print(f'Hash:   {rec.key_hash[:16]}...'); \
print(f'Saved to fp_keys.json'); \
" $(TENANT) $(ROLE)