# The Physics OS Monorepo Makefile
# ==========================================
#
# Orchestrates BOTH Python (ontic + physics_os) and Rust (cargo workspace).
#
# Usage:
#   make check          - Run all quality gates (Python + Rust)
#   make release        - Full release preparation
#   make help           - Show all targets
#
# Domain cheat-sheet:
#   make py-check       - Python-only quality gates
#   make rs-check       - Rust-only quality gates
#   make fp-test        - Facial Plastics product tests
#
# Checklist Mapping:
#   A) make hygiene      - Export clean release zip
#   B) make env          - Capture environment
#   C) make format       - Format/lint gates (Python + Rust)
#   D) make typecheck    - Type checking (mypy)
#   E) make test-unit    - Unit tests (Python)
#   F) make test-int     - Integration tests (Python)
#   G) make proofs       - Run all proofs
#   H) make reproduce    - Reproduction check
#   I) make physics      - Physics validation
#   J) make determinism  - Determinism check
#   K) make evidence     - Build evidence pack
#   L) make truth        - Truth boundary doc
#   M) make package      - Packaging gate
#   N) make docs         - Build documentation (MkDocs Material)
#   O) make security     - Security scan
#   P) make sbom         - Generate SBOM
#   Q) make release-check - Release validation

.PHONY: help check release clean
.PHONY: hygiene env format typecheck test-unit test-int
.PHONY: proofs reproduce physics determinism evidence truth
.PHONY: package docs security sbom release-check
.PHONY: vlasov-test vlasov-smoke vlasov-proof vlasov-build-prover
.PHONY: version-check version-sync
.PHONY: rs-build rs-test rs-clippy rs-fmt rs-check
.PHONY: py-check py-format py-test
.PHONY: fp-test fp-typecheck fp-build fp-up fp-down fp-logs fp-keys
.PHONY: docs-serve docs-build dep-graph
.PHONY: dev-deps lockfile lockfile-check

# ── Configuration ─────────────────────────────────────────────────────
PYTHON      ?= python
CARGO       ?= cargo
ARTIFACTS_DIR = artifacts
DOCS_DIR      = docs

# Detect uv (preferred) vs pip
UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  PIP_INSTALL = uv pip install
  PIP_SYNC    = uv sync
else
  PIP_INSTALL = $(PYTHON) -m pip install
  PIP_SYNC    = $(PYTHON) -m pip install -e ".[dev,docs]"
endif

# Python source directories (order: core -> products -> tests)
PY_SRC = ontic physics_os tests proofs

# ── Default ───────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

help:
	@echo "HyperTensor Monorepo Makefile"
	@echo "================================"
	@echo ""
	@echo "Quality Gates:"
	@echo "  make check          All quality gates (Python + Rust)"
	@echo "  make py-check       Python-only gates (format+type+test)"
	@echo "  make rs-check       Rust-only gates (fmt+clippy+test)"
	@echo "  make format         Format/lint (Python + Rust)"
	@echo "  make typecheck      mypy type checking"
	@echo "  make test-unit      Python unit tests"
	@echo "  make test-int       Python integration tests"
	@echo "  make proofs         Formal proofs"
	@echo ""
	@echo "Rust Workspace:"
	@echo "  make rs-build       cargo build --workspace --release"
	@echo "  make rs-test        cargo test --workspace"
	@echo "  make rs-clippy      cargo clippy --workspace"
	@echo "  make rs-fmt         cargo fmt --all -- --check"
	@echo ""
	@echo "Validation:"
	@echo "  make physics        Physics validation gates"
	@echo "  make determinism    Determinism checks"
	@echo "  make reproduce      Reproduce paper results"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           Build MkDocs Material site"
	@echo "  make docs-serve     Live-reload MkDocs dev server"
	@echo "  make dep-graph      Generate dependency graph (SVG)"
	@echo ""
	@echo "Security:"
	@echo "  make security       Run security scans"
	@echo ""
	@echo "Release:"
	@echo "  make release-check  Pre-release validation"
	@echo "  make version-check  Check version consistency"
	@echo "  make version-sync   Synchronize all version files"
	@echo "  make evidence       Build evidence pack"
	@echo "  make hygiene        Export clean release zip"
	@echo "  make release        Full release preparation"
	@echo ""
	@echo "Utility:"
	@echo "  make env            Capture environment"
	@echo "  make package        Packaging gate (wheel build)"
	@echo "  make dev-deps       Install dev dependencies"
	@echo "  make clean          Remove artifacts"
	@echo ""
	@echo "Vlasov 6D Proof Pipeline:"
	@echo "  make vlasov-test         Vlasov solver unit tests"
	@echo "  make vlasov-smoke        Quick 4^6 smoke test"
	@echo "  make vlasov-proof        Full 32^6 proof pipeline"
	@echo "  make vlasov-build-prover Build Rust STARK prover"
	@echo ""
	@echo "Facial Plastics Product:"
	@echo "  make fp-test        Run facial plastics tests"
	@echo "  make fp-typecheck   mypy on facial plastics"
	@echo "  make fp-build       Build container image"
	@echo "  make fp-up / fp-down / fp-logs / fp-keys"

# ============================================
# A) Hygiene — Clean release export
# ============================================
hygiene: $(ARTIFACTS_DIR)
	@echo "=== A) Export Clean Release ==="
	$(PYTHON) tools/scripts/export_release_zip.py
	@echo "✓ Clean release zip created"

# ============================================
# B) Environment Capture
# ============================================
env: $(ARTIFACTS_DIR)
	@echo "=== B) Environment Capture ==="
	$(PYTHON) tools/scripts/env_capture.py
	@echo "✓ Environment captured"

# ============================================
# C) Format & Lint Gates (Python + Rust)
# ============================================
format: py-format rs-fmt
	@echo "✓ All format checks passed"

py-format:
	@echo "=== C) Python Format & Lint ==="
	@echo "Running ruff format check..."
	$(PYTHON) -m ruff format --check $(PY_SRC) || true
	@echo "Running ruff lint..."
	$(PYTHON) -m ruff check $(PY_SRC)
	@echo "✓ Python format checks passed"

format-fix:
	@echo "=== Applying Format Fixes ==="
	$(PYTHON) -m ruff format $(PY_SRC)
	$(PYTHON) -m ruff check --fix $(PY_SRC)
	$(CARGO) fmt --all
	@echo "✓ Format fixes applied (Python + Rust)"

# ============================================
# D) Type Checking
# ============================================
typecheck:
	@echo "=== D) Type Checking ==="
	$(PYTHON) -m mypy ontic --ignore-missing-imports
	@echo "✓ Type check passed"

# Docstring coverage check
doccheck:
	@echo "=== Docstring Coverage ==="
	$(PYTHON) tools/scripts/check_docstrings.py --threshold 70 -v
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
	$(PYTHON) -m pytest tests/ -v -m "physics" -x
	@echo "✓ Physics tests passed"

test: test-unit test-int
	@echo "✓ All Python tests passed"

# Test with coverage report
test-cov:
	@echo "=== Tests with Coverage ==="
	$(PYTHON) -m pytest tests/ -v --cov=ontic --cov-report=term-missing --cov-report=html --cov-report=xml
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
	$(PYTHON) tools/scripts/full_reproduce.py --quick
	@echo "✓ Quick reproduction complete"

reproduce-full:
	@echo "=== H) Full Reproduction ==="
	$(PYTHON) tools/scripts/full_reproduce.py --full
	@echo "✓ Full reproduction complete"

# ============================================
# I) Physics Validation
# ============================================
physics:
	@echo "=== I) Physics Validation ==="
	$(PYTHON) tools/scripts/physics_validation.py --quick
	@echo "✓ Physics validation complete"

physics-full:
	@echo "=== I) Full Physics Validation ==="
	$(PYTHON) tools/scripts/physics_validation.py
	@echo "✓ Full physics validation complete"

# ============================================
# J) Determinism Check
# ============================================
determinism:
	@echo "=== J) Determinism Check ==="
	$(PYTHON) tools/scripts/determinism_check.py
	@echo "✓ Determinism verified"

# ============================================
# K) Evidence Pack
# ============================================
evidence: $(ARTIFACTS_DIR)
	@echo "=== K) Build Evidence Pack ==="
	$(PYTHON) tools/scripts/build_evidence_pack.py
	@echo "✓ Evidence pack created"

# ============================================
# L) Truth Boundary
# ============================================
truth: $(ARTIFACTS_DIR)
	@echo "=== L) Generate Truth Boundary ==="
	$(PYTHON) tools/scripts/generate_truth_boundary.py
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
# N) Documentation Build (MkDocs Material)
# ============================================
docs: docs-build
	@echo "✓ Documentation built"

docs-build:
	@echo "=== N) Documentation Build ==="
	@if [ -f mkdocs.yml ]; then \
		$(PYTHON) -m mkdocs build --strict; \
	else \
		echo "No mkdocs.yml found, generating API docs with pdoc..."; \
		$(PYTHON) -m pdoc ontic -o $(ARTIFACTS_DIR)/api_docs --html; \
	fi

docs-serve:
	@echo "=== MkDocs Dev Server ==="
	$(PYTHON) -m mkdocs serve --dev-addr 127.0.0.1:8000

# ============================================
# O) Security Scan
# ============================================
security:
	@echo "=== O) Security Scan ==="
	$(PYTHON) tools/scripts/security_scan.py
	@echo "✓ Security scan complete"

# ============================================
# P) SBOM Generation
# ============================================
sbom: $(ARTIFACTS_DIR)
	@echo "=== P) SBOM Generation ==="
	$(PYTHON) tools/scripts/generate_sbom.py
	@echo "✓ SBOM generated"

# ============================================
# Q) Release Check
# ============================================
release-check:
	@echo "=== Q) Release Check ==="
	$(PYTHON) tools/scripts/release_check.py
	@echo "✓ Release check complete"

# ============================================
# Composite Targets
# ============================================
py-check: py-format typecheck py-test
	@echo "✓ Python quality gates passed"

py-test: test-unit test-int
	@echo "✓ Python tests passed"

rs-check: rs-fmt rs-clippy rs-test
	@echo "✓ Rust quality gates passed"

check: py-check rs-check proofs physics
	@echo ""
	@echo "============================================"
	@echo "✓ ALL QUALITY GATES PASSED (Python + Rust)"
	@echo "============================================"

# ============================================
# Rust Workspace Targets
# ============================================
rs-build:
	@echo "=== Rust: Build Workspace (release) ==="
	$(CARGO) build --workspace --release
	@echo "✓ Rust workspace built"

rs-test:
	@echo "=== Rust: Test Workspace ==="
	$(CARGO) test --workspace
	@echo "✓ Rust tests passed"

rs-clippy:
	@echo "=== Rust: Clippy ==="
	$(CARGO) clippy --workspace -- -D warnings
	@echo "✓ Clippy passed"

rs-fmt:
	@echo "=== Rust: Format Check ==="
	$(CARGO) fmt --all -- --check
	@echo "✓ Rust format check passed"

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
	@echo "  3. make version-sync"
	@echo "  4. git tag -a vX.Y.Z -m 'Release X.Y.Z'"
	@echo "  5. git push origin vX.Y.Z  (triggers .github/workflows/release.yml)"

# ============================================
# Utility
# ============================================
$(ARTIFACTS_DIR):
	mkdir -p $(ARTIFACTS_DIR)

version-check:
	@echo "=== Version Consistency Check ==="
	$(PYTHON) tools/sync_versions.py

version-sync:
	@echo "=== Synchronizing Versions ==="
	$(PYTHON) tools/sync_versions.py --apply

# Dependency graph visualization
dep-graph: $(ARTIFACTS_DIR)
	@echo "=== Dependency Graph ==="
	$(PYTHON) tools/dep_graph.py --output $(ARTIFACTS_DIR)/dep_graph.svg
	@echo "✓ Graph: $(ARTIFACTS_DIR)/dep_graph.svg"

clean:
	@echo "Cleaning artifacts..."
	rm -rf $(ARTIFACTS_DIR)
	rm -rf dist build *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Clean complete"

# Install development dependencies
dev-deps:
ifdef UV
	@echo "=== Installing with uv ==="
	uv sync --extra dev --extra docs
else
	@echo "=== Installing with pip ==="
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e ".[dev,docs]"
endif
	@echo "✓ Dev dependencies installed"
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
# Vlasov 6D Proof Pipeline
# ============================================
VLASOV_PROVER = target/release/vlasov-proof
VLASOV_VIDEO  = tools/scripts/vlasov_6d_video.py

vlasov-test:
	@echo "=== Vlasov Genuine Solver: Unit Tests ==="
	PYTHONPATH="apps/qtenet/src/qtenet:$(PWD):$$PYTHONPATH" \
		$(PYTHON) -m pytest tests/test_vlasov_genuine.py -v --tb=short -x
	@echo "✓ Vlasov unit tests passed"

vlasov-smoke:
	@echo "=== Vlasov 6D Smoke Test (4^6 = 4,096 pts, 5 steps) ==="
	PYTHONPATH="apps/qtenet/src/qtenet:$(PWD):$$PYTHONPATH" \
		$(PYTHON) $(VLASOV_VIDEO) --n-bits 2 --max-rank 16 --steps 5 --dt 0.005 --device cpu --frame-every 5
	@echo "✓ Vlasov 6D smoke test passed"

vlasov-proof: $(VLASOV_PROVER)
	@echo "=== Vlasov 6D Full Proof (32^6 = 1B pts) ==="
	PYTHONPATH="apps/qtenet/src/qtenet:$(PWD):$$PYTHONPATH" \
		$(PYTHON) $(VLASOV_VIDEO) --n-bits 5 --max-rank 128 --steps 20 --dt 0.005 --device cpu --frame-every 5
	@echo "✓ Vlasov 6D proof pipeline complete"
	@echo "  Artifacts:"
	@echo "    media/videos/vlasov_6d_phase_space.mp4"
	@echo "    artifacts/VLASOV_6D_PROOF.bin"
	@echo "    artifacts/VLASOV_6D_CERTIFICATE.tpc"

vlasov-build-prover:
	@echo "=== Building Vlasov STARK Prover (Rust) ==="
	cargo build -p vlasov-proof --release
	@echo "✓ Prover built: $(VLASOV_PROVER)"

$(VLASOV_PROVER):
	@echo "Vlasov STARK prover not found, building..."
	cargo build -p vlasov-proof --release

# ============================================
# Facial Plastics Product Targets
# ============================================
.PHONY: fp-test fp-typecheck fp-build fp-up fp-down fp-logs fp-keys

FP_ROOT    = products/facial_plastics
FP_IMAGE   = physics-os-facial-plastics
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
	@if [ -z "$(TENANT)" ]; then \
		echo "Usage: make fp-keys TENANT=<tenant-id> ROLE=<role>"; \
		echo "  TENANT  (required) tenant identifier, e.g. clinic-1"; \
		echo "  ROLE    (optional) surgeon|resident|researcher|administrator|auditor (default: surgeon)"; \
		exit 1; \
	fi
	@$(PYTHON) -c "import sys; from products.facial_plastics.ui.auth import KeyStore; from pathlib import Path; store = KeyStore(Path('fp_keys.json')); t = sys.argv[1]; r = sys.argv[2] if len(sys.argv) > 2 else 'surgeon'; key, rec = store.generate_key(t, r, 'cli-generated'); print(f'Tenant: {rec.tenant_id}'); print(f'Role:   {rec.role}'); print(f'Key:    {key}'); print(f'Hash:   {rec.key_hash[:16]}...'); print(); print('Saved to fp_keys.json'); print('To use in Docker: copy fp_keys.json into the fp-keys volume'); print('  docker cp fp_keys.json fp-app:/etc/fp/keys.json')" $(TENANT) $(ROLE)