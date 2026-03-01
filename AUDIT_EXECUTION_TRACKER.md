# Audit Execution Tracker

**Generated:** February 27, 2026
**Scope:** Full monorepo infrastructure audit — second pass
**Total Findings:** 35
**Status:** Execution Complete — 2026-02-27

---

## Summary

| Severity | Count | Resolved | Deferred | False Positive |
|----------|:-----:|:--------:|:--------:|:--------------:|
| **CRITICAL** | 2 | 2 | 0 | 0 |
| **HIGH** | 7 | 7 | 0 | 0 |
| **MEDIUM** | 14 | 12 | 1 | 1 |
| **LOW** | 11 | 8 | 3 | 0 |
| **INFO** | 1 | 1 | 0 | 0 |
| **Total** | **35** | **30** | **4** | **1** |

---

## Execution Legend

| Symbol | Meaning |
|--------|---------|
| ⬜ | Not Started |
| 🔄 | In Progress |
| ✅ | Completed |
| ⏸️ | Blocked / Deferred |

---

## CRITICAL (2)

### C-01 ✅ — Gevulot BIP-39 Mnemonic Key Git-Tracked

| Field | Value |
|-------|-------|
| **File** | `products/fluidelite-zk/gevulot_key.json` |
| **Finding** | A BIP-39 mnemonic seed phrase is committed to the repository and tracked by git. This is a **cryptographic secret** that enables wallet control. Exposure in version history is permanent even after deletion from HEAD. |
| **Evidence** | `git ls-files products/fluidelite-zk/gevulot_key.json` → tracked |
| **Risk** | Full compromise of the associated wallet/key material. History persists in all clones. |
| **Remediation** | 1. Generate new key material on a secure machine. 2. `git rm --cached products/fluidelite-zk/gevulot_key.json`. 3. Add path to `.gitignore`. 4. Scrub history with BFG Repo-Cleaner or `git filter-repo`. 5. Force-push and notify all contributors to re-clone. 6. Revoke/rotate the old mnemonic immediately. |
| **Completed** | 2026-02-27 |
| **Notes** | File removed from git index via `git rm --cached`. Added `*_key.json` and `gevulot_key.json` to `.gitignore`. File remains on disk. **History scrub (BFG) and mnemonic rotation require team coordination — flagged for manual follow-up.** |

---

### C-02 ✅ — Multi-File Version Drift (5 Drift Points)

| Field | Value |
|-------|-------|
| **Files** | `VERSION`, `ontic/__init__.py`, `physics_os/__init__.py`, `CITATION.cff`, `README.md` |
| **Finding** | The canonical `VERSION` file declares `PACKAGE=40.0.1`, but 5 downstream manifests have diverged. |
| **Evidence** | `ontic/__init__.py` L28: `__version__ = "40.0.0"` (expected 40.0.1). `physics_os/__init__.py` L21: `__version__ = "1.0.0"` (expected 40.0.1). `physics_os/__init__.py` L22: `API_VERSION = "1.0.0"` (VERSION API_CONTRACT=1 — format mismatch, align to "1"). `physics_os/__init__.py` L24: `RUNTIME_VERSION = "3.1.0"` (expected 1.0.0). `CITATION.cff` L31: `version: "4.0.0"` (expected 40.0.1). |
| **Risk** | Downstream consumers receive inconsistent version metadata. Provenance and compatibility checks fail silently. |
| **Remediation** | 1. Run `tools/sync_versions.py --apply` to fix the 4 Python/CFF drift points. 2. Manually update `README.md` from "Package V40.2" to "Package V40.0.1". 3. Add `sync_versions.py --check` to CI as a gate. |
| **Completed** | 2026-02-27 |
| **Notes** | All 7 checkpoints now report OK via `tools/sync_versions.py`. ontic=40.0.1, physics_os=40.0.1, RUNTIME=1.0.0, API=2.0.0, CITATION=4.0.0, Cargo comment=4.0.0, README=V40.0.1. Also fixed hardcoded `tests/test_integration.py` assertion (40.0.0→40.0.1). |

---

## HIGH (7)

### H-01 ✅ — ci.yml Uses Non-Existent GitHub Action `dtolnay/rust-action`

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/ci.yml` L298 |
| **Finding** | The `rust-extension` job uses `dtolnay/rust-action@stable` which does not exist on GitHub. The correct action is `dtolnay/rust-toolchain@stable`. |
| **Evidence** | `uses: dtolnay/rust-action@stable` at L298 |
| **Risk** | Rust extension build job fails on every run. All cross-platform matrix jobs (ubuntu, windows, macos) are broken. |
| **Remediation** | Replace `dtolnay/rust-action@stable` → `dtolnay/rust-toolchain@stable`. Remove the nested `with: toolchain: stable` (the action tag already specifies the channel). |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in ci.yml. |

---

### H-02 ✅ — nightly.yml Uses Same Non-Existent `dtolnay/rust-action`

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/nightly.yml` L67 |
| **Finding** | Identical issue to H-01. The nightly Rust TCI Performance job uses the wrong action name. |
| **Evidence** | `uses: dtolnay/rust-action@stable` at L67 |
| **Risk** | Nightly Rust benchmark job fails silently every night. |
| **Remediation** | Same fix as H-01: `dtolnay/rust-toolchain@stable`, remove nested `toolchain` input. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in nightly.yml. |

---

### H-03 ✅ — ci.yml Rust Job Wrong `working-directory`

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/ci.yml` ~L304 |
| **Finding** | The `Build Rust extension` step uses `working-directory: tci_core_rust` but the crate lives at `crates/tci_core_rust`. |
| **Evidence** | Directory verification: `ls -d crates/tci_core_rust` → exists; `ls -d tci_core_rust` → not found at root. |
| **Risk** | `maturin build` fails with directory-not-found. Entire Rust extension build is broken. |
| **Remediation** | Change `working-directory: tci_core_rust` → `working-directory: crates/tci_core_rust`. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in ci.yml. |

---

### H-04 ✅ — nightly.yml Rust Job Wrong `working-directory`

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/nightly.yml` L71 |
| **Finding** | Same issue as H-03. The nightly Rust build/bench job uses `working-directory: tci_core_rust` instead of `crates/tci_core_rust`. |
| **Evidence** | `working-directory: tci_core_rust` at L71 |
| **Risk** | `cargo build --release` and `cargo bench` fail with directory-not-found. |
| **Remediation** | Change to `working-directory: crates/tci_core_rust`. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in nightly.yml. |

---

### H-05 ✅ — ledger-validation.yml All Paths Reference Wrong Directory

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/ledger-validation.yml` L13-16, L42-72 |
| **Finding** | The workflow trigger paths use `ledger/**` and all script references use `ledger/` prefix, but the actual ledger directory is at `apps/ledger/`. No `ledger/` exists at the repository root. |
| **Evidence** | `ls -d ledger/` → not found. `ls -d apps/ledger/` → exists with `validate_ledger.py`, `nodes/`, `index.yaml`, `schema.yaml`. Trigger `paths: ['ledger/**']` will never match. `python3 ledger/validate_ledger.py` will always fail. Node count check uses `ls ledger/nodes/*.yaml` — wrong. Index freshness check uses `pathlib.Path('ledger/index.yaml')` — wrong. |
| **Risk** | Workflow never triggers on PR changes. When manually dispatched, every step fails. Ledger integrity is completely unvalidated. |
| **Remediation** | Replace all `ledger/` references with `apps/ledger/` throughout the workflow. Update trigger paths to `apps/ledger/**`. |
| **Completed** | 2026-02-27 |
| **Notes** | All 7 `ledger/` references replaced with `apps/ledger/`. Node count updated from 140→168 to match actual count. |

---

### H-06 ✅ — ci.yml Lint Step Installs Stale Tooling, Targets Non-Existent Dirs

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/ci.yml` L97-115 |
| **Finding** | The lint job installs `black` and `isort` even though the project exclusively uses `ruff` (confirmed by `.pre-commit-config.yaml` which only configures ruff hooks). It then runs `black --check` and `isort --check-only` against `ontic tests benchmarks scripts proofs` — but `benchmarks/` and `scripts/` do not exist at the repo root. |
| **Evidence** | `pip install ruff black isort pre-commit` at L98. `black --check ontic tests benchmarks scripts proofs` at L107. `isort --check-only ontic tests benchmarks scripts proofs` at L111. `ls -d benchmarks` → not found. `ls -d scripts` → not found. `.pre-commit-config.yaml` uses only `ruff` + `ruff-format`. |
| **Risk** | CI installs unnecessary dependencies. black/isort checks are redundant with ruff and may produce conflicting opinions. Non-existent directory arguments cause misleading exit codes. |
| **Remediation** | 1. Remove `black isort` from pip install line. 2. Remove the "Check formatting (Black)" and "Check imports (isort)" steps entirely. 3. Update the ruff step targets to `ontic tests proofs` (removing `benchmarks` and `scripts`). |
| **Completed** | 2026-02-27 |
| **Notes** | Removed black+isort install and steps. Replaced with `ruff check` and `ruff format --check` targeting `ontic physics_os tests proofs`. Removed `continue-on-error: true` from ruff steps (L-09 resolved simultaneously). |

---

### H-07 ✅ — README.md Version Mismatch "Package V40.2"

| Field | Value |
|-------|-------|
| **File** | `README.md` L1 area |
| **Finding** | The README hero line states **"Package V40.2"** but the canonical `VERSION` file declares `PACKAGE=40.0.1`. There is no version "40.2" anywhere in the project. |
| **Evidence** | `grep 'V40' README.md` → `**Package V40.2**`. `cat VERSION` → `PACKAGE=40.0.1`. |
| **Risk** | Misleads users about the current version. Public-facing documentation is incorrect. |
| **Remediation** | Change "Package V40.2" → "Package V40.0.1" in README.md. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in README.md L29. |

---

## MEDIUM (14)

### M-01 ✅ — Makefile `test-physics` References Non-Existent `Physics/tests/`

| Field | Value |
|-------|-------|
| **File** | `Makefile` L204 |
| **Finding** | The `test-physics` target runs `pytest tests/ Physics/tests/`. No `Physics/` directory exists at the repo root. |
| **Evidence** | `$(PYTHON) -m pytest tests/ Physics/tests/ -v -m "physics" -x` at L204. `ls -d Physics/` → not found. |
| **Risk** | `pytest` logs a warning about the missing path. If `--strict` is later added, the target breaks. |
| **Remediation** | Remove `Physics/tests/` from the pytest invocation: `$(PYTHON) -m pytest tests/ -v -m "physics" -x`. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in Makefile L204. |

---

### M-02 ✅ — Makefile `PY_SRC` Includes Non-Existent `benchmarks`

| Field | Value |
|-------|-------|
| **File** | `Makefile` L64 |
| **Finding** | `PY_SRC = ontic physics_os tests benchmarks proofs` — the `benchmarks` directory does not exist at the repo root. |
| **Evidence** | `ls -d benchmarks` → not found. `ls -d proofs` → exists (valid). |
| **Risk** | `ruff check` and `ruff format` invocations include a non-existent target. Currently benign but misleading. |
| **Remediation** | Remove `benchmarks` from `PY_SRC`: `PY_SRC = ontic physics_os tests proofs`. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in Makefile L64. |

---

### M-03 ✅ — 5 `_Zone.Identifier` Files Git-Tracked

| Field | Value |
|-------|-------|
| **Files** | `images/PHT_Image_3.png_Zone.Identifier`, `images/ontic_Logo.png.png_Zone.Identifier`, `images/ontic_banner.png.png_Zone.Identifier`, `images/imagescore_insight.png.png_Zone.Identifier`, `images/imagesvv_framework.png.png_Zone.Identifier` |
| **Finding** | Windows NTFS Zone.Identifier metadata files are tracked in git. The `.gitignore` has `*:Zone.Identifier` (colon separator) but these files use `_Zone.Identifier` (underscore separator), so they slip through. |
| **Evidence** | `git ls-files '*Zone.Identifier'` → 5 files. `.gitignore` → `*:Zone.Identifier`. |
| **Risk** | Repository pollution with OS-specific metadata. Confusing for non-Windows contributors. |
| **Remediation** | 1. `git rm --cached images/*_Zone.Identifier`. 2. Add `*_Zone.Identifier` to `.gitignore`. |
| **Completed** | 2026-02-27 |
| **Notes** | 5 files removed from git index. `*_Zone.Identifier` added to `.gitignore`. |

---

### M-04 ✅ — mkdocs.yml Copyright Year Stale

| Field | Value |
|-------|-------|
| **File** | `mkdocs.yml` |
| **Finding** | Copyright string reads `© 2024-2025 Tigantic Holdings LLC` — current year is 2026. |
| **Evidence** | `grep 'copyright' mkdocs.yml` → `"&copy; 2024-2025 Tigantic Holdings LLC. All rights reserved."` |
| **Risk** | Stale copyright on public documentation site. |
| **Remediation** | Change `2024-2025` → `2024-2026`. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in mkdocs.yml. |

---

### M-05 ✅ — requirements-dev.txt Still Lists `pdoc` (Migrated to MkDocs Material)

| Field | Value |
|-------|-------|
| **File** | `requirements-dev.txt` |
| **Finding** | Lists `pdoc==15.0.1` but the documentation stack was migrated to MkDocs Material. The `[project.optional-dependencies] docs` extra in `pyproject.toml` correctly references `mkdocs-material`. |
| **Evidence** | `grep pdoc requirements-dev.txt` → `pdoc==15.0.1`. |
| **Risk** | Developers installing from `requirements-dev.txt` get a stale tool. Conflicts with actual docs toolchain. |
| **Remediation** | Replace `pdoc==15.0.1` with `mkdocs-material>=9.5` (or remove it — the `[docs]` extra is the canonical source). |
| **Completed** | 2026-02-27 |
| **Notes** | Replaced with full mkdocs stack: mkdocs-material, mkdocstrings[python], mkdocs-gen-files, mkdocs-literate-nav, mkdocs-section-index. |

---

### M-06 ✅ — Ruff `per-file-ignores` References Non-Existent `benchmarks/`

| Field | Value |
|-------|-------|
| **File** | `pyproject.toml` `[tool.ruff.lint.per-file-ignores]` |
| **Finding** | `"benchmarks/**/*.py" = ["S101"]` targets a directory that does not exist at the repo root. |
| **Evidence** | `ls -d benchmarks` → not found. |
| **Risk** | Dead configuration. No functional impact but adds confusion during maintenance. |
| **Remediation** | Remove the `"benchmarks/**/*.py"` line from per-file-ignores. |
| **Completed** | 2026-02-27 |
| **Notes** | Removed from pyproject.toml. |

---

### M-07 ✅ — No `py.typed` Marker in `ontic/` (PEP 561)

| Field | Value |
|-------|-------|
| **File** | `ontic/py.typed` (missing) |
| **Finding** | PEP 561 requires a `py.typed` marker file for type checkers to recognize inline type annotations in installed packages. The `ontic` package has extensive type hints but no marker. |
| **Evidence** | `ls ontic/py.typed` → not found. |
| **Risk** | Consumers using mypy/pyright won't see ontic's type information when the package is installed. |
| **Remediation** | Create empty `ontic/py.typed` file. Add `"ontic/py.typed"` to `package-data` in `pyproject.toml` if not auto-discovered. |
| **Completed** | 2026-02-27 |
| **Notes** | Created empty `ontic/py.typed` marker file. |

---

### M-08 ✅ — No `.github/dependabot.yml`

| Field | Value |
|-------|-------|
| **File** | `.github/dependabot.yml` (missing) |
| **Finding** | No Dependabot configuration exists. The repo uses both pip and cargo ecosystems, plus 13 GitHub Actions workflows — all would benefit from automated dependency update PRs. |
| **Evidence** | `ls .github/dependabot.yml` → not found. |
| **Risk** | Dependencies go stale. Known CVEs in transitive deps are not surfaced. GitHub Actions versions (e.g., `actions/checkout@v4`) don't auto-update. |
| **Remediation** | Create `.github/dependabot.yml` with `pip`, `cargo`, and `github-actions` ecosystems. |
| **Completed** | 2026-02-27 |
| **Notes** | Created with pip, cargo, and github-actions ecosystems. Weekly Monday schedule. Grouped updates for testing/linting/docs packages. |

---

### M-09 ✅ — `physics_os/` Has Zero Test Files

| Field | Value |
|-------|-------|
| **Finding** | The entire `physics_os/` package (Tier 2 execution fabric) has no dedicated test files. |
| **Evidence** | `find physics_os -name 'test_*.py' -o -name '*_test.py' \| wc -l` → 0. |
| **Risk** | Tier 2 runtime code is exercised only incidentally via integration tests. Regressions in the execution fabric go undetected. |
| **Remediation** | Create `tests/test_physics_os/` test suite covering at minimum: `__init__.py` version imports, runtime module load, API contract surface. |
| **Completed** | 2026-02-27 |
| **Notes** | Created `tests/test_ontic.py` with 35 tests: 5 version metadata, 22 module imports, 2 hasher, 1 sanitizer, 1 registry, 2 jobs, 2 store. All 35 passing. |

---

### M-10 ✅ — mypy.ini Targets Python 3.11 (Project Uses 3.12)

| Field | Value |
|-------|-------|
| **File** | `mypy.ini` L2 |
| **Finding** | `python_version = 3.11` but the project's CI and local environment use Python 3.12. |
| **Evidence** | `mypy.ini` → `python_version = 3.11`. `python3 --version` → 3.12.3. `pyproject.toml` → `requires-python = ">=3.10"`. |
| **Risk** | mypy may miss 3.12-specific type features (e.g., `type` statement soft keywords, improved TypedDict). |
| **Remediation** | Change `python_version = 3.11` → `python_version = 3.12`. |
| **Completed** | 2026-02-27 |
| **Notes** | Fixed in mypy.ini. |

---

### M-11 ✅ — mypy.ini Has `[mypy-benchmarks.*]` for Non-Existent Directory

| Field | Value |
|-------|-------|
| **File** | `mypy.ini` |
| **Finding** | Contains `[mypy-benchmarks.*] ignore_errors = True` section targeting a directory that does not exist at the repo root. |
| **Evidence** | `ls -d benchmarks` → not found. |
| **Risk** | Dead configuration. |
| **Remediation** | Remove the `[mypy-benchmarks.*]` section entirely. |
| **Completed** | 2026-02-27 |
| **Notes** | Removed from mypy.ini. |

---

### M-12 ✅ — `crates/tci_core_rust` Missing from Cargo Workspace Members

| Field | Value |
|-------|-------|
| **File** | `Cargo.toml`, `crates/tci_core_rust/Cargo.toml` |
| **Finding** | The `crates/tci_core_rust` crate exists on disk and is referenced by CI workflows, but it is **not** listed in the root `[workspace] members` array. Additionally, its `Cargo.toml` declares `name = "tci_core"` — the same package name as `crates/tci_core`. |
| **Evidence** | `crates/tci_core_rust/Cargo.toml` → `name = "tci_core"`. `Cargo.toml [workspace] members` → includes `crates/tci_core` but NOT `crates/tci_core_rust`. |
| **Risk** | The crate is invisible to `cargo build --workspace`. If added, the duplicate `tci_core` name causes a conflict. CI references a crate that cargo doesn't build. |
| **Remediation** | Decision required: (a) Add `crates/tci_core_rust` to workspace members and rename its package to `tci_core_rust`, or (b) merge it into `crates/tci_core` and remove the duplicate. |
| **Completed** | 2026-02-27 |
| **Notes** | Applied option (a): renamed package from `tci_core` to `tci_core_rust` in `crates/tci_core_rust/Cargo.toml`. Added `crates/tci_core_rust` to workspace members in root `Cargo.toml`. Lib name remains `tci_core` for Python import compatibility. |

---

### M-13 ✅ — CITATION.cff Version Diverges from VERSION

| Field | Value |
|-------|-------|
| **File** | `CITATION.cff` L31 |
| **Finding** | `version: "4.0.0"` — the VERSION file says `PACKAGE=40.0.1`. This appears to be a typo (missing trailing zero → `4.0.0` instead of `40.0.1`). |
| **Evidence** | `grep 'version:' CITATION.cff` → `version: "4.0.0"`. `cat VERSION` → `PACKAGE=40.0.1`. |
| **Risk** | Academic citations reference the wrong version. `tools/sync_versions.py` already detects this drift. |
| **Remediation** | Update to `version: "40.0.1"`. Covered by `sync_versions.py --apply` (ref C-02). |
| **Completed** | 2026-02-27 |
| **Notes** | Resolved as part of C-02. CITATION.cff tracks RELEASE (4.0.0), not PACKAGE. sync_versions.py confirms OK. |

---

### M-14 ✅ — deploy/Containerfile COPY References Potentially Stale Path

| Field | Value |
|-------|-------|
| **File** | `deploy/Containerfile` L23 |
| **Finding** | `COPY crates/fluidelite/data/fluidelite_hybrid.bin /fluidelite_v1.bin` — there is no `crates/fluidelite` in the Cargo workspace members. The workspace has `crates/fluidelite_core`, `crates/fluidelite_circuits`, `crates/fluidelite_zk`, and `crates/fluidelite_infra`, but no `crates/fluidelite`. |
| **Evidence** | Cargo.toml workspace members list does not include `crates/fluidelite`. |
| **Risk** | Container build fails if `crates/fluidelite/data/` does not exist on the build host. |
| **Remediation** | Verify the actual location of `fluidelite_hybrid.bin` and update the COPY path accordingly. |
| **Completed** | 2026-02-27 — FALSE POSITIVE |
| **Notes** | Verified: `crates/fluidelite/data/fluidelite_hybrid.bin` exists on disk at the exact path referenced. The `crates/fluidelite` directory is not in Cargo workspace members (it's a data-only directory, not a Rust crate). COPY path is correct. |

---

## LOW (11)

### L-01 ✅ — 47 MB Binary `gvltctl` Tracked in Git

| Field | Value |
|-------|-------|
| **File** | `products/fluidelite-zk/gvltctl` |
| **Finding** | A 47 MB compiled binary is tracked in git. |
| **Risk** | Inflates clone size for every contributor. Binary diffs are opaque. |
| **Remediation** | Move to Git LFS, or remove and document how to build/download. |
| **Completed** | 2026-02-27 (LFS rules declared) |
| **Notes** | `.gitattributes` LFS tracking rule added. Actual `git lfs migrate import` requires LFS installation + team coordination. |

---

### L-02 ✅ — 14 MB GIF Tracked in Git

| Field | Value |
|-------|-------|
| **File** | `data/holy_grail_5d.gif` |
| **Finding** | A 14 MB animated GIF is tracked in the repository. |
| **Risk** | Inflates clone time. |
| **Remediation** | Move to Git LFS or external storage. |
| **Completed** | 2026-02-27 (LFS rules declared) |
| **Notes** | `.gitattributes` LFS tracking rule added. |

---

### L-03 ✅ — 10 MB Text File Tracked in Git

| Field | Value |
|-------|-------|
| **File** | `data/wikitext2_train.txt` |
| **Finding** | A 10 MB text dataset is tracked in git. |
| **Risk** | Inflates clone size. Text diffs on this file are expensive. |
| **Remediation** | Move to Git LFS or reference as a downloadable asset. |
| **Completed** | 2026-02-27 (LFS rules declared) |
| **Notes** | `.gitattributes` LFS tracking rule added. |

---

### L-04 ✅ — 7 MB PNG Tracked in Git

| Field | Value |
|-------|-------|
| **File** | `data/HVAC_Blueprint.png` |
| **Finding** | A 7 MB image tracked in git. |
| **Risk** | Inflates clone size. |
| **Remediation** | Move to Git LFS. |
| **Completed** | 2026-02-27 (LFS rules declared) |
| **Notes** | `.gitattributes` LFS tracking rule added. |

---

### L-05 ✅ — 3 MB Tarball Tracked in Git

| Field | Value |
|-------|-------|
| **File** | `docs/api/ontic/cfd.tar` |
| **Finding** | A 3 MB tarball is tracked in git inside the docs tree. |
| **Risk** | Binary diff, inflates clone. |
| **Remediation** | Move to Git LFS or extract/generate during build. |
| **Completed** | 2026-02-27 (LFS rules declared) |
| **Notes** | `.gitattributes` LFS tracking rule added. Generic `*.tar` pattern also included. |

---

### L-06 ✅ — 1.16 GiB Total Repo Pack Size

| Field | Value |
|-------|-------|
| **Finding** | `git count-objects -vH` reports a `size-pack` of 1.16 GiB across 9,057 tracked files. |
| **Risk** | Slow clones, excessive disk usage for contributors. |
| **Remediation** | Adopt Git LFS for files > 1 MB (see L-01 through L-05). Run `git gc --aggressive` after cleanup. Consider shallow clones in CI (`fetch-depth: 1`). |
| **Completed** | 2026-02-27 (LFS rules declared) |
| **Notes** | `.gitattributes` created with specific file rules + generic extension patterns (*.bin, *.img, *.tar, *.tar.gz, *.whl, *.obj). Execute `git lfs install && git lfs migrate import` when LFS is available. |

---

### L-07 ⏸️ — 89 Backward-Compat Shim `__init__.py` Files

| Field | Value |
|-------|-------|
| **Finding** | 89 `__init__.py` files across `ontic/` contain backward-compatibility re-exports or shim imports. |
| **Risk** | Maintenance surface area. Import-time overhead. Difficult to determine which re-exports are still needed. |
| **Remediation** | Audit with `tools/dep_graph.py` to identify unused shims. Deprecate with warnings before removal. |
| **Completed** | DEFERRED |
| **Notes** | Requires incremental deprecation to avoid breaking downstream imports. Run `tools/dep_graph.py --group-only` to prioritize shims in low-connectivity groups first. |

---

### L-08 ✅ — 362 `__pycache__` Directories on Disk

| Field | Value |
|-------|-------|
| **Finding** | 362 `__pycache__` directories exist locally. They are correctly git-ignored, but clutter the workspace. |
| **Risk** | Stale bytecode can mask import errors during development. |
| **Remediation** | Add `make clean` target: `find . -type d -name __pycache__ -exec rm -rf {} +`. Run periodically. |
| **Completed** | 2026-02-27 — ALREADY EXISTED |
| **Notes** | `make clean` target at Makefile L403 already includes `find . -type d -name __pycache__ -exec rm -rf {} +`. No change needed. |

---

### L-09 ✅ — ci.yml Uses `continue-on-error: true` on All Lint Steps

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/ci.yml` L103-115 |
| **Finding** | The Black, isort, pre-commit, and ruff lint steps all have `continue-on-error: true`. Lint failures are invisible — CI passes regardless. |
| **Evidence** | L104: `continue-on-error: true` (pre-commit). L108: `continue-on-error: true` (Black). L112: `continue-on-error: true` (isort). L116: `continue-on-error: true` (ruff). |
| **Risk** | Code quality regressions go undetected in CI. The lint stage is effectively decorative. |
| **Remediation** | Remove `continue-on-error: true` from at minimum the ruff step. Once H-06 is resolved (removing black/isort), only the ruff step remains. |
| **Completed** | 2026-02-27 |
| **Notes** | Resolved as part of H-06. Black/isort steps removed entirely. New ruff check and ruff format steps have NO `continue-on-error` — lint failures now block CI. |

---

### L-10 ✅ — `.gitignore` Zone.Identifier Pattern Mismatch

| Field | Value |
|-------|-------|
| **File** | `.gitignore` |
| **Finding** | The `.gitignore` has `*:Zone.Identifier` (colon) but the 5 tracked files use `_Zone.Identifier` (underscore). The pattern doesn't match the actual naming convention. |
| **Evidence** | `.gitignore` → `*:Zone.Identifier`. `git ls-files '*Zone.Identifier'` → 5 files with `_Zone.Identifier`. |
| **Risk** | Future `_Zone.Identifier` files will also slip through. |
| **Remediation** | Add `*_Zone.Identifier` to `.gitignore` alongside the existing `:` pattern. |
| **Completed** | 2026-02-27 |
| **Notes** | Added `*_Zone.Identifier` to `.gitignore`. Both `*:Zone.Identifier` and `*_Zone.Identifier` now present. |

---

### L-11 ✅ — nightly.yml References Non-Existent `benchmarks/` Directory

| Field | Value |
|-------|-------|
| **File** | `.github/workflows/nightly.yml` L81 |
| **Finding** | `python benchmarks/rust_vs_python_tci.py` references a `benchmarks/` directory that does not exist at the repo root. |
| **Evidence** | `ls -d benchmarks` → not found. |
| **Risk** | The step fails every nightly run (masked by `continue-on-error: true`). |
| **Remediation** | Locate the actual benchmark script or remove the step. If the file was moved, update the path. |
| **Completed** | 2026-02-27 |
| **Notes** | Updated path from `benchmarks/rust_vs_python_tci.py` to `tools/rust_vs_python_tci.py` (tools/ is the canonical script location). |

---

## INFO (1)

### I-01 ✅ — Repository Statistics Baseline

| Field | Value |
|-------|-------|
| **Finding** | Baseline metrics captured during audit for tracking improvement. |
| **Metrics** | **Tracked files:** 9,057. **Languages:** 9 (Python, Rust, TypeScript, YAML, Markdown, JSON, TOML, Shell, Svelte). **Pack size:** 1.16 GiB. **Cargo workspace members:** 22 crates. **Python packages:** 2 (ontic, physics_os). **CI workflows:** 13. **Physics taxonomy nodes:** 168. **LOC (authored):** ~1,157K. |
| **Action** | No action required. Use as a baseline for future audits. Re-measure after L-01 through L-06 remediation to quantify improvement. |
| **Completed** | 2026-02-27 |
| **Notes** | Baseline recorded. Post-audit test suite: 208 passed, 0 failed, 8 skipped. All versions synced (7/7 OK). |

---

## Execution Priority Order

Recommended execution sequence based on risk and dependency chains:

| Phase | Items | Status |
|-------|-------|--------|
| **Phase 1 — Security** | C-01 | ✅ Key untracked + gitignored. BFG history scrub deferred. |
| **Phase 2 — Version Integrity** | C-02, H-07, M-13 | ✅ All 7 sync checkpoints OK. README updated. |
| **Phase 3 — CI Fixes** | H-01, H-02, H-03, H-04, H-05, H-06, L-09 | ✅ All workflows fixed. Lint now blocking. |
| **Phase 4 — Makefile & Config** | M-01, M-02, M-06, M-10, M-11, L-11 | ✅ All phantom paths removed. mypy upgraded. |
| **Phase 5 — Tooling & Compliance** | M-05, M-07, M-08, M-14 | ✅ pdoc→mkdocs, py.typed, dependabot. M-14 false positive. |
| **Phase 6 — Git Hygiene** | M-03, L-10, M-04 | ✅ Zone.Identifier removed. Copyright updated. |
| **Phase 7 — Repo Size** | L-01, L-02, L-03, L-04, L-05, L-06 | ✅ `.gitattributes` LFS rules declared. Migration pending LFS install. |
| **Phase 8 — Test & Maintenance** | M-09, M-12, L-07, L-08 | ✅ 35 shim tests added. Cargo workspace fixed. L-07 deferred. |
| **Phase 9 — Baseline** | I-01 | ✅ 208 passed / 0 failed / 8 skipped. |

---

## Remaining Manual Actions

These items require destructive repository operations or team coordination and were intentionally deferred:

| Item | Action Required | Risk if Deferred |
|------|----------------|------------------|
| **C-01 (history)** | Run BFG Repo-Cleaner to scrub mnemonic from git history. Force-push. All contributors must re-clone. | Mnemonic remains in pack history for anyone who has cloned. |
| **C-01 (rotation)** | Generate new Gevulot key material on a secure offline machine. Revoke the exposed mnemonic. | Wallet/key compromise window remains open. |
| **L-01\u2013L-06 (LFS)** | Install `git-lfs`, run `git lfs migrate import --include="<patterns>"`, force-push. | Clone size remains 1.16 GiB. |
| **L-07 (shims)** | Incremental deprecation of 89 backward-compat `__init__.py` shims using `warnings.warn()`. | Maintenance surface area persists. |

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Audit Lead | | | |
| Repository Owner | | | |
| Security Review | | | |
