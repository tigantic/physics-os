# Violation_Lies.md (HVAC_CFD integrity audit)

Date: 2026-01-12
Updated: 2026-01-12
Scope: HVAC_CFD documentation + HVAC_CFD/ui implementation

Goal: identify claims that are demonstrably false or materially misleading, and link each one to concrete evidence.

Definitions:
- Lie: an explicit claim/checkmark that contradicts the repo contents.
- Violation: repo behavior contradicts mandatory standards in HVAC_CFD/UI_E.md.txt.

This report intentionally avoids speculation. If something is not proven with evidence links, it is not asserted here.

---

## RESOLVED violations (fixed in this session)

### 1) ~~"No hardcoded credentials or paths in source"~~ ✅ FIXED

**Resolution:** Removed hardcoded `/home/brad/...` path from SceneController.cpp and replaced with `HYPERFOAM_BLUEPRINT_PARSER_PATH` environment variable override.

Evidence of fix:
- HVAC_CFD/ui/src/controllers/SceneController.cpp#L1215-L1234 (now uses env var)

### 2) ~~"-Wall -Wextra -Werror not enforced on Linux/macOS"~~ ✅ FIXED

**Resolution:** Re-enabled `-Werror` on all platforms (including GCC/Clang) and fixed all warnings that surfaced:
- UndoManager.cpp: sign-compare warnings (cast m_cleanIndex to size_t)
- FieldStreamReader.cpp: nodiscard, unused parameters, unused variables
- ProjectValidator.cpp: range-loop-construct warning
- DomainCommands.cpp: sign-compare warnings
- SingleInstance.cpp: ignored ftruncate/write return values
- GeometryImportService.cpp: unused parameter
- ReproBundleService.cpp: unused parameter
- SliceRenderer.cpp: unused variable
- GlyphRenderer.cpp: unused variables

Evidence of fix:
- HVAC_CFD/ui/CMakeLists.txt#L71-L82 (now -Werror on all compilers)
- Build: 28/28 tests passing with -Werror

### 3) ~~Qt baseline inconsistent / stubs shipped for Qt < 6.6~~ ✅ FIXED

**Resolution:** CMakeLists.txt now issues `FATAL_ERROR` if Qt < 6.6. No stub fallback. Qt 6.7.3 installed via aqtinstall for development.

Evidence of fix:
- HVAC_CFD/ui/CMakeLists.txt#L56-L66 (FATAL_ERROR on Qt < 6.6)

### 4) ~~CI claims muddled: multiple workflow copies~~ ✅ FIXED

**Resolution:** Consolidated to single canonical workflow at `.github/workflows/ui-ci.yml`. Deleted duplicates:
- HVAC_CFD/.github/workflows/ui-build.yml (deleted)
- HVAC_CFD/ui/.github/workflows/ci.yml (deleted)
- HVAC_CFD/ui/.github/workflows/test.yml (deleted)
- HVAC_CFD/ui/.github/workflows/release.yml (deleted)

Evidence of fix:
- .github/workflows/ui-ci.yml (single source of truth)

### 5) ~~clang-tidy advisory (|| echo fallback)~~ ✅ FIXED

**Resolution:** clang-tidy in CI now runs without || echo fallback. Failures will gate the build.

Evidence of fix:
- .github/workflows/ui-ci.yml (clang-tidy step is gating)

---

## REMAINING violations (require further work)

### 1) Major test suites declared but skipped/disabled (Documentation drift)

**Status: PARTIALLY ADDRESSED**

The following test suites are now functional:
- Core tests (28 tests passing)
- Accessibility tests
- UI automation tests

The following remain skipped due to missing dependencies:
- Security tests (need full IPC mock implementation)
- Perf/soak tests
- GPU/Viz tests (require GPU context)

Evidence:
- HVAC_CFD/ui/CMakeLists.txt#L419-L462

Required fix:
- Security tests can be enabled once ProjectManager/IPCBridge mocks are complete.

### 2) Test coverage threshold not enforced

**Status: DOCUMENTED AS ADVISORY**

Reality:
- Coverage is collected but no minimum threshold is enforced in CI.

Required fix:
- Add coverage gating or explicitly document that it's advisory.

---

## Addressed this session (Phase 8.2)

### 3) Input validation not comprehensive ✅ FIXED

**Resolution:** Added comprehensive input validation to JobSpec.cpp:
- validateFiniteF() - rejects NaN/Inf values
- validateIntRange() - bounds checking for integers
- validateGridDim() - grid dimension limits (max 4096 per dimension, 100M total cells)

Evidence of fix:
- HVAC_CFD/ui/src/core/JobSpec.cpp (validation helpers at top of file)
- Tests: 28/28 passing

### 4) File writes not crash-safe ✅ FIXED

**Resolution:** JobSpec::saveToFile() now uses QSaveFile for atomic writes.
Writes to temp file, then performs atomic rename.

Evidence of fix:
- HVAC_CFD/ui/src/core/JobSpec.cpp#saveToFile() uses QSaveFile

---

## Local workspace hygiene (not a lie, but worth tightening)

- HVAC_CFD/.venv exists locally, but it is gitignored.
  - Evidence: .gitignore#L67
  - Recommendation: keep it untracked and exclude it from any build/release packaging.

---

## Build verification

```
Date: 2026-01-12
Qt version: 6.7.3 (via aqtinstall)
Build flags: -Wall -Wextra -Wpedantic -Werror
Tests: 28/28 passed
RHI: Enabled (Qt >= 6.6)
```
