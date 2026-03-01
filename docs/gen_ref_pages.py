"""Generate API reference pages for mkdocstrings.

This script is called by the mkdocs gen-files plugin during build.
It walks the ontic package tree and creates a markdown page for
each module, then writes a SUMMARY.md for literate-nav.

Modules that cannot be imported (missing domain-specific deps in CI)
are silently skipped so the docs build does not fail.

See: https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages
"""

import importlib
import io
import logging
import sys
from pathlib import Path

import griffe  # used by mkdocstrings — pre-check that collection works
import mkdocs_gen_files  # type: ignore[import-untyped]

log = logging.getLogger("mkdocs.plugins.gen_ref_pages")

# Sentinel strings that indicate a module's dependencies are missing.
# If any of these appear on stdout/stderr during import, the module
# is skipped (it would pass import but fail mkdocstrings collection).
_IMPORT_ERROR_SENTINELS = (
    "not installed",
    "not available",
    "not found",
    "requires gpu",
    "requires cuda",
    "no module named",
)

# Subdirectories of ontic/ that require hardware or domain-specific
# packages not available in the docs CI environment.  Pages for these
# are skipped to prevent mkdocstrings CollectionError crashes.
_SKIP_DIRS = frozenset({
    "cuda",          # requires pycuda / CUDA toolkit
    "gpu",           # requires GPU runtime
    "triton",        # requires Triton JIT
})

nav = mkdocs_gen_files.Nav()

# Walk ontic source tree
src = Path("ontic")
skipped: list[str] = []

for path in sorted(src.rglob("*.py")):
    # Skip __pycache__, test files, shim-only __init__.py
    if "__pycache__" in str(path):
        continue

    # Skip directories that require unavailable hardware/deps
    rel_parts = path.relative_to(src).parts
    if any(part in _SKIP_DIRS for part in rel_parts):
        continue

    module_path = path.relative_to(".")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.with_suffix("").parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    ident = ".".join(parts)

    # Verify the module can be imported before generating a reference page.
    # Modules with unresolvable dependencies (pycuda, domain-specific libs)
    # are skipped to prevent mkdocstrings CollectionError during build.
    # We also capture stdout/stderr: modules that gracefully degrade by
    # printing "not installed" / "not available" without raising still
    # break griffe collection.
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = cap_out = io.StringIO()
    sys.stderr = cap_err = io.StringIO()
    try:
        importlib.import_module(ident)
    except BaseException:  # noqa: BLE001  — includes SystemExit from GPU-only modules
        skipped.append(ident)
        continue
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    captured = (cap_out.getvalue() + cap_err.getvalue()).lower()
    if any(sentinel in captured for sentinel in _IMPORT_ERROR_SENTINELS):
        skipped.append(ident)
        log.info("Skipping %s — import emitted: %s", ident, captured.strip()[:120])
        continue

    # Final gate: verify griffe (which mkdocstrings uses) can actually
    # collect the module.  If griffe cannot resolve the module, a
    # reference page for it will always crash the build.
    try:
        griffe.load(ident)
    except Exception:  # noqa: BLE001
        skipped.append(ident)
        log.info("Skipping %s — griffe collection failed", ident)
        continue

    nav_parts = list(parts)
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.as_posix())

if skipped:
    log.warning("Skipped %d modules with unresolvable imports: %s", len(skipped), ", ".join(skipped[:10]))

# Write nav file consumed by literate-nav
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
