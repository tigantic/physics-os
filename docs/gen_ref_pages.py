"""Generate API reference pages for mkdocstrings.

This script is called by the mkdocs gen-files plugin during build.
It walks the ontic package tree and creates a markdown page for
each module, then writes a SUMMARY.md for literate-nav.

Modules that cannot be imported (missing domain-specific deps in CI)
are silently skipped so the docs build does not fail.

See: https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages
"""

import importlib
import logging
from pathlib import Path

import mkdocs_gen_files  # type: ignore[import-untyped]

log = logging.getLogger("mkdocs.plugins.gen_ref_pages")

nav = mkdocs_gen_files.Nav()

# Walk ontic source tree
src = Path("ontic")
skipped: list[str] = []

for path in sorted(src.rglob("*.py")):
    # Skip __pycache__, test files, shim-only __init__.py
    if "__pycache__" in str(path):
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
    try:
        importlib.import_module(ident)
    except Exception:  # noqa: BLE001
        skipped.append(ident)
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
