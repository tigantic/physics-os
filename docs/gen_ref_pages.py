"""Generate API reference pages for mkdocstrings.

This script is called by the mkdocs gen-files plugin during build.
It walks the tensornet package tree and creates a markdown page for
each module, then writes a SUMMARY.md for literate-nav.

See: https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages
"""

from pathlib import Path

import mkdocs_gen_files  # type: ignore[import-untyped]

nav = mkdocs_gen_files.Nav()

# Walk tensornet source tree
src = Path("tensornet")
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

    nav_parts = list(parts)
    nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.as_posix())

# Write nav file consumed by literate-nav
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
