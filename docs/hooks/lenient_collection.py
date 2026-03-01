"""MkDocs hook: make mkdocstrings collection errors non-fatal.

mkdocstrings 1.0.x removed the ``on_error: warn`` configuration option.
CollectionError in ``AutoDocProcessor._process_block`` unconditionally
raises ``PluginError``, which aborts the entire build.

This hook monkeypatches ``AutoDocProcessor.run`` so that a
``PluginError`` triggered by collection is downgraded to a warning.
The failing ``:::`` directive is silently dropped from the rendered page,
allowing the rest of the documentation to build successfully.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

log = logging.getLogger("mkdocs.hooks.lenient_collection")

_patched = False


def on_startup(**_kwargs: object) -> None:
    """Patch AutoDocProcessor.run before plugins initialise."""
    global _patched  # noqa: PLW0603
    if _patched:
        return
    _patched = True

    try:
        from mkdocstrings import AutoDocProcessor
    except ImportError:
        log.debug("mkdocstrings not installed — nothing to patch")
        return

    try:
        from mkdocs.exceptions import PluginError
    except ImportError:
        # Older MkDocs, try BuildError
        from mkdocs.exceptions import BuildError as PluginError  # type: ignore[no-redef]

    _original_run = AutoDocProcessor.run

    def _lenient_run(
        self: AutoDocProcessor,
        parent: Element,
        blocks: list[str],
    ) -> None:
        """Wrap ``run`` to catch PluginError from collection failures."""
        try:
            _original_run(self, parent, blocks)
        except PluginError as exc:
            log.warning("mkdocstrings collection skipped: %s", exc)

    AutoDocProcessor.run = _lenient_run  # type: ignore[assignment]
    log.info("Patched mkdocstrings AutoDocProcessor for lenient collection")
