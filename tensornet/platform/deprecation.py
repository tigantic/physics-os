"""
Deprecation & Versioning Policy — decorators and utilities for SemVer-gated
API lifecycle management.

Provides
--------
* ``@deprecated`` — decorator that emits ``DeprecationWarning`` with removal
  version and migration hint.
* ``@since`` — annotates a symbol with the version it was introduced.
* ``check_version_gate`` — enforces "this symbol was removed in vX.Y.Z".
* ``VersionInfo`` — parsed SemVer with comparison.
* ``Deprecation constants`` — used by CI to scan for pending removals.
"""

from __future__ import annotations

import functools
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

__all__ = [
    "deprecated",
    "since",
    "check_version_gate",
    "VersionInfo",
    "PLATFORM_VERSION",
]

F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════════════════════════
# SemVer parsing
# ═══════════════════════════════════════════════════════════════════════════════

_SEMVER_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<pre>[a-zA-Z0-9.]+))?"
    r"(?:\+(?P<build>[a-zA-Z0-9.]+))?$"
)


@dataclass(frozen=True, order=True)
class VersionInfo:
    """
    Parsed Semantic Version (major.minor.patch).

    Comparison operators use the standard SemVer precedence:
    major > minor > patch.  Pre-release tags are stored but do not
    affect ordering (simplification for internal use).
    """

    major: int
    minor: int
    patch: int
    pre: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, s: str) -> "VersionInfo":
        m = _SEMVER_RE.match(s.strip())
        if not m:
            raise ValueError(f"Invalid SemVer string: {s!r}")
        return cls(
            major=int(m.group("major")),
            minor=int(m.group("minor")),
            patch=int(m.group("patch")),
            pre=m.group("pre"),
            build=m.group("build"),
        )

    def __str__(self) -> str:
        s = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre:
            s += f"-{self.pre}"
        if self.build:
            s += f"+{self.build}"
        return s


PLATFORM_VERSION = VersionInfo(2, 0, 0)
"""Current platform version — Phase 7 release target."""


# ═══════════════════════════════════════════════════════════════════════════════
# @deprecated decorator
# ═══════════════════════════════════════════════════════════════════════════════


def deprecated(
    *,
    removal_version: str,
    alternative: Optional[str] = None,
    reason: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Mark a function or class as deprecated.

    Emits ``DeprecationWarning`` on every call, including the version in
    which the symbol will be removed and an optional migration hint.

    Parameters
    ----------
    removal_version : SemVer string (e.g. ``"3.0.0"``).
    alternative : replacement API to suggest.
    reason : why the deprecation happened.

    Example
    -------
    ::

        @deprecated(removal_version="3.0.0", alternative="new_solve()")
        def old_solve(x):
            ...
    """
    rv = VersionInfo.parse(removal_version)

    def decorator(fn: F) -> F:
        msg_parts = [f"{fn.__qualname__} is deprecated"]
        if reason:
            msg_parts.append(f" ({reason})")
        msg_parts.append(f"; will be removed in v{rv}")
        if alternative:
            msg_parts.append(f".  Use {alternative} instead")
        msg = "".join(msg_parts) + "."

        # If current version >= removal version, raise immediately
        if PLATFORM_VERSION >= rv:
            @functools.wraps(fn)
            def errored(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError(
                    f"{fn.__qualname__} was removed in v{rv}.  "
                    f"{'Use ' + alternative + ' instead.' if alternative else ''}"
                )
            return errored  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__deprecated__ = True  # type: ignore[attr-defined]
        wrapper.__removal_version__ = str(rv)  # type: ignore[attr-defined]
        wrapper.__alternative__ = alternative  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# @since decorator
# ═══════════════════════════════════════════════════════════════════════════════


def since(version: str) -> Callable[[F], F]:
    """
    Annotate a function or class with the version it was introduced.

    No runtime effect beyond setting ``__since__`` on the decorated object.

    Example
    -------
    ::

        @since("1.0.0")
        def solve(x):
            ...
    """
    v = VersionInfo.parse(version)

    def decorator(fn: F) -> F:
        fn.__since__ = str(v)  # type: ignore[attr-defined]
        return fn

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Version gate check (for CI / static analysis)
# ═══════════════════════════════════════════════════════════════════════════════


def check_version_gate(
    current: Optional[VersionInfo] = None,
) -> list[dict[str, str]]:
    """
    Walk all modules that have been imported and find symbols marked with
    ``@deprecated``.  Return a list of entries whose ``removal_version``
    has been reached or exceeded by *current*.

    Intended for CI pipelines to fail the build when deprecated code should
    have been removed.

    Returns
    -------
    List of dicts ``{"name": ..., "removal_version": ..., "alternative": ...}``
    where removal is overdue.
    """
    import sys

    current = current or PLATFORM_VERSION
    overdue: list[dict[str, str]] = []

    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("tensornet"):
            continue
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            rv_str = getattr(obj, "__removal_version__", None)
            if rv_str is None:
                continue
            try:
                rv = VersionInfo.parse(rv_str)
            except ValueError:
                continue
            if current >= rv:
                overdue.append({
                    "name": f"{mod_name}.{attr_name}",
                    "removal_version": str(rv),
                    "alternative": getattr(obj, "__alternative__", "") or "",
                })

    return overdue
