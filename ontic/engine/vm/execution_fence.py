"""QTT VM — Execution Fence (Never-Dense Enforcement).

Thread-local context manager that prevents ``to_dense()`` from being
called during the VM dispatch loop.  This upgrades the "never go dense"
rule from a coding convention to a hard runtime invariant.

Usage in the runtime::

    from .execution_fence import vm_dispatch_context, DenseInDispatchError

    with vm_dispatch_context():
        # All opcode dispatch happens here.
        # Any call to to_dense() will raise DenseInDispatchError.
        ...

Usage in QTTTensor / GPUQTTTensor::

    from .execution_fence import assert_not_in_dispatch

    def to_dense(self):
        assert_not_in_dispatch()  # raises if inside dispatch
        ...

§5.3 Rank Governor: "NEVER call to_dense() — kills QTT"
§20.4 IP Boundary: to_dense() is acceptable ONLY for post-execution
    sanitizer/reporting, never inside the timestep dispatch.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator


_thread_local = threading.local()


class DenseInDispatchError(RuntimeError):
    """Raised when to_dense() is called inside the VM dispatch loop.

    This is an automatic rejection per QTT Law §5:
    "to_dense() must NEVER appear in any execution path."
    """


def _is_in_dispatch() -> bool:
    """Check whether the current thread is inside a VM dispatch context."""
    return getattr(_thread_local, "in_dispatch", False)


@contextmanager
def vm_dispatch_context() -> Generator[None, None, None]:
    """Context manager that marks the current thread as inside VM dispatch.

    While active, any call to ``assert_not_in_dispatch()`` will raise
    ``DenseInDispatchError``.  Supports nesting (refcount-based).
    """
    depth = getattr(_thread_local, "dispatch_depth", 0)
    _thread_local.dispatch_depth = depth + 1
    _thread_local.in_dispatch = True
    try:
        yield
    finally:
        _thread_local.dispatch_depth -= 1
        if _thread_local.dispatch_depth == 0:
            _thread_local.in_dispatch = False


def assert_not_in_dispatch() -> None:
    """Raise ``DenseInDispatchError`` if called inside VM dispatch.

    This is the enforcement hook called by ``to_dense()`` on both
    ``QTTTensor`` and ``GPUQTTTensor``.
    """
    if _is_in_dispatch():
        raise DenseInDispatchError(
            "to_dense() called inside VM dispatch loop. "
            "This violates QTT Law §5: 'to_dense() must NEVER appear "
            "in any execution path.' to_dense() is permitted ONLY for "
            "post-execution reporting/sanitization."
        )
