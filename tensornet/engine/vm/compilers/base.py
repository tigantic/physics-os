"""QTT Physics VM — Base compiler interface."""

from __future__ import annotations

import abc

from ..ir import Program


class BaseCompiler(abc.ABC):
    """Abstract base for domain equation compilers.

    Every compiler takes physical parameters + discretization settings
    and produces a ``Program`` containing IR instructions.  The program
    can then be executed on any ``QTTRuntime`` instance.
    """

    @abc.abstractmethod
    def compile(self) -> Program:
        """Compile the domain equations into a VM program."""
        ...

    @property
    @abc.abstractmethod
    def domain(self) -> str:
        """Short machine-readable domain name."""
        ...

    @property
    @abc.abstractmethod
    def domain_label(self) -> str:
        """Human-readable domain description."""
        ...
