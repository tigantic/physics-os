"""
5.7 — LLM-to-Solver Pipeline Enhancement
==========================================

Extends the existing tensornet.intent regex-based NL parsing with
real LLM integration via configurable backends (OpenAI, local, mock)
and full solver dispatch from natural language queries.

Components:
    * LLMBackend — abstract backend with OpenAI / local / mock impls
    * IntentClassifier — classify user query into solver intent
    * SolverDispatcher — route classified intent to solver execution
    * LLMSolverPipeline — end-to-end NL → solution orchestrator
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ── Intent taxonomy ───────────────────────────────────────────────

class SolverIntent(Enum):
    """High-level solver categories parseable from natural language."""
    CFD_STEADY = auto()
    CFD_TRANSIENT = auto()
    STRUCTURAL_STATIC = auto()
    STRUCTURAL_DYNAMIC = auto()
    THERMAL_CONDUCTION = auto()
    THERMAL_CONVECTION = auto()
    ELECTROMAGNETICS = auto()
    QUANTUM_EIGENSOLVER = auto()
    OPTIMIZATION = auto()
    MESH_GENERATION = auto()
    POST_PROCESSING = auto()
    UNKNOWN = auto()


@dataclass
class ParsedQuery:
    """Structured result from NL query analysis."""
    raw_text: str
    intent: SolverIntent
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    domain_keywords: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "intent": self.intent.name,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "domain_keywords": self.domain_keywords,
            "constraints": self.constraints,
        }


# ── LLM Backend abstraction ──────────────────────────────────────

class LLMBackend(ABC):
    """Abstract LLM backend for natural language processing."""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a completion for the given prompt."""
        ...

    @abstractmethod
    def extract_json(self, prompt: str) -> Dict[str, Any]:
        """Generate a completion and parse as JSON."""
        ...


class MockLLMBackend(LLMBackend):
    """Deterministic mock backend using keyword matching.

    Provides a fully functional pipeline without external
    dependencies — suitable for testing and offline operation.
    """

    _INTENT_KEYWORDS: Dict[SolverIntent, List[str]] = {
        SolverIntent.CFD_STEADY: [
            "steady", "incompressible", "laminar", "stokes", "airfoil",
            "pipe flow", "duct", "pressure drop",
        ],
        SolverIntent.CFD_TRANSIENT: [
            "transient", "unsteady", "turbulent", "les", "dns", "vortex",
            "time-dependent", "temporal",
        ],
        SolverIntent.STRUCTURAL_STATIC: [
            "static", "stress", "displacement", "beam", "truss",
            "deformation", "load", "cantilever",
        ],
        SolverIntent.STRUCTURAL_DYNAMIC: [
            "vibration", "modal", "eigenfrequency", "dynamic",
            "impact", "fatigue",
        ],
        SolverIntent.THERMAL_CONDUCTION: [
            "conduction", "heat transfer", "thermal", "temperature",
            "fourier", "insulation",
        ],
        SolverIntent.THERMAL_CONVECTION: [
            "convection", "buoyancy", "rayleigh", "nusselt",
            "natural convection", "forced convection",
        ],
        SolverIntent.ELECTROMAGNETICS: [
            "maxwell", "electromagnetic", "antenna", "waveguide",
            "radar", "em field",
        ],
        SolverIntent.QUANTUM_EIGENSOLVER: [
            "quantum", "eigenvalue", "schrodinger", "hamiltonian",
            "wavefunction", "entanglement",
        ],
        SolverIntent.OPTIMIZATION: [
            "optimize", "minimise", "minimize", "objective",
            "topology optimization", "shape optimization",
        ],
        SolverIntent.MESH_GENERATION: [
            "mesh", "grid", "triangulate", "cells", "refinement",
            "adaptive mesh",
        ],
        SolverIntent.POST_PROCESSING: [
            "visuali", "plot", "contour", "streamline", "render",
            "post-process", "export",
        ],
    }

    _PARAM_PATTERNS = {
        "reynolds": re.compile(r"[Rr]e(?:ynolds)?\s*[=:]\s*([\d.eE+\-]+)"),
        "mach": re.compile(r"[Mm]ach\s*[=:]\s*([\d.eE+\-]+)"),
        "temperature": re.compile(r"[Tt](?:emp(?:erature)?)?\s*[=:]\s*([\d.eE+\-]+)"),
        "pressure": re.compile(r"[Pp](?:ressure)?\s*[=:]\s*([\d.eE+\-]+)"),
        "velocity": re.compile(r"[Vv](?:elocity)?\s*[=:]\s*([\d.eE+\-]+)"),
        "load": re.compile(r"[Ll]oad\s*[=:]\s*([\d.eE+\-]+)"),
        "frequency": re.compile(r"[Ff]req(?:uency)?\s*[=:]\s*([\d.eE+\-]+)"),
        "dimension": re.compile(r"(\d)[dD]"),
    }

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Keyword-based deterministic 'completion'."""
        lower = prompt.lower()
        intent = self._classify(lower)
        params = self._extract_params(prompt)
        response = {
            "intent": intent.name,
            "parameters": params,
            "confidence": 0.85 if intent != SolverIntent.UNKNOWN else 0.1,
        }
        return json.dumps(response)

    def extract_json(self, prompt: str) -> Dict[str, Any]:
        return json.loads(self.complete(prompt))

    def _classify(self, text: str) -> SolverIntent:
        best_intent = SolverIntent.UNKNOWN
        best_score = 0
        for intent, keywords in self._INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_intent = intent
        return best_intent

    def _extract_params(self, text: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name, pattern in self._PARAM_PATTERNS.items():
            m = pattern.search(text)
            if m:
                try:
                    params[name] = float(m.group(1))
                except ValueError:
                    params[name] = m.group(1)
        return params


class OpenAIBackend(LLMBackend):
    """OpenAI API backend (requires OPENAI_API_KEY env var)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        import os
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Call OpenAI chat.completions endpoint."""
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — use MockLLMBackend for offline operation"
            )
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are a physics simulation assistant. "
                    "Classify the user query into a solver intent and extract parameters. "
                    "Reply with JSON only."
                )},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def extract_json(self, prompt: str) -> Dict[str, Any]:
        raw = self.complete(prompt)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw)


# ── Intent classifier ─────────────────────────────────────────────

class IntentClassifier:
    """Classify natural-language queries into SolverIntent with parameters."""

    _SYSTEM_PROMPT = (
        "Given a physics simulation query, respond with JSON:\n"
        '{"intent": "<INTENT>", "parameters": {...}, "confidence": 0.0-1.0}\n'
        "Valid intents: " + ", ".join(i.name for i in SolverIntent) + "\n"
        "Extract numeric parameters where present (reynolds, mach, etc.)."
    )

    def __init__(self, backend: Optional[LLMBackend] = None) -> None:
        self.backend = backend or MockLLMBackend()

    def classify(self, query: str) -> ParsedQuery:
        """Parse a natural-language query into structured form."""
        prompt = f"{self._SYSTEM_PROMPT}\n\nQuery: {query}"
        try:
            result = self.backend.extract_json(prompt)
        except (json.JSONDecodeError, RuntimeError):
            # Fallback to mock
            fallback = MockLLMBackend()
            result = fallback.extract_json(prompt)

        intent_name = result.get("intent", "UNKNOWN")
        try:
            intent = SolverIntent[intent_name]
        except KeyError:
            intent = SolverIntent.UNKNOWN

        return ParsedQuery(
            raw_text=query,
            intent=intent,
            parameters=result.get("parameters", {}),
            confidence=float(result.get("confidence", 0.0)),
            domain_keywords=self._extract_keywords(query),
        )

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        physics_words = {
            "reynolds", "mach", "navier", "stokes", "euler", "heat",
            "stress", "strain", "modal", "turbulence", "laminar",
            "boundary", "mesh", "fea", "cfd", "fem", "quantum",
            "electromagnetic", "thermal", "pressure", "velocity",
        }
        words = set(re.findall(r"\w+", text.lower()))
        return sorted(words & physics_words)


# ── Solver dispatcher ─────────────────────────────────────────────

@dataclass
class SolverResult:
    """Result from solver dispatch."""
    success: bool
    solver_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class SolverDispatcher:
    """Route parsed intent to registered solver functions."""

    def __init__(self) -> None:
        self._registry: Dict[SolverIntent, Callable[..., SolverResult]] = {}

    def register(
        self, intent: SolverIntent, solver_fn: Callable[..., SolverResult],
    ) -> None:
        """Register a solver function for an intent."""
        self._registry[intent] = solver_fn

    def dispatch(self, parsed: ParsedQuery) -> SolverResult:
        """Execute the solver matching the parsed intent."""
        if parsed.intent not in self._registry:
            return SolverResult(
                success=False,
                solver_name="none",
                error=f"No solver registered for intent {parsed.intent.name}",
            )
        fn = self._registry[parsed.intent]
        t0 = time.time()
        try:
            result = fn(**parsed.parameters)
            result.elapsed_seconds = time.time() - t0
            return result
        except Exception as e:
            return SolverResult(
                success=False,
                solver_name=fn.__name__,
                error=str(e),
                elapsed_seconds=time.time() - t0,
            )

    @property
    def registered_intents(self) -> List[str]:
        return [i.name for i in self._registry]


# ── End-to-end pipeline ──────────────────────────────────────────

class LLMSolverPipeline:
    """End-to-end natural language → solver → result pipeline.

    Usage::

        pipe = LLMSolverPipeline()
        pipe.register_solver(SolverIntent.CFD_STEADY, my_cfd_fn)
        result = pipe.run("Solve steady laminar flow, Re=100")
    """

    def __init__(self, backend: Optional[LLMBackend] = None) -> None:
        self.classifier = IntentClassifier(backend)
        self.dispatcher = SolverDispatcher()
        self._history: List[Dict[str, Any]] = []

    def register_solver(
        self, intent: SolverIntent, fn: Callable[..., SolverResult],
    ) -> None:
        self.dispatcher.register(intent, fn)

    def run(self, query: str) -> SolverResult:
        """Process a natural-language query end-to-end."""
        parsed = self.classifier.classify(query)
        result = self.dispatcher.dispatch(parsed)
        self._history.append({
            "query": parsed.to_dict(),
            "result": {
                "success": result.success,
                "solver": result.solver_name,
                "error": result.error,
                "elapsed": result.elapsed_seconds,
            },
        })
        return result

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


__all__ = [
    "SolverIntent",
    "ParsedQuery",
    "LLMBackend",
    "MockLLMBackend",
    "OpenAIBackend",
    "IntentClassifier",
    "SolverResult",
    "SolverDispatcher",
    "LLMSolverPipeline",
]
