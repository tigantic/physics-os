"""Product UI sub-package — HTTP API + single-page application.

Modes
-----
G1  Case Library     – Browse / search / curate case library
G2  Twin Inspect     – Inspect digital twin (mesh, materials, landmarks)
G3  Plan Author      – Author / edit surgical plans with DSL operators
G4  Consult          – Interactive exploration (what-if scenarios)
G5  Report           – Generate / view surgical reports
G6  3D Visualization – Three.js-based 3D mesh viewer
G7  Timeline         – Scrub through simulation timeline
G8  Compare          – Side-by-side pre/post comparison
G9  Contract         – Interaction contract (UI ↔ Plan DSL mapping)
"""

from .api import UIApplication
from .server import start_server

__all__ = ["UIApplication", "start_server"]
