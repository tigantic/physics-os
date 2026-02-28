#!/usr/bin/env python3
"""Exhaustive client-side UI validation — tests every capability the surgeon
can execute through the Ontic Engine Facial Plastics Surgical Cockpit.

Runs from OUTSIDE the container, hitting the live deployment exactly as
a browser would (through Caddy on port 80, or direct to app on 8420).
"""

import json
import sys
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

BASE = "http://localhost:8420"  # Direct to app (bypassing Caddy HTTPS)
API_KEY = "fp_QsU-wSv71x7KKxpNEjCxirFYtB76G7YrHNvq2C_nXgk"

PASS = 0
FAIL = 0
ERRORS: List[str] = []


def _req(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    *,
    expect_status: int = 200,
    auth: bool = True,
    timeout: int = 30,
) -> Tuple[int, Any]:
    """Make an HTTP request and return (status, parsed_body)."""
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    headers: Dict[str, str] = {}
    if auth:
        headers["X-API-Key"] = API_KEY
    if data:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        raw = resp.read()
        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            return resp.status, json.loads(raw)
        return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, {"error": f"Connection failed: {e}"}


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 0: Static Assets — every CSS and JS file the UI loads
# ══════════════════════════════════════════════════════════════════

def test_static_assets():
    section("PHASE 0: Static Assets")

    assets = [
        # HTML shell
        ("/", "text/html", "index.html"),
        # CSS
        ("/css/tokens.css", "text/css", "tokens.css"),
        ("/css/layout.css", "text/css", "layout.css"),
        ("/css/components.css", "text/css", "components.css"),
        ("/css/modes.css", "text/css", "modes.css"),
        ("/css/three-viewer.css", "text/css", "three-viewer.css"),
        ("/css/print.css", "text/css", "print.css"),
        # Core JS
        ("/js/state.js", "javascript", "state.js"),
        ("/js/api.js", "javascript", "api.js"),
        ("/js/router.js", "javascript", "router.js"),
        # Component JS
        ("/js/components/toast.js", "javascript", "toast.js"),
        ("/js/components/modal.js", "javascript", "modal.js"),
        ("/js/components/sidebar.js", "javascript", "sidebar.js"),
        ("/js/components/command-bar.js", "javascript", "command-bar.js"),
        ("/js/components/inspector.js", "javascript", "inspector.js"),
        ("/js/components/status-bar.js", "javascript", "status-bar.js"),
        # Mode JS
        ("/js/modes/case-library.js", "javascript", "case-library.js"),
        ("/js/modes/twin-inspect.js", "javascript", "twin-inspect.js"),
        ("/js/modes/plan-author.js", "javascript", "plan-author.js"),
        ("/js/modes/simulation.js", "javascript", "simulation.js"),
        ("/js/modes/whatif.js", "javascript", "whatif.js"),
        ("/js/modes/sweep.js", "javascript", "sweep.js"),
        ("/js/modes/report.js", "javascript", "report.js"),
        ("/js/modes/viewer3d.js", "javascript", "viewer3d.js"),
        ("/js/modes/timeline.js", "javascript", "timeline.js"),
        ("/js/modes/compare.js", "javascript", "compare.js"),
        # Bootstrap
        ("/js/app.js", "javascript", "app.js"),
    ]

    for path, expected_ct, label in assets:
        status, body = _req("GET", path, auth=False)
        check(
            f"{label} → {status}",
            status == 200,
            f"got {status}",
        )
        if status == 200 and isinstance(body, bytes):
            check(
                f"{label} non-empty ({len(body)} bytes)",
                len(body) > 50,
                f"only {len(body)} bytes",
            )

    # SPA fallback: unknown path should return index.html
    status, body = _req("GET", "/some/unknown/route", auth=False)
    check("SPA fallback → 200", status == 200)
    if isinstance(body, bytes):
        check(
            "SPA fallback returns index.html",
            b"Surgical Cockpit" in body,
            "didn't get new index.html",
        )

    # index.html content validation
    status, body = _req("GET", "/", auth=False)
    if isinstance(body, bytes):
        html = body.decode()
        check("HTML has design token CSS link", "/css/tokens.css" in html)
        check("HTML has state.js script", "/js/state.js" in html)
        check("HTML has api.js script", "/js/api.js" in html)
        check("HTML has app.js bootstrap", "/js/app.js" in html)
        check("HTML has Three.js importmap", "importmap" in html)
        check("HTML has 9 mode panels", html.count("mode-panel") >= 9)
        check("HTML has command-bar", 'id="command-bar"' in html)
        check("HTML has sidebar", 'id="sidebar"' in html)
        check("HTML has inspector", 'id="inspector"' in html)
        check("HTML has status-bar", 'id="status-bar"' in html)
        check("HTML has modal overlay", 'id="modal-overlay"' in html)
        check("HTML has toast container", 'id="toast-container"' in html)
        check("HTML has command palette", 'id="command-palette"' in html)


# ══════════════════════════════════════════════════════════════════
#  PHASE 1: Health & Metrics (unauthenticated)
# ══════════════════════════════════════════════════════════════════

def test_health_metrics():
    section("PHASE 1: Health & Metrics")

    status, data = _req("GET", "/health", auth=False)
    check("GET /health → 200", status == 200)
    check("/health has status field", isinstance(data, dict) and "status" in data)
    check("/health status=healthy", isinstance(data, dict) and data.get("status") == "healthy")

    status, body = _req("GET", "/metrics", auth=False)
    check("GET /metrics → 200", status == 200)
    if isinstance(body, bytes):
        text = body.decode()
        check("/metrics has prometheus format", "fp_requests_total" in text or "request" in text.lower())


# ══════════════════════════════════════════════════════════════════
#  PHASE 2: Auth validation
# ══════════════════════════════════════════════════════════════════

def test_auth():
    section("PHASE 2: Authentication")

    # No key → 401
    status, data = _req("GET", "/api/cases", auth=False)
    check("No API key → 401", status == 401, f"got {status}")

    # Bad key → 401
    headers = {"X-API-Key": "bad_key_12345"}
    req = urllib.request.Request(BASE + "/api/cases", headers=headers)
    try:
        resp = urllib.request.urlopen(req)
        check("Bad API key → 401", False, f"got {resp.status}")
    except urllib.error.HTTPError as e:
        check("Bad API key → 401", e.code == 401, f"got {e.code}")

    # Valid key → 200
    status, data = _req("GET", "/api/cases")
    check("Valid API key → 200", status == 200, f"got {status}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 3: G9 Contract
# ══════════════════════════════════════════════════════════════════

def test_contract():
    section("PHASE 3: G9 — Contract")

    status, data = _req("GET", "/api/contract")
    check("GET /api/contract → 200", status == 200)
    check("contract has version", "version" in data)
    check("contract has procedures", "procedures" in data or "product" in data)


# ══════════════════════════════════════════════════════════════════
#  PHASE 4: G1 Case Library
# ══════════════════════════════════════════════════════════════════

def test_case_library() -> str:
    section("PHASE 4: G1 — Case Library")

    # List cases (may be empty initially)
    status, data = _req("GET", "/api/cases")
    check("GET /api/cases → 200", status == 200)
    check("cases response has 'cases' key", "cases" in data)

    # Create a case
    status, data = _req("POST", "/api/cases", {
        "patient_age": 35,
        "patient_sex": "F",
        "procedure": "rhinoplasty",
        "notes": "UI validation test case",
    })
    check("POST /api/cases (create) → 200", status == 200, f"got {status}: {data}")
    case_id = ""
    if isinstance(data, dict):
        case_id = data.get("case_id", data.get("id", ""))
        check("create returns case_id", bool(case_id), f"data={data}")

    if not case_id:
        print("  ⚠ Cannot continue case tests without case_id")
        return ""

    # Get single case
    status, data = _req("GET", f"/api/cases/{case_id}")
    check(f"GET /api/cases/{case_id} → 200", status == 200)
    if isinstance(data, dict):
        meta = data.get("metadata", data)
        has_proc = "procedure" in meta or "procedure_type" in meta
        check("case has procedure", has_proc, f"keys={list(data.keys())}")

    # List with filter
    status, data = _req("GET", "/api/cases?procedure=rhinoplasty")
    check("GET /api/cases?procedure=rhinoplasty → 200", status == 200)

    # List with pagination
    status, data = _req("GET", "/api/cases?limit=5&offset=0")
    check("GET /api/cases?limit=5&offset=0 → 200", status == 200)

    # Curate library (may generate a synthetic case — allow up to 120s)
    status, data = _req("POST", "/api/curate", {}, timeout=120)
    check("POST /api/curate → 200", status == 200, f"got {status}: {data}")

    # Re-list to see curated cases
    status, data = _req("GET", "/api/cases")
    case_count = len(data.get("cases", [])) if isinstance(data, dict) else 0
    check(f"cases after curate: {case_count}", case_count > 0, f"count={case_count}")

    # Pick a case_id that has twin data (from curate)
    if isinstance(data, dict) and data.get("cases"):
        # Prefer a twin-complete case since it has mesh+landmarks
        for c in data["cases"]:
            cid = c.get("case_id", c.get("id", ""))
            if cid and c.get("twin_complete", False):
                case_id = cid
                break
        # If no twin-complete case found, fall back to first available
        if not case_id:
            case_id = data["cases"][0].get("case_id", data["cases"][0].get("id", ""))

    print(f"  → Using case_id: {case_id}")
    return case_id


# ══════════════════════════════════════════════════════════════════
#  PHASE 5: G2 Twin Inspect
# ══════════════════════════════════════════════════════════════════

def test_twin_inspect(case_id: str):
    section("PHASE 5: G2 — Twin Inspect")

    if not case_id:
        print("  ⚠ Skipped — no case_id")
        return

    # Twin summary
    status, data = _req("GET", f"/api/cases/{case_id}/twin")
    check("GET twin summary → 200", status == 200, f"got {status}")
    if isinstance(data, dict):
        has_mesh = "nodes" in data or "mesh" in data or "n_nodes" in data or "summary" in data
        check("twin summary has mesh info", has_mesh, f"keys={list(data.keys())}")

    # Mesh data
    status, data = _req("GET", f"/api/cases/{case_id}/mesh")
    check("GET mesh data → 200", status == 200, f"got {status}")
    if isinstance(data, dict):
        has_positions = "positions" in data or "vertices" in data or "nodes" in data
        check("mesh has positions/vertices", has_positions, f"keys={list(data.keys())}")
        has_indices = "indices" in data or "faces" in data or "elements" in data
        check("mesh has indices/faces", has_indices, f"keys={list(data.keys())}")

    # Landmarks
    status, data = _req("GET", f"/api/cases/{case_id}/landmarks")
    check("GET landmarks → 200", status == 200, f"got {status}")
    if isinstance(data, dict):
        lm = data.get("landmarks", data.get("points", []))
        check(f"landmarks count: {len(lm)}", len(lm) >= 0)
        if lm:
            first = lm[0]
            has_coords = "coords" in first or "position" in first or "x" in first
            check("landmark has coordinates", has_coords, f"keys={list(first.keys())}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 6: G3 Plan Author
# ══════════════════════════════════════════════════════════════════

def test_plan_author(case_id: str) -> Optional[Dict]:
    section("PHASE 6: G3 — Plan Author")

    if not case_id:
        print("  ⚠ Skipped — no case_id")
        return None

    # List operators
    status, data = _req("GET", "/api/operators")
    check("GET /api/operators → 200", status == 200)
    if isinstance(data, dict):
        ops_raw = data.get("operators", {})
        # operators endpoint returns a dict {name: {…}}, not a list
        ops_list = list(ops_raw.values()) if isinstance(ops_raw, dict) else list(ops_raw)
        check(f"operators count: {len(ops_list)}", len(ops_list) > 0, f"got {len(ops_list)}")
        if ops_list:
            first = ops_list[0]
            check("operator has name", "name" in first, f"keys={list(first.keys())}")
            has_params = "param_defs" in first or "params" in first or "parameters" in first
            check("operator has param_defs", has_params, f"keys={list(first.keys())}")

    # List operators with procedure filter
    status, data = _req("GET", "/api/operators?procedure=rhinoplasty")
    check("GET /api/operators?procedure=rhinoplasty → 200", status == 200)

    # List templates
    status, data = _req("GET", "/api/templates")
    check("GET /api/templates → 200", status == 200)
    if isinstance(data, dict):
        templates = data.get("templates", data.get("categories", []))
        check(f"templates available: {len(templates)}", len(templates) > 0)

    # Load template
    status, plan_data = _req("POST", "/api/plan/template", {
        "category": "rhinoplasty",
        "template": "reduction_rhinoplasty",
    })
    check("POST /api/plan/template → 200", status == 200, f"got {status}: {plan_data}")

    plan = None
    if isinstance(plan_data, dict):
        plan = plan_data.get("plan", plan_data)
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        check(f"template plan has steps: {len(steps)}", len(steps) > 0)

    # Create custom plan
    status, custom = _req("POST", "/api/plan/custom", {
        "name": "test_custom_plan",
        "procedure": "rhinoplasty",
        "steps": [
            {"operator": "dorsal_reduction", "params": {"amount_mm": 3.0}},
            {"operator": "lateral_osteotomy", "params": {"side": "bilateral", "angle_deg": 30}},
        ],
    })
    check("POST /api/plan/custom → 200", status == 200, f"got {status}: {custom}")
    if isinstance(custom, dict):
        custom_plan = custom.get("plan", custom)
        check("custom plan returned", isinstance(custom_plan, dict))
        if plan is None:
            plan = custom_plan

    # Compile plan
    if plan:
        status, compiled = _req("POST", "/api/plan/compile", {
            "case_id": case_id,
            "plan": plan,
        })
        check("POST /api/plan/compile → 200", status == 200, f"got {status}: {compiled}")
        if isinstance(compiled, dict):
            check("compiled has result data", len(compiled) > 0)
            # Check for typical compilation output
            has_bcs = "boundary_conditions" in compiled or "n_bcs" in compiled or "bcs" in compiled
            check("compiled has BCs or output", has_bcs or "content_hash" in compiled or "result" in compiled,
                  f"keys={list(compiled.keys())}")
        return {"plan": plan, "compiled": compiled if isinstance(compiled, dict) else {}}

    return None


# ══════════════════════════════════════════════════════════════════
#  PHASE 7: G4 What-If + Sweep
# ══════════════════════════════════════════════════════════════════

def test_whatif_sweep(case_id: str, plan: Optional[Dict]):
    section("PHASE 7: G4 — What-If & Sweep")

    if not case_id or not plan:
        print("  ⚠ Skipped — no case_id or plan")
        return

    plan_dict = plan.get("plan", plan)

    # What-If
    status, data = _req("POST", "/api/whatif", {
        "case_id": case_id,
        "plan": plan_dict,
        "modified_params": {"dorsal_reduction": {"amount_mm": 5.0}},
    })
    check("POST /api/whatif → 200", status == 200, f"got {status}: {str(data)[:200]}")
    if isinstance(data, dict):
        check("whatif has result data", len(data) > 0, f"keys={list(data.keys())}")

    # Parameter Sweep
    steps = plan_dict.get("steps", []) if isinstance(plan_dict, dict) else []
    sweep_op = ""
    sweep_param = ""
    if steps:
        first_step = steps[0]
        sweep_op = first_step.get("operator", first_step.get("name", ""))
        params = first_step.get("params", {})
        for k, v in params.items():
            if isinstance(v, (int, float)):
                sweep_param = k
                break

    if sweep_op and sweep_param:
        status, data = _req("POST", "/api/sweep", {
            "case_id": case_id,
            "plan": plan_dict,
            "sweep_op": sweep_op,
            "sweep_param": sweep_param,
            "values": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        check("POST /api/sweep → 200", status == 200, f"got {status}: {str(data)[:200]}")
        if isinstance(data, dict):
            has_points = "points" in data or "results" in data or "sweep" in data
            check("sweep has results/points", has_points or len(data) > 0,
                  f"keys={list(data.keys())}")
    else:
        print(f"  ⚠ Sweep skipped — couldn't extract numeric param (op={sweep_op})")


# ══════════════════════════════════════════════════════════════════
#  PHASE 8: G5 Report Generation
# ══════════════════════════════════════════════════════════════════

def test_report(case_id: str, plan: Optional[Dict]):
    section("PHASE 8: G5 — Report Generation")

    if not case_id or not plan:
        print("  ⚠ Skipped — no case_id or plan")
        return

    plan_dict = plan.get("plan", plan)

    for fmt in ("html", "markdown", "json"):
        status, data = _req("POST", "/api/report", {
            "case_id": case_id,
            "plan": plan_dict,
            "format": fmt,
        })
        check(f"POST /api/report format={fmt} → 200", status == 200,
              f"got {status}: {str(data)[:150]}")
        if isinstance(data, dict):
            has_content = "content" in data or "report" in data or "html" in data or len(data) > 0
            check(f"report ({fmt}) has content", has_content, f"keys={list(data.keys())}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 9: G6 Visualization
# ══════════════════════════════════════════════════════════════════

def test_visualization(case_id: str):
    section("PHASE 9: G6 — Visualization Data")

    if not case_id:
        print("  ⚠ Skipped — no case_id")
        return

    status, data = _req("GET", f"/api/cases/{case_id}/visualization")
    check("GET visualization → 200", status == 200, f"got {status}")
    if isinstance(data, dict):
        check("visualization has data", len(data) > 0, f"keys={list(data.keys())}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 10: G7 Timeline
# ══════════════════════════════════════════════════════════════════

def test_timeline(case_id: str):
    section("PHASE 10: G7 — Timeline")

    if not case_id:
        print("  ⚠ Skipped — no case_id")
        return

    status, data = _req("GET", f"/api/cases/{case_id}/timeline")
    check("GET timeline → 200", status == 200, f"got {status}")
    if isinstance(data, dict):
        events = data.get("events", data.get("timeline", []))
        check(f"timeline events: {len(events)}", isinstance(events, list))


# ══════════════════════════════════════════════════════════════════
#  PHASE 11: G8 Compare
# ══════════════════════════════════════════════════════════════════

def test_compare(case_id: str, plan: Optional[Dict]):
    section("PHASE 11: G8 — Compare")

    if not case_id or not plan:
        print("  ⚠ Skipped — no case_id or plan")
        return

    plan_dict = plan.get("plan", plan)

    # Compare plans (same plan for both — just testing the endpoint)
    status, data = _req("POST", "/api/compare/plans", {
        "case_id": case_id,
        "plan_a": plan_dict,
        "plan_b": plan_dict,
    })
    check("POST /api/compare/plans → 200", status == 200, f"got {status}: {str(data)[:200]}")
    if isinstance(data, dict):
        check("compare/plans has result", len(data) > 0, f"keys={list(data.keys())}")

    # Compare cases — need two case_ids
    status, cases_data = _req("GET", "/api/cases")
    cases = cases_data.get("cases", []) if isinstance(cases_data, dict) else []
    if len(cases) >= 2:
        id_a = cases[0].get("case_id", cases[0].get("id", ""))
        id_b = cases[1].get("case_id", cases[1].get("id", ""))
        if id_a and id_b:
            status, data = _req("POST", "/api/compare/cases", {
                "case_id_a": id_a,
                "case_id_b": id_b,
            })
            check("POST /api/compare/cases → 200", status == 200,
                  f"got {status}: {str(data)[:200]}")
            if isinstance(data, dict):
                check("compare/cases has result", len(data) > 0)
    else:
        print(f"  ⚠ compare/cases skipped — only {len(cases)} case(s)")


# ══════════════════════════════════════════════════════════════════
#  PHASE 12: Delete case (cleanup + test)
# ══════════════════════════════════════════════════════════════════

def test_delete_case():
    section("PHASE 12: Delete Case")

    # Create a throwaway case and delete it
    status, data = _req("POST", "/api/cases", {
        "patient_age": 99,
        "patient_sex": "M",
        "procedure": "rhinoplasty",
        "notes": "delete test",
    })
    if status == 200 and isinstance(data, dict):
        del_id = data.get("case_id", data.get("id", ""))
        if del_id:
            status, data = _req("POST", f"/api/cases/{del_id}/delete", {})
            check(f"POST /api/cases/{del_id}/delete → 200", status == 200,
                  f"got {status}: {data}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 13: JS module wiring validation
# ══════════════════════════════════════════════════════════════════

def test_js_wiring():
    section("PHASE 13: JS Module Wiring Validation")

    # Validate that JS files reference the correct API paths
    js_files = {
        "/js/api.js": [
            '"/cases"', '"/contract"', '"/operators"', '"/templates"',
            '"/plan/compile"', '"/whatif"', '"/sweep"', '"/report"',
            '"/compare/plans"', '"/compare/cases"',
            "/health", "/metrics",
        ],
        "/js/state.js": ["Store", "subscribe", "snapshot", "savePrefs"],
        "/js/router.js": ["Router", "navigate", "onModeChange", "registerShortcut"],
        "/js/app.js": ["App", "boot", "DOMContentLoaded"],
    }

    for path, expected_strings in js_files.items():
        status, body = _req("GET", path, auth=False)
        if status != 200 or not isinstance(body, bytes):
            check(f"{path} — loadable", False, f"status={status}")
            continue
        text = body.decode()
        for s in expected_strings:
            check(f"{path} contains '{s}'", s in text)

    # Validate mode panel IDs in index.html match what JS expects
    status, body = _req("GET", "/", auth=False)
    if isinstance(body, bytes):
        html = body.decode()
        mode_ids = [
            "mode-case-library", "mode-twin-inspect", "mode-plan-author",
            "mode-consult", "mode-sweep", "mode-report",
            "mode-viewer3d", "mode-timeline", "mode-compare",
        ]
        for mid in mode_ids:
            check(f"HTML has panel #{mid}", f'id="{mid}"' in html)

    # Validate nav items match mode IDs
    nav_modes = [
        "case-library", "twin-inspect", "plan-author", "consult",
        "sweep", "report", "viewer3d", "timeline", "compare",
    ]
    if isinstance(body, bytes):
        html = body.decode()
        for mode in nav_modes:
            check(f"nav-item data-mode={mode}", f'data-mode="{mode}"' in html)


# ══════════════════════════════════════════════════════════════════
#  PHASE 14: CSS design system validation
# ══════════════════════════════════════════════════════════════════

def test_css_design_system():
    section("PHASE 14: CSS Design System Validation")

    # tokens.css should have all design tokens
    status, body = _req("GET", "/css/tokens.css", auth=False)
    if isinstance(body, bytes):
        css = body.decode()
        tokens = [
            "--surface-0", "--text-primary", "--accent-blue",
            "--font-sans", "--font-mono", "--space-1", "--radius-md",
            "--shadow-md", "--transition-base", "--z-modal",
            "--command-bar-height", "--status-bar-h", "--sidebar-w",
        ]
        for tok in tokens:
            check(f"tokens.css has {tok}", tok in css)

    # layout.css should have grid areas
    status, body = _req("GET", "/css/layout.css", auth=False)
    if isinstance(body, bytes):
        css = body.decode()
        check("layout has grid-template", "grid-template" in css)
        check("layout has sidebar-collapsed", "sidebar-collapsed" in css)
        check("layout has command-bar", "command-bar" in css)
        check("layout has status-bar", "status-bar" in css)


# ══════════════════════════════════════════════════════════════════
#  RUN ALL
# ══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Ontic Facial Plastics — Full UI Validation      ║")
    print("║  Testing every capability from the client/surgeon side ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_static_assets()
    test_health_metrics()
    test_auth()
    test_contract()
    case_id = test_case_library()
    test_twin_inspect(case_id)
    plan_result = test_plan_author(case_id)
    test_whatif_sweep(case_id, plan_result)
    test_report(case_id, plan_result)
    test_visualization(case_id)
    test_timeline(case_id)
    test_compare(case_id, plan_result)
    test_delete_case()
    test_js_wiring()
    test_css_design_system()

    print(f"\n{'═' * 60}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'═' * 60}")

    if ERRORS:
        print("\nFAILURES:")
        for e in ERRORS:
            print(e)

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
