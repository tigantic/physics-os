"""
HyperTensor Facial Plastics — CLI Entry Point
===============================================

Provides command-line access to the full platform pipeline:
ingest, twin build, plan authoring/compilation, simulation,
report generation, UI server, and case library management.

Usage::

    python -m products.facial_plastics.cli <command> [options]

Commands:

    ingest      Import DICOM/photo/surface data into a case.
    twin        Build digital twin from ingested data.
    plan        Create or compile a surgical plan.
    simulate    Run multi-physics simulation.
    report      Generate clinical report.
    serve       Launch the interactive UI server.
    library     List, curate, or augment the case library.
    info        Print platform version and configuration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import textwrap
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast; heavy modules loaded only when needed.
# ---------------------------------------------------------------------------

def _import_core() -> Dict[str, Any]:
    """Import core types and config."""
    from products.facial_plastics.core.types import (
        Modality,
        ProcedureType,
        QualityLevel,
        generate_case_id,
    )
    from products.facial_plastics.core.config import PlatformConfig
    from products.facial_plastics.core.case_bundle import CaseBundle
    return {
        "Modality": Modality,
        "ProcedureType": ProcedureType,
        "QualityLevel": QualityLevel,
        "generate_case_id": generate_case_id,
        "PlatformConfig": PlatformConfig,
        "CaseBundle": CaseBundle,
    }


def _import_data() -> Dict[str, Any]:
    """Import data ingestion modules."""
    from products.facial_plastics.data import (
        CaseLibrary,
        CaseLibraryCurator,
        DicomIngester,
        PhotoIngester,
        SurfaceIngester,
        SyntheticAugmenter,
    )
    return {
        "CaseLibrary": CaseLibrary,
        "CaseLibraryCurator": CaseLibraryCurator,
        "DicomIngester": DicomIngester,
        "PhotoIngester": PhotoIngester,
        "SurfaceIngester": SurfaceIngester,
        "SyntheticAugmenter": SyntheticAugmenter,
    }


def _import_twin() -> Dict[str, Any]:
    """Import twin builder."""
    from products.facial_plastics.twin import TwinBuilder
    return {"TwinBuilder": TwinBuilder}


def _import_plan() -> Dict[str, Any]:
    """Import plan modules."""
    from products.facial_plastics.plan import PlanCompiler, SurgicalPlan
    from products.facial_plastics.plan.operators import (
        RHINOPLASTY_OPERATORS,
        RhinoplastyPlanBuilder,
        FACELIFT_OPERATORS,
        FaceliftPlanBuilder,
        BLEPHAROPLASTY_OPERATORS,
        BlepharoplastyPlanBuilder,
        FILLER_OPERATORS,
        FillerPlanBuilder,
    )
    return {
        "PlanCompiler": PlanCompiler,
        "SurgicalPlan": SurgicalPlan,
        "RHINOPLASTY_OPERATORS": RHINOPLASTY_OPERATORS,
        "RhinoplastyPlanBuilder": RhinoplastyPlanBuilder,
        "FACELIFT_OPERATORS": FACELIFT_OPERATORS,
        "FaceliftPlanBuilder": FaceliftPlanBuilder,
        "BLEPHAROPLASTY_OPERATORS": BLEPHAROPLASTY_OPERATORS,
        "BlepharoplastyPlanBuilder": BlepharoplastyPlanBuilder,
        "FILLER_OPERATORS": FILLER_OPERATORS,
        "FillerPlanBuilder": FillerPlanBuilder,
    }


def _import_sim() -> Dict[str, Any]:
    """Import simulation orchestrator."""
    from products.facial_plastics.sim import SimOrchestrator
    return {"SimOrchestrator": SimOrchestrator}


def _import_reports() -> Dict[str, Any]:
    """Import report builder."""
    from products.facial_plastics.reports import ReportBuilder
    return {"ReportBuilder": ReportBuilder}


# ---------------------------------------------------------------------------
# CLI: top-level
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser with all sub-commands."""
    parser = argparse.ArgumentParser(
        prog="facial_plastics",
        description="HyperTensor Facial Plastics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              facial_plastics ingest dicom /path/to/dicom --case-id abc123
              facial_plastics twin build abc123
              facial_plastics plan create --template rhinoplasty:dorsal_hump_reduction
              facial_plastics plan compile --case-id abc123 --plan plan.json
              facial_plastics simulate abc123 --plan plan.json
              facial_plastics report abc123 --format html --output report.html
              facial_plastics serve --port 8420
              facial_plastics library list
              facial_plastics info
        """),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to platform configuration JSON file.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for case data storage.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging output.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest imaging data into a case.")
    ingest_sub = p_ingest.add_subparsers(dest="ingest_type", required=True)

    p_dicom = ingest_sub.add_parser("dicom", help="Ingest DICOM series.")
    p_dicom.add_argument("path", type=str, help="Path to DICOM directory.")
    p_dicom.add_argument("--case-id", type=str, default=None, help="Existing case ID (auto-creates if omitted).")

    p_photo = ingest_sub.add_parser("photo", help="Ingest clinical photographs.")
    p_photo.add_argument("paths", nargs="+", type=str, help="Paths to photo files.")
    p_photo.add_argument("--case-id", type=str, default=None, help="Existing case ID.")

    p_surface = ingest_sub.add_parser("surface", help="Ingest surface scan (STL/OBJ/PLY).")
    p_surface.add_argument("path", type=str, help="Path to surface scan file.")
    p_surface.add_argument("--case-id", type=str, default=None, help="Existing case ID.")

    # ── twin ──────────────────────────────────────────────────────
    p_twin = sub.add_parser("twin", help="Build or inspect digital twin.")
    twin_sub = p_twin.add_subparsers(dest="twin_action", required=True)

    p_twin_build = twin_sub.add_parser("build", help="Build full digital twin from case data.")
    p_twin_build.add_argument("case_id", type=str, help="Case identifier.")
    p_twin_build.add_argument("--max-elements", type=int, default=None, help="Target mesh element count.")

    p_twin_inspect = twin_sub.add_parser("inspect", help="Print twin summary.")
    p_twin_inspect.add_argument("case_id", type=str, help="Case identifier.")

    # ── plan ──────────────────────────────────────────────────────
    p_plan = sub.add_parser("plan", help="Create, edit, or compile a surgical plan.")
    plan_sub = p_plan.add_subparsers(dest="plan_action", required=True)

    p_plan_create = plan_sub.add_parser("create", help="Create plan from template.")
    p_plan_create.add_argument(
        "--template", type=str, required=True,
        help="Template specification: category:name (e.g., rhinoplasty:dorsal_hump_reduction).",
    )
    p_plan_create.add_argument("--output", "-o", type=str, default=None, help="Output plan JSON file.")

    p_plan_compile = plan_sub.add_parser("compile", help="Compile plan against a case.")
    p_plan_compile.add_argument("--case-id", type=str, required=True, help="Case to compile against.")
    p_plan_compile.add_argument("--plan", type=str, required=True, help="Path to plan JSON file.")
    p_plan_compile.add_argument("--output", "-o", type=str, default=None, help="Output compiled result JSON.")

    p_plan_list = plan_sub.add_parser("list-operators", help="List available operators.")
    p_plan_list.add_argument("--category", type=str, default=None, help="Filter by category.")

    p_plan_tmpl = plan_sub.add_parser("list-templates", help="List available plan templates.")

    # ── simulate ──────────────────────────────────────────────────
    p_sim = sub.add_parser("simulate", help="Run simulation on a compiled plan.")
    p_sim.add_argument("case_id", type=str, help="Case identifier.")
    p_sim.add_argument("--plan", type=str, required=True, help="Path to plan JSON file.")
    p_sim.add_argument("--output", "-o", type=str, default=None, help="Output simulation result JSON.")
    p_sim.add_argument("--heal-days", type=int, default=None, help="Number of healing days to simulate.")

    # ── report ────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Generate clinical report.")
    p_report.add_argument("case_id", type=str, help="Case identifier.")
    p_report.add_argument("--plan", type=str, default=None, help="Path to plan JSON file.")
    p_report.add_argument("--format", type=str, choices=["html", "json", "markdown"], default="json")
    p_report.add_argument("--output", "-o", type=str, default=None, help="Output file path.")

    # ── serve ─────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Launch interactive UI server.")
    p_serve.add_argument("--port", type=int, default=8420, help="Server port (default: 8420).")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    p_serve.add_argument("--no-auth", action="store_true", help="Disable API-key auth (dev only).")
    p_serve.add_argument("--no-rate-limit", action="store_true", help="Disable rate limiting (dev only).")

    # ── library ───────────────────────────────────────────────────
    p_lib = sub.add_parser("library", help="Manage the case library.")
    lib_sub = p_lib.add_subparsers(dest="lib_action", required=True)

    p_lib_list = lib_sub.add_parser("list", help="List cases in library.")
    p_lib_list.add_argument("--procedure", type=str, default=None, help="Filter by procedure.")
    p_lib_list.add_argument("--quality", type=str, default=None, help="Filter by quality level.")
    p_lib_list.add_argument("--limit", type=int, default=50, help="Max cases to display.")

    p_lib_curate = lib_sub.add_parser("curate", help="Run library curation.")
    p_lib_augment = lib_sub.add_parser("augment", help="Generate synthetic augmentation data.")
    p_lib_augment.add_argument("--count", type=int, default=10, help="Number of synthetic samples.")
    p_lib_augment.add_argument("--procedure", type=str, default="rhinoplasty", help="Target procedure.")

    # ── info ──────────────────────────────────────────────────────
    sub.add_parser("info", help="Print platform version and configuration.")

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _load_config(args: argparse.Namespace) -> Any:
    """Load platform configuration from CLI args."""
    core = _import_core()
    config_path = args.config
    if config_path:
        path = pathlib.Path(config_path)
        if not path.exists():
            _error(f"Config file not found: {config_path}")
        with open(path) as f:
            data = json.load(f)
        cfg = core["PlatformConfig"](**data)
    else:
        kwargs: Dict[str, Any] = {}
        if args.data_root:
            kwargs["data_root"] = pathlib.Path(args.data_root)
        cfg = core["PlatformConfig"](**kwargs) if kwargs else core["PlatformConfig"]()
    return cfg


def _load_library(args: argparse.Namespace) -> Any:
    """Instantiate case library."""
    cfg = _load_config(args)
    data = _import_data()
    return data["CaseLibrary"](data_root=cfg.data_root)


def _load_case(library: Any, case_id: str) -> Any:
    """Load case from library or die."""
    bundle = library.load(case_id)
    if bundle is None:
        _error(f"Case not found: {case_id}")
    return bundle


def _error(msg: str) -> None:
    """Print error and exit."""
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _print_json(obj: Any) -> None:
    """Pretty-print a dict/list as JSON."""
    if hasattr(obj, "__dict__"):
        obj = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    print(json.dumps(obj, indent=2, default=str))


def _write_output(data: Any, output_path: Optional[str]) -> None:
    """Write JSON data to file or stdout."""
    text = json.dumps(data, indent=2, default=str)
    if output_path:
        pathlib.Path(output_path).write_text(text)
        print(f"Written to {output_path}")
    else:
        print(text)


# ── ingest ────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Handle ingest sub-commands."""
    core = _import_core()
    data_mod = _import_data()
    library = _load_library(args)

    case_id = args.case_id
    if not case_id:
        case_id = core["generate_case_id"]()
        print(f"Created case: {case_id}")

    bundle = library.load(case_id)
    if bundle is None:
        bundle = core["CaseBundle"](case_id=case_id)

    ingest_type = args.ingest_type

    if ingest_type == "dicom":
        ingester = data_mod["DicomIngester"]()
        dicom_path = pathlib.Path(args.path)
        if not dicom_path.exists():
            _error(f"DICOM path not found: {args.path}")
        result = ingester.ingest(dicom_path)
        bundle.dicom_metadata = result
        print(f"Ingested DICOM from {dicom_path}")

    elif ingest_type == "photo":
        ingester = data_mod["PhotoIngester"]()
        for photo_path_str in args.paths:
            photo_path = pathlib.Path(photo_path_str)
            if not photo_path.exists():
                print(f"WARNING: Photo not found: {photo_path_str}", file=sys.stderr)
                continue
            result = ingester.ingest(photo_path)
            bundle.photos.append(result)
            print(f"Ingested photo: {photo_path}")

    elif ingest_type == "surface":
        ingester = data_mod["SurfaceIngester"]()
        surface_path = pathlib.Path(args.path)
        if not surface_path.exists():
            _error(f"Surface file not found: {args.path}")
        result = ingester.ingest(surface_path)
        bundle.surface_scans.append(result)
        print(f"Ingested surface scan: {surface_path}")

    library.save(bundle)
    print(f"Case {case_id} saved.")


# ── twin ──────────────────────────────────────────────────────────

def cmd_twin(args: argparse.Namespace) -> None:
    """Handle twin sub-commands."""
    action = args.twin_action
    library = _load_library(args)

    if action == "build":
        bundle = _load_case(library, args.case_id)
        twin_mod = _import_twin()
        builder = twin_mod["TwinBuilder"]()
        kwargs: Dict[str, Any] = {}
        if args.max_elements is not None:
            kwargs["max_elements"] = args.max_elements
        twin = builder.build(bundle, **kwargs)
        library.save(bundle)
        print(f"Digital twin built for case {args.case_id}")
        print(f"  Structures segmented: {len(twin.segmentation.structures) if hasattr(twin, 'segmentation') and hasattr(twin.segmentation, 'structures') else 'N/A'}")
        mesh = getattr(twin, "volume_mesh", None)
        if mesh is not None:
            n_nodes = mesh.nodes.shape[0] if hasattr(mesh.nodes, "shape") else len(mesh.nodes)
            n_elems = mesh.elements.shape[0] if hasattr(mesh.elements, "shape") else len(mesh.elements)
            print(f"  Mesh: {n_nodes} nodes, {n_elems} elements")

    elif action == "inspect":
        bundle = _load_case(library, args.case_id)
        twin = getattr(bundle, "digital_twin", None)
        if twin is None:
            _error(f"No digital twin found for case {args.case_id}. Run 'twin build' first.")
        _print_json(twin)


# ── plan ──────────────────────────────────────────────────────────

_PLAN_BUILDERS: Dict[str, Any] = {}

def _get_plan_builders() -> Dict[str, Any]:
    """Lazily load all plan builders."""
    if not _PLAN_BUILDERS:
        plan = _import_plan()
        _PLAN_BUILDERS["rhinoplasty"] = plan["RhinoplastyPlanBuilder"]
        _PLAN_BUILDERS["facelift"] = plan["FaceliftPlanBuilder"]
        _PLAN_BUILDERS["blepharoplasty"] = plan["BlepharoplastyPlanBuilder"]
        _PLAN_BUILDERS["filler"] = plan["FillerPlanBuilder"]
    return _PLAN_BUILDERS


_ALL_OPERATORS: Dict[str, Any] = {}

def _get_all_operators() -> Dict[str, Any]:
    """Lazily load all operator registries."""
    if not _ALL_OPERATORS:
        plan = _import_plan()
        for registry_name in (
            "RHINOPLASTY_OPERATORS",
            "FACELIFT_OPERATORS",
            "BLEPHAROPLASTY_OPERATORS",
            "FILLER_OPERATORS",
        ):
            ops = plan[registry_name]
            _ALL_OPERATORS.update(ops)
    return _ALL_OPERATORS


def cmd_plan(args: argparse.Namespace) -> None:
    """Handle plan sub-commands."""
    action = args.plan_action

    if action == "create":
        template_spec = args.template
        if ":" not in template_spec:
            _error("Template must be 'category:name' (e.g., rhinoplasty:dorsal_hump_reduction).")
        category, template_name = template_spec.split(":", 1)
        builders = _get_plan_builders()
        if category not in builders:
            _error(f"Unknown category '{category}'. Available: {', '.join(sorted(builders.keys()))}")

        builder_cls = builders[category]
        builder = builder_cls()
        fn = getattr(builder, template_name, None)
        if fn is None or not callable(fn):
            methods = [m for m in dir(builder) if not m.startswith("_") and callable(getattr(builder, m))]
            _error(
                f"Unknown template '{template_name}' in '{category}'. "
                f"Available: {', '.join(sorted(methods))}"
            )
            return  # unreachable but satisfies mypy
        plan_obj = fn()
        plan_dict = {
            "name": plan_obj.name,
            "procedure": plan_obj.procedure.value if hasattr(plan_obj.procedure, "value") else str(plan_obj.procedure),
            "steps": [
                {
                    "name": op.name,
                    "operator": op.name,
                    "params": {k: (v.value if hasattr(v, "value") else v) for k, v in op.params.items()},
                    "description": op.description,
                }
                for op in (plan_obj.root.ops if hasattr(plan_obj.root, "ops") else [])
            ],
        }
        _write_output(plan_dict, args.output)

    elif action == "compile":
        library = _load_library(args)
        bundle = _load_case(library, args.case_id)
        plan_path = pathlib.Path(args.plan)
        if not plan_path.exists():
            _error(f"Plan file not found: {args.plan}")
        with open(plan_path) as f:
            plan_data = json.load(f)

        plan_mod = _import_plan()
        compiler = plan_mod["PlanCompiler"]()
        result = compiler.compile(plan_data, bundle)
        result_dict = {
            "status": "success" if not getattr(result, "errors", []) else "error",
            "errors": getattr(result, "errors", []),
            "warnings": getattr(result, "warnings", []),
        }
        _write_output(result_dict, args.output)

    elif action == "list-operators":
        ops = _get_all_operators()
        category_filter = getattr(args, "category", None)
        for name in sorted(ops.keys()):
            factory = ops[name]
            try:
                op = factory()
                cat = getattr(op, "category", "")
                if category_filter and cat != category_filter:
                    continue
                desc = getattr(op, "description", "")
                print(f"  {name:40s} {cat:20s} {desc}")
            except Exception:
                print(f"  {name:40s} (error instantiating)")

    elif action == "list-templates":
        builders = _get_plan_builders()
        for category, builder_cls in sorted(builders.items()):
            builder = builder_cls()
            methods = [m for m in dir(builder) if not m.startswith("_") and callable(getattr(builder, m))]
            print(f"\n{category}:")
            for m in sorted(methods):
                print(f"  {m}")


# ── simulate ──────────────────────────────────────────────────────

def cmd_simulate(args: argparse.Namespace) -> None:
    """Handle simulate command."""
    library = _load_library(args)
    bundle = _load_case(library, args.case_id)

    plan_path = pathlib.Path(args.plan)
    if not plan_path.exists():
        _error(f"Plan file not found: {args.plan}")
    with open(plan_path) as f:
        plan_data = json.load(f)

    sim_mod = _import_sim()
    orchestrator = sim_mod["SimOrchestrator"]()
    result = orchestrator.run(bundle, plan_data, heal_days=args.heal_days)

    result_dict = {
        "status": "completed",
        "case_id": args.case_id,
    }
    if hasattr(result, "__dict__"):
        for k, v in result.__dict__.items():
            if not k.startswith("_"):
                result_dict[k] = v

    _write_output(result_dict, args.output)


# ── report ────────────────────────────────────────────────────────

def cmd_report(args: argparse.Namespace) -> None:
    """Handle report command."""
    library = _load_library(args)
    bundle = _load_case(library, args.case_id)

    reports_mod = _import_reports()
    builder = reports_mod["ReportBuilder"]()

    plan_data = None
    if args.plan:
        plan_path = pathlib.Path(args.plan)
        if not plan_path.exists():
            _error(f"Plan file not found: {args.plan}")
        with open(plan_path) as f:
            plan_data = json.load(f)

    report = builder.generate(
        bundle,
        plan=plan_data,
        output_format=args.format,
    )
    content = getattr(report, "content", report) if not isinstance(report, str) else report
    if args.output:
        pathlib.Path(args.output).write_text(str(content))
        print(f"Report written to {args.output}")
    else:
        print(content)


# ── serve ─────────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> None:
    """Launch the UI server using the WSGI stack."""
    from wsgiref.simple_server import make_server

    from products.facial_plastics.ui.wsgi import create_app

    enable_auth = not getattr(args, "no_auth", False)
    enable_rate_limit = not getattr(args, "no_rate_limit", False)

    app = create_app(
        enable_auth=enable_auth,
        enable_rate_limit=enable_rate_limit,
    )

    mode = "production" if enable_auth else "development (auth disabled)"
    print(
        f"Starting HyperTensor Facial Plastics UI on "
        f"http://{args.host}:{args.port}  [{mode}]"
    )
    print("Press Ctrl+C to stop.")

    httpd = make_server(args.host, args.port, app)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


# ── library ───────────────────────────────────────────────────────

def cmd_library(args: argparse.Namespace) -> None:
    """Handle library sub-commands."""
    action = args.lib_action
    data_mod = _import_data()
    library = _load_library(args)

    if action == "list":
        cases = library.list_cases(
            procedure=args.procedure,
            quality=args.quality,
            limit=args.limit,
        )
        print(f"Cases ({len(cases)}):")
        for c in cases:
            case_id = getattr(c, "case_id", str(c))
            proc = getattr(c, "procedure", "")
            print(f"  {case_id}  {proc}")

    elif action == "curate":
        curator = data_mod["CaseLibraryCurator"](library)
        result = curator.curate()
        _print_json(result)

    elif action == "augment":
        augmenter = data_mod["SyntheticAugmenter"]()
        results = augmenter.augment(
            count=args.count,
            procedure=args.procedure,
        )
        print(f"Generated {len(results)} synthetic samples.")


# ── info ──────────────────────────────────────────────────────────

def cmd_info(args: argparse.Namespace) -> None:
    """Print platform info."""
    from products.facial_plastics import __version__
    cfg = _load_config(args)
    print(f"HyperTensor Facial Plastics v{__version__}")
    print(f"  Data root: {cfg.data_root}")
    print(f"  Operators: {len(_get_all_operators())}")
    builders = _get_plan_builders()
    total_templates = 0
    for _, builder_cls in builders.items():
        builder = builder_cls()
        methods = [m for m in dir(builder) if not m.startswith("_") and callable(getattr(builder, m))]
        total_templates += len(methods)
    print(f"  Templates: {total_templates}")
    print(f"  Procedures: {[p.value for p in _import_core()['ProcedureType']]}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_COMMAND_MAP = {
    "ingest": cmd_ingest,
    "twin": cmd_twin,
    "plan": cmd_plan,
    "simulate": cmd_simulate,
    "report": cmd_report,
    "serve": cmd_serve,
    "library": cmd_library,
    "info": cmd_info,
}


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")

    handler = _COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            _error(str(exc))


if __name__ == "__main__":
    main()
