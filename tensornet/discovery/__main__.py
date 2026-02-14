#!/usr/bin/env python3
"""
Autonomous Discovery Engine CLI

Usage:
    python -m tensornet.discovery --help
    python -m tensornet.discovery discover --domain defi --demo
    python -m tensornet.discovery test
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch


def cmd_discover(args):
    """Run discovery pipeline."""
    from .engine_v2 import DiscoveryEngineV2
    from .pipelines.defi_pipeline import DeFiDiscoveryPipeline
    
    print("=" * 60)
    print("AUTONOMOUS DISCOVERY ENGINE")
    print("=" * 60)
    print(f"Domain: {args.domain}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    if args.domain == "defi":
        if args.demo:
            # Demo mode with synthetic data
            print("[Demo Mode] Using synthetic DeFi data...")
            pipeline = DeFiDiscoveryPipeline()
            
            # Simulate pool with anomaly
            normal_swaps = [{"amount0": i * 100, "tick": 1000 + i} for i in range(50)]
            anomalous_swaps = normal_swaps + [{"amount0": 1000000, "tick": 5000}]
            
            result = pipeline.analyze_pool(
                pool_address="0xDemoPool",
                swap_events=anomalous_swaps,
                liquidity_events=[{"liquidity": 1000}],
                historical_swaps=normal_swaps,
            )
            
            print(f"\n[Results]")
            print(f"  Findings: {len(result.findings)}")
            print(f"  Stages: {len(result.stages)}")
            print(f"  Time: {result.total_time*1000:.1f}ms")
            print()
            
            for i, f in enumerate(result.findings, 1):
                print(f"  {i}. [{f.severity}] {f.primitive}")
                print(f"     {f.summary}")
            
            print()
            print(f"Attestation: {result.attestation_hash}")
            
            if args.output:
                # Generate report
                report = pipeline.generate_immunefi_report(result, "Demo Protocol")
                Path(args.output).write_text(report)
                print(f"\nReport written to: {args.output}")
        
        elif args.input:
            # Load data from file
            print(f"[File Mode] Loading from {args.input}...")
            data = json.loads(Path(args.input).read_text())
            
            pipeline = DeFiDiscoveryPipeline()
            result = pipeline.analyze_pool(
                pool_address=data.get("pool_address", "unknown"),
                swap_events=data.get("swap_events", []),
                liquidity_events=data.get("liquidity_events", []),
                historical_swaps=data.get("historical_swaps"),
            )
            
            print(f"\nFindings: {len(result.findings)}")
            for f in result.findings:
                print(f"  [{f.severity}] {f.primitive}: {f.summary}")
            
            if args.output:
                report = pipeline.generate_immunefi_report(
                    result, 
                    data.get("protocol_name", "Unknown Protocol")
                )
                Path(args.output).write_text(report)
                print(f"\nReport: {args.output}")
        
        else:
            print("Error: Specify --demo or --input FILE")
            return 1
    
    elif args.domain == "plasma":
        # Fusion plasma mode
        from .pipelines.plasma_pipeline import PlasmaDiscoveryPipeline, run_demo as plasma_demo
        
        if args.demo:
            print("[Demo Mode] Running synthetic plasma shot analysis...")
            result = plasma_demo()
            
            if args.output:
                pipeline = PlasmaDiscoveryPipeline()
                report = pipeline.generate_report(result, format="markdown")
                Path(args.output).write_text(report)
                print(f"\nReport written to: {args.output}")
        
        elif args.input:
            print(f"[File Mode] Loading from {args.input}...")
            data = json.loads(Path(args.input).read_text())
            
            from .ingest.plasma import PlasmaShot, PlasmaProfile
            
            shot = PlasmaShot(
                shot_id=data.get("shot_id", "unknown"),
                device=data.get("device", "unknown"),
                plasma_current_kA=data.get("plasma_current_kA", 0),
                magnetic_field_T=data.get("magnetic_field_T", 0),
                electron_density_m3=data.get("electron_density_m3", 0),
                electron_temp_keV=data.get("electron_temp_keV", 0),
                ion_temp_keV=data.get("ion_temp_keV", 0),
                stored_energy_MJ=data.get("stored_energy_MJ", 0),
                confinement_mode=data.get("confinement_mode", "L-mode"),
                elm_events=data.get("elm_events", []),
            )
            
            pipeline = PlasmaDiscoveryPipeline(verbose=True)
            result = pipeline.analyze_shot(shot)
            
            print(f"\nFindings: {len(result.findings)}")
            for f in result.findings:
                print(f"  [{f.get('severity', '')}] {f.get('primitive', '')}: {f.get('description', '')}")
            
            if args.output:
                report = pipeline.generate_report(result, format="markdown")
                Path(args.output).write_text(report)
                print(f"\nReport: {args.output}")
        
        else:
            print("Error: Specify --demo or --input FILE")
            return 1
    
    elif args.domain == "molecular":
        # Molecular/drug discovery mode
        from .pipelines.molecular_pipeline import MolecularDiscoveryPipeline, run_demo as molecular_demo
        from .ingest.molecular import MolecularIngester, create_synthetic_protein
        
        if args.demo:
            print("[Demo Mode] Running synthetic protein analysis...")
            result = molecular_demo()
            
            if args.output:
                pipeline = MolecularDiscoveryPipeline()
                report = pipeline.generate_report(result, target_name="Demo Kinase")
                Path(args.output).write_text(report)
                print(f"\nReport written to: {args.output}")
        
        elif args.input:
            print(f"[File Mode] Loading PDB from {args.input}...")
            pdb_content = Path(args.input).read_text()
            
            ingester = MolecularIngester()
            structure = ingester.from_pdb_string(pdb_content, Path(args.input).stem)
            
            pipeline = MolecularDiscoveryPipeline()
            result = pipeline.analyze_structure(structure, verbose=True)
            
            print(f"\nFindings: {len(result.findings)}")
            for f in result.findings:
                print(f"  [{f.severity}] {f.primitive}: {f.summary}")
            
            print(f"\nHypotheses: {len(result.hypotheses)}")
            for h in result.hypotheses:
                print(f"  → {h.title} ({h.confidence*100:.0f}%)")
            
            if args.output:
                report = pipeline.generate_report(result, target_name=Path(args.input).stem)
                Path(args.output).write_text(report)
                print(f"\nReport: {args.output}")
        
        else:
            print("Error: Specify --demo or --input FILE (PDB file)")
            return 1
    
    elif args.domain == "markets":
        # Financial markets mode
        from .pipelines.markets_pipeline import MarketsDiscoveryPipeline, run_demo as markets_demo
        from .ingest.markets import MarketsIngester, create_synthetic_flash_crash
        
        if args.demo:
            print("[Demo Mode] Running flash crash analysis...")
            result = markets_demo()
            
            if args.output:
                pipeline = MarketsDiscoveryPipeline()
                report = pipeline.generate_report(result)
                Path(args.output).write_text(report)
                print(f"\nReport written to: {args.output}")
        
        elif args.input:
            print(f"[File Mode] Loading from {args.input}...")
            ingester = MarketsIngester()
            snapshot = ingester.from_json(Path(args.input), symbol=Path(args.input).stem)
            
            pipeline = MarketsDiscoveryPipeline()
            result = pipeline.analyze_market(snapshot, verbose=True)
            
            print(f"\nFindings: {len(result.findings)}")
            for f in result.findings:
                print(f"  [{f.severity}] {f.primitive}: {f.summary}")
            
            print(f"\nFlash Crash Detected: {result.flash_crash_detected}")
            print(f"Hypotheses: {len(result.hypotheses)}")
            for h in result.hypotheses:
                print(f"  → {h.title} ({h.confidence*100:.0f}%)")
            
            if args.output:
                report = pipeline.generate_report(result)
                Path(args.output).write_text(report)
                print(f"\nReport: {args.output}")
        
        else:
            print("Error: Specify --demo or --input FILE (JSON market data)")
            return 1
    
    elif args.domain == "raw":
        # Direct engine mode
        print("[Raw Mode] Running core engine...")
        
        if args.input:
            data = torch.load(args.input, weights_only=True)
        else:
            print("Using random data (specify --input for real data)")
            torch.manual_seed(42)
            data = torch.randn(2, 1024)
        
        engine = DiscoveryEngineV2(grid_bits=args.grid_bits or 12)
        result = engine.discover(data)
        
        print(f"\nFindings: {len(result.findings)}")
        for f in result.findings:
            print(f"  [{f.severity}] {f.primitive}: {f.summary}")
        
        print(f"\nAttestation: {result.attestation_hash}")
    
    else:
        print(f"Error: Unknown domain '{args.domain}'")
        print("Available: defi, plasma, molecular, markets, raw")
        return 1
    
    print("\n" + "=" * 60)
    return 0


def cmd_test(args):
    """Run proof tests."""
    import subprocess
    
    print("=" * 60)
    print("RUNNING PROOF TESTS")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent.parent
    
    tests = [
        ("Discovery Engine", project_root / "proofs" / "proof_discovery_engine.py"),
        ("DeFi Pipeline", project_root / "proofs" / "proof_defi_pipeline.py"),
        ("Plasma Pipeline", project_root / "proofs" / "proof_plasma_pipeline.py"),
        ("Molecular Pipeline", project_root / "proofs" / "proof_molecular_pipeline.py"),
        ("Markets Pipeline", project_root / "proofs" / "proof_markets_pipeline.py"),
        ("Live Data Connectors", project_root / "proofs" / "proof_live_data.py"),
        ("Unification API", project_root / "proofs" / "proof_api.py"),
        ("Production Hardening", project_root / "proofs" / "proof_production.py"),
    ]
    
    all_passed = True
    
    for name, path in tests:
        if path.exists():
            print(f"\n[{name}]")
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=not args.verbose,
            )
            if result.returncode != 0:
                all_passed = False
                if not args.verbose:
                    print(f"  FAILED (run with --verbose for details)")
        else:
            print(f"\n[{name}] - SKIPPED (file not found)")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL PROOF TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


def cmd_info(args):
    """Show system information."""
    print("=" * 60)
    print("AUTONOMOUS DISCOVERY ENGINE")
    print("=" * 60)
    print()
    print("Version: 1.8.0")
    print("Status: Full 7-Stage QTT-Native Pipeline + Phase 7 (Production Hardening)")
    print()
    print("Available Domains:")
    print("  - defi      : DeFi protocol analysis (Uniswap, Aave style)")
    print("  - plasma    : Fusion plasma analysis (tokamak/stellarator)")
    print("  - molecular : Drug discovery (protein-ligand, binding sites)")
    print("  - markets   : Financial markets (flash crash, regime detection)")
    print("  - raw       : Direct tensor analysis")
    print()
    print("Pipeline Stages (7 QTT-Native Genesis Primitives):")
    print("  1. OT     : Optimal Transport (distribution drift)")
    print("  2. SGW    : Spectral Graph Wavelets (multi-scale)")
    print("  3. RMT    : Random Matrix Theory (chaos vs integrability)")
    print("  4. TG     : Tropical Geometry (bottlenecks/critical paths)")
    print("  5. RKHS   : Kernel Methods (MMD anomaly)")
    print("  6. PH     : Persistent Homology (topology)")
    print("  7. GA     : Geometric Algebra (invariants)")
    print("  +  HYP    : Hypothesis Generator (synthesis)")
    print()
    print("Usage:")
    print("  python -m tensornet.discovery discover --domain defi --demo")
    print("  python -m tensornet.discovery discover --domain plasma --demo")
    print("  python -m tensornet.discovery discover --domain molecular --demo")
    print("  python -m tensornet.discovery discover --domain markets --demo")
    print("  python -m tensornet.discovery live --mode historical --event flash-crash-2010")
    print("  python -m tensornet.discovery live --mode replay --event gme-2021")
    print("  python -m tensornet.discovery live --mode stream --duration 30")
    print("  python -m tensornet.discovery serve --port 8000")
    print("  python -m tensornet.discovery test")
    print()
    
    # Check imports
    print("Components:")
    try:
        from . import DiscoveryEngine
        print("  ✓ DiscoveryEngine")
    except ImportError as e:
        print(f"  ✗ DiscoveryEngine: {e}")
    
    try:
        from . import DeFiDiscoveryPipeline
        print("  ✓ DeFiDiscoveryPipeline")
    except ImportError as e:
        print(f"  ✗ DeFiDiscoveryPipeline: {e}")
    
    try:
        from . import HypothesisGenerator
        print("  ✓ HypothesisGenerator")
    except ImportError as e:
        print(f"  ✗ HypothesisGenerator: {e}")
    
    try:
        from . import PlasmaDiscoveryPipeline
        print("  ✓ PlasmaDiscoveryPipeline")
    except ImportError as e:
        print(f"  ✗ PlasmaDiscoveryPipeline: {e}")
    
    try:
        from . import MolecularDiscoveryPipeline
        print("  ✓ MolecularDiscoveryPipeline")
    except ImportError as e:
        print(f"  ✗ MolecularDiscoveryPipeline: {e}")
    
    try:
        from . import MarketsDiscoveryPipeline
        print("  ✓ MarketsDiscoveryPipeline")
    except ImportError as e:
        print(f"  ✗ MarketsDiscoveryPipeline: {e}")
    
    try:
        from .connectors import StreamingPipeline, HistoricalDataLoader
        print("  ✓ LiveDataConnectors")
    except ImportError as e:
        print(f"  ✗ LiveDataConnectors: {e}")
    
    try:
        from .api import DiscoveryAPIServer, GPUBackend
        print("  ✓ UnificationAPI")
    except ImportError as e:
        print(f"  ✗ UnificationAPI: {e}")
    
    try:
        from .api.gpu import gpu_available, get_gpu_info
        info = get_gpu_info()
        if info["available"]:
            print(f"  ✓ GPU Acceleration (CUDA {info['cuda_version']})")
        else:
            print("  ○ GPU Acceleration (CPU fallback)")
    except ImportError as e:
        print(f"  ✗ GPU Backend: {e}")
    
    try:
        from .production import (
            CircuitBreaker, RateLimiter, Bulkhead,
            MetricsCollector, HealthChecker, Tracer,
            APIKeyAuth, AuditLogger,
            CacheManager, ConnectionPool, PerformanceProfiler,
        )
        print("  ✓ ProductionHardening")
    except ImportError as e:
        print(f"  ✗ ProductionHardening: {e}")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch")
    
    print()
    print("=" * 60)
    return 0


def cmd_serve(args):
    """Start the API server."""
    try:
        from .api.server import DiscoveryAPIServer, ServerConfig, FASTAPI_AVAILABLE
    except ImportError as e:
        print(f"Error importing API server: {e}")
        print("Install with: pip install fastapi uvicorn")
        return 1
    
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
        return 1
    
    print("=" * 60)
    print("AUTONOMOUS DISCOVERY ENGINE - API SERVER")
    print("=" * 60)
    print()
    print(f"Starting server on {args.host}:{args.port}")
    print(f"GPU: {'disabled' if args.no_gpu else 'enabled'}")
    print(f"Distributed: {'enabled' if args.distributed else 'disabled'}")
    print(f"Debug: {'enabled' if args.debug else 'disabled'}")
    print()
    print("Endpoints:")
    print(f"  - Health:    http://{args.host}:{args.port}/health")
    print(f"  - Info:      http://{args.host}:{args.port}/info")
    print(f"  - Discover:  http://{args.host}:{args.port}/discover")
    print(f"  - Live:      http://{args.host}:{args.port}/live")
    print(f"  - Stream:    ws://{args.host}:{args.port}/stream/{{session_id}}")
    print(f"  - Docs:      http://{args.host}:{args.port}/docs")
    print()
    
    config = ServerConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        enable_gpu=not args.no_gpu,
        enable_distributed=args.distributed,
    )
    
    server = DiscoveryAPIServer(config)
    server.run()
    
    return 0


def cmd_live(args):
    """Run live/historical data analysis."""
    from .connectors.historical import HistoricalDataLoader
    from .connectors.streaming import StreamingPipeline, ReplayPipeline, StreamingConfig
    from .pipelines.markets_pipeline import MarketsDiscoveryPipeline
    
    print("=" * 60)
    print("AUTONOMOUS DISCOVERY ENGINE - LIVE DATA")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    if args.mode == "historical":
        # Load and analyze historical event
        loader = HistoricalDataLoader()
        
        if args.event == "flash-crash-2010":
            event = loader.load_2010_flash_crash()
        elif args.event == "gme-2021":
            event = loader.load_2021_gme_squeeze()
        elif args.event == "lehman-2008":
            event = loader.load_2008_lehman_week()
        else:
            print(f"Unknown event: {args.event}")
            return 1
        
        print(f"Event: {event.name}")
        print(f"Symbol: {event.symbol}")
        print(f"Duration: {event.duration_minutes} minutes")
        print(f"Bars: {len(event.bars)}")
        print(f"Peak Drawdown: {event.peak_drawdown*100:.2f}%")
        print()
        
        # Run pipeline analysis
        print("[Analyzing with Markets Pipeline...]")
        snapshot = event.to_market_snapshot()
        pipeline = MarketsDiscoveryPipeline()
        result = pipeline.analyze_market(snapshot, verbose=True)
        
        print()
        print("=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Findings: {len(result.findings)}")
        print(f"Flash Crash Detected: {result.flash_crash_detected}")
        if result.flash_crash_idx is not None:
            print(f"  At bar: {result.flash_crash_idx}")
        print(f"Hypotheses: {len(result.hypotheses)}")
        print()
        
        for f in result.findings:
            print(f"  [{f.severity}] {f.primitive}: {f.summary}")
        
        print()
        for h in result.hypotheses:
            print(f"  → {h.title} ({h.confidence*100:.0f}%)")
        
        if args.output:
            report = pipeline.generate_report(result)
            Path(args.output).write_text(report)
            print(f"\nReport: {args.output}")
    
    elif args.mode == "replay":
        # Replay historical event with streaming pipeline
        loader = HistoricalDataLoader()
        
        if args.event == "flash-crash-2010":
            event = loader.load_2010_flash_crash()
        elif args.event == "gme-2021":
            event = loader.load_2021_gme_squeeze()
        elif args.event == "lehman-2008":
            event = loader.load_2008_lehman_week()
        else:
            print(f"Unknown event: {args.event}")
            return 1
        
        print(f"Event: {event.name}")
        print(f"Bars: {len(event.bars)}")
        print()
        print("[Replaying with streaming pipeline...]")
        
        config = StreamingConfig(window_size=30, analysis_interval=5)
        replay = ReplayPipeline(config)
        
        results = replay.run(event, speed=1000, stop_on_alert=False)
        
        flash_alerts = [r for r in results if r.flash_crash_alert]
        regime_alerts = [r for r in results if r.regime_change_alert]
        
        print()
        print("=" * 60)
        print("REPLAY RESULTS")
        print("=" * 60)
        print(f"Analyses: {len(results)}")
        print(f"Flash Crash Alerts: {len(flash_alerts)}")
        print(f"Regime Change Alerts: {len(regime_alerts)}")
        
        if flash_alerts:
            first = flash_alerts[0]
            print(f"\nFirst Flash Crash Alert:")
            print(f"  Bar: {first.bar_count}")
            print(f"  Price: ${first.current_price:.2f}")
            for msg in first.alert_messages:
                print(f"  {msg}")
    
    elif args.mode == "stream":
        # Real-time simulated streaming
        print(f"Symbol: {args.symbol}")
        print(f"Duration: {args.duration} seconds")
        print()
        print("[Starting simulated stream...]")
        
        config = StreamingConfig(
            bar_interval_seconds=1,
            window_size=25,
            analysis_interval=5
        )
        
        pipeline = StreamingPipeline(config)
        
        alerts = []
        def on_alert(result):
            alerts.append(result)
            print(f"  ⚠️ ALERT at bar {result.bar_count}: {result.alert_messages}")
        
        pipeline.on_alert = on_alert
        pipeline.start_simulated(args.symbol, initial_price=50000, update_rate=50)
        
        start = time.time()
        results = []
        
        while time.time() - start < args.duration:
            result = pipeline.get_result(timeout=1.0)
            if result:
                results.append(result)
                print(f"  Bar {result.bar_count}: ${result.current_price:.2f}")
        
        pipeline.stop()
        
        stats = pipeline.get_statistics()
        print()
        print("=" * 60)
        print("STREAMING RESULTS")
        print("=" * 60)
        print(f"Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"Total Bars: {stats['total_bars']}")
        print(f"Analyses: {stats['total_analyses']}")
        print(f"Alerts: {stats['total_alerts']}")
        print(f"Bars/sec: {stats['bars_per_second']:.1f}")
    
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    print()
    print("=" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="tensornet.discovery",
        description="Autonomous Discovery Engine - Cross-Primitive Analysis",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # discover command
    discover_parser = subparsers.add_parser("discover", help="Run discovery")
    discover_parser.add_argument(
        "--domain", "-d",
        choices=["defi", "plasma", "molecular", "markets", "raw"],
        default="defi",
        help="Analysis domain",
    )
    discover_parser.add_argument(
        "--input", "-i",
        help="Input file (JSON for defi, .pt for raw)",
    )
    discover_parser.add_argument(
        "--output", "-o",
        help="Output file for report",
    )
    discover_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with demo data",
    )
    discover_parser.add_argument(
        "--grid-bits",
        type=int,
        default=12,
        help="Grid size as power of 2 (default: 12 = 4096)",
    )
    
    # test command
    test_parser = subparsers.add_parser("test", help="Run proof tests")
    test_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full test output",
    )
    
    # live command (Phase 5)
    live_parser = subparsers.add_parser("live", help="Live/historical data analysis")
    live_parser.add_argument(
        "--mode", "-m",
        choices=["replay", "stream", "historical"],
        default="historical",
        help="Analysis mode",
    )
    live_parser.add_argument(
        "--event", "-e",
        choices=["flash-crash-2010", "gme-2021", "lehman-2008"],
        default="flash-crash-2010",
        help="Historical event to analyze",
    )
    live_parser.add_argument(
        "--symbol", "-s",
        default="BTC-USD",
        help="Symbol for streaming mode",
    )
    live_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="Duration in seconds for streaming",
    )
    live_parser.add_argument(
        "--output", "-o",
        help="Output file for report",
    )
    
    # info command
    subparsers.add_parser("info", help="Show system info")
    
    # serve command (Phase 6)
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    serve_parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    serve_parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed mode",
    )
    serve_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    args = parser.parse_args()
    
    if args.command == "discover":
        return cmd_discover(args)
    elif args.command == "test":
        return cmd_test(args)
    elif args.command == "live":
        return cmd_live(args)
    elif args.command == "info":
        return cmd_info(args)
    elif args.command == "serve":
        return cmd_serve(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
