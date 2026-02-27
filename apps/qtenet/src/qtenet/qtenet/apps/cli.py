from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qtenet", description="QTenet CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inventory", help="Show where the generated inventory lives")
    sub.add_parser("doctor", help="Validate environment + capability wiring")

    # Placeholders for future wiring
    sub.add_parser("inspect", help="Inspect an artifact")
    sub.add_parser("compress", help="Compress input into QTT container (facade)")
    sub.add_parser("query", help="Point query")
    sub.add_parser("reconstruct", help="Reconstruct (dense escape hatch)")
    sub.add_parser("benchmark", help="Run benchmarks")
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "inventory":
        print("inventory/qtt_repo_index.json")
        return 0

    if args.cmd == "doctor":
        # Minimal enterprise support command (no deps beyond stdlib + optional torch)
        ok = True
        issues: list[str] = []
        try:
            import torch  # type: ignore

            _ = torch.__version__
            cuda = bool(torch.cuda.is_available())
        except Exception as e:  # noqa: BLE001
            ok = False
            cuda = False
            issues.append(f"torch import failed: {e}")

        inv_path = "inventory/qtt_repo_index.json"
        try:
            import os

            if not os.path.exists(inv_path):
                ok = False
                issues.append(f"missing inventory: {inv_path}")
        except Exception as e:  # noqa: BLE001
            ok = False
            issues.append(f"inventory check failed: {e}")

        print("QTeneT doctor")
        print(f"- torch: {'ok' if not issues or 'torch' not in ' '.join(issues) else 'fail'}")
        print(f"- cuda: {'available' if cuda else 'not-available'}")
        print(f"- inventory: {'present' if 'missing inventory' not in ' '.join(issues) else 'missing'}")
        if issues:
            print("Issues:")
            for it in issues:
                print(f"- {it}")
        return 0 if ok else 3

    raise SystemExit(f"Command '{args.cmd}' is a stub; wire it to implementations.")


if __name__ == "__main__":
    raise SystemExit(main())
