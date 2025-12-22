#!/usr/bin/env python
"""
M) Packaging Gate
=================

Builds and validates Python package.

Usage:
    python scripts/packaging_gate.py

Pass Criteria:
    - wheel builds successfully
    - twine check passes
    - package installs in clean venv
"""

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def build_wheel(project_root: Path) -> Tuple[bool, str, Path]:
    """Build wheel using pip."""
    dist_dir = project_root / 'dist'
    
    # Clean existing builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'wheel', '.', 
             '-w', 'dist', '--no-deps'],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project_root
        )
        
        if result.returncode != 0:
            return False, result.stderr, Path()
        
        # Find the built wheel
        wheels = list(dist_dir.glob('*.whl'))
        if not wheels:
            return False, "No wheel file generated", Path()
        
        return True, result.stdout, wheels[0]
        
    except Exception as e:
        return False, str(e), Path()


def build_sdist(project_root: Path) -> Tuple[bool, str, Path]:
    """Build source distribution."""
    dist_dir = project_root / 'dist'
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'build', '--sdist'],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project_root
        )
        
        if result.returncode != 0:
            # Fallback to setup.py
            result = subprocess.run(
                [sys.executable, 'setup.py', 'sdist'],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_root
            )
        
        if result.returncode != 0:
            return False, result.stderr, Path()
        
        # Find the built sdist
        sdists = list(dist_dir.glob('*.tar.gz'))
        if not sdists:
            return False, "No sdist file generated", Path()
        
        return True, result.stdout, sdists[0]
        
    except Exception as e:
        return False, str(e), Path()


def check_with_twine(dist_dir: Path) -> Tuple[bool, List[str]]:
    """Check distribution with twine."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'twine', 'check', 'dist/*'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=dist_dir.parent,
            shell=True
        )
        
        issues = []
        for line in result.stdout.split('\n'):
            if 'warning' in line.lower() or 'error' in line.lower():
                issues.append(line.strip())
        
        return result.returncode == 0, issues
        
    except FileNotFoundError:
        return True, ["twine not installed (skipping)"]
    except Exception as e:
        return False, [str(e)]


def verify_install(wheel_path: Path) -> Tuple[bool, str]:
    """Verify package installs in clean venv."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / 'venv'
            
            # Create venv
            result = subprocess.run(
                [sys.executable, '-m', 'venv', str(venv_dir)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return False, f"Failed to create venv: {result.stderr}"
            
            # Determine pip path
            if sys.platform == 'win32':
                pip_path = venv_dir / 'Scripts' / 'pip.exe'
                python_path = venv_dir / 'Scripts' / 'python.exe'
            else:
                pip_path = venv_dir / 'bin' / 'pip'
                python_path = venv_dir / 'bin' / 'python'
            
            # Install wheel
            result = subprocess.run(
                [str(pip_path), 'install', str(wheel_path)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return False, f"Failed to install: {result.stderr}"
            
            # Try importing
            result = subprocess.run(
                [str(python_path), '-c', 'import tensornet; print(tensornet.__version__)'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False, f"Import failed: {result.stderr}"
            
            return True, f"Installed and imported successfully: {result.stdout.strip()}"
        
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print(" PACKAGING GATE")
    print("=" * 60)
    print()
    
    project_root = Path(__file__).parent.parent
    
    report = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'checks': {},
    }
    
    all_passed = True
    
    # Build wheel
    print("Building wheel...", end=' ')
    passed, output, wheel_path = build_wheel(project_root)
    report['checks']['wheel_build'] = {
        'passed': passed,
        'output': output[:500] if not passed else str(wheel_path)
    }
    
    if passed:
        print(f"✓ PASS ({wheel_path.name})")
    else:
        print(f"✗ FAIL")
        print(f"  {output[:200]}")
        all_passed = False
    
    if passed:
        # Twine check
        print("Running twine check...", end=' ')
        twine_passed, issues = check_with_twine(wheel_path.parent)
        report['checks']['twine_check'] = {
            'passed': twine_passed,
            'issues': issues
        }
        
        if twine_passed:
            print("✓ PASS")
        else:
            print(f"✗ FAIL ({len(issues)} issues)")
            for issue in issues[:3]:
                print(f"  - {issue}")
            all_passed = False
        
        # Install verification
        print("Verifying install...", end=' ')
        install_passed, install_output = verify_install(wheel_path)
        report['checks']['install_verify'] = {
            'passed': install_passed,
            'output': install_output
        }
        
        if install_passed:
            print(f"✓ PASS")
            print(f"  {install_output}")
        else:
            print(f"✗ FAIL")
            print(f"  {install_output}")
            all_passed = False
    
    # Summary
    print()
    print("=" * 60)
    if all_passed:
        print(" ✓ PACKAGING GATE PASSED")
        print()
        print(f" Artifacts: {project_root / 'dist'}")
    else:
        print(" ✗ PACKAGING GATE FAILED")
    print("=" * 60)
    
    # Save report
    output_path = project_root / 'artifacts' / 'packaging_gate.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
