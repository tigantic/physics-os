#!/usr/bin/env python3
"""
Generate API documentation using pdoc3.

Usage:
    python scripts/generate_api_docs.py
    
Output:
    docs/api/
"""

import subprocess
import sys
from pathlib import Path


# Public modules to document
PUBLIC_MODULES = [
    "tensornet.core",
    "tensornet.cfd",
    "tensornet.dmrg",
    "tensornet.qtt",
]


def main():
    """Generate API documentation."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "docs" / "api"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating API documentation in {output_dir}")
    
    for module in PUBLIC_MODULES:
        # Generate docs using pdoc3
        cmd = [
            sys.executable, "-m", "pdoc",
            "--html",
            "--output-dir", str(output_dir),
            "--force",  # Overwrite existing docs
            module,
        ]
        
        print(f"  Documenting: {module}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                print(f"    Warning: {module} completed with errors")
                if result.stderr:
                    # Print first few error lines
                    for line in result.stderr.split('\n')[:3]:
                        print(f"    {line}")
                
        except FileNotFoundError:
            print("Error: pdoc not found. Install with: pip install pdoc3")
            sys.exit(1)
    
    print()
    print("API documentation generated!")
    print(f"Open {output_dir}/tensornet/index.html in a browser")


if __name__ == "__main__":
    main()
