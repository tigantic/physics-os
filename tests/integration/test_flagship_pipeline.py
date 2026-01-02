"""
Integration test for the flagship pipeline.

This test ensures the flagship pipeline runs end-to-end and produces
verifiable evidence artifacts.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Mark entire module as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestFlagshipPipeline:
    """Test the flagship pipeline produces valid evidence."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.project_root = Path(__file__).parent.parent.parent
        self.demos_dir = self.project_root / "demos"
        self.evidence_dir = self.project_root / "evidence" / "flagship_pack"
        # Set UTF-8 encoding for Windows subprocess
        self.env = os.environ.copy()
        self.env["PYTHONIOENCODING"] = "utf-8"

    def test_flagship_runs_without_error(self):
        """Flagship pipeline completes without raising exceptions."""
        result = subprocess.run(
            [sys.executable, str(self.demos_dir / "flagship_pipeline.py")],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=self.project_root,
            env=self.env,
        )

        # Check for successful completion
        assert result.returncode == 0, f"Pipeline failed:\n{result.stderr}"

    def test_evidence_pack_created(self):
        """Evidence pack directory and manifest exist."""
        # Run the pipeline first
        subprocess.run(
            [sys.executable, str(self.demos_dir / "flagship_pipeline.py")],
            capture_output=True,
            timeout=300,
            cwd=self.project_root,
            env=self.env,
        )

        # Check evidence artifacts
        assert self.evidence_dir.exists(), "Evidence directory not created"
        assert (self.evidence_dir / "manifest.json").exists(), "Manifest not created"
        assert (self.evidence_dir / "verify.py").exists(), "Verify script not created"
        assert (self.evidence_dir / "data").exists(), "Data directory not created"

    def test_verify_script_passes(self):
        """Verification script returns PASS."""
        # Run the pipeline
        subprocess.run(
            [sys.executable, str(self.demos_dir / "flagship_pipeline.py")],
            capture_output=True,
            timeout=300,
            cwd=self.project_root,
            env=self.env,
        )

        # Run verification
        verify_script = self.evidence_dir / "verify.py"
        if verify_script.exists():
            result = subprocess.run(
                [sys.executable, str(verify_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.evidence_dir,
                env=self.env,
            )

            assert result.returncode == 0, f"Verification failed:\n{result.stdout}"
            assert "PASS" in result.stdout, "Expected PASS in output"

    def test_data_files_have_content(self):
        """Data files contain valid numpy arrays."""
        import numpy as np

        # Run the pipeline
        subprocess.run(
            [sys.executable, str(self.demos_dir / "flagship_pipeline.py")],
            capture_output=True,
            timeout=300,
            cwd=self.project_root,
            env=self.env,
        )

        data_dir = self.evidence_dir / "data"
        expected_files = [
            "x_grid.npy",
            "rho_final.npy",
            "u_final.npy",
            "p_final.npy",
            "electron_density.npy",
        ]

        for filename in expected_files:
            filepath = data_dir / filename
            if filepath.exists():
                data = np.load(filepath)
                assert data.size > 0, f"{filename} is empty"
                assert np.isfinite(data).all(), f"{filename} contains NaN/Inf"
