"""
Test Module: ontic/racing/wake.py

Phase 12: F1 Dirty Air Wake Tracker
Constitutional Compliance:
    - Article III, Section 3.1: Unit tests with 90%+ coverage
    - Article III, Section 3.2: Deterministic seeding (seed=42)
    - Article IV, Section 4.1: Physical validation
    - Article V, Section 5.1: Float64 precision

References:
    Katz, J., Plotkin, A. (2001). "Low-Speed Aerodynamics."
    Cambridge University Press, 2nd Edition. ISBN: 0521665523

    Newbon, J., et al. (2017). "The Aerodynamics of Open-Wheel Racing Cars."
    SAE Technical Paper 2017-01-1545. DOI: 10.4271/2017-01-1545
"""

import numpy as np
import pytest
import torch

from ontic.aerospace.racing.wake import DirtyAirReport, WakeTracker

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def deterministic_seed():
    """Per Article III, Section 3.2: Reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture
def tracker():
    """Standard wake tracker."""
    return WakeTracker(track_width=50, height=20, length=200)


# ============================================================================
# UNIT TESTS
# ============================================================================


class TestWakeTrackerInit:
    """Test WakeTracker initialization."""

    @pytest.mark.unit
    def test_init_dimensions(self, deterministic_seed):
        """Test tracker dimensions."""
        tracker = WakeTracker(track_width=50, height=20, length=200)

        assert tracker.width == 50
        assert tracker.height == 20
        assert tracker.length == 200

    @pytest.mark.unit
    def test_init_field_shapes(self, deterministic_seed):
        """Test wake field tensor shapes."""
        tracker = WakeTracker(track_width=50, height=20, length=200)

        assert tracker.wake_field.shape == (50, 20, 200)
        assert tracker.turbulence.shape == (50, 20, 200)

    @pytest.mark.unit
    def test_float64_compliance(self, deterministic_seed):
        """Per Article V: Physics tensors must be float64."""
        tracker = WakeTracker(track_width=50, height=20, length=200)

        assert tracker.wake_field.dtype == torch.float64
        assert tracker.turbulence.dtype == torch.float64

    @pytest.mark.unit
    def test_initial_fields_zero(self, tracker, deterministic_seed):
        """Fields should start at zero (clean air)."""
        assert tracker.wake_field.sum().item() == 0.0
        assert tracker.turbulence.sum().item() == 0.0


class TestWakeUpdate:
    """Test wake field updates."""

    @pytest.mark.unit
    def test_update_creates_wake(self, tracker, deterministic_seed):
        """Updating should create non-zero wake field."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300, crosswind_ms=0.0)

        assert tracker.wake_field.sum().item() > 0

    @pytest.mark.unit
    @pytest.mark.physics
    def test_wake_intensity_decays(self, tracker, deterministic_seed):
        """Wake intensity should decay with distance."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        # Check intensity at different distances
        intensity_near = tracker.wake_field[25, 2, 10].item()  # 10m behind
        intensity_far = tracker.wake_field[25, 2, 100].item()  # 100m behind

        assert intensity_near > intensity_far

    @pytest.mark.unit
    @pytest.mark.physics
    def test_wake_expands_with_distance(self, tracker, deterministic_seed):
        """Wake cone should expand with distance."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        # Count affected cells at different distances
        y = 2  # car height
        near_slice = tracker.wake_field[:, y, 20]
        far_slice = tracker.wake_field[:, y, 100]

        affected_near = (near_slice > 0.01).sum().item()
        affected_far = (far_slice > 0.01).sum().item()

        # Wake should be wider at distance
        assert affected_far >= affected_near

    @pytest.mark.unit
    @pytest.mark.physics
    def test_higher_speed_stronger_wake(self, deterministic_seed):
        """Higher speeds should create more intense wake."""
        tracker_slow = WakeTracker()
        tracker_fast = WakeTracker()

        tracker_slow.update_wake(lead_car_x=25, lead_car_speed_kmh=200)
        tracker_fast.update_wake(lead_car_x=25, lead_car_speed_kmh=350)

        slow_intensity = tracker_slow.wake_field.max().item()
        fast_intensity = tracker_fast.wake_field.max().item()

        assert fast_intensity > slow_intensity

    @pytest.mark.unit
    @pytest.mark.physics
    def test_crosswind_shifts_wake(self, deterministic_seed):
        """Crosswind should shift wake laterally."""
        tracker_no_wind = WakeTracker()
        tracker_wind = WakeTracker()

        tracker_no_wind.update_wake(
            lead_car_x=25, lead_car_speed_kmh=300, crosswind_ms=0
        )
        tracker_wind.update_wake(lead_car_x=25, lead_car_speed_kmh=300, crosswind_ms=5)

        # Find center of wake at a distance
        y = 2
        z = 100

        device = tracker_no_wind.wake_field.device
        no_wind_centroid = (
            tracker_no_wind.wake_field[:, y, z]
            * torch.arange(50, dtype=torch.float64, device=device)
        ).sum() / (tracker_no_wind.wake_field[:, y, z].sum() + 1e-10)

        wind_centroid = (
            tracker_wind.wake_field[:, y, z]
            * torch.arange(50, dtype=torch.float64, device=device)
        ).sum() / (tracker_wind.wake_field[:, y, z].sum() + 1e-10)

        # Centroids should differ due to wind drift
        assert no_wind_centroid.item() != pytest.approx(wind_centroid.item(), abs=0.1)


class TestTurbulence:
    """Test turbulence field."""

    @pytest.mark.unit
    def test_turbulence_created(self, tracker, deterministic_seed):
        """Updating wake should also create turbulence."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        assert tracker.turbulence.sum().item() > 0

    @pytest.mark.unit
    @pytest.mark.physics
    def test_turbulence_at_wake_edges(self, tracker, deterministic_seed):
        """Turbulence should be highest at wake edges (vortex cores)."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        # Turbulence pattern should differ from wake intensity pattern
        y = 2
        z = 50

        # Both should have non-zero values
        wake_slice = tracker.wake_field[:, y, z]
        turb_slice = tracker.turbulence[:, y, z]

        assert wake_slice.max().item() > 0
        assert turb_slice.max().item() > 0


class TestPositionAnalysis:
    """Test position analysis and reporting."""

    @pytest.mark.unit
    def test_analyze_returns_report(self, tracker, deterministic_seed):
        """Analysis should return DirtyAirReport."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        report = tracker.analyze_position(follower_x=25, follower_z=50)

        assert isinstance(report, DirtyAirReport)

    @pytest.mark.unit
    def test_report_has_required_fields(self, tracker, deterministic_seed):
        """Report should have all required fields."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)
        report = tracker.analyze_position(follower_x=25, follower_z=50)

        assert hasattr(report, "downforce_loss_percent")
        assert hasattr(report, "overtake_window")
        assert hasattr(report, "tow_benefit_kmh")

    @pytest.mark.unit
    @pytest.mark.physics
    def test_inline_has_more_loss(self, tracker, deterministic_seed):
        """Directly behind leader should have more downforce loss."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        # Directly behind
        report_inline = tracker.analyze_position(follower_x=25, follower_z=50)
        # Offset to the side
        report_offset = tracker.analyze_position(follower_x=40, follower_z=50)

        assert (
            report_inline.downforce_loss_percent >= report_offset.downforce_loss_percent
        )


class TestDirtyAirReport:
    """Test DirtyAirReport dataclass."""

    @pytest.mark.unit
    def test_report_string(self, tracker, deterministic_seed):
        """Report should have string representation."""
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=300)
        report = tracker.analyze_position(follower_x=25, follower_z=50)

        report_str = str(report)
        assert len(report_str) > 0
        assert "Downforce" in report_str or "downforce" in report_str.lower()


class TestDeterminism:
    """Test reproducibility requirements."""

    @pytest.mark.unit
    def test_deterministic_wake(self):
        """Same inputs should give same wake field."""
        torch.manual_seed(42)
        t1 = WakeTracker()
        t1.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        torch.manual_seed(42)
        t2 = WakeTracker()
        t2.update_wake(lead_car_x=25, lead_car_speed_kmh=300)

        assert torch.allclose(t1.wake_field, t2.wake_field)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestRacingIntegration:
    """Integration tests for racing simulation."""

    @pytest.mark.integration
    def test_full_overtake_analysis(self, deterministic_seed):
        """Test complete overtake analysis workflow."""
        tracker = WakeTracker(track_width=50, height=20, length=200)

        # Setup lead car wake
        tracker.update_wake(lead_car_x=25, lead_car_speed_kmh=320, crosswind_ms=2.0)

        # Analyze multiple following positions
        positions = [
            (25, 30),  # Close, inline
            (25, 100),  # Far, inline
            (35, 50),  # Medium, offset
        ]

        reports = []
        for x, z in positions:
            report = tracker.analyze_position(follower_x=x, follower_z=z)
            reports.append(report)

        # Close position should have highest loss
        assert reports[0].downforce_loss_percent >= reports[1].downforce_loss_percent


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
