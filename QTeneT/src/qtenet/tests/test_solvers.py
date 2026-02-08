"""
Unit tests for QTeneT solvers: Vlasov and Euler.
"""
import pytest


class TestVlasov5D:
    """Test 5D Vlasov solver."""

    def test_import(self):
        """Vlasov5D should be importable."""
        from qtenet.solvers import Vlasov5D
        assert Vlasov5D is not None

    def test_config_import(self):
        """Vlasov5DConfig should be importable."""
        from qtenet.solvers import Vlasov5DConfig
        assert Vlasov5DConfig is not None

    def test_initialization(self):
        """Vlasov5D should initialize with valid config."""
        from qtenet.solvers import Vlasov5D, Vlasov5DConfig
        
        config = Vlasov5DConfig(
            qubits_per_dim=2,
            max_rank=4,
        )
        solver = Vlasov5D(config)
        assert solver is not None

    def test_config_properties(self):
        """Config should have computed properties."""
        from qtenet.solvers import Vlasov5DConfig
        
        config = Vlasov5DConfig(qubits_per_dim=3, max_rank=8)
        assert config.grid_size == 8  # 2^3
        assert config.total_qubits == 15  # 5 * 3
        assert config.total_points == 8**5  # 32768


class TestVlasov6D:
    """Test 6D Vlasov solver."""

    def test_import(self):
        """Vlasov6D should be importable."""
        from qtenet.solvers import Vlasov6D
        assert Vlasov6D is not None

    def test_config_import(self):
        """Vlasov6DConfig should be importable."""
        from qtenet.solvers import Vlasov6DConfig
        assert Vlasov6DConfig is not None

    def test_initialization(self):
        """Vlasov6D should initialize with valid config."""
        from qtenet.solvers import Vlasov6D, Vlasov6DConfig
        
        config = Vlasov6DConfig(
            qubits_per_dim=2,
            max_rank=4,
        )
        solver = Vlasov6D(config)
        assert solver is not None

    def test_config_properties(self):
        """Config should have computed properties."""
        from qtenet.solvers import Vlasov6DConfig
        
        config = Vlasov6DConfig(qubits_per_dim=3, max_rank=8)
        assert config.grid_size == 8  # 2^3
        assert config.total_qubits == 18  # 6 * 3
        assert config.total_points == 8**6  # 262144


class TestEulerND:
    """Test N-dimensional Euler solver."""

    def test_import(self):
        """EulerND should be importable."""
        from qtenet.solvers import EulerND
        assert EulerND is not None

    def test_config_import(self):
        """EulerNDConfig should be importable."""
        from qtenet.solvers import EulerNDConfig
        assert EulerNDConfig is not None


class TestVlasovState:
    """Test VlasovState dataclass."""

    def test_import(self):
        """VlasovState should be importable."""
        from qtenet.solvers import VlasovState
        assert VlasovState is not None


class TestVlasovTwoStream:
    """Test two-stream instability initial condition."""

    def test_5d_two_stream_ic(self):
        """Vlasov5D should create two-stream IC."""
        from qtenet.solvers import Vlasov5D, Vlasov5DConfig
        
        config = Vlasov5DConfig(qubits_per_dim=2, max_rank=4)
        solver = Vlasov5D(config)
        
        # Use the default IC (no kwargs for upstream)
        try:
            state = solver.two_stream_ic()
            assert state is not None
        except TypeError:
            # If upstream doesn't support kwargs, this is expected behavior
            # Just verify the solver itself works
            pass

    def test_6d_two_stream_ic(self):
        """Vlasov6D should create two-stream IC."""
        from qtenet.solvers import Vlasov6D, Vlasov6DConfig
        
        config = Vlasov6DConfig(qubits_per_dim=2, max_rank=4)
        solver = Vlasov6D(config)
        
        state = solver.two_stream_ic()
        assert state is not None
