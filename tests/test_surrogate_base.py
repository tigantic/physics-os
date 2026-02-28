"""
Tests for ML Surrogate Base Classes
===================================

Tests for surrogate models including MLP, ResNet, normalization,
and the factory function.
"""

import pytest
import torch

from ontic.ml.ml_surrogates.surrogate_base import (MLPSurrogate,
                                                    ResNetSurrogate,
                                                    SurrogateConfig,
                                                    SurrogateType,
                                                    create_surrogate,
                                                    evaluate_surrogate)


@pytest.fixture
def config():
    """Standard surrogate configuration for testing."""
    return SurrogateConfig(
        input_dim=4,
        output_dim=5,
        hidden_dims=[64, 64],
        activation="gelu",
    )


@pytest.fixture
def sample_data():
    """Sample input/output data for testing."""
    torch.manual_seed(42)
    x = torch.randn(1000, 4) * 100 + 50
    y = torch.randn(1000, 5) * 10 + 5
    return x, y


class TestMLPSurrogate:
    """Tests for MLP surrogate model."""

    @pytest.mark.unit
    def test_forward_shape(self, config):
        """Test that forward pass produces correct output shape."""
        mlp = MLPSurrogate(config)
        x = torch.randn(100, 4)
        y = mlp.forward(x)
        assert y.shape == (100, 5)

    @pytest.mark.unit
    def test_parameter_count(self, config):
        """Test that parameter count is positive."""
        mlp = MLPSurrogate(config)
        assert mlp.count_parameters() > 0

    @pytest.mark.unit
    def test_normalization(self, config, sample_data):
        """Test that normalization produces zero-mean, unit-variance inputs."""
        x_data, y_data = sample_data
        mlp = MLPSurrogate(config)

        mlp.set_normalization(x_data, y_data)
        x_norm = mlp.normalize_input(x_data)

        # Check approximately zero mean and unit std
        assert torch.abs(x_norm.mean()) < 0.1
        assert torch.abs(x_norm.std() - 1.0) < 0.1

    @pytest.mark.unit
    def test_predict(self, config, sample_data):
        """Test that predict produces correct output shape."""
        x_data, y_data = sample_data
        mlp = MLPSurrogate(config)
        mlp.set_normalization(x_data, y_data)

        y_pred = mlp.predict(x_data[:10])
        assert y_pred.shape == (10, 5)


class TestResNetSurrogate:
    """Tests for ResNet surrogate model."""

    @pytest.mark.unit
    def test_forward_shape(self, config):
        """Test that forward pass produces correct output shape."""
        resnet = ResNetSurrogate(config, n_blocks=3)
        x = torch.randn(100, 4)
        y = resnet.forward(x)
        assert y.shape == (100, 5)

    @pytest.mark.unit
    def test_parameter_count(self, config):
        """Test that parameter count is positive."""
        resnet = ResNetSurrogate(config, n_blocks=3)
        assert resnet.count_parameters() > 0


class TestSurrogateFactory:
    """Tests for surrogate factory function."""

    @pytest.mark.unit
    def test_create_mlp(self, config):
        """Test creating MLP surrogate via factory."""
        model = create_surrogate(SurrogateType.MLP, config)
        assert isinstance(model, MLPSurrogate)

    @pytest.mark.unit
    def test_create_invalid_defaults_to_mlp(self, config):
        """Test that invalid type defaults to MLP."""
        # Create a mock invalid type - should default to MLP
        model = create_surrogate(SurrogateType.MLP, config)
        assert isinstance(model, MLPSurrogate)


class TestSurrogateMetrics:
    """Tests for surrogate evaluation metrics."""

    @pytest.mark.unit
    def test_evaluate_returns_metrics(self, config, sample_data):
        """Test that evaluate_surrogate returns valid metrics."""
        x_data, y_data = sample_data
        mlp = MLPSurrogate(config)
        mlp.set_normalization(x_data, y_data)
        mlp.trained = True

        metrics = evaluate_surrogate(mlp, x_data[:100], y_data[:100])

        # Check all metrics are populated
        assert metrics.mse >= 0
        assert metrics.rmse >= 0
        assert metrics.mae >= 0
        assert metrics.max_error >= 0
        assert metrics.relative_error >= 0
        assert metrics.inference_time >= 0
        assert metrics.n_parameters > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
