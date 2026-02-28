"""
Deep Operator Networks (DeepONet) for CFD.

This module implements DeepONet architectures for learning solution
operators of PDEs. DeepONets learn mappings from input functions
(e.g., initial/boundary conditions) to output functions (solutions).

Key features:
    - Branch-Trunk architecture
    - Multiple input function encoding
    - Flexible trunk architectures
    - Support for vector-valued operators

Reference:
    Lu et al. "Learning nonlinear operators via DeepONet" (2021)

Author: TiganticLabz
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .surrogate_base import CFDSurrogate, SurrogateConfig


@dataclass
class DeepONetConfig(SurrogateConfig):
    """Configuration for Deep Operator Networks."""

    # Branch network (encodes input function)
    branch_input_dim: int = 100  # Number of sensor points
    branch_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    branch_output_dim: int = 64  # Latent dimension p

    # Trunk network (encodes query locations)
    trunk_input_dim: int = 3  # (x, y, t) coordinates
    trunk_hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    trunk_output_dim: int = 64  # Must match branch output

    # Output
    n_outputs: int = 1  # Number of output fields

    # Architecture options
    use_bias: bool = True
    normalize_branch: bool = True

    # Stacked DeepONet for multiple outputs
    stacked: bool = False


class BranchNet(nn.Module):
    """
    Branch network for encoding input functions.

    Maps from sensor observations of input function u(x_i)
    to a latent representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        # Get activation
        act_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }.get(activation, nn.GELU())

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Encode input function.

        Args:
            u: Sensor observations of shape (batch, n_sensors)

        Returns:
            Latent code of shape (batch, p)
        """
        return self.network(u)


class TrunkNet(nn.Module):
    """
    Trunk network for encoding query locations.

    Maps from spatial/temporal coordinates to basis functions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "gelu",
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        act_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }.get(activation, nn.GELU())

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Encode query locations.

        Args:
            y: Query coordinates of shape (batch, n_queries, dim) or (n_queries, dim)

        Returns:
            Basis functions of shape (..., p)
        """
        return self.network(y)


class DeepONet(CFDSurrogate):
    """
    Deep Operator Network for learning solution operators.

    Learns the mapping G: u -> G(u)(y) where:
        - u is an input function (given by sensor values)
        - y is a query location
        - G(u)(y) is the solution at location y

    The approximation is:
        G(u)(y) ≈ sum_k b_k(u) * t_k(y)

    where b_k comes from branch network and t_k from trunk.

    Example:
        >>> config = DeepONetConfig(branch_input_dim=100, trunk_input_dim=3)
        >>> onet = DeepONet(config)
        >>> # u_sensors: (batch, 100) - input function at sensors
        >>> # y_query: (batch, n_points, 3) - query locations
        >>> output = onet(u_sensors, y_query)
    """

    def __init__(self, config: DeepONetConfig):
        # Store as SurrogateConfig for parent
        super_config = SurrogateConfig(
            input_dim=config.branch_input_dim + config.trunk_input_dim,
            output_dim=config.n_outputs,
            hidden_dims=config.branch_hidden_dims,
            activation=config.activation,
        )
        super().__init__(super_config)

        self.onet_config = config
        self.build_network()

    def build_network(self):
        """Build DeepONet architecture."""
        config = self.onet_config

        # Branch network
        self.branch = BranchNet(
            config.branch_input_dim,
            config.branch_hidden_dims,
            config.branch_output_dim * config.n_outputs,
            config.activation,
        )

        # Trunk network
        self.trunk = TrunkNet(
            config.trunk_input_dim,
            config.trunk_hidden_dims,
            config.trunk_output_dim * config.n_outputs,
            config.activation,
        )

        # Optional bias
        if config.use_bias:
            self.bias = nn.Parameter(torch.zeros(config.n_outputs))
        else:
            self.register_buffer("bias", torch.zeros(config.n_outputs))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DeepONet.

        Args:
            u: Input function values at sensors (batch, n_sensors)
            y: Query locations (batch, n_queries, trunk_dim) or (n_queries, trunk_dim)

        Returns:
            Solution at query points (batch, n_queries, n_outputs)
        """
        config = self.onet_config
        batch_size = u.shape[0]

        # Branch encoding: (batch, p * n_outputs)
        b = self.branch(u)

        # Handle different y shapes
        if y.dim() == 2:
            # (n_queries, trunk_dim) -> broadcast to batch
            y = y.unsqueeze(0).expand(batch_size, -1, -1)

        n_queries = y.shape[1]

        # Trunk encoding: (batch, n_queries, p * n_outputs)
        t = self.trunk(y)

        # Reshape for dot product
        # b: (batch, n_outputs, p)
        # t: (batch, n_queries, n_outputs, p)
        b = b.view(batch_size, config.n_outputs, config.branch_output_dim)
        t = t.view(batch_size, n_queries, config.n_outputs, config.trunk_output_dim)

        # Dot product: sum over p dimension
        # output: (batch, n_queries, n_outputs)
        output = torch.einsum("bop,bqop->bqo", b, t)

        # Add bias
        output = output + self.bias

        return output

    def predict(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict with normalization handling."""
        self.eval()
        with torch.no_grad():
            return self.forward(u, y)


class MultiInputDeepONet(DeepONet):
    """
    DeepONet with multiple input function branches.

    Useful when the operator depends on multiple input functions,
    e.g., initial condition and boundary condition.
    """

    def __init__(self, config: DeepONetConfig, n_inputs: int = 2):
        self.n_inputs = n_inputs
        super().__init__(config)

    def build_network(self):
        """Build multi-input DeepONet."""
        config = self.onet_config

        # Multiple branch networks
        self.branches = nn.ModuleList(
            [
                BranchNet(
                    config.branch_input_dim,
                    config.branch_hidden_dims,
                    config.branch_output_dim,
                    config.activation,
                )
                for _ in range(self.n_inputs)
            ]
        )

        # Combine branches
        self.branch_combine = nn.Linear(
            config.branch_output_dim * self.n_inputs,
            config.branch_output_dim * config.n_outputs,
        )

        # Trunk network
        self.trunk = TrunkNet(
            config.trunk_input_dim,
            config.trunk_hidden_dims,
            config.trunk_output_dim * config.n_outputs,
            config.activation,
        )

        if config.use_bias:
            self.bias = nn.Parameter(torch.zeros(config.n_outputs))
        else:
            self.register_buffer("bias", torch.zeros(config.n_outputs))

    def forward(self, u_list: list[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multiple inputs.

        Args:
            u_list: List of input functions [u1, u2, ...]
            y: Query locations

        Returns:
            Solution at query points
        """
        config = self.onet_config
        batch_size = u_list[0].shape[0]

        # Encode each input
        b_list = [branch(u) for branch, u in zip(self.branches, u_list)]

        # Combine
        b_combined = torch.cat(b_list, dim=-1)
        b = self.branch_combine(b_combined)

        # Rest is same as standard DeepONet
        if y.dim() == 2:
            y = y.unsqueeze(0).expand(batch_size, -1, -1)

        n_queries = y.shape[1]
        t = self.trunk(y)

        b = b.view(batch_size, config.n_outputs, config.branch_output_dim)
        t = t.view(batch_size, n_queries, config.n_outputs, config.trunk_output_dim)

        output = torch.einsum("bop,bqop->bqo", b, t)
        return output + self.bias


def create_deeponet(
    branch_dim: int = 100,
    trunk_dim: int = 3,
    latent_dim: int = 64,
    n_outputs: int = 1,
    **kwargs,
) -> DeepONet:
    """
    Factory function to create DeepONet.

    Args:
        branch_dim: Dimension of input function sensors
        trunk_dim: Dimension of query coordinates
        latent_dim: Latent space dimension
        n_outputs: Number of output fields
        **kwargs: Additional configuration

    Returns:
        Configured DeepONet
    """
    config = DeepONetConfig(
        branch_input_dim=branch_dim,
        trunk_input_dim=trunk_dim,
        branch_output_dim=latent_dim,
        trunk_output_dim=latent_dim,
        n_outputs=n_outputs,
        **kwargs,
    )
    return DeepONet(config)


def train_deeponet(
    model: DeepONet,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    s_train: torch.Tensor,
    n_epochs: int = 1000,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Train DeepONet on data.

    Args:
        model: DeepONet to train
        u_train: Input functions (n_samples, n_sensors)
        y_train: Query points (n_samples, n_queries, dim)
        s_train: Target solutions (n_samples, n_queries, n_outputs)
        n_epochs: Number of epochs
        lr: Learning rate
        batch_size: Batch size
        verbose: Print progress

    Returns:
        Training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=50, factor=0.5
    )

    history = {"loss": []}
    n_samples = len(u_train)

    model.train()
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            u_batch = u_train[idx]
            y_batch = y_train[idx]
            s_batch = s_train[idx]

            optimizer.zero_grad()
            pred = model(u_batch, y_batch)
            loss = torch.mean((pred - s_batch) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        history["loss"].append(epoch_loss)
        scheduler.step(epoch_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: loss = {epoch_loss:.6f}")

    model.trained = True
    return history


def test_deep_onet():
    """Test DeepONet implementation."""
    print("Testing Deep Operator Networks...")

    # Create config
    config = DeepONetConfig(
        branch_input_dim=50,
        branch_hidden_dims=[64, 64],
        branch_output_dim=32,
        trunk_input_dim=2,
        trunk_hidden_dims=[64, 64],
        trunk_output_dim=32,
        n_outputs=3,
    )

    # Create DeepONet
    print("\n  Creating DeepONet...")
    onet = DeepONet(config)
    n_params = sum(p.numel() for p in onet.parameters())
    print(f"    Parameters: {n_params:,}")

    # Test forward pass
    print("\n  Testing forward pass...")
    batch_size = 16
    n_sensors = 50
    n_queries = 100
    trunk_dim = 2

    u = torch.randn(batch_size, n_sensors)
    y = torch.randn(batch_size, n_queries, trunk_dim)

    output = onet(u, y)
    assert output.shape == (batch_size, n_queries, 3)
    print(f"    Output shape: {output.shape}")

    # Test with broadcast y
    print("\n  Testing broadcast query points...")
    y_single = torch.randn(n_queries, trunk_dim)
    output2 = onet(u, y_single)
    assert output2.shape == (batch_size, n_queries, 3)
    print(f"    Broadcast output shape: {output2.shape}")

    # Test multi-input DeepONet
    print("\n  Testing multi-input DeepONet...")
    multi_onet = MultiInputDeepONet(config, n_inputs=2)
    u1 = torch.randn(batch_size, n_sensors)
    u2 = torch.randn(batch_size, n_sensors)
    output_multi = multi_onet([u1, u2], y)
    assert output_multi.shape == (batch_size, n_queries, 3)
    print(f"    Multi-input output shape: {output_multi.shape}")

    # Test factory function
    print("\n  Testing factory function...")
    onet2 = create_deeponet(
        branch_dim=100,
        trunk_dim=3,
        latent_dim=64,
        n_outputs=5,
    )
    u_test = torch.randn(8, 100)
    y_test = torch.randn(8, 50, 3)
    out = onet2(u_test, y_test)
    assert out.shape == (8, 50, 5)
    print(f"    Factory output shape: {out.shape}")

    # Test training (quick)
    print("\n  Testing training loop...")
    small_onet = create_deeponet(branch_dim=20, trunk_dim=2, latent_dim=16, n_outputs=1)

    # Generate synthetic data
    n_samples = 100
    u_train = torch.randn(n_samples, 20)
    y_train = torch.randn(n_samples, 10, 2)
    s_train = torch.randn(n_samples, 10, 1)

    history = train_deeponet(
        small_onet,
        u_train,
        y_train,
        s_train,
        n_epochs=50,
        lr=1e-3,
        batch_size=32,
        verbose=False,
    )

    assert len(history["loss"]) == 50
    assert history["loss"][-1] < history["loss"][0]  # Should improve
    print(f"    Final loss: {history['loss'][-1]:.6f}")

    print("\nDeep Operator Networks: All tests passed!")


if __name__ == "__main__":
    test_deep_onet()
