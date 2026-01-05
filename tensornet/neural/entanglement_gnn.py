"""
Entanglement Graph Neural Network Module
=========================================

Graph neural networks for learning entanglement structure
in tensor networks. Uses message passing to learn correlations
between sites and predict entanglement patterns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class NodeFeatures:
    """Features for a node (site) in the entanglement graph.

    Attributes:
        site_index: Index of the site
        local_dim: Local Hilbert space dimension
        entropy: Entanglement entropy at this bond
        bond_dim_left: Bond dimension to left neighbor
        bond_dim_right: Bond dimension to right neighbor
        occupation: Local occupation expectation
        magnetization: Local magnetization (for spin systems)
    """

    site_index: int
    local_dim: int
    entropy: float
    bond_dim_left: int
    bond_dim_right: int
    occupation: float = 0.0
    magnetization: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert to feature tensor."""
        return torch.tensor(
            [
                self.site_index / 100.0,  # Normalized position
                math.log(self.local_dim) / 5.0,
                self.entropy / 10.0,
                math.log(max(self.bond_dim_left, 1)) / 10.0,
                math.log(max(self.bond_dim_right, 1)) / 10.0,
                self.occupation,
                self.magnetization,
            ],
            dtype=torch.float32,
        )


@dataclass
class EdgeFeatures:
    """Features for an edge (bond) in the entanglement graph.

    Attributes:
        source: Source node index
        target: Target node index
        bond_dim: Bond dimension
        truncation_error: Truncation error at this bond
        correlation: Two-point correlation strength
        distance: Distance between sites
    """

    source: int
    target: int
    bond_dim: int
    truncation_error: float
    correlation: float
    distance: int

    def to_tensor(self) -> torch.Tensor:
        """Convert to feature tensor."""
        return torch.tensor(
            [
                math.log(max(self.bond_dim, 1)) / 10.0,
                -math.log10(max(self.truncation_error, 1e-16)) / 16.0,
                self.correlation,
                1.0 / (1.0 + self.distance),  # Decay with distance
            ],
            dtype=torch.float32,
        )


@dataclass
class EntanglementGraph:
    """Graph representation of entanglement structure.

    Attributes:
        nodes: List of node features
        edges: List of edge features
        node_tensor: Batched node feature tensor
        edge_index: Edge connectivity tensor
        edge_tensor: Batched edge feature tensor
    """

    nodes: list[NodeFeatures]
    edges: list[EdgeFeatures]
    node_tensor: torch.Tensor = field(init=False)
    edge_index: torch.Tensor = field(init=False)
    edge_tensor: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        """Build tensors from features."""
        self.node_tensor = torch.stack([n.to_tensor() for n in self.nodes])

        if self.edges:
            sources = [e.source for e in self.edges]
            targets = [e.target for e in self.edges]
            self.edge_index = torch.tensor([sources, targets], dtype=torch.long)
            self.edge_tensor = torch.stack([e.to_tensor() for e in self.edges])
        else:
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_tensor = torch.zeros((0, 4), dtype=torch.float32)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in graph."""
        return len(self.edges)


@dataclass
class GNNConfig:
    """Configuration for entanglement GNN.

    Attributes:
        node_input_dim: Input dimension for node features
        edge_input_dim: Input dimension for edge features
        hidden_dim: Hidden dimension
        num_layers: Number of message passing layers
        output_dim: Output dimension per node
        dropout: Dropout rate
        aggregation: Aggregation method (mean, sum, max)
    """

    node_input_dim: int = 7
    edge_input_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 4
    output_dim: int = 8
    dropout: float = 0.1
    aggregation: str = "mean"


class MessagePassingLayer(nn.Module):
    """Single message passing layer.

    Updates node features by aggregating messages from neighbors.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        aggregation: str = "mean",
    ) -> None:
        """Initialize layer.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            aggregation: Aggregation method
        """
        super().__init__()

        self.aggregation = aggregation

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Layer norm
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            node_features: Node features (num_nodes, node_dim)
            edge_index: Edge indices (2, num_edges)
            edge_features: Edge features (num_edges, edge_dim)

        Returns:
            Updated node features
        """
        num_nodes = node_features.shape[0]

        if edge_index.shape[1] == 0:
            return node_features

        # Get source and target node features
        source_idx = edge_index[0]
        target_idx = edge_index[1]

        source_features = node_features[source_idx]
        target_features = node_features[target_idx]

        # Compute messages
        message_input = torch.cat(
            [source_features, target_features, edge_features], dim=-1
        )
        messages = self.message_net(message_input)

        # Aggregate messages
        aggregated = torch.zeros(num_nodes, messages.shape[-1], device=messages.device)

        if self.aggregation == "mean":
            # Count incoming edges per node
            counts = torch.zeros(num_nodes, device=messages.device)
            counts.scatter_add_(
                0, target_idx, torch.ones_like(target_idx, dtype=torch.float)
            )
            counts = torch.clamp(counts, min=1)

            aggregated.scatter_add_(
                0, target_idx.unsqueeze(-1).expand_as(messages), messages
            )
            aggregated = aggregated / counts.unsqueeze(-1)
        elif self.aggregation == "sum":
            aggregated.scatter_add_(
                0, target_idx.unsqueeze(-1).expand_as(messages), messages
            )
        elif self.aggregation == "max":
            aggregated.scatter_reduce_(
                0, target_idx.unsqueeze(-1).expand_as(messages), messages, reduce="amax"
            )

        # Update node features
        update_input = torch.cat([node_features, aggregated], dim=-1)
        updated = self.update_net(update_input)

        # Residual connection and normalization
        output = self.norm(node_features + updated)

        return output


class EntanglementGNN(nn.Module):
    """Graph neural network for entanglement structure.

    Learns to predict entanglement properties from graph structure.
    """

    def __init__(self, config: GNNConfig) -> None:
        """Initialize GNN.

        Args:
            config: GNN configuration
        """
        super().__init__()

        self.config = config

        # Input projection
        self.node_encoder = nn.Linear(config.node_input_dim, config.hidden_dim)
        self.edge_encoder = nn.Linear(config.edge_input_dim, config.hidden_dim)

        # Message passing layers
        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    node_dim=config.hidden_dim,
                    edge_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim,
                    aggregation=config.aggregation,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Graph-level readout
        self.graph_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        graph: EntanglementGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            graph: Entanglement graph

        Returns:
            Tuple of (node_outputs, graph_output)
        """
        # Encode inputs
        node_features = self.node_encoder(graph.node_tensor)
        edge_features = (
            self.edge_encoder(graph.edge_tensor) if graph.num_edges > 0 else None
        )

        # Message passing
        for layer in self.layers:
            if edge_features is not None:
                node_features = layer(node_features, graph.edge_index, edge_features)
            node_features = self.dropout(node_features)

        # Node-level output
        node_outputs = self.output_head(node_features)

        # Graph-level output (global pooling)
        graph_features = torch.mean(node_features, dim=0)
        graph_output = self.graph_head(graph_features)

        return node_outputs, graph_output


@dataclass
class GNNPrediction:
    """Prediction from entanglement GNN.

    Attributes:
        node_predictions: Per-node predictions
        graph_prediction: Graph-level prediction
        predicted_chi: Predicted bond dimensions
        predicted_entropy: Predicted entanglement entropies
        scaling_type: Predicted scaling type
    """

    node_predictions: torch.Tensor
    graph_prediction: torch.Tensor
    predicted_chi: list[int]
    predicted_entropy: list[float]
    scaling_type: str

    @classmethod
    def from_outputs(
        cls,
        node_outputs: torch.Tensor,
        graph_output: torch.Tensor,
        chi_max: int = 512,
    ) -> GNNPrediction:
        """Create from network outputs.

        Args:
            node_outputs: Node-level outputs
            graph_output: Graph-level output
            chi_max: Maximum chi

        Returns:
            GNNPrediction instance
        """
        # Interpret node outputs
        # [0:2] -> log chi prediction and uncertainty
        # [2:4] -> entropy prediction and uncertainty
        # [4:] -> other features

        predicted_chi = []
        predicted_entropy = []

        for i in range(node_outputs.shape[0]):
            log_chi = float(node_outputs[i, 0]) * 5 + 3  # Scale to reasonable range
            chi = int(math.exp(log_chi))
            chi = max(2, min(chi_max, chi))
            predicted_chi.append(chi)

            entropy = float(node_outputs[i, 2]) * 5  # Scale
            predicted_entropy.append(max(0, entropy))

        # Interpret graph output for scaling type
        scaling_logits = graph_output[:3] if graph_output.numel() >= 3 else graph_output
        scaling_idx = int(torch.argmax(scaling_logits))
        scaling_types = ["area_law", "log_corrected", "volume_law"]
        scaling_type = scaling_types[scaling_idx % len(scaling_types)]

        return cls(
            node_predictions=node_outputs,
            graph_prediction=graph_output,
            predicted_chi=predicted_chi,
            predicted_entropy=predicted_entropy,
            scaling_type=scaling_type,
        )


def build_entanglement_graph(
    num_sites: int,
    entropies: torch.Tensor,
    bond_dims: list[int],
    truncation_errors: list[float] | None = None,
    correlations: torch.Tensor | None = None,
    local_dims: int = 2,
    connect_neighbors: bool = True,
    connect_all: bool = False,
) -> EntanglementGraph:
    """Build entanglement graph from MPS data.

    Args:
        num_sites: Number of sites
        entropies: Per-bond entropies
        bond_dims: Per-bond bond dimensions
        truncation_errors: Per-bond truncation errors
        correlations: Two-point correlation matrix
        local_dims: Local dimension
        connect_neighbors: Connect nearest neighbors
        connect_all: Connect all pairs (for small systems)

    Returns:
        EntanglementGraph
    """
    # Build nodes
    nodes = []
    for i in range(num_sites):
        entropy = float(entropies[i]) if i < len(entropies) else 0.0
        bond_left = bond_dims[i - 1] if i > 0 and i - 1 < len(bond_dims) else 1
        bond_right = bond_dims[i] if i < len(bond_dims) else 1

        nodes.append(
            NodeFeatures(
                site_index=i,
                local_dim=local_dims,
                entropy=entropy,
                bond_dim_left=bond_left,
                bond_dim_right=bond_right,
            )
        )

    # Build edges
    edges = []

    if connect_all and num_sites <= 20:
        # Connect all pairs for small systems
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                corr = float(correlations[i, j]) if correlations is not None else 0.0
                error = (
                    truncation_errors[i]
                    if truncation_errors and i < len(truncation_errors)
                    else 1e-10
                )

                edges.append(
                    EdgeFeatures(
                        source=i,
                        target=j,
                        bond_dim=bond_dims[i] if i < len(bond_dims) else 1,
                        truncation_error=error,
                        correlation=corr,
                        distance=j - i,
                    )
                )
                # Add reverse edge
                edges.append(
                    EdgeFeatures(
                        source=j,
                        target=i,
                        bond_dim=bond_dims[i] if i < len(bond_dims) else 1,
                        truncation_error=error,
                        correlation=corr,
                        distance=j - i,
                    )
                )
    elif connect_neighbors:
        # Connect nearest neighbors
        for i in range(num_sites - 1):
            corr = float(correlations[i, i + 1]) if correlations is not None else 0.0
            error = (
                truncation_errors[i]
                if truncation_errors and i < len(truncation_errors)
                else 1e-10
            )

            edges.append(
                EdgeFeatures(
                    source=i,
                    target=i + 1,
                    bond_dim=bond_dims[i] if i < len(bond_dims) else 1,
                    truncation_error=error,
                    correlation=corr,
                    distance=1,
                )
            )
            # Add reverse edge
            edges.append(
                EdgeFeatures(
                    source=i + 1,
                    target=i,
                    bond_dim=bond_dims[i] if i < len(bond_dims) else 1,
                    truncation_error=error,
                    correlation=corr,
                    distance=1,
                )
            )

    return EntanglementGraph(nodes=nodes, edges=edges)


def predict_entanglement_structure(
    graph: EntanglementGraph,
    model: EntanglementGNN | None = None,
    config: GNNConfig | None = None,
) -> GNNPrediction:
    """Predict entanglement structure using GNN.

    Args:
        graph: Entanglement graph
        model: Trained GNN model
        config: GNN configuration

    Returns:
        GNNPrediction
    """
    if model is None:
        config = config or GNNConfig()
        model = EntanglementGNN(config)
        model.eval()

    with torch.no_grad():
        node_outputs, graph_output = model(graph)

    return GNNPrediction.from_outputs(node_outputs, graph_output)


def train_entanglement_gnn(
    training_graphs: list[EntanglementGraph],
    target_chi: list[list[int]],
    target_entropy: list[list[float]],
    config: GNNConfig | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> EntanglementGNN:
    """Train entanglement GNN.

    Args:
        training_graphs: List of training graphs
        target_chi: Target chi values per graph
        target_entropy: Target entropy values per graph
        config: GNN configuration
        epochs: Training epochs
        learning_rate: Learning rate
        device: Computation device

    Returns:
        Trained EntanglementGNN
    """
    config = config or GNNConfig()
    model = EntanglementGNN(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for graph, chi_targets, entropy_targets in zip(
            training_graphs, target_chi, target_entropy
        ):
            # Move to device
            graph.node_tensor = graph.node_tensor.to(device)
            graph.edge_index = graph.edge_index.to(device)
            graph.edge_tensor = graph.edge_tensor.to(device)

            # Forward pass
            node_outputs, graph_output = model(graph)

            # Compute loss
            chi_pred = node_outputs[:, 0]  # First output is log chi
            entropy_pred = node_outputs[:, 2]  # Third output is entropy

            chi_target = torch.tensor(
                [math.log(c) for c in chi_targets[: len(chi_pred)]],
                dtype=torch.float32,
                device=device,
            )
            entropy_target = torch.tensor(
                entropy_targets[: len(entropy_pred)],
                dtype=torch.float32,
                device=device,
            )

            chi_loss = F.mse_loss(chi_pred[: len(chi_target)], chi_target)
            entropy_loss = F.mse_loss(
                entropy_pred[: len(entropy_target)], entropy_target
            )

            loss = chi_loss + entropy_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    model.eval()

    return model
