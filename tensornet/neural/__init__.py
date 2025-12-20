"""
Neural-Enhanced Tensor Network Module
=====================================

Neural network components for enhancing tensor network algorithms:
- Learned truncation policies
- Bond dimension prediction
- Entanglement structure learning via GNNs
- Adaptive algorithm selection
"""

from tensornet.neural.truncation_policy import (
    TruncationPolicy,
    PolicyNetwork,
    PolicyAction,
    PolicyState,
    RLTruncationAgent,
    train_truncation_policy,
    load_truncation_policy,
)

from tensornet.neural.bond_predictor import (
    BondDimensionPredictor,
    PredictorConfig,
    PredictionResult,
    EntropyFeatures,
    TemporalFeatures,
    train_bond_predictor,
    predict_optimal_chi,
)

from tensornet.neural.entanglement_gnn import (
    EntanglementGNN,
    GNNConfig,
    EntanglementGraph,
    NodeFeatures,
    EdgeFeatures,
    GNNPrediction,
    build_entanglement_graph,
    predict_entanglement_structure,
)

from tensornet.neural.algorithm_selector import (
    AlgorithmSelector,
    AlgorithmType,
    SelectionCriteria,
    AlgorithmRecommendation,
    ProblemFeatures,
    select_algorithm,
    benchmark_algorithms,
)

__all__ = [
    # Truncation policy
    "TruncationPolicy",
    "PolicyNetwork",
    "PolicyAction",
    "PolicyState",
    "RLTruncationAgent",
    "train_truncation_policy",
    "load_truncation_policy",
    # Bond predictor
    "BondDimensionPredictor",
    "PredictorConfig",
    "PredictionResult",
    "EntropyFeatures",
    "TemporalFeatures",
    "train_bond_predictor",
    "predict_optimal_chi",
    # Entanglement GNN
    "EntanglementGNN",
    "GNNConfig",
    "EntanglementGraph",
    "NodeFeatures",
    "EdgeFeatures",
    "GNNPrediction",
    "build_entanglement_graph",
    "predict_entanglement_structure",
    # Algorithm selector
    "AlgorithmSelector",
    "AlgorithmType",
    "SelectionCriteria",
    "AlgorithmRecommendation",
    "ProblemFeatures",
    "select_algorithm",
    "benchmark_algorithms",
]
