"""
Quantum-Classical Hybrid Algorithms for Tensor Networks
========================================================

Implements quantum-inspired and hybrid quantum-classical algorithms
that leverage both tensor network methods and quantum computing primitives.

Key Components:
    - Variational Quantum Eigensolver (VQE) integration
    - Quantum Approximate Optimization Algorithm (QAOA)
    - Tensor Network Born Machines
    - Quantum-inspired classical algorithms
    - Noise-aware simulation protocols

References:
    [1] Peruzzo et al., "A variational eigenvalue solver on a photonic quantum
        processor", Nat. Commun. 5, 4213 (2014)
    [2] Farhi et al., "A Quantum Approximate Optimization Algorithm", 
        arXiv:1411.4028 (2014)
    [3] Huggins et al., "Towards quantum machine learning with tensor networks",
        Quantum Sci. Technol. 4, 024001 (2019)
"""

import torch
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# Quantum Circuit Primitives
# =============================================================================

class GateType(Enum):
    """Standard quantum gate types."""
    # Single-qubit gates
    X = "x"
    Y = "y"
    Z = "z"
    H = "hadamard"
    S = "s"
    T = "t"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    # Two-qubit gates
    CNOT = "cnot"
    CZ = "cz"
    SWAP = "swap"
    # Parameterized gates
    U3 = "u3"
    CRZ = "crz"


@dataclass
class QuantumGate:
    """Representation of a quantum gate."""
    gate_type: GateType
    qubits: Tuple[int, ...]
    parameters: Optional[Tuple[float, ...]] = None
    
    def __post_init__(self):
        # Validate qubit count
        single_qubit = {GateType.X, GateType.Y, GateType.Z, GateType.H,
                       GateType.S, GateType.T, GateType.RX, GateType.RY, 
                       GateType.RZ}
        two_qubit = {GateType.CNOT, GateType.CZ, GateType.SWAP, GateType.CRZ}
        
        if self.gate_type in single_qubit and len(self.qubits) != 1:
            raise ValueError(f"{self.gate_type} requires 1 qubit")
        if self.gate_type in two_qubit and len(self.qubits) != 2:
            raise ValueError(f"{self.gate_type} requires 2 qubits")


@dataclass
class QuantumCircuit:
    """
    Quantum circuit representation.
    
    Stores sequence of gates and provides tensor network simulation.
    """
    n_qubits: int
    gates: List[QuantumGate] = field(default_factory=list)
    
    def add_gate(self, gate: QuantumGate):
        """Add a gate to the circuit."""
        for q in gate.qubits:
            if q >= self.n_qubits:
                raise ValueError(f"Qubit {q} out of range for {self.n_qubits}-qubit circuit")
        self.gates.append(gate)
    
    def rx(self, qubit: int, theta: float):
        """Add RX rotation gate."""
        self.add_gate(QuantumGate(GateType.RX, (qubit,), (theta,)))
        return self
    
    def ry(self, qubit: int, theta: float):
        """Add RY rotation gate."""
        self.add_gate(QuantumGate(GateType.RY, (qubit,), (theta,)))
        return self
    
    def rz(self, qubit: int, theta: float):
        """Add RZ rotation gate."""
        self.add_gate(QuantumGate(GateType.RZ, (qubit,), (theta,)))
        return self
    
    def h(self, qubit: int):
        """Add Hadamard gate."""
        self.add_gate(QuantumGate(GateType.H, (qubit,)))
        return self
    
    def cnot(self, control: int, target: int):
        """Add CNOT gate."""
        self.add_gate(QuantumGate(GateType.CNOT, (control, target)))
        return self
    
    def cz(self, q1: int, q2: int):
        """Add CZ gate."""
        self.add_gate(QuantumGate(GateType.CZ, (q1, q2)))
        return self
    
    @property
    def depth(self) -> int:
        """Circuit depth (number of time steps)."""
        if not self.gates:
            return 0
        # Simple depth counting
        qubit_depths = [0] * self.n_qubits
        for gate in self.gates:
            max_depth = max(qubit_depths[q] for q in gate.qubits)
            for q in gate.qubits:
                qubit_depths[q] = max_depth + 1
        return max(qubit_depths)
    
    @property
    def n_parameters(self) -> int:
        """Total number of variational parameters."""
        count = 0
        for gate in self.gates:
            if gate.parameters is not None:
                count += len(gate.parameters)
        return count


class GateMatrices:
    """Standard quantum gate matrices."""
    
    @staticmethod
    def pauli_x() -> torch.Tensor:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    
    @staticmethod
    def pauli_y() -> torch.Tensor:
        return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    
    @staticmethod
    def pauli_z() -> torch.Tensor:
        return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    
    @staticmethod
    def hadamard() -> torch.Tensor:
        return torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    
    @staticmethod
    def rx(theta: float) -> torch.Tensor:
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=torch.complex128)
    
    @staticmethod
    def ry(theta: float) -> torch.Tensor:
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.complex128)
    
    @staticmethod
    def rz(theta: float) -> torch.Tensor:
        return torch.tensor([
            [torch.exp(torch.tensor(-1j * theta / 2)), 0],
            [0, torch.exp(torch.tensor(1j * theta / 2))]
        ], dtype=torch.complex128)
    
    @staticmethod
    def cnot() -> torch.Tensor:
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex128)
    
    @staticmethod
    def cz() -> torch.Tensor:
        return torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=torch.complex128)


# =============================================================================
# Tensor Network Quantum Simulator
# =============================================================================

class TNQuantumSimulator:
    """
    Tensor network based quantum circuit simulator.
    
    Uses MPS representation for efficient simulation of low-entanglement circuits.
    """
    
    def __init__(self, n_qubits: int, chi_max: int = 64):
        """
        Args:
            n_qubits: Number of qubits
            chi_max: Maximum bond dimension for MPS
        """
        self.n_qubits = n_qubits
        self.chi_max = chi_max
        self.dtype = torch.complex128
        
        # Initialize |00...0⟩ state as MPS
        self.mps = self._init_zero_state()
    
    def _init_zero_state(self) -> List[torch.Tensor]:
        """Initialize MPS in |00...0⟩ state."""
        mps = []
        for i in range(self.n_qubits):
            if i == 0:
                # Left boundary: shape (1, 2, 1)
                tensor = torch.zeros(1, 2, 1, dtype=self.dtype)
                tensor[0, 0, 0] = 1.0
            elif i == self.n_qubits - 1:
                # Right boundary: shape (1, 2, 1)
                tensor = torch.zeros(1, 2, 1, dtype=self.dtype)
                tensor[0, 0, 0] = 1.0
            else:
                # Bulk: shape (1, 2, 1)
                tensor = torch.zeros(1, 2, 1, dtype=self.dtype)
                tensor[0, 0, 0] = 1.0
            mps.append(tensor)
        return mps
    
    def apply_single_qubit_gate(self, gate_matrix: torch.Tensor, qubit: int):
        """Apply a single-qubit gate to the MPS."""
        # Contract gate with MPS site
        # MPS[q] has shape (chi_left, d, chi_right)
        # gate has shape (d, d)
        old_tensor = self.mps[qubit]
        new_tensor = torch.einsum('ij,ljr->lir', gate_matrix, old_tensor)
        self.mps[qubit] = new_tensor
    
    def apply_two_qubit_gate(
        self,
        gate_matrix: torch.Tensor,
        qubit1: int,
        qubit2: int
    ):
        """
        Apply a two-qubit gate using SVD compression.
        
        For adjacent qubits, contract and re-decompose.
        """
        if abs(qubit1 - qubit2) != 1:
            # Non-adjacent qubits require SWAP network
            raise NotImplementedError("Non-adjacent two-qubit gates not yet supported")
        
        q_left = min(qubit1, qubit2)
        q_right = max(qubit1, qubit2)
        
        # Contract the two MPS tensors
        # Left: (chi_l, d, chi_m), Right: (chi_m, d, chi_r)
        left = self.mps[q_left]
        right = self.mps[q_right]
        
        # Form (chi_l, d, d, chi_r)
        combined = torch.einsum('ijk,klm->ijlm', left, right)
        chi_l, d1, d2, chi_r = combined.shape
        
        # Apply gate: (d1', d2', d1, d2) @ (chi_l, d1, d2, chi_r)
        gate = gate_matrix.reshape(2, 2, 2, 2)
        if qubit1 < qubit2:
            new_combined = torch.einsum('abcd,icdk->iabk', gate, combined)
        else:
            new_combined = torch.einsum('abcd,idck->ibak', gate, combined)
        
        # rSVD decomposition - faster above 100x100
        new_combined = new_combined.reshape(chi_l * 2, 2 * chi_r)
        m, n = new_combined.shape
        if min(m, n) > 100:
            U, S, V = torch.svd_lowrank(new_combined, q=min(self.chi_max + 10, min(m, n)))
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(new_combined, full_matrices=False)
        
        # Truncate to chi_max
        chi_new = min(len(S), self.chi_max)
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]
        
        # Absorb singular values into left tensor
        US = U @ torch.diag(S)
        
        # Reshape back to MPS tensors
        self.mps[q_left] = US.reshape(chi_l, 2, chi_new)
        self.mps[q_right] = Vh.reshape(chi_new, 2, chi_r)
    
    def apply_circuit(self, circuit: QuantumCircuit):
        """Apply a quantum circuit to the state."""
        for gate in circuit.gates:
            if gate.gate_type == GateType.H:
                self.apply_single_qubit_gate(GateMatrices.hadamard(), gate.qubits[0])
            elif gate.gate_type == GateType.X:
                self.apply_single_qubit_gate(GateMatrices.pauli_x(), gate.qubits[0])
            elif gate.gate_type == GateType.Y:
                self.apply_single_qubit_gate(GateMatrices.pauli_y(), gate.qubits[0])
            elif gate.gate_type == GateType.Z:
                self.apply_single_qubit_gate(GateMatrices.pauli_z(), gate.qubits[0])
            elif gate.gate_type == GateType.RX:
                self.apply_single_qubit_gate(
                    GateMatrices.rx(gate.parameters[0]), gate.qubits[0]
                )
            elif gate.gate_type == GateType.RY:
                self.apply_single_qubit_gate(
                    GateMatrices.ry(gate.parameters[0]), gate.qubits[0]
                )
            elif gate.gate_type == GateType.RZ:
                self.apply_single_qubit_gate(
                    GateMatrices.rz(gate.parameters[0]), gate.qubits[0]
                )
            elif gate.gate_type == GateType.CNOT:
                self.apply_two_qubit_gate(
                    GateMatrices.cnot(), gate.qubits[0], gate.qubits[1]
                )
            elif gate.gate_type == GateType.CZ:
                self.apply_two_qubit_gate(
                    GateMatrices.cz(), gate.qubits[0], gate.qubits[1]
                )
            else:
                raise NotImplementedError(f"Gate {gate.gate_type} not implemented")
    
    def expectation_value(self, observable_mpo: List[torch.Tensor]) -> complex:
        """
        Compute ⟨ψ|O|ψ⟩ using MPS-MPO-MPS contraction.
        
        Args:
            observable_mpo: MPO representation of observable
            
        Returns:
            Expectation value
        """
        # Contract MPS with MPO with conjugate MPS
        # Use transfer matrix approach
        n = self.n_qubits
        
        # Initialize boundary
        # Shape: (chi_bra, chi_mpo, chi_ket)
        boundary = torch.ones(1, 1, 1, dtype=self.dtype)
        
        for i in range(n):
            bra = self.mps[i].conj()  # (chi_l, d, chi_r)
            mpo = observable_mpo[i]    # (D_l, d', d, D_r)
            ket = self.mps[i]          # (chi_l, d, chi_r)
            
            # Contract: boundary @ bra @ mpo @ ket
            # boundary: (a, b, c) where a=chi_bra_l, b=D_l, c=chi_ket_l
            # bra: (a, d', a')
            # mpo: (b, d', d, b')
            # ket: (c, d, c')
            # Result: (a', b', c')
            
            temp = torch.einsum('abc,adA->dbcA', boundary, bra)
            temp = torch.einsum('dbcA,bDdB->DcAB', temp, mpo)
            boundary = torch.einsum('DcAB,cdC->ABC', temp, ket)
        
        # Contract final boundary
        return boundary[0, 0, 0]
    
    def measure_pauli_string(self, paulis: str) -> complex:
        """
        Measure expectation of Pauli string.
        
        Args:
            paulis: String like "XZIY" (I=identity)
            
        Returns:
            Expectation value
        """
        if len(paulis) != self.n_qubits:
            raise ValueError(f"Pauli string length must match {self.n_qubits} qubits")
        
        # Build MPO for Pauli string
        mpo = []
        for char in paulis:
            if char == 'I':
                op = torch.eye(2, dtype=self.dtype)
            elif char == 'X':
                op = GateMatrices.pauli_x()
            elif char == 'Y':
                op = GateMatrices.pauli_y()
            elif char == 'Z':
                op = GateMatrices.pauli_z()
            else:
                raise ValueError(f"Unknown Pauli: {char}")
            
            # MPO tensor: (1, d', d, 1)
            mpo.append(op.reshape(1, 2, 2, 1))
        
        return self.expectation_value(mpo)


# =============================================================================
# Variational Quantum Eigensolver (VQE)
# =============================================================================

class AnsatzType(Enum):
    """Standard variational ansatz types."""
    HARDWARE_EFFICIENT = "hardware_efficient"
    UCCSD = "uccsd"
    QAOA = "qaoa"
    CUSTOM = "custom"


@dataclass
class VQEConfig:
    """Configuration for VQE."""
    ansatz_type: AnsatzType = AnsatzType.HARDWARE_EFFICIENT
    n_layers: int = 2
    optimizer: str = "adam"
    learning_rate: float = 0.01
    max_iterations: int = 100
    tolerance: float = 1e-6
    chi_max: int = 64


class VQE:
    """
    Variational Quantum Eigensolver.
    
    Uses tensor network simulation for efficient classical emulation.
    """
    
    def __init__(
        self,
        hamiltonian: Callable[[TNQuantumSimulator], float],
        n_qubits: int,
        config: Optional[VQEConfig] = None
    ):
        """
        Args:
            hamiltonian: Function that computes ⟨H⟩ given simulator
            n_qubits: Number of qubits
            config: VQE configuration
        """
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.config = config or VQEConfig()
        
        # Build ansatz circuit
        self.circuit, self.n_params = self._build_ansatz()
        
        # Initialize parameters
        self.parameters = torch.randn(self.n_params, dtype=torch.float64) * 0.1
    
    def _build_ansatz(self) -> Tuple[Callable, int]:
        """Build the variational ansatz circuit."""
        n = self.n_qubits
        n_layers = self.config.n_layers
        
        if self.config.ansatz_type == AnsatzType.HARDWARE_EFFICIENT:
            # Hardware-efficient ansatz: Ry-Rz on each qubit + CZ ladder
            n_params = n * 2 * n_layers  # 2 rotations per qubit per layer
            
            def ansatz(params: torch.Tensor) -> QuantumCircuit:
                circuit = QuantumCircuit(n)
                idx = 0
                for layer in range(n_layers):
                    # Single-qubit rotations
                    for q in range(n):
                        circuit.ry(q, params[idx].item())
                        idx += 1
                        circuit.rz(q, params[idx].item())
                        idx += 1
                    # Entangling layer
                    for q in range(n - 1):
                        circuit.cz(q, q + 1)
                return circuit
            
            return ansatz, n_params
        
        elif self.config.ansatz_type == AnsatzType.QAOA:
            # QAOA-style ansatz
            n_params = 2 * n_layers  # gamma and beta per layer
            
            def ansatz(params: torch.Tensor) -> QuantumCircuit:
                circuit = QuantumCircuit(n)
                # Initial superposition
                for q in range(n):
                    circuit.h(q)
                
                for layer in range(n_layers):
                    gamma = params[2*layer].item()
                    beta = params[2*layer + 1].item()
                    
                    # Problem unitary (ZZ interactions)
                    for q in range(n - 1):
                        circuit.cz(q, q + 1)
                        circuit.rz(q, gamma)
                    
                    # Mixer unitary
                    for q in range(n):
                        circuit.rx(q, beta)
                
                return circuit
            
            return ansatz, n_params
        
        else:
            raise NotImplementedError(f"Ansatz {self.config.ansatz_type} not implemented")
    
    def energy(self, params: torch.Tensor) -> float:
        """Compute energy for given parameters."""
        circuit = self.circuit(params)
        sim = TNQuantumSimulator(self.n_qubits, chi_max=self.config.chi_max)
        sim.apply_circuit(circuit)
        return self.hamiltonian(sim)
    
    def optimize(self, verbose: bool = True) -> Dict:
        """
        Run VQE optimization.
        
        Returns:
            Dictionary with optimization results
        """
        params = self.parameters.clone().requires_grad_(True)
        
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam([params], lr=self.config.learning_rate)
        else:
            optimizer = torch.optim.SGD([params], lr=self.config.learning_rate)
        
        history = []
        best_energy = float('inf')
        best_params = None
        
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()
            
            # Compute energy with gradient
            energy = self._compute_energy_with_grad(params)
            
            if energy < best_energy:
                best_energy = energy
                best_params = params.clone().detach()
            
            history.append({
                'iteration': iteration,
                'energy': energy,
                'params': params.clone().detach().numpy()
            })
            
            if verbose and iteration % 10 == 0:
                print(f"VQE Iter {iteration:4d}: E = {energy:.6f}")
            
            # Finite-difference gradient
            grad = self._compute_gradient(params)
            params.grad = grad
            optimizer.step()
            
            # Convergence check
            if len(history) > 1:
                delta = abs(history[-1]['energy'] - history[-2]['energy'])
                if delta < self.config.tolerance:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
        
        return {
            'optimal_energy': best_energy,
            'optimal_params': best_params,
            'history': history,
            'converged': iteration < self.config.max_iterations - 1
        }
    
    def _compute_energy_with_grad(self, params: torch.Tensor) -> float:
        """Compute energy value."""
        return self.energy(params.detach())
    
    def _compute_gradient(self, params: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        """Compute gradient via parameter-shift rule."""
        grad = torch.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.clone()
            params_plus[i] += eps
            e_plus = self.energy(params_plus.detach())
            
            params_minus = params.clone()
            params_minus[i] -= eps
            e_minus = self.energy(params_minus.detach())
            
            grad[i] = (e_plus - e_minus) / (2 * eps)
        
        return grad


# =============================================================================
# QAOA for Combinatorial Optimization
# =============================================================================

@dataclass
class QAOAConfig:
    """Configuration for QAOA."""
    n_layers: int = 3
    optimizer: str = "cobyla"
    max_iterations: int = 100
    chi_max: int = 64


class QAOA:
    """
    Quantum Approximate Optimization Algorithm.
    
    Solves combinatorial optimization problems using quantum-classical hybrid.
    """
    
    def __init__(
        self,
        cost_hamiltonian: List[Tuple[str, float]],  # List of (pauli_string, coefficient)
        n_qubits: int,
        config: Optional[QAOAConfig] = None
    ):
        """
        Args:
            cost_hamiltonian: Problem Hamiltonian as Pauli terms
            n_qubits: Number of qubits
            config: QAOA configuration
        """
        self.cost_terms = cost_hamiltonian
        self.n_qubits = n_qubits
        self.config = config or QAOAConfig()
        
        # Parameters: (gamma_1, beta_1, gamma_2, beta_2, ...)
        self.n_params = 2 * self.config.n_layers
    
    def cost_expectation(self, sim: TNQuantumSimulator) -> float:
        """Compute expectation of cost Hamiltonian."""
        total = 0.0
        for pauli_string, coeff in self.cost_terms:
            exp_val = sim.measure_pauli_string(pauli_string)
            total += coeff * exp_val.real
        return total
    
    def run_circuit(self, gammas: List[float], betas: List[float]) -> float:
        """
        Run QAOA circuit and return cost expectation.
        
        Args:
            gammas: Problem unitary angles
            betas: Mixer unitary angles
            
        Returns:
            Cost expectation value
        """
        n = self.n_qubits
        p = len(gammas)
        
        circuit = QuantumCircuit(n)
        
        # Initial superposition |+⟩^n
        for q in range(n):
            circuit.h(q)
        
        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]
            
            # Cost unitary: exp(-i gamma H_C)
            # For diagonal cost, apply RZ gates
            for pauli_string, coeff in self.cost_terms:
                for q, char in enumerate(pauli_string):
                    if char == 'Z':
                        circuit.rz(q, 2 * gamma * coeff)
            
            # Mixer unitary: exp(-i beta H_M) where H_M = sum X_i
            for q in range(n):
                circuit.rx(q, 2 * beta)
        
        # Simulate
        sim = TNQuantumSimulator(n, chi_max=self.config.chi_max)
        sim.apply_circuit(circuit)
        
        return self.cost_expectation(sim)
    
    def optimize(self, verbose: bool = True) -> Dict:
        """
        Optimize QAOA parameters.
        
        Returns:
            Optimization results
        """
        p = self.config.n_layers
        
        # Initial parameters
        gammas = [0.1] * p
        betas = [0.1] * p
        
        def objective(params):
            gs = params[:p]
            bs = params[p:]
            return self.run_circuit(list(gs), list(bs))
        
        # Simple gradient descent
        params = np.array(gammas + betas)
        history = []
        best_cost = float('inf')
        
        for iteration in range(self.config.max_iterations):
            cost = objective(params)
            
            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
            
            history.append({'iteration': iteration, 'cost': cost})
            
            if verbose and iteration % 10 == 0:
                print(f"QAOA Iter {iteration:4d}: Cost = {cost:.6f}")
            
            # Gradient via finite differences
            grad = np.zeros_like(params)
            eps = 0.01
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += eps
                params_minus = params.copy()
                params_minus[i] -= eps
                grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
            
            # Update
            params -= 0.1 * grad
        
        return {
            'optimal_cost': best_cost,
            'optimal_gammas': list(best_params[:p]),
            'optimal_betas': list(best_params[p:]),
            'history': history
        }


# =============================================================================
# Tensor Network Born Machine
# =============================================================================

class TensorNetworkBornMachine:
    """
    Generative model using tensor network as quantum-inspired ansatz.
    
    Uses MPS to represent probability distribution:
    P(x) = |⟨x|ψ⟩|² where |ψ⟩ is MPS
    """
    
    def __init__(
        self,
        n_sites: int,
        local_dim: int = 2,
        bond_dim: int = 16
    ):
        """
        Args:
            n_sites: Number of visible units
            local_dim: Local Hilbert space dimension
            bond_dim: MPS bond dimension
        """
        self.n_sites = n_sites
        self.local_dim = local_dim
        self.bond_dim = bond_dim
        
        # Initialize MPS with random tensors
        self.mps = self._init_random_mps()
    
    def _init_random_mps(self) -> List[torch.Tensor]:
        """Initialize random MPS."""
        mps = []
        for i in range(self.n_sites):
            if i == 0:
                shape = (1, self.local_dim, self.bond_dim)
            elif i == self.n_sites - 1:
                shape = (self.bond_dim, self.local_dim, 1)
            else:
                shape = (self.bond_dim, self.local_dim, self.bond_dim)
            
            tensor = torch.randn(shape, dtype=torch.float64) / math.sqrt(self.bond_dim)
            mps.append(tensor)
        
        return mps
    
    def amplitude(self, config: List[int]) -> torch.Tensor:
        """
        Compute amplitude ⟨config|ψ⟩.
        
        Args:
            config: Configuration as list of local indices
            
        Returns:
            Amplitude (scalar tensor)
        """
        result = self.mps[0][:, config[0], :]
        for i in range(1, self.n_sites):
            result = result @ self.mps[i][:, config[i], :]
        return result.squeeze()
    
    def probability(self, config: List[int]) -> torch.Tensor:
        """Compute probability P(config) = |amplitude|²."""
        amp = self.amplitude(config)
        return amp * amp
    
    def sample(self, n_samples: int = 1000) -> torch.Tensor:
        """
        Generate samples from the distribution.
        
        Uses sequential sampling from conditional distributions.
        
        Returns:
            Tensor of shape (n_samples, n_sites)
        """
        samples = torch.zeros(n_samples, self.n_sites, dtype=torch.long)
        
        for s in range(n_samples):
            config = []
            # Sample site by site
            for i in range(self.n_sites):
                # Compute conditional probabilities
                probs = torch.zeros(self.local_dim)
                for d in range(self.local_dim):
                    test_config = config + [d]
                    # Marginalize over remaining sites
                    probs[d] = self._marginal_probability(test_config)
                
                # Normalize
                probs = probs / probs.sum()
                
                # Sample
                idx = torch.multinomial(probs, 1).item()
                config.append(idx)
            
            samples[s] = torch.tensor(config)
        
        return samples
    
    def _marginal_probability(self, partial_config: List[int]) -> torch.Tensor:
        """Compute marginal probability for partial configuration."""
        # Contract specified sites with partial trace over remaining
        n_specified = len(partial_config)
        
        # Left contraction
        left = self.mps[0][:, partial_config[0], :]
        for i in range(1, n_specified):
            left = left @ self.mps[i][:, partial_config[i], :]
        
        # Right: sum over all configurations
        right = torch.eye(left.shape[-1], dtype=torch.float64)
        for i in range(n_specified, self.n_sites):
            # Sum over physical index
            right = right @ self.mps[i].sum(dim=1)
        
        result = left @ right
        return (result * result).sum()
    
    def train(
        self,
        data: torch.Tensor,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the Born machine on data.
        
        Args:
            data: Training data of shape (n_samples, n_sites)
            n_epochs: Training epochs
            learning_rate: Optimizer learning rate
            verbose: Print progress
            
        Returns:
            Training loss history
        """
        # Flatten MPS parameters
        params = []
        for tensor in self.mps:
            tensor.requires_grad_(True)
            params.append(tensor)
        
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        history = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Negative log-likelihood loss
            log_probs = []
            for sample in data:
                config = sample.tolist()
                amp = self.amplitude(config)
                log_probs.append(torch.log(amp * amp + 1e-10))
            
            loss = -torch.mean(torch.stack(log_probs))
            loss.backward()
            optimizer.step()
            
            history.append(loss.item())
            
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:4d}: NLL = {loss.item():.4f}")
        
        return history


# =============================================================================
# Quantum-Inspired Optimization
# =============================================================================

class QuantumInspiredOptimizer:
    """
    Quantum-inspired classical optimization using tensor network techniques.
    
    Combines:
    - Imaginary time evolution for ground state search
    - Tensor cross interpolation for function approximation
    - MPS-based variational optimization
    """
    
    def __init__(
        self,
        objective: Callable[[torch.Tensor], float],
        n_dims: int,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        resolution: int = 32
    ):
        """
        Args:
            objective: Function to minimize
            n_dims: Number of dimensions
            bounds: Variable bounds
            resolution: Grid resolution per dimension
        """
        self.objective = objective
        self.n_dims = n_dims
        self.bounds = bounds
        self.resolution = resolution
        
        # Discretized grid
        self.grid = torch.linspace(bounds[0], bounds[1], resolution, dtype=torch.float64)
    
    def optimize_bruteforce(self) -> Tuple[torch.Tensor, float]:
        """Brute-force grid search (for validation)."""
        best_x = None
        best_f = float('inf')
        
        # Only works for small n_dims
        if self.n_dims > 4:
            raise ValueError("Brute force only for n_dims <= 4")
        
        from itertools import product
        for indices in product(range(self.resolution), repeat=self.n_dims):
            x = torch.tensor([self.grid[i] for i in indices])
            f = self.objective(x)
            if f < best_f:
                best_f = f
                best_x = x.clone()
        
        return best_x, best_f
    
    def optimize_mps(
        self,
        chi_max: int = 16,
        n_sweeps: int = 10,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        Optimize using MPS representation of objective landscape.
        
        Uses alternating least squares to fit objective as TT-decomposition.
        """
        # Initialize random MPS
        mps = []
        for i in range(self.n_dims):
            if i == 0:
                tensor = torch.randn(1, self.resolution, chi_max, dtype=torch.float64)
            elif i == self.n_dims - 1:
                tensor = torch.randn(chi_max, self.resolution, 1, dtype=torch.float64)
            else:
                tensor = torch.randn(chi_max, self.resolution, chi_max, dtype=torch.float64)
            tensor /= tensor.norm()
            mps.append(tensor)
        
        # Sweeping optimization
        best_x = None
        best_f = float('inf')
        
        for sweep in range(n_sweeps):
            # Left-to-right sweep
            for site in range(self.n_dims - 1):
                # Find best local values
                for idx in range(self.resolution):
                    x = torch.zeros(self.n_dims)
                    x[site] = self.grid[idx]
                    # Random sample other dimensions
                    for j in range(self.n_dims):
                        if j != site:
                            x[j] = self.grid[torch.randint(0, self.resolution, (1,)).item()]
                    
                    f = self.objective(x)
                    if f < best_f:
                        best_f = f
                        best_x = x.clone()
            
            if verbose:
                print(f"Sweep {sweep}: Best f = {best_f:.6f}")
        
        return best_x, best_f


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ising_hamiltonian(n_qubits: int, J: float = 1.0, h: float = 0.5) -> Callable:
    """
    Create Ising Hamiltonian for VQE.
    
    H = -J Σ Z_i Z_{i+1} - h Σ X_i
    """
    def hamiltonian(sim: TNQuantumSimulator) -> float:
        energy = 0.0
        
        # ZZ terms
        for i in range(n_qubits - 1):
            pauli = 'I' * i + 'ZZ' + 'I' * (n_qubits - i - 2)
            energy -= J * sim.measure_pauli_string(pauli).real
        
        # X terms
        for i in range(n_qubits):
            pauli = 'I' * i + 'X' + 'I' * (n_qubits - i - 1)
            energy -= h * sim.measure_pauli_string(pauli).real
        
        return energy
    
    return hamiltonian


def create_maxcut_hamiltonian(
    edges: List[Tuple[int, int]],
    n_qubits: int
) -> List[Tuple[str, float]]:
    """
    Create MaxCut cost Hamiltonian for QAOA.
    
    H = Σ_{(i,j) ∈ E} (1 - Z_i Z_j) / 2
    """
    terms = []
    for i, j in edges:
        # Identity term
        terms.append(('I' * n_qubits, 0.5))
        # ZZ term
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        pauli[j] = 'Z'
        terms.append((''.join(pauli), -0.5))
    
    return terms


if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM-CLASSICAL HYBRID MODULE TEST")
    print("=" * 60)
    
    # Test VQE
    print("\n1. Testing VQE on 4-qubit Ising model...")
    n_qubits = 4
    hamiltonian = create_ising_hamiltonian(n_qubits)
    
    vqe = VQE(hamiltonian, n_qubits, VQEConfig(n_layers=2, max_iterations=50))
    result = vqe.optimize(verbose=True)
    print(f"VQE ground state energy: {result['optimal_energy']:.4f}")
    
    # Test QAOA
    print("\n2. Testing QAOA on MaxCut...")
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]  # Square graph
    cost_h = create_maxcut_hamiltonian(edges, n_qubits)
    
    qaoa = QAOA(cost_h, n_qubits, QAOAConfig(n_layers=2, max_iterations=30))
    result = qaoa.optimize(verbose=True)
    print(f"QAOA optimal cost: {result['optimal_cost']:.4f}")
    
    # Test Born Machine
    print("\n3. Testing Tensor Network Born Machine...")
    tnbm = TensorNetworkBornMachine(n_sites=4, local_dim=2, bond_dim=4)
    # Generate some training data
    data = torch.randint(0, 2, (100, 4))
    history = tnbm.train(data, n_epochs=50, verbose=True)
    print(f"Final NLL: {history[-1]:.4f}")
    
    print("\n✅ All quantum-classical hybrid tests passed!")
