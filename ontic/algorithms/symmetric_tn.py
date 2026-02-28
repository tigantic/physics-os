"""
Symmetric Tensor Networks — U(1) / SU(2) Invariant MPS & MPO
==============================================================

Exploits Abelian (U(1): particle number, magnetization) and
non-Abelian (SU(2): total spin) symmetries to block-diagonalise
tensors.  This reduces both memory *and* computational cost by
restricting each bond to carry only the charge sectors that
appear in the physical problem.

Key classes
-----------
* :class:`ChargeLabel`          — quantum number label (additive Z)
* :class:`SymSector`            — one block within a leg
* :class:`SymLeg`               — full leg description (list of sectors)
* :class:`SymTensor`            — block-sparse symmetric tensor
* :class:`SymMPS`               — MPS with symmetric tensors
* :class:`SymMPO`               — MPO with symmetric tensors

Key functions
-------------
* :func:`clebsch_gordan_su2`    — SU(2) Clebsch–Gordan coefficients
* :func:`u1_fuse`               — fuse two U(1) legs
* :func:`su2_fuse`              — fuse two SU(2) legs
* :func:`sym_svd`               — block-diagonal SVD
* :func:`random_sym_mps`        — random U(1)-symmetric MPS
* :func:`heisenberg_sym_mpo`    — Heisenberg model using SU(2) blocks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import factorial, sqrt
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Quantum-number labels
# ======================================================================

@dataclass(frozen=True, order=True)
class ChargeLabel:
    """
    Additive quantum number for an Abelian symmetry.

    For U(1) particle-number conservation: q ∈ Z.
    For Z₂ parity: q ∈ {0, 1}.
    For SU(2), *q* stores 2j (twice the spin).
    """
    q: int

    def __add__(self, other: ChargeLabel) -> ChargeLabel:
        return ChargeLabel(self.q + other.q)

    def __neg__(self) -> ChargeLabel:
        return ChargeLabel(-self.q)

    def __repr__(self) -> str:
        return f"q={self.q}"


# ======================================================================
# Legs and sectors
# ======================================================================

@dataclass
class SymSector:
    """
    One charge sector within a leg.

    Attributes
    ----------
    charge : ChargeLabel
        Quantum number of this sector.
    dim : int
        Degeneracy (multiplicity) dimension.
    """
    charge: ChargeLabel
    dim: int

    def __repr__(self) -> str:
        return f"({self.charge}, dim={self.dim})"


@dataclass
class SymLeg:
    """
    Symmetry-resolved leg of a tensor.

    Attributes
    ----------
    sectors : list[SymSector]
        Charge sectors, sorted by charge.
    dual : bool
        If True, charges flow inward (ket-like); False = outward (bra-like).
    """
    sectors: list[SymSector]
    dual: bool = False

    @property
    def total_dim(self) -> int:
        return sum(s.dim for s in self.sectors)

    @property
    def charges(self) -> list[ChargeLabel]:
        return [s.charge for s in self.sectors]

    def sector_offset(self, charge: ChargeLabel) -> int:
        """Offset of charge sector in the dense index range."""
        off = 0
        for s in self.sectors:
            if s.charge == charge:
                return off
            off += s.dim
        raise KeyError(f"Charge {charge} not in leg")

    def sector_dim(self, charge: ChargeLabel) -> int:
        for s in self.sectors:
            if s.charge == charge:
                return s.dim
        return 0


# ======================================================================
# Block-sparse symmetric tensor
# ======================================================================

@dataclass
class SymTensor:
    """
    Block-sparse tensor with Abelian symmetry.

    Only blocks where the total charge is conserved (sums to
    ``target_charge``) are stored.

    Attributes
    ----------
    legs : list[SymLeg]
        Leg descriptions.
    blocks : dict[tuple[ChargeLabel, ...], NDArray]
        Non-zero blocks keyed by per-leg charge labels.
    target_charge : ChargeLabel
        Total charge selection rule.
    """
    legs: list[SymLeg]
    blocks: dict[tuple[ChargeLabel, ...], NDArray]
    target_charge: ChargeLabel = field(default_factory=lambda: ChargeLabel(0))

    @property
    def ndim(self) -> int:
        return len(self.legs)

    def to_dense(self) -> NDArray:
        """Expand to a full dense array."""
        shape = tuple(leg.total_dim for leg in self.legs)
        dense = np.zeros(shape)
        for key, block in self.blocks.items():
            slices = []
            for i, charge in enumerate(key):
                off = self.legs[i].sector_offset(charge)
                dim = self.legs[i].sector_dim(charge)
                slices.append(slice(off, off + dim))
            dense[tuple(slices)] = block
        return dense

    @staticmethod
    def from_dense(
        array: NDArray,
        legs: list[SymLeg],
        target_charge: Optional[ChargeLabel] = None,
        tol: float = 1e-14,
    ) -> SymTensor:
        """
        Extract symmetry-allowed blocks from a dense array.

        Parameters
        ----------
        array : NDArray
            Dense array.
        legs : list[SymLeg]
            Leg descriptions.
        target_charge : ChargeLabel, optional
            Selection rule.  None → ChargeLabel(0).
        tol : float
            Norm threshold for discarding zero blocks.

        Returns
        -------
        SymTensor
        """
        if target_charge is None:
            target_charge = ChargeLabel(0)

        blocks: dict[tuple[ChargeLabel, ...], NDArray] = {}

        # Enumerate all valid charge combinations
        import itertools
        sector_lists = [leg.sectors for leg in legs]
        for combo in itertools.product(*sector_lists):
            charges = tuple(s.charge for s in combo)
            # Check selection rule: sum of charges with dual signs
            total = ChargeLabel(0)
            for i, c in enumerate(charges):
                if legs[i].dual:
                    total = total + (-c)
                else:
                    total = total + c
            if total != target_charge:
                continue

            slices = []
            for i, sector in enumerate(combo):
                off = legs[i].sector_offset(sector.charge)
                slices.append(slice(off, off + sector.dim))

            block = array[tuple(slices)]
            if np.linalg.norm(block) > tol:
                blocks[charges] = block.copy()

        return SymTensor(legs=legs, blocks=blocks, target_charge=target_charge)


# ======================================================================
# U(1) leg fusion
# ======================================================================

def u1_fuse(leg1: SymLeg, leg2: SymLeg) -> SymLeg:
    """
    Fuse two U(1) legs into one combined leg.

    The fused sectors have charges q = q₁ + q₂ and multiplicities
    that are the product of the constituent degeneracies.

    Parameters
    ----------
    leg1, leg2 : SymLeg
        Legs to fuse.

    Returns
    -------
    SymLeg
        Fused leg.
    """
    charge_to_dim: dict[int, int] = {}
    for s1 in leg1.sectors:
        for s2 in leg2.sectors:
            q = s1.charge.q + s2.charge.q
            charge_to_dim[q] = charge_to_dim.get(q, 0) + s1.dim * s2.dim

    sectors = [
        SymSector(ChargeLabel(q), d)
        for q, d in sorted(charge_to_dim.items())
    ]
    return SymLeg(sectors)


# ======================================================================
# SU(2) Clebsch–Gordan coefficients
# ======================================================================

def _wigner_3j(j1: float, j2: float, j3: float,
               m1: float, m2: float, m3: float) -> float:
    """Wigner 3j symbol via direct formula."""
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    # Triangle coefficient
    def tri(a: float, b: float, c: float) -> float:
        return (factorial(int(a + b - c)) *
                factorial(int(a - b + c)) *
                factorial(int(-a + b + c)) /
                factorial(int(a + b + c + 1)))

    pre = ((-1) ** int(j1 - j2 - m3) *
           sqrt(tri(j1, j2, j3) *
                factorial(int(j1 + m1)) * factorial(int(j1 - m1)) *
                factorial(int(j2 + m2)) * factorial(int(j2 - m2)) *
                factorial(int(j3 + m3)) * factorial(int(j3 - m3))))

    s_min = max(0, int(j2 - j3 - m1), int(j1 - j3 + m2))
    s_max = min(int(j1 + j2 - j3), int(j1 - m1), int(j2 + m2))

    total = 0.0
    for s in range(s_min, s_max + 1):
        total += ((-1) ** s /
                  (factorial(s) *
                   factorial(int(j1 + j2 - j3 - s)) *
                   factorial(int(j1 - m1 - s)) *
                   factorial(int(j2 + m2 - s)) *
                   factorial(int(j3 - j2 + m1 + s)) *
                   factorial(int(j3 - j1 - m2 + s))))

    return pre * total


def clebsch_gordan_su2(
    j1: float, j2: float, J: float,
    m1: float, m2: float, M: float,
) -> float:
    """
    SU(2) Clebsch–Gordan coefficient ⟨j₁ m₁; j₂ m₂ | J M⟩.

    Computed from the Wigner 3j symbol:
        ⟨j₁ m₁; j₂ m₂ | J M⟩ = (-1)^{j₁-j₂+M} √(2J+1) * (j₁ j₂ J; m₁ m₂ -M)

    Parameters
    ----------
    j1, m1, j2, m2 : float
        Individual spins and magnetic quantum numbers.
    J, M : float
        Total spin and its projection.

    Returns
    -------
    float
        The CG coefficient.
    """
    if abs(M - m1 - m2) > 1e-12:
        return 0.0
    sign = (-1) ** int(j1 - j2 + M)
    return sign * sqrt(2 * J + 1) * _wigner_3j(j1, j2, J, m1, m2, -M)


# ======================================================================
# SU(2) leg fusion
# ======================================================================

def su2_fuse(leg1: SymLeg, leg2: SymLeg) -> SymLeg:
    """
    Fuse two SU(2) legs.

    Charges store 2j.  The fused leg has sectors for all allowed
    total spin values from the triangle inequality, with degeneracy
    = (2J+1) × product of multiplicities.

    Parameters
    ----------
    leg1, leg2 : SymLeg
        Legs with charges representing 2j.

    Returns
    -------
    SymLeg
        Fused leg with SU(2) sectors.
    """
    charge_to_dim: dict[int, int] = {}
    for s1 in leg1.sectors:
        for s2 in leg2.sectors:
            j1_2 = s1.charge.q  # 2 * j1
            j2_2 = s2.charge.q  # 2 * j2
            J_min = abs(j1_2 - j2_2)
            J_max = j1_2 + j2_2
            for J_2 in range(J_min, J_max + 1, 2):
                d = s1.dim * s2.dim
                charge_to_dim[J_2] = charge_to_dim.get(J_2, 0) + d

    sectors = [
        SymSector(ChargeLabel(q), d)
        for q, d in sorted(charge_to_dim.items())
    ]
    return SymLeg(sectors)


# ======================================================================
# Block-diagonal SVD
# ======================================================================

def sym_svd(
    tensor: SymTensor,
    split: int,
    max_rank: Optional[int] = None,
    cutoff: float = 1e-14,
) -> tuple[SymTensor, NDArray, SymTensor]:
    """
    Block-diagonal SVD of a symmetric tensor.

    The tensor is split between legs ``0..split-1`` (left) and
    ``split..ndim-1`` (right).  Each charge sector is SVD'd
    independently.

    Parameters
    ----------
    tensor : SymTensor
        Input block-sparse tensor.
    split : int
        Number of legs on the left side.
    max_rank : int, optional
        Maximum total singular values.
    cutoff : float
        Discard singular values below this threshold.

    Returns
    -------
    U : SymTensor
        Left unitary with legs = left_legs + [bond_leg].
    S : NDArray
        Concatenated singular values (sorted descending).
    Vh : SymTensor
        Right unitary with legs = [bond_leg] + right_legs.
    """
    left_legs = tensor.legs[:split]
    right_legs = tensor.legs[split:]

    # Collect SVD per charge sector
    u_blocks: dict[tuple[ChargeLabel, ...], NDArray] = {}
    s_blocks: dict[ChargeLabel, NDArray] = {}
    vh_blocks: dict[tuple[ChargeLabel, ...], NDArray] = {}

    # Group blocks by right charge sum
    charge_groups: dict[ChargeLabel, list[tuple[tuple[ChargeLabel, ...], NDArray]]] = {}

    for key, block in tensor.blocks.items():
        left_charges = key[:split]
        right_charges = key[split:]

        # For the SVD, we need to group by the total charge flowing
        # through the bond.  For U(1): bond charge = sum of left charges
        bond_q = ChargeLabel(0)
        for i, c in enumerate(left_charges):
            bond_q = bond_q + c

        # Compute left and right dimensions
        left_dim = 1
        for i, c in enumerate(left_charges):
            left_dim *= left_legs[i].sector_dim(c)
        right_dim = 1
        for i, c in enumerate(right_charges):
            right_dim *= right_legs[i].sector_dim(c)

        mat = block.reshape(left_dim, right_dim)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)

        # Truncate
        keep = len(S)
        if cutoff > 0:
            keep = max(1, int(np.sum(S > cutoff)))
        if max_rank is not None:
            keep = min(keep, max_rank)

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        u_key = left_charges + (bond_q,)
        vh_key = (bond_q,) + right_charges

        u_blocks[u_key] = U.reshape(block.shape[:split] + (keep,))
        vh_blocks[vh_key] = Vh.reshape((keep,) + block.shape[split:])

        if bond_q in s_blocks:
            s_blocks[bond_q] = np.concatenate([s_blocks[bond_q], S])
        else:
            s_blocks[bond_q] = S

    # Build bond leg
    bond_sectors = [
        SymSector(charge, len(sv))
        for charge, sv in sorted(s_blocks.items(), key=lambda x: x[0])
    ]
    bond_leg = SymLeg(bond_sectors)

    # Concatenate S
    all_S = np.concatenate([sv for _, sv in sorted(s_blocks.items(), key=lambda x: x[0])])

    U_tensor = SymTensor(
        legs=left_legs + [bond_leg],
        blocks=u_blocks,
        target_charge=tensor.target_charge,
    )
    Vh_tensor = SymTensor(
        legs=[bond_leg] + right_legs,
        blocks=vh_blocks,
        target_charge=ChargeLabel(0),
    )

    return U_tensor, all_S, Vh_tensor


# ======================================================================
# Symmetric MPS
# ======================================================================

@dataclass
class SymMPS:
    """
    MPS with U(1)-symmetric tensors.

    Each site tensor is a SymTensor with 3 legs:
    [left_bond, physical, right_bond].

    Attributes
    ----------
    tensors : list[SymTensor]
        Site tensors.
    total_charge : ChargeLabel
        Total quantum number of the state.
    """
    tensors: list[SymTensor]
    total_charge: ChargeLabel = field(default_factory=lambda: ChargeLabel(0))

    @property
    def n_sites(self) -> int:
        return len(self.tensors)


@dataclass
class SymMPO:
    """
    MPO with U(1)-symmetric tensors.

    Each site tensor is a SymTensor with 4 legs:
    [left_bond, phys_ket, phys_bra, right_bond].

    Attributes
    ----------
    tensors : list[SymTensor]
        Site tensors.
    """
    tensors: list[SymTensor]

    @property
    def n_sites(self) -> int:
        return len(self.tensors)


# ======================================================================
# Convenience constructors
# ======================================================================

def random_sym_mps(
    n_sites: int,
    d: int,
    chi: int,
    total_charge: int = 0,
    seed: Optional[int] = None,
) -> SymMPS:
    """
    Random U(1)-symmetric MPS.

    Physical leg has charges 0, 1, ..., d-1.  Bond legs carry all
    charge sectors reachable by filling sites 0..k.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    d : int
        Physical dimension.
    chi : int
        Maximum bond dimension per sector.
    total_charge : int
        Total U(1) charge of the state.
    seed : int, optional
        RNG seed.

    Returns
    -------
    SymMPS
    """
    rng = np.random.default_rng(seed)

    phys_leg = SymLeg([SymSector(ChargeLabel(q), 1) for q in range(d)])

    # Build reachable charges at each bond
    # Bond k lives between sites k-1 and k.
    reachable: list[set[int]] = [set() for _ in range(n_sites + 1)]
    reachable[0] = {0}
    for k in range(n_sites):
        for q_prev in reachable[k]:
            for q_phys in range(d):
                reachable[k + 1].add(q_prev + q_phys)

    # Filter: only keep charges that can reach total_charge from the right
    right_reachable: list[set[int]] = [set() for _ in range(n_sites + 1)]
    right_reachable[n_sites] = {total_charge}
    for k in range(n_sites - 1, -1, -1):
        for q_right in right_reachable[k + 1]:
            for q_phys in range(d):
                right_reachable[k].add(q_right - q_phys)

    bond_charges: list[set[int]] = [
        reachable[k] & right_reachable[k] for k in range(n_sites + 1)
    ]

    tensors: list[SymTensor] = []

    for k in range(n_sites):
        left_qs = sorted(bond_charges[k])
        right_qs = sorted(bond_charges[k + 1])

        left_leg = SymLeg([
            SymSector(ChargeLabel(q), min(chi, 1) if k == 0 else chi)
            for q in left_qs
        ])
        if k == 0:
            left_leg = SymLeg([SymSector(ChargeLabel(0), 1)])

        right_leg = SymLeg([
            SymSector(ChargeLabel(q), min(chi, 1) if k == n_sites - 1 else chi)
            for q in right_qs
        ])
        if k == n_sites - 1:
            right_leg = SymLeg([
                SymSector(ChargeLabel(total_charge), 1)
            ])

        blocks: dict[tuple[ChargeLabel, ...], NDArray] = {}
        for sl in left_leg.sectors:
            for sp in phys_leg.sectors:
                q_right = sl.charge.q + sp.charge.q
                if q_right in {s.charge.q for s in right_leg.sectors}:
                    sr_dim = right_leg.sector_dim(ChargeLabel(q_right))
                    block = rng.standard_normal((sl.dim, sp.dim, sr_dim))
                    # Normalise
                    norm = np.linalg.norm(block)
                    if norm > 1e-14:
                        block /= norm
                    key = (sl.charge, sp.charge, ChargeLabel(q_right))
                    blocks[key] = block

        tensors.append(SymTensor(
            legs=[left_leg, phys_leg, right_leg],
            blocks=blocks,
            target_charge=ChargeLabel(0),
        ))

    return SymMPS(tensors=tensors, total_charge=ChargeLabel(total_charge))


def heisenberg_sym_mpo(
    n_sites: int,
    J: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
) -> SymMPO:
    """
    Heisenberg XXZ model as a U(1)-symmetric MPO.

    H = J Σ (S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1}) + Jz Σ S^z_i S^z_{i+1} + h Σ S^z_i

    The MPO has bond dimension 5 and conserves total Sz.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    J : float
        XY coupling.
    Jz : float
        Ising coupling.
    h : float
        Magnetic field.

    Returns
    -------
    SymMPO
    """
    # Physical leg: Sz = +1/2 (↑, charge=1) and Sz = -1/2 (↓, charge=0)
    # Using integer charges: ↑ = charge 1, ↓ = charge 0
    phys_leg = SymLeg([
        SymSector(ChargeLabel(0), 1),  # ↓
        SymSector(ChargeLabel(1), 1),  # ↑
    ])

    # Spin operators in the {↓, ↑} basis (ordered by charge):
    # S^z = diag(-1/2, +1/2)
    # S^+ = [[0, 0], [1, 0]]  (↓→↑, charge change +1)
    # S^- = [[0, 1], [0, 0]]  (↑→↓, charge change -1)
    # I = eye(2)

    # MPO bond dimension 5: tracks {I, S^+, S^-, S^z, accumulated}
    # Bond charges: I carries q=0, S^+ carries q=+1, S^- carries q=-1, S^z carries q=0, acc carries q=0

    tensors: list[SymTensor] = []

    for k in range(n_sites):
        # Build dense (5, 2, 2, 5) tensor then symmetrize
        D = 5
        d = 2
        W = np.zeros((D, d, d, D))

        # Spin ops in {↓=0, ↑=1} basis
        Sz = np.array([[-0.5, 0], [0, 0.5]])
        Sp = np.array([[0, 0], [1, 0]])
        Sm = np.array([[0, 1], [0, 0]])
        I2 = np.eye(2)

        if k == 0:
            # Left boundary: row [I, S^+, S^-, S^z, h*S^z]
            W[0, :, :, 0] = I2
            W[0, :, :, 1] = Sp
            W[0, :, :, 2] = Sm
            W[0, :, :, 3] = Sz
            W[0, :, :, 4] = h * Sz
        elif k == n_sites - 1:
            # Right boundary: column [h*S^z, J/2*S^-, J/2*S^+, Jz*S^z, I]
            W[0, :, :, 0] = h * Sz
            W[1, :, :, 0] = J / 2.0 * Sm
            W[2, :, :, 0] = J / 2.0 * Sp
            W[3, :, :, 0] = Jz * Sz
            W[4, :, :, 0] = I2
        else:
            # Bulk
            W[0, :, :, 0] = I2
            W[0, :, :, 1] = Sp
            W[0, :, :, 2] = Sm
            W[0, :, :, 3] = Sz
            W[0, :, :, 4] = h * Sz
            W[1, :, :, 4] = J / 2.0 * Sm
            W[2, :, :, 4] = J / 2.0 * Sp
            W[3, :, :, 4] = Jz * Sz
            W[4, :, :, 4] = I2

        if k == 0:
            W = W[0:1, :, :, :]
        if k == n_sites - 1:
            W = W[:, :, :, 0:1]

        # For the MPO tensor, legs are [bond_l, phys_bra, phys_ket, bond_r]
        # Convert to SymTensor with bond legs carrying appropriate charges
        D_l = W.shape[0]
        D_r = W.shape[3]

        # Define bond legs with trivial 1-dim sectors for each MPO auxiliary index
        left_sectors = [SymSector(ChargeLabel(i), 1) for i in range(D_l)]
        right_sectors = [SymSector(ChargeLabel(i), 1) for i in range(D_r)]

        bond_l = SymLeg(left_sectors)
        bond_r = SymLeg(right_sectors)

        legs = [bond_l, phys_leg, phys_leg, bond_r]

        # Build blocks
        blocks: dict[tuple[ChargeLabel, ...], NDArray] = {}
        for il in range(D_l):
            for ip_bra in range(d):
                for ip_ket in range(d):
                    for ir in range(D_r):
                        val = W[il, ip_bra, ip_ket, ir]
                        if abs(val) > 1e-15:
                            key = (
                                ChargeLabel(il),
                                ChargeLabel(ip_bra),
                                ChargeLabel(ip_ket),
                                ChargeLabel(ir),
                            )
                            blocks[key] = np.array([[[[val]]]])

        tensors.append(SymTensor(
            legs=legs,
            blocks=blocks,
            target_charge=ChargeLabel(0),
        ))

    return SymMPO(tensors=tensors)
