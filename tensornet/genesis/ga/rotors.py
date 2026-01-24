"""
Rotor Operations for Geometric Algebra

Rotors are the geometric algebra equivalent of quaternions and rotation
matrices. They provide singularity-free rotations that compose naturally.

A rotor is an even multivector (scalar + bivector + ...) with unit magnitude.
R = exp(-θB/2) for rotation by angle θ in plane B.

Rotation: v' = RvR̃

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import torch
import math
from typing import Optional, Tuple

from tensornet.genesis.ga.multivector import (
    CliffordAlgebra, Multivector, vector
)
from tensornet.genesis.ga.products import (
    geometric_product, outer_product, scalar_product
)
from tensornet.genesis.ga.operations import (
    reverse, normalize, magnitude, magnitude_squared, even_part
)


def rotor_from_bivector(B: Multivector, angle: float) -> Multivector:
    """
    Create a rotor from a bivector and angle.
    
    R = cos(θ/2) - sin(θ/2)B̂
    
    where B̂ is the normalized bivector.
    
    Args:
        B: Bivector defining the rotation plane
        angle: Rotation angle in radians
        
    Returns:
        Rotor
    """
    if not B.is_bivector:
        raise ValueError("B must be a bivector")
    
    B_norm = magnitude(B)
    if B_norm < 1e-14:
        # Identity rotor for zero bivector
        return Multivector.scalar(B.algebra, 1.0)
    
    B_hat = B / B_norm
    half_angle = angle / 2
    
    scalar_part = Multivector.scalar(B.algebra, math.cos(half_angle))
    bivector_part = B_hat * (-math.sin(half_angle))
    
    return scalar_part + bivector_part


def rotor_from_angle_plane(angle: float, B: Multivector) -> Multivector:
    """
    Alias for rotor_from_bivector with swapped arguments.
    
    R = exp(-θB̂/2)
    
    Args:
        angle: Rotation angle in radians
        B: Bivector defining the rotation plane
        
    Returns:
        Rotor
    """
    return rotor_from_bivector(B, angle)


def rotor_from_vectors(a: Multivector, b: Multivector) -> Multivector:
    """
    Create a rotor that rotates vector a to vector b.
    
    Uses the formula: R = normalize(1 + ba)
    This gives the shortest rotation from a to b.
    
    Args:
        a: Source vector
        b: Target vector
        
    Returns:
        Rotor R such that RaR̃ ∝ b
    """
    if not a.is_vector or not b.is_vector:
        raise ValueError("Both inputs must be vectors")
    
    # Normalize inputs
    a_hat = normalize(a)
    b_hat = normalize(b)
    
    # R = normalize(1 + b*a)
    ba = geometric_product(b_hat, a_hat)
    one = Multivector.scalar(a.algebra, 1.0)
    
    R = one + ba
    
    # Handle anti-parallel case (180-degree rotation)
    R_mag = magnitude(R)
    if R_mag < 1e-10:
        # Find perpendicular vector for the plane
        # Use one of the basis vectors that's not parallel
        for i in range(a.algebra.n):
            perp = Multivector.basis_vector(a.algebra, i)
            wedge = outer_product(a_hat, perp)
            if magnitude(wedge) > 0.1:
                # Found a perpendicular direction
                B = outer_product(a_hat, perp)
                return rotor_from_bivector(normalize(B), math.pi)
        raise ValueError("Could not find rotation plane for anti-parallel vectors")
    
    return normalize(R)


def apply_rotor(R: Multivector, v: Multivector) -> Multivector:
    """
    Apply rotor R to multivector v.
    
    v' = RvR̃
    
    This is the "sandwich product" that performs rotation.
    
    Args:
        R: Rotor (unit even multivector)
        v: Multivector to rotate
        
    Returns:
        Rotated multivector
    """
    R_rev = reverse(R)
    Rv = geometric_product(R, v)
    return geometric_product(Rv, R_rev)


def rotor_log(R: Multivector) -> Multivector:
    """
    Compute the logarithm of a rotor.
    
    For R = cos(θ/2) - sin(θ/2)B̂:
    log(R) = -θB̂/2
    
    Returns a bivector representing the rotation.
    
    Args:
        R: Rotor
        
    Returns:
        Bivector -θB̂/2
    """
    algebra = R.algebra
    
    # Extract scalar and bivector parts
    scalar_coeff = R.coeffs[0]
    R_bivector = R.grade_projection(2)
    
    bivector_mag = magnitude(R_bivector)
    
    if bivector_mag < 1e-14:
        # Near identity rotor
        return Multivector.zero(algebra)
    
    # angle = 2 * atan2(|B|, scalar)
    angle = 2 * math.atan2(bivector_mag, scalar_coeff)
    
    # B̂ = B / |B|
    B_hat = R_bivector / bivector_mag
    
    # log(R) = -θB̂/2
    return B_hat * (-angle / 2)


def rotor_sqrt(R: Multivector) -> Multivector:
    """
    Compute the square root of a rotor.
    
    sqrt(R) is the rotor such that sqrt(R)² = R
    
    Args:
        R: Rotor
        
    Returns:
        Square root rotor
    """
    # log(R) gives -θB̂/2
    # sqrt(R) = exp(-θB̂/4) = exp(log(R)/2)
    log_R = rotor_log(R)
    
    from tensornet.genesis.ga.operations import exp
    return exp(log_R / 2)


def interpolate_rotors(R1: Multivector, R2: Multivector, 
                       t: float) -> Multivector:
    """
    Spherical linear interpolation (SLERP) between rotors.
    
    Returns rotor R(t) such that R(0) = R1 and R(1) = R2.
    
    Args:
        R1: Start rotor
        R2: End rotor
        t: Interpolation parameter in [0, 1]
        
    Returns:
        Interpolated rotor
    """
    if t <= 0:
        return R1
    if t >= 1:
        return R2
    
    # R(t) = R1 * (R1^{-1} R2)^t
    R1_inv = reverse(R1)  # For unit rotors, inverse = reverse
    delta = geometric_product(R1_inv, R2)
    
    # delta^t = exp(t * log(delta))
    log_delta = rotor_log(delta)
    
    from tensornet.genesis.ga.operations import exp
    delta_t = exp(log_delta * t)
    
    return geometric_product(R1, delta_t)


def compose_rotors(*rotors: Multivector) -> Multivector:
    """
    Compose multiple rotors.
    
    The composition applies rotors right-to-left:
    compose(R1, R2, R3) = R3 * R2 * R1
    
    So applying to a vector: (R3 R2 R1) v (R3 R2 R1)~ 
    first rotates by R1, then R2, then R3.
    
    Args:
        *rotors: Sequence of rotors
        
    Returns:
        Composed rotor
    """
    if not rotors:
        raise ValueError("Need at least one rotor")
    
    result = rotors[0]
    for R in rotors[1:]:
        result = geometric_product(R, result)
    
    # Re-normalize to avoid accumulated error
    return normalize(result)


def rotor_to_matrix(R: Multivector) -> torch.Tensor:
    """
    Convert a 3D rotor to a rotation matrix.
    
    Only works for Cl(3,0,0) (3D VGA).
    
    Args:
        R: Rotor in 3D VGA
        
    Returns:
        3x3 rotation matrix
    """
    algebra = R.algebra
    if algebra.n != 3:
        raise ValueError("rotor_to_matrix requires 3D algebra")
    
    # Apply rotor to each basis vector
    matrix = torch.zeros(3, 3, dtype=torch.float64)
    
    for i in range(3):
        e_i = Multivector.basis_vector(algebra, i)
        rotated = apply_rotor(R, e_i)
        matrix[:, i] = rotated.vector_part()
    
    return matrix


def matrix_to_rotor(mat: torch.Tensor, algebra: CliffordAlgebra) -> Multivector:
    """
    Convert a 3x3 rotation matrix to a rotor.
    
    Uses the Cayley transform approach.
    
    Args:
        mat: 3x3 rotation matrix
        algebra: 3D Clifford algebra
        
    Returns:
        Corresponding rotor
    """
    if algebra.n != 3:
        raise ValueError("matrix_to_rotor requires 3D algebra")
    if mat.shape != (3, 3):
        raise ValueError("Matrix must be 3x3")
    
    # Use Shepperd's method based on trace
    trace = mat.trace()
    
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    else:
        if mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = 2 * torch.sqrt(1 + mat[0, 0] - mat[1, 1] - mat[2, 2])
            w = (mat[2, 1] - mat[1, 2]) / s
            x = 0.25 * s
            y = (mat[0, 1] + mat[1, 0]) / s
            z = (mat[0, 2] + mat[2, 0]) / s
        elif mat[1, 1] > mat[2, 2]:
            s = 2 * torch.sqrt(1 + mat[1, 1] - mat[0, 0] - mat[2, 2])
            w = (mat[0, 2] - mat[2, 0]) / s
            x = (mat[0, 1] + mat[1, 0]) / s
            y = 0.25 * s
            z = (mat[1, 2] + mat[2, 1]) / s
        else:
            s = 2 * torch.sqrt(1 + mat[2, 2] - mat[0, 0] - mat[1, 1])
            w = (mat[1, 0] - mat[0, 1]) / s
            x = (mat[0, 2] + mat[2, 0]) / s
            y = (mat[1, 2] + mat[2, 1]) / s
            z = 0.25 * s
    
    # Rotor = w + x*e23 + y*e31 + z*e12
    # Note: quaternion (w, x, y, z) maps to rotor scalar + bivector parts
    coeffs = torch.zeros(algebra.dim, dtype=torch.float64)
    coeffs[0] = float(w)  # scalar
    
    # Bivector indices in Cl(3,0,0): e12=3, e13=5, e23=6
    coeffs[3] = float(z)   # e12
    coeffs[5] = float(y)   # e13 -> e31 with sign
    coeffs[6] = float(x)   # e23
    
    R = Multivector(algebra, coeffs)
    return normalize(R)


def euler_to_rotor(yaw: float, pitch: float, roll: float,
                   algebra: CliffordAlgebra) -> Multivector:
    """
    Create rotor from Euler angles (ZYX convention).
    
    Args:
        yaw: Rotation about z-axis (radians)
        pitch: Rotation about y-axis (radians)
        roll: Rotation about x-axis (radians)
        algebra: 3D Clifford algebra
        
    Returns:
        Composed rotor
    """
    if algebra.n != 3:
        raise ValueError("Euler angles require 3D algebra")
    
    from tensornet.genesis.ga.multivector import bivector
    
    # Bivectors for each axis
    e12 = bivector(algebra, {(0, 1): 1.0})  # xy-plane (z-axis rotation)
    e13 = bivector(algebra, {(0, 2): 1.0})  # xz-plane (y-axis rotation)
    e23 = bivector(algebra, {(1, 2): 1.0})  # yz-plane (x-axis rotation)
    
    R_yaw = rotor_from_bivector(e12, yaw)
    R_pitch = rotor_from_bivector(e13, pitch)
    R_roll = rotor_from_bivector(e23, roll)
    
    # ZYX: first roll, then pitch, then yaw
    return compose_rotors(R_roll, R_pitch, R_yaw)


def rotor_to_euler(R: Multivector) -> Tuple[float, float, float]:
    """
    Extract Euler angles from a 3D rotor.
    
    Returns (yaw, pitch, roll) in radians.
    Uses ZYX convention.
    
    Args:
        R: 3D rotor
        
    Returns:
        (yaw, pitch, roll) tuple
    """
    mat = rotor_to_matrix(R)
    
    # Standard extraction for ZYX
    if abs(mat[2, 0]) < 0.9999:
        pitch = -math.asin(float(mat[2, 0]))
        yaw = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
        roll = math.atan2(float(mat[2, 1]), float(mat[2, 2]))
    else:
        # Gimbal lock
        pitch = math.pi / 2 if mat[2, 0] < 0 else -math.pi / 2
        yaw = math.atan2(-float(mat[0, 1]), float(mat[1, 1]))
        roll = 0
    
    return yaw, pitch, roll
