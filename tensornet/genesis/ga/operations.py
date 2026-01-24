"""
Multivector Operations

Implements fundamental operations on multivectors:
- Reverse (grade involution reversal)
- Grade involution
- Clifford conjugate
- Magnitude and normalization
- Inverse
- Grade projection
- Even/odd decomposition
- Dual (with respect to pseudoscalar)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import torch
from typing import List, Optional

from tensornet.genesis.ga.multivector import (
    CliffordAlgebra, Multivector, pseudoscalar
)


def reverse(a: Multivector) -> Multivector:
    """
    Compute the reverse ã of a multivector.
    
    The reverse reverses the order of basis vectors in each blade.
    For a k-blade: ã = (-1)^{k(k-1)/2} a
    
    This is also called the "reversion" or "dagger" operation.
    
    Args:
        a: Multivector
        
    Returns:
        Reverse ã
    """
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for k in range(algebra.dim):
        grade = algebra.blade_grade(k)
        # Sign pattern: +--++--++--...
        # k(k-1)/2 counts the number of swaps to reverse
        sign = (-1) ** (grade * (grade - 1) // 2)
        result[k] = sign * a.coeffs[k]
    
    return Multivector(algebra, result)


def grade_involution(a: Multivector) -> Multivector:
    """
    Compute the grade involution â of a multivector.
    
    The grade involution negates odd-grade components:
    â = sum_k (-1)^k <a>_k
    
    This is also called the "main involution" or "parity".
    
    Args:
        a: Multivector
        
    Returns:
        Grade involution â
    """
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for k in range(algebra.dim):
        grade = algebra.blade_grade(k)
        sign = (-1) ** grade
        result[k] = sign * a.coeffs[k]
    
    return Multivector(algebra, result)


def clifford_conjugate(a: Multivector) -> Multivector:
    """
    Compute the Clifford conjugate a† of a multivector.
    
    The Clifford conjugate combines reverse and grade involution:
    a† = (ã)^ = (â)~
    
    Sign pattern by grade: +, -, -, +, +, -, -, +, ...
    
    Args:
        a: Multivector
        
    Returns:
        Clifford conjugate a†
    """
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for k in range(algebra.dim):
        grade = algebra.blade_grade(k)
        # Combine both involutions
        sign = ((-1) ** grade) * ((-1) ** (grade * (grade - 1) // 2))
        result[k] = sign * a.coeffs[k]
    
    return Multivector(algebra, result)


def magnitude_squared(a: Multivector) -> float:
    """
    Compute the squared magnitude |a|² = <aã>_0.
    
    This is the scalar part of the product of a with its reverse.
    
    For simple blades, this gives the squared norm.
    For general multivectors, this is the natural norm.
    
    Args:
        a: Multivector
        
    Returns:
        Squared magnitude (can be negative for some signatures)
    """
    from tensornet.genesis.ga.products import geometric_product
    
    a_rev = reverse(a)
    product = geometric_product(a, a_rev)
    return float(product.coeffs[0])


def magnitude(a: Multivector) -> float:
    """
    Compute the magnitude |a| = sqrt(|<aã>_0|).
    
    Takes absolute value before sqrt to handle negative signatures.
    
    Args:
        a: Multivector
        
    Returns:
        Magnitude (always non-negative)
    """
    mag_sq = magnitude_squared(a)
    return float(torch.sqrt(torch.tensor(abs(mag_sq))))


def normalize(a: Multivector) -> Multivector:
    """
    Normalize multivector to unit magnitude.
    
    Args:
        a: Multivector
        
    Returns:
        Normalized multivector a / |a|
    """
    mag = magnitude(a)
    if mag < 1e-14:
        raise ValueError("Cannot normalize zero multivector")
    return a / mag


def inverse(a: Multivector) -> Multivector:
    """
    Compute the inverse a^{-1} such that a * a^{-1} = 1.
    
    For a versor (product of vectors): a^{-1} = ã / |a|²
    For general multivectors, this may not exist.
    
    Args:
        a: Multivector (should be a versor for guaranteed inverse)
        
    Returns:
        Inverse multivector
    """
    mag_sq = magnitude_squared(a)
    if abs(mag_sq) < 1e-14:
        raise ValueError("Multivector has zero magnitude squared, cannot invert")
    
    a_rev = reverse(a)
    return a_rev / mag_sq


def grade_projection(a: Multivector, grade: int) -> Multivector:
    """
    Project multivector onto a single grade.
    
    <a>_k extracts only the grade-k components.
    
    Args:
        a: Multivector
        grade: Grade to project onto
        
    Returns:
        Grade-k part of a
    """
    return a.grade_projection(grade)


def even_part(a: Multivector) -> Multivector:
    """
    Extract the even-grade components.
    
    a_+ = sum_{k even} <a>_k
    
    Args:
        a: Multivector
        
    Returns:
        Even part
    """
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for k in range(algebra.dim):
        grade = algebra.blade_grade(k)
        if grade % 2 == 0:
            result[k] = a.coeffs[k]
    
    return Multivector(algebra, result)


def odd_part(a: Multivector) -> Multivector:
    """
    Extract the odd-grade components.
    
    a_- = sum_{k odd} <a>_k
    
    Args:
        a: Multivector
        
    Returns:
        Odd part
    """
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for k in range(algebra.dim):
        grade = algebra.blade_grade(k)
        if grade % 2 == 1:
            result[k] = a.coeffs[k]
    
    return Multivector(algebra, result)


def dual(a: Multivector) -> Multivector:
    """
    Compute the dual a* with respect to the pseudoscalar.
    
    a* = a⌋I^{-1}
    
    where I is the pseudoscalar.
    
    Args:
        a: Multivector
        
    Returns:
        Dual multivector
    """
    from tensornet.genesis.ga.products import left_contraction
    
    algebra = a.algebra
    I = pseudoscalar(algebra)
    I_inv = inverse(I)
    
    return left_contraction(a, I_inv)


def undual(a: Multivector) -> Multivector:
    """
    Compute the undual (inverse of dual).
    
    (a*)* = a for proper involution.
    
    Args:
        a: Multivector (assumed to be a dual)
        
    Returns:
        Undualized multivector
    """
    from tensornet.genesis.ga.products import left_contraction
    
    algebra = a.algebra
    I = pseudoscalar(algebra)
    
    return left_contraction(a, I)


def project_onto(a: Multivector, b: Multivector) -> Multivector:
    """
    Project multivector a onto blade b.
    
    P_b(a) = (a⌋b)⌋b^{-1}
    
    Args:
        a: Multivector to project
        b: Blade to project onto
        
    Returns:
        Projection of a onto b
    """
    from tensornet.genesis.ga.products import left_contraction
    
    b_inv = inverse(b)
    contraction = left_contraction(a, b)
    return left_contraction(contraction, b_inv)


def reject_from(a: Multivector, b: Multivector) -> Multivector:
    """
    Reject multivector a from blade b.
    
    The rejection is the component of a orthogonal to b:
    R_b(a) = a - P_b(a)
    
    Args:
        a: Multivector
        b: Blade to reject from
        
    Returns:
        Rejection of a from b
    """
    projection = project_onto(a, b)
    return a - projection


def reflect_in(a: Multivector, n: Multivector) -> Multivector:
    """
    Reflect multivector a in the hyperplane with normal n.
    
    For a vector: a' = -nan
    
    Args:
        a: Multivector to reflect
        n: Normal vector (grade 1)
        
    Returns:
        Reflected multivector
    """
    from tensornet.genesis.ga.products import geometric_product
    
    if not n.is_vector:
        raise ValueError("Normal must be a vector (grade 1)")
    
    n_norm = normalize(n)
    na = geometric_product(n_norm, a)
    nan = geometric_product(na, n_norm)
    return -nan


def exp(a: Multivector, terms: int = 50) -> Multivector:
    """
    Compute the exponential exp(a) using Taylor series.
    
    For bivectors, this gives a rotor.
    For general multivectors, convergence depends on structure.
    
    Args:
        a: Multivector (typically a bivector for rotors)
        terms: Number of Taylor series terms
        
    Returns:
        exp(a)
    """
    from tensornet.genesis.ga.products import geometric_product
    
    algebra = a.algebra
    result = Multivector.scalar(algebra, 1.0)
    term = Multivector.scalar(algebra, 1.0)
    
    for n in range(1, terms):
        term = geometric_product(term, a) / n
        result = result + term
        
        # Early termination if converged
        if term.norm() < 1e-15:
            break
    
    return result


def log(a: Multivector, terms: int = 50) -> Multivector:
    """
    Compute the logarithm log(a) using series expansion.
    
    Assumes a is close to identity (1 + x with |x| < 1).
    For rotors, use specialized rotor_log.
    
    Args:
        a: Multivector close to identity
        terms: Number of series terms
        
    Returns:
        log(a)
    """
    from tensornet.genesis.ga.products import geometric_product
    
    algebra = a.algebra
    
    # x = a - 1
    x = a - Multivector.scalar(algebra, 1.0)
    
    if x.norm() >= 1:
        raise ValueError("log requires |a - 1| < 1")
    
    # log(1 + x) = x - x²/2 + x³/3 - ...
    result = Multivector.zero(algebra)
    x_power = x
    
    for n in range(1, terms):
        sign = (-1) ** (n + 1)
        result = result + x_power * (sign / n)
        x_power = geometric_product(x_power, x)
        
        if x_power.norm() < 1e-15:
            break
    
    return result
