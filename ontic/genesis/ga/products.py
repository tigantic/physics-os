"""
Geometric Algebra Products

Implements the various products in Clifford algebra:
- Geometric product (full product)
- Inner product (grade-lowering contraction)
- Outer product (wedge, grade-raising)
- Left/right contractions
- Scalar product
- Commutator and anti-commutator
- Regressive product (dual of wedge)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import torch
from typing import Optional

from ontic.genesis.ga.multivector import (
    CliffordAlgebra, Multivector
)


def geometric_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the geometric product ab.
    
    The geometric product is the fundamental operation in Clifford algebra.
    It combines the inner and outer products: ab = a·b + a∧b for vectors.
    
    Args:
        a: First multivector
        b: Second multivector
        
    Returns:
        Product multivector ab
    """
    if a.algebra != b.algebra:
        raise ValueError("Multivectors must be from the same algebra")
    
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for i in range(algebra.dim):
        if abs(a.coeffs[i]) < 1e-14:
            continue
        for j in range(algebra.dim):
            if abs(b.coeffs[j]) < 1e-14:
                continue
            sign, blade = algebra.sign_and_result(i, j)
            if sign != 0:
                result[blade] += sign * a.coeffs[i] * b.coeffs[j]
    
    return Multivector(algebra, result)


def inner_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the inner (dot) product a·b.
    
    The inner product contracts grades. For homogeneous multivectors
    of grades r and s with r ≤ s:
    grade(a·b) = |s - r|
    
    Uses the left contraction definition for consistency.
    
    Args:
        a: First multivector
        b: Second multivector
        
    Returns:
        Inner product a·b
    """
    return left_contraction(a, b)


def outer_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the outer (wedge) product a∧b.
    
    The outer product is the grade-raising part of the geometric product.
    For homogeneous elements of grades r and s:
    grade(a∧b) = r + s
    
    Args:
        a: First multivector  
        b: Second multivector
        
    Returns:
        Outer product a∧b
    """
    if a.algebra != b.algebra:
        raise ValueError("Multivectors must be from the same algebra")
    
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for i in range(algebra.dim):
        if abs(a.coeffs[i]) < 1e-14:
            continue
        grade_a = algebra.blade_grade(i)
        
        for j in range(algebra.dim):
            if abs(b.coeffs[j]) < 1e-14:
                continue
            grade_b = algebra.blade_grade(j)
            
            sign, blade = algebra.sign_and_result(i, j)
            if sign != 0:
                grade_result = algebra.blade_grade(blade)
                # Wedge product only keeps terms where grades add
                if grade_result == grade_a + grade_b:
                    result[blade] += sign * a.coeffs[i] * b.coeffs[j]
    
    return Multivector(algebra, result)


def left_contraction(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the left contraction a⌋b.
    
    The left contraction is defined so that:
    a⌋b has grade grade(b) - grade(a) when grade(a) ≤ grade(b)
    and is 0 otherwise.
    
    Args:
        a: First multivector (contracts into b)
        b: Second multivector
        
    Returns:
        Left contraction a⌋b
    """
    if a.algebra != b.algebra:
        raise ValueError("Multivectors must be from the same algebra")
    
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for i in range(algebra.dim):
        if abs(a.coeffs[i]) < 1e-14:
            continue
        grade_a = algebra.blade_grade(i)
        
        for j in range(algebra.dim):
            if abs(b.coeffs[j]) < 1e-14:
                continue
            grade_b = algebra.blade_grade(j)
            
            if grade_a > grade_b:
                continue
            
            sign, blade = algebra.sign_and_result(i, j)
            if sign != 0:
                grade_result = algebra.blade_grade(blade)
                # Left contraction: result grade = grade_b - grade_a
                if grade_result == grade_b - grade_a:
                    result[blade] += sign * a.coeffs[i] * b.coeffs[j]
    
    return Multivector(algebra, result)


def right_contraction(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the right contraction a⌊b.
    
    The right contraction is the "dual" of left contraction:
    a⌊b has grade grade(a) - grade(b) when grade(a) ≥ grade(b)
    
    Args:
        a: First multivector
        b: Second multivector (contracts into a)
        
    Returns:
        Right contraction a⌊b
    """
    if a.algebra != b.algebra:
        raise ValueError("Multivectors must be from the same algebra")
    
    algebra = a.algebra
    result = torch.zeros(algebra.dim, dtype=torch.float64)
    
    for i in range(algebra.dim):
        if abs(a.coeffs[i]) < 1e-14:
            continue
        grade_a = algebra.blade_grade(i)
        
        for j in range(algebra.dim):
            if abs(b.coeffs[j]) < 1e-14:
                continue
            grade_b = algebra.blade_grade(j)
            
            if grade_a < grade_b:
                continue
            
            sign, blade = algebra.sign_and_result(i, j)
            if sign != 0:
                grade_result = algebra.blade_grade(blade)
                # Right contraction: result grade = grade_a - grade_b
                if grade_result == grade_a - grade_b:
                    result[blade] += sign * a.coeffs[i] * b.coeffs[j]
    
    return Multivector(algebra, result)


def scalar_product(a: Multivector, b: Multivector) -> float:
    """
    Compute the scalar product <a,b>.
    
    This is the grade-0 part of the geometric product.
    Equivalent to the coefficient of 1 in ab.
    
    Args:
        a: First multivector
        b: Second multivector
        
    Returns:
        Scalar value
    """
    product = geometric_product(a, b)
    return float(product.coeffs[0])


def commutator_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the commutator product [a,b] = (ab - ba)/2.
    
    The commutator extracts the anti-symmetric part of the product.
    For bivectors, this gives the Lie algebra structure.
    
    Args:
        a: First multivector
        b: Second multivector
        
    Returns:
        Commutator [a,b]
    """
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab - ba) / 2


def anticommutator_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the anti-commutator product {a,b} = (ab + ba)/2.
    
    The anti-commutator extracts the symmetric part of the product.
    
    Args:
        a: First multivector
        b: Second multivector
        
    Returns:
        Anti-commutator {a,b}
    """
    ab = geometric_product(a, b)
    ba = geometric_product(b, a)
    return (ab + ba) / 2


def regressive_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the regressive (vee) product a∨b.
    
    The regressive product is the dual of the wedge product:
    a∨b = (a*∧b*)*
    
    where * denotes the dual with respect to the pseudoscalar.
    
    This is useful in projective geometry for computing meets.
    
    Args:
        a: First multivector
        b: Second multivector
        
    Returns:
        Regressive product a∨b
    """
    from ontic.genesis.ga.operations import dual
    
    # a∨b = dual(dual(a) ∧ dual(b))
    dual_a = dual(a)
    dual_b = dual(b)
    wedge_duals = outer_product(dual_a, dual_b)
    return dual(wedge_duals)


def sandwich_product(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the sandwich product aba^{-1}.
    
    This is the fundamental transformation in GA.
    For a versor (product of vectors), this performs
    a sequence of reflections.
    
    Args:
        a: Versor (transformer)
        b: Multivector to transform
        
    Returns:
        Transformed multivector aba^{-1}
    """
    from ontic.genesis.ga.operations import inverse
    
    ab = geometric_product(a, b)
    a_inv = inverse(a)
    return geometric_product(ab, a_inv)


def versor_sandwich(a: Multivector, b: Multivector, 
                    a_reverse: Optional[Multivector] = None) -> Multivector:
    """
    Compute the versor sandwich product abã.
    
    For unit versors, this is aba^{-1} = abã where ã is the reverse.
    More efficient than sandwich_product when we already have the reverse.
    
    Args:
        a: Unit versor
        b: Multivector to transform
        a_reverse: Pre-computed reverse of a (optional)
        
    Returns:
        Transformed multivector abã
    """
    from ontic.genesis.ga.operations import reverse
    
    if a_reverse is None:
        a_reverse = reverse(a)
    
    ab = geometric_product(a, b)
    return geometric_product(ab, a_reverse)
