"""
Conformal Geometric Algebra (CGA)

CGA embeds Euclidean space into a higher-dimensional space with a
degenerate metric, enabling unified treatment of points, lines, planes,
circles, and spheres as algebraic objects.

For 3D CGA: Cl(4,1,0) with 32 components
- Adds two extra basis vectors: e+ (squares to +1) and e- (squares to -1)
- Points map to the null cone: X = x + (1/2)x²e∞ + e₀
- e₀ = (e- - e+)/2 (origin)
- e∞ = e- + e+ (point at infinity)

Copyright (c) 2026 Tigantic Holdings LLC. All Rights Reserved.
"""

from __future__ import annotations
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass

from tensornet.genesis.ga.multivector import (
    CliffordAlgebra, Multivector, vector
)
from tensornet.genesis.ga.products import (
    geometric_product, outer_product, inner_product, regressive_product
)
from tensornet.genesis.ga.operations import (
    reverse, normalize, magnitude, dual
)


@dataclass
class ConformalGA:
    """
    Conformal Geometric Algebra for 3D.
    
    Uses Cl(4,1,0) with basis {e1, e2, e3, e+, e-}.
    Constructs null vectors e₀ (origin) and e∞ (infinity).
    
    Indices:
        0-2: Euclidean basis e1, e2, e3
        3: e+ (positive)
        4: e- (negative)
    """
    algebra: CliffordAlgebra = None
    
    def __post_init__(self):
        # Cl(4,1,0): 4 positive (e1,e2,e3,e+), 1 negative (e-)
        self.algebra = CliffordAlgebra(p=4, q=1, r=0)
    
    @property
    def dim(self) -> int:
        """Dimension of the Euclidean space (3)."""
        return 3
    
    def e_plus(self) -> Multivector:
        """Basis vector that squares to +1 (e+)."""
        return Multivector.basis_vector(self.algebra, 3)
    
    def e_minus(self) -> Multivector:
        """Basis vector that squares to -1 (e-)."""
        return Multivector.basis_vector(self.algebra, 4)
    
    def e_origin(self) -> Multivector:
        """
        Origin point e₀ = (e- - e+)/2.
        
        This is a null vector (e₀² = 0).
        """
        e_p = self.e_plus()
        e_m = self.e_minus()
        return (e_m - e_p) / 2
    
    def e_infinity(self) -> Multivector:
        """
        Point at infinity e∞ = e- + e+.
        
        This is a null vector (e∞² = 0).
        e₀ · e∞ = -1
        """
        e_p = self.e_plus()
        e_m = self.e_minus()
        return e_m + e_p
    
    def euclidean_basis(self, i: int) -> Multivector:
        """Get i-th Euclidean basis vector (0, 1, or 2)."""
        if i < 0 or i >= 3:
            raise ValueError("Euclidean index must be 0, 1, or 2")
        return Multivector.basis_vector(self.algebra, i)


def point_to_cga(cga: ConformalGA, point: List[float]) -> Multivector:
    """
    Embed a Euclidean point into CGA.
    
    X = x + (1/2)|x|²e∞ + e₀
    
    where x = x₁e₁ + x₂e₂ + x₃e₃
    
    Args:
        cga: Conformal GA instance
        point: [x, y, z] coordinates
        
    Returns:
        CGA point (null vector)
    """
    if len(point) != 3:
        raise ValueError("Point must be 3D")
    
    # Euclidean part
    x = Multivector.zero(cga.algebra)
    for i, c in enumerate(point):
        x = x + Multivector.basis_vector(cga.algebra, i, c)
    
    # Squared norm
    x_sq = sum(c**2 for c in point)
    
    # CGA point
    e0 = cga.e_origin()
    einf = cga.e_infinity()
    
    return x + einf * (x_sq / 2) + e0


def cga_to_point(cga: ConformalGA, X: Multivector) -> List[float]:
    """
    Extract Euclidean point from CGA representation.
    
    Normalizes by the e∞ coefficient.
    
    Args:
        cga: Conformal GA instance
        X: CGA point
        
    Returns:
        [x, y, z] coordinates
    """
    # Get the e∞ coefficient for normalization
    einf = cga.e_infinity()
    einf_coeff = -float(inner_product(X, einf).coeffs[0])
    
    if abs(einf_coeff) < 1e-14:
        raise ValueError("Point is at infinity")
    
    # Extract Euclidean components
    result = []
    for i in range(3):
        blade_idx = 1 << i
        result.append(float(X.coeffs[blade_idx]) / einf_coeff)
    
    return result


def cga_line(cga: ConformalGA, p1: Multivector, p2: Multivector) -> Multivector:
    """
    Create a CGA line through two points.
    
    Line L = P₁ ∧ P₂ ∧ e∞
    
    Args:
        cga: Conformal GA instance
        p1: First CGA point
        p2: Second CGA point
        
    Returns:
        CGA line (3-blade)
    """
    einf = cga.e_infinity()
    p1_p2 = outer_product(p1, p2)
    return outer_product(p1_p2, einf)


def cga_plane(cga: ConformalGA, normal: List[float], distance: float) -> Multivector:
    """
    Create a CGA plane with given normal and distance from origin.
    
    Plane π = n + d·e∞
    
    where n is the unit normal vector.
    
    Args:
        cga: Conformal GA instance
        normal: [nx, ny, nz] unit normal
        distance: Distance from origin (signed)
        
    Returns:
        CGA plane (1-vector in dual space)
    """
    # Normalize normal
    norm = sum(c**2 for c in normal) ** 0.5
    normal = [c / norm for c in normal]
    
    # n = n₁e₁ + n₂e₂ + n₃e₃
    n = Multivector.zero(cga.algebra)
    for i, c in enumerate(normal):
        n = n + Multivector.basis_vector(cga.algebra, i, c)
    
    einf = cga.e_infinity()
    
    return n + einf * distance


def cga_plane_from_points(cga: ConformalGA, p1: Multivector, 
                          p2: Multivector, p3: Multivector) -> Multivector:
    """
    Create a CGA plane through three points.
    
    Plane π = P₁ ∧ P₂ ∧ P₃ ∧ e∞
    
    Args:
        cga: Conformal GA instance
        p1, p2, p3: Three CGA points
        
    Returns:
        CGA plane (4-blade)
    """
    einf = cga.e_infinity()
    p123 = outer_product(outer_product(p1, p2), p3)
    return outer_product(p123, einf)


def cga_circle(cga: ConformalGA, center: List[float], 
               normal: List[float], radius: float) -> Multivector:
    """
    Create a CGA circle.
    
    Circle C = S ∧ π
    where S is a sphere through the circle and π is the plane of the circle.
    
    Args:
        cga: Conformal GA instance
        center: [x, y, z] center of circle
        normal: [nx, ny, nz] normal to circle plane
        radius: Radius of circle
        
    Returns:
        CGA circle (3-blade)
    """
    # Create sphere centered at center with given radius
    S = cga_sphere(cga, center, radius)
    
    # Create plane through center with given normal
    # Distance = center · normal
    distance = sum(c * n for c, n in zip(center, normal))
    pi = cga_plane(cga, normal, distance)
    
    return outer_product(S, pi)


def cga_sphere(cga: ConformalGA, center: List[float], radius: float) -> Multivector:
    """
    Create a CGA sphere.
    
    Sphere S = c - (r²/2)e∞
    
    where c is the CGA center point.
    
    Args:
        cga: Conformal GA instance
        center: [x, y, z] center coordinates
        radius: Sphere radius
        
    Returns:
        CGA sphere (1-vector)
    """
    c = point_to_cga(cga, center)
    einf = cga.e_infinity()
    
    return c - einf * (radius ** 2 / 2)


def cga_point_pair(cga: ConformalGA, p1: List[float], p2: List[float]) -> Multivector:
    """
    Create a CGA point pair (0-sphere).
    
    Point pair PP = P₁ ∧ P₂
    
    Args:
        cga: Conformal GA instance
        p1, p2: Two Euclidean points
        
    Returns:
        CGA point pair (2-blade)
    """
    P1 = point_to_cga(cga, p1)
    P2 = point_to_cga(cga, p2)
    return outer_product(P1, P2)


def meet(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the meet (intersection) of two geometric objects.
    
    The meet is the regressive product: a ∨ b = (a* ∧ b*)*
    
    Args:
        a: First CGA object
        b: Second CGA object
        
    Returns:
        Intersection
    """
    return regressive_product(a, b)


def join(a: Multivector, b: Multivector) -> Multivector:
    """
    Compute the join (span) of two geometric objects.
    
    The join is the outer product: a ∧ b
    
    Args:
        a: First CGA object
        b: Second CGA object
        
    Returns:
        Joined object
    """
    return outer_product(a, b)


def cga_translator(cga: ConformalGA, direction: List[float]) -> Multivector:
    """
    Create a CGA translator (translation motor).
    
    T = 1 - (d/2)e∞
    
    where d = d₁e₁ + d₂e₂ + d₃e₃ is the translation vector.
    Translation: X' = TXT̃
    
    Args:
        cga: Conformal GA instance
        direction: [dx, dy, dz] translation vector
        
    Returns:
        Translator versor
    """
    # Build translation vector
    d = Multivector.zero(cga.algebra)
    for i, c in enumerate(direction):
        d = d + Multivector.basis_vector(cga.algebra, i, c)
    
    einf = cga.e_infinity()
    d_einf = geometric_product(d, einf)
    
    one = Multivector.scalar(cga.algebra, 1.0)
    return one - d_einf / 2


def apply_translator(T: Multivector, X: Multivector) -> Multivector:
    """
    Apply translator to a CGA object.
    
    X' = TXT̃
    
    Args:
        T: Translator
        X: CGA object
        
    Returns:
        Translated object
    """
    T_rev = reverse(T)
    TX = geometric_product(T, X)
    return geometric_product(TX, T_rev)


def cga_dilator(cga: ConformalGA, factor: float) -> Multivector:
    """
    Create a CGA dilator (scaling versor).
    
    D = exp(λ e∞e₀ / 2)
    
    Scaling: X' = DXD̃ scales by e^λ.
    
    Args:
        cga: Conformal GA instance
        factor: Scaling factor (exponential)
        
    Returns:
        Dilator versor
    """
    import math
    
    e0 = cga.e_origin()
    einf = cga.e_infinity()
    
    e0_einf = geometric_product(e0, einf)
    
    # D = cosh(λ/2) + sinh(λ/2) * (e∞e₀)
    half_lambda = math.log(factor) / 2
    
    one = Multivector.scalar(cga.algebra, math.cosh(half_lambda))
    biv = e0_einf * math.sinh(half_lambda)
    
    return one + biv


def cga_reflector(cga: ConformalGA, plane: Multivector) -> Multivector:
    """
    Reflect an object in a CGA plane.
    
    Reflection: X' = -πXπ
    
    Args:
        cga: Conformal GA instance
        plane: CGA plane
        
    Returns:
        The plane itself (used as reflector)
    """
    return plane


def reflect_in_plane(X: Multivector, plane: Multivector) -> Multivector:
    """
    Reflect CGA object X in the given plane.
    
    X' = -πXπ
    
    Args:
        X: CGA object
        plane: CGA plane
        
    Returns:
        Reflected object
    """
    pX = geometric_product(plane, X)
    pXp = geometric_product(pX, plane)
    return -pXp


def distance_point_to_point(cga: ConformalGA, P1: Multivector, 
                            P2: Multivector) -> float:
    """
    Compute Euclidean distance between two CGA points.
    
    d² = -2(P₁ · P₂)
    
    (for normalized points where P · e∞ = -1)
    
    Args:
        cga: Conformal GA instance
        P1, P2: CGA points
        
    Returns:
        Euclidean distance
    """
    # Normalize points
    einf = cga.e_infinity()
    
    w1 = -float(inner_product(P1, einf).coeffs[0])
    w2 = -float(inner_product(P2, einf).coeffs[0])
    
    if abs(w1) < 1e-14 or abs(w2) < 1e-14:
        raise ValueError("Point at infinity")
    
    P1_norm = P1 / w1
    P2_norm = P2 / w2
    
    dot = float(inner_product(P1_norm, P2_norm).coeffs[0])
    
    return (abs(-2 * dot)) ** 0.5
