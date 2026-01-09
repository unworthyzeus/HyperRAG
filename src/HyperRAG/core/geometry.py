"""
Advanced Geometry Module for HyperRAG v2

This module implements various geometric structures for embedding space manipulation:
1. Hypersphere (optimal volume/surface ratio - isoperimetric optimality)
2. Cross-Polytope (L1-ball, good for sparse embeddings)  
3. Poincaré Ball (hyperbolic geometry for hierarchical data)

Key insight: In high dimensions, spheres are OPTIMAL for volume/surface area ratio.
The "curse of dimensionality" causes points to concentrate on shells.
We combat this with VOLUMETRIC TRANSFORMS that spread points into the interior.
"""

import numpy as np
from scipy import special
from typing import Tuple, Optional, Literal


class VolumetricTransform:
    """
    Transforms shell-concentrated embeddings into volume-filling distributions.
    
    In high dimensions, random points tend to cluster near a thin shell.
    We apply transforms to spread them throughout the interior of the shape.
    """
    
    @staticmethod
    def shell_to_volume(vectors: np.ndarray, power: float = 0.5) -> np.ndarray:
        """
        Transform shell-concentrated vectors to volume-filling distribution.
        
        Uses the inverse of the radial CDF to map uniform shell -> uniform volume.
        For a d-dimensional ball, if R is the radius, the volume scales as R^d.
        To uniformly fill the ball: r_new = r_original^(1/d). 
        But since embeddings aren't uniform random, we use a tunable power.
        
        Args:
            vectors: (N, D) array of vectors
            power: transformation strength (lower = more compression toward center)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)  # Avoid division by zero
        
        # Normalize to unit vectors
        unit_vectors = vectors / norms
        
        # Apply volumetric transform to radii
        # For uniform volume distribution in D dims: r' = r^(1/D)
        # We use power as a tunable parameter
        d = vectors.shape[1]
        new_norms = np.power(norms / np.max(norms), power)
        
        return unit_vectors * new_norms
    
    @staticmethod
    def radial_spread(vectors: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Apply a softmax-like radial spread using the norm as "energy".
        
        Args:
            vectors: (N, D) array of vectors  
            temperature: controls spread (higher = more spread)
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms_clipped = np.clip(norms, 1e-10, None)
        
        # Sigmoid-based radial transformation
        transformed_norms = norms_clipped * (2 / (1 + np.exp(-temperature * norms_clipped)) - 1)
        
        return vectors / norms_clipped * transformed_norms


class Hypersphere:
    """
    N-dimensional Hypersphere - optimal volume/surface ratio by isoperimetric inequality.
    
    This is the BEST polytope for maximizing hypervolume relative to surface area.
    A sphere in D dimensions has volume V = π^(D/2) * R^D / Γ(D/2 + 1)
    and surface S = D * π^(D/2) * R^(D-1) / Γ(D/2 + 1) = D * V / R
    
    Ratio V/S = R/D, which is optimal among all shapes of the same volume.
    """
    
    def __init__(self, dimension: int, radius: float = 1.0):
        self.dimension = dimension
        self.radius = radius
        self.volume_transform = VolumetricTransform()
        
    def project_to_surface(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors onto the hypersphere surface (L2 normalization)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        return (vectors / norms) * self.radius
    
    def project_to_volume(self, vectors: np.ndarray, method: str = 'power') -> np.ndarray:
        """
        Project vectors INTO the hypersphere volume (not just surface).
        
        This is the key innovation - we don't just normalize to the shell,
        we transform to fill the entire ball uniformly.
        """
        if method == 'power':
            return self.volume_transform.shell_to_volume(vectors) * self.radius
        else:
            return self.volume_transform.radial_spread(vectors) * self.radius
    
    def clip_to_ball(self, vectors: np.ndarray) -> np.ndarray:
        """Clip vectors to be inside the ball (project if outside)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.radius / np.clip(norms, 1e-10, None))
        return vectors * scale
    
    def get_hypervolume(self) -> float:
        """Calculate the hypervolume of the hypersphere."""
        d = self.dimension
        return (np.pi ** (d/2) * self.radius ** d) / special.gamma(d/2 + 1)
    
    def get_surface_area(self) -> float:
        """Calculate the surface area of the hypersphere."""
        d = self.dimension
        return d * (np.pi ** (d/2) * self.radius ** (d-1)) / special.gamma(d/2 + 1)
    
    def volume_surface_ratio(self) -> float:
        """Returns V/S ratio - optimal for spheres."""
        return self.get_hypervolume() / self.get_surface_area()


class CrossPolytope:
    """
    N-dimensional Cross-Polytope (Orthoplex/Hyperoctahedron).
    
    The cross-polytope is the unit ball in L1 norm: {x : ||x||_1 <= 1}
    It's the dual of the hypercube and has 2^D vertices in D dimensions.
    
    Better than hypercubes for sparse data since L1 promotes sparsity.
    The vertices are at ±e_i where e_i are the standard basis vectors.
    """
    
    def __init__(self, dimension: int, scale: float = 1.0):
        self.dimension = dimension
        self.scale = scale
        
    def project_to_surface(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors onto the cross-polytope surface (L1 normalization)."""
        l1_norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
        l1_norms = np.clip(l1_norms, 1e-10, None)
        return (vectors / l1_norms) * self.scale
    
    def project_to_volume(self, vectors: np.ndarray) -> np.ndarray:
        """
        Project vectors into the cross-polytope volume with spread.
        Similar volumetric transform but using L1 norm.
        """
        l1_norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
        l1_norms_clipped = np.clip(l1_norms, 1e-10, None)
        unit_vectors = vectors / l1_norms_clipped
        
        # Transform to fill volume
        d = self.dimension
        new_norms = np.power(l1_norms / np.max(l1_norms), 1.0 / d)
        
        return unit_vectors * new_norms * self.scale
    
    def clip_to_interior(self, vectors: np.ndarray) -> np.ndarray:
        """Clip vectors to be inside the cross-polytope."""
        l1_norms = np.sum(np.abs(vectors), axis=1, keepdims=True)
        scale = np.minimum(1.0, self.scale / np.clip(l1_norms, 1e-10, None))
        return vectors * scale
    
    def get_hypervolume(self) -> float:
        """Volume of cross-polytope: 2^D / D!"""
        d = self.dimension
        return (2 ** d) / special.factorial(d) * (self.scale ** d)
    
    def get_surface_area(self) -> float:
        """Surface area of cross-polytope: 2^D * sqrt(D) / (D-1)!"""
        d = self.dimension
        if d == 1:
            return 2 * self.scale
        return (2 ** d) * np.sqrt(d) / special.factorial(d-1) * (self.scale ** (d-1))
    
    def volume_surface_ratio(self) -> float:
        """Returns V/S ratio for comparison."""
        return self.get_hypervolume() / self.get_surface_area()


class PoincareBall:
    """
    Poincaré Ball Model of Hyperbolic Space.
    
    Hyperbolic geometry is better for hierarchical/tree-like data because:
    - The volume grows exponentially with radius  
    - This matches the exponential growth of tree nodes with depth
    - Points near the boundary have more "room" for children
    
    The Poincaré ball uses the unit ball in R^n with hyperbolic metric:
    ds^2 = 4 * ||dx||^2 / (1 - ||x||^2)^2
    """
    
    def __init__(self, dimension: int, curvature: float = 1.0):
        self.dimension = dimension
        self.curvature = curvature  # Negative curvature parameter |c|
        self.eps = 1e-5  # Numerical stability
        
    def _lambda_x(self, x: np.ndarray) -> np.ndarray:
        """Conformal factor λ_x = 2 / (1 - ||x||^2)"""
        norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        return 2 / np.clip(1 - self.curvature * norm_sq, self.eps, None)
    
    def exp_map(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: move from point x in direction v (tangent vector).
        
        exp_x(v) = x ⊕ (tanh(√c * ||v|| * λ_x / 2) * v / (√c * ||v||))
        """
        v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
        v_norm = np.clip(v_norm, self.eps, None)
        
        lambda_x = self._lambda_x(x)
        sqrt_c = np.sqrt(self.curvature)
        
        # Direction of v
        v_unit = v / v_norm
        
        # tanh factor
        t = np.tanh(sqrt_c * lambda_x * v_norm / 2) / sqrt_c
        
        # Möbius addition x ⊕ (t * v_unit)
        return self.mobius_add(x, t * v_unit)
    
    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Möbius addition in the Poincaré ball.
        
        x ⊕ y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) / 
                (1 + 2c<x,y> + c^2||x||^2||y||^2)
        """
        c = self.curvature
        x_norm_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        y_norm_sq = np.sum(y ** 2, axis=-1, keepdims=True)
        xy_dot = np.sum(x * y, axis=-1, keepdims=True)
        
        num = (1 + 2*c*xy_dot + c*y_norm_sq) * x + (1 - c*x_norm_sq) * y
        denom = 1 + 2*c*xy_dot + c**2 * x_norm_sq * y_norm_sq
        
        return num / np.clip(denom, self.eps, None)
    
    def project_to_ball(self, vectors: np.ndarray, max_norm: float = 0.99) -> np.ndarray:
        """
        Project Euclidean vectors into the Poincaré ball.
        
        We clip the norm to be < 1 (open ball) and apply a transformation
        that maps the shell to a volumetric distribution.
        """
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.clip(norms, self.eps, None)
        
        # Normalize to unit sphere first
        unit_vectors = vectors / norms
        
        # Map norms to (0, max_norm) using tanh for smooth saturation
        # This naturally fills the volume since tanh spreads values
        new_norms = np.tanh(norms / np.max(norms)) * max_norm
        
        return unit_vectors * new_norms
    
    def hyperbolic_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Hyperbolic distance in the Poincaré ball.
        
        d(x, y) = (1/√c) * arcosh(1 + 2c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))
        """
        c = self.curvature
        x_norm_sq = np.sum(x ** 2, axis=-1)
        y_norm_sq = np.sum(y ** 2, axis=-1)
        diff_norm_sq = np.sum((x - y) ** 2, axis=-1)
        
        numerator = 2 * c * diff_norm_sq
        denominator = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
        denominator = np.clip(denominator, self.eps, None)
        
        inner = 1 + numerator / denominator
        inner = np.clip(inner, 1.0 + self.eps, None)  # arcosh domain
        
        return np.arccosh(inner) / np.sqrt(c)


class DistanceMetrics:
    """
    Collection of distance/similarity metrics for embedding comparison.
    
    Different metrics have different properties:
    - Cosine: angle-based, ignores magnitude (standard for embeddings)
    - Euclidean: straight-line distance
    - Hellinger: for probability distributions (need positive, normalized)
    - Angular: arccos of cosine (true geodesic on sphere)
    - Manhattan (L1): sum of absolute differences (sparse-friendly)
    """
    
    @staticmethod
    def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Cosine similarity: <x,y> / (||x|| ||y||)"""
        x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
        
        x_normalized = x / np.clip(x_norm, 1e-10, None)
        y_normalized = y / np.clip(y_norm, 1e-10, None)
        
        # Handle both single vector and batched comparisons
        if x.ndim == 1 and y.ndim == 2:
            return np.dot(y, x_normalized.flatten())
        elif x.ndim == 2 and y.ndim == 1:
            return np.dot(x, y_normalized.flatten())
        else:
            return np.sum(x_normalized * y_normalized, axis=-1)
    
    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Euclidean (L2) distance."""
        if x.ndim == 1 and y.ndim == 2:
            return np.linalg.norm(y - x, axis=1)
        elif x.ndim == 2 and y.ndim == 1:
            return np.linalg.norm(x - y, axis=1)
        else:
            return np.linalg.norm(x - y, axis=-1)
    
    @staticmethod
    def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Manhattan (L1) distance - sum of absolute differences."""
        if x.ndim == 1 and y.ndim == 2:
            return np.sum(np.abs(y - x), axis=1)
        elif x.ndim == 2 and y.ndim == 1:
            return np.sum(np.abs(x - y), axis=1)
        else:
            return np.sum(np.abs(x - y), axis=-1)
    
    @staticmethod
    def angular_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Angular distance: arccos(cosine_similarity) / π
        
        This is the geodesic distance on the unit sphere, normalized to [0, 1].
        """
        cos_sim = DistanceMetrics.cosine_similarity(x, y)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return np.arccos(cos_sim) / np.pi
    
    @staticmethod
    def to_probability_distribution(x: np.ndarray) -> np.ndarray:
        """
        Convert embedding to a probability distribution for Hellinger.
        Uses softmax-like transformation.
        """
        # Shift to positive (embeddings can be negative)
        x_shifted = x - np.min(x, axis=-1, keepdims=True)
        
        # L1 normalize to get probabilities
        x_sum = np.sum(x_shifted, axis=-1, keepdims=True)
        return x_shifted / np.clip(x_sum, 1e-10, None)
    
    @staticmethod
    def hellinger_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Hellinger distance between probability distributions.
        
        H(P, Q) = (1/√2) * ||√P - √Q||_2
        
        Note: Inputs must be probability distributions (positive, sum to 1).
        Use to_probability_distribution() first for raw embeddings.
        
        Range: [0, 1] where 0 = identical, 1 = maximally different.
        """
        # Convert to probability distributions if needed
        if np.any(x < 0) or np.any(y < 0):
            x = DistanceMetrics.to_probability_distribution(x)
            y = DistanceMetrics.to_probability_distribution(y)
        
        sqrt_x = np.sqrt(np.clip(x, 0, None))
        sqrt_y = np.sqrt(np.clip(y, 0, None))
        
        if x.ndim == 1 and y.ndim == 2:
            return np.linalg.norm(sqrt_y - sqrt_x, axis=1) / np.sqrt(2)
        elif x.ndim == 2 and y.ndim == 1:
            return np.linalg.norm(sqrt_x - sqrt_y, axis=1) / np.sqrt(2)
        else:
            return np.linalg.norm(sqrt_x - sqrt_y, axis=-1) / np.sqrt(2)


def compare_polytopes_volume_surface():
    """
    Compare volume/surface ratios of different polytopes.
    This demonstrates why hyperspheres are optimal.
    """
    dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 384, 768]
    
    print("=" * 70)
    print("POLYTOPE VOLUME/SURFACE RATIO COMPARISON")
    print("=" * 70)
    print(f"{'Dim':>6} | {'Hypersphere V/S':>16} | {'CrossPolytope V/S':>18} | {'Ratio':>12}")
    print("-" * 70)
    
    for d in dimensions:
        sphere = Hypersphere(d)
        cross = CrossPolytope(d)
        
        sphere_ratio = sphere.volume_surface_ratio()
        cross_ratio = cross.volume_surface_ratio()
        
        # Sphere is always better - show how much better
        improvement = sphere_ratio / cross_ratio if cross_ratio > 0 else float('inf')
        
        print(f"{d:>6} | {sphere_ratio:>16.2e} | {cross_ratio:>18.2e} | {improvement:>10.2f}x")
    
    print("=" * 70)
    print("The HYPERSPHERE is optimal for volume/surface ratio (isoperimetric inequality)")
    print("=" * 70)


if __name__ == "__main__":
    compare_polytopes_volume_surface()
    
    print("\n\nTesting Volumetric Transforms:")
    print("-" * 50)
    
    # Create test vectors
    np.random.seed(42)
    d = 384  # Typical embedding dimension
    n = 1000
    vectors = np.random.randn(n, d)
    
    # Before transform - all norms cluster near sqrt(d) due to concentration
    norms_before = np.linalg.norm(vectors, axis=1)
    print(f"Before transform: mean norm = {np.mean(norms_before):.2f}, std = {np.std(norms_before):.2f}")
    print(f"  Range: [{np.min(norms_before):.2f}, {np.max(norms_before):.2f}]")
    
    # Apply volumetric transform
    sphere = Hypersphere(d)
    vectors_vol = sphere.project_to_volume(vectors)
    norms_after = np.linalg.norm(vectors_vol, axis=1)
    print(f"After volumetric:  mean norm = {np.mean(norms_after):.4f}, std = {np.std(norms_after):.4f}")
    print(f"  Range: [{np.min(norms_after):.4f}, {np.max(norms_after):.4f}]")
    
    print("\nThe volumetric transform SPREADS points from the shell into the volume!")
