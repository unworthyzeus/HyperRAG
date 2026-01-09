"""
HyperRAG - Geometric RAG Optimization
"""

from .core.geometry import (
    Hypersphere,
    CrossPolytope,
    PoincareBall,
    DistanceMetrics,
    VolumetricTransform,
)

from .core.engines import (
    StandardRAG,
    HypersphereRAG,
    CrossPolytopeRAG,
    HyperbolicRAG,
    HellingerRAG,
    create_rag_engine,
)

from .advanced.advanced import (
    HybridGeometricRAG,
    EnsembleRAG,
    ContrastiveVolumetricRAG,
    CrossPolytopeHybridRAG,
)

from .experimental.experimental import (
    WhitenedRAG,
    ClusterTreeRAG,
)

__version__ = "3.1.0"
__all__ = [
    # Geometry
    "Hypersphere",
    "CrossPolytope", 
    "PoincareBall",
    "DistanceMetrics",
    "VolumetricTransform",
    # Engines
    "StandardRAG",
    "HypersphereRAG",
    "CrossPolytopeRAG",
    "HyperbolicRAG",
    "HellingerRAG",
    "create_rag_engine",
    # Advanced
    "HybridGeometricRAG",
    "EnsembleRAG",
    "ContrastiveVolumetricRAG",
    "CrossPolytopeHybridRAG",
    # Experimental
    "WhitenedRAG",
    "ClusterTreeRAG",
]
