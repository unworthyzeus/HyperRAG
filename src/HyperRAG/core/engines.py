"""
HyperRAG v2 - Advanced RAG Engine with Geometric Embeddings

This module implements a RAG engine that leverages geometric structures
for improved retrieval:

1. Hyperspherical Embeddings with Volumetric Transform
2. Multiple distance metrics (Cosine, Euclidean, Hellinger, Angular, Hyperbolic)
3. Poincaré Ball for hierarchical data
4. Comparison framework against standard RAG

Key insight: Standard embeddings concentrate on a thin shell in high dimensions.
By applying volumetric transforms and using appropriate metrics, we can
potentially improve retrieval performance.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
from .geometry import (
    Hypersphere, 
    CrossPolytope, 
    PoincareBall, 
    DistanceMetrics,
    VolumetricTransform
)


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    document: str
    score: float
    index: int
    metadata: Optional[dict] = None


class BaseRAGEngine:
    """Base class for RAG engines."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading model {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        # Get embedding dimension
        dummy = self.encoder.encode(["test"])
        self.dim = dummy.shape[1]
        print(f"Embedding dimension: {self.dim}")
        
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def ingest(self, documents: List[str]) -> None:
        """Ingest documents into the RAG system."""
        raise NotImplementedError
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        """Query the RAG system."""
        raise NotImplementedError


class StandardRAG(BaseRAGEngine):
    """
    Standard RAG using cosine similarity on the hypersphere surface.
    This is the baseline for comparison.
    """
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        self.embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # L2 normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.clip(norms, 1e-10, None)
        
        print(f"Ingested {len(documents)} documents (cosine similarity).")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.encoder.encode([query_text])[0]
        q_emb = q_emb / np.linalg.norm(q_emb)
        
        # Cosine similarity (dot product of normalized vectors)
        scores = np.dot(self.embeddings, q_emb)
        
        # Sort descending (higher = more similar)
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(scores[idx]),
                index=int(idx)
            ))
        
        return results


class HypersphereRAG(BaseRAGEngine):
    """
    RAG using Hyperspherical embeddings with volumetric transform.
    
    Instead of just normalizing to the surface, we spread points
    throughout the hypersphere's interior using volumetric transforms.
    
    This combats the curse of dimensionality where all points
    cluster on a thin shell.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        metric: Literal['cosine', 'euclidean', 'angular', 'hellinger'] = 'euclidean',
        volumetric: bool = True
    ):
        super().__init__(model_name)
        self.geometry = Hypersphere(self.dim)
        self.metric = metric
        self.volumetric = volumetric
        
        # Select distance function
        self.distance_fn = {
            'cosine': lambda x, y: 1 - DistanceMetrics.cosine_similarity(x, y),
            'euclidean': DistanceMetrics.euclidean_distance,
            'angular': DistanceMetrics.angular_distance,
            'hellinger': DistanceMetrics.hellinger_distance
        }[metric]
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        if self.volumetric:
            print("Applying volumetric transform (spreading to fill hypersphere)...")
            self.embeddings = self.geometry.project_to_volume(raw_embeddings)
        else:
            print("Normalizing to hypersphere surface...")
            self.embeddings = self.geometry.project_to_surface(raw_embeddings)
        
        # Debug: check norm distribution
        norms = np.linalg.norm(self.embeddings, axis=1)
        print(f"Norm distribution: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")
        print(f"Ingested {len(documents)} documents (metric: {self.metric}, volumetric: {self.volumetric}).")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.encoder.encode([query_text])[0]
        
        if self.volumetric:
            q_emb = self.geometry.project_to_volume(q_emb.reshape(1, -1))[0]
        else:
            q_emb = self.geometry.project_to_surface(q_emb.reshape(1, -1))[0]
        
        # Calculate distances
        distances = self.distance_fn(q_emb, self.embeddings)
        
        # Sort ascending (lower = more similar)
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(distances[idx]),
                index=int(idx)
            ))
        
        return results


class CrossPolytopeRAG(BaseRAGEngine):
    """
    RAG using Cross-Polytope (L1-ball) geometry.
    
    The cross-polytope is optimal for SPARSE embeddings because:
    - L1 norm promotes sparsity
    - Manhattan distance is more robust to outliers
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', volumetric: bool = True):
        super().__init__(model_name)
        self.geometry = CrossPolytope(self.dim)
        self.volumetric = volumetric
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        if self.volumetric:
            print("Applying volumetric transform (L1 ball)...")
            self.embeddings = self.geometry.project_to_volume(raw_embeddings)
        else:
            print("Normalizing to L1 surface...")
            self.embeddings = self.geometry.project_to_surface(raw_embeddings)
        
        print(f"Ingested {len(documents)} documents (cross-polytope, volumetric: {self.volumetric}).")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.encoder.encode([query_text])[0]
        
        if self.volumetric:
            q_emb = self.geometry.project_to_volume(q_emb.reshape(1, -1))[0]
        else:
            q_emb = self.geometry.project_to_surface(q_emb.reshape(1, -1))[0]
        
        # Manhattan distance
        distances = DistanceMetrics.manhattan_distance(q_emb, self.embeddings)
        
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(distances[idx]),
                index=int(idx)
            ))
        
        return results


class HyperbolicRAG(BaseRAGEngine):
    """
    RAG using Poincaré Ball (Hyperbolic Geometry).
    
    Hyperbolic space is better for HIERARCHICAL data because:
    - Volume grows exponentially with radius
    - This matches the exponential growth of tree-like structures
    - Concepts with parent-child relationships are naturally embedded
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', curvature: float = 1.0):
        super().__init__(model_name)
        self.geometry = PoincareBall(self.dim, curvature)
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        print("Projecting to Poincaré ball (hyperbolic space)...")
        self.embeddings = self.geometry.project_to_ball(raw_embeddings)
        
        norms = np.linalg.norm(self.embeddings, axis=1)
        print(f"Poincaré norms: mean={np.mean(norms):.4f}, max={np.max(norms):.4f}")
        print(f"Ingested {len(documents)} documents (hyperbolic).")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.encoder.encode([query_text])[0]
        q_emb = self.geometry.project_to_ball(q_emb.reshape(1, -1))[0]
        
        # Hyperbolic distance
        distances = self.geometry.hyperbolic_distance(q_emb, self.embeddings)
        
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(distances[idx]),
                index=int(idx)
            ))
        
        return results


class HellingerRAG(BaseRAGEngine):
    """
    RAG using Hellinger distance on probability distributions.
    
    Hellinger distance is a proper metric that:
    - Has range [0, 1]
    - Satisfies triangle inequality
    - Is symmetric
    - Is better for comparing density functions
    
    We convert embeddings to probability distributions first.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        print("Converting to probability distributions for Hellinger distance...")
        # Convert to probability distributions
        self.embeddings = DistanceMetrics.to_probability_distribution(raw_embeddings)
        
        print(f"Ingested {len(documents)} documents (Hellinger).")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.encoder.encode([query_text])[0]
        q_emb = DistanceMetrics.to_probability_distribution(q_emb)
        
        # Hellinger distance
        distances = DistanceMetrics.hellinger_distance(q_emb, self.embeddings)
        
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(distances[idx]),
                index=int(idx)
            ))
        
        return results


def create_rag_engine(
    engine_type: Literal['standard', 'hypersphere', 'cross', 'hyperbolic', 'hellinger'] = 'standard',
    model_name: str = 'all-MiniLM-L6-v2',
    **kwargs
) -> BaseRAGEngine:
    """Factory function to create RAG engines."""
    
    engines = {
        'standard': StandardRAG,
        'hypersphere': HypersphereRAG,
        'cross': CrossPolytopeRAG,  
        'hyperbolic': HyperbolicRAG,
        'hellinger': HellingerRAG
    }
    
    if engine_type not in engines:
        raise ValueError(f"Unknown engine type: {engine_type}. Choose from: {list(engines.keys())}")
    
    return engines[engine_type](model_name=model_name, **kwargs)


if __name__ == "__main__":
    # Quick demo
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Hyperdodecahedrons are complex 4D shapes with 120 cells.",
        "Machine learning uses vector embeddings for semantic similarity.",
        "Geometry in high dimensions exhibits counter-intuitive properties.",
        "Apple pie recipe: combine apples, sugar, flour, and butter.",
        "The 120-cell is a regular polychoron with 120 dodecahedral cells.",
        "RAG systems retrieve relevant information to augment LLM responses.",
        "The Poincaré ball model represents hyperbolic space efficiently.",
        "Hellinger distance measures similarity between probability distributions.",
        "Cross-polytopes are dual to hypercubes and promote L1 sparsity."
    ]
    
    query = "Tell me about 4D geometric shapes"
    
    print("\n" + "=" * 70)
    print("HyperRAG v2 - Multi-Metric Comparison Demo")
    print("=" * 70)
    
    engines = {
        'Standard (Cosine)': create_rag_engine('standard'),
        'Hypersphere (Euclidean+Vol)': create_rag_engine('hypersphere', metric='euclidean'),
        'Hypersphere (Hellinger)': create_rag_engine('hypersphere', metric='hellinger'),
        'Cross-Polytope (L1)': create_rag_engine('cross'),
        'Hyperbolic (Poincaré)': create_rag_engine('hyperbolic'),
        'Hellinger Distance': create_rag_engine('hellinger'),
    }
    
    for name, engine in engines.items():
        print(f"\n--- {name} ---")
        engine.ingest(docs)
        results = engine.query(query, top_k=3)
        for r in results:
            print(f"  [{r.score:.4f}] {r.document[:60]}...")
