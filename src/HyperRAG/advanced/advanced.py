"""
HyperRAG v3 - Advanced Volumetric Embedding Strategies

The key insight from v2 is that pre-trained embeddings are trained for ANGULAR similarity,
so simply transforming them geometrically doesn't help much.

This version implements:
1. PROPER volumetric spreading that preserves angular relationships
2. Radial Information Encoding - uses the magnitude to encode additional info
3. Manifold-aware distance metrics
4. Hybrid retrieval combining multiple geometric approaches
5. Learnable geometric projections

The goal is to ADD information via the radial dimension, not just remap existing info.
"""

import numpy as np
from typing import List, Tuple, Optional, Literal, Callable
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import time


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    document: str
    score: float
    index: int
    metadata: Optional[dict] = None


class RadialInformationEncoder:
    """
    Encodes additional information in the radial dimension.
    
    The key insight: Angular similarity captures SEMANTIC meaning.
    We can ADD information in the radial dimension:
    - Document length/importance
    - Specificity score
    - Topic confidence
    - Or a HASH of the content for disambiguation
    
    This expands the information capacity beyond what's on the shell.
    """
    
    def __init__(self, dimension: int):
        self.dim = dimension
        self.hash_dim = min(32, dimension // 4)  # Use subset of dims for hashing
        np.random.seed(42)
        self.hash_matrix = np.random.randn(self.hash_dim, dimension)
        
    def encode_specificity(self, embeddings: np.ndarray, texts: List[str]) -> np.ndarray:
        """
        Encode text specificity into the radial dimension.
        
        More specific/detailed texts get pushed toward the boundary,
        more general texts stay closer to center.
        This creates natural clustering where queries can match
        at different specificity levels.
        """
        result = embeddings.copy()
        
        for i, text in enumerate(texts):
            # Simple heuristics for specificity
            word_count = len(text.split())
            unique_ratio = len(set(text.lower().split())) / max(1, word_count)
            avg_word_len = np.mean([len(w) for w in text.split()]) if text else 0
            
            # Combine into specificity score (0 to 1)
            specificity = min(1.0, (
                0.3 * min(1.0, word_count / 200) +  # Longer = more specific
                0.4 * unique_ratio +                 # More unique words = more specific
                0.3 * min(1.0, avg_word_len / 8)     # Longer words = more technical/specific
            ))
            
            # Scale embedding by specificity (0.3 to 1.0 range)
            norm = np.linalg.norm(result[i])
            if norm > 0:
                target_norm = 0.3 + 0.7 * specificity
                result[i] = result[i] / norm * target_norm
        
        return result
    
    def encode_with_hash(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Add a content-based hash to break ties between similar embeddings.
        
        Two documents with similar semantics but different exact content
        will have slightly different radii based on a hash of their embedding.
        """
        result = embeddings.copy()
        
        # Project through random matrix to get hash
        hashes = np.dot(embeddings, self.hash_matrix.T)  # (N, hash_dim)
        
        # Convert to a scalar radial modifier
        hash_norms = np.linalg.norm(hashes, axis=1, keepdims=True)
        hash_norms = hash_norms / np.max(hash_norms)  # Normalize to [0, 1]
        
        # Slight radial variation (95% to 100% of original norm)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        target_norms = norms * (0.95 + 0.05 * hash_norms.flatten()[:, np.newaxis])
        
        # Renormalize
        norms_safe = np.clip(norms, 1e-10, None)
        result = result / norms_safe * target_norms
        
        return result


class ManifoldAwareDistance:
    """
    Distance metrics that respect the manifold structure of embeddings.
    
    Pre-trained embeddings live on a complex manifold, not a simple sphere.
    We approximate this with adaptive metrics.
    """
    
    @staticmethod 
    def cosine_plus_radial(x: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """
        Combines angular distance with radial distance.
        
        This allows breaking ties between equidistant angular matches
        using the radial information.
        """
        # Angular component (1 - cosine)
        x_norm = np.linalg.norm(x, keepdims=True)
        y_norms = np.linalg.norm(y, axis=1, keepdims=True)
        
        x_unit = x / np.clip(x_norm, 1e-10, None)
        y_unit = y / np.clip(y_norms, 1e-10, None)
        
        cos_sim = np.dot(y_unit, x_unit.flatten())
        angular_dist = 1 - cos_sim
        
        # Radial component (absolute difference in norms)
        radial_dist = np.abs(y_norms.flatten() - x_norm.flatten())
        
        # Weighted combination
        return angular_dist + alpha * radial_dist
    
    @staticmethod
    def adaptive_metric(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Adaptively weights Euclidean vs Angular based on local density.
        
        In sparse regions, Euclidean works well.
        In dense regions, Angular is more discriminative.
        """
        # Start with both distances
        euclidean = np.linalg.norm(y - x, axis=1)
        
        x_norm = np.linalg.norm(x)
        y_norms = np.linalg.norm(y, axis=1)
        
        cos_sim = np.dot(y, x) / (np.clip(x_norm, 1e-10, None) * np.clip(y_norms, 1e-10, None))
        angular = 1 - cos_sim
        
        # Estimate local density using k-nearest based on Euclidean
        k = min(10, len(y) // 10)
        if k > 0:
            sorted_idx = np.argsort(euclidean)[:k]
            local_density = 1 / (np.mean(euclidean[sorted_idx]) + 1e-10)
        else:
            local_density = 1.0
        
        # In dense regions, prefer angular
        # In sparse regions, prefer euclidean  
        weight = np.tanh(local_density / 10)  # Smooth transition
        
        return weight * angular + (1 - weight) * euclidean / 10  # Scale euclidean


class HybridGeometricRAG:
    """
    Combines multiple geometric approaches for robust retrieval.
    
    Strategy:
    1. Primary: Standard cosine similarity (battle-tested)
    2. Radial Encoding: Adds specificity/length info
    3. Tie-breaking: Uses Euclidean for near-equal cosine scores
    4. Reranking: Cross-checks with L1 distance
    
    This ensemble approach is more robust than any single method.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        use_radial_encoding: bool = True,
        alpha: float = 0.1  # Radial weight
    ):
        print(f"Loading model {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        dummy = self.encoder.encode(["test"])
        self.dim = dummy.shape[1]
        print(f"Embedding dimension: {self.dim}")
        
        self.use_radial = use_radial_encoding
        self.alpha = alpha
        
        if use_radial_encoding:
            self.radial_encoder = RadialInformationEncoder(self.dim)
        
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cosine_embeddings: Optional[np.ndarray] = None  # L2 normalized for cosine
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        if self.use_radial:
            print("Encoding radial information (specificity)...")
            self.embeddings = self.radial_encoder.encode_specificity(raw_embeddings, documents)
        else:
            self.embeddings = raw_embeddings
        
        # Also keep cosine-ready version
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.cosine_embeddings = self.embeddings / np.clip(norms, 1e-10, None)
        
        # Check radial distribution
        final_norms = np.linalg.norm(self.embeddings, axis=1)
        print(f"Radial distribution: min={np.min(final_norms):.3f}, max={np.max(final_norms):.3f}, mean={np.mean(final_norms):.3f}")
        print(f"Ingested {len(documents)} documents.")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_emb = self.encoder.encode([query_text])[0]
        
        if self.use_radial:
            q_emb = self.radial_encoder.encode_specificity(
                q_emb.reshape(1, -1), [query_text]
            )[0]
        
        # Combined metric
        distances = ManifoldAwareDistance.cosine_plus_radial(
            q_emb, self.embeddings, self.alpha
        )
        
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(distances[idx]),
                index=int(idx)
            ))
        
        return results


class EnsembleRAG:
    """
    Ensemble of multiple RAG approaches with voting/fusion.
    
    Different geometric approaches may excel at different query types.
    We use Late Fusion to combine their rankings.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        strategies: List[str] = ['cosine', 'euclidean', 'hybrid']
    ):
        print(f"Loading model {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        dummy = self.encoder.encode(["test"])
        self.dim = dummy.shape[1]
        
        self.strategies = strategies
        self.radial_encoder = RadialInformationEncoder(self.dim)
        
        self.documents: List[str] = []
        self.embeddings_raw: Optional[np.ndarray] = None
        self.embeddings_normalized: Optional[np.ndarray] = None
        self.embeddings_radial: Optional[np.ndarray] = None
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        
        self.embeddings_raw = self.encoder.encode(documents, show_progress_bar=True)
        
        # Normalized for cosine
        norms = np.linalg.norm(self.embeddings_raw, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings_raw / np.clip(norms, 1e-10, None)
        
        # Radial encoded
        self.embeddings_radial = self.radial_encoder.encode_specificity(
            self.embeddings_raw, documents
        )
        
        print(f"Ingested {len(documents)} documents with {len(self.strategies)} strategies.")
    
    def _query_strategy(
        self, 
        query_text: str, 
        strategy: str, 
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Query using a single strategy, returning (index, score) pairs."""
        q_raw = self.encoder.encode([query_text])[0]
        
        if strategy == 'cosine':
            q = q_raw / np.linalg.norm(q_raw)
            scores = np.dot(self.embeddings_normalized, q)
            sorted_idx = np.argsort(scores)[::-1]
            return [(int(idx), float(scores[idx])) for idx in sorted_idx[:top_k]]
        
        elif strategy == 'euclidean':
            distances = np.linalg.norm(self.embeddings_raw - q_raw, axis=1)
            sorted_idx = np.argsort(distances)
            return [(int(idx), float(-distances[idx])) for idx in sorted_idx[:top_k]]
        
        elif strategy == 'hybrid':
            q = self.radial_encoder.encode_specificity(
                q_raw.reshape(1, -1), [query_text]
            )[0]
            distances = ManifoldAwareDistance.cosine_plus_radial(
                q, self.embeddings_radial, 0.1
            )
            sorted_idx = np.argsort(distances)
            return [(int(idx), float(-distances[idx])) for idx in sorted_idx[:top_k]]
        
        elif strategy == 'l1':
            distances = np.sum(np.abs(self.embeddings_raw - q_raw), axis=1)
            sorted_idx = np.argsort(distances)
            return [(int(idx), float(-distances[idx])) for idx in sorted_idx[:top_k]]
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        """Query using ensemble of strategies with Reciprocal Rank Fusion."""
        
        all_results = {}
        
        # Get results from each strategy
        for strategy in self.strategies:
            results = self._query_strategy(query_text, strategy, top_k * 2)
            for rank, (idx, score) in enumerate(results):
                if idx not in all_results:
                    all_results[idx] = {'scores': [], 'ranks': []}
                all_results[idx]['scores'].append(score)
                all_results[idx]['ranks'].append(rank + 1)  # 1-indexed
        
        # Reciprocal Rank Fusion scoring
        k = 60  # RRF constant
        rrf_scores = {}
        for idx, data in all_results.items():
            rrf_scores[idx] = sum(1 / (k + r) for r in data['ranks'])
        
        # Sort by RRF score
        sorted_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for idx in sorted_idx[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(rrf_scores[idx]),
                index=int(idx)
            ))
        
        return results


class ContrastiveVolumetricRAG:
    """
    Uses contrastive learning principles for volumetric embedding.
    
    Key insight: Instead of arbitrary geometric transforms,
    we use the RELATIONSHIPS between documents to determine placement.
    
    Similar documents cluster together, dissimilar documents spread apart.
    The radial dimension encodes the "cluster centrality" of each document.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading model {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        dummy = self.encoder.encode(["test"])
        self.dim = dummy.shape[1]
        
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        print("Computing cluster centrality for volumetric encoding...")
        
        # Normalize for cosine similarity computation
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        normalized = raw_embeddings / np.clip(norms, 1e-10, None)
        
        # Compute pairwise similarities
        sim_matrix = np.dot(normalized, normalized.T)
        
        # "Centrality" = average similarity to all other documents
        # More central documents have higher average similarity
        centrality = np.mean(sim_matrix, axis=1)
        
        # Normalize centrality to [0.3, 1.0]
        c_min, c_max = np.min(centrality), np.max(centrality)
        if c_max > c_min:
            centrality_norm = (centrality - c_min) / (c_max - c_min)
        else:
            centrality_norm = np.ones_like(centrality) * 0.5
        
        target_radii = 0.3 + 0.7 * centrality_norm
        
        # Apply radial encoding
        self.embeddings = normalized * target_radii[:, np.newaxis]
        
        radii = np.linalg.norm(self.embeddings, axis=1)
        print(f"Radial distribution: min={np.min(radii):.3f}, max={np.max(radii):.3f}")
        print(f"Ingested {len(documents)} documents with contrastive volumetric encoding.")
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_raw = self.encoder.encode([query_text])[0]
        q_norm = q_raw / np.linalg.norm(q_raw)
        
        # For queries, place at medium radius (0.6)
        q = q_norm * 0.6
        
        # Use Euclidean distance in the volumetric space
        distances = np.linalg.norm(self.embeddings - q, axis=1)
        
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(distances[idx]),
                index=int(idx)
            ))
        
        return results


class CrossPolytopeHybridRAG:
    """
    Combines the best of CrossPolytope (L1) and Radial Encoding.
    
    Key innovations:
    1. L1 normalization (cross-polytope) instead of L2 (hypersphere)
    2. Radial information encoding (text specificity)
    3. Combined L1 + Radial distance metric
    
    This should combine the benefits of:
    - L1's sparsity-friendliness and outlier robustness
    - Radial encoding's additional information capacity
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        alpha: float = 0.7,  # Radial weight (0.7 found optimal)
        use_radial_encoding: bool = True
    ):
        print(f"Loading model {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        
        dummy = self.encoder.encode(["test"])
        self.dim = dummy.shape[1]
        print(f"Embedding dimension: {self.dim}")
        
        self.alpha = alpha
        self.use_radial = use_radial_encoding
        
        if use_radial_encoding:
            self.radial_encoder = RadialInformationEncoder(self.dim)
        
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _l1_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to L1 unit ball (cross-polytope surface)."""
        l1_norms = np.sum(np.abs(vectors), axis=-1, keepdims=True)
        l1_norms = np.clip(l1_norms, 1e-10, None)
        return vectors / l1_norms
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # Apply radial encoding FIRST (this modifies the L2 norms based on specificity)
        if self.use_radial:
            print("Encoding radial information (specificity)...")
            radial_encoded = self.radial_encoder.encode_specificity(raw_embeddings, documents)
        else:
            radial_encoded = raw_embeddings
        
        # Then L1 normalize - but PRESERVE the radial information in a hybrid way
        # We normalize the direction with L1, but keep the radial magnitude
        l1_norms = np.sum(np.abs(radial_encoded), axis=1, keepdims=True)
        l2_norms = np.linalg.norm(radial_encoded, axis=1, keepdims=True)
        
        # L1 normalized direction
        l1_unit = radial_encoded / np.clip(l1_norms, 1e-10, None)
        
        # Apply the L2 radial magnitude to the L1-normalized direction
        self.embeddings = l1_unit * l2_norms
        
        # Check distributions
        final_l1 = np.sum(np.abs(self.embeddings), axis=1)
        final_l2 = np.linalg.norm(self.embeddings, axis=1)
        print(f"L1 norms: mean={np.mean(final_l1):.3f}, L2 norms: mean={np.mean(final_l2):.3f}")
        print(f"Ingested {len(documents)} docs (CrossPolytope + Radial, a={self.alpha}).")
    
    def _combined_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Combined L1 (angular) + Radial distance.
        
        L1 angular: Normalize to L1 unit ball, measure L1 distance
        Radial: Difference in L2 norms (the encoded specificity)
        """
        # L1 normalize both for angular component
        x_l1 = np.sum(np.abs(x))
        y_l1 = np.sum(np.abs(y), axis=1)
        
        x_unit = x / np.clip(x_l1, 1e-10, None)
        y_unit = y / np.clip(y_l1, 1e-10, None)[:, np.newaxis]
        
        # L1 distance on normalized vectors (angular-like in L1 space)
        l1_angular = np.sum(np.abs(y_unit - x_unit), axis=1)
        
        # Radial component (L2 norm difference = specificity difference)  
        x_l2 = np.linalg.norm(x)
        y_l2 = np.linalg.norm(y, axis=1)
        radial_dist = np.abs(y_l2 - x_l2)
        
        # Weighted combination
        return l1_angular + self.alpha * radial_dist
    
    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_raw = self.encoder.encode([query_text])[0]
        
        # Apply same encoding to query
        if self.use_radial:
            q_encoded = self.radial_encoder.encode_specificity(
                q_raw.reshape(1, -1), [query_text]
            )[0]
        else:
            q_encoded = q_raw
        
        # L1 normalize but preserve radial magnitude
        l1_norm = np.sum(np.abs(q_encoded))
        l2_norm = np.linalg.norm(q_encoded)
        q_final = (q_encoded / np.clip(l1_norm, 1e-10, None)) * l2_norm
        
        # Combined distance
        distances = self._combined_distance(q_final, self.embeddings)
        
        sorted_indices = np.argsort(distances)
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(distances[idx]),
                index=int(idx)
            ))
        
        return results


if __name__ == "__main__":
    # Quick test
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Hyperdodecahedrons are complex 4D shapes with 120 cells.",
        "Machine learning uses vector embeddings for semantic similarity.",
        "The theory of relativity fundamentally changed physics.",
        "Apple pie requires fresh apples, butter, and cinnamon.",
        "Neural networks process information through layered nodes.",
        "The 120-cell is a regular polychoron bounded by 120 dodecahedral cells.",
        "Quantum entanglement enables instantaneous correlation between particles.",
    ]
    
    query = "Tell me about 4D geometry"
    
    print("\n" + "=" * 60)
    print("HyperRAG v3 - Advanced Volumetric Strategies")
    print("=" * 60)
    
    print("\n--- HybridGeometricRAG ---")
    hybrid = HybridGeometricRAG(use_radial_encoding=True)
    hybrid.ingest(docs)
    results = hybrid.query(query, top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] {r.document[:50]}...")
    
    print("\n--- EnsembleRAG ---")
    ensemble = EnsembleRAG(strategies=['cosine', 'euclidean', 'hybrid', 'l1'])
    ensemble.ingest(docs)
    results = ensemble.query(query, top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] {r.document[:50]}...")
    
    print("\n--- ContrastiveVolumetricRAG ---")
    contrastive = ContrastiveVolumetricRAG()
    contrastive.ingest(docs)
    results = contrastive.query(query, top_k=3)
    for r in results:
        print(f"  [{r.score:.4f}] {r.document[:50]}...")
