"""
Experimental RAG Engines preventing Semantic Collapse.

Approaches:
1. WhitenedRAG: Applies ZCA whitening to force an isotropic distribution, removing global "common" signals that cause crowding/collapse.
2. ClusterTreeRAG (Fractal): Hierarchical retrieval that zooms into local clusters and re-normalizes space, allowing "fine-grained" distinctions that are usually lost in global noise.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Optional
from ..core.engines import BaseRAGEngine, RetrievalResult

class WhitenedRAG(BaseRAGEngine):
    """
    RAG engine that applies Whitening (Zero-phase Component Analysis - ZCA).
    
    Problem Solved: 
    In high dimensions, embedding clouds often look like "cigars" or narrow cones 
    (anisotropy). Most dimensions are correlated or useless, leading to "hubness" 
    and semantic collapse where everything looks somewhat similar to everything else.
    
    Solution:
    We force the embedding distribution to be isotropic (spherical covariance).
    1. Center the data (subtract mean).
    2. Rotate/Scale so covariance is Identity (Whitening).
    This maximizes the information capacity of the available dimensions.
    """
    
    def __init__(self, model_name='intfloat/e5-large-v2'):
        super().__init__(model_name)
        self.mean = None
        self.transform_matrix = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        # Get raw embeddings (do not normalize yet)
        embeddings = self.encoder.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        
        print("Computing Whitening Transform...")
        # 1. Center
        self.mean = np.mean(embeddings, axis=0)
        centered = embeddings - self.mean
        
        # 2. Compute Covariance
        cov = np.dot(centered.T, centered) / (len(documents) - 1)
        
        # 3. Eigen decomposition
        U, S, V = np.linalg.svd(cov)
        
        # 4. Calculate ZCA Matrix: U * diag(1/sqrt(S)) * U.T
        # Add epsilon to S to avoid division by zero
        epsilon = 1e-5
        inv_sqrt_S = np.diag(1.0 / np.sqrt(S + epsilon))
        self.transform_matrix = np.dot(np.dot(U, inv_sqrt_S), U.T)
        
        # 5. Apply Transform
        print("Applying whitening to corpus...")
        self.embeddings = np.dot(centered, self.transform_matrix)
        
        # 6. Final L2 Normalize helps for stability after whitening
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.clip(norms, 1e-10, None)
        
        print("Ingest complete.")

    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        # Encode
        q_raw = self.encoder.encode([query_text])[0]
        
        # Apply EXACT same transform
        q_centered = q_raw - self.mean
        q_whitened = np.dot(q_centered, self.transform_matrix)
        
        # Normalize
        q_norm = q_whitened / np.linalg.norm(q_whitened)
        
        # Cosine similarity in Whitened Space
        scores = np.dot(self.embeddings, q_norm)
        
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in sorted_indices[:top_k]:
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=float(scores[idx]),
                index=int(idx)
            ))
        return results


class ClusterTreeRAG(BaseRAGEngine):
    """
    RAG engine that fights "Semantic Crowding" by creating local metric spaces.
    
    Problem:
    With 3000+ docs, "semantic collapse" occurs. All "Science" docs bunch together.
    Global cosine similarity can't distinguish between "Quantum Entanglement" and 
    "Quantum Superposition" well because they are both overwhelmed by the massive 
    "Science" vector component they share.
    
    Solution:
    1. Cluster the data (e.g., 20 clusters).
    2. Calculate "Residuals": Subtract the cluster centroid from the document vectors.
    3. Retrieval:
       a. Find nearest cluster(s).
       b. Inside the cluster, compare Query-Residual vs Doc-Residual.
       
    This is effectively "zooming in" to the local manifold and re-expanding it 
    to fill the coordinate space, removing the common "noise".
    """
    
    def __init__(self, model_name='intfloat/e5-large-v2', n_clusters=20):
        super().__init__(model_name)
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_centroids = None
        self.cluster_docs = {} # cluster_id -> list of (global_index, local_embedding)
        self.doc_registry = [] # global_index -> document_text
        
    def ingest(self, documents: List[str]) -> None:
        self.doc_registry = documents
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize first for clustering (spherical k-means approx)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embs = embeddings / np.clip(norms, 1e-10, None)
        
        print(f"Clustering into {self.n_clusters} local manifolds...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = self.kmeans.fit_predict(normalized_embs)
        self.cluster_centroids = self.kmeans.cluster_centers_
        
        # Process each cluster
        print("Computing local residuals...")
        self.cluster_docs = {i: [] for i in range(self.n_clusters)}
        
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            # Residual: The vector pointing from centroid to doc
            # This represents the "unique" info of the doc relative to its cluster topic
            centroid = self.cluster_centroids[label]
            residual = emb - centroid
            
            # Normalize residual to emphasize DIRECTION of difference
            res_norm = np.linalg.norm(residual)
            if res_norm > 1e-9:
                residual = residual / res_norm
                
            self.cluster_docs[label].append((i, residual))
            
        print("Ingest complete.")

    def query(self, query_text: str, top_k: int = 5) -> List[RetrievalResult]:
        q_raw = self.encoder.encode([query_text])[0]
        q_unit = q_raw / np.linalg.norm(q_raw)
        
        # 1. Find top clusters (routing)
        # Cosine similarity to centroids
        cluster_scores = np.dot(self.cluster_centroids, q_unit)
        # Search top 3 clusters to be safe (fuzzy routing)
        top_clusters = np.argsort(cluster_scores)[::-1][:3]
        
        candidates = []
        
        # 2. Local Search
        for c_idx in top_clusters:
            centroid = self.cluster_centroids[c_idx]
            
            # Calculate Query Residual relative to this cluster
            # "How is my query different from the generic topic of this cluster?"
            q_residual = q_raw - centroid
            q_res_norm = np.linalg.norm(q_residual)
            if q_res_norm > 1e-9:
                q_residual = q_residual / q_res_norm
            
            docs_in_cluster = self.cluster_docs[c_idx]
            
            for global_idx, doc_residual in docs_in_cluster:
                # Similarity of the *differences*
                # High score if the query deviates from the centroid in the SAME WAY the doc does
                score = np.dot(doc_residual, q_residual)
                
                # We can blend this with global score, but let's try pure local first
                # to maximize the "noise reduction" effect.
                candidates.append((score, global_idx))
        
        # Sort candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in candidates[:top_k]:
            results.append(RetrievalResult(
                document=self.doc_registry[idx],
                score=float(score),
                index=int(idx)
            ))
            
        return results
