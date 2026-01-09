"""
Benchmark for Experimental RAG Engines
Tests WhitenedRAG and ClusterTreeRAG on e5-large-v2
"""

import time
import sys
import gc
import numpy as np
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from typing import Dict, List

from HyperRAG import WhitenedRAG, ClusterTreeRAG

# --- Wrapper Classes for 4GB VRAM (Batch Size 4) ---

class WhitenedTinyBatch(WhitenedRAG):
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} docs (Whitened, BS=4)...")
        # RAW embeddings (no norm)
        embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=4, convert_to_numpy=True)
        
        print("Computing ZCA Transform...")
        self.mean = np.mean(embeddings, axis=0)
        centered = embeddings - self.mean
        cov = np.dot(centered.T, centered) / (len(documents) - 1)
        U, S, V = np.linalg.svd(cov)
        epsilon = 1e-5
        self.transform_matrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S+epsilon))), U.T)
        
        print("Whitening corpus...")
        self.embeddings = np.dot(centered, self.transform_matrix)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.clip(norms, 1e-10, None)
        print("Ingest Done.")

class ClusterTreeTinyBatch(ClusterTreeRAG):
    def ingest(self, documents: List[str]) -> None:
        self.doc_registry = documents
        print(f"Encoding {len(documents)} docs (ClusterTree, BS=4)...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=4, convert_to_numpy=True)
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embs = embeddings / np.clip(norms, 1e-10, None)
        
        print(f"Clustering into {self.n_clusters} manifolds...")
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        labels = self.kmeans.fit_predict(normalized_embs)
        self.cluster_centroids = self.kmeans.cluster_centers_
        
        print("Computing residuals...")
        self.cluster_docs = {i: [] for i in range(self.n_clusters)}
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            centroid = self.cluster_centroids[label]
            residual = emb - centroid # Raw difference
            res_norm = np.linalg.norm(residual)
            if res_norm > 1e-9:
                residual = residual / res_norm
            self.cluster_docs[label].append((i, residual))
        print("Ingest Done.")

# --- Eval Logic ---

def load_data(num_docs=3000, num_queries=800):
    print(f"Loading SQuAD ({num_docs} docs)...")
    dataset = load_dataset("squad", split="validation")
    documents = []
    qa_pairs = []
    seen = set()
    for item in dataset:
        if item['context'] not in seen:
            seen.add(item['context'])
            documents.append(item['context'])
        qa_pairs.append((item['question'], item['context']))
        
    if len(documents) < num_docs:
        dataset_train = load_dataset("squad", split="train")
        for item in dataset_train:
            if item['context'] not in seen:
                seen.add(item['context'])
                documents.append(item['context'])
            qa_pairs.append((item['question'], item['context']))
            if len(documents) >= num_docs: break
            
    return documents[:num_docs], qa_pairs[:num_queries]

def evaluate(rag, qa_pairs) -> Dict:
    hits_1 = hits_5 = 0
    rr_sum = 0
    n = len(qa_pairs)
    for question, answer in qa_pairs:
        results = rag.query(question, top_k=5)
        for i, r in enumerate(results):
            doc = r.document
            if doc == answer or answer in doc:
                if i == 0: hits_1 += 1
                if i < 5: hits_5 += 1
                rr_sum += 1.0 / (i + 1)
                break
    return {'mrr': rr_sum/n, 'hits_5': hits_5/n*100}

def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass

# --- Run ---

def run_exp():
    docs, qa = load_data()
    model = 'intfloat/e5-large-v2'
    print(f"Target: {model}, Baseline MRR: ~0.7923")
    
    # 1. Whitening
    print("\n>>> Testing WhitenedRAG (Isotropic Correction)...")
    try:
        eng = WhitenedTinyBatch(model)
        start = time.time()
        eng.ingest(docs)
        print(f"Ingest time: {time.time()-start:.2f}s")
        m = evaluate(eng, qa)
        print(f"Whitened Stats: {m}")
        del eng
        cleanup()
    except Exception as e:
        print(f"Whiteneing Failed: {e}")
        
    # 2. ClusterTree
    k = int(np.sqrt(len(docs))) # Rule of thumb
    print(f"\n>>> Testing ClusterTreeRAG (Locally Adaptive, k={k})...")
    try:
        eng = ClusterTreeTinyBatch(model, n_clusters=k)
        start = time.time()
        eng.ingest(docs)
        print(f"Ingest time: {time.time()-start:.2f}s")
        m = evaluate(eng, qa)
        print(f"ClusterTree Stats: {m}")
        del eng
        cleanup()
    except Exception as e:
        print(f"ClusterTree Failed: {e}")

if __name__ == "__main__":
    run_exp()

