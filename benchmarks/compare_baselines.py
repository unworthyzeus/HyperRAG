"""
Comprehensive RAG Baseline Comparison

Compare our geometric RAG approaches against:
1. sentence-transformers util.semantic_search (official baseline)
2. Multiple embedding models
3. FAISS vector store (industry standard)

This gives us a proper comparison against established baselines.
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("FAISS not installed. Install with: pip install faiss-cpu")

from src import (
    StandardRAG,
    CrossPolytopeRAG,
    HybridGeometricRAG,
    CrossPolytopeHybridRAG,
)


@dataclass
class Result:
    document: str
    score: float
    index: int


class SentenceTransformersBaseline:
    """
    Uses sentence-transformers util.semantic_search - the official implementation.
    This is what most tutorials and production systems use.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading model {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.corpus_embeddings = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        self.corpus_embeddings = self.encoder.encode(
            documents, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print(f"Ingested {len(documents)} docs (sentence-transformers baseline).")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Result]:
        query_embedding = self.encoder.encode(query_text, convert_to_tensor=True)
        
        # Use official semantic_search utility
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]
        
        results = []
        for hit in hits:
            results.append(Result(
                document=self.documents[hit['corpus_id']],
                score=float(hit['score']),
                index=int(hit['corpus_id'])
            ))
        return results


class FAISSBaseline:
    """
    Uses FAISS for vector similarity search - industry standard.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_type: str = 'flat'):
        if not HAS_FAISS:
            raise ImportError("FAISS not installed")
        
        print(f"Loading model {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.index_type = index_type
        self.documents = []
        self.index = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        dim = embeddings.shape[1]
        
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
        elif self.index_type == 'ivf':
            nlist = min(100, len(documents) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings.astype('float32'))
        
        self.index.add(embeddings.astype('float32'))
        print(f"Ingested {len(documents)} docs (FAISS {self.index_type}).")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Result]:
        query_embedding = self.encoder.encode([query_text])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(Result(
                    document=self.documents[idx],
                    score=float(score),
                    index=int(idx)
                ))
        return results


def load_data(num_samples=300):
    """Load SQuAD dataset."""
    if not HAS_DATASETS:
        print("datasets library not available")
        return None, None
    
    print("Loading SQuAD...")
    dataset = load_dataset("squad", split="validation")
    
    documents = []
    qa_pairs = []
    seen = set()
    
    for item in dataset:
        context = item['context']
        question = item['question']
        
        if context not in seen:
            seen.add(context)
            documents.append(context)
        
        qa_pairs.append((question, context))
        
        if len(documents) >= num_samples:
            break
    
    return documents, qa_pairs[:200]


def evaluate(rag, name: str, documents: List[str], qa_pairs: List[Tuple[str, str]]):
    """Evaluate a RAG engine."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    start = time.time()
    rag.ingest(documents)
    ingest_time = time.time() - start
    
    hits_1 = hits_5 = hits_10 = 0
    rr_sum = 0
    
    start = time.time()
    for question, answer_doc in qa_pairs:
        results = rag.query(question, top_k=10)
        
        for i, r in enumerate(results):
            doc = r.document if hasattr(r, 'document') else r[0]
            if doc == answer_doc or answer_doc in doc:
                if i == 0:
                    hits_1 += 1
                if i < 5:
                    hits_5 += 1
                if i < 10:
                    hits_10 += 1
                rr_sum += 1.0 / (i + 1)
                break
    
    query_time = time.time() - start
    n = len(qa_pairs)
    mrr = rr_sum / n
    
    print(f"MRR: {mrr:.4f}, H@1: {hits_1}/{n}, H@5: {hits_5}/{n}, H@10: {hits_10}/{n}")
    print(f"Time: ingest={ingest_time:.2f}s, query={query_time:.2f}s")
    
    return {
        'name': name,
        'mrr': mrr,
        'hits_1': hits_1 / n * 100,
        'hits_5': hits_5 / n * 100,
        'hits_10': hits_10 / n * 100,
        'ingest_time': ingest_time,
        'query_time': query_time,
    }


def run_comparison():
    """Run full comparison."""
    documents, qa_pairs = load_data(300)
    
    if documents is None:
        return
    
    print(f"\nComparing with {len(documents)} docs, {len(qa_pairs)} queries\n")
    
    results = []
    
    # Different models to test
    models = [
        'all-MiniLM-L6-v2',      # Fast, good quality (384 dim)
        'all-mpnet-base-v2',      # Higher quality (768 dim)
        # 'BAAI/bge-small-en-v1.5', # BGE model (384 dim)
    ]
    
    for model in models:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model}")
        print(f"{'#'*70}")
        
        # 1. Sentence-transformers baseline
        try:
            baseline = SentenceTransformersBaseline(model_name=model)
            r = evaluate(baseline, f"ST Baseline ({model.split('/')[-1]})", documents, qa_pairs)
            results.append(r)
        except Exception as e:
            print(f"Error with ST baseline: {e}")
        
        # 2. FAISS baseline
        if HAS_FAISS:
            try:
                faiss_rag = FAISSBaseline(model_name=model, index_type='flat')
                r = evaluate(faiss_rag, f"FAISS Flat ({model.split('/')[-1]})", documents, qa_pairs)
                results.append(r)
            except Exception as e:
                print(f"Error with FAISS: {e}")
        
        # Only test our custom engines with the first model for speed
        if model == 'all-MiniLM-L6-v2':
            # 3. Our StandardRAG (should match ST baseline)
            std = StandardRAG(model_name=model)
            r = evaluate(std, "Our StandardRAG", documents, qa_pairs)
            results.append(r)
            
            # 4. CrossPolytope L1
            cross = CrossPolytopeRAG(model_name=model, volumetric=True)
            r = evaluate(cross, "Our CrossPolytope L1", documents, qa_pairs)
            results.append(r)
            
            # 5. Hybrid Radial (our best)
            hybrid = HybridGeometricRAG(model_name=model, use_radial_encoding=True, alpha=0.7)
            r = evaluate(hybrid, "Our Hybrid Radial (a=0.7)", documents, qa_pairs)
            results.append(r)
            
            # 6. CrossPolytope+Radial
            cp_hybrid = CrossPolytopeHybridRAG(model_name=model, alpha=0.7)
            r = evaluate(cp_hybrid, "Our CrossPolytope+Radial", documents, qa_pairs)
            results.append(r)
    
    # Summary
    print("\n" + "=" * 90)
    print("COMPREHENSIVE COMPARISON - Sorted by MRR")
    print("=" * 90)
    print(f"{'Engine':<40} | {'MRR':>8} | {'H@1':>7} | {'H@5':>7} | {'H@10':>7}")
    print("-" * 90)
    
    for r in sorted(results, key=lambda x: x['mrr'], reverse=True):
        print(f"{r['name']:<40} | {r['mrr']:>8.4f} | {r['hits_1']:>6.1f}% | {r['hits_5']:>6.1f}% | {r['hits_10']:>6.1f}%")
    
    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    
    baseline = next((r for r in results if 'ST Baseline' in r['name'] and 'MiniLM' in r['name']), None)
    our_best = next((r for r in results if 'Hybrid Radial' in r['name']), None)
    
    if baseline and our_best:
        improvement = (our_best['mrr'] - baseline['mrr']) / baseline['mrr'] * 100
        print(f"Official ST Baseline: MRR={baseline['mrr']:.4f}")
        print(f"Our Best (Hybrid Radial): MRR={our_best['mrr']:.4f}")
        print(f"Improvement over official baseline: {improvement:+.2f}%")
    
    return results


if __name__ == "__main__":
    results = run_comparison()

