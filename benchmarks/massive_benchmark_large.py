"""
MASSIVE Benchmark - Large Models Extension
Tests 1024-dim models with tiny batch sizes for 4GB VRAM.

Models:
- BAAI/bge-large-en-v1.5 (1024 dim)
- intfloat/e5-large-v2 (1024 dim)
"""

import numpy as np
import time
import json
import sys
import gc
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

try:
    from datasets import load_dataset
except ImportError:
    print("datasets library not available")
    exit(1)

from HyperRAG import HybridGeometricRAG
from HyperRAG import StandardRAG, CrossPolytopeRAG

# Re-use classes from massive_benchmark.py but with stricter memory control
class STBaseline:
    def __init__(self, model_name: str, batch_size: int = 4):
        self.encoder = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.documents = []
        self.corpus_embeddings = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        self.corpus_embeddings = self.encoder.encode(
            documents, 
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=self.batch_size
        )
    
    def query(self, query_text: str, top_k: int = 10):
        query_embedding = self.encoder.encode(query_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]
        return [type('R', (), {
            'document': self.documents[h['corpus_id']], 
            'score': h['score'], 
            'index': h['corpus_id']
        })() for h in hits]

class HybridRAGTinyBatch(HybridGeometricRAG):
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents (batch_size=4)...")
        # Very small batch size for encoding
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=4)
        
        if self.use_radial:
            print("Encoding radial information (specificity)...")
            self.embeddings = self.radial_encoder.encode_specificity(raw_embeddings, documents)
        else:
            self.embeddings = raw_embeddings
        
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.cosine_embeddings = self.embeddings / np.clip(norms, 1e-10, None)
        
        final_norms = np.linalg.norm(self.embeddings, axis=1)
        print(f"Radial distribution: min={np.min(final_norms):.3f}, max={np.max(final_norms):.3f}, mean={np.mean(final_norms):.3f}")
        print(f"Ingested {len(documents)} documents.")

class CrossPolytopeTinyBatch(CrossPolytopeRAG):
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents (batch_size=4)...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=4)
        
        if self.volumetric:
            print("Applying volumetric transform (L1 ball)...")
            self.embeddings = self.geometry.project_to_volume(raw_embeddings)
        else:
            self.embeddings = self.geometry.project_to_surface(raw_embeddings)
        
        print(f"Ingested {len(documents)} documents (cross-polytope, volumetric: {self.volumetric}).")

def load_large_data(num_docs: int = 3000, num_queries: int = 800):
    print(f"Loading SQuAD (target: {num_docs} docs, {num_queries} queries)...")
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
    
    if len(documents) < num_docs:
        print("Loading from train split for more data...")
        train = load_dataset("squad", split="train")
        for item in train:
            context = item['context']
            question = item['question']
            if context not in seen:
                seen.add(context)
                documents.append(context)
            qa_pairs.append((question, context))
            if len(documents) >= num_docs:
                break
    
    return documents[:num_docs], qa_pairs[:num_queries]

def evaluate(rag, qa_pairs) -> Dict:
    hits_1 = hits_5 = hits_10 = 0
    rr_sum = 0
    for question, answer_doc in qa_pairs:
        results = rag.query(question, top_k=10)
        for i, r in enumerate(results):
            doc = r.document if hasattr(r, 'document') else r[0]
            if doc == answer_doc or answer_doc in doc:
                if i == 0: hits_1 += 1
                if i < 5: hits_5 += 1
                if i < 10: hits_10 += 1
                rr_sum += 1.0 / (i + 1)
                break
    n = len(qa_pairs)
    return {'mrr': rr_sum/n, 'hits_1': hits_1/n*100, 'hits_5': hits_5/n*100, 'hits_10': hits_10/n*100}

def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def run_large_model_benchmark():
    documents, qa_pairs = load_large_data(num_docs=3000, num_queries=800)
    
    # Large 1024-dim models with tiny batch size
    models = [
        ('BAAI/bge-large-en-v1.5', 1024, 4),
        ('intfloat/e5-large-v2', 1024, 4)
    ]
    
    alphas = [0.1, 0.2, 0.3]
    
    for model_name, dim, batch_size in models:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name} ({dim} dims, batch_size={batch_size})")
        print(f"{'#'*80}")
        
        # 1. Baseline
        print(f"\n[1/5] ST Baseline...")
        baseline_mrr = 0
        try:
            baseline = STBaseline(model_name, batch_size=batch_size)
            start = time.time()
            baseline.ingest(documents)
            print(f"  Ingest: {time.time()-start:.2f}s")
            metrics = evaluate(baseline, qa_pairs)
            print(f"  MRR: {metrics['mrr']:.4f}, H@1: {metrics['hits_1']:.1f}%, H@5: {metrics['hits_5']:.1f}%")
            baseline_mrr = metrics['mrr']
            del baseline
            cleanup()
        except Exception as e:
            print(f"  Error: {e}")
            
        # 2. CrossPolytope L1
        print(f"\n[2/5] CrossPolytope L1...")
        try:
            cross = CrossPolytopeTinyBatch(model_name=model_name, volumetric=True)
            cross.ingest(documents)
            metrics = evaluate(cross, qa_pairs)
            print(f"  MRR: {metrics['mrr']:.4f}, H@1: {metrics['hits_1']:.1f}%, H@5: {metrics['hits_5']:.1f}%")
            if baseline_mrr:
                print(f"  vs Baseline: {(metrics['mrr'] - baseline_mrr) / baseline_mrr * 100:+.2f}%")
            del cross
            cleanup()
        except Exception as e:
            print(f"  Error: {e}")
            
        # 3-5. Hybrid Radial
        for i, alpha in enumerate(alphas):
            print(f"\n[{3+i}/5] Hybrid Radial (alpha={alpha})...")
            try:
                hybrid = HybridRAGTinyBatch(model_name=model_name, use_radial_encoding=True, alpha=alpha)
                hybrid.ingest(documents)
                metrics = evaluate(hybrid, qa_pairs)
                print(f"  MRR: {metrics['mrr']:.4f}, H@1: {metrics['hits_1']:.1f}%, H@5: {metrics['hits_5']:.1f}%")
                if baseline_mrr:
                    print(f"  vs Baseline: {(metrics['mrr'] - baseline_mrr) / baseline_mrr * 100:+.2f}%")
                del hybrid
                cleanup()
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    print("Starting EXTRA LARGE MODEL benchmark (Tiny Batch)...")
    run_large_model_benchmark()

