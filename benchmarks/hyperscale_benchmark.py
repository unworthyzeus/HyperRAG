"""
Hyper-Scale Benchmark
Tests "Semantic Collapse" hypothesis by scaling up to 10,000+ documents.

Hypothesis:
1. Retrieval accuracy drops significantly as N increases (Collapse).
2. Hybrid Radial resists this collapse better than Standard Cosine.

Target Models:
- all-MiniLM-L6-v2 (Small, likely to collapse early)
- intfloat/e5-large-v2 (Large, robust baseline)
"""

import time
import sys
import gc
import numpy as np
import json
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from typing import Dict, List, Tuple

from HyperRAG import StandardRAG, HybridGeometricRAG

# --- Optimized Classes for Scale with PROPER BATCH CONTROL ---

class BatchStandardRAG(StandardRAG):
    def __init__(self, model_name, batch_size=32):
        super().__init__(model_name)
        self.batch_size = batch_size
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} docs (Standard, BS={self.batch_size})...")
        self.embeddings = self.encoder.encode(
            documents, 
            batch_size=self.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.clip(norms, 1e-10, None)

class BatchHybridRAG(HybridGeometricRAG):
    def __init__(self, model_name, batch_size=32, **kwargs):
        super().__init__(model_name, **kwargs)
        self.batch_size = batch_size
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} docs (Hybrid, BS={self.batch_size})...")
        # Ensure we pass batch_size to encode
        raw_embeddings = self.encoder.encode(
            documents, 
            batch_size=self.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        if self.use_radial:
            self.embeddings = self.radial_encoder.encode_specificity(raw_embeddings, documents)
        else:
            self.embeddings = raw_embeddings
            
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.cosine_embeddings = self.embeddings / np.clip(norms, 1e-10, None)

class ScaleEvaluator:
    def __init__(self, model_name, batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"Loading {model_name} (Evaluation Batch Size: {batch_size})")
        # We don't pre-load here effectively because we re-init wrappers
        # In a real app we'd share the encoder object.
        # But for benchmark simplicity and memory safety (clearing between runs)
        # we will let them init their own, but we destroy them aggressively.

    def run_experiment(self, documents: List[str], qa_pairs, doc_count_label: str):
        results = {}
        
        # 1. Standard RAG
        print(f"\n--- Testing Standard RAG ({doc_count_label}) ---")
        try:
            rag = BatchStandardRAG(self.model_name, batch_size=self.batch_size)
            start = time.time()
            rag.ingest(documents)
            t_ingest = time.time() - start
            
            metrics = self.evaluate(rag, qa_pairs)
            results['Standard'] = metrics
            print(f"Standard: {metrics} (Ingest: {t_ingest:.1f}s)")
            
            del rag
            self.cleanup()
        except Exception as e:
            print(f"Standard Failed: {e}")

        # 2. Hybrid Radial (a=0.1 for Large, a=0.7 for Small)
        alpha = 0.1 if "large" in self.model_name else 0.7
        print(f"\n--- Testing Hybrid Radial a={alpha} ({doc_count_label}) ---")
        try:
            rag = BatchHybridRAG(self.model_name, batch_size=self.batch_size, use_radial_encoding=True, alpha=alpha)
            start = time.time()
            rag.ingest(documents)
            
            metrics = self.evaluate(rag, qa_pairs)
            results['Hybrid'] = metrics
            print(f"Hybrid: {metrics}")
            
            del rag
            self.cleanup()
        except Exception as e:
            print(f"Hybrid Failed: {e}")
            
        return results

    def evaluate(self, rag, qa_pairs) -> Dict:
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

    def cleanup(self):
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except: pass

# --- Data Loading ---

def load_data_at_scale(target_docs=10000, target_queries=1000):
    print(f"Loading SQuAD (Target: {target_docs} docs)...")
    
    documents = []
    qa_pairs = []
    seen = set()
    
    # Validation split first
    dataset_val = load_dataset("squad", split="validation")
    for item in dataset_val:
        ctx = item['context']
        if ctx not in seen:
            seen.add(ctx)
            documents.append(ctx)
        if len(qa_pairs) < target_queries:
            qa_pairs.append((item['question'], ctx))
            
    # Train split if needed
    if len(documents) < target_docs:
        print("Fetching from train split...")
        dataset_train = load_dataset("squad", split="train")
        for item in dataset_train:
            ctx = item['context']
            if ctx not in seen:
                seen.add(ctx)
                documents.append(ctx)
            # Add more queries if we are short (unlikely)
            if len(qa_pairs) < target_queries:
                qa_pairs.append((item['question'], ctx))
            
            if len(documents) >= target_docs:
                break
                
    print(f"Loaded {len(documents)} docs, {len(qa_pairs)} queries.")
    return documents[:target_docs], qa_pairs[:target_queries]

# --- Main Run ---

def run_hyperscale():
    # Scales to test
    scales = [1000, 5000, 10000, 15000] 
    
    # Models
    models = [
        ("sentence-transformers/all-MiniLM-L6-v2", 32), # Fast
        ("intfloat/e5-large-v2", 2) # VERY SAFE batch size for 1024 dim
    ]
    
    # Ensure we get enough docs
    full_docs, qa_pairs = load_data_at_scale(target_docs=15000, target_queries=500) # Reduce queries to 500 for speed
    
    summary = []
    
    for model_name, bs in models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} (Batch Size: {bs})")
        print(f"{'='*60}")
        
        evaluator = ScaleEvaluator(model_name, batch_size=bs)
        
        for scale in scales:
            if scale > len(full_docs):
                print(f"Skipping scale {scale} (not enough docs)")
                continue
                
            subset_docs = full_docs[:scale]
            print(f"\n>>> SCALE: {scale} Documents")
            
            res = evaluator.run_experiment(subset_docs, qa_pairs, f"{scale} docs")
            
            if 'Standard' in res and 'Hybrid' in res:
                diff = res['Hybrid']['mrr'] - res['Standard']['mrr']
                summary.append({
                    "model": model_name,
                    "scale": scale,
                    "standard_mrr": res['Standard']['mrr'],
                    "hybrid_mrr": res['Hybrid']['mrr'],
                    "diff": diff
                })
                print(f"Result: Std={res['Standard']['mrr']:.4f}, Hybrid={res['Hybrid']['mrr']:.4f}, Diff={diff:+.4f}")
    
    print("\n\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))
    
    # Save
    with open("results/semantic_collapse_study.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_hyperscale()

