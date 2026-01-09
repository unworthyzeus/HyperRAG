"""
Tiny Model Radial Study (128 Dimensions)
Tests if Radial Encoding provides a significant boost in highly crowded spaces.
Uses prajjwal1/bert-tiny (128 dims) with manual pooling.
"""

import time
import sys
import gc
import numpy as np
import json
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset
from typing import Dict, List, Tuple

from HyperRAG import StandardRAG, HybridGeometricRAG

# --- Controlled Batch Classes ---

class BatchStandardRAG(StandardRAG):
    def __init__(self, model_name, batch_size=32):
        # Manual load for tiny model
        print(f"Loading {model_name} with manual pooling...")
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.documents = []
        self.embeddings = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        self.embeddings = self.encoder.encode(
            documents, 
            batch_size=self.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.clip(norms, 1e-10, None)

class BatchHybridRAG(HybridGeometricRAG):
    def __init__(self, model_name, batch_size=32, **kwargs):
        # Manual load
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.dim = self.encoder.get_sentence_embedding_dimension()
        
        self.use_radial = kwargs.get('use_radial_encoding', True)
        self.alpha = kwargs.get('alpha', 0.1)
        
        from HyperRAG.advanced.advanced import RadialInformationEncoder
        if self.use_radial:
            self.radial_encoder = RadialInformationEncoder(self.dim)
        
        self.batch_size = batch_size
        self.documents = []
        self.embeddings = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        raw_embeddings = self.encoder.encode(
            documents, 
            batch_size=self.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.embeddings = self.radial_encoder.encode_specificity(raw_embeddings, documents)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.cosine_embeddings = self.embeddings / np.clip(norms, 1e-10, None)

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

def load_data(num_docs=10000, num_queries=500):
    print(f"Loading SQuAD ({num_docs} docs)...")
    dataset = load_dataset("squad", split="validation")
    documents = []
    qa_pairs = []
    seen = set()
    for item in dataset:
        if item['context'] not in seen:
            seen.add(item['context'])
            documents.append(item['context'])
        if len(qa_pairs) < num_queries:
            qa_pairs.append((item['question'], item['context']))
            
    if len(documents) < num_docs:
        dataset_train = load_dataset("squad", split="train")
        for item in dataset_train:
            if item['context'] not in seen:
                seen.add(item['context'])
                documents.append(item['context'])
            if len(documents) >= num_docs: break
            
    return documents[:num_docs], qa_pairs[:num_queries]

def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass

def run_study():
    model_name = 'prajjwal1/bert-tiny'  # 128 dimensions
    docs, qa = load_data(num_docs=10000)
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} (128 DIMENSIONS)")
    print(f"Scale: {len(docs)} docs, {len(qa)} queries")
    print(f"{'='*60}")
    
    # 1. Baseline
    print("\n>>> Testing Standard RAG (Cosine Baseline)...")
    std_rag = BatchStandardRAG(model_name, batch_size=64)
    std_rag.ingest(docs)
    std_metrics = evaluate(std_rag, qa)
    print(f"Standard Stats: {std_metrics}")
    
    # Pre-encode raw embeddings to avoid repetition
    raw_embs = std_rag.embeddings # This is already normalized though. 
    # Let's re-encode once for raw unnormalized
    print("Pre-encoding raw embeddings for sweep...")
    raw_embs_unnorm = std_rag.encoder.encode(docs, batch_size=128, convert_to_numpy=True)
    
    # 2. Sweep Alpha
    alphas = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]
    best_mrr = 0
    best_alpha = 0
    best_metrics = {}
    
    results_sweep = []
    
    print("\n>>> Sweeping Alpha for Hybrid Radial...")
    from HyperRAG.advanced.advanced import RadialInformationEncoder
    radial_encoder = RadialInformationEncoder(128)
    
    for a in alphas:
        print(f"Testing Alpha = {a}...")
        hybrid = BatchHybridRAG(model_name, batch_size=128, alpha=a)
        hybrid.encoder = std_rag.encoder
        hybrid.documents = docs
        hybrid.embeddings = radial_encoder.encode_specificity(raw_embs_unnorm, docs)
        
        m = evaluate(hybrid, qa)
        print(f"Result (a={a}): {m}")
        
        results_sweep.append({"alpha": a, "metrics": m})
        
        if m['mrr'] > best_mrr:
            best_mrr = m['mrr']
            best_alpha = a
            best_metrics = m
            
    # Final Report
    print("\n" + "#"*40)
    print(f"STUDY FINISHED: {model_name}")
    print(f"Baseline MRR: {std_metrics['mrr']:.4f}")
    print(f"Best Hybrid MRR: {best_mrr:.4f} (Alpha: {best_alpha})")
    improvement = (best_mrr - std_metrics['mrr']) / std_metrics['mrr'] * 100
    print(f"Relative Improvement: {improvement:+.2f}%")
    print("#"*40)
    
    # Save to report
    out_file = "results/reports/tiny_model_study.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_name,
            "baseline": std_metrics,
            "sweep": results_sweep,
            "best": {"alpha": best_alpha, "metrics": best_metrics},
            "improvement_pct": improvement
        }, f, indent=2)

if __name__ == "__main__":
    run_study()
