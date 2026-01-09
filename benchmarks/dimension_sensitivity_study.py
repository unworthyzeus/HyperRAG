"""
Dimension Sensitivity Study
Tests how the effectiveness of Radial Encoding scales with model dimensionality.
Models: 128D (tiny), 256D (mini), 512D (small), 768D (base)
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
from HyperRAG.advanced.advanced import RadialInformationEncoder

class BatchRAGWrapper:
    """Wrapper to handle manual loading and batching for BERT-style models."""
    def __init__(self, model_name, batch_size=64):
        print(f"Loading {model_name}...")
        word_embedding_model = models.Transformer(model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        
    def get_embeddings(self, documents: List[str]):
        return self.encoder.encode(
            documents, 
            batch_size=self.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )

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

def load_data(num_docs=5000, num_queries=300):
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
    return documents[:num_docs], qa_pairs[:num_queries]

def run_study():
    # Model configs: (name, dimension)
    models_to_test = [
        ('prajjwal1/bert-tiny', 128),
        ('prajjwal1/bert-mini', 256),
        ('prajjwal1/bert-small', 512),
        ('sentence-transformers/all-distilroberta-v1', 768)
    ]
    
    docs, qa = load_data()
    alphas = [0.0, 0.05, 0.1, 0.3, 0.7] # 0.0 is baseline
    
    results = []
    
    for model_name, dim in models_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_name} ({dim} dims)")
        print(f"{'='*60}")
        
        try:
            wrapper = BatchRAGWrapper(model_name)
            raw_embs = wrapper.get_embeddings(docs)
            
            # Setup Engines
            # Standard (Baseline)
            std_rag = StandardRAG(model_name)
            std_rag.encoder = wrapper.encoder # Share encoder
            std_rag.documents = docs
            # Normalize raw embs for standard
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
            std_rag.embeddings = raw_embs / np.clip(norms, 1e-10, None)
            
            baseline_m = evaluate(std_rag, qa)
            print(f"Baseline MRR: {baseline_m['mrr']:.4f}")
            
            # Hybrid Sweep
            radial_encoder = RadialInformationEncoder(dim)
            best_mrr = baseline_m['mrr']
            best_alpha = 0.0
            
            findings = {"model": model_name, "dim": dim, "baseline": baseline_m, "sweep": []}
            
            for a in alphas:
                if a == 0.0: continue
                
                hybrid = HybridGeometricRAG(model_name, alpha=a)
                hybrid.encoder = wrapper.encoder
                hybrid.documents = docs
                hybrid.radial_encoder = radial_encoder
                hybrid.embeddings = radial_encoder.encode_specificity(raw_embs, docs)
                
                m = evaluate(hybrid, qa)
                print(f"Alpha {a}: MRR={m['mrr']:.4f} (Diff: {m['mrr']-baseline_m['mrr']:+.4f})")
                
                findings["sweep"].append({"alpha": a, "mrr": m['mrr']})
                if m['mrr'] > best_mrr:
                    best_mrr = m['mrr']
                    best_alpha = a
            
            findings["best_alpha"] = best_alpha
            findings["gain"] = (best_mrr - baseline_m['mrr']) / (baseline_m['mrr'] + 1e-10) * 100
            results.append(findings)
            
            # Cleanup for next model
            del wrapper, raw_embs, std_rag, radial_encoder
            gc.collect()
            
        except Exception as e:
            print(f"Failed model {model_name}: {e}")

    # Final Summary Table
    print("\n" + "!"*60)
    print("FINAL DIMENSION SENSITIVITY SUMMARY")
    print("!"*60)
    print(f"{'Dims':<6} | {'Model':<30} | {'Baseline':<10} | {'Gain %':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['dim']:<6} | {r['model']:<30} | {r['baseline']['mrr']:<10.4f} | {r['gain']:>+7.2f}%")
    
    with open("results/reports/dimension_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_study()
