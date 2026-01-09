"""
Alpha sweep for HybridGeometricRAG
Test different radial weights to find the optimal value
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from HyperRAG import HybridGeometricRAG

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_data(num_samples=300):
    if not HAS_DATASETS:
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


def evaluate(alpha, documents, qa_pairs):
    """Evaluate HybridGeometricRAG with given alpha"""
    rag = HybridGeometricRAG(use_radial_encoding=True, alpha=alpha)
    rag.ingest(documents)
    
    hits = 0
    rr_sum = 0
    
    for question, answer_doc in qa_pairs:
        results = rag.query(question, top_k=10)
        
        for i, r in enumerate(results):
            if r.document == answer_doc or answer_doc in r.document:
                if i < 5:
                    hits += 1
                rr_sum += 1.0 / (i + 1)
                break
    
    mrr = rr_sum / len(qa_pairs)
    hits_5 = hits / len(qa_pairs) * 100
    
    return mrr, hits_5


if __name__ == "__main__":
    documents, qa_pairs = load_data(300)
    
    if documents is None:
        print("Datasets library not available")
        exit(1)
    
    print(f"\nTesting with {len(documents)} docs, {len(qa_pairs)} queries\n")
    
    # Test range of alpha values
    alphas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    print(f"{'Alpha':<10} | {'MRR':>10} | {'Hits@5':>10}")
    print("-" * 40)
    
    results = []
    for alpha in alphas:
        mrr, hits5 = evaluate(alpha, documents, qa_pairs)
        print(f"{alpha:<10.2f} | {mrr:>10.4f} | {hits5:>9.1f}%")
        results.append((alpha, mrr, hits5))
    
    print("\n" + "=" * 40)
    best = max(results, key=lambda x: x[1])
    print(f"Best alpha: {best[0]} with MRR={best[1]:.4f}, Hits@5={best[2]:.1f}%")

