"""
Test CrossPolytopeHybridRAG vs other best performers
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from src import (
    StandardRAG,
    CrossPolytopeRAG,
    HybridGeometricRAG,
    CrossPolytopeHybridRAG,
)

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


def evaluate(rag, name, documents, qa_pairs):
    """Evaluate a RAG engine"""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")
    
    start = time.time()
    rag.ingest(documents)
    ingest_time = time.time() - start
    
    hits_1 = hits_5 = 0
    rr_sum = 0
    
    start = time.time()
    for question, answer_doc in qa_pairs:
        results = rag.query(question, top_k=10)
        
        for i, r in enumerate(results):
            if r.document == answer_doc or answer_doc in r.document:
                if i == 0:
                    hits_1 += 1
                if i < 5:
                    hits_5 += 1
                rr_sum += 1.0 / (i + 1)
                break
    
    query_time = time.time() - start
    mrr = rr_sum / len(qa_pairs)
    
    print(f"Results: MRR={mrr:.4f}, Hits@1={hits_1}, Hits@5={hits_5}")
    print(f"Time: ingest={ingest_time:.2f}s, query={query_time:.2f}s")
    
    return mrr, hits_1 / len(qa_pairs) * 100, hits_5 / len(qa_pairs) * 100


if __name__ == "__main__":
    documents, qa_pairs = load_data(300)
    
    if documents is None:
        print("Datasets library not available")
        exit(1)
    
    print(f"Testing with {len(documents)} docs, {len(qa_pairs)} queries\n")
    
    results = []
    
    # Test configurations
    configs = [
        ("Standard Cosine (baseline)", lambda: StandardRAG()),
        ("CrossPolytope L1", lambda: CrossPolytopeRAG(volumetric=True)),
        ("Hybrid Radial (a=0.7)", lambda: HybridGeometricRAG(use_radial_encoding=True, alpha=0.7)),
        ("CrossPolytope+Radial (a=0.5)", lambda: CrossPolytopeHybridRAG(alpha=0.5)),
        ("CrossPolytope+Radial (a=0.7)", lambda: CrossPolytopeHybridRAG(alpha=0.7)),
        ("CrossPolytope+Radial (a=1.0)", lambda: CrossPolytopeHybridRAG(alpha=1.0)),
    ]
    
    for name, create_fn in configs:
        rag = create_fn()
        mrr, h1, h5 = evaluate(rag, name, documents, qa_pairs)
        results.append((name, mrr, h1, h5))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Sorted by MRR")
    print("=" * 70)
    print(f"{'Engine':<35} | {'MRR':>8} | {'H@1':>7} | {'H@5':>7}")
    print("-" * 70)
    
    for name, mrr, h1, h5 in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"{name:<35} | {mrr:>8.4f} | {h1:>6.1f}% | {h5:>6.1f}%")
    
    # Best improvement
    baseline = next(r for r in results if "baseline" in r[0])
    best = max(results, key=lambda x: x[1])
    improvement = (best[1] - baseline[1]) / baseline[1] * 100
    
    print(f"\nBest: {best[0]} with MRR={best[1]:.4f} (+{improvement:.1f}% vs baseline)")

