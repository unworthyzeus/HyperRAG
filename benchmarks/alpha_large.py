"""
Alpha tuning for larger corpus (1000 docs)
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

try:
    from datasets import load_dataset
except ImportError:
    print("datasets not available")
    exit(1)

from HyperRAG import HybridGeometricRAG


class STBaseline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.corpus_embeddings = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        self.corpus_embeddings = self.encoder.encode(documents, convert_to_tensor=True, show_progress_bar=True)
    
    def query(self, query_text: str, top_k: int = 10):
        query_embedding = self.encoder.encode(query_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]
        return [type('R', (), {'document': self.documents[h['corpus_id']], 'score': h['score'], 'index': h['corpus_id']})() for h in hits]


def load_data():
    print("Loading SQuAD (1000 docs, 500 queries)...")
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
        
        if len(documents) >= 1000:
            break
    
    return documents, qa_pairs[:500]


def evaluate(rag, qa_pairs):
    rr_sum = 0
    for q, a in qa_pairs:
        results = rag.query(q, top_k=10)
        for i, r in enumerate(results):
            if r.document == a or a in r.document:
                rr_sum += 1.0 / (i + 1)
                break
    return rr_sum / len(qa_pairs)


if __name__ == "__main__":
    documents, qa_pairs = load_data()
    print(f"Testing with {len(documents)} docs, {len(qa_pairs)} queries\n")
    
    # Baseline
    print("Computing baseline...")
    baseline = STBaseline()
    baseline.ingest(documents)
    baseline_mrr = evaluate(baseline, qa_pairs)
    print(f"Baseline MRR: {baseline_mrr:.4f}\n")
    
    # Test many alpha values
    alphas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    print(f"{'Alpha':<10} | {'MRR':>10} | {'vs Baseline':>12}")
    print("-" * 40)
    
    best_alpha = 0
    best_mrr = 0
    
    for alpha in alphas:
        rag = HybridGeometricRAG(use_radial_encoding=True, alpha=alpha)
        rag.ingest(documents)
        mrr = evaluate(rag, qa_pairs)
        diff = (mrr - baseline_mrr) / baseline_mrr * 100
        print(f"{alpha:<10.2f} | {mrr:>10.4f} | {diff:>+11.2f}%")
        
        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} with MRR={best_mrr:.4f}")
    print(f"Baseline: {baseline_mrr:.4f}")

