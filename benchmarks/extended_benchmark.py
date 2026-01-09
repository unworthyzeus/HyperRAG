"""
Extended Benchmark - Multiple Models, More Data

Tests our approach against multiple embedding models with larger dataset.
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("datasets library not available")

from HyperRAG import HybridGeometricRAG, CrossPolytopeHybridRAG
from HyperRAG import StandardRAG, CrossPolytopeRAG


@dataclass
class BenchmarkResult:
    model: str
    engine: str
    mrr: float
    hits_1: float
    hits_5: float
    hits_10: float
    num_docs: int
    num_queries: int
    ingest_time: float
    query_time: float


class STBaseline:
    """Sentence-transformers baseline using util.semantic_search"""
    
    def __init__(self, model_name: str):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.corpus_embeddings = None
        
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        self.corpus_embeddings = self.encoder.encode(
            documents, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
    
    def query(self, query_text: str, top_k: int = 10):
        query_embedding = self.encoder.encode(query_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]
        
        results = []
        for hit in hits:
            results.append(type('Result', (), {
                'document': self.documents[hit['corpus_id']],
                'score': float(hit['score']),
                'index': int(hit['corpus_id'])
            })())
        return results


def load_squad_extended(num_docs: int = 1000, num_queries: int = 500):
    """Load more data from SQuAD."""
    if not HAS_DATASETS:
        return None, None
    
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
        
        if len(documents) >= num_docs and len(qa_pairs) >= num_queries:
            break
    
    # Limit queries
    qa_pairs = qa_pairs[:num_queries]
    
    print(f"Loaded {len(documents)} documents, {len(qa_pairs)} queries")
    return documents, qa_pairs


def evaluate(rag, documents, qa_pairs) -> Dict:
    """Evaluate a RAG engine."""
    
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
    
    return {
        'mrr': rr_sum / n,
        'hits_1': hits_1 / n * 100,
        'hits_5': hits_5 / n * 100,
        'hits_10': hits_10 / n * 100,
        'ingest_time': ingest_time,
        'query_time': query_time,
    }


def run_extended_benchmark():
    """Run comprehensive benchmark."""
    
    # Load more data
    documents, qa_pairs = load_squad_extended(num_docs=1000, num_queries=500)
    
    if documents is None:
        print("Could not load data")
        return
    
    # Models to test
    models = [
        ('all-MiniLM-L6-v2', 384),      # Fast, good quality
        ('all-mpnet-base-v2', 768),      # Higher quality
        ('multi-qa-MiniLM-L6-cos-v1', 384),  # Trained on QA pairs
        # ('BAAI/bge-small-en-v1.5', 384),   # BGE model
    ]
    
    all_results = []
    
    for model_name, dim in models:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name} ({dim} dims)")
        print(f"{'#'*80}")
        
        # 1. ST Baseline (official implementation)
        print(f"\n[1/4] ST Baseline...")
        try:
            baseline = STBaseline(model_name)
            metrics = evaluate(baseline, documents, qa_pairs)
            result = BenchmarkResult(
                model=model_name,
                engine="ST Baseline",
                num_docs=len(documents),
                num_queries=len(qa_pairs),
                **metrics
            )
            all_results.append(result)
            print(f"  MRR: {metrics['mrr']:.4f}, H@5: {metrics['hits_5']:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 2. Our StandardRAG (should match baseline)
        print(f"\n[2/4] Our StandardRAG...")
        try:
            std = StandardRAG(model_name=model_name)
            metrics = evaluate(std, documents, qa_pairs)
            result = BenchmarkResult(
                model=model_name,
                engine="Our StandardRAG",
                num_docs=len(documents),
                num_queries=len(qa_pairs),
                **metrics
            )
            all_results.append(result)
            print(f"  MRR: {metrics['mrr']:.4f}, H@5: {metrics['hits_5']:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 3. CrossPolytope L1
        print(f"\n[3/4] CrossPolytope L1...")
        try:
            cross = CrossPolytopeRAG(model_name=model_name, volumetric=True)
            metrics = evaluate(cross, documents, qa_pairs)
            result = BenchmarkResult(
                model=model_name,
                engine="CrossPolytope L1",
                num_docs=len(documents),
                num_queries=len(qa_pairs),
                **metrics
            )
            all_results.append(result)
            print(f"  MRR: {metrics['mrr']:.4f}, H@5: {metrics['hits_5']:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 4. Hybrid Radial (our best)
        print(f"\n[4/4] Hybrid Radial (a=0.7)...")
        try:
            hybrid = HybridGeometricRAG(model_name=model_name, use_radial_encoding=True, alpha=0.7)
            metrics = evaluate(hybrid, documents, qa_pairs)
            result = BenchmarkResult(
                model=model_name,
                engine="Hybrid Radial (a=0.7)",
                num_docs=len(documents),
                num_queries=len(qa_pairs),
                **metrics
            )
            all_results.append(result)
            print(f"  MRR: {metrics['mrr']:.4f}, H@5: {metrics['hits_5']:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary Table
    print("\n" + "=" * 100)
    print("EXTENDED BENCHMARK RESULTS - 1000 docs, 500 queries")
    print("=" * 100)
    print(f"{'Model':<30} | {'Engine':<25} | {'MRR':>8} | {'H@1':>7} | {'H@5':>7} | {'H@10':>7}")
    print("-" * 100)
    
    # Group by model
    for model_name, _ in models:
        model_results = [r for r in all_results if r.model == model_name]
        for r in sorted(model_results, key=lambda x: x.mrr, reverse=True):
            short_model = model_name.split('/')[-1][:28]
            print(f"{short_model:<30} | {r.engine:<25} | {r.mrr:>8.4f} | {r.hits_1:>6.1f}% | {r.hits_5:>6.1f}% | {r.hits_10:>6.1f}%")
        print("-" * 100)
    
    # Overall comparison
    print("\n" + "=" * 100)
    print("IMPROVEMENT OVER BASELINE (by model)")
    print("=" * 100)
    
    for model_name, _ in models:
        model_results = {r.engine: r for r in all_results if r.model == model_name}
        baseline = model_results.get("ST Baseline")
        hybrid = model_results.get("Hybrid Radial (a=0.7)")
        
        if baseline and hybrid:
            improvement = (hybrid.mrr - baseline.mrr) / baseline.mrr * 100
            print(f"{model_name:<40}: Baseline MRR={baseline.mrr:.4f}, Hybrid MRR={hybrid.mrr:.4f}, Improvement={improvement:+.2f}%")
    
    # Save results
    output = [asdict(r) for r in all_results]
    with open('results/extended_benchmark.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to results/extended_benchmark.json")
    
    return all_results


if __name__ == "__main__":
    results = run_extended_benchmark()

