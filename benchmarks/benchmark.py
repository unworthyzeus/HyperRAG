"""
HyperRAG v3 Comprehensive Benchmark

Tests all engines from v2 and v3 on real datasets.
Focused on finding the BEST geometric approach for RAG.
"""

import numpy as np
import time
import json
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

from .engines import (
    StandardRAG,
    HypersphereRAG,
    CrossPolytopeRAG,
    HyperbolicRAG,
    HellingerRAG,
)

from .advanced import (
    HybridGeometricRAG,
    EnsembleRAG,
    ContrastiveVolumetricRAG,
)


@dataclass 
class BenchmarkResult:
    engine_name: str
    dataset_name: str
    num_documents: int
    num_queries: int
    hits_at_1: int
    hits_at_3: int  
    hits_at_5: int
    hits_at_10: int
    mrr: float
    ingest_time: float
    query_time: float


def load_squad_v2(num_samples: int = 500) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Load SQuAD dataset."""
    if not HAS_DATASETS:
        return generate_fallback_data(num_samples)
    
    print(f"Loading SQuAD (up to {num_samples} contexts)...")
    try:
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
        
        print(f"Loaded {len(documents)} contexts, {len(qa_pairs)} QA pairs")
        return documents, qa_pairs
        
    except Exception as e:
        print(f"Error: {e}")
        return generate_fallback_data(num_samples)


def generate_fallback_data(num_samples: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Generate synthetic data as fallback."""
    import random
    
    domains = [
        ("mathematics", [
            "The Pythagorean theorem states that a² + b² = c² for right triangles.",
            "Euler's identity e^(iπ) + 1 = 0 connects five fundamental constants.",
            "The derivative measures the instantaneous rate of change of a function.",
            "Integration is the reverse process of differentiation in calculus.",
            "Prime numbers have exactly two distinct divisors: 1 and themselves.",
        ]),
        ("physics", [
            "Einstein's mass-energy equivalence E=mc² revolutionized physics.",
            "Quantum mechanics describes particles as both waves and particles.",
            "Newton's laws describe the relationship between force and motion.",
            "Entropy always increases in isolated systems according to thermodynamics.",
            "The speed of light is approximately 299,792,458 meters per second.",
        ]),
        ("biology", [
            "DNA contains the genetic instructions for all living organisms.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "Mitochondria are the powerhouses of the cell, producing ATP.",
            "Evolution through natural selection was proposed by Darwin.",
            "The human genome contains approximately 3 billion base pairs.",
        ]),
        ("computer science", [
            "Machine learning algorithms learn patterns from training data.",
            "Neural networks are inspired by biological brain structures.",
            "Recursion is a technique where a function calls itself.",
            "Big O notation describes algorithm time complexity.",
            "Data structures organize and store data efficiently.",
        ]),
        ("geometry", [
            "A hyperdodecahedron is a 4-dimensional polytope with 120 cells.",
            "The 120-cell has 600 vertices and 1200 edges in 4D space.",
            "Platonic solids are the only regular convex polyhedra in 3D.",
            "Non-Euclidean geometry includes hyperbolic and elliptic spaces.",
            "Fractals exhibit self-similarity at different scales.",
        ]),
    ]
    
    documents = []
    qa_pairs = []
    
    for domain, facts in domains:
        for fact in facts:
            doc = f"In {domain}: {fact} This fundamental concept is widely studied."
            documents.append(doc)
            question = f"What is a key concept in {domain}?"
            qa_pairs.append((question, doc))
    
    # Add noise
    for i in range(num_samples - len(documents)):
        domain = random.choice([d[0] for d in domains])
        doc = f"Additional notes on {domain}: " + " ".join(["word"] * 20)
        documents.append(doc)
    
    print(f"Generated {len(documents)} synthetic documents")
    return documents, qa_pairs


def evaluate_engine(engine, documents: List[str], qa_pairs: List[Tuple[str, str]], 
                   engine_name: str, top_k: int = 10) -> BenchmarkResult:
    """Evaluate a single engine."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {engine_name}")
    print(f"{'='*50}")
    
    # Ingest
    start = time.time()
    engine.ingest(documents)
    ingest_time = time.time() - start
    
    # Query
    hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    
    start = time.time()
    for question, answer_doc in qa_pairs:
        results = engine.query(question, top_k=top_k)
        
        rank = None
        for i, r in enumerate(results):
            if r.document == answer_doc or answer_doc in r.document:
                rank = i + 1
                break
        
        if rank:
            for k in hits_at_k.keys():
                if rank <= k:
                    hits_at_k[k] += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    query_time = time.time() - start
    mrr = np.mean(reciprocal_ranks)
    
    result = BenchmarkResult(
        engine_name=engine_name,
        dataset_name="squad",
        num_documents=len(documents),
        num_queries=len(qa_pairs),
        hits_at_1=hits_at_k[1],
        hits_at_3=hits_at_k[3],
        hits_at_5=hits_at_k[5],
        hits_at_10=hits_at_k[10],
        mrr=mrr,
        ingest_time=ingest_time,
        query_time=query_time
    )
    
    print(f"Results: Hits@1={result.hits_at_1}, Hits@5={result.hits_at_5}, MRR={result.mrr:.4f}")
    return result


def run_full_benchmark(num_samples: int = 300):
    """Run comprehensive benchmark on all engine variants."""
    
    # Load data
    documents, qa_pairs = load_squad_v2(num_samples)
    
    # Limit queries for faster testing
    qa_pairs = qa_pairs[:min(200, len(qa_pairs))]
    
    print(f"\nBenchmarking with {len(documents)} documents, {len(qa_pairs)} queries")
    
    # Define all engines
    engine_configs = [
        # v2 engines
        ("1. Standard Cosine", lambda: StandardRAG()),
        ("2. Hypersphere Euclidean", lambda: HypersphereRAG(metric='euclidean', volumetric=True)),
        ("3. Hypersphere Angular", lambda: HypersphereRAG(metric='angular', volumetric=True)),
        ("4. Hypersphere Hellinger", lambda: HypersphereRAG(metric='hellinger', volumetric=True)),
        ("5. CrossPolytope L1", lambda: CrossPolytopeRAG(volumetric=True)),
        ("6. Hyperbolic Poincare", lambda: HyperbolicRAG()),
        ("7. Hellinger Direct", lambda: HellingerRAG()),
        
        # v3 engines  
        ("8. Hybrid Radial", lambda: HybridGeometricRAG(use_radial_encoding=True, alpha=0.1)),
        ("9. Hybrid Radial (a=0.2)", lambda: HybridGeometricRAG(use_radial_encoding=True, alpha=0.2)),
        ("10. Ensemble CEHL", lambda: EnsembleRAG(strategies=['cosine', 'euclidean', 'hybrid', 'l1'])),
        ("11. Ensemble CE", lambda: EnsembleRAG(strategies=['cosine', 'euclidean'])),
        ("12. Contrastive Vol", lambda: ContrastiveVolumetricRAG()),
    ]
    
    results = {}
    
    for name, create_engine in engine_configs:
        try:
            engine = create_engine()
            result = evaluate_engine(engine, documents, qa_pairs, name)
            results[name] = result
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Print summary table
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY - Sorted by MRR")
    print("=" * 90)
    print(f"{'Engine':<30} | {'Hits@1':>8} | {'Hits@5':>8} | {'MRR':>8} | {'Time':>8}")
    print("-" * 90)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].mrr, reverse=True)
    
    for name, r in sorted_results:
        h1_pct = 100 * r.hits_at_1 / r.num_queries
        h5_pct = 100 * r.hits_at_5 / r.num_queries
        print(f"{name:<30} | {h1_pct:>7.1f}% | {h5_pct:>7.1f}% | {r.mrr:>8.4f} | {r.query_time:>6.2f}s")
    
    # Save results
    output = {name: asdict(r) for name, r in results.items()}
    with open('benchmark_v3_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to benchmark_v3_results.json")
    
    # Analysis
    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)
    
    best = sorted_results[0]
    baseline = results.get("1. Standard Cosine")
    
    if baseline:
        improvement = (best[1].mrr - baseline.mrr) / baseline.mrr * 100
        print(f"Best engine: {best[0]} with MRR={best[1].mrr:.4f}")
        print(f"Baseline (Standard Cosine): MRR={baseline.mrr:.4f}")
        print(f"Improvement: {improvement:+.2f}%")
    
    return results


if __name__ == "__main__":
    results = run_full_benchmark(num_samples=300)

