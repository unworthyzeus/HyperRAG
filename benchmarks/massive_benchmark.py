"""
MASSIVE Benchmark - Best Models, Maximum Data
GPU-optimized version with reduced batch sizes for 4GB VRAM

Tests our geometric approaches with:
- Top-tier embedding models (BGE, E5, mpnet)
- 3000 documents (reduced for memory)
- 800 queries
- 3 alpha values per model
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer, util
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("datasets library not available")
    exit(1)

from HyperRAG import HybridGeometricRAG
from HyperRAG import StandardRAG, CrossPolytopeRAG


@dataclass
class Result:
    model: str
    engine: str
    alpha: float
    mrr: float
    hits_1: float
    hits_5: float
    hits_10: float
    num_docs: int
    num_queries: int


class STBaseline:
    """Sentence-transformers baseline with configurable batch size"""
    
    def __init__(self, model_name: str, batch_size: int = 32):
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


class HybridRAGSmallBatch(HybridGeometricRAG):
    """HybridGeometricRAG with smaller batch size for GPU memory"""
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents (batch_size=8)...")
        
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=8)
        
        if self.use_radial:
            print("Encoding radial information (specificity)...")
            self.embeddings = self.radial_encoder.encode_specificity(raw_embeddings, documents)
        else:
            self.embeddings = raw_embeddings
        
        # Also keep cosine-ready version
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.cosine_embeddings = self.embeddings / np.clip(norms, 1e-10, None)
        
        final_norms = np.linalg.norm(self.embeddings, axis=1)
        print(f"Radial distribution: min={np.min(final_norms):.3f}, max={np.max(final_norms):.3f}, mean={np.mean(final_norms):.3f}")
        print(f"Ingested {len(documents)} documents.")


class CrossPolytopeSmallBatch(CrossPolytopeRAG):
    """CrossPolytopeRAG with smaller batch size"""
    
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents (batch_size=8)...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=8)
        
        if self.volumetric:
            print("Applying volumetric transform (L1 ball)...")
            self.embeddings = self.geometry.project_to_volume(raw_embeddings)
        else:
            self.embeddings = self.geometry.project_to_surface(raw_embeddings)
        
        print(f"Ingested {len(documents)} documents (cross-polytope, volumetric: {self.volumetric}).")


def load_large_data(num_docs: int = 3000, num_queries: int = 800):
    """Load data from SQuAD."""
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
    
    # Also load from train split for more data
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
    
    documents = documents[:num_docs]
    qa_pairs = qa_pairs[:num_queries]
    
    print(f"Loaded {len(documents)} documents, {len(qa_pairs)} queries")
    return documents, qa_pairs


def evaluate(rag, qa_pairs) -> Dict:
    """Evaluate a RAG engine."""
    hits_1 = hits_5 = hits_10 = 0
    rr_sum = 0
    
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
    
    n = len(qa_pairs)
    return {
        'mrr': rr_sum / n,
        'hits_1': hits_1 / n * 100,
        'hits_5': hits_5 / n * 100,
        'hits_10': hits_10 / n * 100,
    }


def run_massive_benchmark():
    """Run comprehensive benchmark with best models."""
    
    # Load data (reduced for GPU memory)
    documents, qa_pairs = load_large_data(num_docs=3000, num_queries=800)
    
    # Top embedding models with batch sizes for 4GB VRAM
    # Format: (model_name, dimension, batch_size)
    models = [
        ('all-mpnet-base-v2', 768, 16),               # Good model, medium batch
        ('BAAI/bge-base-en-v1.5', 768, 16),           # BGE base (smaller than large)
        ('intfloat/e5-base-v2', 768, 16),             # E5 base (smaller than large)
    ]
    
    # Alpha values to test
    alphas = [0.1, 0.2, 0.3]
    
    all_results = []
    
    for model_name, dim, batch_size in models:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name} ({dim} dims, batch_size={batch_size})")
        print(f"# Documents: {len(documents)}, Queries: {len(qa_pairs)}")
        print(f"{'#'*80}")
        
        # 1. Baseline
        print(f"\n[1/5] ST Baseline...")
        try:
            baseline = STBaseline(model_name, batch_size=batch_size)
            start = time.time()
            baseline.ingest(documents)
            ingest_time = time.time() - start
            print(f"  Ingest time: {ingest_time:.2f}s")
            
            start = time.time()
            metrics = evaluate(baseline, qa_pairs)
            query_time = time.time() - start
            print(f"  Query time: {query_time:.2f}s")
            print(f"  MRR: {metrics['mrr']:.4f}, H@1: {metrics['hits_1']:.1f}%, H@5: {metrics['hits_5']:.1f}%")
            
            all_results.append(Result(
                model=model_name,
                engine="ST Baseline",
                alpha=0,
                num_docs=len(documents),
                num_queries=len(qa_pairs),
                **metrics
            ))
            baseline_mrr = metrics['mrr']
            
            # Free memory
            del baseline
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"  Error: {e}")
            baseline_mrr = 0
            continue
        
        # 2. CrossPolytope L1
        print(f"\n[2/5] CrossPolytope L1...")
        try:
            cross = CrossPolytopeSmallBatch(model_name=model_name, volumetric=True)
            cross.ingest(documents)
            metrics = evaluate(cross, qa_pairs)
            print(f"  MRR: {metrics['mrr']:.4f}, H@1: {metrics['hits_1']:.1f}%, H@5: {metrics['hits_5']:.1f}%")
            print(f"  vs Baseline: {(metrics['mrr'] - baseline_mrr) / baseline_mrr * 100:+.2f}%")
            
            all_results.append(Result(
                model=model_name,
                engine="CrossPolytope L1",
                alpha=0,
                num_docs=len(documents),
                num_queries=len(qa_pairs),
                **metrics
            ))
            
            del cross
            gc.collect()
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # 3-5. Hybrid Radial with different alphas
        for i, alpha in enumerate(alphas):
            print(f"\n[{3+i}/5] Hybrid Radial (alpha={alpha})...")
            try:
                hybrid = HybridRAGSmallBatch(model_name=model_name, use_radial_encoding=True, alpha=alpha)
                hybrid.ingest(documents)
                metrics = evaluate(hybrid, qa_pairs)
                print(f"  MRR: {metrics['mrr']:.4f}, H@1: {metrics['hits_1']:.1f}%, H@5: {metrics['hits_5']:.1f}%")
                print(f"  vs Baseline: {(metrics['mrr'] - baseline_mrr) / baseline_mrr * 100:+.2f}%")
                
                all_results.append(Result(
                    model=model_name,
                    engine=f"Hybrid Radial (a={alpha})",
                    alpha=alpha,
                    num_docs=len(documents),
                    num_queries=len(qa_pairs),
                    **metrics
                ))
                
                del hybrid
                gc.collect()
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 110)
    print(f"MASSIVE BENCHMARK RESULTS - {len(documents)} docs, {len(qa_pairs)} queries")
    print("=" * 110)
    print(f"{'Model':<30} | {'Engine':<25} | {'MRR':>8} | {'H@1':>7} | {'H@5':>7} | {'H@10':>7}")
    print("-" * 110)
    
    # Group by model
    for model_name, _, _ in models:
        model_results = [r for r in all_results if r.model == model_name]
        baseline = next((r for r in model_results if r.engine == "ST Baseline"), None)
        
        for r in sorted(model_results, key=lambda x: x.mrr, reverse=True):
            short_model = model_name.split('/')[-1][:28]
            diff = ""
            if baseline and r.engine != "ST Baseline":
                diff_pct = (r.mrr - baseline.mrr) / baseline.mrr * 100
                diff = f" ({diff_pct:+.2f}%)"
            print(f"{short_model:<30} | {r.engine:<25} | {r.mrr:>8.4f} | {r.hits_1:>6.1f}% | {r.hits_5:>6.1f}% | {r.hits_10:>6.1f}%{diff}")
        print("-" * 110)
    
    # Best per model
    print("\n" + "=" * 110)
    print("BEST CONFIGURATION PER MODEL")
    print("=" * 110)
    
    for model_name, _, _ in models:
        model_results = [r for r in all_results if r.model == model_name]
        if not model_results:
            continue
            
        baseline = next((r for r in model_results if r.engine == "ST Baseline"), None)
        best = max(model_results, key=lambda x: x.mrr)
        
        if baseline:
            improvement = (best.mrr - baseline.mrr) / baseline.mrr * 100
            print(f"{model_name}:")
            print(f"  Baseline: MRR={baseline.mrr:.4f}")
            print(f"  Best: {best.engine} with MRR={best.mrr:.4f} ({improvement:+.2f}%)")
            print()
    
    # Save results
    output = [asdict(r) for r in all_results]
    with open('results/massive_benchmark.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to results/massive_benchmark.json")
    
    return all_results


if __name__ == "__main__":
    print("Starting MASSIVE benchmark (GPU-optimized)...")
    print("3000 docs, 800 queries, batch_size=8-16 for GPU memory.\n")
    results = run_massive_benchmark()

