"""
HyperRAG v2 Benchmark Suite

Comprehensive benchmarking using REAL datasets from HuggingFace:
1. SQuAD - Reading comprehension questions
2. Natural Questions (NQ) - Google's open-domain QA dataset  
3. MS MARCO - Real web queries
4. TriviaQA - Trivia questions
5. HotpotQA - Multi-hop reasoning

This provides proper evaluation instead of synthetic data.
"""

import numpy as np
import time
import json
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: 'datasets' library not installed. Install with: pip install datasets")

from rag_engine_v2 import (
    StandardRAG,
    HypersphereRAG,
    CrossPolytopeRAG,
    HyperbolicRAG,
    HellingerRAG,
    create_rag_engine
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    engine_name: str
    dataset_name: str
    num_documents: int
    num_queries: int
    
    # Accuracy metrics
    hits_at_1: int
    hits_at_3: int  
    hits_at_5: int
    hits_at_10: int
    mrr: float  # Mean Reciprocal Rank
    
    # Timing
    ingest_time: float
    query_time: float
    avg_query_time: float


class DatasetLoader:
    """
    Loads and prepares datasets for RAG evaluation.
    """
    
    @staticmethod
    def load_squad(num_samples: int = 1000) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Load Stanford Question Answering Dataset.
        Returns: (documents, [(question, answer_doc), ...])
        """
        if not HAS_DATASETS:
            return DatasetLoader._generate_synthetic(num_samples)
        
        print(f"Loading SQuAD dataset (up to {num_samples} samples)...")
        try:
            dataset = load_dataset("squad", split="validation", trust_remote_code=True)
            
            documents = []
            qa_pairs = []
            
            seen_contexts = set()
            for item in dataset:
                if len(documents) >= num_samples:
                    break
                    
                context = item['context']
                question = item['question']
                
                # Use context as the document
                if context not in seen_contexts:
                    seen_contexts.add(context)
                    documents.append(context)
                    doc_idx = len(documents) - 1
                else:
                    doc_idx = list(seen_contexts).index(context)
                
                qa_pairs.append((question, documents[doc_idx]))
            
            print(f"Loaded {len(documents)} documents and {len(qa_pairs)} QA pairs from SQuAD")
            return documents, qa_pairs
            
        except Exception as e:
            print(f"Error loading SQuAD: {e}")
            return DatasetLoader._generate_synthetic(num_samples)
    
    @staticmethod
    def load_natural_questions(num_samples: int = 500) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Load Google Natural Questions dataset.
        """
        if not HAS_DATASETS:
            return DatasetLoader._generate_synthetic(num_samples)
        
        print(f"Loading Natural Questions dataset (up to {num_samples} samples)...")
        try:
            # Use the simplified version
            dataset = load_dataset("natural_questions", "default", split="validation", 
                                   trust_remote_code=True, streaming=True)
            
            documents = []
            qa_pairs = []
            
            for i, item in enumerate(dataset):
                if len(qa_pairs) >= num_samples:
                    break
                
                # Extract document text
                doc_text = item.get('document', {}).get('tokens', {}).get('token', [])
                if isinstance(doc_text, list):
                    doc_text = ' '.join(doc_text[:500])  # Limit length
                
                question = item.get('question', {}).get('text', '')
                
                if doc_text and question and len(doc_text) > 50:
                    documents.append(doc_text)
                    qa_pairs.append((question, doc_text))
            
            print(f"Loaded {len(documents)} documents from Natural Questions")
            return documents, qa_pairs
            
        except Exception as e:
            print(f"Error loading NQ (trying MS MARCO instead): {e}")
            return DatasetLoader.load_ms_marco(num_samples)
    
    @staticmethod
    def load_ms_marco(num_samples: int = 1000) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Load MS MARCO dataset - real web queries and passages.
        """
        if not HAS_DATASETS:
            return DatasetLoader._generate_synthetic(num_samples)
        
        print(f"Loading MS MARCO dataset (up to {num_samples} samples)...")
        try:
            dataset = load_dataset("ms_marco", "v1.1", split="validation", 
                                   trust_remote_code=True, streaming=True)
            
            documents = []
            qa_pairs = []
            
            for item in dataset:
                if len(qa_pairs) >= num_samples:
                    break
                
                query = item.get('query', '')
                passages = item.get('passages', {})
                
                if isinstance(passages, dict):
                    passage_texts = passages.get('passage_text', [])
                    is_selected = passages.get('is_selected', [])
                    
                    for text, selected in zip(passage_texts, is_selected):
                        if text and len(text) > 20:
                            documents.append(text)
                            if selected:
                                qa_pairs.append((query, text))
            
            print(f"Loaded {len(documents)} passages from MS MARCO")
            return documents, qa_pairs
            
        except Exception as e:
            print(f"Error loading MS MARCO: {e}")
            return DatasetLoader._generate_synthetic(num_samples)
    
    @staticmethod
    def load_triviaqa(num_samples: int = 500) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Load TriviaQA dataset.
        """
        if not HAS_DATASETS:
            return DatasetLoader._generate_synthetic(num_samples)
        
        print(f"Loading TriviaQA dataset (up to {num_samples} samples)...")
        try:
            dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation",
                                   trust_remote_code=True)
            
            documents = []
            qa_pairs = []
            
            for item in dataset:
                if len(qa_pairs) >= num_samples:
                    break
                
                question = item.get('question', '')
                answer = item.get('answer', {})
                answer_text = answer.get('value', '') if isinstance(answer, dict) else str(answer)
                
                # Create a pseudo-document from the answer
                if question and answer_text:
                    doc = f"The answer to '{question}' is: {answer_text}"
                    documents.append(doc)
                    qa_pairs.append((question, doc))
            
            print(f"Loaded {len(documents)} QA pairs from TriviaQA")
            return documents, qa_pairs
            
        except Exception as e:
            print(f"Error loading TriviaQA: {e}")
            return DatasetLoader._generate_synthetic(num_samples)
    
    @staticmethod
    def load_hotpotqa(num_samples: int = 500) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Load HotpotQA - multi-hop reasoning dataset.
        """
        if not HAS_DATASETS:
            return DatasetLoader._generate_synthetic(num_samples)
        
        print(f"Loading HotpotQA dataset (up to {num_samples} samples)...")
        try:
            dataset = load_dataset("hotpot_qa", "fullwiki", split="validation",
                                   trust_remote_code=True, streaming=True)
            
            documents = []
            qa_pairs = []
            doc_to_idx = {}
            
            for item in dataset:
                if len(qa_pairs) >= num_samples:
                    break
                
                question = item.get('question', '')
                context = item.get('context', {})
                
                if isinstance(context, dict):
                    titles = context.get('title', [])
                    sentences = context.get('sentences', [])
                    
                    for title, sents in zip(titles, sentences):
                        if sents:
                            doc = f"{title}: " + ' '.join(sents)
                            if doc not in doc_to_idx:
                                doc_to_idx[doc] = len(documents)
                                documents.append(doc)
                
                # Use the first supporting fact as the target
                supporting_facts = item.get('supporting_facts', {})
                if isinstance(supporting_facts, dict):
                    sf_titles = supporting_facts.get('title', [])
                    if sf_titles:
                        target_title = sf_titles[0]
                        # Find doc with this title
                        for doc in documents:
                            if doc.startswith(f"{target_title}:"):
                                qa_pairs.append((question, doc))
                                break
            
            print(f"Loaded {len(documents)} contexts from HotpotQA")
            return documents, qa_pairs
            
        except Exception as e:
            print(f"Error loading HotpotQA: {e}")
            return DatasetLoader._generate_synthetic(num_samples)
    
    @staticmethod
    def _generate_synthetic(num_samples: int) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Fallback: Generate synthetic data if datasets can't be loaded.
        """
        import random
        import string
        
        print(f"Generating {num_samples} synthetic documents (fallback mode)...")
        
        topics = [
            "quantum physics", "machine learning", "ancient history", "molecular biology",
            "astrophysics", "organic chemistry", "cognitive psychology", "computer architecture",
            "evolutionary biology", "number theory", "linguistics", "climate science",
            "neuroscience", "game theory", "cryptography", "materials science"
        ]
        
        facts = [
            "The principle of superposition allows quantum states to exist simultaneously.",
            "Neural networks learn hierarchical representations through backpropagation.",
            "The Great Pyramid of Giza was built around 2560 BCE.",
            "DNA replication is semi-conservative and requires multiple enzymes.",
            "Black holes emit Hawking radiation due to quantum effects near the event horizon.",
            "Enzymes lower activation energy to catalyze biochemical reactions.",
            "Working memory has a capacity of approximately 4-7 chunks of information.",
            "Modern CPUs use pipelining and branch prediction for efficiency.",
            "Natural selection drives adaptation through differential reproductive success.",
            "Prime numbers are infinitely numerous as proven by Euclid.",
            "Chomsky proposed that humans have an innate language acquisition device.",
            "The greenhouse effect traps infrared radiation in Earth's atmosphere.",
            "Neurons communicate through electrical and chemical synaptic transmission.",
            "Nash equilibrium represents stable strategies in non-cooperative games.",
            "RSA encryption relies on the difficulty of factoring large primes.",
            "Graphene is a single layer of carbon atoms in a hexagonal lattice."
        ]
        
        documents = []
        qa_pairs = []
        
        for i, fact in enumerate(facts[:num_samples]):
            topic = topics[i % len(topics)]
            doc = f"In the field of {topic}: {fact} This fundamental concept has been extensively studied by researchers worldwide and has important applications in various domains. Further investigation reveals complex interactions with other phenomena."
            documents.append(doc)
            
            # Generate a question-like query
            question = f"What is an important principle in {topic}?"
            qa_pairs.append((question, doc))
        
        # Add some noise documents
        for _ in range(min(500, num_samples)):
            noise_topic = random.choice(topics)
            noise = ''.join(random.choices(string.ascii_lowercase + ' ', k=100))
            doc = f"General notes on {noise_topic}: {noise}."
            documents.append(doc)
        
        return documents, qa_pairs


class RAGBenchmark:
    """
    Comprehensive benchmark suite for comparing RAG engines.
    """
    
    def __init__(self, dataset_name: str = 'squad', num_samples: int = 500):
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.documents: List[str] = []
        self.qa_pairs: List[Tuple[str, str]] = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the specified dataset."""
        loaders = {
            'squad': DatasetLoader.load_squad,
            'nq': DatasetLoader.load_natural_questions,
            'msmarco': DatasetLoader.load_ms_marco,
            'triviaqa': DatasetLoader.load_triviaqa,
            'hotpotqa': DatasetLoader.load_hotpotqa,
            'synthetic': DatasetLoader._generate_synthetic
        }
        
        loader = loaders.get(self.dataset_name, DatasetLoader._generate_synthetic)
        self.documents, self.qa_pairs = loader(self.num_samples)
        
        if not self.qa_pairs:
            print("WARNING: No QA pairs loaded, generating synthetic fallback.")
            self.documents, self.qa_pairs = DatasetLoader._generate_synthetic(100)
    
    def evaluate_engine(
        self, 
        engine,
        engine_name: str,
        top_k: int = 10
    ) -> BenchmarkResult:
        """
        Evaluate a single RAG engine.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {engine_name}")
        print(f"{'='*60}")
        
        # Ingest
        start = time.time()
        engine.ingest(self.documents)
        ingest_time = time.time() - start
        
        # Query
        hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        reciprocal_ranks = []
        
        start = time.time()
        for question, answer_doc in self.qa_pairs:
            results = engine.query(question, top_k=top_k)
            
            # Find rank of correct answer
            rank = None
            for i, r in enumerate(results):
                # Check if answer doc is in the result
                if r.document == answer_doc or answer_doc in r.document or r.document in answer_doc:
                    rank = i + 1
                    break
            
            # Update hits@k
            if rank is not None:
                for k in hits_at_k.keys():
                    if rank <= k:
                        hits_at_k[k] += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        query_time = time.time() - start
        
        # Calculate MRR
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        result = BenchmarkResult(
            engine_name=engine_name,
            dataset_name=self.dataset_name,
            num_documents=len(self.documents),
            num_queries=len(self.qa_pairs),
            hits_at_1=hits_at_k[1],
            hits_at_3=hits_at_k[3],
            hits_at_5=hits_at_k[5],
            hits_at_10=hits_at_k[10],
            mrr=mrr,
            ingest_time=ingest_time,
            query_time=query_time,
            avg_query_time=query_time / len(self.qa_pairs) if self.qa_pairs else 0
        )
        
        # Print summary
        self._print_result(result)
        
        return result
    
    def _print_result(self, r: BenchmarkResult):
        """Print formatted result."""
        print(f"\n{r.engine_name} Results:")
        print(f"  Documents: {r.num_documents}, Queries: {r.num_queries}")
        print(f"  Hits@1:  {r.hits_at_1}/{r.num_queries} ({100*r.hits_at_1/r.num_queries:.1f}%)")
        print(f"  Hits@3:  {r.hits_at_3}/{r.num_queries} ({100*r.hits_at_3/r.num_queries:.1f}%)")
        print(f"  Hits@5:  {r.hits_at_5}/{r.num_queries} ({100*r.hits_at_5/r.num_queries:.1f}%)")
        print(f"  Hits@10: {r.hits_at_10}/{r.num_queries} ({100*r.hits_at_10/r.num_queries:.1f}%)")
        print(f"  MRR: {r.mrr:.4f}")
        print(f"  Ingest: {r.ingest_time:.2f}s, Query: {r.query_time:.2f}s")
    
    def run_all(self) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark on all RAG engine variants.
        """
        results = {}
        
        # Define all engine configurations to test
        configs = [
            ('Standard (Cosine)', lambda: StandardRAG()),
            ('Hypersphere (Euclidean)', lambda: HypersphereRAG(metric='euclidean', volumetric=True)),
            ('Hypersphere (Angular)', lambda: HypersphereRAG(metric='angular', volumetric=True)),
            ('Hypersphere (Cosine-Vol)', lambda: HypersphereRAG(metric='cosine', volumetric=True)),
            ('Hypersphere (Hellinger)', lambda: HypersphereRAG(metric='hellinger', volumetric=True)),
            ('Hypersphere (No Vol)', lambda: HypersphereRAG(metric='euclidean', volumetric=False)),
            ('CrossPolytope (L1)', lambda: CrossPolytopeRAG(volumetric=True)),
            ('Hyperbolic (Poincaré)', lambda: HyperbolicRAG()),
            ('Hellinger Direct', lambda: HellingerRAG()),
        ]
        
        for name, engine_fn in configs:
            try:
                engine = engine_fn()
                results[name] = self.evaluate_engine(engine, name)
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
        
        return results


def run_comprehensive_benchmark(
    datasets: List[str] = ['squad', 'synthetic'],
    samples_per_dataset: int = 500
):
    """
    Run comprehensive benchmark across multiple datasets.
    """
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*70}")
        
        benchmark = RAGBenchmark(dataset_name=dataset, num_samples=samples_per_dataset)
        results = benchmark.run_all()
        all_results[dataset] = results
    
    # Summary comparison
    print("\n" + "=" * 90)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 90)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"{'Engine':<30} | {'Hits@1':>8} | {'Hits@5':>8} | {'MRR':>8} | {'Time':>8}")
        print("-" * 70)
        
        # Sort by MRR
        sorted_results = sorted(results.items(), key=lambda x: x[1].mrr, reverse=True)
        
        for name, r in sorted_results:
            h1_pct = 100 * r.hits_at_1 / r.num_queries
            h5_pct = 100 * r.hits_at_5 / r.num_queries
            print(f"{name:<30} | {h1_pct:>7.1f}% | {h5_pct:>7.1f}% | {r.mrr:>8.4f} | {r.query_time:>6.2f}s")
    
    # Save results
    output_file = 'benchmark_results_v2.json'
    with open(output_file, 'w') as f:
        # Convert dataclasses to dicts
        serializable = {
            ds: {name: asdict(r) for name, r in results.items()}
            for ds, results in all_results.items()
        }
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    # Check for datasets library
    if not HAS_DATASETS:
        print("\n" + "=" * 60)
        print("To use real datasets, install the datasets library:")
        print("  pip install datasets")
        print("Falling back to synthetic data for now...")
        print("=" * 60 + "\n")
    
    # Run benchmark
    results = run_comprehensive_benchmark(
        datasets=['squad', 'synthetic'],  # Add 'msmarco', 'triviaqa' etc if available
        samples_per_dataset=300  # Adjust based on your hardware
    )
