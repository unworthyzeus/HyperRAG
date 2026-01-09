"""
Benchmark Alternatives - Testing Exotic Geometries on e5-large-v2
"""

import numpy as np
import time
import sys
import gc
sys.path.insert(0, 'src')

from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
from datasets import load_dataset

from HyperRAG import HyperbolicRAG, HellingerRAG, HypersphereRAG, DistanceMetrics

# --- Tiny Batch Wrappers for 4GB VRAM ---

class HyperbolicTinyBatch(HyperbolicRAG):
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents (Hyperbolic, batch_size=4)...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=4)
        print("Projecting to Poincaré ball...")
        self.embeddings = self.geometry.project_to_ball(raw_embeddings)
        print("Ingested.")

class HellingerTinyBatch(HellingerRAG):
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents (Hellinger, batch_size=4)...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=4)
        print("Converting to probability distributions...")
        self.embeddings = DistanceMetrics.to_probability_distribution(raw_embeddings)
        print("Ingested.")

class HypersphereEuclideanTinyBatch(HypersphereRAG):
    def ingest(self, documents: List[str]) -> None:
        self.documents = documents
        print(f"Encoding {len(documents)} documents (Euclidean+Volumetric, batch_size=4)...")
        raw_embeddings = self.encoder.encode(documents, show_progress_bar=True, batch_size=4)
        if self.volumetric:
            print("Applying volumetric transform...")
            self.embeddings = self.geometry.project_to_volume(raw_embeddings)
        else:
            self.embeddings = self.geometry.project_to_surface(raw_embeddings)
        print("Ingested.")

# --- Evaluation Logic ---

def load_data(num_docs=3000, num_queries=800):
    print(f"Loading SQuAD (target: {num_docs} docs)...")
    dataset = load_dataset("squad", split="validation")
    documents = []
    qa_pairs = []
    seen = set()
    for item in dataset:
        if item['context'] not in seen:
            seen.add(item['context'])
            documents.append(item['context'])
        qa_pairs.append((item['question'], item['context']))
    
    # Fill from train if needed
    if len(documents) < num_docs:
        dataset_train = load_dataset("squad", split="train")
        for item in dataset_train:
            if item['context'] not in seen:
                seen.add(item['context'])
                documents.append(item['context'])
            qa_pairs.append((item['question'], item['context']))
            if len(documents) >= num_docs: break
            
    return documents[:num_docs], qa_pairs[:num_queries]

def evaluate(rag, qa_pairs) -> Dict:
    hits_1 = hits_5 = 0
    rr_sum = 0
    n = len(qa_pairs)
    
    for question, answer_doc in qa_pairs:
        results = rag.query(question, top_k=5)
        for i, r in enumerate(results):
            doc = r.document
            if doc == answer_doc or answer_doc in doc:
                if i == 0: hits_1 += 1
                if i < 5: hits_5 += 1
                rr_sum += 1.0 / (i + 1)
                break
    return {'mrr': rr_sum/n, 'hits_1': hits_1/n*100, 'hits_5': hits_5/n*100}

def cleanup():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

# --- Main ---

def run_benchmark():
    docs, qa = load_data()
    model_name = 'intfloat/e5-large-v2'
    
    print(f"\nTargeting Model: {model_name}")
    print("Baseline MRR from previous run: ~0.7923")
    
    results = []
    
    # 1. Hyperbolic (Poincare)
    print("\n>>> Testing Hyperbolic (Poincare Ball)...")
    try:
        engine = HyperbolicTinyBatch(model_name=model_name)
        start = time.time()
        engine.ingest(docs)
        metrics = evaluate(engine, qa)
        print(f"Hyperbolic Reuslts: {metrics}")
        results.append(("Hyperbolic", metrics))
        del engine
        cleanup()
    except Exception as e:
        print(f"Hyperbolic Failed: {e}")

    # 2. Hellinger
    print("\n>>> Testing Hellinger Distance...")
    try:
        engine = HellingerTinyBatch(model_name=model_name)
        engine.ingest(docs)
        metrics = evaluate(engine, qa)
        print(f"Hellinger Results: {metrics}")
        results.append(("Hellinger", metrics))
        del engine
        cleanup()
    except Exception as e:
        print(f"Hellinger Failed: {e}")

    # 3. Hypersphere Euclidean + Volumetric
    print("\n>>> Testing Hypersphere (Euclidean + Volumetric)...")
    try:
        # Note: StandardRAG uses Cosine on Surface. 
        # This uses Euclidean Distance inside the Volume (L2 ball interior).
        engine = HypersphereEuclideanTinyBatch(model_name=model_name, metric='euclidean', volumetric=True)
        engine.ingest(docs)
        metrics = evaluate(engine, qa)
        print(f"Hypersphere Volumetric Results: {metrics}")
        results.append(("Hypersphere (Euclidean+Vol)", metrics))
        del engine
        cleanup()
    except Exception as e:
        print(f"Hypersphere Vol Failed: {e}")

    print("\n--- Summary ---")
    for name, m in results:
        print(f"{name}: MRR={m['mrr']:.4f}, H@5={m['hits_5']:.1f}%")

if __name__ == "__main__":
    run_benchmark()

