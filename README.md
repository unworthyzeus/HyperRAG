# HyperRAG v3 - Geometric RAG Optimization

## Project Goal
Investigate whether alternative geometric structures (beyond the standard hypersphere with cosine similarity) can improve RAG retrieval accuracy, specifically targeting "Semantic Collapse" in large document sets.

## Key Results (on SQuAD dataset)

| Rank | Approach | MRR | Hits@5 | vs Baseline |
|------|----------|-----|--------|-------------|
| **1** | **Hybrid Radial (a=0.7)** | **0.6722** | **90.5%** | **+4.9%** |
| 2 | CrossPolytope+Radial | 0.6649 | 89.5% | +3.8% |
| 3 | CrossPolytope L1 | 0.6606 | 88.5% | +3.1% |
| 4 | *ST mpnet-base (larger model)* | *0.6541* | *88.0%* | *+2.1%* |
| 5 | **sentence-transformers baseline** | **0.6408** | **88.0%** | **baseline** |

### Large Model Results (1024 dims - e5-large-v2)

| Engine | MRR | H@5 | Notes |
|--------|-----|-----|-------|
| **ST Baseline (e5-large)** | **0.7923** | **94.6%** | Best absolute performance |
| Hybrid Radial (a=0.1) | 0.7918 | 94.5% | Statistically tied |
| Whitened RAG (ZCA) | 0.7134 | 86.5% | **-10%** (Failed) |
| ClusterTree RAG | 0.6930 | 82.1% | **-12.5%** (Failed) |

**Key insight**: On highly optimized large models (e5-large), standard cosine similarity is extremely hard to beat. The embedding space is already "perfectly" shaped for cosine. Geometric interventions like **Whitening** or **Clustering** destroy this learned structure.

## Semantic Collapse & Solutions

**"Semantic Collapse"** is the phenomenon where RAG precision drops as the corpus grows. Our 15k-document study confirmed:
1.  **Scaling Gain**: With `e5-large-v2`, the improvement of **Hybrid Radial** over the baseline grows with the corpus (**+0.0031 MRR** at 15k docs vs +0.0024 at 1k).
2.  **Dimension Threshold**: Radial encoding requires models with **>=768 dimensions**. In tiny models (128D - bert-tiny), it introduces noise (-5% MRR).
3.  **Alpha Scaling**: The $\alpha$ parameter must scale inversely with corpus size (e.g., $\alpha=0.05$ for large scale).

**What DOESN'T Work (The "Negative Results" Hall of Shame):**
- **Whitening (ZCA)**: -10% MRR.
- **Clustering**: -12.5% MRR.
- **Hellinger**: -19% MRR.
- **Applying to <512D models**: -1.6% to -5% MRR.

## Recommended Configurations

### 1. Maximum Performance (Unlimited Resources)
Use **`intfloat/e5-large-v2`** with Standard Cosine Similarity.
```python
rag = StandardRAG(model_name='intfloat/e5-large-v2')
```

### 2. High Performance (Constrained/Standard)
Use **`HybridGeometricRAG`** with `all-mpnet-base-v2`.
```python
# Use small alpha (0.1 - 0.2) for larger corpora
rag = HybridGeometricRAG(model_name='all-mpnet-base-v2', use_radial_encoding=True, alpha=0.2)
```

### 3. Efficiency / Sparse Data
Use **`CrossPolytopeRAG`** (L1 distance).
```python
rag = CrossPolytopeRAG(volumetric=True)
```

## Architecture

```
HyperRAG/
├── README.md               # Quick start and results
├── docs/                   # Detailed studies and theory
│   ├── RESEARCH.md         # Semantic Collapse & Prior Art
│   ├── techniques_explained.md # Geometric theory deep-dive
│   └── large_model_analysis.md
├── src/                    # Source code
│   └── HyperRAG/           # Main package
│       ├── core/           # Geometry and Bases
│       ├── advanced/       # Hybrid and Radial strategies
│       └── experimental/    # Research trials (Whitening, etc.)
├── benchmarks/             # Study scripts
│   ├── hyperscale_benchmark.py # Semantic Collapse study
│   └── ...
├── results/                # Collected data
│   ├── logs/               # Raw execution outputs
│   ├── reports/            # JSON summaries
│   └── plots/              # Visualizations
└── scripts/                # Helper tools
```

## Future Work

1.  **Learned Radial Projections**: Instead of using heuristics (word count), train a small MLP to predict the optimal "radius" for a document based on its content.
2.  **Query-dependent geometry**: Switch metrics based on whether the query appears to be "lookup" (L1) or "thematic" (Cosine).
3.  **Test on 100k+ Documents**: Validate if "Semantic Collapse" is truly mitigated at massive scale.

## References

- Isoperimetric inequality: Sphere maximizes V/S ratio
- Cross-polytope: Dual of hypercube, L1 unit ball
- Poincaré ball: Hyperbolic geometry for hierarchical data
- **Semantic Collapse**: Degradation of retrieval at scale due to embedding crowding.
