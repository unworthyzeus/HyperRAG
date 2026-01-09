# Large Model Benchmark Analysis (GPU Optimized)

This report summarizes the performance of 1024-dimension embedding models (`BAAI/bge-large-en-v1.5` and `intfloat/e5-large-v2`) under memory-constrained conditions (batch size = 4).

## Executive Summary

- **Best Overall Model:** `intfloat/e5-large-v2` (Baseline MRR: **0.7923**)
- **HyperRAG Impact:** 
    - **bge-large-en-v1.5:** Small improvement with Hybrid Radial (a=0.1) -> **0.7577** (+0.40% over baseline).
    - **e5-large-v2:** No improvement over the strong baseline. Hybrid Radial (a=0.1) was statistically tied (**0.7918** vs 0.7923), while other geometric transforms degraded performance slightly.
- **Hardware Viability:** Successfully ran 1024-dim models on standard consumer hardware by reducing batch size to 4. Ingest time increased (~99s for 3000 docs) but remained practical.

## Detailed Results

### 1. BAAI/bge-large-en-v1.5 (1024 dims)

| Approach | MRR | Hits@5 | vs Baseline |
|----------|-----|--------|-------------|
| **ST Baseline** | **0.7547** | **92.5%** | **-** |
| Hybrid Radial (a=0.1) | 0.7577 | 92.8% | +0.40% |
| Hybrid Radial (a=0.2) | 0.7549 | 92.6% | +0.03% |
| CrossPolytope L1 | 0.7452 | 91.8% | -1.26% |

> **Insight:** `bge-large` benefits slightly from the Hybrid Radial approach with a low alpha (0.1), suggesting some useful radial signal exists but is subtle.

### 2. intfloat/e5-large-v2 (1024 dims)

| Approach | MRR | Hits@5 | vs Baseline |
|----------|-----|--------|-------------|
| **ST Baseline** | **0.7923** | **94.6%** | **-** |
| Hybrid Radial (a=0.1) | 0.7918 | 94.5% | -0.06% |
| CrossPolytope L1 | 0.7752 | - | -2.16% |

> **Insight:** `e5-large-v2` is an exceptionally strong baseline. Its embedding space appears highly optimized for cosine similarity, leaving little room for geometric re-interpretation to add value without re-training. The L1 transformation (`CrossPolytope`) actively hurts performance here (-2.16%), significantly more than with other models.

## Conclusion

For constrained hardware (4GB VRAM):
1.  **If you want maximum accuracy:** Use `intfloat/e5-large-v2` with the standard baseline. It outperforms everything else, including `bge-large` + HyperRAG.
2.  **If you are using `bge-large`:** Use `Hybrid Radial (a=0.1)` for a free 0.4% boost.
3.  **CrossPolytope L1** seems less effective on these high-dimensional (1024d) models compared to the 384d/768d models, possibly due to the "curse of dimensionality" affecting L1/L2 ratios differently or the specific pre-training of these large models.
