# HyperRAG Techniques Explained

This document details the geometric embedding techniques used in HyperRAG and compares them to standard approaches.

## 1. Standard approach: Hypersphere (Cosine Similarity)

Most modern embedding models (like BERT, RoBERTa, E5, BGE) produce vectors that are normalized to have a length (L2 norm) of 1.
Geometrically, this means all data points sit on the surface of a **Hypersphere** (a high-dimensional sphere).

- **Metric**: Cosine Similarity (equivalent to dot product on normalized vectors).
- **Geometry**: Surface of an L2 unit ball.
- **Intuition**: We care about the *direction* of the vector, not its magnitude. "King" and "Queen" point in similar directions; "Apple" points elsewhere.
- **Limitation**: In very high dimensions (e.g., 1024d), the surface area is vast, and points can become indistinguishably far apart or crowded depending on the training. The interior volume is unused.

## 2. CrossPolytope L1 (The "Sparse" Geometry)

The **CrossPolytope** is the high-dimensional generalization of an octahedron. It is the unit ball for the **L1 norm** (Manhattan distance).

- **Transformation**:
    1.  Take the standard L2 vector.
    2.  Project it onto the surface or volume of the CrossPolytope.
    3.  This often involves normalizing by the sum of absolute values ($|x_1| + |x_2| + ... = 1$) instead of the square root of sum of squares.
- **Metric**: **Manhattan Distance** (L1). sum of absolute differences $\sum |x_i - y_i|$.
- **Why it usually works**: High-dimensional semantic spaces are often sparse (many dimensions are near zero). L1 distance is often more meaningful than L2 for sparse data because it doesn't penalize small differences as harshly (no squaring) and is more robust to outliers.
- **Why it failed on e5-large**: `e5-large` is a dense model highly optimized for Cosine. Forcing it into an L1 geometry might distort the carefully tuned angular relationships without adding enough value from sparsity.

## 3. Hybrid Radial (The "Information Density" Approach)

Standard models discard the **magnitude** (length) of the vector, placing everything on the sphere surface.  Hybrid Radial attempts to reclaim this lost dimension to encode **specificity** or **information density**.

- **Concept**:
    - Queries are often broad ("animals").
    - Documents are specific ("The physiological structure of the African Elephant...").
    - A specific document should ideally be "contained" within a broad query, or they should have a relationship defined by more than just angle.
- **Mechanism**:
    1.  **Radial Encoding**: We measure how "specific" a text is (e.g., entropy of the embedding, or distance to a centroid).
    2.  We scale the embedding vector length based on this specificity. Rare/specific items get longer (or shorter, depending on configuration) vectors.
    3.  **Hybrid Distance**: The final similarity score is a weighted sum of:
        - **Angular Distance** (Cosine/L1): "Are they about the same topic?"
        - **Radial Distance** (Difference in lengths): "Do they match in specificity level?"
- **Alpha ($\alpha$) Parameter**: Controls how much the radial difference matters.
    - Low $\alpha$ (0.1): Mostly cosine similarity, with a small nudge from radial info.
    - High $\alpha$ (0.7): Strong penalty if the specificity levels don't match.

## 4. Other Alternatives Tested

### Hyperbolic (Poincaré Ball)
- **Geometry**: Non-Euclidean geometry with negative curvature.
- **Intuition**: Space expands exponentially as you move away from the center. This is perfect for **trees** and **hierarchies** (e.g., WordNet, file systems).
- **Use Case**: If your data has a strong "is-a" taxonomy (Animal -> Mammal -> Dog).

### Hellinger Distance
- **Geometry**: Distance between probability distributions.
- **Transformation**: Treat the embedding vector (made non-negative) as a probability distribution of "concepts".
- **Use Case**: Comparing "mixtures" of topics.
