# Neural Visual Search Engine

An image retrieval system that finds visually similar images using deep learning feature extraction and nearest neighbor search. Benchmarks **5 CNN architectures** and **6 search algorithms** on Caltech-101 (8,677 images, 101 categories) and Caltech-256, achieving up to **96% retrieval accuracy** with fine-tuned features and **microsecond-scale** query times via PCA-compressed nearest neighbor search.

## Results

### Retrieval Accuracy (ResNet-50 Features)

| Dataset | Algorithm | Pretrained | Fine-Tuned |
|---------|-----------|-----------|-----------|
| Caltech-101 | Brute Force | 87.06% | 89.48% |
| Caltech-101 | PCA + Brute Force | 87.65% | 89.39% |
| Caltech-256 | Brute Force | 58.38% | **96.01%** |
| Caltech-256 | PCA + Brute Force | 56.64% | 95.34% |

Fine-tuning on Caltech-256 yields a **+37.6% accuracy jump** (58.38% → 96.01%), demonstrating the impact of domain adaptation on retrieval quality.

### Fine-Tuning Training Progress (ResNet-50 on Caltech-101)

| Epoch | Loss | Training Accuracy |
|-------|------|-------------------|
| 1 | 3.089 | 33.2% |
| 5 | 1.187 | 67.3% |
| 10 | 0.884 | 74.1% |

### Search Algorithms Benchmarked

| Algorithm | Type | Library |
|-----------|------|---------|
| Brute Force (L2) | Exact | Scikit-Learn |
| k-d Tree | Exact | Scikit-Learn |
| Ball Tree | Exact | Scikit-Learn |
| Annoy | Approximate | Spotify Annoy |
| HNSW | Approximate | NMSLib |
| LSH (Cross-Polytope) | Approximate | Falconn |

All algorithms benchmarked on both full 2,048-d feature vectors and PCA-compressed 100-d vectors, with timing for single-query and batch (1,000 queries) modes.

## Architecture

```
Input Image → CNN Feature Extraction → PCA Dimensionality Reduction → Nearest Neighbor Search → Top-K Results
```

**Feature Extraction:** Extract 2,048-d embeddings from the penultimate layer of pretrained CNNs (ImageNet weights), then optionally fine-tune on the target dataset. Models benchmarked: VGG-16, VGG-19, ResNet-50, InceptionV3, MobileNet.

**Dimensionality Reduction:** PCA compresses 2,048-d → 100-d feature vectors, significantly accelerating search with minimal accuracy loss (87.06% → 87.65% on Caltech-101, slight improvement due to noise reduction).

**Search:** Index all image features and query using exact (brute force, k-d tree, ball tree) and approximate (Annoy, NMSLib HNSW, Falconn LSH) nearest neighbor algorithms.

**Visualization:** t-SNE projections to visualize feature space clustering — shows clear separation improvement after fine-tuning.

## Project Structure

```
├── 1_feature_extraction.ipynb            # Extract features from 5 pretrained CNNs, fine-tune ResNet-50
├── 2_similarity_search_level_1.ipynb     # Nearest neighbor search + result visualization
├── 2_similarity_search_level_2.ipynb     # Benchmark 6 algorithms (index + query timing)
├── 2_similarity_search_level_3.ipynb     # Calculate retrieval accuracies across datasets
├── 3_reduce_feature_length_with_pca.ipynb    # PCA optimization experiments
├── 4_improving_accuracy_with_fine_tuning.ipynb   # Fine-tune CNNs, t-SNE before/after comparison
└── README.md
```

## Tech Stack

- **Python**, **TensorFlow/Keras** — CNN feature extraction and fine-tuning
- **Scikit-Learn** — PCA, brute force, k-d tree, ball tree
- **Annoy** (Spotify) — Approximate nearest neighbors
- **NMSLib** — Non-Metric Space Library (HNSW graph-based search)
- **Falconn** — Locality-sensitive hashing
- **NumPy**, **Matplotlib** — Data processing and visualization

## Datasets

- [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) — 8,677 images across 101 object categories
- [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) — 30,607 images across 256 object categories

## Getting Started

```bash
git clone https://github.com/armansidhu3/neural-visual-search-engine.git
cd neural-visual-search-engine
pip install -r requirements.txt
```

Download the Caltech-101 and Caltech-256 datasets, then run the notebooks in order (1 → 2 → 3 → 4).
