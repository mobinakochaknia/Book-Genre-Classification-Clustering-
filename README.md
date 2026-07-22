# Book Genre Classification & Clustering 📚

**Repo name suggestion:** `book-genre-classification-clustering`
**Short description:** An end-to-end NLP pipeline — built from scratch, without scikit-learn's models — for classifying and clustering books by genre from their descriptions.

## Overview

This notebook works with a dataset of book descriptions and metadata to solve two related problems: **predicting a book's genre** (classification) and **discovering natural groupings** among book descriptions (clustering) — comparing TF-IDF against sentence embeddings as text representations throughout. A deliberate constraint of the project is that scikit-learn's classifiers, vectorizers, scalers, splitters, metrics, and clustering algorithms are **not** used (only `PCA`, for visualization); every model and metric is implemented from first principles.

## Part 1 — Book Genre Classification

- Mapping multi-label genre tags down to a single target genre per book (priority-based resolution)
- Cleaning the dataset and filtering to the most common classes
- Building **TF-IDF** features from scratch and comparing them against precomputed **sentence embeddings**
- Custom train/dev/test splitting and evaluation utilities
- Handling class imbalance and comparing mitigation strategies
- Training and comparing classifiers on both representations (TF-IDF-based models and sentence-embedding-based models, including an MLP on dense TF-IDF features)
- Model selection based on the accuracy/efficiency trade-off, followed by final test-set evaluation
- **Error analysis:** confusion matrix, per-class F1 scores, most common confusion pairs
- Interpreting which words drive the TF-IDF + Logistic Regression model's predictions

## Part 2 — Book Description Clustering

- PCA visualization of the embedding space before clustering
- Clustering evaluation metrics implemented from scratch
- **K-Means** and **Spherical K-Means** implemented and compared from scratch
- **DBSCAN** implemented from scratch, with a parameter sensitivity analysis
- **Hierarchical Agglomerative Clustering (HAC)** implemented from scratch, with merge-distance dendrogram-style plots
- Final comparison across all four clustering algorithms, followed by cluster interpretation and deep-dives into individual clusters (including inspecting DBSCAN noise points)

## Tech Stack

- **NumPy / SciPy (sparse)** — vectorized math and sparse TF-IDF matrices
- **pandas** — dataset handling
- **scikit-learn** — used *only* for PCA (dimensionality reduction for visualization)
- **Matplotlib** — all plots and visualizations
- **gdown** — automated dataset download (for Colab / cloud runs)
- **tqdm** — progress tracking

## Project Structure

```
book-genre-classification-clustering/
└── book_genre_classification_and_clustering.ipynb
```

## Getting Started

```bash
git clone https://github.com/<your-username>/book-genre-classification-clustering.git
cd book-genre-classification-clustering
pip install numpy scipy pandas matplotlib scikit-learn gdown tqdm
jupyter notebook book_genre_classification_and_clustering.ipynb
```

The notebook expects an input CSV containing a `description_emb` column (precomputed sentence embeddings); a cell for automatic download is included for Colab environments.
