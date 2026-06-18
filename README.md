# Sentence Clustering with BERT Embeddings & K-Means

## Overview

This project implements an unsupervised text clustering pipeline that combines transformer-based sentence embeddings with a custom K-Means clustering algorithm. The goal is to group semantically similar sentences and evaluate clustering quality using internal metrics.

---

## Techniques Used

### 1. BERT-based Sentence Embedding
Sentences are converted into dense vector representations using a pre-trained **BERT model (bert-base-uncased)** from Hugging Face Transformers.

- Tokenization with padding and truncation
- Batch inference to improve memory efficiency
- Pooler output used as sentence embedding

---

### 2. Dimensionality Reduction (PCA)
To improve efficiency and reduce noise, sentence embeddings are compressed using **Principal Component Analysis (PCA)**:

- Reduces embeddings to 5 dimensions
- Helps improve clustering stability and computation speed

---

### 3. Custom K-Means Clustering
A from-scratch implementation of the K-Means algorithm is used:

- Cosine distance as similarity metric
- Random centroid initialization
- Iterative update of cluster assignments and centroids
- Normalization of centroids after each update

---

### 4. Model Evaluation

Clustering performance is evaluated using:

- **Silhouette Score** (measures intra-cluster cohesion and inter-cluster separation)
- **Sum of Squared Errors (SSE)** (measures cluster compactness)
- Elbow method for estimating optimal K

---

## Pipeline

1. Load and clean raw text data
2. Generate sentence embeddings using BERT
3. Apply PCA for dimensionality reduction
4. Run K-Means clustering for different values of K
5. Evaluate clusters using Silhouette score and SSE
6. Select optimal K and output final labels
7. Visualize results

---

## Output

- `sentence_vectors_pca.npy` → processed sentence embeddings
- `Silhouette_Analysis_Plot.png` → Silhouette score vs K
- `SSE.png` → Elbow method visualization
- `labels.txt` / `Silhouette_labels.txt` / `SSE_labels.txt` → clustering results

---

## Dependencies

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Notes

- BERT inference is performed without gradient computation (`torch.no_grad()`)
- Batch processing is used for memory efficiency
- Cosine distance is used instead of Euclidean distance for better semantic clustering
