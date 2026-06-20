# Cardiovascular Heart Attack Prediction using Graph Neural Networks

A graph-based approach to cardiovascular risk prediction — patients are modeled as nodes in a similarity graph rather than independent rows, allowing a Graph Convolutional Network (GCN) to learn from neighborhood-level relational patterns.

## Problem

Most heart-attack risk prediction models (SVM, Random Forest, Logistic Regression, etc.) treat each patient as an independent sample. This ignores the fact that clinically similar patients often share latent risk patterns that flat, row-wise models can't capture.

## Approach

1. **Preprocessing** — cleaned clinical data (invalid entries removed, numeric type conversion, Min-Max normalization)
2. **Patient Similarity Graph** — built using K-Nearest Neighbors (K=10, Euclidean distance) on normalized clinical features
3. **GCN Model** — 2 Graph Convolutional layers (32 → 16 dims) + dropout (0.4) + fully connected output layer
4. **Training** — Adam optimizer (lr=0.005), cross-entropy loss, 200 epochs, node-level train/test masking (80/20)

## Dataset

[Heart Disease Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/mfarhaannazirkhan/heart-dataset) — 1,888 cleaned patient records, 14 clinical features (age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, vessel count, thalassemia, target).

## Results

| Metric | Score |
|---|---|
| Accuracy | 87.30% |
| Precision | 87.76% |
| Recall | 87.76% |
| F1 Score | 87.76% |
| ROC-AUC | 0.937 |

> **Note:** The original Colab notebook for this project was lost (deleted from Drive before backup). The notebook in this repo is a faithful reconstruction from the project's documented methodology (same architecture, hyperparameters, and dataset). Results are close to but not identical to the originally reported numbers — this is expected, since model weight initialization and the train/test split are randomized, and a small dataset (1,888 patients) naturally produces some run-to-run variance. Re-running this notebook with different seeds typically yields 83-87% accuracy and 0.88-0.94 AUC.

Full visualizations (KNN similarity graph, training loss curve, confusion matrix, ROC curve) are in the notebook.

## How to Run

```bash
pip install -r requirements.txt
```

Open `gcn_heart_attack_prediction.ipynb` in Jupyter or Colab and run all cells. The dataset (`cleaned_merged_heart_dataset.csv`) is included in this repo.

## Notes

This is an independent academic / semester research project, not peer-reviewed or published. Built to explore graph-based learning approaches for structured clinical data beyond standard classification pipelines.
