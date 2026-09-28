# Patient Similarity Graph + GCN for Heart Disease Prediction

> An independent academic project exploring whether modelling patients as nodes in a similarity graph, instead of independent rows, helps heart disease classification.
>
> **Not peer-reviewed or published. This is a learning project, not a clinical tool.**

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red) ![Status](https://img.shields.io/badge/Status-Academic%20project-lightgrey)

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Results](#results)
5. [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
6. [Limitations](#limitations)
7. [Repository Structure](#repository-structure)
8. [Reproducing the Results](#reproducing-the-results)
9. [Tech Stack](#tech-stack)

---

## Overview

Standard tabular models treat each patient record as an independent feature vector. In this project, each patient becomes a **node** connected to their most similar patients, and a **Graph Convolutional Network (GCN)** propagates information across those connections.

| | |
|---|---|
| **Task** | Binary node classification (heart disease: yes / no) |
| **Graph** | k-nearest-neighbour patient similarity graph (k = 10) |
| **Model** | 2-layer GCN + dropout + linear classifier |
| **Evaluation** | Stratified 80/20 split, with bootstrap 95% confidence intervals |

---

## Dataset

| | |
|---|---|
| **File** | `cleaned_merged_heart_dataset.csv` |
| **Patients** | 1,888 |
| **Features** | 14 |
| **Target** | Binary (heart disease: yes / no) |
| **Source** | `[add dataset name / link, e.g. Kaggle or UCI]` |

---

## Method

| Step | Description |
|---|---|
| **1. Preprocessing** | Features scaled to [0, 1] with `MinMaxScaler` |
| **2. Graph construction** | K-nearest neighbours (k = 10, Euclidean distance) on the scaled features; edges made undirected and de-duplicated |
| **3. Model** | Two `GCNConv` layers (32 then 16 hidden units, ReLU, dropout 0.4), followed by a linear classification layer |
| **4. Training** | Adam optimiser (lr = 0.005), cross-entropy loss, 200 epochs |
| **5. Evaluation** | Stratified 80/20 train/test split (random seed 13) |
| **6. Uncertainty** | Bootstrap resampling of the test set (see [below](#bootstrap-confidence-intervals)) |

---

## Results

Held-out test set:

| Metric | Point estimate | 95% bootstrap CI |
|---|---|---|
| Accuracy | 0.8730 | 0.8412 to 0.9048 |
| Precision | 0.8776 | 0.8316 to 0.9242 |
| Recall | 0.8776 | 0.8298 to 0.9235 |
| F1 score | 0.8776 | 0.8430 to 0.9109 |
| ROC-AUC | 0.9366 | 0.9131 to 0.9584 |

---

## Bootstrap Confidence Intervals

A single train/test split gives one number per metric. To show how much these numbers could vary, 95% confidence intervals were estimated with the **percentile bootstrap**:

1. Take the trained model's predictions on the test set (`y_true`, `y_pred`, `y_prob`).
2. Resample the test set with replacement, keeping its original size.
3. Recompute each metric on the resample.
4. Repeat 1,000 times (resamples containing only one class are skipped).
5. Report the 2.5th and 97.5th percentiles as the 95% interval.

**Scope:** the model is trained once and only the test set is resampled. The intervals therefore reflect variation from test-set sampling only. They do not capture variation from different train/test splits, random seeds or hyperparameters.

---

## Limitations

- **Transductive setup:** the graph is built over all patients (train and test), so test-node features take part in message passing during training. Labels of test nodes are not used, but this differs from a strictly inductive evaluation.
- **Single split and seed:** results come from one stratified split. Repeating over several seeds or using cross-validation would give a fuller picture.
- **No baseline yet:** the GCN has not been compared against non-graph baselines (e.g. Logistic Regression, Random Forest) on the same split, so this project does not show that the graph structure improves performance.
- **Small dataset, one source:** no external validation.

---

## Repository Structure

```
.
├── [notebook_name].ipynb            # full pipeline: graph, training, evaluation, bootstrap
├── cleaned_merged_heart_dataset.csv # dataset
└── README.md
```

---

## Reproducing the Results

**1. Install dependencies**

```bash
pip install torch torch_geometric scikit-learn pandas numpy matplotlib networkx
```

**2. Add the data**

Place `cleaned_merged_heart_dataset.csv` in the working directory (or update the path in the notebook).

**3. Run the notebook**

Run `[notebook_name].ipynb` from top to bottom. The random seed is fixed to 13.

---

## Tech Stack

Python · PyTorch · PyTorch Geometric · scikit-learn · pandas · NumPy · Matplotlib · NetworkX
