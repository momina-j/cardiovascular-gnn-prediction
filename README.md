Patient Similarity Graph + GCN for Heart Disease Prediction

An independent academic project exploring whether modelling patients as nodes in a similarity graph, instead of independent rows, helps heart disease classification. Not peer-reviewed or published. This is a learning project, not a clinical tool.

Idea

Standard tabular models treat each patient record as an independent feature vector. Here, each patient becomes a node connected to their most similar patients, and a Graph Convolutional Network (GCN) propagates information across those connections.

Dataset
Cleaned, merged heart disease dataset (cleaned_merged_heart_dataset.csv)
1,888 patients, 14 features, binary target (heart disease: yes/no)

Method
Preprocessing: features scaled to [0, 1] with MinMaxScaler.
Graph construction: K-nearest neighbours (k = 10, Euclidean distance) on the scaled features. Edges are made undirected and de-duplicated.
Model: two GCNConv layers (32 then 16 hidden units, ReLU, dropout 0.4) followed by a linear classification layer.
Training: Adam (lr = 0.005), cross-entropy loss, 200 epochs.
Evaluation: stratified 80/20 train/test split (random seed 13).
Uncertainty: bootstrap resampling of the test set (see below).
Results (held-out test set)
Metric	Point estimate	95% bootstrap CI
Accuracy	0.8730	0.8412 to 0.9048
Precision	0.8776	0.8316 to 0.9242
Recall	0.8776	0.8298 to 0.9235
F1 score	0.8776	0.8430 to 0.9109
ROC-AUC	0.9366	0.9131 to 0.9584
Bootstrap confidence intervals

A single train/test split gives one number per metric. To show how much these numbers could vary, I estimated 95% confidence intervals with the percentile bootstrap:

Take the trained model's predictions on the test set (y_true, y_pred, y_prob).
Resample the test set with replacement, keeping its original size.
Recompute each metric on the resample.
Repeat 1,000 times (resamples containing only one class are skipped).
Report the 2.5th and 97.5th percentiles as the 95% interval.

The model is trained once; only the test set is resampled. The intervals therefore reflect variation from test-set sampling only. They do not capture variation from different train/test splits, random seeds or hyperparameters.

Limitations
Transductive setup: the graph is built over all patients (train and test), so test-node features take part in message passing during training. Labels of test nodes are not used, but this differs from a strictly inductive evaluation.
Single split and seed: results come from one stratified split. Repeating over several seeds or cross-validation would give a fuller picture.
No baseline yet: the GCN has not been compared against non-graph baselines (e.g. Logistic Regression, Random Forest) on the same split, so this project does not show that the graph structure improves performance.
Small dataset, one source: no external validation.
Reproducing
bash
pip install torch torch_geometric scikit-learn pandas numpy matplotlib networkx
Put cleaned_merged_heart_dataset.csv in the working directory (or update the path in the notebook).
Run [notebook_name].ipynb from top to bottom. The random seed is fixed to 13.
Tech stack

Python, PyTorch, PyTorch Geometric, scikit-learn, pandas, NumPy, Matplotlib, NetworkX
