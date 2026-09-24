# Groupwise Signed Hypergraph Learning for Resting-State fMRI Analysis

Code for signed hypergraph construction from resting-state fMRI functional
connectivity and the signed hypergraph network used for classification.

## Installation

```bash
pip install -r requirements.txt
```

## Data format

All inputs are NumPy arrays, stored separately for the training set and the test set:

| File | Shape | Content |
|---|---|---|
| `x_train.npy`, `x_test.npy` | (N, \|V\|, \|V\|) | Pearson correlation matrix of each subject |
| `y_train.npy`, `y_test.npy` | (N,) | binary labels (values > 0 are treated as the positive class) |

## 1. Signed hypergraph construction

```bash
python -m hypergraph_construction.run_construction --config configs/construction.yaml
```

This writes `H_train.npy` and `H_test.npy` (shape (N, |V|, |E|)) to `output_dir`.

The incidence matrices are obtained by minimising Eq. (7) under row-wise
normalisation with Algorithm 1 (`hypergraph_construction/construct.py`). Two settings
are supported through the `setting` field:

- `transductive`: the hypergraphs of the training and test subjects are constructed
  jointly from their functional connectivity. No labels are used during construction.
- `inductive`: the hypergraphs are constructed from the training subjects only. Each
  test subject is then initialised from the training-derived template and optimised
  independently with its own connectivity and the training-anchored sparsity term of
  Eq. (14) (`infer_hypergraph`).

Main parameters (`configs/construction.yaml`):

| Parameter | Meaning |
|---|---|
| `num_hyperedges` | number of hyperedges \|E\| |
| `alpha`, `beta`, `lam` | weights of the smoothness, diversity and groupwise sparsity terms |
| `learning_rate` | step size γ of Algorithm 1; if `null`, γ = `step_size` · N · \|V\|² |
| `max_iters`, `tol` | T_max and ε of Algorithm 1 |

Because every term of Eq. (7) is averaged over subjects and normalised by |V|² (or |E|²),
the gradient with respect to a single incidence matrix scales with 1 / (N |V|²). Setting
γ proportional to N |V|² keeps the per-subject step size independent of the cohort
size and the atlas.

## 2. Linear SVM on the incidence matrices

```bash
python run_svm.py --train-hypergraph data/H_train.npy --train-labels data/y_train.npy \
                  --test-hypergraph data/H_test.npy --test-labels data/y_test.npy
```

## 3. Signed hypergraph network

```bash
python train_hgnn.py --config configs/model.yaml
```

The network (`model/`) uses the rows of the Pearson correlation matrix as node
features, groupwise top-k pooling (Algorithm 2), signed hypergraph convolution
(Eq. 11) and a two-layer fully connected readout. It is trained on the training set
for a fixed number of epochs with Adam and a step learning-rate schedule, and is then
evaluated once on the test set (ACC, AUC, SEN, SPE). Results are written to
`output_dir/test_metrics.json`.

## Repository structure

```
hypergraph_construction/
    objective.py        Eq. (7) and the inductive objective (Eqs. 13-14), analytic gradients
    initialization.py   groupwise k-means initialisation (Algorithm 1, steps 1-5)
    construct.py        Algorithm 1 and inductive inference
    run_construction.py command-line entry point
model/
    layers.py           signed hypergraph convolution, groupwise pooling
    network.py          signed hypergraph network
    engine.py           training, prediction, metrics
run_svm.py              linear SVM on vectorised incidence matrices
train_hgnn.py           network training and test evaluation
configs/                example configurations
```
