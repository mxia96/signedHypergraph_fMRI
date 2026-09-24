"""Linear SVM (default parameters) on the vectorised incidence matrices.

python run_svm.py --train-hypergraph data/H_train.npy --train-labels data/y_train.npy \
                  --test-hypergraph data/H_test.npy --test-labels data/y_test.npy
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.svm import SVC

from model.engine import classification_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-hypergraph", required=True)
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--test-hypergraph", required=True)
    ap.add_argument("--test-labels", required=True)
    args = ap.parse_args()

    H_train = np.load(args.train_hypergraph)
    H_test = np.load(args.test_hypergraph)
    y_train = (np.load(args.train_labels) > 0).astype(int)
    y_test = (np.load(args.test_labels) > 0).astype(int)

    clf = SVC(kernel="linear").fit(H_train.reshape(len(H_train), -1), y_train)
    score = clf.decision_function(H_test.reshape(len(H_test), -1))
    metrics = classification_metrics(y_test, score)
    print("test  " + "  ".join(f"{k} {v * 100:.1f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
