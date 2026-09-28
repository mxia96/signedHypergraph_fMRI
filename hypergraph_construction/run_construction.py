"""Build and save signed hypergraphs for the training and test sets."""
from __future__ import annotations

import argparse
import os

import numpy as np

from .config import ConstructionConfig
from .construct import apply_threshold, construct_hypergraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = ConstructionConfig.from_yaml(args.config)

    P_train = np.load(cfg.train_correlation).astype(np.float64)
    P_test = np.load(cfg.test_correlation).astype(np.float64)
    print(f"train {P_train.shape}, test {P_test.shape}")

    H_all, info = construct_hypergraph(np.concatenate([P_train, P_test]), cfg, cfg.verbose)
    print(f"construction: {info['iterations']} iterations, J = {info['objective']:.6f}")
    H_all = apply_threshold(H_all, cfg.threshold, cfg.tol)
    if cfg.threshold > 0:
        print(f"threshold {cfg.threshold}: {np.mean(H_all == 0) * 100:.1f}% of coefficients set to zero")
    H_train, H_test = H_all[:len(P_train)], H_all[len(P_train):]

    os.makedirs(cfg.output_dir, exist_ok=True)
    np.save(os.path.join(cfg.output_dir, "H_train.npy"), H_train.astype(np.float32))
    np.save(os.path.join(cfg.output_dir, "H_test.npy"), H_test.astype(np.float32))
    print(f"saved H_train {H_train.shape} and H_test {H_test.shape} to {cfg.output_dir}")


if __name__ == "__main__":
    main()
