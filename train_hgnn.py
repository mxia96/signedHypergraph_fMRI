"""Train the signed hypergraph network on the training set and evaluate it on the test set.

python train_hgnn.py --config configs/model.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import HypergraphDataset, ModelConfig, SignedHypergraphNet, classification_metrics, predict, train


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = ModelConfig.from_yaml(args.config)
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    train_set = HypergraphDataset(np.load(cfg.train_features), np.load(cfg.train_hypergraph),
                                  np.load(cfg.train_labels))
    test_set = HypergraphDataset(np.load(cfg.test_features), np.load(cfg.test_hypergraph),
                                 np.load(cfg.test_labels))
    num_roi = train_set.features.shape[1]
    print(f"train {len(train_set)}, test {len(test_set)}, |V| = {num_roi}, "
          f"|E| = {train_set.hypergraph.shape[2]}, device = {device}")

    # a trailing mini-batch of size 1 cannot be batch-normalised
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=len(train_set) % cfg.batch_size == 1)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    model = SignedHypergraphNet(cfg, num_roi).to(device)
    train(model, train_loader, cfg, device)

    y_test, score = predict(model, test_loader, device)
    metrics = classification_metrics(y_test, score)
    print("test  " + "  ".join(f"{k} {v * 100:.1f}" for k, v in metrics.items()))

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model.pt"))
    with open(os.path.join(cfg.output_dir, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
