from __future__ import annotations

from dataclasses import dataclass, fields

import yaml


@dataclass
class ModelConfig:
    # data: node features (N, |V|, |V|), incidence matrices (N, |V|, |E|), labels (N,)
    train_features: str = "data/x_train.npy"
    train_hypergraph: str = "data/H_train.npy"
    train_labels: str = "data/y_train.npy"
    test_features: str = "data/x_test.npy"
    test_hypergraph: str = "data/H_test.npy"
    test_labels: str = "data/y_test.npy"
    output_dir: str = "outputs"

    # architecture
    hidden_features: int = 32
    input_activation: str = "tanh"
    depth: int = 1
    conv_dropout: float = 0.2
    pool_ratio: float = 0.8
    pool_momentum: float = 0.9
    readout_hidden: int = 64
    readout_dropout: float = 0.0

    # optimisation (Section 2.6)
    lr: float = 1e-3
    weight_decay: float = 1e-3
    scheduler_step: int = 10
    scheduler_gamma: float = 0.8
    epochs: int = 60
    batch_size: int = 60

    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)
