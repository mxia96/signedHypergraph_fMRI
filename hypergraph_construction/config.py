from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import yaml


@dataclass
class ConstructionConfig:
    # data: Pearson correlation stacks of shape (N, |V|, |V|)
    train_correlation: str = "data/x_train.npy"
    test_correlation: str = "data/x_test.npy"
    output_dir: str = "data"
    # transductive: construct jointly from train + test connectivity (no labels used)
    # inductive: construct from training subjects, then infer test subjects (Section 2.7)
    setting: str = "transductive"

    # Eq. (7)
    num_hyperedges: int = 15
    alpha: float = 0.6
    beta: float = 0.06
    lam: float = 0.03

    # Algorithm 1
    learning_rate: Optional[float] = None   # gamma; None -> step_size * N * |V|^2
    step_size: float = 1e-3
    max_iters: int = 2000                   # T_max
    tol: float = 1e-6                       # epsilon
    init_random_state: int = 42

    # coefficients with |h_ij| < threshold are set to zero in the saved incidence matrices and every
    # row is re-normalised to unit l2 norm (0 disables)
    threshold: float = 0.05

    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ConstructionConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**raw)
        if cfg.setting not in ("transductive", "inductive"):
            raise ValueError("setting must be 'transductive' or 'inductive'")
        return cfg
