from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import yaml


@dataclass
class ConstructionConfig:
    # data: Pearson correlation stacks (N, |V|, |V|)
    train_correlation: str = "data/x_train.npy"
    test_correlation: str = "data/x_test.npy"
    output_dir: str = "data"

    # objective
    num_hyperedges: int = 15
    alpha: float = 0.6
    beta: float = 0.06
    lam: float = 0.03

    # optimisation
    learning_rate: Optional[float] = None   # None -> step_size * N * |V|^2
    step_size: float = 1e-3
    max_iters: int = 2000
    tol: float = 1e-6
    init_random_state: int = 42

    # zero |h_ij| < threshold and re-normalise rows (0 disables)
    threshold: float = 0.0

    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ConstructionConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)
