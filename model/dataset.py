from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset


class HypergraphDataset(Dataset):
    """(label, node features, incidence matrix) triples. Labels are mapped to {0, 1} by y > 0."""

    def __init__(self, features, hypergraph, labels):
        self.features = np.asarray(features, dtype=np.float32)
        self.hypergraph = np.asarray(hypergraph, dtype=np.float32)
        self.labels = (np.asarray(labels) > 0).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx:idx + 1], self.features[idx], self.hypergraph[idx]
