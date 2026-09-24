from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GroupwiseTopKPooling, HypergraphConv

_ACTIVATIONS = {"none": nn.Identity, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
                "elu": nn.ELU, "leaky_relu": nn.LeakyReLU, "sigmoid": nn.Sigmoid}


class SignedHypergraphNet(nn.Module):
    """Signed hypergraph network (Section 2.3).

    Node features are the rows of the subject's Pearson correlation matrix. Each
    block applies groupwise pooling (Algorithm 2) followed by a signed hypergraph
    convolution (Eq. 11). The readout concatenates all node embeddings and applies
    a two-layer fully connected classifier.
    """

    def __init__(self, cfg, num_roi):
        super().__init__()
        out = cfg.hidden_features
        self.input_proj = nn.Linear(num_roi, out)
        self.input_act = _ACTIVATIONS[cfg.input_activation]()
        self.pools = nn.ModuleList(
            [GroupwiseTopKPooling(out, num_roi, cfg.pool_ratio, cfg.pool_momentum)
             for _ in range(cfg.depth)])
        self.convs = nn.ModuleList(
            [HypergraphConv(out, out, dropout=cfg.conv_dropout) for _ in range(cfg.depth)])
        self.readout = nn.Sequential(
            nn.Linear(out * num_roi, cfg.readout_hidden),
            nn.BatchNorm1d(cfg.readout_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.readout_dropout),
            nn.Linear(cfg.readout_hidden, 1),
        )

    def forward(self, x, H):
        x = self.input_act(self.input_proj(x))
        for pool, conv in zip(self.pools, self.convs):
            x, H, _ = pool(x, H)
            x = F.relu(conv(x, H))
        return self.readout(x.flatten(start_dim=1))
