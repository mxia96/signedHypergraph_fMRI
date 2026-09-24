from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class HypergraphConv(nn.Module):
    """Signed hypergraph convolution, Eq. (11): D^{-1/2} H Delta^{-1} H^T D^{-1/2} X Theta.

    Vertex and hyperedge degrees are computed from absolute incidence weights.
    Nodes or hyperedges with zero degree (e.g. rows removed by pooling) receive
    no messages.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)
        torch.nn.init.xavier_normal_(self.lin.weight)
        self.bias = Parameter(torch.zeros(out_features))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, H: Tensor) -> Tensor:
        x = self.dropout(self.lin(x))
        dv = H.abs().sum(dim=2)
        de = H.abs().sum(dim=1)
        dv_isqrt = torch.where(dv > 0, dv.clamp_min(1e-12).rsqrt(), torch.zeros_like(dv))
        de_inv = torch.where(de > 0, 1.0 / de.clamp_min(1e-12), torch.zeros_like(de))
        x = x * dv_isqrt.unsqueeze(-1)
        x = torch.bmm(H.transpose(1, 2), x) * de_inv.unsqueeze(-1)
        x = torch.bmm(H, x) * dv_isqrt.unsqueeze(-1)
        return x + self.bias


class GroupwiseTopKPooling(nn.Module):
    """Groupwise node pooling (Algorithm 2).

    Node scores are averaged over the mini-batch and accumulated in a persistent
    buffer (initialised to 1, updated with momentum during training only). The
    same top-k ROIs are kept for every subject; the node dimension stays |V| and
    removed rows are set to zero.
    """

    def __init__(self, in_channels: int, num_regions: int, ratio: float = 0.8,
                 momentum: float = 0.9):
        super().__init__()
        self.num_regions = num_regions
        self.ratio = ratio
        self.momentum = momentum
        self.register_buffer("score_buffer", torch.ones(num_regions))
        self.weight = Parameter(torch.empty(1, in_channels))
        nn.init.uniform_(self.weight)

    def forward(self, x: Tensor, H: Tensor):
        score = torch.tanh((x * self.weight).sum(dim=-1) / self.weight.norm(p=2))
        if self.training:
            with torch.no_grad():
                self.score_buffer.mul_(self.momentum).add_((1 - self.momentum) * score.mean(dim=0))
        k = int(math.ceil(self.ratio * self.num_regions))
        idx = torch.topk(self.score_buffer, k).indices
        mask = torch.zeros(self.num_regions, device=x.device, dtype=x.dtype)
        mask[idx] = 1.0
        x = x * score.unsqueeze(-1) * mask.view(1, -1, 1)
        H = H * mask.view(1, -1, 1)
        return x, H, idx
