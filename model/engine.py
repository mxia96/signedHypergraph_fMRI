from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def classification_metrics(y_true, score, threshold=0.0):
    """ACC, AUC, SEN, SPE from binary labels and real-valued scores (logits / decision values)."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "ACC": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, score),
        "SEN": tp / (tp + fn) if tp + fn else 0.0,
        "SPE": tn / (tn + fp) if tn + fp else 0.0,
    }


def train(model, loader, cfg, device):
    """Train for a fixed number of epochs (Adam + StepLR). The test set is not used here."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler_step,
                                                gamma=cfg.scheduler_gamma)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for labels, feats, H in loader:
            labels, feats, H = labels.to(device), feats.to(device), H.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats, H), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        scheduler.step()
        if epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"epoch {epoch:3d}  train loss {total / max(batches, 1):.4f}")
    return model


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, labels = [], []
    for y, feats, H in loader:
        scores.append(model(feats.to(device), H.to(device)).cpu().numpy().ravel())
        labels.append(y.numpy().ravel())
    return np.concatenate(labels), np.concatenate(scores)
