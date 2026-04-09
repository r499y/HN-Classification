import math
import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, brier_score_loss
from scipy.stats import spearmanr, pearsonr

def bin_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_prob)
    y, p = y_true[m], y_prob[m]
    if y.size < 2 or np.unique(y).size < 2:
        return {"AUC": np.nan, "PR-AUC": np.nan}
    return {"AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}

def cont_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y, p = y_true[m], y_pred[m]
    out = {"MAE": np.nan, "Spearman": np.nan, "Pearson": np.nan}
    if y.size >= 2:
        out["MAE"] = mean_absolute_error(y, p)
        try:
            out["Spearman"] = spearmanr(y, p).correlation
        except Exception:
            pass
        try:
            out["Pearson"] = pearsonr(y, p)[0]
        except Exception:
            pass
    return out

def compute_pos_weight(dataset, indices):
    ys = [float(dataset[i]["y_bin"]) for i in indices]
    pos = sum(ys)
    neg = len(ys) - pos
    pos_weight = (neg + 1e-6) / (pos + 1e-6)
    return torch.tensor([pos_weight], dtype=torch.float32)

def instance_dropout(H, p=0.1, min_keep=128):
    if H.size(0) <= min_keep or p <= 0:
        return H
    keep = torch.rand(H.size(0), device=H.device) > p
    if keep.sum().item() < min_keep:
        idx = torch.randperm(H.size(0), device=H.device)[:min_keep]
        keep = torch.zeros(H.size(0), dtype=torch.bool, device=H.device); keep[idx] = True
    return H[keep]

def feature_jitter(H, sigma=0.02):
    if sigma <= 0: return H
    return H + sigma * torch.randn_like(H)

def attention_entropy(attn):
    a = attn.clamp_min(1e-8)
    H = -(a * a.log()).sum()
    return H / math.log(attn.numel() + 1e-8)  # ∈ [0,1]

def smooth_targets(yb, eps=0.05):
    return yb*(1-eps) + 0.5*eps
