import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

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
