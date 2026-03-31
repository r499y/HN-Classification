import numpy as np
import torch

from src.utils.logging import _scalar


def infer_loader_test(model, loader, device):
    """
    Come infer_fold ma robusto a y mancanti (test set).
    Ritorna y_true (può contenere NaN), y_prob, patient_id.
    """
    model.eval()
    ys, ps, pids = [], [], []
    with torch.no_grad():
        for batch in loader:
            H = batch.get("bag_feats", batch.get("H"))
            if isinstance(H, np.ndarray):
                H = torch.from_numpy(H)
            H = H.to(device).float() if H is not None else None
            if H is None or H.ndim != 2 or H.size(0) == 0:
                H = torch.zeros((1, 768), dtype=torch.float32, device=device)

            out = model(H)
            logit = out["logit_bin"]
            prob = torch.sigmoid(logit).view(-1)[0].item()
            ps.append(float(prob))

            yb = batch.get("y_bin", None)
            if yb is None:
                ys.append(np.nan)
            else:
                y_val = _scalar(yb)
                ys.append(float(y_val))

            pid = batch.get("patient_id")
            if isinstance(pid, (list, tuple)):
                pid = pid[0]
            if isinstance(pid, torch.Tensor):
                pid = pid.item() if pid.ndim == 0 else pid[0].item()
            pids.append(str(pid))

    return np.array(ys), np.array(ps), np.array(pids)


def get_mu_sigma(logit_stats, split_seed, fold):
    row = logit_stats[
        (logit_stats["split_seed"] == split_seed) &
        (logit_stats["fold"] == fold)
    ]
    if row.empty:
        raise ValueError(f"Nessuna stat di rescaling per split_seed={split_seed}, fold={fold}")
    row = row.iloc[0]
    return float(row["logit_mean"]), float(row["logit_std"])
