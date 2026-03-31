import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from src.training.losses import instance_dropout, feature_jitter, attention_entropy, smooth_targets
from src.models.heads_clam import clam_instance_loss
from src.utils.logging import _scalar

def run_one_epoch_v2(model, loader, phase, optimizer=None,
                     alpha_reg=0.5, pos_weight=None,
                     lambda_attn=0.0, label_smooth_eps=0.0,
                     jitter_sigma=0.02, inst_drop_p=0.1, min_keep=128,
                     use_clam_inst_loss=False, clam_topk=8,
                     device=None, max_grad_norm=2.0,lambda_inst=1e-3,coverage_acc=None):
    assert phase in {"train","val","test"}
    training = (phase == "train")
    model.train(training)

    all_bin, all_prob = [], []
    all_cont_true, all_cont_pred = [], []

    tot_loss, n_steps = 0.0, 0
    for batch in loader:
        # batch fields attesi: H (N,384), y_bin (scalar 0/1), y_cont (scalar float), has_cont (0/1)
        batch["H"] = batch["bag_feats"]
        H = batch["H"].to(device)                 # (N, D)
        y_bin_val  = _scalar(batch.get("y_bin"))
        assert np.isfinite(y_bin_val), "y_bin mancante/NaN: quel paziente non dovrebbe essere nel fold di train/val."
                # --- coverage logging nel MAIN via dict mutabile ---
        if (coverage_acc is not None) and (phase == "train"):
            pid = batch["patient_id"]
            # se per caso il collate restituisse una lista
            pid = str(pid)  # <<< coerente su tutte le epoche
            sel = batch.get("sel_idx", None)
            n_all = batch.get("n_all", None)
            if (sel is not None) and (n_all is not None):
                idx = sel.cpu().numpy()
                N   = int(n_all.item())
                buf = coverage_acc.get(pid)
                if (buf is None) or (buf.size != N):
                    buf = np.zeros(N, dtype=bool)
                    coverage_acc[pid] = buf
                if idx.size > 0:
                    buf[idx] = True

        y_cont_val = _scalar(batch.get("y_cont"), default=float("nan"))
        hasc_val   = _scalar(batch.get("has_cont"), default=float("nan"))
        # se has_cont non è valido, deducilo da y_cont (NaN => 0)
        if not np.isfinite(hasc_val):
            hasc_val = 1.0 if np.isfinite(y_cont_val) else 0.0
        
        yb       = torch.tensor([y_bin_val],  dtype=torch.float32, device=device)
        yhat     = torch.tensor([y_cont_val], dtype=torch.float32, device=device)
        has_cont = torch.tensor([hasc_val],   dtype=torch.float32, device=device)

        # Bag vuote: fallback 1xD zeros
        if H.ndim != 2 or H.size(0) == 0:
            H = torch.zeros((1, 384), dtype=torch.float32, device=device)

        # Augment/Reg only in training
        if training:
            H = instance_dropout(H, p=inst_drop_p, min_keep=min_keep)
            H = feature_jitter(H, sigma=jitter_sigma)

        out = model(H)
        logit = out["logit_bin"]
        y_cont_pred = out["y_cont"]
        attn = out["attn"]

        # --- Loss binaria con pos_weight e (opz.) label smoothing
        if label_smooth_eps > 0:
            yb_eff = smooth_targets(yb, eps=label_smooth_eps)
        else:
            yb_eff = yb
        pw = None
        if pos_weight is not None:
            pw = pos_weight.to(device)
        loss_bin = F.binary_cross_entropy_with_logits(logit.unsqueeze(0), yb_eff, pos_weight=pw)

        # --- Loss continua (mask su has_cont)
        if torch.isfinite(yhat).all() and has_cont.sum() > 0:
            loss_reg = F.smooth_l1_loss(y_cont_pred.unsqueeze(0), yhat, reduction="none")
            loss_reg = (loss_reg * has_cont).sum() / (has_cont.sum() + 1e-6)
        else:
            loss_reg = torch.tensor(0.0, device=device)

        loss_inst = torch.tensor(0.0, device=device)
        if use_clam_inst_loss and ("inst_logits" in out):
            # assicurati che clam_instance_loss sia MEDIA su tile, non somma
            loss_inst = lambda_inst * clam_instance_loss(out["inst_logits"], attn, yb.item(), k=clam_topk)

        loss_attn_reg = torch.tensor(0.0, device=device)
        if lambda_attn > 0:
            loss_attn_reg = - lambda_attn * attention_entropy(attn)

        loss = loss_bin  + loss_inst + loss_attn_reg+ alpha_reg * loss_reg

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # metriche
        prob = torch.sigmoid(logit).detach().item()
        all_bin.append(yb.item()); all_prob.append(prob)
        if has_cont.item() > 0.5 and torch.isfinite(yhat).item():
            all_cont_true.append(yhat.item()); all_cont_pred.append(y_cont_pred.detach().item())

        tot_loss += loss.detach().item()
        n_steps += 1

    # aggrega metriche
    bm = bin_metrics(all_bin, all_prob)
    cm = cont_metrics(all_cont_true, all_cont_pred)
    avg_loss = tot_loss / max(n_steps, 1)
    return {"loss": avg_loss, **bm, **cm}
