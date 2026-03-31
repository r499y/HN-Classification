import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers_attention import GatedAttn, xavier_

class CLAMHead(nn.Module):
    """
    CLAM single-branch (binario). Attenzione gated + istanza-classifier (per instance-loss opzionale).
    Restituisce: bag_emb, attn, logit_bin, y_cont, inst_logits.
    """
    def __init__(self, in_dim=384, attn_dim=128, dropout=0.1):
        super().__init__()
        self.attn = GatedAttn(in_dim=in_dim, attn_dim=attn_dim, dropout=dropout)
        self.fc_bin = nn.Linear(in_dim, 1)  # bag -> logit binario
        self.fc_reg = nn.Linear(in_dim, 1)  # bag -> valore continuo
        self.inst_cls = nn.Linear(in_dim, 1) # per instance-loss (top-k)
        self.apply(xavier_)

    def forward(self, H):  # H: (N, D)
        bag_emb, attn = self.attn(H)                    # (D,), (N,)
        logit_bin = self.fc_bin(bag_emb).squeeze(-1)    # scalar
        y_cont   = self.fc_reg(bag_emb).squeeze(-1)     # scalar
        inst_logits = self.inst_cls(H).squeeze(-1)      # (N,)
        return {
            "bag_emb": bag_emb,
            "attn": attn,
            "logit_bin": logit_bin,
            "y_cont": y_cont,
            "inst_logits": inst_logits
        }

def clam_topk_indices(attn, k=8):
    k = min(k, attn.numel())
    return torch.topk(attn, k=k, largest=True).indices

def clam_instance_loss(inst_logits, attn, y_bin, k=8):
    """
    Instance-loss semplice stile CLAM:
    - se y=1 -> top-k (per attn) verso positivo
    - se y=0 -> top-k verso negativo
    """
    idx = clam_topk_indices(attn, k)
    logits = inst_logits[idx]                    # (k,)
    target = torch.full_like(logits, float(y_bin))
    return F.binary_cross_entropy_with_logits(logits, target)
