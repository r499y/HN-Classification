import math
import torch
import torch.nn as nn
from src.models.layers_attention import TransformerEncoderLayerRetAttn, RelPosBias2D, _xavier_


class DSMILHead(nn.Module):
    """
    DSMIL (dual-stream):
    - stream istanza: inst_logits per tile
    - stream bag: attenzione guidata dalla top-istanza (query = h_top)
    - fusione: convex combo tra p_inst(max) e p_bag, poi riconvertita a logit
    """
    def __init__(self, in_dim=384, attn_dim=128, alpha=0.5, dropout=0.1, use_v=True, fuse_mode="prob"):
        super().__init__()
        self.inst_cls = nn.Linear(in_dim, 1)        # istanza -> logit
        self.q = nn.Linear(in_dim, attn_dim)        # Q(h_top)
        self.k = nn.Linear(in_dim, attn_dim)        # K(H)
        self.v = nn.Linear(in_dim, in_dim) if use_v else nn.Identity()  # V(H)
        self.fc_bag = nn.Linear(in_dim, 1)          # bag_emb -> logit bin
        self.fc_reg = nn.Linear(in_dim, 1)          # bag_emb -> y_cont
        self.drop = nn.Dropout(dropout)
        self.alpha = float(alpha)
        assert 0.0 <= self.alpha <= 1.0
        assert fuse_mode in {"prob", "logit"}
        self.fuse_mode = fuse_mode
        self.apply(_xavier_)

    @staticmethod
    def _logit_from_prob(p, eps=1e-6):
        p = p.clamp(eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)

    def forward(self, H):            # H: (N, D)
        # ---- 1) stream istanza
        inst_logits = self.inst_cls(H).squeeze(-1)      # (N,)
        # inutile passare da sigmoid per l'argmax (monotona):
        top_index = torch.argmax(inst_logits)           # scalar

        # ---- 2) attenzione guidata da h_top
        h_top = H[top_index]                             # (D,)
        q = self.q(self.drop(h_top))                     # (A,)
        K = self.k(self.drop(H))                         # (N, A)
        scores = torch.matmul(K, q) / math.sqrt(K.size(-1))  # (N,)
        # calcola attn in float32 per stabilità con AMP/bfloat16
        attn = torch.softmax(scores.float(), dim=0).to(H.dtype)   # (N,)
        bag_emb = torch.sum(self.v(H) * attn.unsqueeze(-1), dim=0)  # (D,)

        # ---- 3) teste finali
        bag_logit = self.fc_bag(bag_emb).squeeze(-1)     # scalar
        y_cont    = self.fc_reg(bag_emb).squeeze(-1)     # scalar

        # ---- 4) fusione
        if self.fuse_mode == "prob":
            p_inst = torch.sigmoid(inst_logits).max()    # scalar
            p_bag  = torch.sigmoid(bag_logit)           # scalar
            p_mix  = self.alpha * p_inst + (1.0 - self.alpha) * p_bag
            logit_bin = self._logit_from_prob(p_mix)     # per BCEWithLogits
        else:  # "logit" (come il tuo codice attuale)
            logit_bin = self.alpha * inst_logits.max() + (1.0 - self.alpha) * bag_logit

        return {
            "bag_emb": bag_emb,
            "attn": attn,
            "logit_bin": logit_bin,
            "y_cont": y_cont,
            "inst_logits": inst_logits,
            "top_index": top_index
        }

class TransMILHead(nn.Module):
    """
    TransMIL completo:
    - Proiezione input -> d_model
    - Token CLS
    - K encoder layers con MHA che ritorna attn maps
    - Bias posizionale 2D relativo opzionale (slide-level)
    - Logit binario/regressione dal CLS
    - Attenzione di output = media teste della mappa CLS->token dell'ULTIMO layer
    - Top-K opzionale a monte per ridurre O(N^2)
    """
    def __init__(
        self,
        in_dim=384,
        d_model=384,
        n_heads=8,
        n_layers=2,
        dim_ff=1024,
        dropout=0.1,
        use_relpos2d=True,     # True per SlideBag; False per PatientBag
        topk_tokens: int = None  # es: 2048 per ridurre costo; None = usa tutti
    ):
        super().__init__()
        self.in_proj = nn.Identity() if in_dim == d_model else nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        self.layers = nn.ModuleList([
            TransformerEncoderLayerRetAttn(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        self.use_relpos2d = bool(use_relpos2d)
        self.relpos = RelPosBias2D(n_heads) if self.use_relpos2d else None
        self.n_heads = n_heads
        self.topk_tokens = topk_tokens

        # heads sul CLS
        self.fc_bin = nn.Linear(d_model, 1)
        # self.logit_norm = nn.LayerNorm(1)   # oppure nn.LayerNorm(1) se preferisci
        self.fc_reg = nn.Linear(d_model, 1)

        self.drop = nn.Dropout(dropout)
        self.apply(_xavier_)

    def _maybe_topk(self, H, coords_norm=None):
        """
        Riduce il numero di token (N) a top-k in base alla norma del feature vector.
        Metodo semplice/stabile per gestire bag enormi.
        Ritorna H_reduced, coords_reduced, idx_kept
        """
        if (self.topk_tokens is None) or (H.size(0) <= self.topk_tokens):
            N = H.size(0)
            idx = torch.arange(N, device=H.device)
            return H, coords_norm, idx

        # score semplice: ||h||2
        scores = torch.norm(H, dim=1)  # (N,)
        k = min(self.topk_tokens, H.size(0))
        topk = torch.topk(scores, k=k, largest=True, sorted=False).indices
        Hk = H[topk]
        ck = coords_norm[topk] if (coords_norm is not None) else None
        return Hk, ck, topk

    def forward(self, H, coords_norm=None):
        """
        H: (N, D)
        coords_norm: (N, 2) in [0,1] per slide (solo se use_relpos2d=True)
        returns dict: bag_emb, attn (N,), logit_bin, y_cont
        """
        device = H.device
        dtype  = H.dtype

        # N==0 fallback
        if H.ndim != 2 or H.size(0) == 0:
            H = torch.zeros((1, H.size(-1) if H.ndim==2 else 384), device=device, dtype=dtype)

        # top-k opzionale per O(N^2)
        Hk, ck, kept = self._maybe_topk(H, coords_norm)

        # proiezione + CLS
        X = self.in_proj(Hk)          # (Nk, d_model)
        X = self.drop(X)
        B = 1
        CLS = self.cls.expand(B, 1, X.size(-1))         # (1,1,D)
        S = torch.cat([CLS, X.unsqueeze(0)], dim=1)     # (1, 1+Nk, D)

        # bias posizionale
        if self.use_relpos2d and (ck is not None):
            attn_bias = self.relpos(ck.unsqueeze(0))    # (1, h, L, L) con L=1+Nk
        else:
            attn_bias = None

        # encoder
        attn_last = None
        Z = S
        for layer in self.layers:
            Z, attn_map = layer(Z, attn_bias=attn_bias)  # attn_map: (B, L, L) (media heads)
            attn_last = attn_map                         # tieni l'ultimo

        cls_out = Z[:, 0, :]        # (1, D)
        tok_out = Z[:, 1:, :]       # (1, Nk, D)

        # teste finali
        logit_bin = self.fc_bin(cls_out).squeeze()
        y_cont    = self.fc_reg(cls_out).squeeze()
                          
        # ----- teste finali -----
        # cls_out: (1, D)
        # logit_raw = self.fc_bin(cls_out)          # (1, 1)
        # logit_cal = self.logit_norm(logit_raw)      # (1, 1) rescalato
        # logit_bin = logit_cal.squeeze()           # scalar per BCEWithLogits

        # # regressione continua (raw, in R)
        # y_cont = self.fc_reg(cls_out).squeeze()
        

        # attenzione da restituire = CLS->token dall'ultimo layer (media heads)
        # attn_last: (1, L, L) in fp32; prendo riga CLS (0) verso i token (1:)
        attn_cls_to_tok = attn_last[:, 0, 1:].squeeze(0).float()  # (Nk,)

        # se abbiamo fatto top-k, rimappa su N con zeri dove token scartati
        if (self.topk_tokens is not None) and (kept is not None) and (kept.numel() != H.size(0)):
            attn_full = torch.zeros(H.size(0), dtype=attn_cls_to_tok.dtype, device=attn_cls_to_tok.device)
            attn_full[kept] = attn_cls_to_tok
            attn_out = attn_full
        else:
            attn_out = attn_cls_to_tok  # (N,)

        return {
            "bag_emb": cls_out.squeeze(0),   # (D,)
            "attn": attn_out,                # (N,)
            "logit_bin": logit_bin,          # scalar (logit per BCEWithLogits)
            "y_cont": y_cont                 # scalar
        }

def build_mil_head(kind: str, **kwargs) -> nn.Module:
    kind = (kind or "").lower()
    if kind == "clam":
        return CLAMHead(**kwargs)
    if kind == "dsmil":
        return DSMILHead(**kwargs)
    if kind == "transmil":
        return TransMILHead(**kwargs)
    raise ValueError(f"mil_head '{kind}' non supportato. Usa: 'clam' | 'dsmil' | 'transmil'")
