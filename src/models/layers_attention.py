import torch
import torch.nn as nn
import torch.nn.functional as F

def xavier_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

def _xavier_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class GatedAttn(nn.Module):
    """Gated Attention (Ilse et al.), backbone di CLAM."""
    def __init__(self, in_dim=384, attn_dim=128, dropout=0.0):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim, bias=True)
        self.U = nn.Linear(in_dim, attn_dim, bias=True)
        self.w = nn.Linear(attn_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(dropout)
        self.apply(xavier_)

    def forward(self, H):           # H: (N, D)
        Vh = self.tanh(self.V(H))   # (N, A)
        Uh = self.sigmoid(self.U(H))
        A = self.drop(Vh * Uh)      # (N, A)
        A = self.w(A).squeeze(-1)   # (N,)
        A = torch.softmax(A, dim=0) # somma=1 nel bag
        Z = torch.sum(H * A.unsqueeze(-1), dim=0)  # (D,)
        return Z, A

class RelPosBias2D(nn.Module):
    """
    Bias posizionale 2D relativo:
    - Input: coords_norm (B, N, 2) con valori in [0,1] per ogni token (NO CLS)
    - Costruisce bias per tutte le coppie (i,j): b_ij = MLP([dx, dy, |dx|, |dy|, r, r^2])
    - Restituisce (B, n_heads, L, L) dove L = N+1 (includo CLS con bias = 0 verso/da altri)
    """
    def __init__(self, n_heads: int, hidden: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(),
            nn.Linear(hidden, n_heads)
        )
        self.apply(_xavier_)

    def forward(self, coords_norm: torch.Tensor) -> torch.Tensor:
        """
        coords_norm: (B, N, 2) in [0,1]
        returns: attn_bias (B, n_heads, N+1, N+1)
        """
        B, N, _ = coords_norm.shape
        if N == 0:
            # no tokens -> solo CLS (1)
            return coords_norm.new_zeros(B, self.n_heads, 1, 1)

        xy = coords_norm  # (B,N,2)
        xi = xy.unsqueeze(2)              # (B,N,1,2)
        xj = xy.unsqueeze(1)              # (B,1,N,2)
        d  = xi - xj                      # (B,N,N,2)
        dx, dy = d[..., 0], d[..., 1]     # (B,N,N)

        r2 = dx*dx + dy*dy
        r = torch.sqrt(r2 + 1e-12)
        feat = torch.stack([dx, dy, dx.abs(), dy.abs(), r, r2], dim=-1)  # (B,N,N,6)

        bh = self.mlp(feat)  # (B, N, N, n_heads)
        bh = bh.permute(0, 3, 1, 2).contiguous()  # (B, n_heads, N, N)

        # inserisco una riga/colonna per CLS con bias=0
        L = N + 1
        out = torch.zeros((B, self.n_heads, L, L), dtype=bh.dtype, device=bh.device)
        out[:, :, 1:, 1:] = bh  # CLS(0) senza bias
        return out

class MultiheadSelfAttention(nn.Module):
    """
    MHA custom (Q=K=V=X proiettato) che:
    - supporta attn_bias (B, h, L, L) additivo
    - ritorna attn_weights per ispezione
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.apply(_xavier_)

    def forward(self, x, attn_bias=None):
        """
        x: (B, L, D)
        attn_bias: (B, h, L, L) or None
        returns: (out, attn_weights_meanHeads)
            out: (B, L, D)
            attn_weights_meanHeads: (B, L, L) (media sulle teste) in fp32
        """
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)  # ciascuno (B, L, D)

        # reshape per heads
        def split_heads(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, L, d)
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # scaled dot-product
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,h,L,L)
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        # softmax in fp32 per stabilità
        attn_weights = F.softmax(attn_scores.float(), dim=-1)  # (B,h,L,L, fp32)
        attn_weights = attn_weights.to(v.dtype)
        attn = self.drop(attn_weights)

        out = torch.matmul(attn, v)  # (B,h,L,d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B,L,D)
        out = self.proj(out)
        # media teste per ispezione (ritorno fp32)
        attn_mean = attn_weights.mean(dim=1)  # (B,L,L)
        return out, attn_mean

class TransformerEncoderLayerRetAttn(nn.Module):
    def __init__(self, d_model=384, n_heads=8, dim_ff=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.drop2 = nn.Dropout(dropout)
        self.apply(_xavier_)

    def forward(self, x, attn_bias=None):
        """
        x: (B,L,D)
        attn_bias: (B,h,L,L) or None
        returns: x_out, attn_meanHeads (B,L,L)
        """
        # pre-norm
        y, attn_map = self.attn(self.norm1(x), attn_bias=attn_bias)
        x = x + self.drop1(y)
        y = self.ff(self.norm2(x))
        x = x + self.drop2(y)
        return x, attn_map
