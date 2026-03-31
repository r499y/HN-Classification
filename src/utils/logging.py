import numpy as np
import torch

def _scalar(x, default=float("nan")):
    import torch
    if x is None:
        return default
    if torch.is_tensor(x):
        return float(x.view(-1)[0].item()) if x.numel() else default
    try:
        return float(x)
    except Exception:
        return default

def _norm_pid(x):
    import pandas as pd
    try:
        if isinstance(x,(list,tuple)): x=x[0]
        if hasattr(x,'item'): x=x.item()
        if isinstance(x,(int,float)) and not(pd.isna(x)): return str(int(x))
        return str(x)
    except Exception:
        return str(x)
