import ast
import os
import numpy as np
import torch

def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(str(x))
    except: return []

def to_numpy_f32(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if isinstance(x, (list, tuple)):
        arr = [to_numpy_f32(t) for t in x]
        arr = [t for t in arr if t is not None]
        if not arr: return None
        return np.concatenate(arr, 0) if arr[0].ndim==2 else np.stack(arr, 0)
    if isinstance(x, dict):
        for k in ("features","feats","X","x","embeddings","data"):
            if k in x: return to_numpy_f32(x[k])
        if len(x)==1: return to_numpy_f32(next(iter(x.values())))
        return None
    return None

def load_pt_matrix(pt_path, D_expected=768):
    if not os.path.exists(pt_path):
        return None
    try:
        obj = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"[SKIP load] {pt_path}: {e}")
        return None
    X = to_numpy_f32(obj)
    if X is None:
        print(f"[WARN] {pt_path}: formato non riconosciuto"); return None
    if X.ndim == 1: X = X[None, :]
    if X.shape[1] != D_expected:
        print(f"[WARN] {pt_path}: D={X.shape[1]} != {D_expected}; skipping"); return None
    return X
