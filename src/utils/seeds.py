import os
import random
import numpy as np
import torch

def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_loader_seed(seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    def _wif(_):
        random.seed(seed); np.random.seed(seed)
    return g, _wif
