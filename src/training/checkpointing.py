import numpy as np
from sklearn.metrics import average_precision_score

def calc_ap_lift(y,p):
    prev=float(np.mean(y)); pr=float(average_precision_score(y,p)) if len(np.unique(y))>1 else 0.0
    return pr-prev, pr, prev
