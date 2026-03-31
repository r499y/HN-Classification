import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.cluster import MiniBatchKMeans
from src.utils.io import parse_list, load_pt_matrix

class PatientBagsDataset(Dataset):
    def __init__(self, manifest_csv,
                 norm_mode="slide_center_l2",
                 sample_per_patient=2000,
                 centroid_frac=0.85,
                 kmeans_k=96,
                 random_state=42,
                 # kcenter_frac=None,           # se None usa centroid_frac
                 # diversity_mode="farthest",   # "farthest" (consigliato) | "random"
                 # diversity_temp=0.75
                ):
        self.df = pd.read_csv(manifest_csv)
        self.df["feat_paths_list"] = self.df["feat_paths"].apply(parse_list)
        self.norm_mode = norm_mode
        self.sample_per_patient = sample_per_patient
        self.centroid_frac = centroid_frac
        self.kmeans_k = kmeans_k
        self.rs = np.random.RandomState(random_state)
        self.split = "train"                 # "train" o "val" (impostalo quando crei ds)
        self.base_seed = 42                  # o quello che già usi
        self.fold = 0                        # impostalo da fuori per riproducibilità per-fold

        # cache val e rotazione train
        self.val_idx_cache = {}              # {patient_id: np.ndarray di indici fissi per VAL}
        self.train_perm = {}                 # {patient_id: permutazione intera degli indici tile}
        self.train_ptr  = {}                 # {patient_id: puntatore corrente nella permutazione}
        self._epoch = 0

        # coverage (debug)
        self._cover = {}                     # {patient_id: np.ndarray bool visti}
        # self.kcenter_frac  = centroid_frac if kcenter_frac is None else float(kcenter_frac)
        # self.diversity_mode = diversity_mode
        # self.diversity_temp = float(diversity_temp)
        
    def set_epoch(self, e: int):
        self._epoch = int(e)

    def set_split(self, split: str, fold: int = 0):
        assert split in ("train", "val")
        self.split = split
        self.fold = int(fold)
        # ---------- K-CENTER (FPS) + DIVERSITY HELPERS ----------

    @staticmethod
    def _pairwise_sq_dists(A, B):
        # A: (Na, D), B: (Nb, D) → (Na, Nb) con ||a-b||^2
        # stabile e veloce su float32
        AA = (A*A).sum(axis=1, keepdims=True)
        BB = (B*B).sum(axis=1, keepdims=True).T
        AB = A @ B.T
        D2 = np.maximum(AA + BB - 2.0*AB, 0.0)
        return D2

    def _kcenter_fps(self, X, k, rng=None, init_idx=None):
        """
        K-center greedy (farthest point sampling):
        - parte da init_idx (o dal punto a norma massima se None)
        - aggiunge ogni volta il punto più lontano dal set corrente
        X: (N, D) float32
        k: numero di punti da selezionare
        """
        N = X.shape[0]
        if k <= 0 or N == 0:
            return np.zeros((0,), dtype=int)
        k = min(k, N)

        # inizializzazione
        if init_idx is None:
            # primo punto: massimo della norma (stabile) o random deterministico
            norms = np.linalg.norm(X, axis=1)
            start = int(norms.argmax())
        else:
            start = int(init_idx)

        selected = [start]
        # mantieni le distanze minime di ogni punto al set selezionato
        d2_min = self._pairwise_sq_dists(X, X[[start], :]).reshape(-1)

        for _ in range(1, k):
            # scegli il più lontano dai selezionati
            nxt = int(np.argmax(d2_min))
            selected.append(nxt)
            # aggiorna d2_min con la nuova colonna
            d2_new = self._pairwise_sq_dists(X, X[[nxt], :]).reshape(-1)
            # distanza al set = min distanza a uno dei selezionati
            d2_min = np.minimum(d2_min, d2_new)

        return np.array(selected, dtype=int)

    def _diversity_fill(self, X, pool_idx, already_sel, m, rng, temp=0.5):
        """
        Riempi con punti "diversi" dal set già selezionato.
        Strategy 'farthest-softmax': probabilità ∝ softmax( min_dist^2 / temp )
        Se m<=0 o pool vuoto, ritorna [].
        """
        if m <= 0 or len(pool_idx) == 0:
            return np.zeros((0,), dtype=int)

        # distanza di ciascun pool-point al set selezionato
        X_pool = X[pool_idx]
        X_sel  = X[already_sel] if len(already_sel) else None
        if X_sel is None or X_sel.shape[0] == 0:
            # nessun selezionato: puro random (deterministico)
            return rng.choice(pool_idx, size=min(m, len(pool_idx)), replace=False)

        D2 = self._pairwise_sq_dists(X_pool, X_sel)   # (|pool|, |sel|)
        d2_min = D2.min(axis=1)                       # min distanza al set = diversità

        # softmax con temperatura
        s = d2_min / max(1e-6, float(temp))
        s -= s.max()  # stabilità
        p = np.exp(s)
        p /= p.sum() if p.sum() > 0 else 1.0

        m = min(m, len(pool_idx))
        # campionamento senza rimpiazzo via weighted sampling (approssimazione iterativa)
        chosen_rel = []
        avail = np.arange(len(pool_idx))
        probs = p.copy()
        for _ in range(m):
            j = rng.choice(avail, p=probs[avail]/probs[avail].sum())
            chosen_rel.append(int(j))
            # rimuovi j
            avail = avail[avail != j]
            # opzionale: aggiorna le probabilità penalizzando i vicini di j
            # qui manteniamo semplice (buono abbastanza in pratica)

        return pool_idx[np.array(chosen_rel, dtype=int)]

    def _sample_centroids_random(self, X_pool, n_cent, n_rand, rng=None):
        N = X_pool.shape[0]
        if N == 0:
            return np.zeros((0,), dtype=int)

        # # CLAMP su k per evitare cluster inutili
        # k_base = getattr(self, "kmeans_k", 32)
        # k = min(k_base, max(2, N // 10), N)
        k_min,k_max=16,96
        k=int(np.clip(int(round(np.sqrt(N))),k_min,min(k_max,N)))
        # random_state deterministico se rng è passato
        rs = None if rng is None else int(rng.randint(0, 2**31-1))

        # --- KMeans ---
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init='auto', random_state=rs)
        km.fit(X_pool)
        labels = km.labels_
        centers = km.cluster_centers_

        # indice della tile più vicina ad ogni centro
        nearest = []
        for c in range(k):
            idx_c = np.where(labels == c)[0]
            if idx_c.size == 0: 
                continue
            dif = X_pool[idx_c] - centers[c]
            d2 = np.einsum("ij,ij->i", dif, dif)
            nearest.append(idx_c[int(d2.argmin())])
        nearest = np.array(nearest, dtype=int)

        # prendi n_cent centroidi
        if nearest.size > n_cent > 0:
            sel_cent = (rng.choice(nearest, size=n_cent, replace=False) if rng is not None
                        else np.random.choice(nearest, size=n_cent, replace=False))
        else:
            sel_cent = nearest[:n_cent] if n_cent > 0 else np.zeros((0,), dtype=int)

        # random dal resto
        mask = np.ones(N, dtype=bool)
        mask[sel_cent] = False
        rem = np.where(mask)[0]
        n_rand = max(0, n_rand)
        if rem.size > 0 and n_rand > 0:
            sel_rand = (rng.choice(rem, size=min(n_rand, rem.size), replace=False) if rng is not None
                        else np.random.choice(rem, size=min(n_rand, rem.size), replace=False))
        else:
            sel_rand = np.zeros((0,), dtype=int)

        sel = np.concatenate([sel_cent, sel_rand])
        # se per qualsiasi motivo sono meno di (n_cent+n_rand), completa dal resto
        need = (n_cent + n_rand) - sel.size
        if need > 0 and rem.size > 0:
            extra = (rng.choice(rem, size=min(need, rem.size), replace=False) if rng is not None
                    else np.random.choice(rem, size=min(need, rem.size), replace=False))
            sel = np.unique(np.concatenate([sel, extra]))

        return sel[:(n_cent + n_rand)]
    

        
    def _select_indices_with_rng(self, X_full, idx_pool, rng, M_target):
        # scegli quanti centroidi/random vuoi come fai ora
        n_cent = int(round(self.centroid_frac * M_target))
        n_rand = M_target - n_cent
        # se hai X_full (features) disponibile, lavora sul sottoinsieme
        if X_full is not None:
            X_pool = X_full[idx_pool]
            rel = self._sample_centroids_random(X_pool, n_cent, n_rand, rng=rng)  # vedi patch sotto
            return idx_pool[rel]
        # fallback: puro shuffle deterministico
        idx = idx_pool.copy()
        rng.shuffle(idx)
        return idx[:M_target]
    
    def _fixed_val_indices(self, pid, X, rng, M_target):
        if not hasattr(self, "val_idx_cache"):
            self.val_idx_cache = {}
        if pid not in self.val_idx_cache:
            idx_pool = np.arange(X.shape[0])
            chosen   = self._select_indices_with_rng(X, idx_pool, rng, M_target)  # <<< passa X, non pid
            self.val_idx_cache[pid] = chosen
        return self.val_idx_cache[pid]


    
    def _next_block_train(self, pid, N, block):
        # inizializza permutazione per paziente
        if (pid not in self.train_perm) or (self.train_perm[pid].size != N):
            rng = np.random.default_rng((hash((str(pid), self.base_seed, self.fold)) & 0xffffffff))
            self.train_perm[pid] = rng.permutation(N)
            self.train_ptr[pid]  = 0
        # finestra scorrevole (rotazione circolare)
        p  = self.train_ptr[pid]
        idx = self.train_perm[pid]
        if p + block <= N:
            pool = idx[p:p+block]
        else:
            k = (p + block) - N
            pool = np.concatenate([idx[p:], idx[:k]])
        # stride “aggressivo ma non tutto”: metà pool
        stride = max(1, block // 2)
        self.train_ptr[pid] = (p + stride) % N
        return pool

    
    def coverage_report(self):
        return {pid: float(c.mean()) for pid, c in self._cover.items()} if self._cover else {}

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _l2_norm_rows(X, eps=1e-8):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, eps)

    def _apply_norm(self, X_list):
        if self.norm_mode == "none":
            return np.concatenate(X_list, 0)
        if self.norm_mode == "tile_l2":
            return np.concatenate([self._l2_norm_rows(X) for X in X_list], 0)
        if self.norm_mode == "slide_center":
            return np.concatenate([X - X.mean(0, keepdims=True) for X in X_list], 0)
        if self.norm_mode == "slide_center_l2":
            return np.concatenate([self._l2_norm_rows(X - X.mean(0, keepdims=True)) for X in X_list], 0)
        if self.norm_mode == "slide_zscore":
            return np.concatenate([(X - X.mean(0, keepdims=True))/(X.std(0, keepdims=True)+1e-6) for X in X_list], 0)
        raise ValueError(f"Unknown norm_mode: {self.norm_mode}")


    def __getitem__(self, i):
        
        def _stable_seed(pid, fold, base_seed):
            key = f"{pid}_{fold}_{base_seed}"
            h = hashlib.md5(key.encode("utf-8")).hexdigest()
            # prendi 8 hex (32 bit)
            return int(h[:8], 16)
        row = self.df.iloc[i]
        pid = row["patient_id"]
        # y_bin = int(row["y_bin"]) if pd.notna(row["y_bin"]) else None
        # # has_cont = bool(row["has_cont"]) if "has_cont" in row else False
        # # y_cont = float(row["y_cont"]) if ("y_cont" in row and pd.notna(row["y_cont"])) else None
        # y_cont_val = row.get("y_cont")
        # has_cont = pd.notna(y_cont_val)
        # y_cont = float(y_cont_val) if has_cont else None
        y_bin_val = row.get("y_bin", None)
        y_bin = int(y_bin_val) if (y_bin_val is not None and pd.notna(y_bin_val)) else None
        
        y_cont_val = row.get("y_cont", None)
        has_cont = (y_cont_val is not None and pd.notna(y_cont_val))
        y_cont = float(y_cont_val) if has_cont else None
        X_list, n_tiles_raw = [], 0
        for pt in row["feat_paths_list"]:
            if not isinstance(pt,str) or not os.path.exists(pt): continue
            X = load_pt_matrix(pt)
            if X is None: continue
            n_tiles_raw += X.shape[0]; X_list.append(X)
        if len(X_list)==0: X = np.zeros((0,768),np.float32)
        else: X = self._apply_norm(X_list)

        N_all = X.shape[0]
        M_target = min(self.sample_per_patient, N_all) if N_all>0 else 0

        # default per il caso M_target==0
        sel_idx = np.zeros((0,), dtype=np.int32)
        N_all_orig = N_all

        if M_target > 0:
            if getattr(self, "split", "train") == "val":
                # VAL deterministico (versione con KMeans che hai scelto)
                # rng_val = np.random.RandomState((hash((pid, self.fold, self.base_seed)) & 0xffffffff))
                seed = _stable_seed(pid, self.fold, self.base_seed)
                rng_val = np.random.RandomState(seed)
                idx = self._fixed_val_indices(pid, X, rng_val, M_target)  # PASSI X
            else:
                # TRAIN: rotazione senza sostituzione + selezione kmeans+random
                pool_size = max(M_target, int(1.5 * M_target))  # pool > target per diversità
                idx_pool = self._next_block_train(pid, N_all, pool_size)
                rng_tr = np.random.RandomState((hash((pid, getattr(self, "_epoch", 0), getattr(self, "base_seed", 42))) & 0xffffffff))
                idx = self._select_indices_with_rng(X, idx_pool, rng_tr, M_target)

            # ---- QUI: LOG COVERAGE (salva gli indici PRIMA di tagliare X) ----
            sel_idx = np.asarray(idx, dtype=np.int32)
            N_all_orig = N_all

            # taglia realmente le feature
            X = X[idx]

        bag_feats = torch.from_numpy(X)  # X è già float32

        return {
            "bag_feats": bag_feats,
            "patient_id": pid,
            "y_bin": torch.tensor([y_bin], dtype=torch.float32) if y_bin is not None else None,
            "y_cont": torch.tensor([y_cont/100.0], dtype=torch.float32) if y_cont is not None else None,
            "has_cont": torch.tensor([1.0 if has_cont else 0.0], dtype=torch.float32),
            "n_tiles_raw": n_tiles_raw,
            "n_wsi": int(row.get("n_wsi", 0)),
            # --- campi per coverage nel MAIN ---
            "sel_idx": torch.from_numpy(sel_idx),                 # (M_target,) int32
            "n_all": torch.tensor(N_all_orig, dtype=torch.int32), # scalar int32
        }

def collate_patient(batch):
    if len(batch)==1: return batch[0]
