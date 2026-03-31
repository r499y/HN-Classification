#!/usr/bin/env python3
"""Extract TITAN/CONCH tile embeddings from a WSI given a tile-index parquet.

This version is robust for *external* cohorts (e.g., TCGA) where SVS metadata may
miss MPP. It will automatically read tiles in level-0 pixel coordinates when the
parquet contains (x0,y0,w0,h0) (or similar), avoiding units='mpp'.

Optional: apply stain normalization (Macenko/Reinhard) *after* reading tiles and
*before* the model transform. This is the right place to normalize externals.

Author: Nicolò Raganato (patched for external cohorts)
"""

import argparse
import json
import hashlib
import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

from tiatoolbox.wsicore.wsireader import WSIReader


# ---------------------------
# Small helpers
# ---------------------------


def sha1_of_file(path, chunk=1024 * 1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_params(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def load_titan_conch(device, local_files_only=False):
    titan = AutoModel.from_pretrained(
        "MahmoodLab/TITAN",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    titan.to(device).eval()
    conch, eval_transform = titan.return_conch()
    conch.to(device).eval()
    return conch, eval_transform


def pick_xy_cols(df):
    for a, b in [
        ("x0", "y0"),
        ("x", "y"),
        ("X", "Y"),
        ("loc_x", "loc_y"),
        ("center_x", "center_y"),
        ("cx", "cy"),
    ]:
        if a in df.columns and b in df.columns:
            return a, b
    raise RuntimeError(f"x/y columns not found in parquet: {list(df.columns)}")


def pick_wh_cols(df):
    for a, b in [
        ("w0", "h0"),
        ("w", "h"),
        ("W", "H"),
        ("width", "height"),
    ]:
        if a in df.columns and b in df.columns:
            return a, b
    return None, None


def pick_id_col(df):
    for c in ["istologico", "patient_id", "case_id", "slide_id", "sample_id"]:
        if c in df.columns:
            return c
    return None


def qc_rgb_mean_std(arr_u8):
    a = arr_u8.astype(np.float32) / 255.0
    return float(a.mean()), float(a.std())


def ensure_rgb_u8(arr):
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.shape[-1] >= 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def read_tile_level0(wsi, x0, y0, w0, h0, out_size):
    """Read a tile using level-0 pixel coords (robust when MPP is missing)."""
    arr = wsi.read_rect(
        location=(int(x0), int(y0)),
        size=(int(w0), int(h0)),
        resolution=0,
        units="level",
    )
    arr = ensure_rgb_u8(np.asarray(arr))
    img = Image.fromarray(arr).convert("RGB")
    if img.size != (out_size, out_size):
        img = img.resize((out_size, out_size), resample=Image.BILINEAR)
    return img


def read_tile_mpp(wsi, x, y, out_size, read_mpp):
    """Read a tile at given MPP (training/internal cohorts with valid metadata)."""
    arr = wsi.read_rect(
        location=(int(x), int(y)),
        size=(int(out_size), int(out_size)),
        resolution=(float(read_mpp), float(read_mpp)),
        units="mpp",
    )
    arr = ensure_rgb_u8(np.asarray(arr))
    return Image.fromarray(arr).convert("RGB")


# ---------------------------
# Optional stain normalization
# ---------------------------

_WORKER_NORM = None


def _build_normalizer(stain_method: str, target_png: str):
    if stain_method == "none":
        return None
    from tiatoolbox.tools.stainnorm import MacenkoNormalizer, ReinhardNormalizer

    target_rgb = np.asarray(Image.open(target_png).convert("RGB")).astype(np.uint8)
    if stain_method == "macenko":
        norm = MacenkoNormalizer()
    elif stain_method == "reinhard":
        norm = ReinhardNormalizer()
    else:
        raise ValueError("stain_method must be one of: none, macenko, reinhard")
    norm.fit(target_rgb)
    return norm


def _worker_init(worker_id: int, stain_method: str, target_png: str, *args, **kwargs):
    global _WORKER_NORM
    if stain_method == "none":
        _WORKER_NORM = None
        return
    _WORKER_NORM = _build_normalizer(stain_method, target_png)


def _get_normalizer(fallback_norm=None):
    # In workers: use global singleton; in main (num_workers=0): dataset holds fallback_norm
    return _WORKER_NORM if _WORKER_NORM is not None else fallback_norm


def apply_stainnorm_pil(img: Image.Image, norm) -> Image.Image:
    if norm is None:
        return img
    arr = np.asarray(img).astype(np.uint8)
    arr = norm.transform(arr).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


# ---------------------------
# Dataset
# ---------------------------


class TilesDataset(Dataset):
    def __init__(
        self,
        wsi,
        df_keep: pd.DataFrame,
        xcol: str,
        ycol: str,
        wcol: str | None,
        hcol: str | None,
        read_mode: str,
        read_mpp: float,
        out_size: int,
        eval_transform,
        stain_method: str = "none",
        stain_target_png: str | None = None,
        norm_main=None,
    ):
        self.wsi = wsi
        self.df = df_keep
        self.xcol, self.ycol = xcol, ycol
        self.wcol, self.hcol = wcol, hcol
        self.read_mode = read_mode
        self.read_mpp = read_mpp
        self.out_size = out_size
        self.eval_transform = eval_transform
        self.stain_method = stain_method
        self.stain_target_png = stain_target_png
        self.norm_main = norm_main  # used only when num_workers=0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]

        if self.read_mode == "level0":
            img = read_tile_level0(
                self.wsi,
                r[self.xcol],
                r[self.ycol],
                r[self.wcol],
                r[self.hcol],
                self.out_size,
            )
        elif self.read_mode == "mpp":
            img = read_tile_mpp(
                self.wsi,
                r[self.xcol],
                r[self.ycol],
                self.out_size,
                self.read_mpp,
            )
        else:
            raise RuntimeError(f"Unsupported read_mode: {self.read_mode}")

        # Stain norm (after read, before model transform)
        norm = _get_normalizer(self.norm_main)
        if self.stain_method != "none":
            img = apply_stainnorm_pil(img, norm)

        out = self.eval_transform(img)
        if isinstance(out, dict) and "pixel_values" in out:
            x_t = out["pixel_values"]
        elif isinstance(out, torch.Tensor):
            x_t = out
        else:
            raise TypeError(f"Tipo output transform non gestito: {type(out)}")

        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)
        return x_t[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svs", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--read_mpp", type=float, default=0.5)
    ap.add_argument("--out_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")

    # Read mode: auto chooses level0 if w0/h0 exist, else mpp.
    ap.add_argument(
        "--read_mode",
        choices=["auto", "level0", "mpp"],
        default="auto",
        help="Tile read mode. 'auto' uses level0 if parquet has w0/h0, else mpp.",
    )

    # QC
    ap.add_argument("--tissue_thr", type=float, default=0.35)
    ap.add_argument("--mean_thr", type=float, default=0.50)
    ap.add_argument("--std_thr", type=float, default=0.08)

    # Stain normalization (recommended for external cohort)
    ap.add_argument(
        "--stain_method",
        choices=["none", "macenko", "reinhard"],
        default="none",
        help="Optional stain normalization applied per tile before model transform.",
    )
    ap.add_argument(
        "--stain_target_png",
        default=None,
        help="PNG path for stain normalization target (e.g., ref_target_mosaic.png). Required if stain_method!=none.",
    )

    args = ap.parse_args()

    if args.stain_method != "none" and not args.stain_target_png:
        raise SystemExit("[ERR] --stain_target_png is required when --stain_method != none")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slide_id = Path(args.svs).stem
    out_pt = outdir / f"{slide_id}.pt"
    out_meta = outdir / f"{slide_id}.json"

    # --- load parquet ---
    df = pd.read_parquet(args.parquet).copy()
    xcol, ycol = pick_xy_cols(df)
    wcol, hcol = pick_wh_cols(df)
    idcol = pick_id_col(df)
    istologico = str(df[idcol].iloc[0]) if idcol else ""

    # Read-mode resolution
    if args.read_mode == "auto":
        read_mode = "level0" if (wcol is not None and hcol is not None) else "mpp"
    else:
        read_mode = args.read_mode

    if read_mode == "level0" and (wcol is None or hcol is None):
        raise RuntimeError(
            "read_mode=level0 requires w/h columns in parquet (e.g., w0/h0). "
            f"Columns={list(df.columns)}"
        )

    # basic filter
    if "tissue_frac" in df.columns:
        df = df[df["tissue_frac"].fillna(0) >= args.tissue_thr].copy()
    df = df.reset_index(drop=True)
    n_raw = len(df)

    # --- idempotency ---
    params = dict(
        model="MahmoodLab/TITAN_CONCH",
        read_mode=read_mode,
        read_mpp=args.read_mpp,
        out_size=args.out_size,
        tissue_thr=args.tissue_thr,
        mean_thr=args.mean_thr,
        std_thr=args.std_thr,
        stain_method=args.stain_method,
        stain_target_png=str(args.stain_target_png) if args.stain_target_png else None,
    )
    params_hash = hash_params(params)
    if out_meta.exists() and out_pt.exists():
        try:
            meta_old = json.loads(out_meta.read_text())
            if meta_old.get("params_hash") == params_hash and meta_old.get("n_tiles", 0) > 0:
                print(f"[SKIP] {slide_id} already extracted with same params.")
                return
        except Exception:
            pass

    # --- WSI ---
    wsi = WSIReader.open(args.svs)

    # --- QC mean/std on raw tiles (before stain norm) ---
    keep_mask = np.zeros(n_raw, dtype=bool)
    means = np.zeros(n_raw, dtype=np.float32)
    stds = np.zeros(n_raw, dtype=np.float32)

    print(f"[INFO] QC on {n_raw} tiles for slide {slide_id} (read_mode={read_mode})...")
    for i in range(n_raw):
        r = df.iloc[i]
        if read_mode == "level0":
            img = read_tile_level0(wsi, r[xcol], r[ycol], r[wcol], r[hcol], args.out_size)
            arr = np.asarray(img).astype(np.uint8)
        else:
            arr = wsi.read_rect(
                location=(int(r[xcol]), int(r[ycol])),
                size=(args.out_size, args.out_size),
                resolution=(args.read_mpp, args.read_mpp),
                units="mpp",
            )
            arr = ensure_rgb_u8(np.asarray(arr))

        mu, sd = qc_rgb_mean_std(arr)
        means[i], stds[i] = mu, sd
        keep_mask[i] = (mu >= args.mean_thr) and (sd >= args.std_thr)

    row_idx = np.flatnonzero(keep_mask).astype(np.int32)
    if row_idx.size == 0:
        print(f"[WARN] No tiles passed QC for {slide_id}. Saving empty container.")
        torch.save(
            {
                "feats": torch.empty(0, 768),
                "row_idx": torch.empty(0, dtype=torch.int32),
                "parquet_path": str(Path(args.parquet).resolve()),
                "parquet_sha1": sha1_of_file(args.parquet),
                "slide_id": slide_id,
                "istologico": istologico,
            },
            out_pt,
        )
        out_meta.write_text(
            json.dumps(
                {
                    "slide_id": slide_id,
                    "svs": str(Path(args.svs).resolve()),
                    "parquet": str(Path(args.parquet).resolve()),
                    "n_raw": int(n_raw),
                    "n_tiles": 0,
                    "qc_rate_kept": 0.0,
                    "params": params,
                    "params_hash": params_hash,
                },
                indent=2,
            )
        )
        return

    print(
        f"[INFO] QC kept {row_idx.size}/{n_raw} tiles "
        f"({row_idx.size / max(1, n_raw):.3f} ratio)"
    )
    print(
        f"[INFO] mean(mean_rgb_kept)={means[keep_mask].mean():.3f}, "
        f"mean(std_rgb_kept)={stds[keep_mask].mean():.3f}"
    )

    df_keep = df.loc[row_idx].reset_index(drop=True)

    # --- model ---
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[INFO] Using device: {device}")
    conch, eval_transform = load_titan_conch(device, local_files_only=True)

    # stain normalizer setup
    norm_main = None
    worker_init_fn = None
    if args.stain_method != "none":
        if args.num_workers == 0:
            norm_main = _build_normalizer(args.stain_method, args.stain_target_png)
        else:
            worker_init_fn = partial(_worker_init, stain_method=args.stain_method, target_png=args.stain_target_png)

    # --- DataLoader ---
    ds = TilesDataset(
        wsi=wsi,
        df_keep=df_keep,
        xcol=xcol,
        ycol=ycol,
        wcol=wcol,
        hcol=hcol,
        read_mode=read_mode,
        read_mpp=args.read_mpp,
        out_size=args.out_size,
        eval_transform=eval_transform,
        stain_method=args.stain_method,
        stain_target_png=args.stain_target_png,
        norm_main=norm_main,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else 2,
        worker_init_fn=worker_init_fn,
    )

    # --- forward ---
    t0 = time.time()
    feats_chunks = []
    n_seen = 0
    print(f"[INFO] Running CONCH on {row_idx.size} tiles...")
    with torch.inference_mode():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f = conch(xb)
            else:
                f = conch(xb)
            feats_chunks.append(f.float().cpu())
            n_seen += xb.size(0)
            if n_seen % 2000 == 0:
                print(f"  [INFO] processed {n_seen}/{row_idx.size} tiles...")

    t1 = time.time()
    feats = torch.cat(feats_chunks, dim=0)

    out_obj = {
        "feats": feats.half(),
        "row_idx": torch.from_numpy(row_idx),
        "parquet_path": str(Path(args.parquet).resolve()),
        "parquet_sha1": sha1_of_file(args.parquet),
        "slide_id": slide_id,
        "istologico": istologico,
    }
    torch.save(out_obj, out_pt)

    meta = {
        "slide_id": slide_id,
        "svs": str(Path(args.svs).resolve()),
        "parquet": str(Path(args.parquet).resolve()),
        "n_raw": int(n_raw),
        "n_tiles": int(row_idx.size),
        "feature_dim": 768,
        "times": {
            "seconds": round(t1 - t0, 3),
            "tiles_per_sec": round(row_idx.size / max(t1 - t0, 1e-6), 1),
        },
        "qc": {
            "mean_thr": args.mean_thr,
            "std_thr": args.std_thr,
            "kept_ratio": round(row_idx.size / max(1, n_raw), 4),
            "mean_rgb_mean_kept": float(means[keep_mask].mean()),
            "std_rgb_mean_kept": float(stds[keep_mask].mean()),
        },
        "params": params,
        "params_hash": params_hash,
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[OK] {slide_id}: kept {row_idx.size}/{n_raw} -> {out_pt}")


if __name__ == "__main__":
    main()
