#!/usr/bin/env python3
import argparse, json, hashlib, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel

from tiatoolbox.wsicore.wsireader import WSIReader


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
        ("x", "y"),
        ("x0", "y0"),
        ("X", "Y"),
        ("loc_x", "loc_y"),
        ("center_x", "center_y"),
        ("cx", "cy"),
    ]:
        if a in df.columns and b in df.columns:
            return a, b
    raise RuntimeError(f"x/y columns not found in parquet: {list(df.columns)}")


def pick_id_col(df):
    for c in ["istologico", "patient_id", "case_id", "slide_id", "sample_id"]:
        if c in df.columns:
            return c
    return None  # optional


def qc_rgb_mean_std(arr_u8):
    a = arr_u8.astype(np.float32) / 255.0
    return float(a.mean()), float(a.std())


class TilesDataset(Dataset):
    def __init__(self, wsi, coords, read_mpp, out_size, eval_transform):
        self.wsi = wsi
        self.coords = coords
        self.read_mpp = read_mpp
        self.out_size = out_size
        self.eval_transform = eval_transform  # di CONCH/TITAN

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        x, y = self.coords[i]
        arr = self.wsi.read_rect(
            location=(int(x), int(y)),
            size=(self.out_size, self.out_size),
            resolution=(self.read_mpp, self.read_mpp),
            units="mpp",
        )

        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.shape[2] >= 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr).convert("RGB")

        out = self.eval_transform(img)
        if isinstance(out, dict) and "pixel_values" in out:
            x_t = out["pixel_values"]
        elif isinstance(out, torch.Tensor):
            x_t = out
        else:
            raise TypeError(f"Tipo output transform non gestito: {type(out)}")

        if x_t.ndim == 3:
            x_t = x_t.unsqueeze(0)  # [1, C, H, W]

        return x_t[0]  # [C, H, W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svs", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--read_mpp", type=float, default=0.5)
    # PER TITAN/CONCH: patch a 20x da 512x512
    ap.add_argument("--out_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    # QC semplice
    ap.add_argument("--tissue_thr", type=float, default=0.35)
    ap.add_argument("--mean_thr", type=float, default=0.50)
    ap.add_argument("--std_thr", type=float, default=0.08)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slide_id = Path(args.svs).stem
    out_pt = outdir / f"{slide_id}.pt"
    out_meta = outdir / f"{slide_id}.json"

    # --- carica parquet e prepara indici ---
    df = pd.read_parquet(args.parquet).copy()
    xcol, ycol = pick_xy_cols(df)
    idcol = pick_id_col(df)
    istologico = str(df[idcol].iloc[0]) if idcol else ""

    if "tissue_frac" in df.columns:
        df = df[df["tissue_frac"].fillna(0) >= args.tissue_thr].copy()

    df = df.reset_index(drop=True)
    n_raw = len(df)

    # --- idempotenza su params ---
    params = dict(
        model="MahmoodLab/TITAN_CONCH",
        read_mpp=args.read_mpp,
        out_size=args.out_size,
        tissue_thr=args.tissue_thr,
        mean_thr=args.mean_thr,
        std_thr=args.std_thr,
    )
    params_hash = hash_params(params)
    if out_meta.exists() and out_pt.exists():
        try:
            meta_old = json.loads(out_meta.read_text())
            if (
                meta_old.get("params_hash") == params_hash
                and meta_old.get("n_tiles", 0) > 0
            ):
                print(f"[SKIP] {slide_id} already extracted with same params.")
                return
        except Exception:
            pass

    # --- WSI reader ---
    wsi = WSIReader.open(args.svs)

    # --- QC mean/std RGB ---
    keep_mask = np.zeros(n_raw, dtype=bool)
    means = np.zeros(n_raw, dtype=np.float32)
    stds = np.zeros(n_raw, dtype=np.float32)

    print(f"[INFO] QC on {n_raw} tiles for slide {slide_id}...")
    for i in range(n_raw):
        r = df.iloc[i]
        arr = wsi.read_rect(
            location=(int(r[xcol]), int(r[ycol])),
            size=(args.out_size, args.out_size),
            resolution=(args.read_mpp, args.read_mpp),
            units="mpp",
        )
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.shape[2] >= 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
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
    coords = list(
        zip(
            df_keep[xcol].astype(int).tolist(),
            df_keep[ycol].astype(int).tolist(),
        )
    )

    # --- modello TITAN+CONCH ---
    device = (
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[INFO] Using device: {device}")
    conch, eval_transform = load_titan_conch(device, local_files_only=True)

    # --- DataLoader ---
    ds = TilesDataset(
        wsi=wsi,
        coords=coords,
        read_mpp=args.read_mpp,
        out_size=args.out_size,
        eval_transform=eval_transform,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else 2,
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
                    f = conch(xb)  # (B,768)
            else:
                f = conch(xb)
            feats_chunks.append(f.float().cpu())
            n_seen += xb.size(0)
            if n_seen % 500 == 0:
                print(f"  [INFO] processed {n_seen}/{row_idx.size} tiles...")

    t1 = time.time()
    feats = torch.cat(feats_chunks, dim=0)  # (N_keep,768)

    out_obj = {
        "feats": feats.half(),  # fp16
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
    print(f"[OK] {slide_id}: kept {row_idx.size}/{n_raw} → {out_pt}")


if __name__ == "__main__":
    main()

