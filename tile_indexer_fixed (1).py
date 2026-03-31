#!/usr/bin/env python3
"""
Tile Indexer.
Outputs per-slide Parquet (or CSV) with tile coords & QC, plus a global manifest.
Author: Nicolò Raganato
"""
import argparse, os, re, hashlib, math, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd


import pyarrow as pa
import pyarrow.parquet as pq
HAVE_ARROW = True


from PIL import Image
Image.MAX_IMAGE_PIXELS = None



from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools.tissuemask import OtsuTissueMasker

warnings.filterwarnings("ignore", category=UserWarning)

EXTS = {".svs"}


def md5_of_file(path, chunk=1024*1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def get_mpp_xy(wsi):
    mpp = wsi.info.mpp
    if isinstance(mpp, (tuple, list, np.ndarray)):
        return float(mpp[0]), float(mpp[-1])
    return float(mpp), float(mpp)

def read_lowres_rgb(wsi, res_mpp, W0, H0):
    mppx, mppy = get_mpp_xy(wsi)
    Wm = int(math.ceil(W0 * (mppx / res_mpp)))
    Hm = int(math.ceil(H0 * (mppy / res_mpp)))
    img = wsi.read_rect(location=(0, 0), size=(Wm, Hm), resolution=res_mpp, units="mpp")
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr

def build_mask(wsi, res_mpp=4.0):
    """Return low-res RGB, boolean mask, and scale factors baseline->mask-grid."""
    W0, H0 = wsi.info.level_dimensions[0]
    mppx, mppy = get_mpp_xy(wsi)
    Wm = int(math.ceil(W0 * (mppx / res_mpp)))
    Hm = int(math.ceil(H0 * (mppy / res_mpp)))
    img_low = wsi.read_rect(location=(0, 0), size=(Wm, Hm), resolution=res_mpp, units="mpp")
    img_low = np.asarray(img_low)
    if img_low.ndim == 2:
        img_low = np.stack([img_low, img_low, img_low], axis=-1)
    if img_low.shape[-1] == 4:
        img_low = img_low[:, :, :3]
    mask = OtsuTissueMasker().fit_transform([img_low])[0].astype(bool)
    sx = Wm / W0
    sy = Hm / H0
    return img_low, mask, sx, sy

#erode rimossa perche stabilito che tralascia la maggiorparte del tessuto->inutile
def grid_params(wsi, tile_px, tile_mpp, stride):
    """Compute baseline-level (level-0) tile width/height and strides accounting for anisotropy."""
    W0, H0 = wsi.info.level_dimensions[0]
    mppx, mppy = get_mpp_xy(wsi)
    tw0 = int(round(tile_px  * (tile_mpp / mppx)))
    th0 = int(round(tile_px  * (tile_mpp / mppy)))
    dx0 = int(round(stride   * (tile_mpp / mppx)))
    dy0 = int(round(stride   * (tile_mpp / mppy)))
    tw0 = max(1, tw0); th0 = max(1, th0); dx0 = max(1, dx0); dy0 = max(1, dy0)
    return (W0, H0, tw0, th0, dx0, dy0)

def rect_on_mask_coords(x0, y0, tw0, th0, sx, sy):
    """Map baseline rect to mask-grid integer rect (clipped)."""
    x = int(round(x0 * sx)); y = int(round(y0 * sy))
    w = int(round(tw0 * sx)); h = int(round(th0 * sy))
    return x, y, w, h

def tile_qc_from_lowres(img_low, mask, rect_mask):
    """Compute QC on mask-grid window using low-res image and mask.
    Returns tissue_frac.
    """
    x, y, w, h = rect_mask
    H, W = mask.shape
    x2 = min(W, max(0, x + w)); y2 = min(H, max(0, y + h))
    x1 = min(W, max(0, x));     y1 = min(H, max(0, y))
    if x1 >= x2 or y1 >= y2:
        return 0.0
    m = mask[y1:y2, x1:x2]
    if m.size == 0: return 0.0
    tissue_frac = float(m.mean())
    #non voglio il filtro bianco quindi l ho rimosso
    return tissue_frac

def extract_histological_number(path: Path, csv_mapping=None):
    """Estrae il numero istologico dal nome del file.
    Se csv_mapping è fornito, usa la tabella di mappatura,
    altrimenti cerca pattern nel nome del file.
    """
    filename = path.name
    
    # Se abbiamo la tabella di mappatura, usala
    if csv_mapping is not None and filename in csv_mapping:
        return csv_mapping[filename]
    
    # Fallback: cerca pattern nel nome del file
    base = path.stem
    # Cerca pattern tipo H12345, h12345
    m = re.search(r"[Hh](\d+)", base)
    if m:
        return m.group(1)
    # Cerca pattern di numeri lunghi (presumibilmente numero istologico)
    m = re.search(r"(\d{4,})", base)
    if m:
        return m.group(1)
    # Se non trova nulla, restituisce None
    return None

def load_histological_mapping(csv_path):
    """Carica la mappatura file.svs -> numero istologico dal CSV"""
    if not csv_path or not Path(csv_path).exists():
        return None
    
    mapping = {}
    try:
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ';;;' in line:
                    parts = line.split(';;;')
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        hist_number = parts[1].strip()
                        mapping[filename] = hist_number
        return mapping
    except Exception as e:
        print(f"Warning: Could not load histological mapping from {csv_path}: {e}")
        return None

#come lo guessa l id della slide? prende il barcode dal xml oppure il nome del file senza il .svs?
def guess_slide_id(path: Path):
    base = path.stem
    m = re.search(r"(\d{5,})", base)
    return m.group(1) if m else base

def index_one_slide(path, out_root, tile_px=256, tile_mpp=0.5, stride=256,
                    mask_res_mpp=4.0,
                    tissue_thr=0.60,
                    write_parquet=True, csv_mapping=None):
    path = Path(path)
    slide_id = guess_slide_id(path)
    slide_md5 = md5_of_file(path)
    histological_number=extract_histological_number(path,csv_mapping)
    wsi = WSIReader.open(str(path))
    W0, H0 = wsi.info.level_dimensions[0]
    
    # lowres image + mask
    img_low, mask, sx, sy = build_mask(wsi, res_mpp=mask_res_mpp)
    Hm, Wm = mask.shape
    #rimosso erode mask
    
    W0, H0, tw0, th0, dx0, dy0 = grid_params(wsi, tile_px, tile_mpp, stride)

    # iterate grid
    records = []
    n_total = 0
    n_center_ok = 0
    n_qc_ok = 0

    y = 0
    while y + th0 <= H0:
        x = 0
        while x + tw0 <= W0:
            n_total += 1
            # center in eroded tissue?
            cx_m = int((x + tw0/2) * sx)
            cy_m = int((y + th0/2) * sy)
            #sostituito mask_e con mask
            if (0 <= cx_m < Wm) and (0 <= cy_m < Hm) and mask[cy_m, cx_m]:
                n_center_ok += 1
            rect_m = rect_on_mask_coords(x, y, tw0, th0, sx, sy)
            tissue_frac = tile_qc_from_lowres(img_low, mask, rect_m)
            #filtro white rimosso
            qc_ok = (tissue_frac >= tissue_thr)
            if qc_ok:
                n_qc_ok += 1
                tile_id = f"{slide_md5[:8]}_{x}_{y}_{tw0}x{th0}"
                #rimossi da rec tutti i parametri relativi a erode e white
                rec = dict(
                    slide_id=slide_id, slide_md5=slide_md5,histological_number=histological_number,
                    x0=x, y0=y, w0=tw0, h0=th0,
                    tile_px=tile_px, tile_mpp=tile_mpp, stride=stride,
                    mask_res_mpp=mask_res_mpp,
                    tissue_frac=tissue_frac,
                    center_ok=True, qc_ok=True, tile_id=tile_id,
                    W0=W0, H0=H0,
                    filename=str(path.name), abspath=str(path)
                )
                records.append(rec)
            x += dx0
        y += dy0

    # write per-slide parquet/csv
    out_dir = Path(out_root) / "tile_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_records(records)
    out_path_parquet = out_dir / f"{slide_id}.parquet"
    out_path_csv = out_dir / f"{slide_id}.csv"

    if df.empty:
    	# se vuoto, non scrivere niente (o scrivi parquet vuoto, a scelta)
    	out_path = None

    elif HAVE_ARROW and write_parquet:
    	table = pa.Table.from_pandas(df)
    	pq.write_table(table, out_path_parquet)
    	out_path = out_path_parquet

    else:
    	raise RuntimeError("Parquet non disponibile (manca pyarrow o write_parquet=False).")


    # meta json
    #rimossi da meta tutti i parametri relativi a erode e white
    n_tiles_total = len(records)  # numero di tile che hanno passato QC
    
    meta = dict(
        slide_id=slide_id, slide_md5=slide_md5,
        filename=str(path.name), abspath=str(path),
        histological_number=histological_number,
        W0=W0, H0=H0, tile_px=tile_px, tile_mpp=tile_mpp, stride=stride,
        mask_res_mpp=mask_res_mpp,
        tissue_thr=tissue_thr,
        n_total=n_total, n_center_ok=n_center_ok, n_qc_ok=n_qc_ok,
        n_tiles_total=n_tiles_total,
        index_path=str(out_path)
    )
    meta_dir = Path(out_root) / "qc" / "meta"
    with open(meta_dir / f"{slide_id}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta, str(out_path)

def walk_wsi(dir_path):
    return sorted([p for p in Path(dir_path).rglob("*") if p.suffix.lower() in EXTS])

def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--wsi_dir", help="Directory containing .svs files")
    group.add_argument("--wsi_path", help="Single .svs file to process")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--csv_mapping", help="Path to CSV file mapping filename.svs to histological number")
    ap.add_argument("--tile_px", type=int, default=256)
    ap.add_argument("--tile_mpp", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--mask_res_mpp", type=float, default=4.0)
    ap.add_argument("--tissue_thr", type=float, default=0.60)
    ap.add_argument("--index", type=int, default=-1, help="If >=0, process only the file at that index (job arrays).")
    ap.add_argument("--no_parquet", action="store_true", help="Force CSV output instead of Parquet.")
    args = ap.parse_args()

    # LOG DIAGNOSTICO
    print(f"[CFG] wsi_dir={args.wsi_dir} wsi_path={args.wsi_path} out_dir={args.out_dir} "
          f"tile_px={args.tile_px} tile_mpp={args.tile_mpp} stride={args.stride} "
          f"mask_res_mpp={args.mask_res_mpp} tissue_thr={args.tissue_thr} no_parquet={args.no_parquet}")

    # Carica mappatura
    csv_mapping = load_histological_mapping(args.csv_mapping)
    if args.csv_mapping and csv_mapping is None:
        print(f"Warning: Could not load CSV mapping from {args.csv_mapping}", file=sys.stderr)

    # Determina i file
    if args.wsi_path:
        wsi_paths = [Path(args.wsi_path)]
        if not wsi_paths[0].exists():
            print(f"WSI file not found: {args.wsi_path}", file=sys.stderr)
            sys.exit(2)
    else:
        wsi_paths = walk_wsi(args.wsi_dir)
        if not wsi_paths:
            print(f"No WSI found in {args.wsi_dir}.", file=sys.stderr)
            sys.exit(2)

    manifest_rows = []
    out_dir = Path(args.out_dir)
    manifest_csv = out_dir / "tile_index_manifest.csv"
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.index >= 0:
        i = args.index
        if i >= len(wsi_paths):
            print(f"Index {i} out of range (n={len(wsi_paths)}).", file=sys.stderr)
            sys.exit(3)
        wsi_targets = [wsi_paths[i]]
    else:
        wsi_targets = wsi_paths

    for p in wsi_targets:
        try:
            meta, out_path = index_one_slide(
                p, str(out_dir),
                tile_px=args.tile_px, tile_mpp=args.tile_mpp, stride=args.stride,
                mask_res_mpp=args.mask_res_mpp,
                tissue_thr=args.tissue_thr,
                write_parquet=(not args.no_parquet),
                csv_mapping=csv_mapping
            )
            manifest_rows.append(meta)
        except Exception as e:
            # lascia traccia chiara in stderr
            print(f"[ERROR] {p}: {e}", file=sys.stderr)

    if not manifest_rows:
        print("Nessuna slide indicizzata (manifest vuoto).", file=sys.stderr)
        sys.exit(4)

    mdf = pd.DataFrame(manifest_rows)
    if manifest_csv.exists():
        try:
            old = pd.read_csv(manifest_csv)
            mdf = pd.concat([old, mdf]).drop_duplicates(subset=["slide_md5"]).reset_index(drop=True)
        except Exception as e:
            print(f"Warning: manifest precedente corrotto/non leggibile: {e}", file=sys.stderr)

    mdf.to_csv(manifest_csv, index=False)
    print(f"Wrote manifest: {manifest_csv} ({len(mdf)} slides indexed)")


if __name__ == "__main__":
    main()



