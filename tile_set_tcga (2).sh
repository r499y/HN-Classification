#!/bin/bash
#SBATCH -J tile_index_tcga
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --array=0-9%2
#SBATCH -o /hpcnfs/home/ieo7627/logs/%x_%A_%a.out
#SBATCH -e /hpcnfs/home/ieo7627/logs/%x_%A_%a.err
#SBATCH --requeue

set -euo pipefail
mkdir -p /hpcnfs/home/ieo7627/logs

# --- CONFIG ---
PYTHON_BIN="/hpcnfs/scratch/LN/Nicolo_envs/crc-tia/bin/python"
SCRIPT="/hpcnfs/home/ieo7627/tile_indexer_fixed_v3.py"

# Root that contains many subfolders, one .svs each
EXTERNAL_ROOT="/hpcscratch/ieo/ieo7627/TCGA"

# One-time generated list of absolute .svs paths (one per line)
SVS_LIST="/hpcnfs/home/ieo7627/svs_list_missing_6.txt"

# Output directory (keep it separate from training)
OUT_DIR="/hpcscratch/ieo/ieo7627/parquet_TITAN_TCGA/more_6"

TILE_PX=512; TILE_MPP=0.5; STRIDE=512
MASK_MPP=4.0; TISSUE_THR=0.40
FALLBACK_MPP=0.252
FALLBACK_MAX_SIDE=2048

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- PRECHECK ---
[[ -x "$PYTHON_BIN" ]] || { echo "[ERR] python env non eseguibile: $PYTHON_BIN" >&2; exit 90; }
[[ -f "$SCRIPT"      ]] || { echo "[ERR] script non trovato: $SCRIPT" >&2; exit 1; }
[[ -d "$EXTERNAL_ROOT" ]] || { echo "[ERR] EXTERNAL_ROOT non esiste: $EXTERNAL_ROOT" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# --- BUILD LIST (only once) ---
if [[ ! -f "$SVS_LIST" ]]; then
  echo "[INFO] Building SVS list at $SVS_LIST"
  find "$EXTERNAL_ROOT" -type f -name "*.svs" | sort > "$SVS_LIST"
fi

TOTAL=$(wc -l < "$SVS_LIST")
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "[SKIP] Index ${SLURM_ARRAY_TASK_ID} >= $TOTAL"
  exit 0
fi

LINE=$((SLURM_ARRAY_TASK_ID + 1))
WSI_PATH=$(sed -n "${LINE}p" "$SVS_LIST")
[[ -n "$WSI_PATH" && -f "$WSI_PATH" ]] || { echo "[ERR] WSI non trovato: $WSI_PATH" >&2; exit 2; }

echo "[$(date)] START idx=${SLURM_ARRAY_TASK_ID}  WSI=${WSI_PATH}"
srun -c "${SLURM_CPUS_PER_TASK}" "$PYTHON_BIN" "$SCRIPT" \
  --wsi_path "$WSI_PATH" \
  --out_dir "$OUT_DIR" \
  --tile_px $TILE_PX --tile_mpp $TILE_MPP --stride $STRIDE \
  --mask_res_mpp $MASK_MPP --tissue_thr $TISSUE_THR \
  --fallback_mpp $FALLBACK_MPP --fallback_max_side $FALLBACK_MAX_SIDE
rc=$?
echo "[$(date)] DONE  idx=${SLURM_ARRAY_TASK_ID}  rc=${rc}"
exit $rc
