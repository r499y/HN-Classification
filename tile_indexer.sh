#!/bin/bash
#SBATCH -J tile_index_list
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH --array=0-56%8
#SBATCH -o /hpcnfs/home/ieo7627/logs/%x_%A_%a.out
#SBATCH -e /hpcnfs/home/ieo7627/logs/%x_%A_%a.err
#SBATCH --requeue

set -e -o pipefail
mkdir -p /hpcnfs/home/ieo7627/logs

# --- CONFIG ---
PYTHON_BIN="/hpcnfs/scratch/LN/Nicolo_envs/crc-tia/bin/python"
SCRIPT="/hpcnfs/home/ieo7627/tile_indexer_fixed.py"
CSV_MAPPING="/hpcnfs/home/ieo7627/isto-svs-cs3.csv"
WSI_DIR="/hpcnfs/data/LN/P_LN_MITICO/MITICO_01122025"
OUT_DIR="/hpcnfs/data/LN/P_LN_MITICO/parquet_TITAN"

TILE_PX=512; TILE_MPP=0.5; STRIDE=512
MASK_MPP=4.0; TISSUE_THR=0.4

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# --- PRECHECK ---
[[ -x "$PYTHON_BIN" ]] || { echo "[ERR] python env non eseguibile: $PYTHON_BIN" >&2; exit 90; }
[[ -f "$SCRIPT"      ]] || { echo "[ERR] script non trovato: $SCRIPT" >&2; exit 1; }
[[ -f "$CSV_MAPPING" ]] || { echo "[ERR] CSV non trovato: $CSV_MAPPING" >&2; exit 1; }
[[ -d "$WSI_DIR"     ]] || { echo "[ERR] WSI_DIR non esiste: $WSI_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

# --- BOUND CHECK su CSV (righe totali) ---
TOTAL=$(wc -l < "$CSV_MAPPING")
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "[SKIP] Index ${SLURM_ARRAY_TASK_ID} >= $TOTAL"
  exit 0
fi

# --- PARSE riga i-esima (1-based) ---
LINE=$((SLURM_ARRAY_TASK_ID + 1))
ROW=$(sed -n "${LINE}p" "$CSV_MAPPING")
ROW=${ROW#$'\xEF\xBB\xBF'}                      # strip BOM se presente
FILENAME=$(echo "$ROW" | cut -d';' -f1 | xargs) # campo prima di ';;;'

# Se FILENAME è assoluto usalo, altrimenti prepend WSI_DIR
if [[ "$FILENAME" = /* ]]; then
  WSI_PATH="$FILENAME"
else
  WSI_PATH="${WSI_DIR%/}/${FILENAME}"
fi

[[ -n "$FILENAME" && -f "$WSI_PATH" ]] || { echo "[ERR] WSI non trovato: $WSI_PATH (riga $LINE: '$FILENAME')" >&2; exit 2; }

echo "[$(date)] START idx=${SLURM_ARRAY_TASK_ID}  WSI=${WSI_PATH}"
srun -c "${SLURM_CPUS_PER_TASK}" "$PYTHON_BIN" "$SCRIPT" \
  --wsi_path "$WSI_PATH" \
  --out_dir "$OUT_DIR" \
  --csv_mapping "$CSV_MAPPING" \
  --tile_px $TILE_PX --tile_mpp $TILE_MPP --stride $STRIDE \
  --mask_res_mpp $MASK_MPP --tissue_thr $TISSUE_THR
rc=$?
echo "[$(date)] DONE  idx=${SLURM_ARRAY_TASK_ID}  rc=${rc}"
exit $rc
