#!/bin/bash
#SBATCH -p medium
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-9%2
#SBATCH -o /hpcnfs/home/ieo7627/logs/%x_%A_%a.out
#SBATCH -e /hpcnfs/home/ieo7627/logs/%x_%A_%a.err
#SBATCH --job-name=fe_titan_tcga

set -e -o pipefail
trap 'echo "[ERR] line $LINENO rc=$? cmd: $BASH_COMMAND" >&2' ERR
export PYTHONUNBUFFERED=1

# =============================
# ENV / PYTHON
# =============================
ENV_PREFIX="/hpcnfs/scratch/LN/Nicolo_envs/crc-tia"
PY="$ENV_PREFIX/bin/python"
[ -x "$PY" ] || { echo "[ERR] python non trovato: $PY"; exit 12; }

: "${SLURM_CPUS_PER_TASK:=16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# =============================
# INPUTS
# TSV format (3 columns): slide_id<TAB>svs_path<TAB>parquet_path
# Recommended: generate from tile_index_manifest.csv after indexing.
# =============================
LIST="/hpcnfs/home/ieo7627/slides_for_feature_extraction.tsv"

# Where to write per-slide .pt + .json
OUTDIR="/hpcscratch/ieo/ieo7627/features/more_6"

# Feature extractor script (patched for external cohorts)
SCRIPT="/hpcnfs/home/ieo7627/extract_features_titan_external_v3.py"

# =============================
# READ / MODEL SETTINGS
# =============================
READ_MPP=0.5
OUT_SIZE=512
BATCH_SIZE=2048
NUM_WORKERS="${SLURM_CPUS_PER_TASK}"

# QC (applied BEFORE stain normalization)
TISSUE_THR=0.35
MEAN_THR=0.50
STD_THR=0.08

# =============================
# STAIN NORMALIZATION (external recommended)
# =============================
STAIN_METHOD="macenko"  # none | macenko | reinhard
STAIN_TARGET_PNG="/hpcnfs/home/ieo7627/out_stainref/ref_target_mosaic.png"

# =============================
# PICK LINE
# =============================
: "${SLURM_ARRAY_TASK_ID:=0}"
[ -f "$LIST" ] || { echo "[ERR] slides.tsv mancante: $LIST"; exit 20; }

# If the file has a header, we start from line 2, else from line 1.
FIRST_LINE=$(head -n 1 "$LIST" || true)
START=1
if echo "$FIRST_LINE" | grep -qiE '^slide_id\t'; then
  START=2
fi
LINE_NO=$((SLURM_ARRAY_TASK_ID + START))
LINE=$(sed -n "${LINE_NO}p" "$LIST" || true)

# If the task id is beyond file length, exit cleanly (useful with oversized arrays)
if [ -z "$LINE" ]; then
  echo "[SKIP] task_id=${SLURM_ARRAY_TASK_ID} beyond list length (line ${LINE_NO} empty)."
  exit 0
fi

IFS=$'\t' read -r SLIDE_ID SVS PARQ <<< "$LINE"

[ -f "$SVS" ]  || { echo "[ERR] WSI non trovata: $SVS"; exit 30; }
[ -f "$PARQ" ] || { echo "[ERR] Parquet non trovato: $PARQ"; exit 31; }

mkdir -p "$OUTDIR"

echo "[START] $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID slide=$SLIDE_ID host=$(hostname)"
nvidia-smi || echo "[WARN] nvidia-smi fallita (GPU non visibile?)"
"$PY" -c "import sys,torch; print('exe:',sys.executable); print('cuda:',torch.cuda.is_available())" || true

# -----------------------------
# RUN
# -----------------------------
CMD=(
  "$PY" -u "$SCRIPT"
  --svs "$SVS"
  --parquet "$PARQ"
  --outdir "$OUTDIR"
  --read_mode auto
  --read_mpp "$READ_MPP"
  --out_size "$OUT_SIZE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --tissue_thr "$TISSUE_THR"
  --mean_thr "$MEAN_THR"
  --std_thr "$STD_THR"
)

if [ "$STAIN_METHOD" != "none" ]; then
  CMD+=( --stain_method "$STAIN_METHOD" --stain_target_png "$STAIN_TARGET_PNG" )
fi

echo "[CMD] ${CMD[@]}"

srun --gpu-bind=none "${CMD[@]}"
rc=$?

echo "[DONE]  $(date) slide=$SLIDE_ID rc=$rc"
exit $rc
