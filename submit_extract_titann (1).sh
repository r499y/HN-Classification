#!/bin/bash
#SBATCH -p medium
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-56%4
#SBATCH -o /hpcnfs/home/ieo7627/logs/%x_%A_%a.out
#SBATCH -e /hpcnfs/home/ieo7627/logs/%x_%A_%a.err
#SBATCH --job-name=fe_lunit

set -e -o pipefail
trap 'echo "[ERR] line $LINENO rc=$? cmd: $BASH_COMMAND" >&2' ERR
export PYTHONUNBUFFERED=1


ENV_PREFIX="/hpcnfs/scratch/LN/Nicolo_envs/crc-tia"
PY="$ENV_PREFIX/bin/python"
[ -x "$PY" ] || { echo "[ERR] python non trovato: $PY"; exit 12; }


: "${SLURM_CPUS_PER_TASK:=16}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"


OUTDIR="/hpcnfs/data/LN/P_LN_MITICO/features/TITAN"
LIST="/hpcnfs/home/ieo7627/slides_titan3.tsv"
LOGDIR="/hpcnfs/home/ieo7627/logs"




: "${SLURM_ARRAY_TASK_ID:=0}"
[ -f "$LIST" ] || { echo "[ERR] slides.tsv mancante: $LIST"; exit 20; }
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+2))p" "$LIST") || { echo "[ERR] sed fallita"; exit 21; }
[ -n "$LINE" ] || { echo "[ERR] linea vuota per task_id=$SLURM_ARRAY_TASK_ID"; exit 22; }


IFS=$'\t' read -r SLIDE_ID SVS PARQ <<< "$LINE"

[ -f "$SVS" ]  || { echo "[ERR] WSI non trovata: $SVS"; exit 30; }
[ -f "$PARQ" ] || { echo "[ERR] Parquet non trovato: $PARQ"; exit 31; }

echo "[START] $(date) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID slide=$SLIDE_ID host=$(hostname)"
nvidia-smi || echo "[WARN] nvidia-smi fallita (GPU non visibile?)"
"$PY" -c "import sys,torch; print('exe:',sys.executable); print('cuda:',torch.cuda.is_available())" || true

#sed -i 's/\t/    /g' /hpcnfs/home/ieo7627/extract_features_lunit.py

srun --gpu-bind=none "$PY" -u /hpcnfs/home/ieo7627/extract_features_titan.py \
  --svs "$SVS" \
  --parquet "$PARQ" \
  --outdir "$OUTDIR" \
  --read_mpp 0.5 \
  --out_size 512 \
  --batch_size 2048 \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --tissue_thr 0.35 \
  --mean_thr 0.50 \
  --std_thr 0.08

rc=$?
echo "[DONE]  $(date) slide=$SLIDE_ID rc=$rc"
exit $rc

