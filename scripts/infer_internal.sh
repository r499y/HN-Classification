#!/bin/bash
set -euo pipefail

# Esegue il notebook di inferenza interna (held-out interno).

NB_IN="${NB_IN:-notebooks/03_case_studies_heatmaps.ipynb}"
NB_OUT="${NB_OUT:-notebooks/03_case_studies_heatmaps.executed.ipynb}"

if ! command -v jupyter >/dev/null 2>&1; then
  echo "[ERR] jupyter non trovato nell'ambiente corrente." >&2
  exit 1
fi

echo "[INFO] Running internal inference notebook: ${NB_IN}"
jupyter nbconvert \
  --to notebook \
  --execute "${NB_IN}" \
  --output "$(basename "${NB_OUT}")" \
  --output-dir "$(dirname "${NB_OUT}")"

echo "[OK] Internal inference notebook eseguito: ${NB_OUT}"
