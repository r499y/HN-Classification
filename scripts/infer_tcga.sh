#!/bin/bash
set -euo pipefail

# Esegue il notebook di inferenza esterna/risultati (TCGA).

NB_IN="${NB_IN:-notebooks/02_results_figures.ipynb}"
NB_OUT="${NB_OUT:-notebooks/02_results_figures.executed.ipynb}"

if ! command -v jupyter >/dev/null 2>&1; then
  echo "[ERR] jupyter non trovato nell'ambiente corrente." >&2
  exit 1
fi

echo "[INFO] Running TCGA inference notebook: ${NB_IN}"
jupyter nbconvert \
  --to notebook \
  --execute "${NB_IN}" \
  --output "$(basename "${NB_OUT}")" \
  --output-dir "$(dirname "${NB_OUT}")"

echo "[OK] TCGA inference notebook eseguito: ${NB_OUT}"
