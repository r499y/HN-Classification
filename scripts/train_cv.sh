#!/bin/bash
set -euo pipefail

# Esegue il notebook di training in modo riproducibile.
# Richiede jupyter installato nell'env corrente.

NB_IN="${NB_IN:-notebooks/01_exploration.ipynb}"
NB_OUT="${NB_OUT:-notebooks/01_exploration.executed.ipynb}"

if ! command -v jupyter >/dev/null 2>&1; then
  echo "[ERR] jupyter non trovato nell'ambiente corrente." >&2
  exit 1
fi

echo "[INFO] Running training notebook: ${NB_IN}"
jupyter nbconvert \
  --to notebook \
  --execute "${NB_IN}" \
  --output "$(basename "${NB_OUT}")" \
  --output-dir "$(dirname "${NB_OUT}")"

echo "[OK] Training notebook eseguito: ${NB_OUT}"
