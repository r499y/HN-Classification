# HN-Classification

Repository per pipeline deep learning su WSI CRC (task HN vs LN).

## Struttura
- `src/`: codice sorgente (data, models, training, inference, eval, utils)
- `scripts/`: launcher shell (HPC + esecuzione notebook)
- `configs/`: configurazioni di base
- `notebooks/`: notebook originali di training/inference/figure
- `docs/`: documentazione e materiali tesi

## Prerequisiti
Python `>=3.10`.

Installazione dipendenze:
```bash
pip install -e .
# oppure:
pip install -r requirements.txt
```

## Esecuzione pipeline

### 1) Tile indexing
```bash
python src/data/manifests.py --help
```
oppure launcher HPC:
```bash
scripts/run_indexing.sh
```

### 2) Feature extraction (internal/external)
```bash
python src/inference/predict_internal.py --help
python src/inference/predict_external_tcga.py --help
```
oppure launcher HPC:
```bash
scripts/run_features_internal.sh
scripts/run_features_external.sh
```

### 3) Training / Inference notebook-driven
```bash
scripts/train_cv.sh
scripts/infer_internal.sh
scripts/infer_tcga.sh
```

## File/asset necessari (non versionati qui)
Per un run end-to-end servono anche i file dati/manifest usati nei notebook/script:
- manifest CSV pazienti (training/test),
- file `.pt` di feature per slide,
- WSI `.svs`,
- eventuali checkpoint `.pth`,
- file di supporto per rescaling/calibration (es. `logit_rescaling_stats.csv`),
- asset stain target per esterno (quando richiesto).

Vedi `docs/data_contracts.md` e `docs/pipeline.md` per i dettagli operativi.
