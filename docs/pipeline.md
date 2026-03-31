# Pipeline

1. Tile indexing (`src/data/manifests.py`)
2. Feature extraction internal (`src/inference/predict_internal.py`)
3. Feature extraction external (`src/inference/predict_external_tcga.py`)
4. Training / validation (placeholder modules in `src/training`)
5. Inference + evaluation (placeholders in `src/inference`, `src/eval`)

> Nota: training/eval modulari sono stati scaffoldati e vanno popolati con codice già disponibile nei tuoi file locali non ancora caricati.
