# Pipeline

## 1) Indexing WSI -> tile index
- Entry point: `src/data/manifests.py`
- Input: cartella WSI (`--wsi_dir`) o singolo file (`--wsi_path`)
- Output:
  - `tile_index/*.parquet` (o CSV con `--no_parquet`)
  - `qc/meta/*.json`
  - `tile_index_manifest.csv`

## 2) Feature extraction (internal)
- Entry point: `src/inference/predict_internal.py`
- Input:
  - `--svs`
  - `--parquet`
- Output:
  - `*.pt` (feature tensor + row_idx)
  - `*.json` (metadata/QC)

## 3) Feature extraction (external / TCGA)
- Entry point: `src/inference/predict_external_tcga.py`
- Extra: `--read_mode auto|level0|mpp` + opzionale stain normalization.

## 4) Training / model selection
- Funzioni core estratte nei moduli:
  - `src/data/patient_dataset.py`
  - `src/models/*`
  - `src/training/*`
- Esecuzione pratica corrente:
  - notebook `notebooks/01_exploration.ipynb`
  - script `scripts/train_cv.sh`

## 5) Inference interna/esterna e figure
- Notebook:
  - `notebooks/03_case_studies_heatmaps.ipynb` (internal/interpretabilità)
  - `notebooks/02_results_figures.ipynb` (external/results)
- Script:
  - `scripts/infer_internal.sh`
  - `scripts/infer_tcga.sh`

## Known gaps per nuovi utilizzatori
- Path HPC hardcoded nei launcher storici (`scripts/run_*.sh`).
- Asset esterni non versionati (manifest, checkpoint, csv di rescaling, dataset).
