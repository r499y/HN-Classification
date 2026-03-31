# Data Contracts

## Tile index parquet
Expected columns (auto-detected where possible):
- coordinates: `x,y` or `x0,y0` (+ optional `w0,h0`)
- `tissue_frac` (optional but recommended)
- identifiers: `patient_id`/`slide_id`/`istologico` (optional set)

## Feature tensor (`.pt`)
Dictionary keys used by current extractors:
- `feats` (N, D)
- `row_idx`
- `parquet_path`
- `parquet_sha1`
- `slide_id`
- `istologico`

## Metadata (`.json`)
- params + params_hash
- n_raw / n_tiles
- timing and QC diagnostics
