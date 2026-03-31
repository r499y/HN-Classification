# HN-Classification

Repository ristrutturata per pipeline deep learning su WSI CRC (task HN vs LN).

## Struttura
- `src/`: codice sorgente modulare.
- `scripts/`: launcher shell/HPC.
- `configs/`: configurazioni (placeholder + base).
- `notebooks/`: notebook esplorativi e figure.
- `docs/`: documentazione pipeline/contratti/esperimenti.

## Stato attuale
Questo refactor usa **solo codice già presente in repo**. Dove mancava codice separato (es. training engine modulare), sono stati inseriti placeholder espliciti da popolare successivamente.

## Quickstart (bozza)
1. Indicizzazione tile: `scripts/run_indexing.sh`
2. Feature extraction internal: `scripts/run_features_internal.sh`
3. Feature extraction external: `scripts/run_features_external.sh`
4. Training/inference: moduli in `src/training` e `src/inference` (placeholder + codice estratto)

## Nota
I path HPC hardcoded nei launcher sono stati mantenuti come riferimento operativo originale e andranno parametrizzati nei prossimi step.
