# WA-STGAT

Anonymous implementation accompanying the submitted manuscript:
**"WA-STGAT: A Context-Aware Spatio-Temporal Graph Architecture for Dynamic Urban Mobility Pricing"**

## Overview

WA-STGAT dynamically modulates spatial edge weights in a ride-hailing marketplace graph using exogenous environmental signals. This artifact includes the model architecture, experiment configuration, preprocessing pipeline, and constrained-supply marketplace dispatch simulator used to estimate preserved Gross Merchandise Value (GMV).

The default configuration is intentionally small and reproducible. It downloads public NYC TLC yellow taxi trip records and historical weather data, builds a 15-minute spatio-temporal demand tensor over taxi zones, trains WA-STGAT, and evaluates the dispatch simulator.

## Repository Layout

```text
configs/default.yaml      Experiment configuration
scripts/preprocess.py     Public data download and feature construction
scripts/run_experiment.sh End-to-end reproducibility script
src/model.py              WA-STGAT architecture and baseline module
src/train.py              Training entry point
src/evaluate.py           Evaluation and marketplace simulation
src/utils.py              Dataset and dataloader utilities
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing Results

Run the full pipeline from the repository root:

```bash
bash scripts/run_experiment.sh
```

Or run the stages independently:

```bash
python scripts/preprocess.py --config configs/default.yaml
python src/train.py --config configs/default.yaml
python src/evaluate.py --config configs/default.yaml
```

The preprocessing step downloads public NYC TLC data when the configured parquet file is not present locally. Generated data tensors, graph files, checkpoints, and raw parquet files are ignored by git.

## Configuration

The default artifact settings are in `configs/default.yaml`. The included demo window covers January 1-3, 2024, with a small graph-construction threshold suitable for fast reviewer reproduction.

## Anonymity Note

This repository is prepared for anonymous peer review. Please do not add author names, institutional paths, personal email addresses, or acknowledgements until the review process no longer requires anonymity.
