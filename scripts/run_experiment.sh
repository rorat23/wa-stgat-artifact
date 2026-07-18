#!/bin/bash
set -e


# CRITICAL: Add the repository root directory to the Python path
export PYTHONPATH=.


echo "Starting WA-STGAT Reproducibility Pipeline..."


echo ">> 1. Preprocessing Data"
python scripts/preprocess.py --config configs/default.yaml


echo ">> 2. Training Model"
python src/train.py --config configs/default.yaml


echo ">> 3. Evaluating Economic Impact"
python src/evaluate.py --config configs/default.yaml


echo "Pipeline Execution Complete!"


