#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Running Hyperparameter Optimisation locally..."
echo "Project directory: $PROJECT_DIR"

EXPERIMENT_NAME=${1:-"local_experiment"}
GENERATIONS=${2:-10}
POPULATION_SIZE=${3:-20}

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

python src/main.py \
  --experiment-name "$EXPERIMENT_NAME" \
  --generations "$GENERATIONS" \
  --population-size "$POPULATION_SIZE" \
  --config configs/default.yaml

echo "Optimisation complete! Results saved to: results/$EXPERIMENT_NAME"
