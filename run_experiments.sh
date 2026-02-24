#!/usr/bin/env bash
# Run Single-Model CQL pipeline over 5 random seeds.
#
# Usage:
#   ./run_experiments.sh [--n_cases N] [--confounded]
#
# Defaults: --n_cases 10000, no confounding.

set -euo pipefail

N_CASES=10000
CONFOUNDED=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_cases)
            N_CASES="$2"
            shift 2
            ;;
        --confounded)
            CONFOUNDED="--confounded"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

SEEDS=(42 123 456 789 1024)

echo "======================================"
echo "Single-Model CQL Experiment Runner"
echo "n_cases=${N_CASES}  confounded=${CONFOUNDED:-false}"
echo "Seeds: ${SEEDS[*]}"
echo "======================================"

for SEED in "${SEEDS[@]}"; do
    EVAL_SEED="99${SEED}"

    echo ""
    echo "======================================"
    echo "Seed: ${SEED}  (eval seed: ${EVAL_SEED})"
    echo "======================================"

    echo "[1/4] Generating data (seed=${SEED})..."
    python singleModelCQL/generate_data.py \
        --n_cases "${N_CASES}" \
        --seed "${SEED}" \
        ${CONFOUNDED}

    echo "[2/4] Converting data..."
    python singleModelCQL/convert_data.py \
        --n_cases "${N_CASES}" \
        ${CONFOUNDED}

    echo "[3/4] Training..."
    python singleModelCQL/train.py \
        --n_cases "${N_CASES}" \
        ${CONFOUNDED}

    echo "[4/4] Evaluating (eval seed=${EVAL_SEED})..."
    python singleModelCQL/evaluate.py \
        --n_cases "${N_CASES}" \
        --seed "${EVAL_SEED}" \
        ${CONFOUNDED}

    echo "Seed ${SEED} complete."
done

echo ""
echo "======================================"
echo "All seeds finished."
echo "======================================"
