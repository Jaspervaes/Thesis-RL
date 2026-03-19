#!/usr/bin/env bash
# Run both Single-Model and Multi-Model CQL pipelines over 5 seeds.
#
# Usage:
#   ./run_all.sh [--n_cases N] [--confounded]

set -euo pipefail

ARGS="$@"

echo ""
echo "######################################"
echo "#  Single-Model CQL"
echo "######################################"
bash run_experiments.sh ${ARGS}

echo ""
echo "######################################"
echo "#  Multi-Model CQL"
echo "######################################"
bash run_experiments_multi.sh ${ARGS}

echo ""
echo "======================================"
echo "Both methods finished."
echo "======================================"
