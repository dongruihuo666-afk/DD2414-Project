#!/usr/bin/env bash
# TEMPLATE ONLY: review site partition, node-local data and authorized resources.
#SBATCH --job-name=dd2414-new-depth
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${DD2414_PROJECT_ROOT:?Set an absolute source checkout}"
: "${DD2414_PYTHON:?Set the existing mamba environment Python}"
: "${DD2414_DEPLOYMENT:?Set your private deployment JSON path}"
: "${DD2414_EXPERIMENT:?Set the reviewed experiment JSON path}"
cd "$DD2414_PROJECT_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "$DD2414_PYTHON" tools/launch_training.py \
  --deployment "$DD2414_DEPLOYMENT" --experiment "$DD2414_EXPERIMENT" --execute
