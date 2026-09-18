#!/usr/bin/env bash
#SBATCH --job-name=lgrag-crud-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=lgrag-crud-%j.out
#SBATCH --error=lgrag-crud-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${PYTHON_BIN:?Export existing project venv Python path}"
: "${HF_HOME:?Export existing offline model cache}"
: "${PILOT_DATASET:?Export prepared dataset path}"
cd "${SLURM_SUBMIT_DIR:?}"
test -f "$PILOT_DATASET"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
run_dir="$PWD/.tmp/cluster/crud-pilot-$SLURM_JOB_ID"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
scontrol show job "$SLURM_JOB_ID" > "$run_dir/slurm-start.txt"
for arm in project-vector llamaindex-vector project-vector-rerank llamaindex-vector-rerank project-full-rerank; do
  system=project
  profile=vector
  extra=()
  if [[ "$arm" == llamaindex-* ]]; then system=llamaindex; fi
  if [[ "$arm" == *-full-* ]]; then profile=full; fi
  if [[ "$arm" == *-rerank ]]; then extra+=(--rerank); fi
  printf '\nARM_BEGIN %s\n' "$arm"
  srun "$PYTHON_BIN" -u -B scripts/external_comparison.py \
    --dataset "$PILOT_DATASET" --out "$run_dir/$arm" \
    --split dev --system "$system" --profile "$profile" \
    --embedding-backend model --require-cuda --top-k 10 --candidate-k 40 "${extra[@]}"
  printf 'ARM_COMPLETE %s\n' "$arm"
done
printf '\nCRUD_PILOT_ALL_FIVE_ARMS_COMPLETED\n'
