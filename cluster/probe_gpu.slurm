#!/usr/bin/env bash
#SBATCH --job-name=lgrag-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=lgrag-probe-%j.out
#SBATCH --error=lgrag-probe-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch, never run GPU work on the login node}"
: "${PYTHON_BIN:?Export the absolute Python interpreter path before submission}"
cd "${SLURM_SUBMIT_DIR:?}"
test -f scripts/probe_gpu.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
output_dir="${RUN_ROOT:-$PWD/.tmp/cluster}/probe-${SLURM_JOB_ID}"
mkdir -p "$output_dir"
scontrol show job "$SLURM_JOB_ID" > "$output_dir/slurm-start.txt"
args=()
if [[ "${PROBE_MODELS:-0}" == 1 ]]; then
  args+=(--models)
fi
srun "$PYTHON_BIN" -B scripts/probe_gpu.py --out "$output_dir/probe.json" "${args[@]}"
