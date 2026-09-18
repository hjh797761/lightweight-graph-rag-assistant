#!/usr/bin/env bash
#SBATCH --job-name=lgrag-crud-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=lgrag-ablation-%j.out
#SBATCH --error=lgrag-ablation-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${PYTHON_BIN:?Export existing project venv Python path}"
: "${HF_HOME:?Export existing offline model cache}"
: "${PILOT_DATASET:?Export unchanged pilot dataset path}"
cd "${SLURM_SUBMIT_DIR:?}"
test -f "$PILOT_DATASET"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Existing bridge scoring uses set iteration; keep it identical across arm processes.
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
run_dir="$PWD/.tmp/cluster/crud-ablation-$SLURM_JOB_ID"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
scontrol show job "$SLURM_JOB_ID" > "$run_dir/slurm-start.txt"
# name | implementation | profile | reranker enabled | final output override
arms=(
  'project-vector|project|vector|0|0'
  'llamaindex-vector|llamaindex|vector|0|0'
  'project-vector-rerank|project|vector|1|0'
  'llamaindex-vector-rerank|llamaindex|vector|1|0'
  'project-full-rerank|project|full|1|0'
  'project-full-rerank-fixed10|project|full|1|10'
  'project-vector-rerank-fixed3|project|vector|1|3'
  'llamaindex-vector-rerank-fixed3|llamaindex|vector|1|3'
  'project-full-rerank-fixed3|project|full|1|3'
  'project-keyword-rerank-fixed10|project|vector_keyword|1|10'
  'project-graph-rerank-fixed10|project|vector_graph|1|10'
  'project-topic-graph-rerank-fixed10|project|vector_topic_graph|1|10'
)
for spec in "${arms[@]}"; do
  IFS='|' read -r arm system profile rerank fixed <<< "$spec"
  extra=()
  if [[ "$rerank" == 1 ]]; then extra+=(--rerank); fi
  if [[ "$fixed" != 0 ]]; then extra+=(--fixed-output-k "$fixed"); fi
  printf '\nARM_BEGIN %s\n' "$arm"
  srun "$PYTHON_BIN" -u -B scripts/external_comparison.py \
    --dataset "$PILOT_DATASET" --out "$run_dir/$arm" \
    --split dev --system "$system" --profile "$profile" \
    --embedding-backend model --require-cuda --top-k 10 --candidate-k 40 "${extra[@]}"
  printf 'ARM_COMPLETE %s\n' "$arm"
done
printf '\nCRUD_ABLATION_ALL_TWELVE_ARMS_COMPLETED\n'
