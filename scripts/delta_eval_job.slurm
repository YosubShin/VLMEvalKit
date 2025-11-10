#!/bin/bash
#SBATCH --job-name=VLMEvalKit
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH -A bchy-delta-gpu

set -euo pipefail

module purge >/dev/null 2>&1 || true

RESULTS_ROOT="${KOA_ML_RESULTS_ROOT:-$HOME/koa-results}"
JOB_DIR="${KOA_RUN_DIR:-${RESULTS_ROOT}/${SLURM_JOB_ID}}"
REPO_DIR="${JOB_DIR}/repo"
RESULTS_DIR="${JOB_DIR}/results"
mkdir -p "${REPO_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ -d "${REPO_DIR}" ]]; then
  cd "${REPO_DIR}"
fi

echo "Writing outputs to ${RESULTS_DIR}"

echo "==== Job Info ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "KOA_ML_CODE_ROOT=${KOA_ML_CODE_ROOT:-unset}"
echo "KOA_RUN_DIR=${KOA_RUN_DIR:-unset}"
echo "KOA_RUN_METADATA_DIR=${KOA_RUN_METADATA_DIR:-unset}"
echo "KOA_SHARED_ENV=${KOA_SHARED_ENV:-unset}"
echo

echo "==== GPU Info ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not available"
fi
echo

echo "==== GPU Memory Detection ===="
# Query first GPU's name and total VRAM (in MB)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | xargs)

echo "Detected GPU: ${GPU_NAME} with ${GPU_MEM} MB VRAM"

# Default settings
BATCH_SIZE=4
DTYPE="bfloat16"

# Adjust batch size and other configs based on VRAM capacity
if (( GPU_MEM < 20000 )); then
    echo "⚠️ GPU has <20GB VRAM — too small for Qwen2.5-VL-7B vLLM."
    exit 1
elif (( GPU_MEM < 30000 )); then
    BATCH_SIZE=2
elif (( GPU_MEM < 50000 )); then
    BATCH_SIZE=4
elif (( GPU_MEM < 80000 )); then
    BATCH_SIZE=16
else
    BATCH_SIZE=32
fi

echo "Using batch size: ${BATCH_SIZE}"

echo "==== Python Environment ====="
if command -v python >/dev/null 2>&1; then
  which python
  python --version
else
  echo "python not found"
fi
echo

source "scripts/setup_env.sh"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_GPU_MEMORY_UTILIZATION=0.85

# Allow overriding the model path while keeping a filesystem-safe label for outputs.
CUSTOM_MODEL="${CUSTOM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
CUSTOM_MODEL_SAFE="${CUSTOM_MODEL//\//_}"

# --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct \
# --pass-custom-model /home/yosubs/koa_scratch/Qwen3-VL/qwen-vl-finetune/output/11k/qwen3-vl-8b-instruct/checkpoint-520 \

srun --mpi=none uv run run.py \
--pass-custom-model "${CUSTOM_MODEL}" \
--data Omni3DBench \
--work-dir ${RESULTS_DIR} \
--use-vllm \
--max-output-tokens 4096 \
--batch-size ${BATCH_SIZE} \
--api-nproc 8 \
--reuse \
--judge gpt-4o-mini

srun uv run scripts/dcvlr_standalone_scorer.py \
      --benchmarks Omni3DBench \
      --input-dir "${RESULTS_DIR}/${CUSTOM_MODEL_SAFE}" \
      --llm-backend openai --model gpt-4o-mini \
      --num-workers 8 \
      --verbose
