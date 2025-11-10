#!/bin/bash
#SBATCH --job-name=VLMEvalKit
#SBATCH --partition=kill-shared
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint="hopper|ampere"
#SBATCH --gres=gpu:1

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

if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ MIG- ]]; then
  echo "Detected CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "This job was scheduled onto a MIG slice, which vLLM cannot handle yet. Exiting early so the job can be resubmitted to a full GPU."
  exit 2
fi

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

# --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct \
# --pass-custom-model Qwen/Qwen3-VL-8B-Instruct \
# --pass-custom-model /home/yosubs/koa_scratch/Qwen3-VL/qwen-vl-finetune/output/11k/qwen3-vl-8b-instruct/checkpoint-520 \
# --data VMCBench_DEV LiveXivTQA OlympiadBench Omni3DBench atomic_dataset electro_dataset mechanics_dataset optics_dataset quantum_dataset statistics_dataset \

srun uv run run.py \
--pass-custom-model /home/yosubs/koa_scratch/oumi/output/qwen2_5_vl_7b_walton_random_1k_1 \
--mode infer \
--data VMCBench_DEV LiveXivTQA OlympiadBench Omni3DBench atomic_dataset electro_dataset mechanics_dataset optics_dataset quantum_dataset statistics_dataset \
--work-dir /home/yosubs/koa_scratch/VLMEvalKit/outputs \
--use-vllm \
--max-output-tokens 8192 \
--batch-size ${BATCH_SIZE} \
--api-nproc 8 \
--reuse \
--judge gpt-4o-mini

srun uv run scripts/dcvlr_standalone_scorer.py \
      --benchmarks VMCBench_DEV LiveXivTQA OlympiadBench Omni3DBench atomic_dataset electro_dataset mechanics_dataset optics_dataset quantum_dataset statistics_dataset \
      --input-dir /home/yosubs/koa_scratch/VLMEvalKit/outputs/_home_yosubs_koa_scratch_oumi_output_qwen2_5_vl_7b_walton_random_1k_1  \
      --llm-backend openai --model gpt-4o-mini \
      --num-workers 8 \
      --verbose