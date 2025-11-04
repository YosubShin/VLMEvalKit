#!/bin/bash
#SBATCH --job-name=VLMEvalKit
#SBATCH --partition=kill-shared
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_pcie:1

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

srun uv run run.py \
--pass-custom-model /home/yosubs/koa_scratch/Qwen3-VL/qwen-vl-finetune/output/11k/qwen3-vl-8b-instruct/checkpoint-520 \
--data Omni3DBench \
--work-dir ${RESULTS_DIR} \
--use-vllm \
--max-output-tokens 4096 \
--batch-size 8 \
--api-nproc 8 \
--reuse \
--judge gpt-4o-mini
