#!/bin/bash
#SBATCH --job-name=VLMEvalKit
#SBATCH --partition=kill-shared
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200_nvl:1

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
# --pass-custom-model Qwen/Qwen3-VL-8B-Instruct \
# --pass-custom-model /home/yosubs/koa_scratch/Qwen3-VL/qwen-vl-finetune/output/11k/qwen3-vl-8b-instruct/checkpoint-520 \
# --data VMCBench_DEV LiveXivTQA OlympiadBench Omni3DBench atomic_dataset electro_dataset mechanics_dataset optics_dataset quantum_dataset statistics_dataset \

srun uv run run.py \
--pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct \
--mode infer \
--data OlympiadBench \
--work-dir /home/yosubs/koa_scratch/VLMEvalKit/outputs \
--use-vllm \
--max-output-tokens 8192 \
--batch-size 64 \
--api-nproc 16 \
--reuse \
--judge gpt-4o-mini

# srun uv run scripts/dcvlr_standalone_scorer.py \
#       --benchmarks VMCBench_DEV LiveXivTQA OlympiadBench Omni3DBench atomic_dataset electro_dataset mechanics_dataset optics_dataset quantum_dataset statistics_dataset \
#       --input-dir /home/yosubs/koa_scratch/VLMEvalKit/outputs/Qwen_Qwen2.5-VL-7B-Instruct  \
#       --llm-backend openai --model gpt-4o-mini \
#       --resume --verbose