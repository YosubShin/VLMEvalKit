#!/bin/bash
#SBATCH --job-name=DCVLR_Scorer
#SBATCH --partition=kill-shared
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

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

source "scripts/setup_env_simple.sh"

#     --input-dir /home/yosubs/koa_scratch/VLMEvalKit/outputs/Qwen_Qwen2.5-VL-7B-Instruct \
#      --input-dir /home/yosubs/koa_scratch/VLMEvalKit/outputs/_home_yosubs_koa_scratch_oumi_output_qwen2_5_vl_7b_walton_hard_cosyn_traces_1k  \

srun uv run scripts/dcvlr_standalone_scorer.py \
      --benchmarks VMCBench_DEV LiveXivTQA OlympiadBench Omni3DBench atomic_dataset electro_dataset mechanics_dataset optics_dataset quantum_dataset statistics_dataset \
      --input-dir /home/yosubs/koa_scratch/VLMEvalKit/outputs/_home_yosubs_koa_scratch_oumi_output_qwen2_5_vl_7b_walton_hard_cosyn_traces_1k  \
      --llm-backend openai --model gpt-4o-mini \
      --verbose
