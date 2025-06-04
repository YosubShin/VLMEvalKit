# WandB Integration for VLMEvalKit

This directory contains a WandB (Weights & Biases) integration script for logging VLMEvalKit evaluation results to experiment tracking, with full support for VLLM batch processing.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install wandb pandas torch transformers vllm
   ```

2. **Login to WandB:**
   ```bash
   wandb login
   ```
   Follow the prompts to authenticate with your WandB account.

## Usage

### 1. Run Evaluation and Log to WandB

Run a new evaluation and automatically log results:

```bash
# Basic usage - console output from run.py will be displayed in real-time
python scripts/wandb_logger.py --run-and-log --model GPT4o --dataset MMBench_DEV_EN

# With custom project name
python scripts/wandb_logger.py --run-and-log --model GPT4o --dataset MMBench_DEV_EN --project my-vlm-experiments

# VLLM Batch Processing (NEW!) - 2-4x faster inference
python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --batch-size 4

# Conservative batch size for memory-constrained environments
python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --batch-size 2

# With verbose batch processing monitoring
python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --batch-size 4 --verbose

# With additional run.py arguments
python scripts/wandb_logger.py --run-and-log --model GPT4o --dataset MMBench_DEV_EN --run-args --verbose --api-nproc 8
```

### 2. Log Existing Results

Log results from previous evaluations:

```bash
# Log specific model/dataset combination
python scripts/wandb_logger.py --model GPT4o --dataset MMBench_DEV_EN --work-dir ./outputs

# Log all existing results in the work directory
python scripts/wandb_logger.py --log-all --work-dir ./outputs
```

### 3. Advanced Options

```bash
# Add custom tags and notes
python scripts/wandb_logger.py \
  --model GPT4o \
  --dataset MMBench_DEV_EN \
  --tags baseline-eval production \
  --notes "Baseline evaluation with updated prompts"

# Use custom WandB project
python scripts/wandb_logger.py \
  --log-all \
  --project vlm-comparison-2024 \
  --work-dir ./outputs
```

## VLLM Batch Processing Support

### Supported Models

The following models support VLLM batch processing for 2-4x speedup:

- **Molmo**: `molmo-7B-D-0924`, `molmo-72B-D-0924` (fully implemented)
- **Qwen2-VL**: `qwen2-vl-7b-instruct`, `qwen2-vl-72b-instruct`
- **Llama4**: `llama-4-11b-vision-instruct`, `llama-4-90b-vision-instruct`
- **Gemma3**: `gemma-3-2b-it`, `gemma-3-9b-it`

### Batch Size Recommendations

| Batch Size | Memory Usage | Use Case | Expected Speedup |
|------------|--------------|----------|------------------|
| 2          | Low          | 8-16GB VRAM | 1.5-2.0x |
| 4          | Medium       | 16-32GB VRAM | 2.0-3.0x |
| 8          | High         | 32GB+ VRAM | 2.5-4.0x |

### Example Batch Processing Commands

```bash
# Compare sequential vs batch processing performance
python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --dataset MMBench_DEV_EN \
  --tags sequential --notes "Baseline sequential processing"

python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --dataset MMBench_DEV_EN \
  --use-vllm --batch-size 4 --tags batch-4 --notes "Batch processing with size 4"

# Test different batch sizes
for batch_size in 2 4 8; do
  python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --dataset MMBench_DEV_EN \
    --use-vllm --batch-size $batch_size --tags batch-sweep --notes "Batch size: $batch_size"
done
```

## What Gets Logged

### Metrics
- **Accuracy**: Calculated from prediction/answer columns if available
- **Dataset Statistics**: Sample counts, categories, etc.
- **Numerical Metrics**: Any numerical columns in result files (mean/std)
- **Custom Metrics**: Extracted from JSON/CSV result files
- **Batch Processing Stats**: Efficiency metrics when using VLLM batching

### Artifacts
- **Result Files**: Raw evaluation outputs (.xlsx, .csv, .json files)
- **Predictions**: Model predictions and ground truth when available

### Configuration
- **Model Info**: Model name and configuration parameters
- **Dataset Info**: Dataset name and metadata
- **Framework**: VLMEvalKit version and settings
- **VLLM Settings**: Backend type, batch size, processing mode

## Features

### Real-Time Output Streaming

When using `--run-and-log`, the console output from `run.py` is displayed in real-time:

- **Live Progress**: See evaluation progress as it happens
- **Error Visibility**: Immediately see any errors or warnings
- **No Lost Output**: All console output is preserved and displayed
- **Combined Streams**: Both stdout and stderr are shown together

This means you get the same experience as running `run.py` directly, but with automatic WandB logging when complete.

## WandB Dashboard

After logging, you can view results in your WandB dashboard:

1. **Runs Table**: Compare metrics across different models/datasets
2. **Charts**: Visualize accuracy trends and performance comparisons  
3. **Artifacts**: Download and inspect raw result files
4. **System Metrics**: Resource usage during evaluation (if available)

## Examples

### Example 1: Benchmark Multiple Models

```bash
# Run evaluations for multiple models
for model in GPT4o InternVL2-8B QwenVL-Chat; do
    python scripts/wandb_logger.py --run-and-log --model $model --dataset MMBench_DEV_EN
done
```

### Example 2: Batch Upload Existing Results

```bash
# Upload all existing results from a completed benchmark run
python scripts/wandb_logger.py --log-all --work-dir ./outputs --project vlm-benchmark-2024
```

### Example 3: Compare Model Variants

```bash
# Compare different configurations of the same model
python scripts/wandb_logger.py --run-and-log --model Qwen2-VL-7B-Instruct --dataset VQAv2_VAL --tags baseline
python scripts/wandb_logger.py --run-and-log --model Qwen2-VL-7B-Instruct-AWQ --dataset VQAv2_VAL --tags quantized
```

## Configuration

### Environment Variables

- `WANDB_PROJECT`: Default project name
- `WANDB_ENTITY`: WandB team/username  
- `WANDB_API_KEY`: API key for authentication
- `WANDB_MODE`: Set to "offline" for offline logging

### Project Organization

The script automatically organizes runs with:
- **Project**: Configurable (default: "vlmeval-benchmark")
- **Run Name**: `{model_name}_{dataset_name}`
- **Tags**: Model name, dataset name, and custom tags
- **Config**: Model parameters, dataset info, framework details

## Troubleshooting

### Common Issues

1. **"wandb not installed"**
   ```bash
   pip install wandb
   ```

2. **"WandB not configured"**
   ```bash
   wandb login
   ```

3. **"No result files found"**
   - Ensure evaluation completed successfully
   - Check `--work-dir` path is correct
   - Verify model/dataset names match exactly

4. **Permission errors**
   - Ensure you have write access to the work directory
   - Check WandB project permissions

### Debug Mode

Add verbose logging:
```bash
python scripts/wandb_logger.py --run-and-log --model GPT4o --dataset MMBench_DEV_EN --verbose
```

## Integration with Existing Workflows

### CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmark and log to WandB
  env:
    WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
  run: |
    python scripts/wandb_logger.py --run-and-log --model ${{ matrix.model }} --dataset ${{ matrix.dataset }}
```

### Slurm Job Integration

```bash
#!/bin/bash
#SBATCH --job-name=vlm-eval-wandb
#SBATCH --nodes=1
#SBATCH --gpus=1

export WANDB_API_KEY="your-api-key"
python scripts/wandb_logger.py --run-and-log --model InternVL2-8B --dataset MMBench_DEV_EN
```

## Files Created

The WandB logger script:
- ✅ `/scripts/wandb_logger.py` - Main integration script
- ✅ `/scripts/README_wandb.md` - This documentation  
- ✅ Updated `requirements.txt` to include `wandb`

Start experimenting with WandB logging by running:
```bash
python scripts/wandb_logger.py --help
```