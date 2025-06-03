# Molmo VLLM Integration

This document describes the VLLM integration for Molmo models in VLMEvalKit.

## Overview

Molmo models from AllenAI now support VLLM acceleration for faster inference. This integration provides significant speedup for batch evaluation while maintaining compatibility with the existing transformers-based pipeline.

## Supported Models

The following Molmo models support VLLM acceleration:

- `molmoE-1B-0924` - MolmoE 1B parameter model
- `molmo-7B-D-0924` - Molmo 7B Dense model  
- `molmo-7B-O-0924` - Molmo 7B Optimized model
- `molmo-72B-0924` - Molmo 72B model

## Requirements

### Base Requirements
- PyTorch with CUDA support
- transformers >= 4.46.0
- einops
- pillow

### VLLM Requirements
- vllm (latest version with Molmo support)
- CUDA-capable GPU(s)
- Sufficient GPU memory (varies by model size)

### Installation

```bash
# Install base requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers einops pillow

# Install VLLM (ensure latest version with Molmo support)
pip install vllm
```

## Usage

### Command Line

#### Basic Usage
```bash
# Run with VLLM acceleration
python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm

# Run without VLLM (standard transformers)
python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN
```

#### Advanced Options
```bash
# Custom GPU memory utilization
python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --gpu-utils 0.8

# Verbose output
python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm --verbose
```

### JSON Configuration

```json
{
    "model": {
        "molmo-7b-vllm": {
            "class": "molmo",
            "model_path": "allenai/Molmo-7B-D-0924",
            "use_vllm": true,
            "max_new_tokens": 200,
            "temperature": 0.0,
            "gpu_utils": 0.9
        },
        "molmo-7b-transformers": {
            "class": "molmo", 
            "model_path": "allenai/Molmo-7B-D-0924",
            "use_vllm": false,
            "max_crops": 36
        }
    },
    "data": {
        "MMBench_DEV_EN": {
            "class": "ImageMCQDataset",
            "dataset": "MMBench_DEV_EN"
        }
    }
}
```

### Python API

```python
from vlmeval.vlm.molmo import molmo

# Initialize with VLLM
model = molmo(
    model_path="allenai/Molmo-7B-D-0924",
    use_vllm=True,
    max_new_tokens=200,
    temperature=0.0,
    gpu_utils=0.9,
    verbose=True
)

# Initialize without VLLM  
model_transformers = molmo(
    model_path="allenai/Molmo-7B-D-0924",
    use_vllm=False,
    max_crops=36
)

# Generate response
message = [
    {"type": "text", "value": "What do you see in this image?"},
    {"type": "image", "value": "/path/to/image.jpg"}
]

response = model.generate_inner(message, dataset="MMBench_DEV_EN")
```

## Automatic Truncation

Molmo models now include automatic context length management to prevent "maximum length exceeded" warnings and errors.

### How It Works

- **Auto-truncation** is enabled by default (`auto_truncate=True`)
- When input exceeds `max_context_length`, content is intelligently truncated
- **Images are preserved** - truncation primarily affects text content
- **Smart truncation** keeps the beginning and end of text, removing middle content
- A `[TRUNCATED]` marker indicates where content was removed

### Example with Truncation

```python
# Model with custom context length and truncation settings
model = molmo(
    model_path="allenai/Molmo-7B-D-0924",
    max_context_length=2048,  # Shorter context for demonstration
    auto_truncate=True,       # Enable auto-truncation (default)
    verbose=True              # Show truncation warnings
)

# Long message that exceeds context length
very_long_message = [
    {"type": "text", "value": "Very long text content..." * 1000},
    {"type": "image", "value": "/path/to/image.jpg"}
]

# Model automatically truncates text while preserving image
response = model.generate_inner(very_long_message)  # No length warnings!
```

### Disabling Truncation

```python
# Disable auto-truncation to get original behavior
model = molmo(
    model_path="allenai/Molmo-7B-D-0924",
    auto_truncate=False  # Disable truncation
)
```

## Configuration Parameters

### VLLM-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_vllm` | bool | False | Enable VLLM acceleration |
| `gpu_utils` | float | 0.9 | GPU memory utilization (0.0-1.0) |
| `max_new_tokens` | int | 200 | Maximum tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature |
| `verbose` | bool | False | Enable verbose logging |
| `max_context_length` | int | 4096 | Maximum context length in tokens |
| `auto_truncate` | bool | True | Automatically truncate long inputs |

### Model-Specific Parameters  

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_crops` | int | 36 | Maximum image crops (transformers only) |

### Automatic Parameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| `max_model_len` | 4096 | Maximum sequence length (Molmo's actual context length) |
| `limit_mm_per_prompt` | 24 | Maximum images per prompt |
| `tensor_parallel_size` | auto | Determined by available GPUs |

## Performance Considerations

### GPU Requirements

| Model | Minimum VRAM | Recommended VRAM | Tensor Parallel |
|-------|--------------|------------------|-----------------|
| MolmoE-1B | 4GB | 6GB | 1 |
| Molmo-7B | 14GB | 18GB | 1-2 |
| Molmo-72B | 144GB | 160GB | 8 |

### Optimization Tips

1. **Tensor Parallelism**: Automatically configured based on available GPUs
   - 8+ GPUs: TP=8
   - 4-7 GPUs: TP=4  
   - 2-3 GPUs: TP=2
   - 1 GPU: TP=1

2. **Memory Management**: Adjust `gpu_utils` based on other GPU usage
   ```bash
   # Conservative memory usage
   --gpu-utils 0.7
   ```

3. **Batch Processing**: VLLM excels at processing multiple requests
   ```bash
   # Process multiple datasets efficiently
   python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN VQAv2_VAL --use-vllm
   ```

## Environment Setup

### Required Environment Variables

```bash
# Recommended for multiprocessing
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Optional: Disable V1 engine for compatibility  
export VLLM_USE_V1=0
```

### CUDA Setup

```bash
# Verify CUDA availability
python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"
python -c \"import torch; print(f'GPU count: {torch.cuda.device_count()}')\"
```

## Troubleshooting

### Common Issues

#### 1. VLLM Import Error
```
ImportError: No module named 'vllm'
```
**Solution**: Install VLLM with Molmo support
```bash
pip install vllm
```

#### 2. CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solutions**:
- Reduce `gpu_utils`: `--gpu-utils 0.7`
- Use smaller model: `molmoE-1B-0924` instead of `molmo-7B-D-0924`
- Enable tensor parallelism across multiple GPUs

#### 3. Multiprocessing Issues
```
RuntimeError: spawn is not supported
```
**Solution**: Set environment variable
```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

#### 4. Model Loading Errors
```
ValueError: trust_remote_code is required for Molmo models
```
**Solution**: This is automatically handled in the VLLM integration

#### 5. Context Length Warnings (FIXED)
```
This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (4096)
```
**Solution**: This warning has been eliminated with automatic truncation
- Auto-truncation is enabled by default (`auto_truncate=True`)
- Long inputs are automatically truncated to fit within context length
- Images are preserved; text content is intelligently truncated
- Disable with `auto_truncate=False` if needed

```python
# Enable verbose mode to see truncation warnings
model = molmo(
    model_path="allenai/Molmo-7B-D-0924",
    verbose=True,           # Shows when truncation occurs
    max_context_length=4096 # Adjust context length if needed
)
```

#### 6. Max Model Length Error (FIXED)
```
ValueError: User-specified max_model_len (16384) is greater than the derived max_model_len (max_position_embeddings=4096)
```
**Solution**: This error has been fixed in the latest version
- Molmo models have a maximum context length of 4096 tokens
- VLLM configuration now automatically uses the correct max_model_len (4096)
- The auto-truncation feature ensures inputs fit within this limit
- If you need to override: set environment variable `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

```bash
# If you still encounter this error, try:
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
```

### Debug Mode

Enable verbose logging for debugging:

```python
model = molmo(
    model_path="allenai/Molmo-7B-D-0924",
    use_vllm=True,
    verbose=True
)
```

### Performance Monitoring

Monitor GPU usage during inference:
```bash
# Terminal 1: Run evaluation
python run.py --model molmo-7B-D-0924 --dataset MMBench_DEV_EN --use-vllm

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

## Validation

Test the integration with the provided test script:

```bash
python scripts/test_molmo_vllm.py
```

This will verify:
- ✓ Model initialization (both VLLM and transformers)
- ✓ Method availability 
- ✓ Content preparation
- ✓ Configuration integration
- ✓ Inference pipeline integration

## Datasets Support

Molmo with VLLM supports all standard VLMEvalKit datasets:

### Image Datasets
- MMBench, MMBench_CN
- VQAv2, TextVQA, DocVQA
- COCO Captioning
- AI2D, ScienceQA
- ChartQA, InfoVQA
- And more...

### Specialized Prompts
The integration preserves Molmo's specialized prompting for different datasets:
- `vqa2:` for VQA tasks
- `a_okvqa_mc:` for multiple choice
- `ai2_diagram:` for AI2D
- `chart_qa:` for ChartQA
- And more...

## Benchmarks

Performance comparison (approximate, varies by hardware):

| Configuration | Throughput | Memory Usage | 
|---------------|------------|--------------|
| Molmo-7B + Transformers | 1x | 100% |
| Molmo-7B + VLLM (1 GPU) | 2-3x | 90% |
| Molmo-7B + VLLM (2 GPU) | 4-5x | 85% |

## Contributing

When contributing to Molmo VLLM support:

1. Test both VLLM and transformers backends
2. Verify dataset-specific prompting works correctly
3. Check memory usage and performance
4. Update documentation for new features
5. Run the test suite: `python scripts/test_molmo_vllm.py`

## References

- [Molmo Paper](https://arxiv.org/abs/2409.17146)
- [Molmo Models on Hugging Face](https://huggingface.co/collections/allenai/molmo-66f379e6fe3b8ef090a8ca19)
- [VLLM Documentation](https://docs.vllm.ai/)
- [VLLM Molmo Support PR](https://github.com/vllm-project/vllm/pull/9016)