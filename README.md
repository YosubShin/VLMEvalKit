![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>A Toolkit for Evaluating Large Vision-Language Models. </b>

<b>DCVLR Edition </b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

English | [简体中文](/docs/zh-CN/README_zh-CN.md) | [日本語](/docs/ja/README_ja.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OC Learderboard </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF Leaderboard</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 Evaluation Records</a> •
<a href="https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard">🤗 HF Video Leaderboard</a> •

<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 Report</a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>
</div>

**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-based answer extraction**.

This is the **Data Curation for Vision-Language Reasoning (DCVLR)** edition of VLMEvalKit, introduced by [Oumi AI](github.com/oumi-ai) in support of the [2025 Competition Workshop at NeurIPS](https://dcvlr-neurips.github.io/). This edition includes several new datasets, refines support for the Molmo series of models, and adds useful features (see 🔥 Enhanced Features).

Modifications to VLMEvalKit are by @penfever, @oelachqar, @NimrodShabtay and @rohunagrawal.

## 🚀 Key Features

- **🤖 Custom Model Support**: Automatically detect and evaluate models from HuggingFace repositories with `--pass-custom-model`
- **⚡ VLLM Acceleration**: High-performance inference with batch processing for supported models
- **📊 Advanced Logging**: Comprehensive evaluation tracking with WandB integration
- **💾 Detailed Response Saving**: Save raw model responses and LLM judge explanations
- **🔧 Token Control**: Global and model-specific output token limits
- **🎯 One-Command Evaluation**: Minimal setup for comprehensive benchmarking

## 🆕 News

> We have presented a [**comprehensive survey**](https://arxiv.org/pdf/2411.15296) on the evaluation of large multi-modality models, jointly with [**MME Team**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) and [**LMMs-Lab**](https://lmms-lab.github.io) 🔥🔥🔥
- **[2025-05-24]** To facilitate faster evaluations for large-scale or thinking models, **VLMEvalKit supports multi-node distributed inference** using **LMDeploy**  (supports *InternVL Series, QwenVL Series, LLaMa4*) or **VLLM**(supports *QwenVL Series, Qwen2.5-VL Series, LLaMa4, Molmo, Gemma3*). You can activate this feature by adding the ```use_lmdeploy``` or ```use_vllm``` flag to your custom model configuration in [config.py](vlmeval/config.py) or use ```--use-vllm --batch-size N``` for automatic batch processing. Custom models from HuggingFace repositories are automatically detected for VLLM compatibility. Leverage these tools to significantly speed up your evaluation workflows 🔥🔥🔥
- **[2025-05-24]** Supported Models: **InternVL3 Series, Gemini-2.5-Pro, Kimi-VL, LLaMA4, NVILA, Qwen2.5-Omni, Phi4, SmolVLM2, Grok, SAIL-VL-1.5, WeThink-Qwen2.5VL-7B, Bailingmm, VLM-R1, Taichu-VLR**. Supported Benchmarks: **HLE-Bench, MMVP, MM-AlignBench, Creation-MMBench, MM-IFEval, OmniDocBench, OCR-Reasoning, EMMA, ChaXiv，MedXpertQA, Physics, MSEarthMCQ, MicroBench, MMSci, VGRP-Bench, wildDoc, TDBench, VisuLogic, CVBench, LEGO-Puzzles, Video-MMLU, QBench-Video, MME-CoT, VLM2Bench, VMCBench, MOAT, Spatial457 Benchmark**. Please refer to [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) for more details. Thanks to all contributors 🔥🔥🔥
- **[2025-02-20]** Supported Models: **InternVL2.5 Series, Qwen2.5VL Series, QVQ-72B, Doubao-VL, Janus-Pro-7B, MiniCPM-o-2.6, InternVL2-MPO, LLaVA-CoT, Hunyuan-Standard-Vision, Ovis2, Valley, SAIL-VL, Ross, Long-VITA, EMU3, SmolVLM**. Supported Benchmarks: **MMMU-Pro, WeMath, 3DSRBench, LogicVista, VL-RewardBench, CC-OCR, CG-Bench, CMMMU, WorldSense**. Thanks to all contributors 🔥🔥🔥
- **[2024-12-11]** Supported [**NaturalBench**](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- **[2024-12-02]** Supported [**VisOnlyQA**](https://github.com/psunlpgroup/VisOnlyQA/), a benchmark for evaluating the visual perception capabilities 🔥🔥🔥
- **[2024-11-26]** Supported [**Ovis1.6-Gemma2-27B**](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-27B), thanks to [**runninglsy**](https://github.com/runninglsy) 🔥🔥🔥
- **[2024-11-25]** Create a new flag `VLMEVALKIT_USE_MODELSCOPE`. By setting this environment variable, you can download the video benchmarks supported from [**modelscope**](https://www.modelscope.cn) 🔥🔥🔥
- **[2024-11-25]** Supported [**VizWiz**](https://vizwiz.org/tasks/vqa/) benchmark 🔥🔥🔥
- **[2024-11-22]** Supported the inference of [**MMGenBench**](https://mmgenbench.alsoai.com), thanks [**lerogo**](https://github.com/lerogo) 🔥🔥🔥
- **[2024-11-22]** Supported [**Dynamath**](https://huggingface.co/datasets/DynaMath/DynaMath_Sample), a multimodal math benchmark comprising of 501 SEED problems and 10 variants generated based on random seeds. The benchmark can be used to measure the robustness of MLLMs in multi-modal math solving 🔥🔥🔥
- **[2024-11-21]** Integrated a new config system to enable more flexible evaluation settings. Check the [Document](/docs/en/ConfigSystem.md) or run `python run.py --help` for more details 🔥🔥🔥
- **[2024-11-21]** Supported [**QSpatial**](https://andrewliao11.github.io/spatial_prompt/), a multimodal benchmark for Quantitative Spatial Reasoning (determine the size / distance, e.g.), thanks [**andrewliao11**](https://github.com/andrewliao11)  for providing the official support 🔥🔥🔥
- **[2024-11-21]** Supported [**MM-Math**](https://github.com/kge-sun/mm-math), a new multimodal math benchmark comprising of ~6K middle school multi-modal reasoning math problems. GPT-4o-20240806 achieces 22.5% accuracy on this benchmark 🔥🔥🔥

## 🏗️ QuickStart

See [[QuickStart](/docs/en/Quickstart.md) | [快速开始](/docs/zh-CN/Quickstart.md)] for a quick start guide.

## 🔥 Enhanced Features

### 🤖 Custom Model Support

Automatically detect and evaluate any HuggingFace model without manual configuration:

```bash
# Evaluate a custom Qwen2.5-VL model
python run.py --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct --data MMBench_DEV_EN

# Evaluate a custom LLaVA model with multiple datasets
python run.py --pass-custom-model liuhaotian/llava-v1.5-7b --data MMBench_DEV_EN MMMU_DEV_VAL

# Combine custom model with existing models
python run.py --model GPT4o --pass-custom-model microsoft/Phi-3-vision-128k-instruct --data MMBench_DEV_EN

# Use custom model with VLLM acceleration
python run.py --pass-custom-model oumi-ai/Molmo-7B-D-0924 --data MMBench_DEV_EN --use-vllm --batch-size 4
```

**Supported Architectures**: Automatically detects 25+ model families including Qwen2-VL, Qwen2.5-VL, LLaVA, InternVL, MiniCPM, Phi, Molmo, Aria, Pixtral, SmolVLM, IDEFICS, CogVLM, DeepSeek, Llama-Vision, Gemma, VILA, Ovis, Bunny, Cambrian, Mantis, Moondream, and more.

### ⚡ VLLM Acceleration & Batch Processing

High-performance inference with 2-4x speedup for supported models:

```bash
# Enable VLLM with batch processing
python run.py --model molmo-7B-D-0924 --data MMBench_DEV_EN --use-vllm --batch-size 4

# Custom models with VLLM (auto-detected compatibility)
python run.py --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct --data MMBench_DEV_EN --use-vllm --batch-size 8

# Memory-constrained environments
python run.py --model molmo-7B-D-0924 --data MMBench_DEV_EN --use-vllm --batch-size 2 --verbose
```

**VLLM-Compatible Models**: 
- **Qwen2-VL & Qwen2.5-VL**: All variants automatically detected
- **Molmo**: All sizes with full batch processing support
- **Llama-4**: Scout and other variants
- **Gemma3**: Selected configurations
- **Custom Models**: Automatic compatibility detection

**Performance**: Up to 3.2x speedup with batch size 8 on large datasets (see [VLLM_BATCH_PROCESSING.md](/docs/VLLM_BATCH_PROCESSING.md)).

### 📊 WandB Experiment Tracking

Comprehensive experiment tracking and visualization with Weights & Biases:

```bash
# Log evaluation results to WandB
python scripts/wandb_logger.py --run-and-log --model GPT4o --data MMBench_DEV_EN

# Batch evaluation with WandB logging
python scripts/wandb_logger.py --run-and-log --model molmo-7B-D-0924 --data MMBench_DEV_EN MMMU_DEV_VAL --use-vllm --batch-size 4

# Custom model evaluation with WandB
python scripts/wandb_logger.py --run-and-log --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct --data MMBench_DEV_EN

# Log existing results
python scripts/wandb_logger.py --model GPT4o --data MMBench_DEV_EN --work-dir ./outputs

# Bulk upload all results
python scripts/wandb_logger.py --log-all --work-dir ./outputs
```

**Features**:
- Automatic metric extraction from result files
- Model configuration tracking
- Dataset statistics and analysis
- Performance monitoring for batch processing
- Experiment comparison and visualization

### 💾 Advanced Response Saving

Save detailed evaluation data for analysis and debugging:

```bash
# Save raw model responses
python run.py --model GPT4o --data quantum_dataset --save-detailed-eval

# Save LLM judge responses (for supported datasets)
python run.py --model GPT4o --data mechanics_dataset --save-judge-responses

# Save both with custom format
python run.py --model GPT4o --data atomic_dataset --save-detailed-eval --save-judge-responses --response-format xlsx

# Supported formats: json (default), csv, xlsx
python run.py --model GPT4o --data OlympiadBench --save-detailed-eval --response-format csv
```

**Supported Datasets**:
- **Judge Responses**: Yale_physics datasets (quantum_dataset, mechanics_dataset, atomic_dataset, electro_dataset, optics_dataset, statistics_dataset)
- **Detailed Evaluation**: Yale_physics, OlympiadBench, VMCBench_DEV

### 🔧 Token Control

Fine-grained control over model output length:

```bash
# Global token override (supersedes all other limits)
python run.py --model GPT4o --data MMBench_DEV_EN --max-output-tokens 2048

# With custom models
python run.py --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct --data MMBench_DEV_EN --max-output-tokens 1024

# Combined with other features
python run.py --pass-custom-model oumi-ai/Molmo-7B-D-0924 --data MMBench_DEV_EN --use-vllm --batch-size 4 --max-output-tokens 512
```

**Benefits**:
- Consistent output lengths across models
- Cost control for API models
- Memory optimization for large-scale evaluation
- Standardized comparison conditions

### 🎯 Complete Workflow Examples

```bash
# Complete custom model evaluation with all features
python scripts/wandb_logger.py --run-and-log \
  --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct \
  --data MMBench_DEV_EN MMMU_DEV_VAL \
  --use-vllm --batch-size 4 \
  --max-output-tokens 1024 \
  --save-detailed-eval \
  --response-format json \
  --verbose

# Research workflow: evaluate multiple custom models
for model in "Qwen/Qwen2.5-VL-7B-Instruct" "microsoft/Phi-3-vision-128k-instruct" "oumi-ai/Molmo-7B-D-0924"; do
  python scripts/wandb_logger.py --run-and-log \
    --pass-custom-model "$model" \
    --data MMBench_DEV_EN \
    --use-vllm --batch-size 4 \
    --max-output-tokens 1024
done

# Production evaluation with comprehensive logging
python scripts/wandb_logger.py --run-and-log \
  --model GPT4o \
  --data quantum_dataset mechanics_dataset atomic_dataset \
  --save-judge-responses \
  --save-detailed-eval \
  --response-format xlsx \
  --max-output-tokens 2048
```

## 🛠️ Command-Line Parameters

### New Parameters

| Parameter | Description | Example | 
|-----------|-------------|---------|
| `--pass-custom-model` | HuggingFace repository for automatic model detection | `--pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct` |
| `--max-output-tokens` | Global override for maximum output tokens | `--max-output-tokens 2048` |
| `--save-detailed-eval` | Save comprehensive evaluation data | `--save-detailed-eval` |
| `--save-judge-responses` | Save raw LLM judge responses | `--save-judge-responses` |
| `--response-format` | Format for detailed responses | `--response-format xlsx` |
| `--use-vllm` | Enable VLLM backend for acceleration | `--use-vllm` |
| `--batch-size` | Batch size for VLLM inference | `--batch-size 4` |

### WandB Logger Script

The `scripts/wandb_logger.py` provides comprehensive experiment tracking with all the same parameters as `run.py` plus additional WandB-specific options:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--run-and-log` | Run evaluation and log to WandB | `--run-and-log` |
| `--log-all` | Log all existing results in work directory | `--log-all` |
| `--project` | WandB project name | `--project my-vlm-experiments` |
| `--tags` | Additional tags for WandB run | `--tags baseline experiment-1` |
| `--notes` | Notes for WandB run | `--notes "Testing new model"` |

### Usage Examples

```bash
# Complete evaluation with all new features
python run.py \
  --pass-custom-model Qwen/Qwen2.5-VL-7B-Instruct \
  --data MMBench_DEV_EN MMMU_DEV_VAL \
  --use-vllm --batch-size 4 \
  --max-output-tokens 1024 \
  --save-detailed-eval \
  --save-judge-responses \
  --response-format json \
  --verbose

# WandB logging with custom model
python scripts/wandb_logger.py \
  --run-and-log \
  --pass-custom-model microsoft/Phi-3-vision-128k-instruct \
  --data MMBench_DEV_EN \
  --project my-vlm-research \
  --tags phi3-vision baseline \
  --max-output-tokens 2048

# Batch upload existing results to WandB
python scripts/wandb_logger.py \
  --log-all \
  --work-dir ./outputs \
  --project vlm-benchmark-results
```

## 📊 Datasets, Models, and Evaluation Results

### Evaluation Results

**The performance numbers on our official multi-modal leaderboards can be downloaded from here!**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [**Download All DETAILED Results**](http://opencompass.openxlab.space/assets/OpenVLM.json).

Check **Supported Benchmarks** Tab in [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) to view all supported image & video benchmarks (70+).

Check **Supported LMMs** Tab in [**VLMEvalKit Features**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb) to view all supported LMMs, including commercial APIs, open-source models, and more (200+).

**Transformers Version Recommendation:**

Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`.
- **Please use** `transformers==4.36.2` **for**: `Moondream1`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`, `VILA Series`, `Llama-3-MixSenseV1_1`, `Parrot-7B`, `PLLaVA Series`.
- **Please use** `transformers==4.40.0` **for**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **Please use** `transformers==4.42.0` **for**: `AKI`.
- **Please use** `transformers==4.44.0` **for**: `Moondream2`, `H2OVL series`.
- **Please use** `transformers==4.45.0` **for**: `Aria`.
- **Please use** `transformers==latest` **for**: `LLaVA-Next series`, `PaliGemma-3B`, `Chameleon series`, `Video-LLaVA-7B-HF`, `Ovis series`, `Mantis series`, `MiniCPM-V2.6`, `OmChat-v2.0-13B-sinlge-beta`, `Idefics-3`, `GLM-4v-9B`, `VideoChat2-HD`, `RBDash_72b`, `Llama-3.2 series`, `Kosmos series`.

**Torchvision Version Recommendation:**

Note that some VLMs may not be able to run under certain torchvision versions, we recommend the following settings to evaluate each VLM:

- **Please use** `torchvision>=0.16` **for**: `Moondream series` and `Aria`

**Flash-attn Version Recommendation:**

Note that some VLMs may not be able to run under certain flash-attention versions, we recommend the following settings to evaluate each VLM:

- **Please use** `pip install flash-attn --no-build-isolation` **for**: `Aria`

```python
# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.
```

## 🛠️ Development Guide

To develop custom benchmarks, VLMs, or simply contribute other codes to **VLMEvalKit**, please refer to [[Development_Guide](/docs/en/Development.md) | [开发指南](/docs/zh-CN/Development.md)].

**Call for contributions**

To promote the contribution from the community and share the corresponding credit (in the next report update):

- All Contributions will be acknowledged in the report.
- Contributors with 3 or more major contributions (implementing an MLLM, benchmark, or major feature) can join the author list of [VLMEvalKit Technical Report](https://www.arxiv.org/abs/2407.11691) on ArXiv. Eligible contributors can create an issue or dm kennyutc in [VLMEvalKit Discord Channel](https://discord.com/invite/evDT4GZmxN).

Here is a [contributor list](/docs/en/Contributors.md) we curated based on the records.

## 🎯 The Goal of VLMEvalKit

**The codebase is designed to:**

1. Provide an **easy-to-use**, **opensource evaluation toolkit** to make it convenient for researchers & developers to evaluate existing LVLMs and make evaluation results **easy to reproduce**.
2. Make it easy for VLM developers to evaluate their own models. To evaluate the VLM on multiple supported benchmarks, one just need to **implement a single `generate_inner()` function**, all other workloads (data downloading, data preprocessing, prediction inference, metric calculation) are handled by the codebase.

**The codebase is not designed to:**

1. Reproduce the exact accuracy number reported in the original papers of all **3rd party benchmarks**. The reason can be two-fold:
   1. VLMEvalKit uses **generation-based evaluation** for all VLMs (and optionally with **LLM-based answer extraction**). Meanwhile, some benchmarks may use different approaches (SEEDBench uses PPL-based evaluation, *eg.*). For those benchmarks, we compare both scores in the corresponding result. We encourage developers to support other evaluation paradigms in the codebase.
   2. By default, we use the same prompt template for all VLMs to evaluate on a benchmark. Meanwhile, **some VLMs may have their specific prompt templates** (some may not covered by the codebase at this time). We encourage VLM developers to implement their own prompt template in VLMEvalKit, if that is not covered currently. That will help to improve the reproducibility.

## 🖊️ Citation

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

If you wish to refer to DCVLR in your work, please cite the following:

```bib
@misc{DCVLR: Data Curation for Vision-Language Reasoning,
  author = {Feuer, Benjamin and Tripathi, Rohun and Elachqar, Oussama and Zhang, Yuhui and Hulkund, Neha and Nguyen, Thao and Shabtay, Nimrod and Udandarao, Vishaal and Wang, Xiaohan and Webb, Stefan and Koukoumidis, Emmanouil and Schmidt, Ludwig and Xie, Saining and Yeung-Levy, Serena and Liang, Paul and Beery, Sara and Gkioxari, Georgia}
  month = June,
  title = {{DCVLR}},
  year = {2025}
}
```

Please use the following BibTeX entry to cite VLMEvalKit:

```bib
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>

[github-contributors-link]: https://github.com/open-compass/VLMEvalKit/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/VLMEvalKit?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/VLMEvalKit/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/VLMEvalKit?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/VLMEvalKit/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/VLMEvalKit?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/VLMEvalKit/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/VLMEvalKit?color=white&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/VLMEvalKit/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/VLMEvalKit?color=ffcb47&labelColor=black&style=flat-square
