# MLX Forge

**Fine-tune, experiment with, and run LLMs locally on your Mac.**

MLX Forge is a comprehensive fine-tuning framework built on Apple's [MLX](https://github.com/ml-explore/mlx), optimized for Apple Silicon. It supports 25+ model architectures, 8 training methods, speculative decoding, vision models, and streaming datasets.

## Features

- **8 Training Methods**: LoRA, DoRA, Full Fine-Tuning, DPO, GRPO, ORPO, KTO, SimPO
- **25+ Architectures**: Llama, Qwen, Gemma, Phi, Mixtral, DeepSeek V2/V3, Mamba, Cohere, and more
- **Browser-Based Studio UI**: Training wizard, live loss charts, inference playground
- **OpenAI-Compatible API**: Serve fine-tuned models with `/v1/chat/completions`
- **Speculative Decoding**: 1.5-2x faster inference with draft models
- **Vision Model Support**: Fine-tune and run VLMs via mlx-vlm integration
- **HuggingFace Integration**: Load 200k+ datasets, push models to Hub
- **GGUF Quantized Export**: Q4_0/Q8_0 for Ollama/llama.cpp deployment
- **QLoRA Support**: 4-bit quantized fine-tuning for memory efficiency
- **Streaming Data**: Train on datasets that don't fit in RAM
- **Auto Memory Safety**: Adjusts batch size and checkpointing per hardware

## Comparison

| Feature | MLX Forge | mlx-lm | mlx-tune |
|---------|-----------|--------|----------|
| Studio UI | Yes | No | No |
| OpenAI API | Yes | Yes | No |
| LoRA / DoRA | Yes | Yes | Yes / No |
| Full FT | Yes | Yes | No |
| DPO | Yes | No | Yes |
| GRPO | Yes | No | Yes |
| ORPO / KTO / SimPO | Yes | No | Yes |
| Speculative Decoding | Yes | Yes | No |
| Vision Training | Yes | No | No |
| GGUF Quantized Export | Yes | Yes | No |
| Streaming Data | Yes | No | No |
| HF Datasets | Yes | Yes | No |
| Hub Upload | Yes | Yes | No |
| Sequence Packing | Yes | No | No |
| Job Queue | Yes | No | No |
| Auto Memory Safety | Yes | No | No |
| Architectures | 25+ | 40+ | 10+ |

## Quick Start

```bash
pip install mlx-forge

# Train a model
mlx-forge train --config config.yaml

# Launch Studio UI
mlx-forge studio

# Serve with OpenAI API
mlx-forge serve --model Qwen/Qwen3-0.6B
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX 0.18.0+
