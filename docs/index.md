# MLX Forge

**Fine-tune, experiment with, and run LLMs locally on your Mac.**

MLX Forge is a comprehensive LoRA/DoRA/Full fine-tuning framework built on Apple's [MLX](https://github.com/ml-explore/mlx) framework, optimized for Apple Silicon.

## Features

- **Multiple Training Methods**: LoRA, DoRA, Full Fine-Tuning, DPO, GRPO
- **Browser-Based Studio UI**: Training wizard, live loss charts, inference playground
- **OpenAI-Compatible API**: Serve fine-tuned models with `/v1/chat/completions`
- **HuggingFace Integration**: Load 200k+ datasets, push models to Hub
- **GGUF Export**: Deploy to Ollama/llama.cpp
- **QLoRA Support**: 4-bit quantized fine-tuning for memory efficiency
- **Sequence Packing**: Train faster with packed sequences
- **7 Model Architectures**: Llama, Qwen2/3, Phi-3/4, Gemma, Mistral

## Comparison

| Feature | MLX Forge | mlx-lm | mlx-tune |
|---------|-----------|--------|----------|
| Studio UI | Yes | No | No |
| OpenAI API | Yes | Yes | No |
| LoRA | Yes | Yes | Yes |
| DoRA | Yes | Yes | No |
| Full FT | Yes | Yes | No |
| GRPO | Yes | No | Yes |
| GGUF Export | Yes | Yes | No |
| HF Datasets | Yes | Yes | No |
| Hub Upload | Yes | Yes | No |
| Sequence Packing | Yes | No | No |
| Job Queue | Yes | No | No |

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
