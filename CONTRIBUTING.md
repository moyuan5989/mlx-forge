# Contributing to MLX Forge

Thanks for your interest in contributing to MLX Forge! This guide covers the development setup, coding standards, and submission process.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/moyuan5989/mlx-forge.git
cd mlx-forge

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev,studio]"
```

## Running Tests

```bash
# Run the full test suite (510 tests)
.venv/bin/python -m pytest tests/ -v

# Run a specific test file
.venv/bin/python -m pytest tests/test_config.py -v

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=mlx_forge --cov-report=term-missing
```

All tests must pass before submitting a PR.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
# Check for issues
ruff check .

# Auto-fix
ruff check --fix .

# Format
ruff format .
```

Key conventions:
- Line length: 100 characters
- Python 3.10+ (use `X | Y` union syntax, not `Union[X, Y]`)
- Type hints for function signatures
- Pydantic v2 models with `extra="forbid"` for configs
- `from __future__ import annotations` in all modules

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <subject>

<optional body>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
- `feat: add Phi-4 architecture support`
- `fix: handle empty batches in sequence packing`
- `perf: optimize gradient accumulation loop`

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Run the full test suite and ruff
4. Submit a PR with a clear description of the change

PRs should:
- Include tests for new functionality
- Not break existing tests
- Follow existing code patterns and conventions
- Keep changes focused (one feature/fix per PR)

## Adding a New Architecture

1. **Create the architecture file** at `mlx_forge/models/architectures/newmodel.py`:

```python
from dataclasses import dataclass
from mlx_forge.models._base import BaseModelArgs

@dataclass
class NewModelArgs(BaseModelArgs):
    # Model-specific fields
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    # ...

    @classmethod
    def from_dict(cls, config: dict) -> "NewModelArgs":
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})

class NewModel(nn.Module):
    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        # inputs: (B, T) token IDs
        # returns: (B, T, vocab_size) logits
        ...
```

2. **Register it** in `mlx_forge/models/registry.py`:

```python
SUPPORTED_ARCHITECTURES = {
    ...
    "newmodel": "mlx_forge.models.architectures.newmodel",
}
```

3. **Add tests** in `tests/test_model_loading.py`

4. **Test with a real model** by downloading a small variant from HF

## Building the Frontend

The Studio frontend is a React + Vite app:

```bash
cd studio-frontend
npm install
npm run build
```

The build output goes to `mlx_forge/studio/frontend/` and is served by the FastAPI backend.

## Project Structure

```
mlx_forge/
├── adapters/           # LoRA targeting, application, fusing
├── cli/                # CLI commands (prepare, train, generate, studio, data)
├── config.py           # Pydantic config models
├── data/               # Data pipeline (formats, batching, packing, catalog)
├── inference/          # Text generation, sampling, KV cache
├── models/             # Model registry, loader, architectures
├── studio/             # FastAPI backend + React frontend
└── trainer/            # Training loop, checkpointing, callbacks
```

## Questions?

Open an [issue](https://github.com/moyuan5989/mlx-forge/issues) for bugs, feature requests, or questions.
