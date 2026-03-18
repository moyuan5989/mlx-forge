"""M14 — Adapter export and model merge tests."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_forge.adapters.fuse import fuse_model, save_fused_model
from mlx_forge.adapters.lora import LoRAEmbedding, LoRALinear

# ── Fuse correctness ──


class TestFuseModel:
    def test_fuse_lora_linear(self):
        """Fused weights = W + (scale/r) * B @ A."""
        base = nn.Linear(16, 32)
        mx.eval(base.parameters())
        lora = LoRALinear.from_base(base, r=4, scale=8.0)
        mx.eval(lora.parameters())

        # Compute expected fused weight
        expected = base.weight + (8.0 / 4) * (lora.lora_b @ lora.lora_a)
        mx.eval(expected)

        fused = lora.fuse()
        mx.eval(fused.parameters())

        diff = mx.abs(fused.weight - expected)
        assert mx.max(diff).item() < 1e-5

    def test_fuse_lora_embedding(self):
        """Fused embedding weights = W + (scale/r) * A @ B."""
        base = nn.Embedding(100, 32)
        mx.eval(base.parameters())
        lora = LoRAEmbedding.from_base(base, r=4, scale=8.0)
        mx.eval(lora.parameters())

        expected = base.weight + (8.0 / 4) * (lora.lora_a @ lora.lora_b)
        mx.eval(expected)

        fused = lora.fuse()
        mx.eval(fused.parameters())

        diff = mx.abs(fused.weight - expected)
        assert mx.max(diff).item() < 1e-5

    def test_fuse_model_replaces_lora_modules(self):
        """fuse_model should replace LoRA modules with plain Linear."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(16, 32)
                self.linear2 = nn.Linear(32, 16)

        model = TinyModel()
        mx.eval(model.parameters())

        # Wrap linear1 with LoRA
        lora1 = LoRALinear.from_base(model.linear1, r=4, scale=8.0)
        model.linear1 = lora1
        mx.eval(model.parameters())

        assert isinstance(model.linear1, LoRALinear)
        fuse_model(model)
        assert isinstance(model.linear1, nn.Linear)
        assert isinstance(model.linear2, nn.Linear)

    def test_fuse_no_lora(self):
        """Model with no LoRA modules should not change."""

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 32)

        model = TinyModel()
        mx.eval(model.parameters())
        fuse_model(model)
        assert isinstance(model.linear, nn.Linear)

    def test_fuse_preserves_output(self):
        """Fused model should produce same output as LoRA model."""
        base = nn.Linear(16, 32)
        mx.eval(base.parameters())

        lora = LoRALinear.from_base(base, r=4, scale=8.0)
        mx.eval(lora.parameters())

        x = mx.random.normal((2, 16))
        lora_out = lora(x)
        mx.eval(lora_out)

        # Get lora params before fuse
        lora_a = lora.lora_a
        lora_b = lora.lora_b

        fused = lora.fuse()
        mx.eval(fused.parameters())
        fused_out = fused(x)
        mx.eval(fused_out)

        diff = mx.max(mx.abs(lora_out - fused_out))
        assert diff.item() < 1e-4

    def test_fuse_with_bias(self):
        """Fusing should preserve bias."""
        base = nn.Linear(16, 32, bias=True)
        mx.eval(base.parameters())

        lora = LoRALinear.from_base(base, r=4, scale=8.0)
        mx.eval(lora.parameters())

        fused = lora.fuse()
        assert fused.bias is not None


# ── save_fused_model ──


class TestSaveFusedModel:
    def test_creates_safetensors(self, tmp_path):
        """save_fused_model should create model.safetensors."""
        model = nn.Linear(16, 32)
        mx.eval(model.parameters())

        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "config.json").write_text("{}")
        (tokenizer_dir / "tokenizer.json").write_text("{}")

        output = tmp_path / "output"
        save_fused_model(model, tokenizer_dir, output)

        assert (output / "model.safetensors").exists()
        assert (output / "config.json").exists()
        assert (output / "tokenizer.json").exists()

    def test_copies_tokenizer_files(self, tmp_path):
        """All matching tokenizer files should be copied."""
        model = nn.Linear(8, 8)
        mx.eval(model.parameters())

        tokenizer_dir = tmp_path / "tok"
        tokenizer_dir.mkdir()
        for f in ["config.json", "tokenizer_config.json", "special_tokens_map.json"]:
            (tokenizer_dir / f).write_text("{}")

        output = tmp_path / "out"
        save_fused_model(model, tokenizer_dir, output)

        assert (output / "config.json").exists()
        assert (output / "tokenizer_config.json").exists()
        assert (output / "special_tokens_map.json").exists()

    def test_missing_tokenizer_files_skipped(self, tmp_path):
        """Missing tokenizer files should not cause errors."""
        model = nn.Linear(8, 8)
        mx.eval(model.parameters())

        tokenizer_dir = tmp_path / "empty_tok"
        tokenizer_dir.mkdir()

        output = tmp_path / "out"
        save_fused_model(model, tokenizer_dir, output)
        assert (output / "model.safetensors").exists()


# ── CLI argument parsing ──


class TestExportCLI:
    def test_export_subcommand_exists(self):
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--run-id", "test-run"])
        assert args.command == "export"
        assert args.run_id == "test-run"
        assert args.output_dir is None
        assert args.checkpoint is None

    def test_export_with_all_args(self):
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "export",
            "--run-id", "my-run",
            "--output-dir", "/tmp/export",
            "--checkpoint", "step-100",
        ])
        assert args.run_id == "my-run"
        assert args.output_dir == "/tmp/export"
        assert args.checkpoint == "step-100"


# ── API endpoint ──


@pytest.fixture
def test_client():
    try:
        from httpx import ASGITransport, AsyncClient
    except ImportError:
        pytest.skip("httpx not installed")
    from mlx_forge.studio.server import create_app

    app = create_app(runs_dir="/tmp/mlxforge_test_export/runs")
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_export_endpoint_validates_run_id(test_client):
    r = await test_client.post("/api/v1/runs/..secret/export")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_export_endpoint_404_for_missing_run(test_client):
    r = await test_client.post("/api/v1/runs/nonexistent-run/export")
    assert r.status_code == 404
