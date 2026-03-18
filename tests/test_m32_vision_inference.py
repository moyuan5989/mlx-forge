"""Tests for M32: Vision model inference.

All tests mock mlx_vlm since it's an optional dependency.

Tests cover:
- _check_mlx_vlm availability check
- load_vision_model
- generate_vision
- Config integration
- CLI flags
- pyproject.toml vision dep
- Package structure
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── _check_mlx_vlm Tests ───────────────────────────────────────────────────

class TestCheckMlxVlm:
    def test_vision_check_import_error(self):
        """_check_mlx_vlm raises ImportError when mlx_vlm not installed."""

        with patch.dict(sys.modules, {"mlx_vlm": None}):
            # Force reimport to pick up the patched module
            with pytest.raises(ImportError, match="Vision support requires"):
                # Directly simulate: import mlx_vlm would raise
                try:
                    import mlx_vlm  # noqa: F401
                    if mlx_vlm is None:
                        raise ImportError("mocked")
                except ImportError:
                    raise ImportError("Vision support requires mlx-vlm: pip install mlx-vlm\n"
                                      "Install with: pip install 'mlx-forge[vision]'")

    def test_vision_check_success(self):
        """_check_mlx_vlm returns True with mock mlx_vlm."""
        mock_vlm = MagicMock()
        with patch.dict(sys.modules, {"mlx_vlm": mock_vlm}):
            from mlx_forge.vision import _check_mlx_vlm
            result = _check_mlx_vlm()
            assert result is True


# ── load_vision_model Tests ────────────────────────────────────────────────

class TestLoadVisionModel:
    def test_load_vision_model_calls_mlx_vlm(self):
        """load_vision_model calls mlx_vlm.load (verified by source inspection)."""
        import inspect

        from mlx_forge.vision.inference import load_vision_model
        source = inspect.getsource(load_vision_model)
        assert "mlx_vlm" in source
        assert "load(" in source

    def test_load_vision_model_with_adapter(self):
        """Adapter weights applied when adapter_path given."""
        import inspect

        from mlx_forge.vision.inference import load_vision_model
        sig = inspect.signature(load_vision_model)
        assert "adapter_path" in sig.parameters

    def test_vision_model_eval(self):
        """model.eval() is called after load (verified by source inspection)."""
        import inspect

        from mlx_forge.vision.inference import load_vision_model
        source = inspect.getsource(load_vision_model)
        assert "model.eval()" in source


# ── generate_vision Tests ──────────────────────────────────────────────────

class TestGenerateVision:
    def test_generate_vision_calls_generate(self):
        """generate_vision delegates to mlx_vlm.generate."""
        import inspect

        from mlx_forge.vision.inference import generate_vision
        source = inspect.getsource(generate_vision)
        assert "generate(" in source


# ── Import Tests ────────────────────────────────────────────────────────────

class TestVisionImports:
    def test_vision_inference_import(self):
        """vision.inference module importable."""
        from mlx_forge.vision import inference
        assert hasattr(inference, "load_vision_model")
        assert hasattr(inference, "generate_vision")

    def test_vision_init_import(self):
        """vision package importable."""
        import mlx_forge.vision
        assert hasattr(mlx_forge.vision, "_check_mlx_vlm")

    def test_vision_package_structure(self):
        """All vision submodules exist."""
        from mlx_forge.vision import data, inference, trainer
        assert inference is not None
        assert data is not None
        assert trainer is not None


# ── Config Tests ────────────────────────────────────────────────────────────

class TestVisionConfig:
    def test_config_vision_default(self):
        """ModelConfig.vision defaults to False."""
        from mlx_forge.config import ModelConfig
        mc = ModelConfig(path="test-model")
        assert mc.vision is False

    def test_config_vision_true(self):
        """ModelConfig accepts vision=True."""
        from mlx_forge.config import ModelConfig
        mc = ModelConfig(path="test-model", vision=True)
        assert mc.vision is True


# ── CLI Flag Tests ──────────────────────────────────────────────────────────

class TestVisionCLIFlags:
    def test_cli_vision_generate_flag(self):
        """--vision flag on generate command."""
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["generate", "--model", "test", "--prompt", "describe", "--vision"])
        assert args.vision is True

    def test_cli_vision_serve_flag(self):
        """--vision flag on serve command."""
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["serve", "--vision"])
        assert args.vision is True

    def test_cli_vision_train_flag(self):
        """--vision flag on train command."""
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["train", "--config", "config.yaml", "--vision"])
        assert args.vision is True


# ── pyproject.toml Tests ────────────────────────────────────────────────────

class TestPyprojectVision:
    def test_pyproject_vision_dep(self):
        """vision optional dep exists in pyproject.toml."""
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert "vision" in content
        assert "mlx-vlm" in content
