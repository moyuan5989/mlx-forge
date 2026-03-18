"""Tests for M22: GGUF Export."""

import json
import struct

import numpy as np
import pytest

from mlx_forge.export.gguf_constants import GGUF_MAGIC, GGUF_VERSION, GGMLType
from mlx_forge.export.weight_mapping import (
    SUPPORTED_GGUF_ARCHITECTURES,
    get_weight_map,
    translate_weight_name,
)


class TestWeightMapping:
    """Test weight name translation."""

    def test_llama_embedding(self):
        name = translate_weight_name("model.embed_tokens.weight", "llama")
        assert name == "token_embd.weight"

    def test_llama_layer_qkv(self):
        assert translate_weight_name("model.layers.0.self_attn.q_proj.weight", "llama") == "blk.0.attn_q.weight"
        assert translate_weight_name("model.layers.5.self_attn.k_proj.weight", "llama") == "blk.5.attn_k.weight"
        assert translate_weight_name("model.layers.31.self_attn.v_proj.weight", "llama") == "blk.31.attn_v.weight"

    def test_llama_mlp(self):
        assert translate_weight_name("model.layers.0.mlp.gate_proj.weight", "llama") == "blk.0.ffn_gate.weight"
        assert translate_weight_name("model.layers.0.mlp.up_proj.weight", "llama") == "blk.0.ffn_up.weight"
        assert translate_weight_name("model.layers.0.mlp.down_proj.weight", "llama") == "blk.0.ffn_down.weight"

    def test_llama_norm(self):
        assert translate_weight_name("model.norm.weight", "llama") == "output_norm.weight"
        assert translate_weight_name("model.layers.0.input_layernorm.weight", "llama") == "blk.0.attn_norm.weight"

    def test_llama_lm_head(self):
        assert translate_weight_name("lm_head.weight", "llama") == "output.weight"

    def test_mistral_uses_llama_map(self):
        assert translate_weight_name("model.embed_tokens.weight", "mistral") == "token_embd.weight"

    def test_qwen2_uses_llama_map(self):
        assert translate_weight_name("model.layers.0.self_attn.q_proj.weight", "qwen2") == "blk.0.attn_q.weight"

    def test_phi3_qkv_proj(self):
        assert translate_weight_name("model.layers.0.self_attn.qkv_proj.weight", "phi3") == "blk.0.attn_qkv.weight"

    def test_phi3_gate_up(self):
        assert translate_weight_name("model.layers.0.mlp.gate_up_proj.weight", "phi3") == "blk.0.ffn_gate_up.weight"

    def test_unknown_weight_returns_none(self):
        assert translate_weight_name("some.random.weight", "llama") is None

    def test_unsupported_architecture(self):
        with pytest.raises(ValueError, match="not supported"):
            get_weight_map("unsupported_arch")

    def test_supported_architectures(self):
        assert "llama" in SUPPORTED_GGUF_ARCHITECTURES
        assert "mistral" in SUPPORTED_GGUF_ARCHITECTURES
        assert "qwen2" in SUPPORTED_GGUF_ARCHITECTURES
        assert "phi3" in SUPPORTED_GGUF_ARCHITECTURES


class TestGGUFConstants:
    """Test GGUF format constants."""

    def test_magic_number(self):
        assert GGUF_MAGIC == 0x46475547

    def test_version(self):
        assert GGUF_VERSION == 3

    def test_tensor_types(self):
        assert GGMLType.F32 == 0
        assert GGMLType.F16 == 1


class TestGGUFWriter:
    """Test GGUF file writing."""

    def _create_model_dir(self, tmp_path, model_type="llama", num_layers=2, hidden=64, heads=4):
        """Create a minimal model directory for testing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Write config.json
        config = {
            "model_type": model_type,
            "hidden_size": hidden,
            "num_hidden_layers": num_layers,
            "num_attention_heads": heads,
            "num_key_value_heads": heads,
            "intermediate_size": hidden * 4,
            "max_position_embeddings": 2048,
            "vocab_size": 100,
            "rms_norm_eps": 1e-5,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        # Write minimal model.safetensors with numpy arrays
        from safetensors.numpy import save_file

        weights = {
            "model.embed_tokens.weight": np.random.randn(100, hidden).astype(np.float32),
            "model.norm.weight": np.random.randn(hidden).astype(np.float32),
            "lm_head.weight": np.random.randn(100, hidden).astype(np.float32),
        }
        for i in range(num_layers):
            weights[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
            weights[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
            weights[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
            weights[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(hidden, hidden).astype(np.float32)
            weights[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(hidden * 4, hidden).astype(np.float32)
            weights[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(hidden * 4, hidden).astype(np.float32)
            weights[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(hidden, hidden * 4).astype(np.float32)
            weights[f"model.layers.{i}.input_layernorm.weight"] = np.random.randn(hidden).astype(np.float32)
            weights[f"model.layers.{i}.post_attention_layernorm.weight"] = np.random.randn(hidden).astype(np.float32)

        save_file(weights, str(model_dir / "model.safetensors"))

        return model_dir

    def test_convert_creates_file(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = self._create_model_dir(tmp_path)
        output = tmp_path / "output.gguf"
        result = convert_to_gguf(model_dir, output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_gguf_header_magic(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = self._create_model_dir(tmp_path)
        output = tmp_path / "output.gguf"
        convert_to_gguf(model_dir, output)

        with open(output, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            version = struct.unpack("<I", f.read(4))[0]
        assert magic == GGUF_MAGIC
        assert version == GGUF_VERSION

    def test_gguf_tensor_count(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = self._create_model_dir(tmp_path, num_layers=1)
        output = tmp_path / "output.gguf"
        convert_to_gguf(model_dir, output)

        with open(output, "rb") as f:
            f.read(8)  # magic + version
            tensor_count = struct.unpack("<Q", f.read(8))[0]
        # 1 layer: 9 per-layer + 3 global = 12
        assert tensor_count == 12

    def test_f16_quantization(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = self._create_model_dir(tmp_path, num_layers=1)
        output_f16 = tmp_path / "f16.gguf"
        output_f32 = tmp_path / "f32.gguf"
        convert_to_gguf(model_dir, output_f16, quantization="f16")
        convert_to_gguf(model_dir, output_f32, quantization="f32")
        # F16 should be roughly half the size of F32
        assert output_f16.stat().st_size < output_f32.stat().st_size

    def test_unsupported_arch_error(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = {"model_type": "unsupported_model"}
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "model.safetensors").write_bytes(b"")

        with pytest.raises(ValueError, match="not supported"):
            convert_to_gguf(model_dir, tmp_path / "out.gguf")

    def test_missing_config_error(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = tmp_path / "empty_model"
        model_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="config.json"):
            convert_to_gguf(model_dir, tmp_path / "out.gguf")

    def test_missing_weights_error(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
        with pytest.raises(FileNotFoundError, match="model.safetensors"):
            convert_to_gguf(model_dir, tmp_path / "out.gguf")

    def test_invalid_quantization_error(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = self._create_model_dir(tmp_path)
        with pytest.raises(ValueError, match="Unsupported quantization"):
            convert_to_gguf(model_dir, tmp_path / "out.gguf", quantization="q3_k")

    def test_output_dir_created(self, tmp_path):
        from mlx_forge.export.gguf_writer import convert_to_gguf
        model_dir = self._create_model_dir(tmp_path)
        output = tmp_path / "nested" / "dir" / "model.gguf"
        convert_to_gguf(model_dir, output)
        assert output.exists()


class TestCLIIntegration:
    """Test CLI format flag."""

    def test_export_format_flag(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["export", "--run-id", "test", "--format", "gguf"])
        assert args.format == "gguf"

    def test_export_format_default(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["export", "--run-id", "test"])
        assert args.format == "safetensors"

    def test_export_format_choices(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "--run-id", "test", "--format", "invalid"])
