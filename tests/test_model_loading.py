"""Tests for self-contained model loading (M8)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from cortexlab.models._base import BaseModelArgs
from cortexlab.models.registry import (
    get_model_classes,
    is_supported,
    list_supported_architectures,
)


class TestBaseModelArgs:
    """Tests for BaseModelArgs.from_dict()."""

    def test_from_dict_filters_unknown_keys(self):
        """Test that from_dict filters out unknown keys."""

        @dataclass
        class TestArgs(BaseModelArgs):
            hidden_size: int
            num_layers: int

        config = {
            "hidden_size": 1024,
            "num_layers": 12,
            "unknown_key": "should be ignored",
            "another_unknown": 42,
        }

        args = TestArgs.from_dict(config)
        assert args.hidden_size == 1024
        assert args.num_layers == 12
        assert not hasattr(args, "unknown_key")
        assert not hasattr(args, "another_unknown")

    def test_from_dict_with_all_known_keys(self):
        """Test from_dict when all keys are known."""

        @dataclass
        class TestArgs(BaseModelArgs):
            hidden_size: int
            num_layers: int
            dropout: float = 0.1

        config = {"hidden_size": 512, "num_layers": 6, "dropout": 0.2}

        args = TestArgs.from_dict(config)
        assert args.hidden_size == 512
        assert args.num_layers == 6
        assert args.dropout == 0.2

    def test_from_dict_uses_defaults(self):
        """Test that from_dict uses default values for missing optional fields."""

        @dataclass
        class TestArgs(BaseModelArgs):
            hidden_size: int
            dropout: float = 0.1

        config = {"hidden_size": 256}

        args = TestArgs.from_dict(config)
        assert args.hidden_size == 256
        assert args.dropout == 0.1


class TestModelRegistry:
    """Tests for the model registry."""

    def test_list_supported_architectures(self):
        """Test that we can list supported architectures."""
        supported = list_supported_architectures()
        assert isinstance(supported, list)
        assert len(supported) > 0
        assert "llama" in supported
        assert "qwen3" in supported

    def test_is_supported_for_known_types(self):
        """Test is_supported returns True for known types."""
        assert is_supported("llama") is True
        assert is_supported("qwen3") is True

    def test_is_supported_for_remapped_types(self):
        """Test is_supported handles remapped types."""
        assert is_supported("mistral") is True  # Remaps to llama

    def test_is_supported_for_unknown_types(self):
        """Test is_supported returns False for unknown types."""
        assert is_supported("unknown_model") is False
        assert is_supported("gpt4") is False

    def test_get_model_classes_for_qwen3(self):
        """Test getting model classes for qwen3."""
        config = {"model_type": "qwen3"}
        Model, ModelArgs = get_model_classes(config)

        assert Model is not None
        assert ModelArgs is not None
        assert hasattr(Model, "__call__")
        assert hasattr(ModelArgs, "from_dict")

    def test_get_model_classes_for_llama(self):
        """Test getting model classes for llama."""
        config = {"model_type": "llama"}
        Model, ModelArgs = get_model_classes(config)

        assert Model is not None
        assert ModelArgs is not None

    def test_get_model_classes_for_remapped_type(self):
        """Test that remapped types (mistral -> llama) work."""
        config = {"model_type": "mistral"}
        Model, ModelArgs = get_model_classes(config)

        # Should return llama classes
        assert Model is not None
        assert ModelArgs is not None

    def test_get_model_classes_raises_for_unknown(self):
        """Test that unknown model types raise ValueError."""
        config = {"model_type": "unknown_architecture"}

        with pytest.raises(ValueError) as exc_info:
            get_model_classes(config)

        error_msg = str(exc_info.value)
        assert "not supported" in error_msg.lower()
        assert "unknown_architecture" in error_msg
        # Should list supported architectures
        assert "llama" in error_msg or "qwen3" in error_msg

    def test_get_model_classes_raises_for_missing_model_type(self):
        """Test that missing model_type raises ValueError."""
        config = {"hidden_size": 1024}  # No model_type

        with pytest.raises(ValueError) as exc_info:
            get_model_classes(config)

        assert "model_type" in str(exc_info.value).lower()


class TestModelInstantiation:
    """Tests for model instantiation with mock weights."""

    def test_qwen3_model_instantiation(self):
        """Test that Qwen3 model can be instantiated."""
        from cortexlab.models.architectures.qwen3 import Model, ModelArgs

        config = {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "head_dim": 16,
            "tie_word_embeddings": True,
            "rms_norm_eps": 1e-6,
        }

        args = ModelArgs.from_dict(config)
        model = Model(args)

        assert model is not None
        assert hasattr(model, "layers")
        assert len(model.layers) == 2

    def test_llama_model_instantiation(self):
        """Test that Llama model can be instantiated."""
        from cortexlab.models.architectures.llama import Model, ModelArgs

        config = {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
        }

        args = ModelArgs.from_dict(config)
        model = Model(args)

        assert model is not None
        assert hasattr(model, "layers")
        assert len(model.layers) == 2

    def test_model_has_layers_property(self):
        """Test that models expose layers property for LoRA targeting."""
        from cortexlab.models.architectures.qwen3 import Model, ModelArgs

        config = {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "head_dim": 16,
            "tie_word_embeddings": True,
            "rms_norm_eps": 1e-6,
        }

        args = ModelArgs.from_dict(config)
        model = Model(args)

        # Should be able to access layers for LoRA
        layers = model.layers
        assert len(layers) == 4
        assert hasattr(layers[0], "self_attn")
        assert hasattr(layers[0], "mlp")


class TestLoaderFunctions:
    """Tests for loader utility functions."""

    def test_load_config_file_not_found(self, tmp_path):
        """Test load_config raises FileNotFoundError for missing config."""
        from cortexlab.models.loader import load_config

        with pytest.raises(FileNotFoundError) as exc_info:
            load_config(tmp_path)

        assert "config.json" in str(exc_info.value)

    def test_load_config_success(self, tmp_path):
        """Test load_config successfully reads config.json."""
        from cortexlab.models.loader import load_config

        config_data = {"model_type": "qwen3", "hidden_size": 1024}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))

        result = load_config(tmp_path)
        assert result == config_data

    def test_load_weights_file_not_found(self, tmp_path):
        """Test load_weights raises FileNotFoundError for missing weights."""
        from cortexlab.models.loader import load_weights

        with pytest.raises(FileNotFoundError) as exc_info:
            load_weights(tmp_path)

        assert "safetensors" in str(exc_info.value).lower()


class TestSanitization:
    """Tests for model weight sanitization."""

    def test_qwen3_sanitize_removes_lm_head_when_tied(self):
        """Test that Qwen3 sanitize removes lm_head when embeddings are tied."""
        from cortexlab.models.architectures.qwen3 import Model, ModelArgs

        config = {
            "model_type": "qwen3",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1000,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "head_dim": 16,
            "tie_word_embeddings": True,
            "rms_norm_eps": 1e-6,
        }

        args = ModelArgs.from_dict(config)
        model = Model(args)

        weights = {
            "model.embed_tokens.weight": "tensor1",
            "lm_head.weight": "tensor2",  # Should be removed
            "model.layers.0.self_attn.q_proj.weight": "tensor3",
        }

        sanitized = model.sanitize(weights)
        assert "lm_head.weight" not in sanitized
        assert "model.embed_tokens.weight" in sanitized

    def test_llama_sanitize_removes_rotary_emb(self):
        """Test that Llama sanitize removes rotary_emb.inv_freq."""
        from cortexlab.models.architectures.llama import Model, ModelArgs

        config = {
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "vocab_size": 1000,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
        }

        args = ModelArgs.from_dict(config)
        model = Model(args)

        weights = {
            "model.embed_tokens.weight": "tensor1",
            "model.layers.0.self_attn.rotary_emb.inv_freq": "should_remove",
            "model.layers.0.self_attn.q_proj.weight": "tensor3",
        }

        sanitized = model.sanitize(weights)
        assert "rotary_emb.inv_freq" not in str(sanitized.keys())
        assert "model.embed_tokens.weight" in sanitized
