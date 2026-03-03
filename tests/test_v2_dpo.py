"""Tests for V2 DPO training (losses, data format, trainer).

V2: Uses per-token labels (-100 masking) instead of offset-based masking.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest


# ── Loss Functions ──────────────────────────────────────────────────────────


class TestSFTLoss:
    """Test that SFT loss works with per-token labels."""

    def test_sft_loss_basic(self):
        """SFT loss computes cross-entropy on non-masked tokens only."""
        from lmforge.losses.sft import SFTLoss

        loss_fn = SFTLoss()

        model = _make_dummy_model(vocab_size=10)
        input_ids = mx.array([[1, 2, 3, 4, 5, 0, 0]], dtype=mx.int32)
        labels = mx.array([[-100, -100, 3, 4, 5, -100, -100]], dtype=mx.int32)

        loss, ntoks = loss_fn(model, input_ids, labels)
        mx.eval(loss, ntoks)

        assert loss.item() > 0
        assert ntoks.item() > 0

    def test_sft_loss_backward_compat(self):
        """Module-level loss_fn still works."""
        from lmforge.losses.sft import loss_fn

        model = _make_dummy_model(vocab_size=10)
        input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

        loss, ntoks = loss_fn(model, input_ids, labels)
        mx.eval(loss, ntoks)

        assert loss.item() > 0

    def test_sft_packed_loss(self):
        """Packed SFT loss respects segment boundaries."""
        from lmforge.losses.sft import SFTLoss

        loss_fn = SFTLoss()
        model = _make_dummy_model(vocab_size=10)

        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 0]], dtype=mx.int32)
        labels = mx.array([[-100, 2, 3, -100, 5, 6, 7, -100]], dtype=mx.int32)
        segment_ids = mx.array([[0, 0, 0, 1, 1, 1, 1, -1]], dtype=mx.int32)

        loss, ntoks = loss_fn.packed(model, input_ids, labels, segment_ids)
        mx.eval(loss, ntoks)

        assert loss.item() > 0
        assert ntoks.item() > 0

    def test_trainer_loss_compat(self):
        """Trainer module-level functions still work."""
        from lmforge.trainer.trainer import loss_fn, loss_fn_packed

        model = _make_dummy_model(vocab_size=10)
        input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

        loss, ntoks = loss_fn(model, input_ids, labels)
        mx.eval(loss, ntoks)
        assert loss.item() > 0


class TestDPOLoss:
    """Test DPO loss implementation with per-token labels."""

    def test_dpo_loss_reference_free(self):
        """SimPO (reference-free DPO) computes loss without reference model."""
        from lmforge.losses.dpo import DPOLoss

        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        model = _make_dummy_model(vocab_size=10)

        chosen_ids = mx.array([[1, 2, 3, 4, 5, 0, 0, 0]], dtype=mx.int32)
        chosen_labels = mx.array([[-100, -100, 3, 4, 5, -100, -100, -100]], dtype=mx.int32)
        rejected_ids = mx.array([[1, 2, 6, 7, 8, 9, 0, 0]], dtype=mx.int32)
        rejected_labels = mx.array([[-100, -100, 6, 7, 8, 9, -100, -100]], dtype=mx.int32)

        loss, ntoks = loss_fn(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        mx.eval(loss, ntoks)

        assert loss.item() > 0
        assert ntoks.item() > 0

    def test_dpo_loss_standard_requires_reference(self):
        """Standard DPO raises error without reference log-probs."""
        from lmforge.losses.dpo import DPOLoss

        loss_fn = DPOLoss(beta=0.1, reference_free=False)
        model = _make_dummy_model(vocab_size=10)

        chosen_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        chosen_labels = mx.array([[-100, -100, 3, 4, 5]], dtype=mx.int32)
        rejected_ids = mx.array([[1, 2, 6, 7, 8]], dtype=mx.int32)
        rejected_labels = mx.array([[-100, -100, 6, 7, 8]], dtype=mx.int32)

        with pytest.raises(ValueError, match="reference model"):
            loss_fn(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels)

    def test_dpo_loss_standard_with_reference(self):
        """Standard DPO works with reference log-probs."""
        from lmforge.losses.dpo import DPOLoss

        loss_fn = DPOLoss(beta=0.1, reference_free=False)
        model = _make_dummy_model(vocab_size=10)

        chosen_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        chosen_labels = mx.array([[-100, -100, 3, 4, 5]], dtype=mx.int32)
        rejected_ids = mx.array([[1, 2, 6, 7, 8]], dtype=mx.int32)
        rejected_labels = mx.array([[-100, -100, 6, 7, 8]], dtype=mx.int32)

        ref_chosen_logps = mx.array([-2.0])
        ref_rejected_logps = mx.array([-3.0])

        loss, ntoks = loss_fn(
            model, chosen_ids, chosen_labels, rejected_ids, rejected_labels,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )
        mx.eval(loss, ntoks)

        assert loss.item() > 0

    def test_dpo_beta_affects_loss(self):
        """Higher beta produces different loss value."""
        from lmforge.losses.dpo import DPOLoss

        model = _make_dummy_model(vocab_size=10)
        chosen_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        chosen_labels = mx.array([[-100, -100, 3, 4, 5]], dtype=mx.int32)
        rejected_ids = mx.array([[1, 2, 6, 7, 8]], dtype=mx.int32)
        rejected_labels = mx.array([[-100, -100, 6, 7, 8]], dtype=mx.int32)

        loss_low = DPOLoss(beta=0.01, reference_free=True)
        loss_high = DPOLoss(beta=1.0, reference_free=True)

        l1, _ = loss_low(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        l2, _ = loss_high(model, chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        mx.eval(l1, l2)

        assert l1.item() != l2.item()

    def test_sequence_logprobs(self):
        """_sequence_logprobs returns correct shapes."""
        from lmforge.losses.dpo import DPOLoss

        loss_fn = DPOLoss()
        model = _make_dummy_model(vocab_size=10)

        input_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        labels = mx.array([[-100, -100, 3, 4, 5]], dtype=mx.int32)

        logps, ntoks = loss_fn._sequence_logprobs(model, input_ids, labels)
        mx.eval(logps, ntoks)

        assert logps.shape == (1,)  # One per sequence
        assert logps.item() <= 0  # Log-probs are non-positive


# ── Data Format ─────────────────────────────────────────────────────────────


class TestPreferenceFormat:
    """Test preference pair format detection and validation."""

    def test_detect_preference_format(self):
        """Detects preference format from chosen/rejected keys."""
        from lmforge.data.formats import detect_format

        samples = [{
            "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "good"}],
            "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
        }]
        assert detect_format(samples) == "preference"

    def test_preference_over_chat(self):
        """Preference format takes priority even if 'messages' is also present."""
        from lmforge.data.formats import detect_format

        samples = [{
            "chosen": [{"role": "user", "content": "hi"}],
            "rejected": [{"role": "user", "content": "hi"}],
            "messages": [{"role": "user", "content": "hi"}],
        }]
        assert detect_format(samples) == "preference"

    def test_validate_preference_valid(self):
        """Valid preference samples pass validation."""
        from lmforge.data.formats import validate_samples

        samples = [{
            "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}],
        }]
        errors = validate_samples(samples, "preference")
        assert errors == []

    def test_validate_preference_missing_chosen(self):
        """Missing 'chosen' field is caught."""
        from lmforge.data.formats import validate_samples

        samples = [{"rejected": [{"role": "user", "content": "hi"}]}]
        errors = validate_samples(samples, "preference")
        assert any("chosen" in e for e in errors)

    def test_validate_preference_bad_message(self):
        """Bad messages in chosen/rejected are caught."""
        from lmforge.data.formats import validate_samples

        samples = [{"chosen": [{"role": "user"}], "rejected": [{"content": "hi"}]}]
        errors = validate_samples(samples, "preference")
        assert len(errors) > 0

    def test_validate_preference_empty_list(self):
        """Empty chosen/rejected lists are caught."""
        from lmforge.data.formats import validate_samples

        samples = [{"chosen": [], "rejected": []}]
        errors = validate_samples(samples, "preference")
        assert len(errors) == 2  # Both empty

    def test_original_formats_unchanged(self):
        """Chat, completions, text detection still works."""
        from lmforge.data.formats import detect_format

        assert detect_format([{"messages": []}]) == "chat"
        assert detect_format([{"prompt": "x", "completion": "y"}]) == "completions"
        assert detect_format([{"text": "hello"}]) == "text"

    def test_unknown_format_error_includes_preference(self):
        """Error message for unknown format mentions preference keys."""
        from lmforge.data.formats import detect_format

        with pytest.raises(ValueError, match="chosen.*rejected"):
            detect_format([{"foo": "bar"}])


# ── Preference Tokenization ─────────────────────────────────────────────────


class TestPreferenceTokenization:
    """Test preference pair tokenization with per-token labels."""

    def test_tokenize_preference(self):
        """Preference tokenization produces chosen/rejected with labels."""
        from lmforge.data.preprocessing import tokenize_dataset

        tokenizer = _make_mock_tokenizer()
        samples = [{
            "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "good"}],
            "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
        }]

        result = tokenize_dataset(samples, tokenizer, "preference")
        assert len(result) == 1
        assert "chosen_input_ids" in result[0]
        assert "rejected_input_ids" in result[0]
        assert "chosen_labels" in result[0]
        assert "rejected_labels" in result[0]

    def test_tokenize_preference_labels(self):
        """Preference tokenization produces labels with -100 masking."""
        from lmforge.data.preprocessing import tokenize_dataset

        tokenizer = _make_mock_tokenizer()
        samples = [{
            "chosen": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "good"}],
            "rejected": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bad"}],
        }]

        result = tokenize_dataset(samples, tokenizer, "preference", mask_prompt=True)
        # Labels should have some -100 values (masked prompt)
        assert -100 in result[0]["chosen_labels"]
        assert -100 in result[0]["rejected_labels"]


# ── Preference Batching ──────────────────────────────────────────────────────


class TestPreferenceBatching:
    """Test preference pair batching with per-token labels."""

    def test_iterate_preference_batches_shapes(self):
        """Preference batches have correct shapes."""
        from lmforge.data.batching import iterate_preference_batches

        dataset = [
            {"chosen_input_ids": list(range(10)), "chosen_labels": [-100, -100, -100] + list(range(3, 10)),
             "rejected_input_ids": list(range(8)), "rejected_labels": [-100, -100, -100] + list(range(3, 8))}
            for _ in range(8)
        ]
        config = _make_config(batch_size=4)

        batches = list(iterate_preference_batches(dataset, config))
        assert len(batches) == 2

        chosen_ids, chosen_labels, rejected_ids, rejected_labels = batches[0]
        assert chosen_ids.shape[0] == 4
        assert chosen_labels.shape[0] == 4
        assert rejected_ids.shape[0] == 4
        assert rejected_labels.shape[0] == 4
        # input_ids and labels should have same shape
        assert chosen_ids.shape == chosen_labels.shape
        assert rejected_ids.shape == rejected_labels.shape

    def test_preference_batch_padding(self):
        """Preference batches are padded to multiple of 32."""
        from lmforge.data.batching import iterate_preference_batches

        dataset = [
            {"chosen_input_ids": list(range(10)), "chosen_labels": list(range(10)),
             "rejected_input_ids": list(range(15)), "rejected_labels": list(range(15))}
            for _ in range(4)
        ]
        config = _make_config(batch_size=4)

        batches = list(iterate_preference_batches(dataset, config))
        _, _, rejected_ids, _ = batches[0]

        # Max len is 15, padded to 32
        assert rejected_ids.shape[1] == 32


# ── Config Extension ─────────────────────────────────────────────────────────


class TestConfigDPO:
    """Test DPO config extension."""

    def test_training_type_defaults_to_sft(self):
        """training_type defaults to 'sft' for backward compatibility."""
        from lmforge.config import TrainingParams

        params = TrainingParams()
        assert params.training_type == "sft"

    def test_training_type_dpo(self):
        """training_type can be set to 'dpo'."""
        from lmforge.config import TrainingParams

        params = TrainingParams(training_type="dpo", steps_per_save=100)
        assert params.training_type == "dpo"

    def test_dpo_beta_default(self):
        """dpo_beta defaults to 0.1."""
        from lmforge.config import TrainingParams

        params = TrainingParams()
        assert params.dpo_beta == 0.1

    def test_dpo_reference_free_default(self):
        """dpo_reference_free defaults to True (SimPO)."""
        from lmforge.config import TrainingParams

        params = TrainingParams()
        assert params.dpo_reference_free is True

    def test_existing_config_backward_compat(self):
        """Existing V1 configs without training_type still work."""
        from lmforge.config import TrainingConfig
        import yaml

        config_yaml = """
schema_version: 1
model:
  path: test/model
adapter:
  preset: attention-qv
data:
  train: train.jsonl
  valid: val.jsonl
training:
  num_iters: 100
  steps_per_save: 100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config = TrainingConfig.from_yaml(f.name)

        assert config.training.training_type == "sft"
        assert config.training.dpo_beta == 0.1


# ── Trainer Refactor ─────────────────────────────────────────────────────────


class TestTrainerRefactor:
    """Test that trainer refactor preserved V1 behavior."""

    def test_trainer_alias(self):
        """Trainer is an alias for SFTTrainer."""
        from lmforge.trainer.trainer import Trainer, SFTTrainer

        assert Trainer is SFTTrainer

    def test_base_trainer_abstract(self):
        """BaseTrainer raises NotImplementedError for abstract methods."""
        from lmforge.trainer.trainer import BaseTrainer

        model = _make_dummy_model(vocab_size=10)
        config = _make_full_config()

        trainer = BaseTrainer(model, config, [], [],
                              checkpoint_manager=MagicMock())

        with pytest.raises(NotImplementedError):
            trainer._build_step_functions(None, None)

    def test_sft_trainer_creates(self):
        """SFTTrainer can be instantiated."""
        from lmforge.trainer.trainer import SFTTrainer

        model = _make_dummy_model(vocab_size=10)
        config = _make_full_config()

        trainer = SFTTrainer(model, config, [], [],
                             checkpoint_manager=MagicMock())
        assert trainer.loss is not None

    def test_dpo_trainer_creates(self):
        """DPOTrainer can be instantiated."""
        from lmforge.trainer.dpo_trainer import DPOTrainer

        model = _make_dummy_model(vocab_size=10)
        config = _make_full_config(training_type="dpo")

        trainer = DPOTrainer(model, config, [], [],
                             checkpoint_manager=MagicMock())
        assert trainer.loss is not None
        assert trainer.loss.reference_free is True

    def test_clip_grad_norm(self):
        """clip_grad_norm still works after refactor."""
        from lmforge.trainer.trainer import clip_grad_norm

        grads = {"w": mx.array([3.0, 4.0])}
        clipped = clip_grad_norm(grads, max_norm=1.0)
        mx.eval(clipped)

        norm = mx.sqrt((clipped["w"] ** 2).sum())
        mx.eval(norm)
        assert norm.item() <= 1.0 + 1e-5


# ── Helpers ──────────────────────────────────────────────────────────────────


class _DummyModel(nn.Module):
    """Minimal model for testing losses."""

    def __init__(self, vocab_size: int, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x, cache=None):
        return self.head(self.embed(x))


def _make_dummy_model(vocab_size: int = 10, hidden_dim: int = 32):
    model = _DummyModel(vocab_size, hidden_dim)
    mx.eval(model.parameters())
    return model


def _make_mock_tokenizer():
    """Create a mock tokenizer for testing."""
    mock = MagicMock()
    mock.eos_token_id = 0

    def apply_chat_template(messages, tokenize=True, add_generation_prompt=False):
        # Return increasing integers based on number of messages
        n = sum(len(m.get("content", "")) for m in messages)
        return list(range(1, n + 5))

    mock.apply_chat_template = apply_chat_template
    mock.encode = lambda text, add_special_tokens=True: list(range(1, len(text) + 1))
    return mock


@dataclass
class _MockDataConfig:
    train: str = "train.jsonl"
    valid: str = "val.jsonl"
    max_seq_length: int = 2048
    mask_prompt: bool = True
    packing: bool = False


@dataclass
class _MockTrainingParams:
    batch_size: int = 4
    num_iters: int = 100
    learning_rate: float = 1e-5
    optimizer: str = "adam"
    optimizer_config: dict = None
    lr_schedule: object = None
    grad_accumulation_steps: int = 1
    max_grad_norm: float = None
    seed: int = 42
    gradient_checkpointing: bool = False
    steps_per_report: int = 10
    steps_per_eval: int = 50
    steps_per_save: int = 50
    val_batches: int = 25
    keep_last_n_checkpoints: int = 3
    training_type: str = "sft"
    dpo_beta: float = 0.1
    dpo_reference_free: bool = True

    def __post_init__(self):
        if self.optimizer_config is None:
            self.optimizer_config = {}


@dataclass
class _MockRuntimeConfig:
    run_dir: str = "~/.lmforge/runs"
    eager: bool = True
    report_to: str = None
    wandb_project: str = None


@dataclass
class _MockConfig:
    schema_version: int = 1
    model: object = None
    adapter: object = None
    data: object = None
    training: object = None
    runtime: object = None


def _make_config(batch_size: int = 4):
    return _MockConfig(
        data=_MockDataConfig(),
        training=_MockTrainingParams(batch_size=batch_size),
    )


def _make_full_config(training_type: str = "sft"):
    return _MockConfig(
        data=_MockDataConfig(),
        training=_MockTrainingParams(training_type=training_type),
        runtime=_MockRuntimeConfig(),
    )
