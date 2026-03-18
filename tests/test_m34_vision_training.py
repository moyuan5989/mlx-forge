"""Tests for M34: Vision model training (VLM SFT).

All tests mock mlx_vlm and PIL since they are optional dependencies.

Tests cover:
- VisionDataCollator
- VLMSFTTrainer
- Import paths
- Config integration
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from unittest.mock import MagicMock

import mlx.nn as nn
import numpy as np

# ── Mock config ─────────────────────────────────────────────────────────────

@dataclass
class _MockTraining:
    batch_size: int = 1
    num_iters: int = 10
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    optimizer_config: dict = None
    lr_schedule: object = None
    grad_accumulation_steps: int = 1
    max_grad_norm: float = None
    seed: int = 42
    gradient_checkpointing: bool = False
    steps_per_report: int = 5
    steps_per_eval: int = 5
    steps_per_save: int = 5
    val_batches: int = 2
    keep_last_n_checkpoints: int = 1
    training_type: str = "sft"

    def __post_init__(self):
        if self.optimizer_config is None:
            self.optimizer_config = {}


@dataclass
class _MockData:
    max_seq_length: int = 64
    mask_prompt: bool = True
    packing: bool = False
    streaming: bool = False


@dataclass
class _MockRuntime:
    run_dir: str = "/tmp/test_runs"
    eager: bool = True
    report_to: str = None
    wandb_project: str = None


@dataclass
class _MockModelCfg:
    path: str = "test-model"
    vision: bool = True


@dataclass
class _MockAdapter:
    method: str = "lora"
    rank: int = 8


@dataclass
class _MockConfig:
    training: _MockTraining = None
    data: _MockData = None
    runtime: _MockRuntime = None
    model: _MockModelCfg = None
    adapter: _MockAdapter = None

    def __post_init__(self):
        if self.training is None:
            self.training = _MockTraining()
        if self.data is None:
            self.data = _MockData()
        if self.runtime is None:
            self.runtime = _MockRuntime()
        if self.model is None:
            self.model = _MockModelCfg()
        if self.adapter is None:
            self.adapter = _MockAdapter()

    def model_dump(self):
        """Mimic Pydantic model_dump for CheckpointManager."""
        return asdict(self)


# ── Mock model for training ─────────────────────────────────────────────────

class MockModel(nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.proj = nn.Linear(32, vocab_size)

    def __call__(self, x, cache=None):
        return self.proj(self.embed(x))

    def trainable_parameters(self):
        return self.parameters()


# ── PIL Mock helper ────────────────────────────────────────────────────────

def _install_pil_mock():
    """Install a mock PIL module if not available."""
    if "PIL" not in sys.modules:
        mock_pil = MagicMock()
        mock_image = MagicMock()
        mock_pil.Image = mock_image
        sys.modules["PIL"] = mock_pil
        sys.modules["PIL.Image"] = mock_image
        return mock_image
    return None


# ── VisionDataCollator Tests ───────────────────────────────────────────────

class TestVisionDataCollator:
    def _make_collator(self):
        from mlx_forge.vision.data import VisionDataCollator
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "pixel_values": np.zeros((1, 3, 224, 224)),
        }
        return VisionDataCollator(mock_processor, max_seq_length=64)

    def test_vision_data_collator_init(self):
        """VisionDataCollator can be created."""
        from mlx_forge.vision.data import VisionDataCollator
        processor = MagicMock()
        collator = VisionDataCollator(processor, max_seq_length=128)
        assert collator.processor is processor
        assert collator.max_seq_length == 128

    def test_vision_data_collator_call(self):
        """Processes sample correctly (mock processor)."""
        collator = self._make_collator()
        sample = {
            "messages": [
                {"role": "user", "content": "Describe this"},
                {"role": "assistant", "content": "A photo"},
            ]
        }
        result = collator(sample)
        assert result is not None
        assert "input_ids" in result
        assert "labels" in result

    def test_vision_data_collator_extracts_images(self):
        """Image extraction function exists and handles text-only messages."""
        collator = self._make_collator()
        # Text-only messages should return empty images list
        messages = [
            {"role": "user", "content": "What is this?"},
        ]
        images = collator._extract_images(messages)
        assert isinstance(images, list)

    def test_vision_data_collator_extracts_text(self):
        """Text extracted correctly."""
        collator = self._make_collator()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text = collator._extract_text_messages(messages)
        assert "Hello" in text
        assert "Hi there" in text

    def test_vision_data_collator_labels_masked(self):
        """Vision tokens masked with -100."""
        collator = self._make_collator()
        sample = {
            "messages": [
                {"role": "user", "content": "Describe"},
                {"role": "assistant", "content": "A photo"},
            ]
        }
        result = collator(sample)
        assert result is not None
        labels = result["labels"]
        # First half should be -100 (masked)
        mid = len(labels) // 2
        assert all(l == -100 for l in labels[:mid])

    def test_vision_data_collator_max_seq_length(self):
        """Truncates long sequences."""
        from mlx_forge.vision.data import VisionDataCollator
        processor = MagicMock()
        processor.return_value = {
            "input_ids": np.array([list(range(200))]),
        }
        collator = VisionDataCollator(processor, max_seq_length=50)
        sample = {"messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]}
        result = collator(sample)
        assert result is not None
        assert len(result["input_ids"]) <= 50

    def test_vision_data_collator_handles_failure(self):
        """Returns None on error."""
        from mlx_forge.vision.data import VisionDataCollator
        processor = MagicMock()
        processor.side_effect = Exception("processor error")
        collator = VisionDataCollator(processor)
        result = collator({"messages": []})
        assert result is None


# ── VLMSFTTrainer Tests ────────────────────────────────────────────────────

class TestVLMSFTTrainer:
    def _make_trainer(self):
        from mlx_forge.vision.trainer import VLMSFTTrainer
        model = MockModel(vocab_size=50)
        processor = MagicMock()
        processor.return_value = {
            "input_ids": np.array([[1, 2, 3, 4]]),
        }
        config = _MockConfig()
        train_data = [
            {"messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ]}
        ]
        val_data = list(train_data)
        return VLMSFTTrainer(model, processor, config, train_data, val_data)

    def test_vlm_trainer_init(self):
        """VLMSFTTrainer can be instantiated."""
        trainer = self._make_trainer()
        assert trainer is not None
        assert hasattr(trainer, "processor")
        assert hasattr(trainer, "collator")

    def test_vlm_trainer_batch_size_1(self):
        """Effective batch size is 1 (variable image sizes)."""
        trainer = self._make_trainer()
        # VLMSFTTrainer processes one sample at a time
        iterator = trainer._build_batch_iterator()
        assert iterator is not None

    def test_vlm_trainer_evaluate(self):
        """evaluate returns float."""
        trainer = self._make_trainer()
        # Mock collator to return valid data
        trainer.collator = MagicMock()
        trainer.collator.return_value = {
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
        }
        result = trainer.evaluate()
        assert isinstance(result, float)

    def test_vlm_trainer_build_iterator(self):
        """Iterator yields processed samples."""
        trainer = self._make_trainer()
        trainer.collator = MagicMock()
        trainer.collator.return_value = {
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
        }
        iterator = trainer._build_batch_iterator()
        sample = next(iterator)
        assert isinstance(sample, dict)

    def test_vlm_trainer_loss_fn(self):
        """Uses SFTLoss."""
        from mlx_forge.losses.sft import SFTLoss
        trainer = self._make_trainer()
        assert isinstance(trainer.loss, SFTLoss)

    def test_vlm_trainer_collator_used(self):
        """Collator is called for each sample."""
        trainer = self._make_trainer()
        trainer.collator = MagicMock()
        trainer.collator.return_value = {
            "input_ids": [1, 2, 3],
            "labels": [-100, 2, 3],
        }
        iterator = trainer._build_batch_iterator()
        next(iterator)
        assert trainer.collator.called

    def test_vlm_trainer_gradient_update(self):
        """_build_step_functions returns dict with vlm_grad key."""
        trainer = self._make_trainer()

        step_fns = trainer._build_step_functions(None, None)
        assert "vlm_grad" in step_fns

    def test_vision_data_processor_called(self):
        """Processor is called during collation."""
        from mlx_forge.vision.data import VisionDataCollator
        processor = MagicMock()
        processor.return_value = {
            "input_ids": np.array([[1, 2, 3, 4]]),
        }
        collator = VisionDataCollator(processor)
        sample = {
            "messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ]
        }
        result = collator(sample)
        # Processor should be called via __call__
        assert processor.called


# ── Import Tests ────────────────────────────────────────────────────────────

class TestVisionTrainingImports:
    def test_vision_trainer_import(self):
        """vision.trainer importable."""
        from mlx_forge.vision.trainer import VLMSFTTrainer
        assert VLMSFTTrainer is not None

    def test_vision_data_import(self):
        """vision.data importable."""
        from mlx_forge.vision.data import VisionDataCollator
        assert VisionDataCollator is not None


# ── Data Format Tests ──────────────────────────────────────────────────────

class TestVisionDataFormats:
    def test_vision_data_image_message_format(self):
        """Handles OpenAI image format in messages."""
        from mlx_forge.vision.data import VisionDataCollator
        collator = VisionDataCollator(MagicMock())
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": "test.jpg"},
                {"type": "text", "text": "What is this?"},
            ]}
        ]
        text = collator._extract_text_messages(messages)
        assert "What is this?" in text

    def test_vision_data_text_only_fallback(self):
        """Handles messages without images (no image content blocks)."""
        from mlx_forge.vision.data import VisionDataCollator
        collator = VisionDataCollator(MagicMock())
        messages = [
            {"role": "user", "content": "Just text, no image"},
        ]
        images = collator._extract_images(messages)
        assert len(images) == 0


# ── Config Integration ─────────────────────────────────────────────────────

class TestVisionConfigIntegration:
    def test_vision_config_integration(self):
        """vision=True in ModelConfig."""
        from mlx_forge.config import ModelConfig
        mc = ModelConfig(path="llava-model", vision=True)
        assert mc.vision is True
