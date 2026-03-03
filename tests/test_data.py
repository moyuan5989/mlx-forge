"""Tests for data pipeline (V2 — per-token labels)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmforge.data.batching import iterate_batches
from lmforge.data.formats import detect_format, validate_samples


class TestFormatDetection:
    def test_detect_chat_format(self):
        """Test detection of chat format."""
        samples = [{"messages": [{"role": "user", "content": "Hello"}]}]
        assert detect_format(samples) == "chat"

    def test_detect_completions_format(self):
        """Test detection of completions format."""
        samples = [{"prompt": "Hello", "completion": "Hi there!"}]
        assert detect_format(samples) == "completions"

    def test_detect_text_format(self):
        """Test detection of text format."""
        samples = [{"text": "This is a sample text."}]
        assert detect_format(samples) == "text"

    def test_unknown_format_raises(self):
        """Test that unknown format raises ValueError with helpful message."""
        samples = [{"unknown_key": "value"}]
        with pytest.raises(ValueError) as exc_info:
            detect_format(samples)
        assert "unknown" in str(exc_info.value).lower()
        assert "unknown_key" in str(exc_info.value)


class TestFormatValidation:
    def test_validate_chat_samples(self):
        """Test validation of chat format samples."""
        valid_samples = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
        ]
        errors = validate_samples(valid_samples, "chat")
        assert len(errors) == 0

        invalid_samples = [{"text": "wrong format"}]
        errors = validate_samples(invalid_samples, "chat")
        assert len(errors) > 0
        assert "messages" in errors[0].lower()

        invalid_samples = [{"messages": "not a list"}]
        errors = validate_samples(invalid_samples, "chat")
        assert len(errors) > 0

        invalid_samples = [{"messages": [{"content": "Hello"}]}]
        errors = validate_samples(invalid_samples, "chat")
        assert len(errors) > 0
        assert "role" in errors[0].lower()

    def test_validate_completions_samples(self):
        """Test validation of completions format samples."""
        valid_samples = [{"prompt": "Hello", "completion": "Hi!"}]
        errors = validate_samples(valid_samples, "completions")
        assert len(errors) == 0

        invalid_samples = [{"completion": "Hi!"}]
        errors = validate_samples(invalid_samples, "completions")
        assert len(errors) > 0
        assert "prompt" in errors[0].lower()

        invalid_samples = [{"prompt": "Hello"}]
        errors = validate_samples(invalid_samples, "completions")
        assert len(errors) > 0
        assert "completion" in errors[0].lower()

    def test_validate_text_samples(self):
        """Test validation of text format samples."""
        valid_samples = [{"text": "Sample text"}]
        errors = validate_samples(valid_samples, "text")
        assert len(errors) == 0

        invalid_samples = [{"prompt": "wrong"}]
        errors = validate_samples(invalid_samples, "text")
        assert len(errors) > 0
        assert "text" in errors[0].lower()


class TestFingerprinting:
    def test_fingerprint_format(self, tmp_dir):
        """Test that fingerprint has correct format."""
        from lmforge.data.backend import compute_fingerprint

        data_file = tmp_dir / "test.jsonl"
        data_file.write_text('{"text": "sample"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        tokenizer = MockTokenizer()
        fingerprint = compute_fingerprint(str(data_file), tokenizer)

        assert fingerprint.startswith("sha256:")
        assert len(fingerprint) == 71  # "sha256:" + 64 hex chars

    def test_same_inputs_same_fingerprint(self, tmp_dir):
        """Test that same inputs produce same fingerprint."""
        from lmforge.data.backend import compute_fingerprint

        data_file = tmp_dir / "test.jsonl"
        data_file.write_text('{"text": "sample"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        tokenizer = MockTokenizer()
        fp1 = compute_fingerprint(str(data_file), tokenizer)
        fp2 = compute_fingerprint(str(data_file), tokenizer)

        assert fp1 == fp2

    def test_different_data_different_fingerprint(self, tmp_dir):
        """Test that different data produces different fingerprint."""
        from lmforge.data.backend import compute_fingerprint

        file1 = tmp_dir / "test1.jsonl"
        file1.write_text('{"text": "sample1"}\n')

        file2 = tmp_dir / "test2.jsonl"
        file2.write_text('{"text": "sample2"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        tokenizer = MockTokenizer()
        fp1 = compute_fingerprint(str(file1), tokenizer)
        fp2 = compute_fingerprint(str(file2), tokenizer)

        assert fp1 != fp2


class TestBatching:
    def test_batch_shapes_match_contract(self, sample_config_dict):
        """Test that batch shapes match V2 contract: (B, T) input_ids + (B, T) labels."""
        from lmforge.config import TrainingConfig

        dataset = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
            {"input_ids": [6, 7, 8, 9], "labels": [-100, -100, 8, 9]},
            {"input_ids": [10, 11, 12], "labels": [10, 11, 12]},
            {"input_ids": [13, 14, 15, 16], "labels": [-100, 14, 15, 16]},
        ]

        config = TrainingConfig(**sample_config_dict)
        batches = list(iterate_batches(dataset, config))

        assert len(batches) == 1  # 4 samples, batch_size=4

        input_ids, labels = batches[0]

        # Check shapes: both (B, T)
        assert input_ids.shape == (config.training.batch_size, input_ids.shape[1])
        assert labels.shape == (config.training.batch_size, labels.shape[1])
        assert input_ids.shape == labels.shape

        # Check dtypes
        assert "int32" in str(input_ids.dtype)
        assert "int32" in str(labels.dtype)

    def test_labels_padded_with_minus_100(self, sample_config_dict):
        """Labels should be padded with -100 (not 0)."""
        from lmforge.config import TrainingConfig
        import numpy as np

        dataset = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5, 6, 7, 8], "labels": [-100, -100, 6, 7, 8]},
            {"input_ids": [9, 10], "labels": [9, 10]},
            {"input_ids": [11, 12, 13, 14], "labels": [-100, 12, 13, 14]},
        ]

        config = TrainingConfig(**sample_config_dict)
        batches = list(iterate_batches(dataset, config))
        _, labels = batches[0]

        labels_np = np.array(labels)
        # Padding positions should be -100
        # Shortest sequence is length 2, padded to 32
        # So positions 2+ in that row should be -100
        assert (labels_np[:, -1] == -100).all()  # Last column is all padding

    def test_padding_to_multiple_of_32(self, sample_config_dict):
        """Test that sequences are padded to nearest multiple of 32."""
        from lmforge.config import TrainingConfig

        dataset = [
            {"input_ids": [1] * 37, "labels": [1] * 37},
            {"input_ids": [2] * 35, "labels": [2] * 35},
            {"input_ids": [3] * 30, "labels": [3] * 30},
            {"input_ids": [4] * 32, "labels": [4] * 32},
        ]

        config = TrainingConfig(**sample_config_dict)
        batches = list(iterate_batches(dataset, config))

        input_ids, _ = batches[0]
        T = input_ids.shape[1]

        assert T % 32 == 0
        assert T >= 37
        assert T == 64  # Next multiple of 32 after 37
