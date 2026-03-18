"""Tests for M33: Streaming data pipeline.

Tests cover:
- StreamingHFDataset (mocked)
- StreamingJSONLDataset
- Config streaming field
- Streaming batching integration
- tokenize_single format dispatch
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ── StreamingJSONLDataset Tests ─────────────────────────────────────────────

class TestStreamingJSONLDataset:
    def test_streaming_jsonl_no_len(self):
        """StreamingJSONLDataset has no __len__."""
        from mlx_forge.data.streaming import StreamingJSONLDataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
            f.flush()
            ds = StreamingJSONLDataset(f.name, tokenizer=MagicMock())
            assert not hasattr(ds, "__len__")

    def test_streaming_jsonl_cycles(self):
        """Cycles through file (infinite iterator)."""
        from mlx_forge.data.streaming import StreamingJSONLDataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
            f.flush()

            mock_tokenizer = MagicMock()
            ds = StreamingJSONLDataset(f.name, tokenizer=mock_tokenizer)

            # Mock tokenize_single to return valid data
            with patch("mlx_forge.data.preprocessing.tokenize_single") as mock_tok:
                mock_tok.return_value = {"input_ids": [1, 2, 3], "labels": [1, 2, 3]}
                it = iter(ds)
                items = [next(it) for _ in range(3)]
                assert len(items) == 3  # Cycled through single sample 3 times

    def test_streaming_jsonl_yields_dicts(self):
        """Yields tokenized dicts."""
        from mlx_forge.data.streaming import StreamingJSONLDataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "test"}) + "\n")
            f.flush()

            mock_tokenizer = MagicMock()
            ds = StreamingJSONLDataset(f.name, tokenizer=mock_tokenizer)

            with patch("mlx_forge.data.preprocessing.tokenize_single") as mock_tok:
                mock_tok.return_value = {"input_ids": [1, 2], "labels": [1, 2]}
                item = next(iter(ds))
                assert isinstance(item, dict)
                assert "input_ids" in item

    def test_streaming_jsonl_missing_file(self):
        """FileNotFoundError on missing file."""
        from mlx_forge.data.streaming import StreamingJSONLDataset

        with pytest.raises(FileNotFoundError):
            StreamingJSONLDataset("/nonexistent/file.jsonl", tokenizer=MagicMock())

    def test_streaming_jsonl_skip_bad_lines(self):
        """Skips malformed JSON lines."""
        from mlx_forge.data.streaming import StreamingJSONLDataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            f.write(json.dumps({"text": "valid"}) + "\n")
            f.flush()

            mock_tokenizer = MagicMock()
            ds = StreamingJSONLDataset(f.name, tokenizer=mock_tokenizer)

            with patch("mlx_forge.data.preprocessing.tokenize_single") as mock_tok:
                mock_tok.return_value = {"input_ids": [1, 2], "labels": [1, 2]}
                item = next(iter(ds))
                assert isinstance(item, dict)

    def test_streaming_jsonl_skip_empty_lines(self):
        """Skips empty lines."""
        from mlx_forge.data.streaming import StreamingJSONLDataset

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"text": "valid"}) + "\n")
            f.flush()

            mock_tokenizer = MagicMock()
            ds = StreamingJSONLDataset(f.name, tokenizer=mock_tokenizer)

            with patch("mlx_forge.data.preprocessing.tokenize_single") as mock_tok:
                mock_tok.return_value = {"input_ids": [1], "labels": [1]}
                item = next(iter(ds))
                assert isinstance(item, dict)


# ── StreamingHFDataset Tests (mocked) ───────────────────────────────────────

class TestStreamingHFDataset:
    def test_streaming_hf_dataset_no_len(self):
        """StreamingHFDataset has no __len__."""
        from mlx_forge.data.streaming import StreamingHFDataset

        assert not hasattr(StreamingHFDataset, "__len__")

    @patch("datasets.load_dataset")
    def test_streaming_hf_dataset_init(self, mock_load):
        """Can initialize with mocked datasets.load_dataset."""
        from mlx_forge.data.streaming import StreamingHFDataset

        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_load.return_value = mock_ds

        ds = StreamingHFDataset("test/dataset", tokenizer=MagicMock())
        mock_load.assert_called_once()
        assert ds.ds is mock_ds

    @patch("datasets.load_dataset")
    def test_streaming_hf_dataset_shuffle(self, mock_load):
        """Shuffle buffer applied."""
        from mlx_forge.data.streaming import StreamingHFDataset

        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_load.return_value = mock_ds

        ds = StreamingHFDataset("test/dataset", tokenizer=MagicMock(),
                                shuffle_buffer=500)
        mock_ds.shuffle.assert_called_once_with(buffer_size=500)

    @patch("datasets.load_dataset")
    def test_streaming_hf_dataset_columns(self, mock_load):
        """Column mapping works."""
        from mlx_forge.data.streaming import StreamingHFDataset

        mock_ds = MagicMock()
        mock_ds.shuffle.return_value = mock_ds
        mock_load.return_value = mock_ds

        columns = {"text": "content", "prompt": "input"}
        ds = StreamingHFDataset("test/dataset", tokenizer=MagicMock(),
                                columns=columns)
        assert ds.columns == columns

        # Test _convert_row
        row = {"content": "hello", "input": "test"}
        result = ds._convert_row(row)
        assert result == {"text": "hello", "prompt": "test"}


# ── Config Tests ────────────────────────────────────────────────────────────

class TestStreamingConfig:
    def test_config_streaming_default(self):
        """streaming defaults to False."""
        from mlx_forge.config import DataConfig
        dc = DataConfig(train="data.jsonl", valid="val.jsonl")
        assert dc.streaming is False

    def test_config_streaming_bool(self):
        """streaming accepts True."""
        from mlx_forge.config import DataConfig
        dc = DataConfig(train="data.jsonl", valid="val.jsonl", streaming=True)
        assert dc.streaming is True

    def test_data_config_streaming_field(self):
        """DataConfig has streaming field."""

        from mlx_forge.config import DataConfig
        # Check that streaming is in the model fields
        assert "streaming" in DataConfig.model_fields


# ── Batching Integration ───────────────────────────────────────────────────

class TestStreamingBatching:
    def test_streaming_batching_integration(self):
        """Streaming works with _iterate_batches_streaming."""
        from mlx_forge.data.batching import _iterate_batches_streaming

        def fake_stream():
            for i in range(10):
                yield {"input_ids": list(range(i, i + 8)),
                       "labels": list(range(i, i + 8))}

        batches = list(_iterate_batches_streaming(
            fake_stream(), batch_size=2, max_seq_length=16
        ))
        assert len(batches) > 0
        for input_ids, labels in batches:
            assert input_ids.shape[0] <= 2


# ── tokenize_single Tests ──────────────────────────────────────────────────

class TestTokenizeSingle:
    def test_tokenize_single_chat(self):
        """tokenize_single handles chat format."""
        from mlx_forge.data.preprocessing import tokenize_single

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 4, 5])
        tokenizer.encode = MagicMock(return_value=[1, 2])

        sample = {"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]}

        result = tokenize_single(sample, tokenizer, max_seq_length=64)
        assert result is not None
        assert "input_ids" in result

    def test_tokenize_single_text(self):
        """tokenize_single handles text format."""
        from mlx_forge.data.preprocessing import tokenize_single

        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        sample = {"text": "Hello world"}
        result = tokenize_single(sample, tokenizer, max_seq_length=64)
        assert result is not None
        assert "input_ids" in result

    def test_tokenize_single_completions(self):
        """tokenize_single handles completions format."""
        from mlx_forge.data.preprocessing import tokenize_single

        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        sample = {"prompt": "Question?", "completion": "Answer."}
        result = tokenize_single(sample, tokenizer, max_seq_length=64)
        assert result is not None
        assert "input_ids" in result

    def test_tokenize_single_unknown(self):
        """tokenize_single returns None for unknown format."""
        from mlx_forge.data.preprocessing import tokenize_single

        tokenizer = MagicMock()
        sample = {"unknown_key": "value"}
        result = tokenize_single(sample, tokenizer)
        assert result is None


# ── Import Tests ────────────────────────────────────────────────────────────

class TestStreamingImports:
    def test_streaming_imports(self):
        """streaming module importable."""
        from mlx_forge.data import streaming
        assert hasattr(streaming, "StreamingHFDataset")
        assert hasattr(streaming, "StreamingJSONLDataset")

    def test_streaming_in_train_dispatch(self):
        """Data __init__ or batching references streaming."""
        import inspect

        from mlx_forge.data import batching
        source = inspect.getsource(batching)
        assert "streaming" in source.lower() or "_iterate_batches_streaming" in source
