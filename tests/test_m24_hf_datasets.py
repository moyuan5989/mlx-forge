"""Tests for M24: HuggingFace Datasets Integration."""


import pytest

from mlx_forge.data.hf_loader import (
    apply_column_mapping,
    auto_detect_columns,
    save_as_jsonl,
)


class MockDataset:
    """Mock HF dataset for testing."""

    def __init__(self, data: list[dict], columns: list[str]):
        self._data = data
        self._columns = columns

    @property
    def column_names(self):
        return self._columns

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def select(self, indices):
        return MockDataset(
            [self._data[i] for i in indices],
            self._columns,
        )


class TestAutoDetectColumns:
    """Test column format auto-detection."""

    def test_detect_messages_format(self):
        ds = MockDataset([], ["messages", "id"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "messages"

    def test_detect_sharegpt_format(self):
        ds = MockDataset([], ["conversations", "source"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "sharegpt"

    def test_detect_alpaca_format(self):
        ds = MockDataset([], ["instruction", "input", "output"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "alpaca"
        assert "input" in mapping

    def test_detect_alpaca_without_input(self):
        ds = MockDataset([], ["instruction", "output"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "alpaca"

    def test_detect_text_format(self):
        ds = MockDataset([], ["text", "meta"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "text"

    def test_detect_preference_format(self):
        ds = MockDataset([], ["prompt", "chosen", "rejected"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "preference"

    def test_detect_qa_format(self):
        ds = MockDataset([], ["question", "answer", "context"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "qa"

    def test_detect_content_fallback(self):
        ds = MockDataset([], ["content", "id"])
        mapping = auto_detect_columns(ds)
        assert mapping["format"] == "text"

    def test_unknown_columns_raises(self):
        ds = MockDataset([], ["col_a", "col_b"])
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            auto_detect_columns(ds)


class TestApplyColumnMapping:
    """Test column mapping conversion."""

    def test_messages_format(self):
        ds = MockDataset(
            [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}],
            ["messages"],
        )
        result = apply_column_mapping(ds, {"format": "messages", "messages": "messages"})
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "user"

    def test_alpaca_format(self):
        ds = MockDataset(
            [{"instruction": "Translate to French", "input": "Hello", "output": "Bonjour"}],
            ["instruction", "input", "output"],
        )
        result = apply_column_mapping(ds, {"format": "alpaca", "instruction": "instruction", "output": "output", "input": "input"})
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "user"
        assert "Hello" in result[0]["messages"][0]["content"]
        assert result[0]["messages"][1]["content"] == "Bonjour"

    def test_sharegpt_format(self):
        ds = MockDataset(
            [{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}],
            ["conversations"],
        )
        result = apply_column_mapping(ds, {"format": "sharegpt", "conversations": "conversations"})
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_text_format(self):
        ds = MockDataset(
            [{"text": "Some training text"}],
            ["text"],
        )
        result = apply_column_mapping(ds, {"format": "text", "text": "text"})
        assert result[0]["text"] == "Some training text"

    def test_preference_format(self):
        ds = MockDataset(
            [{"prompt": "Q?", "chosen": "Good", "rejected": "Bad"}],
            ["prompt", "chosen", "rejected"],
        )
        result = apply_column_mapping(ds, {"format": "preference", "prompt": "prompt", "chosen": "chosen", "rejected": "rejected"})
        assert result[0]["prompt"] == "Q?"
        assert result[0]["chosen"] == "Good"

    def test_qa_format(self):
        ds = MockDataset(
            [{"question": "What?", "answer": "That."}],
            ["question", "answer"],
        )
        result = apply_column_mapping(ds, {"format": "qa", "question": "question", "answer": "answer"})
        assert result[0]["messages"][0]["content"] == "What?"


class TestSaveAsJsonl:
    """Test JSONL saving."""

    def test_save_creates_file(self, tmp_path):
        samples = [{"text": "hello"}, {"text": "world"}]
        path = save_as_jsonl(samples, tmp_path / "test.jsonl")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_save_creates_parent_dirs(self, tmp_path):
        path = save_as_jsonl([{"text": "hi"}], tmp_path / "sub" / "dir" / "test.jsonl")
        assert path.exists()


class TestConfigIntegration:
    """Test DataConfig changes for HF datasets."""

    def test_hf_dataset_field(self):
        from mlx_forge.config import DataConfig
        cfg = DataConfig(hf_dataset="tatsu-lab/alpaca")
        assert cfg.hf_dataset == "tatsu-lab/alpaca"
        assert cfg.hf_split == "train"

    def test_hf_dataset_with_split(self):
        from mlx_forge.config import DataConfig
        cfg = DataConfig(hf_dataset="dataset-id", hf_split="test")
        assert cfg.hf_split == "test"

    def test_hf_dataset_mutual_exclusion_with_train(self):
        from mlx_forge.config import DataConfig
        with pytest.raises(ValueError):
            DataConfig(train="data.jsonl", valid="val.jsonl", hf_dataset="dataset-id")

    def test_train_still_works(self):
        from mlx_forge.config import DataConfig
        cfg = DataConfig(train="data.jsonl", valid="val.jsonl")
        assert cfg.train == "data.jsonl"

    def test_hf_max_samples(self):
        from mlx_forge.config import DataConfig
        cfg = DataConfig(hf_dataset="ds", hf_max_samples=100)
        assert cfg.hf_max_samples == 100


class TestCLIIntegration:
    """Test CLI subcommand for HF import."""

    def test_hf_import_parser(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "data", "hf-import", "tatsu-lab/alpaca",
            "--split", "train",
            "--max-samples", "100",
            "--name", "alpaca",
        ])
        assert args.data_command == "hf-import"
        assert args.dataset_id == "tatsu-lab/alpaca"
        assert args.split == "train"
        assert args.max_samples == 100
        assert args.name == "alpaca"
