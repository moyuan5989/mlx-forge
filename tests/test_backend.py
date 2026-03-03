"""Tests for Arrow-based storage backend (datasets library)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestSaveAndLoad:
    """Test save_tokenized and load_tokenized."""

    def test_save_and_load_sft(self, tmp_path, monkeypatch):
        """Save SFT samples and load them back."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        samples = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
            {"input_ids": [6, 7, 8], "labels": [-100, 7, 8]},
            {"input_ids": [10, 11, 12, 13], "labels": [10, 11, 12, 13]},
        ]

        path = backend.save_tokenized("test-ds", "test/model", samples)
        assert path.exists()

        ds = backend.load_tokenized("test-ds", "test/model")
        assert len(ds) == 3
        assert ds[0]["input_ids"] == [1, 2, 3, 4, 5]
        assert ds[0]["labels"] == [-100, -100, 3, 4, 5]
        assert ds[2]["labels"] == [10, 11, 12, 13]

    def test_save_and_load_preference(self, tmp_path, monkeypatch):
        """Save preference samples and load them back."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        samples = [
            {
                "chosen_input_ids": [1, 2, 3],
                "chosen_labels": [-100, 2, 3],
                "rejected_input_ids": [1, 4, 5],
                "rejected_labels": [-100, 4, 5],
            },
        ]

        backend.save_tokenized("pref-ds", "test/model", samples)
        ds = backend.load_tokenized("pref-ds", "test/model")

        assert len(ds) == 1
        assert ds[0]["chosen_input_ids"] == [1, 2, 3]
        assert ds[0]["rejected_labels"] == [-100, 4, 5]

    def test_empty_dataset_raises(self, tmp_path, monkeypatch):
        """Saving empty dataset should raise ValueError."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        with pytest.raises(ValueError, match="empty"):
            backend.save_tokenized("empty", "test/model", [])


class TestTokenizedExists:
    """Test tokenized_exists function."""

    def test_exists_after_save(self, tmp_path, monkeypatch):
        """tokenized_exists returns True after save."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        assert not backend.tokenized_exists("ds", "model/a")

        backend.save_tokenized("ds", "model/a", [
            {"input_ids": [1, 2], "labels": [1, 2]},
        ])

        assert backend.tokenized_exists("ds", "model/a")

    def test_not_exists_for_different_model(self, tmp_path, monkeypatch):
        """tokenized_exists returns False for a different model."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        backend.save_tokenized("ds", "model/a", [
            {"input_ids": [1, 2], "labels": [1, 2]},
        ])

        assert not backend.tokenized_exists("ds", "model/b")


class TestGetProcessedPath:
    """Test path computation."""

    def test_path_format(self):
        """Path should use -- to replace / in model ID."""
        from lmforge.data.backend import get_processed_path

        path = get_processed_path("my-data", "org/model-name")
        assert "my-data--org--model-name" in str(path)
        assert "processed" in str(path)

    def test_path_deterministic(self):
        """Same inputs should produce same path."""
        from lmforge.data.backend import get_processed_path

        p1 = get_processed_path("ds", "model/x")
        p2 = get_processed_path("ds", "model/x")
        assert p1 == p2


class TestMetadata:
    """Test metadata saving."""

    def test_meta_json_created(self, tmp_path, monkeypatch):
        """meta.json should be created alongside the dataset."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        samples = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [4, 5]},
        ]

        path = backend.save_tokenized("meta-test", "m/1", samples)
        meta_path = path / "meta.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["dataset_name"] == "meta-test"
        assert meta["model_id"] == "m/1"
        assert meta["num_samples"] == 2
        assert meta["total_tokens"] == 5  # 3 + 2
        assert meta["max_length"] == 3
        assert meta["min_length"] == 2
        assert meta["format"] == "sft"
        assert meta["schema_version"] == 2
        assert "created_at" in meta

    def test_preference_meta(self, tmp_path, monkeypatch):
        """Preference dataset metadata should report format='preference'."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        samples = [
            {
                "chosen_input_ids": [1, 2, 3],
                "chosen_labels": [-100, 2, 3],
                "rejected_input_ids": [4, 5, 6, 7],
                "rejected_labels": [-100, 5, 6, 7],
            },
        ]

        path = backend.save_tokenized("pref-meta", "m/1", samples)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        assert meta["format"] == "preference"
        assert meta["total_tokens"] == 7  # 3 + 4
        assert meta["max_length"] == 4  # max(3, 4)


class TestFingerprint:
    """Test compute_fingerprint."""

    def test_fingerprint_format(self, tmp_path):
        """Fingerprint should start with 'sha256:' and be 71 chars."""
        from lmforge.data.backend import compute_fingerprint

        data_file = tmp_path / "test.jsonl"
        data_file.write_text('{"text": "sample"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        fp = compute_fingerprint(str(data_file), MockTokenizer())
        assert fp.startswith("sha256:")
        assert len(fp) == 71  # "sha256:" (7) + 64 hex chars

    def test_same_inputs_same_fingerprint(self, tmp_path):
        """Same inputs should produce same fingerprint."""
        from lmforge.data.backend import compute_fingerprint

        data_file = tmp_path / "test.jsonl"
        data_file.write_text('{"text": "sample"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        fp1 = compute_fingerprint(str(data_file), MockTokenizer())
        fp2 = compute_fingerprint(str(data_file), MockTokenizer())
        assert fp1 == fp2

    def test_different_data_different_fingerprint(self, tmp_path):
        """Different data should produce different fingerprint."""
        from lmforge.data.backend import compute_fingerprint

        f1 = tmp_path / "a.jsonl"
        f1.write_text('{"text": "aaa"}\n')
        f2 = tmp_path / "b.jsonl"
        f2.write_text('{"text": "bbb"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0}
            chat_template = None

        fp1 = compute_fingerprint(str(f1), MockTokenizer())
        fp2 = compute_fingerprint(str(f2), MockTokenizer())
        assert fp1 != fp2


class TestListProcessed:
    """Test list_processed function."""

    def test_empty_dir(self, tmp_path, monkeypatch):
        """Empty dir returns empty list."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))
        assert backend.list_processed() == []

    def test_lists_saved_datasets(self, tmp_path, monkeypatch):
        """list_processed returns metadata for saved datasets."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        backend.save_tokenized("ds1", "m/a", [
            {"input_ids": [1, 2], "labels": [1, 2]},
        ])
        backend.save_tokenized("ds2", "m/b", [
            {"input_ids": [3, 4, 5], "labels": [3, 4, 5]},
        ])

        results = backend.list_processed()
        assert len(results) == 2

        names = {r["dataset_name"] for r in results}
        assert "ds1" in names
        assert "ds2" in names


class TestDeleteProcessed:
    """Test delete_processed function."""

    def test_delete_existing(self, tmp_path, monkeypatch):
        """Delete an existing processed dataset."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        backend.save_tokenized("to-delete", "m/1", [
            {"input_ids": [1], "labels": [1]},
        ])

        assert backend.tokenized_exists("to-delete", "m/1")
        assert backend.delete_processed("to-delete", "m/1")
        assert not backend.tokenized_exists("to-delete", "m/1")

    def test_delete_nonexistent(self, tmp_path, monkeypatch):
        """Deleting non-existent dataset returns False."""
        from lmforge.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))
        assert not backend.delete_processed("nope", "m/1")
