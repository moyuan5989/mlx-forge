"""Tests for data validation (lmforge data validate)."""

from __future__ import annotations

import json

import pytest

from lmforge.data.validate import validate_file, ValidationReport


@pytest.fixture
def chat_file(tmp_path):
    """Create a valid chat JSONL file."""
    samples = [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
        {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "See you!"}]},
    ]
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return str(path)


@pytest.fixture
def completions_file(tmp_path):
    """Create a valid completions JSONL file."""
    samples = [
        {"prompt": "Capital of France?", "completion": "Paris"},
        {"prompt": "1+1?", "completion": "2"},
    ]
    path = tmp_path / "completions.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in samples))
    return str(path)


class TestValidateFile:
    def test_valid_chat_file(self, chat_file):
        report = validate_file(chat_file)
        assert report.ok
        assert report.format == "chat"
        assert report.num_samples == 2
        assert report.num_duplicates == 0

    def test_valid_completions_file(self, completions_file):
        report = validate_file(completions_file)
        assert report.ok
        assert report.format == "completions"
        assert report.num_samples == 2

    def test_missing_file(self):
        report = validate_file("/nonexistent/path.jsonl")
        assert not report.ok
        assert "not found" in report.errors[0].lower()

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        report = validate_file(str(path))
        assert not report.ok
        assert "empty" in report.errors[0].lower()


class TestChatQuality:
    def test_trailing_user_turn_warning(self, tmp_path):
        samples = [
            {"messages": [{"role": "user", "content": "Hello"}]},
        ]
        path = tmp_path / "trailing.jsonl"
        path.write_text(json.dumps(samples[0]))
        report = validate_file(str(path))
        assert any("trailing" in w.lower() or "ends with user" in w.lower() for w in report.warnings)

    def test_empty_content_warning(self, tmp_path):
        samples = [
            {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "Hi"}]},
        ]
        path = tmp_path / "empty_content.jsonl"
        path.write_text(json.dumps(samples[0]))
        report = validate_file(str(path))
        assert any("empty content" in w.lower() for w in report.warnings)

    def test_consecutive_roles_warning(self, tmp_path):
        samples = [
            {"messages": [
                {"role": "user", "content": "A"},
                {"role": "user", "content": "B"},
                {"role": "assistant", "content": "C"},
            ]},
        ]
        path = tmp_path / "consec.jsonl"
        path.write_text(json.dumps(samples[0]))
        report = validate_file(str(path))
        assert any("consecutive" in w.lower() for w in report.warnings)


class TestDuplicateDetection:
    def test_detects_duplicates(self, tmp_path):
        sample = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
        path = tmp_path / "dupes.jsonl"
        path.write_text("\n".join(json.dumps(sample) for _ in range(3)))
        report = validate_file(str(path))
        assert report.num_duplicates == 2

    def test_no_duplicates(self, chat_file):
        report = validate_file(chat_file)
        assert report.num_duplicates == 0


class TestOverlapDetection:
    def test_detects_overlap(self, tmp_path):
        shared = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
        train_only = {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Later"}]}

        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text("\n".join([json.dumps(shared), json.dumps(train_only)]))
        val_path.write_text(json.dumps(shared))

        report = validate_file(str(train_path), val_path=str(val_path))
        assert report.overlap_count == 1

    def test_no_overlap(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text(json.dumps({"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]}))
        val_path.write_text(json.dumps({"messages": [{"role": "user", "content": "C"}, {"role": "assistant", "content": "D"}]}))
        report = validate_file(str(train_path), val_path=str(val_path))
        assert report.overlap_count == 0


class TestLengthStats:
    def test_length_stats_computed(self, chat_file):
        report = validate_file(chat_file)
        assert "min" in report.length_stats
        assert "max" in report.length_stats
        assert "mean" in report.length_stats
        assert "p50" in report.length_stats
        assert "p95" in report.length_stats
        assert report.length_stats["min"] > 0
