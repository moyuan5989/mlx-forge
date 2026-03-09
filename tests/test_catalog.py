"""Tests for dataset catalog, converter, and registry."""

from __future__ import annotations

import json


class TestCatalogEntries:
    """Test that catalog entries are valid."""

    def test_catalog_not_empty(self):
        """Catalog should contain datasets."""
        from cortexlab.data.catalog import DATASET_CATALOG

        assert len(DATASET_CATALOG) > 0

    def test_catalog_entries_have_required_fields(self):
        """All catalog entries should have required fields."""
        from cortexlab.data.catalog import DATASET_CATALOG

        for entry_id, profile in DATASET_CATALOG.items():
            assert profile.id == entry_id, f"ID mismatch: {profile.id} vs {entry_id}"
            assert profile.source, f"Missing source for {entry_id}"
            assert profile.display_name, f"Missing display_name for {entry_id}"
            assert profile.category, f"Missing category for {entry_id}"
            assert profile.format in ("chat", "completions", "text", "preference"), \
                f"Invalid format '{profile.format}' for {entry_id}"
            assert profile.total_samples > 0, f"Invalid total_samples for {entry_id}"
            assert profile.license, f"Missing license for {entry_id}"

    def test_catalog_categories(self):
        """Catalog should cover multiple categories."""
        from cortexlab.data.catalog import DATASET_CATALOG

        categories = {p.category for p in DATASET_CATALOG.values()}
        assert "general" in categories
        assert "code" in categories
        assert "math" in categories

    def test_catalog_has_preference_datasets(self):
        """Catalog should include preference datasets for DPO."""
        from cortexlab.data.catalog import DATASET_CATALOG

        preference_datasets = [
            p for p in DATASET_CATALOG.values() if p.format == "preference"
        ]
        assert len(preference_datasets) >= 2

    def test_catalog_to_dict(self):
        """DatasetProfile.to_dict() returns valid dict."""
        from cortexlab.data.catalog import DATASET_CATALOG

        for profile in DATASET_CATALOG.values():
            d = profile.to_dict()
            assert isinstance(d, dict)
            assert "id" in d
            assert "source" in d
            assert "display_name" in d


class TestConverters:
    """Test dataset format converters."""

    def test_rename_converter(self):
        """Rename converter maps columns correctly."""
        from cortexlab.data.converter import _convert_rename

        dataset = [
            {"instruction": "Say hi", "response": "Hello!"},
            {"instruction": "What?", "response": "Hi there"},
        ]
        mapping = {"instruction": "prompt", "response": "completion"}

        result = _convert_rename(dataset, mapping)
        assert len(result) == 2
        assert result[0]["prompt"] == "Say hi"
        assert result[0]["completion"] == "Hello!"

    def test_alpaca_converter(self):
        """Alpaca converter combines instruction + input."""
        from cortexlab.data.converter import _convert_alpaca

        dataset = [
            {"instruction": "Translate", "input": "Hello", "output": "Hola"},
            {"instruction": "Say hi", "input": "", "output": "Hello!"},
        ]
        mapping = {"instruction": "instruction", "input": "input", "output": "output"}

        result = _convert_alpaca(dataset, mapping)
        assert len(result) == 2
        assert "Hello" in result[0]["prompt"]  # Input appended
        assert result[0]["completion"] == "Hola"
        assert result[1]["prompt"] == "Say hi"  # No input appended

    def test_sharegpt_converter(self):
        """ShareGPT converter converts from/value to role/content."""
        from cortexlab.data.converter import _convert_sharegpt

        dataset = [
            {"conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]},
        ]

        result = _convert_sharegpt(dataset, {"conversations": "conversations"})
        assert len(result) == 1
        msgs = result[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hi"
        assert msgs[1]["role"] == "assistant"

    def test_text_converter(self):
        """Text converter renames text column."""
        from cortexlab.data.converter import _convert_text

        dataset = [{"content": "Hello world"}]
        result = _convert_text(dataset, {"text": "content"})
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_chat_messages_converter(self):
        """Chat messages converter normalizes messages."""
        from cortexlab.data.converter import _convert_chat_messages

        dataset = [
            {"messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]},
        ]

        result = _convert_chat_messages(dataset, {"messages": "messages"})
        assert len(result) == 1
        assert len(result[0]["messages"]) == 2

    def test_preference_converter_string(self):
        """Preference converter handles string chosen/rejected."""
        from cortexlab.data.converter import _convert_preference

        dataset = [
            {"prompt": "Hi", "chosen": "Good response", "rejected": "Bad response"},
        ]

        result = _convert_preference(dataset, {"chosen": "chosen", "rejected": "rejected"})
        assert len(result) == 1
        assert "chosen" in result[0]
        assert "rejected" in result[0]
        assert result[0]["chosen"][-1]["content"] == "Good response"

    def test_preference_converter_messages(self):
        """Preference converter handles message list chosen/rejected."""
        from cortexlab.data.converter import _convert_preference

        dataset = [
            {
                "chosen": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Good"}],
                "rejected": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Bad"}],
            },
        ]

        result = _convert_preference(dataset, {"chosen": "chosen", "rejected": "rejected"})
        assert len(result) == 1


class TestConvertDataset:
    """Test the main convert_dataset function."""

    def test_convert_with_profile(self):
        """convert_dataset uses profile's column mapping."""
        from cortexlab.data.catalog import ColumnMapping, DatasetProfile
        from cortexlab.data.converter import convert_dataset

        profile = DatasetProfile(
            id="test",
            source="test/test",
            display_name="Test",
            category="test",
            format="completions",
            description="Test dataset",
            license="MIT",
            total_samples=1,
            avg_tokens=10,
            columns=ColumnMapping(
                type="rename",
                mapping={"q": "prompt", "a": "completion"},
            ),
        )

        hf_dataset = [{"q": "Question?", "a": "Answer!"}]
        result = convert_dataset(hf_dataset, profile)
        assert result[0]["prompt"] == "Question?"
        assert result[0]["completion"] == "Answer!"


class TestRegistry:
    """Test dataset registry operations."""

    def test_list_empty_registry(self, tmp_path):
        """Empty registry returns empty list."""
        from cortexlab.data.registry import DatasetRegistry

        registry = DatasetRegistry(base_dir=str(tmp_path / "datasets"))
        assert registry.list_datasets() == []

    def test_import_local(self, tmp_path):
        """Import a local JSONL file."""
        from cortexlab.data.registry import DatasetRegistry

        # Create a test JSONL file
        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps({"text": "Hello world"}) + "\n")
            f.write(json.dumps({"text": "Goodbye"}) + "\n")

        registry = DatasetRegistry(base_dir=str(tmp_path / "datasets"))
        path = registry.import_local(str(data_file), name="my-test")

        assert path.exists()
        assert (path / "data.jsonl").exists()
        assert (path / "meta.json").exists()

        # List should include it
        datasets = registry.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["id"] == "my-test"

    def test_get_samples(self, tmp_path):
        """Preview samples from imported dataset."""
        from cortexlab.data.registry import DatasetRegistry

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"Sample {i}"}) + "\n")

        registry = DatasetRegistry(base_dir=str(tmp_path / "datasets"))
        registry.import_local(str(data_file), name="test-ds")

        samples = registry.get_samples("test-ds", n=3)
        assert len(samples) == 3
        assert samples[0]["text"] == "Sample 0"

    def test_delete_dataset(self, tmp_path):
        """Delete a dataset from registry."""
        from cortexlab.data.registry import DatasetRegistry

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps({"text": "Hello"}) + "\n")

        registry = DatasetRegistry(base_dir=str(tmp_path / "datasets"))
        registry.import_local(str(data_file), name="to-delete")

        assert len(registry.list_datasets()) == 1
        assert registry.delete_dataset("to-delete")
        assert len(registry.list_datasets()) == 0

    def test_delete_nonexistent(self, tmp_path):
        """Deleting non-existent dataset returns False."""
        from cortexlab.data.registry import DatasetRegistry

        registry = DatasetRegistry(base_dir=str(tmp_path / "datasets"))
        assert not registry.delete_dataset("nonexistent")

    def test_list_catalog(self):
        """list_catalog returns full catalog."""
        from cortexlab.data.registry import DatasetRegistry

        registry = DatasetRegistry()
        catalog = registry.list_catalog()
        assert len(catalog) > 0

    def test_get_dataset(self, tmp_path):
        """Get metadata for a specific dataset."""
        from cortexlab.data.registry import DatasetRegistry

        data_file = tmp_path / "test.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps({"text": "Hello"}) + "\n")

        registry = DatasetRegistry(base_dir=str(tmp_path / "datasets"))
        registry.import_local(str(data_file), name="specific")

        meta = registry.get_dataset("specific")
        assert meta is not None
        assert meta["id"] == "specific"

        # Non-existent
        assert registry.get_dataset("nope") is None
