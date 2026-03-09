"""Tests for config system (M1)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from cortexlab.config import (
    AdapterConfig,
    DataConfig,
    DataSourceConfig,
    LRScheduleConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    TrainingParams,
)


class TestTrainingConfigLoading:
    def test_valid_config_from_dict(self, sample_config_dict):
        """Test loading a valid config from a dict."""
        config = TrainingConfig(**sample_config_dict)
        assert config.schema_version == 1
        assert config.model.path == "Qwen/Qwen3-0.6B"
        assert config.adapter.preset == "attention-qv"
        assert config.adapter.rank == 8
        assert config.training.batch_size == 4
        assert config.training.num_iters == 1000

    def test_valid_config_from_yaml(self, tmp_dir, sample_config_dict):
        """Test loading a valid config from a YAML file."""
        yaml_path = tmp_dir / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = TrainingConfig.from_yaml(str(yaml_path))
        assert config.schema_version == 1
        assert config.model.path == "Qwen/Qwen3-0.6B"
        assert config.adapter.preset == "attention-qv"

    def test_schema_version_present(self, sample_config_dict):
        """Test that schema_version field is present and correct."""
        config = TrainingConfig(**sample_config_dict)
        assert hasattr(config, "schema_version")
        assert config.schema_version == 1


class TestConfigValidation:
    def test_extra_fields_rejected(self, sample_config_dict):
        """Test that extra='forbid' rejects unknown fields."""
        # Extra field at top level
        bad_config = sample_config_dict.copy()
        bad_config["unknown_field"] = "value"
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**bad_config)
        assert "extra_forbidden" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()

        # Extra field in nested model
        bad_config = sample_config_dict.copy()
        bad_config["model"]["unknown_field"] = "value"
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**bad_config)
        assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()

    def test_missing_required_field_raises(self):
        """Test that missing required fields raise ValidationError."""
        # Missing 'model' field
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(
                schema_version=1,
                adapter={"preset": "attention-qv"},
                data={"train": "train.jsonl", "valid": "valid.jsonl"},
                training={},
            )
        assert "model" in str(exc_info.value).lower()

        # Missing required field in nested model
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(trust_remote_code=False)  # missing 'path'
        assert "path" in str(exc_info.value).lower()

    def test_invalid_optimizer_raises(self, sample_config_dict):
        """Test that invalid optimizer enum value raises ValidationError."""
        bad_config = sample_config_dict.copy()
        bad_config["training"]["optimizer"] = "invalid_optimizer"
        with pytest.raises(ValidationError) as exc_info:
            TrainingConfig(**bad_config)
        assert "optimizer" in str(exc_info.value).lower()


class TestAdapterConfigValidation:
    def test_targets_and_preset_mutual_exclusion(self):
        """Test that specifying both targets and preset raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            AdapterConfig(
                method="lora",
                targets=["*.q_proj"],
                preset="attention-qv",
                rank=8,
            )
        error_msg = str(exc_info.value).lower()
        assert "targets" in error_msg or "preset" in error_msg or "both" in error_msg

    def test_neither_targets_nor_preset_raises(self):
        """Test that specifying neither targets nor preset raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            AdapterConfig(method="lora", rank=8)
        error_msg = str(exc_info.value).lower()
        assert "targets" in error_msg or "preset" in error_msg


class TestTrainingParamsValidation:
    def test_steps_per_save_not_multiple_of_grad_accum_raises(self):
        """Test that steps_per_save not being a multiple of grad_accumulation_steps raises."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingParams(
                steps_per_save=100,
                grad_accumulation_steps=3,  # 100 % 3 != 0
            )
        error_msg = str(exc_info.value).lower()
        assert "steps_per_save" in error_msg or "grad_accumulation" in error_msg or "multiple" in error_msg

    def test_steps_per_save_multiple_of_grad_accum_passes(self, sample_config_dict):
        """Test that steps_per_save being a multiple of grad_accumulation_steps passes."""
        # Default values: steps_per_save=100, grad_accumulation_steps=1 (100 % 1 == 0)
        params = TrainingParams(**sample_config_dict["training"])
        assert params.steps_per_save == 100
        assert params.grad_accumulation_steps == 1

        # Test with other valid multiples
        params = TrainingParams(steps_per_save=200, grad_accumulation_steps=4)
        assert params.steps_per_save == 200
        assert params.grad_accumulation_steps == 4

        params = TrainingParams(steps_per_save=300, grad_accumulation_steps=10)
        assert params.steps_per_save == 300
        assert params.grad_accumulation_steps == 10


class TestLRScheduleConfig:
    def test_lr_schedule_optional(self):
        """Test that lr_schedule is optional in TrainingParams."""
        params = TrainingParams()
        assert params.lr_schedule is None

    def test_lr_schedule_with_values(self):
        """Test that lr_schedule accepts valid config."""
        schedule = LRScheduleConfig(
            name="cosine_decay",
            arguments=[1e-5, 1000],
            warmup=100,
            warmup_init=0.0,
        )
        assert schedule.name == "cosine_decay"
        assert schedule.arguments == [1e-5, 1000]
        assert schedule.warmup == 100
        assert schedule.warmup_init == 0.0


class TestRuntimeConfig:
    def test_runtime_config_defaults(self):
        """Test that RuntimeConfig has correct defaults."""
        runtime = RuntimeConfig()
        assert runtime.run_dir == "~/.cortexlab/runs"
        assert runtime.eager is False
        assert runtime.report_to is None
        assert runtime.wandb_project is None

    def test_runtime_config_in_training_config(self, sample_config_dict):
        """Test that RuntimeConfig is optional with defaults in TrainingConfig."""
        config_no_runtime = sample_config_dict.copy()
        del config_no_runtime["runtime"]

        config = TrainingConfig(**config_no_runtime)
        assert config.runtime.run_dir == "~/.cortexlab/runs"
        assert config.runtime.eager is False


class TestDataSourceConfig:
    def test_valid_path_source(self):
        """Test valid data source with path."""
        src = DataSourceConfig(path="./data.jsonl", weight=0.5)
        assert src.path == "./data.jsonl"
        assert src.weight == 0.5

    def test_valid_dataset_source(self):
        """Test valid data source with catalog dataset."""
        src = DataSourceConfig(dataset="alpaca", weight=1.0)
        assert src.dataset == "alpaca"

    def test_both_path_and_dataset_raises(self):
        """Test that specifying both path and dataset raises."""
        with pytest.raises(ValidationError):
            DataSourceConfig(path="./data.jsonl", dataset="alpaca")

    def test_neither_path_nor_dataset_raises(self):
        """Test that specifying neither path nor dataset raises."""
        with pytest.raises(ValidationError):
            DataSourceConfig(weight=1.0)

    def test_zero_weight_raises(self):
        """Test that zero weight raises."""
        with pytest.raises(ValidationError):
            DataSourceConfig(path="./data.jsonl", weight=0.0)

    def test_negative_weight_raises(self):
        """Test that negative weight raises."""
        with pytest.raises(ValidationError):
            DataSourceConfig(path="./data.jsonl", weight=-1.0)


class TestDataConfigSources:
    def test_train_or_sources_required(self):
        """Test that either train or sources must be provided."""
        with pytest.raises(ValidationError):
            DataConfig(valid="./val.jsonl")

    def test_train_and_sources_exclusive(self):
        """Test that train and sources are mutually exclusive."""
        with pytest.raises(ValidationError):
            DataConfig(
                train="./train.jsonl",
                valid="./val.jsonl",
                sources=[DataSourceConfig(path="./other.jsonl")],
            )

    def test_sources_config_valid(self):
        """Test valid sources configuration."""
        cfg = DataConfig(
            valid="./val.jsonl",
            sources=[
                DataSourceConfig(path="./data1.jsonl", weight=0.6),
                DataSourceConfig(dataset="alpaca", weight=0.4),
            ],
        )
        assert cfg.sources is not None
        assert len(cfg.sources) == 2
        assert cfg.train is None

    def test_single_train_still_works(self):
        """Test backward-compatible single train path."""
        cfg = DataConfig(train="./train.jsonl", valid="./val.jsonl")
        assert cfg.train == "./train.jsonl"
        assert cfg.sources is None
