"""Pydantic v2 config models for MLX Forge.

All schemas match V0_DESIGN_FREEZE.md §2.1 exactly.
V2 additions are backward-compatible optional fields.
"""

from __future__ import annotations

from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class LRScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: list
    warmup: int = 0
    warmup_init: float = 0.0


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bits: int = 4
    group_size: int = 64

    @model_validator(mode="after")
    def validate_bits(self) -> QuantizationConfig:
        if self.bits not in (4, 8):
            raise ValueError(
                f"Quantization bits must be 4 or 8, got {self.bits}."
            )
        if self.group_size not in (32, 64, 128):
            raise ValueError(
                f"Quantization group_size must be 32, 64, or 128, got {self.group_size}."
            )
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None  # HF revision/commit hash (None = latest)
    quantization: Optional[QuantizationConfig] = None


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["lora", "dora", "full"] = "lora"
    targets: Optional[list[str]] = None
    preset: Optional[str] = None
    num_layers: Optional[int] = None
    rank: int = 8
    scale: float = 20.0
    dropout: float = 0.0

    @model_validator(mode="after")
    def validate_targeting(self) -> AdapterConfig:
        if self.method == "full":
            return self  # Full FT doesn't need targets or presets
        if self.targets is not None and self.preset is not None:
            raise ValueError("Specify 'targets' or 'preset', not both.")
        if self.targets is None and self.preset is None:
            raise ValueError(
                "Must specify 'targets' (glob patterns) or 'preset'. "
                "Available presets: attention-qv, attention-all, mlp, all-linear."
            )
        return self


class DataSourceConfig(BaseModel):
    """A single data source for dataset mixing."""

    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None       # Local JSONL path
    dataset: Optional[str] = None    # Catalog dataset ID
    weight: float = 1.0              # Sampling weight

    @model_validator(mode="after")
    def validate_source(self) -> DataSourceConfig:
        if self.path is None and self.dataset is None:
            raise ValueError("DataSourceConfig must specify 'path' or 'dataset'.")
        if self.path is not None and self.dataset is not None:
            raise ValueError("Specify 'path' or 'dataset', not both.")
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive, got {self.weight}.")
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: Optional[str] = None
    valid: Optional[str] = None
    test: Optional[str] = None
    dataset: Optional[str] = None          # Named dataset from catalog
    sources: Optional[list[DataSourceConfig]] = None  # Multi-dataset mixing
    max_seq_length: int = 2048
    mask_prompt: bool = True
    packing: bool = False

    # HuggingFace Datasets integration
    hf_dataset: Optional[str] = None       # HF dataset ID
    hf_split: str = "train"
    hf_subset: Optional[str] = None
    hf_columns: Optional[dict[str, str]] = None
    hf_max_samples: Optional[int] = None

    @model_validator(mode="after")
    def validate_data_source(self) -> DataConfig:
        sources_count = sum([
            self.train is not None,
            self.sources is not None,
            self.hf_dataset is not None,
        ])
        if sources_count == 0:
            raise ValueError("Must specify one of 'train', 'sources', or 'hf_dataset'.")
        if sources_count > 1:
            raise ValueError("Specify only one of 'train', 'sources', or 'hf_dataset'.")
        if self.hf_dataset is None and self.valid is None:
            raise ValueError("Must specify 'valid' when using 'train' or 'sources'.")
        return self


class TrainingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 2
    num_iters: int = 1000
    learning_rate: float = 1e-5
    optimizer: Literal["adam", "adamw", "sgd", "adafactor"] = "adam"
    optimizer_config: dict = {}
    lr_schedule: Optional[LRScheduleConfig] = None
    grad_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None
    seed: int = 42

    gradient_checkpointing: bool = False
    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    val_batches: int = 25
    keep_last_n_checkpoints: int = 3

    # V2: Training type and DPO parameters
    training_type: Literal["sft", "dpo", "grpo"] = "sft"
    dpo_beta: float = 0.1
    dpo_reference_free: bool = True

    # GRPO parameters
    grpo_num_generations: int = 4
    grpo_beta: float = 0.1
    grpo_clip_range: float = 0.2
    grpo_max_completion_length: int = 256
    grpo_reward_function: str = "length"

    @model_validator(mode="after")
    def validate_save_accum(self) -> TrainingParams:
        if self.steps_per_save % self.grad_accumulation_steps != 0:
            raise ValueError(
                f"steps_per_save ({self.steps_per_save}) must be a multiple of "
                f"grad_accumulation_steps ({self.grad_accumulation_steps})."
            )
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_dir: str = "~/.mlxforge/runs"
    eager: bool = False
    report_to: Optional[str] = None
    wandb_project: Optional[str] = None


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    model: ModelConfig
    adapter: AdapterConfig
    data: DataConfig
    training: TrainingParams
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def from_yaml(cls, path: str) -> TrainingConfig:
        """Load a TrainingConfig from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
