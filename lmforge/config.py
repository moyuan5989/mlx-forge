"""Pydantic v2 config models for LMForge.

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

    method: Literal["lora"] = "lora"
    targets: Optional[list[str]] = None
    preset: Optional[str] = None
    num_layers: Optional[int] = None
    rank: int = 8
    scale: float = 20.0
    dropout: float = 0.0

    @model_validator(mode="after")
    def validate_targeting(self) -> AdapterConfig:
        if self.targets is not None and self.preset is not None:
            raise ValueError("Specify 'targets' or 'preset', not both.")
        if self.targets is None and self.preset is None:
            raise ValueError(
                "Must specify 'targets' (glob patterns) or 'preset'. "
                "Available presets: attention-qv, attention-all, mlp, all-linear."
            )
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: str
    valid: str
    test: Optional[str] = None
    dataset: Optional[str] = None   # Named dataset from catalog
    mix: Optional[str] = None       # Named mix config (future)
    max_seq_length: int = 2048
    mask_prompt: bool = True
    packing: bool = False


class TrainingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 4
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
    training_type: Literal["sft", "dpo"] = "sft"
    dpo_beta: float = 0.1
    dpo_reference_free: bool = True

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

    run_dir: str = "~/.lmforge/runs"
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
