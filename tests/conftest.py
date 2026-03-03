"""Shared pytest fixtures for LMForge tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def sample_config_dict():
    """Return a valid TrainingConfig dict matching V0_DESIGN_FREEZE.md §2.1."""
    return {
        "schema_version": 1,
        "model": {
            "path": "Qwen/Qwen3-0.6B",
            "trust_remote_code": False,
        },
        "adapter": {
            "method": "lora",
            "preset": "attention-qv",
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
        },
        "data": {
            "train": "./data/train.jsonl",
            "valid": "./data/valid.jsonl",
            "max_seq_length": 2048,
            "mask_prompt": True,
        },
        "training": {
            "batch_size": 4,
            "num_iters": 1000,
            "learning_rate": 1e-5,
            "optimizer": "adam",
            "grad_accumulation_steps": 1,
            "seed": 42,
            "steps_per_report": 10,
            "steps_per_eval": 200,
            "steps_per_save": 100,
            "val_batches": 25,
            "keep_last_n_checkpoints": 3,
        },
        "runtime": {
            "run_dir": "~/.lmforge/runs",
            "eager": False,
        },
    }
