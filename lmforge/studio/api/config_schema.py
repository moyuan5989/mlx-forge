"""Config Schema API — serves Pydantic schemas for dynamic UI forms."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v2/schema", tags=["schema"])


@router.get("")
def get_training_config_schema():
    """Return the TrainingConfig JSON Schema with UI metadata.

    The frontend renders forms dynamically from this schema.
    Adding a field to config.py auto-generates the UI field.
    """
    from lmforge.config import TrainingConfig
    schema = TrainingConfig.model_json_schema()

    # Add UI metadata to schema properties
    _add_ui_metadata(schema)

    return schema


@router.get("/training-params")
def get_training_params_schema():
    """Return just the TrainingParams schema for the config form."""
    from lmforge.config import TrainingParams
    schema = TrainingParams.model_json_schema()
    _add_ui_metadata(schema)
    return schema


@router.get("/model")
def get_model_config_schema():
    """Return the ModelConfig schema."""
    from lmforge.config import ModelConfig
    return ModelConfig.model_json_schema()


@router.get("/adapter")
def get_adapter_config_schema():
    """Return the AdapterConfig schema."""
    from lmforge.config import AdapterConfig
    return AdapterConfig.model_json_schema()


@router.get("/data")
def get_data_config_schema():
    """Return the DataConfig schema."""
    from lmforge.config import DataConfig
    return DataConfig.model_json_schema()


# UI metadata for schema fields
_UI_METADATA = {
    "batch_size": {
        "group": "training",
        "label": "Batch Size",
        "description": "Number of samples per training step",
        "ui_type": "slider",
        "min": 1,
        "max": 64,
        "step": 1,
    },
    "num_iters": {
        "group": "training",
        "label": "Training Steps",
        "description": "Total number of training iterations",
        "ui_type": "number",
        "min": 1,
    },
    "learning_rate": {
        "group": "training",
        "label": "Learning Rate",
        "description": "Optimizer learning rate",
        "ui_type": "number",
        "min": 0,
        "step": 1e-6,
    },
    "max_seq_length": {
        "group": "data",
        "label": "Max Sequence Length",
        "description": "Maximum tokens per training sample",
        "ui_type": "slider",
        "min": 128,
        "max": 8192,
        "step": 128,
    },
    "rank": {
        "group": "adapter",
        "label": "LoRA Rank",
        "description": "Rank of the LoRA adapters (higher = more capacity)",
        "ui_type": "slider",
        "min": 1,
        "max": 128,
        "step": 1,
    },
    "gradient_checkpointing": {
        "group": "performance",
        "label": "Gradient Checkpointing",
        "description": "Reduce memory by recomputing activations (~30% slower)",
        "ui_type": "toggle",
    },
    "packing": {
        "group": "performance",
        "label": "Sequence Packing",
        "description": "Pack short sequences together for 2-5x speedup",
        "ui_type": "toggle",
    },
    "training_type": {
        "group": "training",
        "label": "Training Type",
        "description": "SFT for supervised fine-tuning, DPO for preference alignment",
        "ui_type": "select",
        "options": ["sft", "dpo"],
    },
    "dpo_beta": {
        "group": "dpo",
        "label": "DPO Beta",
        "description": "Temperature for preference strength (higher = stronger)",
        "ui_type": "slider",
        "min": 0.01,
        "max": 1.0,
        "step": 0.01,
        "visible_when": {"training_type": "dpo"},
    },
    "dpo_reference_free": {
        "group": "dpo",
        "label": "Reference-Free (SimPO)",
        "description": "Skip reference model to save ~2x memory",
        "ui_type": "toggle",
        "visible_when": {"training_type": "dpo"},
    },
}


def _add_ui_metadata(schema: dict) -> None:
    """Annotate schema properties with UI rendering hints."""
    props = schema.get("properties", {})
    for key, meta in _UI_METADATA.items():
        if key in props:
            props[key]["x-ui"] = meta

    # Recurse into nested schemas
    for key, prop in props.items():
        if "$ref" in prop:
            ref_name = prop["$ref"].split("/")[-1]
            defs = schema.get("$defs", {})
            if ref_name in defs:
                _add_ui_metadata(defs[ref_name])
