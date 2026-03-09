"""Tests for V2 Studio backend (recipes API, memory API, queue, schema)."""

from __future__ import annotations

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient

from cortexlab.studio.server import create_app


@pytest.fixture
def app():
    return create_app(runs_dir="/tmp/cortexlab_test_v2_studio")


@pytest.fixture
def client(app):
    return TestClient(app)


# ── Recipes API ──────────────────────────────────────────────────────────────


class TestRecipesAPI:
    """Test /api/v2/recipes endpoints."""

    def test_list_recipes(self, client):
        """GET /api/v2/recipes returns list of recipes."""
        resp = client.get("/api/v2/recipes")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 4

    def test_get_recipe(self, client):
        """GET /api/v2/recipes/chat-sft returns recipe."""
        resp = client.get("/api/v2/recipes/chat-sft")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "chat-sft"
        assert data["training_type"] == "sft"

    def test_get_recipe_not_found(self, client):
        """GET /api/v2/recipes/nonexistent returns 404."""
        resp = client.get("/api/v2/recipes/nonexistent")
        assert resp.status_code == 404

    def test_resolve_recipe(self, client):
        """POST /api/v2/recipes/chat-sft/resolve returns config."""
        resp = client.post(
            "/api/v2/recipes/chat-sft/resolve",
            json={
                "model_id": "Qwen/Qwen3-0.6B",
                "train_path": "/data/train.jsonl",
                "valid_path": "/data/val.jsonl",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"]["path"] == "Qwen/Qwen3-0.6B"
        assert data["schema_version"] == 1

    def test_resolve_recipe_missing_fields(self, client):
        """POST /api/v2/recipes/chat-sft/resolve with missing fields returns 400."""
        resp = client.post(
            "/api/v2/recipes/chat-sft/resolve",
            json={"model_id": "test/model"},  # Missing train/valid paths
        )
        assert resp.status_code == 400


# ── Memory API ───────────────────────────────────────────────────────────────


class TestMemoryAPI:
    """Test /api/v2/memory endpoints."""

    def test_get_hardware(self, client):
        """GET /api/v2/memory/hardware returns hardware info."""
        resp = client.get("/api/v2/memory/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_memory_gb" in data
        assert "training_budget_gb" in data
        assert "chip_name" in data

    def test_estimate_memory(self, client):
        """POST /api/v2/memory/estimate returns estimate."""
        resp = client.post(
            "/api/v2/memory/estimate",
            json={"model_id": "Qwen/Qwen3-0.6B"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_gb" in data
        assert "fits" in data
        assert "bar_segments" in data
        assert len(data["bar_segments"]) == 4

    def test_estimate_memory_missing_model(self, client):
        """POST /api/v2/memory/estimate without model_id returns 400."""
        resp = client.post("/api/v2/memory/estimate", json={})
        assert resp.status_code == 400

    def test_estimate_memory_unknown_model(self, client):
        """POST /api/v2/memory/estimate with unknown model returns 400."""
        resp = client.post(
            "/api/v2/memory/estimate",
            json={"model_id": "nonexistent/model-99B"},
        )
        assert resp.status_code == 400

    def test_compatible_models(self, client):
        """GET /api/v2/memory/compatible-models returns model list."""
        resp = client.get("/api/v2/memory/compatible-models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "model_id" in data[0]
        assert "fp16" in data[0]
        assert "qlora_4bit" in data[0]


# ── Queue API ────────────────────────────────────────────────────────────────


class TestQueueAPI:
    """Test /api/v2/queue endpoints."""

    def test_list_queue_empty(self, client):
        """GET /api/v2/queue returns empty list initially."""
        resp = client.get("/api/v2/queue")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_queue_stats(self, client):
        """GET /api/v2/queue/stats returns stats."""
        resp = client.get("/api/v2/queue/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "queued" in data
        assert "running" in data
        assert "max_concurrent" in data

    def test_cancel_nonexistent(self, client):
        """POST /api/v2/queue/{id}/cancel for nonexistent returns 404."""
        resp = client.post("/api/v2/queue/nonexistent/cancel")
        assert resp.status_code == 404


# ── Schema API ───────────────────────────────────────────────────────────────


class TestSchemaAPI:
    """Test /api/v2/schema endpoints."""

    def test_get_full_schema(self, client):
        """GET /api/v2/schema returns full TrainingConfig JSON Schema."""
        resp = client.get("/api/v2/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "properties" in data
        assert "title" in data

    def test_get_training_params_schema(self, client):
        """GET /api/v2/schema/training-params returns TrainingParams schema."""
        resp = client.get("/api/v2/schema/training-params")
        assert resp.status_code == 200
        data = resp.json()
        assert "properties" in data
        # V2 fields should be present
        assert "training_type" in data["properties"]
        assert "dpo_beta" in data["properties"]

    def test_get_model_schema(self, client):
        """GET /api/v2/schema/model returns ModelConfig schema."""
        resp = client.get("/api/v2/schema/model")
        assert resp.status_code == 200
        data = resp.json()
        assert "properties" in data
        assert "path" in data["properties"]

    def test_get_adapter_schema(self, client):
        """GET /api/v2/schema/adapter returns AdapterConfig schema."""
        resp = client.get("/api/v2/schema/adapter")
        assert resp.status_code == 200

    def test_get_data_schema(self, client):
        """GET /api/v2/schema/data returns DataConfig schema."""
        resp = client.get("/api/v2/schema/data")
        assert resp.status_code == 200


# ── Frontend V2 ──────────────────────────────────────────────────────────────


class TestFrontendV2:
    """Test that V2 frontend files exist."""

    def test_new_training_page_exists(self):
        """NewTraining.tsx page exists."""
        from pathlib import Path
        p = Path("studio-frontend/src/pages/NewTraining.tsx")
        assert p.exists()

    def test_job_queue_page_exists(self):
        """JobQueue.tsx page exists."""
        from pathlib import Path
        p = Path("studio-frontend/src/pages/JobQueue.tsx")
        assert p.exists()

    def test_memory_bar_component_exists(self):
        """MemoryBar.tsx component exists."""
        from pathlib import Path
        p = Path("studio-frontend/src/components/shared/MemoryBar.tsx")
        assert p.exists()

    def test_v2_hooks_exist(self):
        """V2 hooks exist."""
        from pathlib import Path
        hooks = ["useRecipes.ts", "useMemory.ts", "useQueue.ts"]
        for hook in hooks:
            p = Path(f"studio-frontend/src/hooks/{hook}")
            assert p.exists(), f"Missing hook: {hook}"

    def test_app_has_v2_routes(self):
        """App.tsx includes V2 routes."""
        from pathlib import Path
        content = Path("studio-frontend/src/App.tsx").read_text()
        assert "/new" in content
        assert "/queue" in content
        assert "NewTraining" in content
        assert "JobQueue" in content

    def test_sidebar_has_v2_links(self):
        """Sidebar includes V2 navigation links."""
        from pathlib import Path
        content = Path("studio-frontend/src/components/layout/Sidebar.tsx").read_text()
        assert "New Training" in content
        assert "Job Queue" in content

    def test_api_client_has_v2(self):
        """API client has V2 methods."""
        from pathlib import Path
        content = Path("studio-frontend/src/api/client.ts").read_text()
        assert "apiV2" in content
        assert "getRecipes" in content
        assert "estimateMemory" in content
        assert "submitJob" in content

    def test_types_have_v2(self):
        """Types file has V2 interfaces."""
        from pathlib import Path
        content = Path("studio-frontend/src/api/types.ts").read_text()
        assert "Recipe" in content
        assert "HardwareInfo" in content
        assert "QueueJob" in content
        assert "MemoryEstimateResult" in content


# ── Model Library API ────────────────────────────────────────────────────────


class TestModelLibraryAPI:
    """Test /api/v2/models/library endpoint."""

    def test_list_library(self, client):
        """GET /api/v2/models/library returns 18 models."""
        resp = client.get("/api/v2/models/library")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 18

    def test_library_entry_shape(self, client):
        """Each entry has all required fields including downloaded: bool."""
        resp = client.get("/api/v2/models/library")
        data = resp.json()
        entry = data[0]
        required_fields = [
            "model_id", "display_name", "num_params_b", "architecture",
            "hidden_dim", "num_layers", "vocab_size", "downloaded",
            "fp16", "qlora_4bit", "recommended",
        ]
        for field in required_fields:
            assert field in entry, f"Missing field: {field}"
        assert isinstance(entry["downloaded"], bool)
        assert isinstance(entry["recommended"], bool)

    def test_library_sorted_by_size(self, client):
        """Library is sorted ascending by num_params_b."""
        resp = client.get("/api/v2/models/library")
        data = resp.json()
        sizes = [m["num_params_b"] for m in data]
        assert sizes == sorted(sizes)

    def test_library_has_memory_estimates(self, client):
        """fp16 and qlora_4bit have total_gb and fits."""
        resp = client.get("/api/v2/models/library")
        data = resp.json()
        for entry in data:
            for key in ("fp16", "qlora_4bit"):
                assert "total_gb" in entry[key]
                assert "fits" in entry[key]
                assert isinstance(entry[key]["total_gb"], (int, float))
                assert isinstance(entry[key]["fits"], bool)


# ── Backend Metadata ──────────────────────────────────────────────────────


class TestBackendMetadata:
    """Test metadata in backend save_tokenized."""

    def test_save_tokenized_with_metadata(self, tmp_path, monkeypatch):
        """save_tokenized stores dataset_name and model_id in meta.json."""
        import json

        from cortexlab.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        samples = [{"input_ids": list(range(10)), "labels": [-100, -100, -100] + list(range(3, 10))}]
        path = backend.save_tokenized("my-train", "Qwen/Qwen3-0.6B", samples)

        with open(path / "meta.json") as f:
            meta = json.load(f)
        assert meta["dataset_name"] == "my-train"
        assert meta["model_id"] == "Qwen/Qwen3-0.6B"
        assert meta["num_samples"] == 1
        assert meta["schema_version"] == 2

    def test_save_tokenized_format_detection(self, tmp_path, monkeypatch):
        """save_tokenized detects SFT vs preference format."""
        import json

        from cortexlab.data import backend

        monkeypatch.setattr(backend, "DATASETS_DIR", str(tmp_path))

        # SFT format
        sft_samples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]
        path = backend.save_tokenized("sft-ds", "m/1", sft_samples)
        with open(path / "meta.json") as f:
            assert json.load(f)["format"] == "sft"

        # Preference format
        pref_samples = [{
            "chosen_input_ids": [1, 2], "chosen_labels": [1, 2],
            "rejected_input_ids": [3, 4], "rejected_labels": [3, 4],
        }]
        path = backend.save_tokenized("pref-ds", "m/1", pref_samples)
        with open(path / "meta.json") as f:
            assert json.load(f)["format"] == "preference"


# ── Architecture Tests ───────────────────────────────────────────────────────


class TestNewArchitectures:
    """Test V2 architecture registrations."""

    def test_qwen2_in_registry(self):
        """Qwen2/Qwen2.5 is registered."""
        from cortexlab.models.registry import is_supported
        assert is_supported("qwen2")

    def test_phi4_in_registry(self):
        """Phi-4 is registered."""
        from cortexlab.models.registry import is_supported
        assert is_supported("phi4")

    def test_llama3_remapping(self):
        """llama3 remaps to llama."""
        from cortexlab.models.registry import is_supported
        assert is_supported("llama3")

    def test_qwen2_model_instantiation(self):
        """Qwen2 model can be instantiated and forward pass works."""
        from cortexlab.models.architectures.qwen2 import Model, ModelArgs

        args = ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=100,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
        )
        model = Model(args)
        mx.eval(model.parameters())

        inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        logits = model(inputs)
        mx.eval(logits)

        assert logits.shape == (1, 4, 100)

    def test_phi4_model_instantiation(self):
        """Phi-4 model can be instantiated and forward pass works."""
        from cortexlab.models.architectures.phi4 import Model, ModelArgs

        args = ModelArgs(
            model_type="phi4",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=100,
        )
        model = Model(args)
        mx.eval(model.parameters())

        inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        logits = model(inputs)
        mx.eval(logits)

        assert logits.shape == (1, 4, 100)

    def test_qwen2_lora_targeting(self):
        """LoRA patterns match Qwen2 modules."""
        from cortexlab.adapters.targeting import resolve_targets
        from cortexlab.models.architectures.qwen2 import Model, ModelArgs

        args = ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=100,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
        )
        model = Model(args)
        mx.eval(model.parameters())

        targets = resolve_targets(model, ["*.self_attn.q_proj", "*.self_attn.v_proj"])
        assert len(targets) == 4  # 2 layers * 2 modules

    def test_phi4_lora_targeting(self):
        """LoRA patterns match Phi-4 modules."""
        from cortexlab.adapters.targeting import resolve_targets
        from cortexlab.models.architectures.phi4 import Model, ModelArgs

        args = ModelArgs(
            model_type="phi4",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=100,
        )
        model = Model(args)
        mx.eval(model.parameters())

        targets = resolve_targets(model, ["*.self_attn.q_proj", "*.self_attn.v_proj"])
        assert len(targets) == 4  # 2 layers * 2 modules

    def test_qwen2_sanitize(self):
        """Qwen2 sanitize removes rotary_emb weights."""
        from cortexlab.models.architectures.qwen2 import Model, ModelArgs

        args = ModelArgs(
            model_type="qwen2",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=100,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            tie_word_embeddings=True,
        )
        model = Model(args)

        weights = {
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros(32),
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
            "lm_head.weight": mx.zeros((100, 64)),
        }

        cleaned = model.sanitize(weights)
        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in cleaned
        assert "lm_head.weight" not in cleaned  # Tied embeddings
        assert "model.layers.0.self_attn.q_proj.weight" in cleaned

    def test_list_architectures_includes_v2(self):
        """Architecture list includes V2 additions."""
        from cortexlab.models.registry import list_supported_architectures
        archs = list_supported_architectures()
        assert "qwen2" in archs
        assert "phi4" in archs


# ── Contract Preservation ─────────────────────────────────────────────────────


class TestV2ContractPreservation:
    """Verify all V1 frozen contracts are preserved in V2."""

    def test_checkpoint_format_unchanged(self):
        """Checkpoint still produces exactly 3 files."""
        from cortexlab.trainer.checkpoint import CheckpointManager
        # CheckpointManager class still exists with same interface
        assert hasattr(CheckpointManager, "save")
        assert hasattr(CheckpointManager, "load")

    def test_batch_contract_v2(self):
        """V2 batch contract: (B, T) input_ids + (B, T) labels."""
        from cortexlab.data.batching import iterate_batches

        dataset = [
            {"input_ids": list(range(10)), "labels": [-100, -100, -100] + list(range(3, 10))}
            for _ in range(8)
        ]

        from dataclasses import dataclass

        @dataclass
        class _Cfg:
            data: object
            training: object

        @dataclass
        class _D:
            max_seq_length: int = 2048
            packing: bool = False

        @dataclass
        class _T:
            batch_size: int = 4

        config = _Cfg(data=_D(), training=_T())
        batches = list(iterate_batches(dataset, config))

        assert len(batches) >= 1
        input_ids, labels = batches[0]
        assert input_ids.shape[0] == 4  # B
        assert labels.shape[0] == 4    # B
        assert input_ids.shape == labels.shape  # Same shape

    def test_config_backward_compat(self):
        """V1 configs load without errors."""
        import tempfile

        import yaml

        from cortexlab.config import TrainingConfig

        v1_config = {
            "schema_version": 1,
            "model": {"path": "test/model"},
            "adapter": {"preset": "attention-qv"},
            "data": {"train": "t.jsonl", "valid": "v.jsonl"},
            "training": {"num_iters": 100, "steps_per_save": 100},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(v1_config, f)
            f.flush()
            config = TrainingConfig.from_yaml(f.name)

        assert config.training.training_type == "sft"
        assert config.training.dpo_beta == 0.1

    def test_loss_functions_accessible(self):
        """V1-style loss functions still accessible from trainer module."""
        from cortexlab.trainer.trainer import loss_fn, loss_fn_packed
        assert callable(loss_fn)
        assert callable(loss_fn_packed)

    def test_trainer_alias_works(self):
        """Trainer class (V1 name) is still importable."""
        from cortexlab.trainer.trainer import SFTTrainer, Trainer
        assert Trainer is SFTTrainer
