"""Tests for M11: Studio Backend — FastAPI server, REST API, WebSocket hub.

Tests cover:
- Service layer (RunService, ModelService, DatasetService, MetricsWatcher)
- REST API endpoints (runs, models, datasets, training, inference)
- WebSocket hubs (training metrics, inference streaming)
- CLI command registration
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

# FastAPI test client
from fastapi.testclient import TestClient


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_runs_dir(tmp_path):
    """Create a temporary runs directory with mock run data."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return runs_dir


@pytest.fixture
def mock_run(tmp_runs_dir):
    """Create a mock run directory with config, metrics, and checkpoint."""
    run_id = "20260207-120000-sft-test-model-abcd"
    run_dir = tmp_runs_dir / run_id
    run_dir.mkdir()

    # Write config.yaml
    config = {
        "model": {"path": "test/model"},
        "adapter": {"method": "lora", "preset": "attention-qv", "rank": 8},
        "data": {"train": "train.jsonl", "valid": "val.jsonl"},
        "training": {"num_iters": 100, "batch_size": 2, "learning_rate": 1e-5,
                      "steps_per_report": 10, "steps_per_eval": 50, "steps_per_save": 50},
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Write manifest.json
    manifest = {"schema_version": 1, "lmforge_version": "0.1.0"}
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    # Write metrics
    logs_dir = run_dir / "logs"
    logs_dir.mkdir()
    metrics = [
        {"event": "train", "step": 10, "train_loss": 2.5, "learning_rate": 1e-5},
        {"event": "train", "step": 20, "train_loss": 2.3, "learning_rate": 1e-5},
        {"event": "eval", "step": 50, "val_loss": 2.1},
        {"event": "train", "step": 50, "train_loss": 2.0, "learning_rate": 1e-5},
        {"event": "train", "step": 100, "train_loss": 1.8, "learning_rate": 1e-5},
        {"event": "eval", "step": 100, "val_loss": 1.7},
    ]
    with open(logs_dir / "metrics.jsonl", "w") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")

    # Write checkpoint
    ckpt_dir = run_dir / "checkpoints" / "step-0000100"
    ckpt_dir.mkdir(parents=True)
    state = {"schema_version": 1, "step": 100, "epoch": 5, "trained_tokens": 50000,
             "best_val_loss": 1.7, "learning_rate": 1e-5, "rng_seed": 42}
    with open(ckpt_dir / "state.json", "w") as f:
        json.dump(state, f)

    return run_id


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Create a temporary datasets directory with mock processed data."""
    datasets_dir = tmp_path / "datasets"
    processed_dir = datasets_dir / "processed"
    processed_dir.mkdir(parents=True)

    # Create a mock processed dataset
    ds_name = "train--test--model"
    ds_dir = processed_dir / ds_name
    ds_dir.mkdir()
    meta = {
        "schema_version": 2,
        "dataset_name": "train",
        "model_id": "test/model",
        "num_samples": 1000,
        "total_tokens": 500000,
        "min_length": 10,
        "max_length": 2048,
        "mean_length": 500.0,
        "format": "sft",
    }
    with open(ds_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    return datasets_dir


@pytest.fixture
def app(tmp_runs_dir, tmp_cache_dir):
    """Create a test FastAPI app with mock service configuration."""
    from lmforge.studio.server import create_app
    from lmforge.studio.api import datasets, runs
    from lmforge.studio.services.dataset_service import DatasetService
    from lmforge.studio.services.run_service import RunService

    app = create_app(runs_dir=str(tmp_runs_dir))

    # Override dataset service to use tmp datasets dir
    datasets.set_dataset_service(DatasetService(datasets_dir=str(tmp_cache_dir)))

    return app


@pytest.fixture
def client(app):
    """Create a test HTTP client."""
    return TestClient(app)


# ============================================================
# RunService Tests
# ============================================================

class TestRunService:
    def test_list_runs_empty(self, tmp_runs_dir):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        assert service.list_runs() == []

    def test_list_runs_with_run(self, tmp_runs_dir, mock_run):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        runs = service.list_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == mock_run
        assert runs[0]["model"] == "test/model"
        assert runs[0]["num_iters"] == 100

    def test_get_run(self, tmp_runs_dir, mock_run):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        run = service.get_run(mock_run)
        assert run is not None
        assert run["id"] == mock_run
        assert "config" in run
        assert "manifest" in run
        assert run["config"]["model"]["path"] == "test/model"

    def test_get_run_not_found(self, tmp_runs_dir):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        assert service.get_run("nonexistent") is None

    def test_get_metrics(self, tmp_runs_dir, mock_run):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        metrics = service.get_metrics(mock_run)
        assert len(metrics["train"]) == 4
        assert len(metrics["eval"]) == 2
        assert metrics["train"][0]["step"] == 10
        assert metrics["eval"][-1]["val_loss"] == 1.7

    def test_get_checkpoints(self, tmp_runs_dir, mock_run):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        checkpoints = service.get_checkpoints(mock_run)
        assert len(checkpoints) == 1
        assert checkpoints[0]["name"] == "step-0000100"
        assert checkpoints[0]["state"]["step"] == 100

    def test_delete_run(self, tmp_runs_dir, mock_run):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        assert service.delete_run(mock_run) is True
        assert service.get_run(mock_run) is None

    def test_delete_run_not_found(self, tmp_runs_dir):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        assert service.delete_run("nonexistent") is False

    def test_infer_status_completed(self, tmp_runs_dir, mock_run):
        from lmforge.studio.services.run_service import RunService
        service = RunService(str(tmp_runs_dir))
        run = service.get_run(mock_run)
        # current_step == num_iters == 100 -> completed
        assert run["status"] == "completed"

    def test_infer_status_stopped(self, tmp_runs_dir):
        """A run with current_step < num_iters and old mtime -> stopped."""
        from lmforge.studio.services.run_service import RunService

        run_id = "20260207-130000-sft-test-abcd"
        run_dir = tmp_runs_dir / run_id
        run_dir.mkdir()

        config = {
            "model": {"path": "test/model"},
            "training": {"num_iters": 1000},
        }
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        logs_dir = run_dir / "logs"
        logs_dir.mkdir()
        with open(logs_dir / "metrics.jsonl", "w") as f:
            f.write(json.dumps({"event": "train", "step": 50, "train_loss": 2.0}) + "\n")

        # Set mtime to >60s ago
        old_time = time.time() - 120
        os.utime(logs_dir / "metrics.jsonl", (old_time, old_time))

        service = RunService(str(tmp_runs_dir))
        run = service.get_run(run_id)
        assert run["status"] == "stopped"


# ============================================================
# ModelService Tests
# ============================================================

class TestModelService:
    def test_list_models_empty(self, tmp_path):
        from lmforge.studio.services.model_service import ModelService
        service = ModelService(str(tmp_path / "nonexistent"))
        assert service.list_models() == []

    def test_list_models_with_model(self, tmp_path):
        from lmforge.studio.services.model_service import ModelService

        # Create a mock HF cache structure
        model_dir = tmp_path / "models--test--model"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)

        # Write a config.json
        config = {"model_type": "llama", "hidden_size": 2048}
        with open(snapshot_dir / "config.json", "w") as f:
            json.dump(config, f)

        service = ModelService(str(tmp_path))
        models = service.list_models()
        assert len(models) == 1
        assert models[0]["id"] == "test/model"
        assert models[0]["architecture"] == "llama"
        assert models[0]["supported"] is True

    def test_get_model(self, tmp_path):
        from lmforge.studio.services.model_service import ModelService

        model_dir = tmp_path / "models--Qwen--Qwen3-0.8B"
        snapshot_dir = model_dir / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        config = {"model_type": "qwen3"}
        with open(snapshot_dir / "config.json", "w") as f:
            json.dump(config, f)

        service = ModelService(str(tmp_path))
        model = service.get_model("Qwen/Qwen3-0.8B")
        assert model is not None
        assert model["id"] == "Qwen/Qwen3-0.8B"
        assert model["architecture"] == "qwen3"

    def test_get_model_not_found(self, tmp_path):
        from lmforge.studio.services.model_service import ModelService
        service = ModelService(str(tmp_path))
        assert service.get_model("nonexistent/model") is None

    def test_list_supported_architectures(self, tmp_path):
        from lmforge.studio.services.model_service import ModelService
        service = ModelService(str(tmp_path))
        archs = service.list_supported_architectures()
        assert "llama" in archs
        assert "qwen3" in archs
        assert "gemma" in archs


# ============================================================
# DatasetService Tests
# ============================================================

class TestDatasetService:
    def test_list_datasets_empty(self, tmp_path):
        from lmforge.studio.services.dataset_service import DatasetService
        service = DatasetService(str(tmp_path / "nonexistent"))
        assert service.list_datasets() == []

    def test_list_datasets(self, tmp_cache_dir):
        from lmforge.studio.services.dataset_service import DatasetService
        service = DatasetService(datasets_dir=str(tmp_cache_dir))
        datasets = service.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["dataset_name"] == "train"
        assert datasets[0]["num_samples"] == 1000

    def test_get_dataset(self, tmp_cache_dir):
        from lmforge.studio.services.dataset_service import DatasetService
        service = DatasetService(datasets_dir=str(tmp_cache_dir))
        ds = service.get_dataset("train--test--model")
        assert ds is not None
        assert ds["total_tokens"] == 500000

    def test_get_dataset_not_found(self, tmp_cache_dir):
        from lmforge.studio.services.dataset_service import DatasetService
        service = DatasetService(datasets_dir=str(tmp_cache_dir))
        assert service.get_dataset("nonexistent") is None

    def test_delete_dataset(self, tmp_cache_dir):
        from lmforge.studio.services.dataset_service import DatasetService
        service = DatasetService(datasets_dir=str(tmp_cache_dir))
        assert service.delete_dataset("train--test--model") is True
        assert service.get_dataset("train--test--model") is None

    def test_delete_dataset_not_found(self, tmp_cache_dir):
        from lmforge.studio.services.dataset_service import DatasetService
        service = DatasetService(datasets_dir=str(tmp_cache_dir))
        assert service.delete_dataset("nonexistent") is False


# ============================================================
# MetricsWatcher Tests
# ============================================================

class TestMetricsWatcher:
    def test_poll_empty(self, tmp_path):
        from lmforge.studio.services.metrics_watcher import MetricsWatcher
        watcher = MetricsWatcher(tmp_path / "metrics.jsonl")
        assert watcher.poll() == []

    def test_poll_new_entries(self, tmp_path):
        from lmforge.studio.services.metrics_watcher import MetricsWatcher

        metrics_path = tmp_path / "metrics.jsonl"
        # Create empty file first, then create watcher (starts at end)
        metrics_path.touch()
        watcher = MetricsWatcher(metrics_path)

        # Write some metrics after watcher init
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": 10, "loss": 2.5}) + "\n")
            f.write(json.dumps({"step": 20, "loss": 2.3}) + "\n")

        entries = watcher.poll()
        assert len(entries) == 2
        assert entries[0]["step"] == 10
        assert entries[1]["step"] == 20

        # Second poll should return empty (no new data)
        assert watcher.poll() == []

    def test_poll_incremental(self, tmp_path):
        from lmforge.studio.services.metrics_watcher import MetricsWatcher

        metrics_path = tmp_path / "metrics.jsonl"
        metrics_path.touch()
        watcher = MetricsWatcher(metrics_path)

        # Write first batch
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": 1}) + "\n")
        assert len(watcher.poll()) == 1

        # Write second batch
        with open(metrics_path, "a") as f:
            f.write(json.dumps({"step": 2}) + "\n")
            f.write(json.dumps({"step": 3}) + "\n")
        entries = watcher.poll()
        assert len(entries) == 2
        assert entries[0]["step"] == 2

    def test_reset(self, tmp_path):
        from lmforge.studio.services.metrics_watcher import MetricsWatcher

        metrics_path = tmp_path / "metrics.jsonl"
        with open(metrics_path, "w") as f:
            f.write(json.dumps({"step": 1}) + "\n")

        watcher = MetricsWatcher(metrics_path)
        # Watcher starts at end, so poll returns nothing
        assert watcher.poll() == []

        # Reset and poll again
        watcher.reset()
        entries = watcher.poll()
        assert len(entries) == 1
        assert entries[0]["step"] == 1


# ============================================================
# REST API Tests
# ============================================================

class TestRunsAPI:
    def test_list_runs_empty(self, client):
        resp = client.get("/api/v1/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_runs(self, client, mock_run):
        resp = client.get("/api/v1/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["id"] == mock_run

    def test_get_run(self, client, mock_run):
        resp = client.get(f"/api/v1/runs/{mock_run}")
        assert resp.status_code == 200
        run = resp.json()
        assert run["id"] == mock_run
        assert "config" in run

    def test_get_run_not_found(self, client):
        resp = client.get("/api/v1/runs/nonexistent")
        assert resp.status_code == 404

    def test_get_metrics(self, client, mock_run):
        resp = client.get(f"/api/v1/runs/{mock_run}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["train"]) == 4
        assert len(data["eval"]) == 2

    def test_get_config(self, client, mock_run):
        resp = client.get(f"/api/v1/runs/{mock_run}/config")
        assert resp.status_code == 200
        config = resp.json()
        assert config["model"]["path"] == "test/model"

    def test_get_checkpoints(self, client, mock_run):
        resp = client.get(f"/api/v1/runs/{mock_run}/checkpoints")
        assert resp.status_code == 200
        ckpts = resp.json()
        assert len(ckpts) == 1
        assert ckpts[0]["name"] == "step-0000100"

    def test_delete_run(self, client, mock_run):
        resp = client.delete(f"/api/v1/runs/{mock_run}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp = client.get(f"/api/v1/runs/{mock_run}")
        assert resp.status_code == 404

    def test_delete_run_not_found(self, client):
        resp = client.delete("/api/v1/runs/nonexistent")
        assert resp.status_code == 404


class TestModelsAPI:
    def test_list_supported(self, client):
        resp = client.get("/api/v1/models/supported")
        assert resp.status_code == 200
        archs = resp.json()
        assert "llama" in archs
        assert "qwen3" in archs


class TestDatasetsAPI:
    def test_list_datasets(self, client):
        resp = client.get("/api/v1/datasets")
        assert resp.status_code == 200
        datasets = resp.json()
        assert len(datasets) == 1
        assert datasets[0]["dataset_name"] == "train"

    def test_get_dataset(self, client):
        resp = client.get("/api/v1/datasets/train--test--model")
        assert resp.status_code == 200
        ds = resp.json()
        assert ds["num_samples"] == 1000

    def test_get_dataset_not_found(self, client):
        resp = client.get("/api/v1/datasets/nonexistent")
        assert resp.status_code == 404

    def test_delete_dataset(self, client):
        resp = client.delete("/api/v1/datasets/train--test--model")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestTrainingAPI:
    def test_list_active_empty(self, client):
        resp = client.get("/api/v1/training/active")
        assert resp.status_code == 200
        assert resp.json() == []


class TestInferenceAPI:
    def test_generate_missing_model(self, client):
        resp = client.post("/api/v1/inference/generate", json={})
        assert resp.status_code == 400
        assert "model" in resp.json()["detail"]

    def test_generate_missing_prompt(self, client):
        resp = client.post("/api/v1/inference/generate", json={"model": "test"})
        assert resp.status_code == 400
        assert "prompt" in resp.json()["detail"]

    def test_inference_status(self, client):
        resp = client.get("/api/v1/inference/status")
        assert resp.status_code == 200
        assert resp.json()["loaded_model"] is None


# ============================================================
# WebSocket Tests
# ============================================================

class TestTrainingWebSocket:
    def test_ws_training_not_found(self, client):
        with client.websocket_connect("/ws/training/nonexistent") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not found" in msg["detail"]

    def test_ws_training_receives_metrics(self, client, mock_run, tmp_runs_dir):
        """Connect to training WS and verify it streams new metrics."""
        with client.websocket_connect(f"/ws/training/{mock_run}") as ws:
            # Append a new metric to the file
            metrics_path = tmp_runs_dir / mock_run / "logs" / "metrics.jsonl"
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"event": "train", "step": 110, "train_loss": 1.5}) + "\n")

            # Send stop to end the loop after one poll cycle
            ws.send_json({"type": "stop"})

            # Collect messages until stopped
            messages = []
            while True:
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] == "stopped":
                    break

            # Should have received the new metric + stopped message
            metric_msgs = [m for m in messages if m["type"] == "metric"]
            assert len(metric_msgs) >= 1
            assert metric_msgs[0]["data"]["step"] == 110


# ============================================================
# CLI Tests
# ============================================================

class TestCLI:
    def test_studio_command_registered(self):
        """Verify 'studio' is a valid subcommand."""
        from lmforge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["studio", "--port", "9999"])
        assert args.command == "studio"
        assert args.port == 9999
        assert args.host == "127.0.0.1"

    def test_studio_default_args(self):
        from lmforge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["studio"])
        assert args.port == 8741
        assert args.host == "127.0.0.1"


# ============================================================
# Server Tests
# ============================================================

class TestServer:
    def test_frontend_page(self, client):
        """Verify the frontend HTML page is served (SPA or placeholder)."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_cors_headers(self, client):
        """Verify CORS is configured for localhost."""
        resp = client.options(
            "/api/v1/runs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS preflight should return appropriate headers
        assert resp.status_code == 200

    def test_openapi_docs(self, client):
        """Verify Swagger UI is available."""
        resp = client.get("/docs")
        assert resp.status_code == 200
