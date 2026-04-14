"""M44 Tests: Train-to-Serve Bridge — forge files, resolution, CLI, adapter by run_id."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ─── Test ForgeSpec ───


class TestForgeSpec:
    """Tests for ForgeSpec model bundle."""

    def test_from_yaml(self, tmp_path):
        from mlx_forge.forge import ForgeSpec

        forge_file = tmp_path / "test.yaml"
        forge_file.write_text(
            yaml.dump({
                "name": "my-bot",
                "base": "Qwen/Qwen3-0.6B",
                "adapter": "run:my-run",
                "system": "You are helpful.",
                "parameters": {"temperature": 0.5},
            })
        )

        forge = ForgeSpec.from_yaml(forge_file)
        assert forge.name == "my-bot"
        assert forge.base == "Qwen/Qwen3-0.6B"
        assert forge.adapter == "run:my-run"
        assert forge.system == "You are helpful."
        assert forge.parameters["temperature"] == 0.5

    def test_from_yaml_minimal(self, tmp_path):
        forge_file = tmp_path / "minimal.yaml"
        forge_file.write_text(yaml.dump({"base": "some-model"}))

        from mlx_forge.forge import ForgeSpec

        forge = ForgeSpec.from_yaml(forge_file)
        assert forge.base == "some-model"
        assert forge.adapter is None
        assert forge.system is None

    def test_from_yaml_missing_base(self, tmp_path):
        forge_file = tmp_path / "bad.yaml"
        forge_file.write_text(yaml.dump({"name": "bad"}))

        from mlx_forge.forge import ForgeSpec

        with pytest.raises(ValueError, match="missing 'base'"):
            ForgeSpec.from_yaml(forge_file)

    def test_from_yaml_not_found(self):
        from mlx_forge.forge import ForgeSpec

        with pytest.raises(FileNotFoundError):
            ForgeSpec.from_yaml("/nonexistent/path.yaml")

    def test_from_run(self, tmp_path):
        from mlx_forge.forge import ForgeSpec

        # Create mock run directory
        run_dir = tmp_path / "runs" / "my-run"
        ckpt_dir = run_dir / "checkpoints" / "step-0001000"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapters.safetensors").touch()

        config = {"model": {"path": "Qwen/Qwen3-0.6B"}}
        (run_dir / "config.yaml").write_text(yaml.dump(config))

        with patch("mlx_forge.forge.RUNS_DIR", tmp_path / "runs"):
            forge = ForgeSpec.from_run("my-run")
            assert forge.name == "my-run"
            assert forge.base == "Qwen/Qwen3-0.6B"
            assert forge.adapter is not None
            assert "step-0001000" in forge.adapter

    def test_from_run_not_found(self):
        from mlx_forge.forge import ForgeSpec

        with patch("mlx_forge.forge.RUNS_DIR", Path("/nonexistent")):
            with pytest.raises(FileNotFoundError):
                ForgeSpec.from_run("missing-run")

    def test_save(self, tmp_path):
        from mlx_forge.forge import ForgeSpec

        forge = ForgeSpec(
            name="test-forge",
            base="model-id",
            adapter="/path/to/adapter",
            system="Hello",
        )
        path = forge.save(tmp_path / "test-forge.yaml")
        assert path.exists()

        loaded = ForgeSpec.from_yaml(path)
        assert loaded.name == "test-forge"
        assert loaded.base == "model-id"
        assert loaded.adapter == "/path/to/adapter"

    def test_save_default_path(self, tmp_path):
        from mlx_forge.forge import ForgeSpec

        with patch("mlx_forge.forge.FORGES_DIR", tmp_path):
            forge = ForgeSpec(name="myforge", base="model")
            path = forge.save()
            assert path == tmp_path / "myforge.yaml"
            assert path.exists()

    def test_to_load_args(self):
        from mlx_forge.forge import ForgeSpec

        forge = ForgeSpec(name="x", base="model-id", adapter="/path/adapter")
        args = forge.to_load_args()
        assert args["model_path"] == "model-id"
        assert args["adapter_path"] == "/path/adapter"

    def test_to_load_args_no_adapter(self):
        from mlx_forge.forge import ForgeSpec

        forge = ForgeSpec(name="x", base="model-id")
        args = forge.to_load_args()
        assert args["model_path"] == "model-id"
        assert "adapter_path" not in args

    def test_resolve_adapter_run(self, tmp_path):
        from mlx_forge.forge import ForgeSpec

        run_dir = tmp_path / "runs" / "run1"
        ckpt_dir = run_dir / "checkpoints" / "step-0000500"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapters.safetensors").touch()

        with patch("mlx_forge.forge.RUNS_DIR", tmp_path / "runs"):
            forge = ForgeSpec(name="x", base="m", adapter="run:run1")
            resolved = forge.resolve_adapter_path()
            assert "step-0000500" in resolved

    def test_resolve_adapter_path(self):
        from mlx_forge.forge import ForgeSpec

        forge = ForgeSpec(name="x", base="m", adapter="/some/path")
        assert forge.resolve_adapter_path() == "/some/path"

    def test_resolve_adapter_none(self):
        from mlx_forge.forge import ForgeSpec

        forge = ForgeSpec(name="x", base="m")
        assert forge.resolve_adapter_path() is None


# ─── Test list/get/delete forges ───


class TestForgeOperations:
    """Tests for forge CRUD operations."""

    def test_list_forges(self, tmp_path):
        from mlx_forge.forge import ForgeSpec, list_forges

        with patch("mlx_forge.forge.FORGES_DIR", tmp_path):
            ForgeSpec(name="a", base="model-a").save(tmp_path / "a.yaml")
            ForgeSpec(name="b", base="model-b").save(tmp_path / "b.yaml")

            forges = list_forges()
            names = [f.name for f in forges]
            assert "a" in names
            assert "b" in names

    def test_list_forges_empty(self, tmp_path):
        from mlx_forge.forge import list_forges

        with patch("mlx_forge.forge.FORGES_DIR", tmp_path):
            assert list_forges() == []

    def test_get_forge(self, tmp_path):
        from mlx_forge.forge import ForgeSpec, get_forge

        with patch("mlx_forge.forge.FORGES_DIR", tmp_path):
            ForgeSpec(name="x", base="model").save(tmp_path / "x.yaml")
            forge = get_forge("x")
            assert forge is not None
            assert forge.base == "model"

    def test_get_forge_not_found(self, tmp_path):
        from mlx_forge.forge import get_forge

        with patch("mlx_forge.forge.FORGES_DIR", tmp_path):
            assert get_forge("nonexistent") is None

    def test_delete_forge(self, tmp_path):
        from mlx_forge.forge import ForgeSpec, delete_forge

        with patch("mlx_forge.forge.FORGES_DIR", tmp_path):
            ForgeSpec(name="x", base="m").save(tmp_path / "x.yaml")
            assert delete_forge("x") is True
            assert not (tmp_path / "x.yaml").exists()
            assert delete_forge("x") is False


# ─── Test Forge Resolution in ModelPool ───


class TestForgeResolution:
    """Tests for forge: prefix resolution in ModelPool."""

    def test_pool_resolves_forge_prefix(self):
        """ModelPool recognizes forge: prefix."""
        from mlx_forge.serving.model_pool import ModelPool

        pool = ModelPool(max_models=2)

        # Mock the forge loading
        with patch.object(pool, "_load_from_forge") as mock_load:
            mock_mgr = MagicMock()
            mock_mgr.load = MagicMock()
            mock_mgr.snapshot_base_weights = MagicMock()

            def side_effect(mgr, name):
                mgr._model = MagicMock()
                mgr._tokenizer = MagicMock()
                mgr._model_id = f"forge:{name}"

            mock_load.side_effect = side_effect

            with patch("mlx_forge.serving.model_pool.ModelManager") as MockMM:
                instance = MockMM.return_value
                instance.snapshot_base_weights = MagicMock()
                pool.get("forge:my-bot")
                mock_load.assert_called_once()

    def test_non_forge_prefix_normal_load(self):
        """Non-forge: prefix loads normally."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool()
        mgr = MagicMock()
        mgr.model_id = "regular-model"
        pool._models["regular-model"] = ManagedModel(
            manager=mgr, model_id="regular-model", keep_alive=300
        )

        result = pool.get("regular-model")
        assert result is mgr


# ─── Test Adapter by Run ID ───


class TestAdapterByRunId:
    """Tests for loading adapters by training run ID."""

    def test_resolve_adapter_request_with_path(self):
        from mlx_forge.serving.routes import AdapterLoadRequest, _resolve_adapter_path

        req = AdapterLoadRequest(adapter_path="/some/path")
        assert _resolve_adapter_path(req) == "/some/path"

    def test_resolve_adapter_request_with_run_id(self, tmp_path):
        """run_id resolves to best checkpoint path."""
        from mlx_forge.serving.routes import AdapterLoadRequest

        # Create mock run structure
        run_dir = tmp_path / "my-run"
        ckpt_dir = run_dir / "checkpoints" / "step-0001000"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "adapters.safetensors").touch()

        req = AdapterLoadRequest(run_id="my-run")

        # Patch the runs_dir inside the function
        import mlx_forge.serving.routes as routes_mod
        original_fn = routes_mod._resolve_adapter_path

        def patched_resolve(request):
            if request.run_id:
                rd = tmp_path / request.run_id
                if not rd.exists():
                    raise FileNotFoundError(f"Run '{request.run_id}' not found")
                cd = rd / "checkpoints"
                best = cd / "best"
                if best.exists():
                    return str(best.resolve())
                ckpts = sorted(
                    [d for d in cd.iterdir() if d.is_dir() and not d.is_symlink()]
                )
                if ckpts:
                    return str(ckpts[-1])
                raise FileNotFoundError(f"No checkpoints in run '{request.run_id}'")
            return original_fn(request)

        result = patched_resolve(req)
        assert "step-0001000" in result

    def test_resolve_adapter_neither(self):
        from mlx_forge.serving.routes import AdapterLoadRequest, _resolve_adapter_path

        req = AdapterLoadRequest()
        with pytest.raises(ValueError, match="Provide either"):
            _resolve_adapter_path(req)

    def test_adapter_request_has_run_id_field(self):
        from mlx_forge.serving.routes import AdapterLoadRequest

        req = AdapterLoadRequest(run_id="my-training-run")
        assert req.run_id == "my-training-run"
        assert req.adapter_path is None


# ─── Test CLI ───


class TestForgeCLI:
    """Tests for forge CLI commands."""

    def test_forge_parser_exists(self):
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["forge", "list"])
        assert args.command == "forge"
        assert args.forge_command == "list"

    def test_forge_create_parser(self):
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "forge", "create", "my-bot", "--from-run", "run-123"
        ])
        assert args.name == "my-bot"
        assert args.from_run == "run-123"

    def test_forge_create_with_base(self):
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "forge", "create", "my-bot", "--base", "Qwen/Qwen3-0.6B"
        ])
        assert args.base == "Qwen/Qwen3-0.6B"

    def test_forge_delete_parser(self):
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["forge", "delete", "my-bot"])
        assert args.forge_command == "delete"
        assert args.name == "my-bot"

    def test_forge_show_parser(self):
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["forge", "show", "my-bot"])
        assert args.forge_command == "show"

    def test_forge_cmd_handler_exists(self):
        from mlx_forge.cli.forge_cmd import run_forge

        assert callable(run_forge)
