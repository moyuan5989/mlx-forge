"""Tests for M21: HuggingFace Hub Upload."""

import json
from unittest.mock import MagicMock, patch

from mlx_forge.hub.upload import generate_model_card, push_adapter_only, push_to_hub


class TestModelCard:
    """Test model card generation."""

    def test_basic_card(self):
        config = {
            "model": {"path": "Qwen/Qwen3-0.6B"},
            "adapter": {"method": "lora", "rank": 8, "scale": 20.0, "preset": "attention-qv"},
            "training": {"learning_rate": 1e-5, "batch_size": 2, "num_iters": 1000, "optimizer": "adam"},
            "data": {"max_seq_length": 2048},
        }
        card = generate_model_card(config)
        assert "Qwen/Qwen3-0.6B" in card
        assert "lora" in card
        assert "8" in card

    def test_card_has_yaml_frontmatter(self):
        config = {
            "model": {"path": "test/model"},
            "adapter": {"method": "lora", "rank": 4},
            "training": {},
            "data": {},
        }
        card = generate_model_card(config)
        assert card.startswith("---")
        assert "library_name: mlx-forge" in card
        assert "base_model: test/model" in card

    def test_card_with_metrics(self):
        config = {
            "model": {"path": "test/model"},
            "adapter": {"method": "dora", "rank": 16},
            "training": {},
            "data": {},
        }
        metrics = {"final_train_loss": 0.5432, "best_val_loss": 0.6123}
        card = generate_model_card(config, metrics=metrics)
        assert "0.5432" in card
        assert "0.6123" in card

    def test_card_with_custom_base_model(self):
        config = {
            "model": {"path": "wrong/model"},
            "adapter": {"method": "lora"},
            "training": {},
            "data": {},
        }
        card = generate_model_card(config, base_model="correct/model")
        assert "correct/model" in card

    def test_card_with_targets(self):
        config = {
            "model": {"path": "test/model"},
            "adapter": {"method": "lora", "targets": ["*.q_proj", "*.v_proj"]},
            "training": {},
            "data": {},
        }
        card = generate_model_card(config)
        assert "q_proj" in card

    def test_card_dora_method(self):
        config = {
            "model": {"path": "test/model"},
            "adapter": {"method": "dora", "rank": 16},
            "training": {},
            "data": {},
        }
        card = generate_model_card(config)
        assert "dora" in card

    def test_card_has_tags(self):
        config = {
            "model": {"path": "test/model"},
            "adapter": {},
            "training": {},
            "data": {},
        }
        card = generate_model_card(config)
        assert "mlx" in card
        assert "apple-silicon" in card

    def test_card_has_usage_example(self):
        config = {
            "model": {"path": "test/model"},
            "adapter": {},
            "training": {},
            "data": {},
        }
        card = generate_model_card(config)
        assert "mlx_forge.generate" in card


class TestPushToHub:
    """Test push_to_hub function."""

    @patch("huggingface_hub.HfApi")
    def test_push_creates_repo(self, MockHfApi, tmp_path):
        mock_api = MagicMock()
        mock_repo_url = MagicMock()
        mock_repo_url.url = "https://huggingface.co/test/repo"
        mock_api.create_repo.return_value = mock_repo_url
        MockHfApi.return_value = mock_api

        # Create a dummy file
        (tmp_path / "model.safetensors").write_bytes(b"fake")

        url = push_to_hub(tmp_path, "test/repo")
        assert url == "https://huggingface.co/test/repo"
        mock_api.create_repo.assert_called_once_with(
            repo_id="test/repo", private=False, exist_ok=True
        )

    @patch("huggingface_hub.HfApi")
    def test_push_private_repo(self, MockHfApi, tmp_path):
        mock_api = MagicMock()
        mock_repo_url = MagicMock()
        mock_repo_url.url = "https://huggingface.co/test/repo"
        mock_api.create_repo.return_value = mock_repo_url
        MockHfApi.return_value = mock_api

        (tmp_path / "model.safetensors").write_bytes(b"fake")

        push_to_hub(tmp_path, "test/repo", private=True)
        mock_api.create_repo.assert_called_once_with(
            repo_id="test/repo", private=True, exist_ok=True
        )

    @patch("huggingface_hub.HfApi")
    def test_push_adapter_only_patterns(self, MockHfApi, tmp_path):
        mock_api = MagicMock()
        mock_repo_url = MagicMock()
        mock_repo_url.url = "https://huggingface.co/test/repo"
        mock_api.create_repo.return_value = mock_repo_url
        MockHfApi.return_value = mock_api

        (tmp_path / "adapters.safetensors").write_bytes(b"fake")

        push_to_hub(tmp_path, "test/repo", adapter_only=True)
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["allow_patterns"] is not None
        assert "adapters.safetensors" in call_kwargs["allow_patterns"]

    @patch("huggingface_hub.HfApi")
    def test_push_full_model_no_filter(self, MockHfApi, tmp_path):
        mock_api = MagicMock()
        mock_repo_url = MagicMock()
        mock_repo_url.url = "https://huggingface.co/test/repo"
        mock_api.create_repo.return_value = mock_repo_url
        MockHfApi.return_value = mock_api

        push_to_hub(tmp_path, "test/repo", adapter_only=False)
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["allow_patterns"] is None

    @patch("huggingface_hub.HfApi")
    def test_push_with_token(self, MockHfApi, tmp_path):
        mock_api = MagicMock()
        mock_repo_url = MagicMock()
        mock_repo_url.url = "https://huggingface.co/test/repo"
        mock_api.create_repo.return_value = mock_repo_url
        MockHfApi.return_value = mock_api

        push_to_hub(tmp_path, "test/repo", token="hf_test123")
        MockHfApi.assert_called_once_with(token="hf_test123")


class TestPushAdapterOnly:
    """Test push_adapter_only function."""

    @patch("mlx_forge.hub.upload.push_to_hub")
    def test_creates_adapter_config(self, mock_push, tmp_path):
        (tmp_path / "adapters.safetensors").write_bytes(b"fake")
        mock_push.return_value = "https://huggingface.co/test/repo"

        config = {
            "model": {"path": "test/model"},
            "adapter": {"method": "lora", "rank": 8, "scale": 20.0},
            "training": {},
            "data": {},
        }

        push_adapter_only(tmp_path, "test/repo", config)

        # Check adapter_config.json was created
        config_file = tmp_path / "adapter_config.json"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert data["method"] == "lora"
        assert data["rank"] == 8

    @patch("mlx_forge.hub.upload.push_to_hub")
    def test_creates_readme(self, mock_push, tmp_path):
        (tmp_path / "adapters.safetensors").write_bytes(b"fake")
        mock_push.return_value = "https://huggingface.co/test/repo"

        config = {
            "model": {"path": "test/model"},
            "adapter": {"method": "lora"},
            "training": {},
            "data": {},
        }

        push_adapter_only(tmp_path, "test/repo", config)

        readme = tmp_path / "README.md"
        assert readme.exists()
        assert "mlx-forge" in readme.read_text()

    @patch("mlx_forge.hub.upload.push_to_hub")
    def test_calls_push_with_adapter_only(self, mock_push, tmp_path):
        (tmp_path / "adapters.safetensors").write_bytes(b"fake")
        mock_push.return_value = "url"

        config = {
            "model": {"path": "test/model"},
            "adapter": {},
            "training": {},
            "data": {},
        }

        push_adapter_only(tmp_path, "test/repo", config, private=True)
        mock_push.assert_called_once_with(
            tmp_path, "test/repo", adapter_only=True, private=True, token=None,
        )


class TestCLIFlags:
    """Test CLI flag parsing for push-to-hub."""

    def test_export_parser_has_push_flags(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        # Parse export with push flags
        args = parser.parse_args([
            "export", "--run-id", "test-run",
            "--push-to-hub", "user/model",
            "--adapter-only",
            "--private",
        ])
        assert args.push_to_hub == "user/model"
        assert args.adapter_only is True
        assert args.private is True

    def test_export_parser_push_optional(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["export", "--run-id", "test-run"])
        assert args.push_to_hub is None
        assert args.adapter_only is False
        assert args.private is False
