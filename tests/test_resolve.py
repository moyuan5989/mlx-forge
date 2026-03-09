"""Tests for model resolution (M7)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cortexlab.models.resolve import is_hf_repo_id, resolve_model


class TestHFRepoIDDetection:
    def test_detects_hf_repo_id(self):
        """Test that HF repo IDs are correctly identified."""
        assert is_hf_repo_id("Qwen/Qwen3-0.8B") is True
        assert is_hf_repo_id("meta-llama/Llama-3.2-3B") is True
        assert is_hf_repo_id("org/model") is True
        assert is_hf_repo_id("org/model/subpath") is True

    def test_detects_local_paths(self, tmp_path):
        """Test that local paths are correctly identified."""
        # Existing directory
        local_dir = tmp_path / "model"
        local_dir.mkdir()
        assert is_hf_repo_id(str(local_dir)) is False

        # Absolute path
        assert is_hf_repo_id("/absolute/path/to/model") is False

        # Relative paths
        assert is_hf_repo_id("./relative/path") is False
        assert is_hf_repo_id("../parent/path") is False

    def test_detects_nonexistent_local_paths(self):
        """Test that paths that look local but don't exist are still treated as local."""
        # Absolute path that doesn't exist
        assert is_hf_repo_id("/nonexistent/path") is False

        # Relative path patterns
        assert is_hf_repo_id("./nonexistent") is False
        assert is_hf_repo_id("../nonexistent") is False


class TestLocalPathResolution:
    def test_resolves_existing_local_path(self, tmp_path):
        """Test that existing local paths resolve correctly."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()

        resolved = resolve_model(str(local_dir))

        assert resolved.source_id == str(local_dir)
        assert resolved.resolved_revision is None
        assert resolved.is_local is True
        assert Path(resolved.local_path) == local_dir.resolve()

    def test_raises_for_missing_local_path(self):
        """Test that missing local paths raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_model("/nonexistent/model/path")

        assert "does not exist" in str(exc_info.value).lower()


class TestHFResolutionOffline:
    @patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"})
    @patch("huggingface_hub.snapshot_download")
    def test_offline_mode_uses_cache(self, mock_snapshot):
        """Test that offline mode skips model_info and uses local cache."""
        # Setup mock
        mock_snapshot.return_value = "/cache/models/model-id/snapshots/abc123"

        resolved = resolve_model("Qwen/Qwen3-0.8B")

        # Should call snapshot_download with local_files_only=True
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["local_files_only"] is True
        assert call_kwargs["repo_id"] == "Qwen/Qwen3-0.8B"

        assert resolved.source_id == "Qwen/Qwen3-0.8B"
        assert resolved.local_path == "/cache/models/model-id/snapshots/abc123"

    @patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"})
    @patch("huggingface_hub.snapshot_download")
    def test_offline_cache_miss_raises_clear_error(self, mock_snapshot):
        """Test that offline mode with cache miss gives clear error."""
        # Simulate cache miss
        mock_snapshot.side_effect = FileNotFoundError("Model not found")

        with pytest.raises(ValueError) as exc_info:
            resolve_model("Qwen/Qwen3-0.8B")

        error_msg = str(exc_info.value)
        assert "not found in local cache" in error_msg.lower()
        assert "offline mode" in error_msg.lower()
        assert "download" in error_msg.lower()


class TestHFResolutionOnline:
    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.model_info")
    def test_online_mode_resolves_latest_revision(self, mock_info, mock_snapshot):
        """Test that online mode resolves to latest commit hash."""
        # Setup mocks
        mock_info.return_value = Mock(sha="abc123def456")
        mock_snapshot.return_value = "/cache/models/model-id/snapshots/abc123def456"

        resolved = resolve_model("Qwen/Qwen3-0.8B")

        # Should call model_info to get latest revision
        mock_info.assert_called_once_with("Qwen/Qwen3-0.8B", token=None)

        # Should download with that revision
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args[1]
        assert call_kwargs["repo_id"] == "Qwen/Qwen3-0.8B"
        assert call_kwargs["revision"] == "abc123def456"
        assert call_kwargs["local_files_only"] is False

        assert resolved.source_id == "Qwen/Qwen3-0.8B"
        assert resolved.resolved_revision == "abc123def456"
        assert resolved.is_local is False

    @patch("huggingface_hub.snapshot_download")
    def test_pinned_revision_skips_model_info(self, mock_snapshot):
        """Test that providing a revision skips model_info call."""
        mock_snapshot.return_value = "/cache/models/model-id/snapshots/pinned123"

        with patch("huggingface_hub.model_info") as mock_info:
            resolve_model("Qwen/Qwen3-0.8B", revision="pinned123")

            # Should NOT call model_info when revision is provided
            mock_info.assert_not_called()

            # Should download with pinned revision
            mock_snapshot.assert_called_once()
            call_kwargs = mock_snapshot.call_args[1]
            assert call_kwargs["revision"] == "pinned123"


class TestHFErrorHandling:
    @patch("huggingface_hub.model_info")
    def test_gated_model_raises_permission_error(self, mock_info):
        """Test that gated models produce clear auth instructions."""
        from huggingface_hub.utils import GatedRepoError

        # Create a mock response object
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}

        mock_info.side_effect = GatedRepoError("Authentication required", response=mock_response)

        with pytest.raises(PermissionError) as exc_info:
            resolve_model("meta-llama/Llama-3.2-3B")

        error_msg = str(exc_info.value)
        assert "gated model" in error_msg.lower()
        assert "huggingface-cli login" in error_msg
        assert "HF_TOKEN" in error_msg

    @patch("huggingface_hub.model_info")
    def test_repo_not_found_raises_value_error(self, mock_info):
        """Test that non-existent repos produce clear error."""
        from huggingface_hub.utils import RepositoryNotFoundError

        # Create a mock response object
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}

        mock_info.side_effect = RepositoryNotFoundError("Repository not found", response=mock_response)

        with pytest.raises(ValueError) as exc_info:
            resolve_model("nonexistent/model")

        error_msg = str(exc_info.value)
        assert "not found" in error_msg.lower()
        assert "huggingface.co" in error_msg.lower()

    @patch("huggingface_hub.model_info")
    def test_network_error_raises_value_error(self, mock_info):
        """Test that network errors produce clear error."""
        mock_info.side_effect = ConnectionError("Network error")

        with pytest.raises(ValueError) as exc_info:
            resolve_model("Qwen/Qwen3-0.8B")

        error_msg = str(exc_info.value)
        assert "network unavailable" in error_msg.lower()
        assert "HF_HUB_OFFLINE" in error_msg


class TestResolutionMetadata:
    @patch("huggingface_hub.snapshot_download")
    @patch("huggingface_hub.model_info")
    def test_resolution_metadata_for_hf_model(self, mock_info, mock_snapshot):
        """Test that resolution metadata is correct for HF models."""
        mock_info.return_value = Mock(sha="abc123")
        mock_snapshot.return_value = "/cache/path"

        resolved = resolve_model("Qwen/Qwen3-0.8B")

        metadata = resolved.resolution_metadata
        assert metadata["source_id"] == "Qwen/Qwen3-0.8B"
        assert metadata["resolved_revision"] == "abc123"
        assert metadata["local_path"] == "/cache/path"
        assert metadata["is_local"] is False

    def test_resolution_metadata_for_local_model(self, tmp_path):
        """Test that resolution metadata is correct for local models."""
        local_dir = tmp_path / "model"
        local_dir.mkdir()

        resolved = resolve_model(str(local_dir))

        metadata = resolved.resolution_metadata
        assert metadata["source_id"] == str(local_dir)
        assert metadata["local_path"] == str(local_dir.resolve())
        assert metadata["is_local"] is True
