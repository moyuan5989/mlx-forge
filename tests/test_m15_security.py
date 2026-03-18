"""M15 — Security hardening tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlx_forge.studio.security import validate_safe_name, validate_safe_path

# ── validate_safe_name ──


class TestValidateSafeName:
    def test_valid_simple(self):
        assert validate_safe_name("my-run-01") == "my-run-01"

    def test_valid_with_dots(self):
        assert validate_safe_name("run.v2.3") == "run.v2.3"

    def test_valid_with_underscores(self):
        assert validate_safe_name("my_run_01") == "my_run_01"

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name("../../etc/passwd")

    def test_rejects_slash(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name("foo/bar")

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name("foo\\bar")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name("")

    def test_rejects_dot_start(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name(".hidden")

    def test_rejects_hyphen_start(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name("-flag")

    def test_rejects_null_byte(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name("name\x00evil")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_safe_name("a" * 129)

    def test_max_length_ok(self):
        name = "a" * 128
        assert validate_safe_name(name) == name

    def test_custom_label(self):
        with pytest.raises(ValueError, match="Invalid run_id"):
            validate_safe_name("../bad", label="run_id")


# ── validate_safe_path ──


class TestValidateSafePath:
    def test_valid_subpath(self, tmp_path):
        child = tmp_path / "sub" / "file.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = validate_safe_path(Path("sub/file.txt"), tmp_path)
        assert result == child

    def test_rejects_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="Path escapes root"):
            validate_safe_path(Path("../../etc/passwd"), tmp_path)

    def test_rejects_absolute(self, tmp_path):
        with pytest.raises(ValueError, match="Path escapes root"):
            validate_safe_path(Path("/etc/passwd"), tmp_path)

    def test_rejects_null_byte(self, tmp_path):
        with pytest.raises(ValueError, match="null byte"):
            validate_safe_path(Path("foo\x00bar"), tmp_path)


# ── API endpoint validation (via ASGI test client) ──


@pytest.fixture
def test_client():
    """Create a test client for the Studio app."""
    try:
        from httpx import ASGITransport, AsyncClient
    except ImportError:
        pytest.skip("httpx not installed")
    from mlx_forge.studio.server import create_app

    app = create_app(runs_dir="/tmp/mlxforge_test_security/runs")
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_runs_endpoint_rejects_dotdot(test_client):
    """Names starting with dot are rejected."""
    r = await test_client.get("/api/v1/runs/..secret")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_runs_endpoint_rejects_space(test_client):
    """Names with spaces are rejected."""
    r = await test_client.get("/api/v1/runs/bad name")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_runs_delete_rejects_invalid(test_client):
    r = await test_client.delete("/api/v1/runs/-badname")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_data_library_rejects_dotstart(test_client):
    r = await test_client.get("/api/v2/data/datasets/.hidden")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_valid_run_id_passes_validation(test_client):
    # valid name, run won't exist → 404 not 400
    r = await test_client.get("/api/v1/runs/valid-run-01")
    assert r.status_code == 404
