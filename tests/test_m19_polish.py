"""M19 — Frontend polish, docs, and version bump tests."""

from __future__ import annotations


class TestVersionBump:
    def test_version_py(self):
        from mlx_forge._version import __version__
        assert __version__ == "0.3.0"

    def test_pyproject_version(self):
        from pathlib import Path

        import tomllib

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == "0.3.0"


class TestContributing:
    def test_no_broken_urls(self):
        from pathlib import Path

        text = (Path(__file__).parent.parent / "CONTRIBUTING.md").read_text()
        assert "MLX Forge.git" not in text
        assert "MLX Forge/issues" not in text
        assert "mlx-forge.git" in text


class TestChangelog:
    def test_has_030_entry(self):
        from pathlib import Path

        text = (Path(__file__).parent.parent / "CHANGELOG.md").read_text()
        assert "## [0.3.0]" in text
        assert "## [0.2.11]" in text


class TestTransformersPinned:
    def test_upper_bound(self):
        from pathlib import Path

        import tomllib

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        deps = data["project"]["dependencies"]
        transformers_dep = [d for d in deps if d.startswith("transformers")]
        assert len(transformers_dep) == 1
        assert "<5.0" in transformers_dep[0]


class TestSecurityModule:
    def test_importable(self):
        from mlx_forge.studio.security import validate_safe_name, validate_safe_path
        assert callable(validate_safe_name)
        assert callable(validate_safe_path)


class TestFuseModule:
    def test_importable(self):
        from mlx_forge.adapters.fuse import fuse_model, save_fused_model
        assert callable(fuse_model)
        assert callable(save_fused_model)
