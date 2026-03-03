"""Tests for V2 recipe system."""

from __future__ import annotations

import pytest

from lmforge.recipes.registry import Recipe, get_recipe, list_recipes
from lmforge.recipes.auto_config import resolve_config


class TestRecipeRegistry:
    """Test recipe loading and discovery."""

    def test_list_recipes_returns_all(self):
        """list_recipes returns 4 built-in recipes."""
        recipes = list_recipes()
        assert len(recipes) >= 4

    def test_list_recipes_returns_recipe_objects(self):
        """Each recipe is a Recipe dataclass."""
        recipes = list_recipes()
        for r in recipes:
            assert isinstance(r, Recipe)

    def test_get_recipe_by_id(self):
        """Can retrieve a recipe by its ID."""
        recipe = get_recipe("chat-sft")
        assert recipe is not None
        assert recipe.id == "chat-sft"
        assert recipe.name == "Chat Fine-Tuning"

    def test_get_nonexistent_recipe(self):
        """get_recipe returns None for unknown ID."""
        recipe = get_recipe("nonexistent-recipe")
        assert recipe is None

    def test_recipe_ids_are_unique(self):
        """All recipe IDs are unique."""
        recipes = list_recipes()
        ids = [r.id for r in recipes]
        assert len(ids) == len(set(ids))


class TestBuiltInRecipes:
    """Test the 4 built-in recipe templates."""

    def test_chat_sft_recipe(self):
        """Chat SFT recipe has correct properties."""
        r = get_recipe("chat-sft")
        assert r is not None
        assert r.training_type == "sft"
        assert r.data_format == "chat"
        assert len(r.recommended_models) > 0
        assert "adapter" in r.config_template

    def test_instruction_sft_recipe(self):
        """Instruction SFT recipe has correct properties."""
        r = get_recipe("instruction-sft")
        assert r is not None
        assert r.training_type == "sft"
        assert r.data_format == "completions"

    def test_writing_style_recipe(self):
        """Writing style recipe uses text format."""
        r = get_recipe("writing-style")
        assert r is not None
        assert r.training_type == "sft"
        assert r.data_format == "text"
        # Writing style should have higher rank for more capacity
        template_adapter = r.config_template.get("adapter", {})
        assert template_adapter.get("rank", 0) >= 16

    def test_preference_dpo_recipe(self):
        """Preference DPO recipe has correct properties."""
        r = get_recipe("preference-dpo")
        assert r is not None
        assert r.training_type == "dpo"
        assert r.data_format == "preference"
        training = r.config_template.get("training", {})
        assert training.get("training_type") == "dpo"
        assert training.get("dpo_reference_free") is True

    def test_all_recipes_have_recommended_models(self):
        """All recipes have at least one recommended model."""
        for r in list_recipes():
            assert len(r.recommended_models) > 0, f"{r.id} has no recommended models"

    def test_recipe_to_dict(self):
        """Recipe serializes to dict correctly."""
        r = get_recipe("chat-sft")
        d = r.to_dict()
        assert d["id"] == "chat-sft"
        assert "config_template" in d
        assert "recommended_models" in d


class TestAutoConfig:
    """Test auto-configuration rule resolution."""

    def test_resolve_basic(self):
        """resolve_config produces a complete config."""
        recipe = get_recipe("chat-sft")
        config = resolve_config(
            recipe,
            model_id="Qwen/Qwen3-0.6B",
            train_path="/data/train.jsonl",
            valid_path="/data/val.jsonl",
        )

        assert config["schema_version"] == 1
        assert config["model"]["path"] == "Qwen/Qwen3-0.6B"
        assert config["data"]["train"] == "/data/train.jsonl"
        assert config["data"]["valid"] == "/data/val.jsonl"
        assert "training" in config

    def test_resolve_with_overrides(self):
        """User overrides are applied."""
        recipe = get_recipe("chat-sft")
        config = resolve_config(
            recipe,
            model_id="Qwen/Qwen3-0.6B",
            train_path="/data/train.jsonl",
            valid_path="/data/val.jsonl",
            overrides={"training.learning_rate": 1e-4},
        )

        assert config["training"]["learning_rate"] == 1e-4

    def test_resolve_dpo_recipe(self):
        """DPO recipe sets training_type correctly."""
        recipe = get_recipe("preference-dpo")
        config = resolve_config(
            recipe,
            model_id="Qwen/Qwen3-4B",
            train_path="/data/train.jsonl",
            valid_path="/data/val.jsonl",
        )

        assert config["training"]["training_type"] == "dpo"

    def test_resolve_with_small_dataset(self):
        """Small dataset triggers auto-config rules."""
        from lmforge.models.memory import HardwareProfile

        recipe = get_recipe("chat-sft")
        hw = HardwareProfile(total_memory_gb=64.0, training_budget_gb=48.0)
        config = resolve_config(
            recipe,
            model_id="Qwen/Qwen3-0.6B",
            train_path="/data/train.jsonl",
            valid_path="/data/val.jsonl",
            hardware=hw,
            dataset_samples=100,
        )

        assert config["training"]["num_iters"] == 500

    def test_resolve_with_low_memory(self):
        """Low memory triggers QLoRA auto-config."""
        from lmforge.models.memory import HardwareProfile

        recipe = get_recipe("chat-sft")
        hw = HardwareProfile(total_memory_gb=12.0, training_budget_gb=9.0)
        config = resolve_config(
            recipe,
            model_id="Qwen/Qwen3-4B",
            train_path="/data/train.jsonl",
            valid_path="/data/val.jsonl",
            hardware=hw,
        )

        assert config["model"]["quantization"]["bits"] == 4
