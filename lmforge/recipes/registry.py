"""Recipe registry — loading, discovery, and application.

Recipes are pre-configured training templates that combine
a recommended model, dataset format, hyperparameters, and
auto-configuration rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Recipe:
    """A training recipe template."""
    id: str
    name: str
    description: str
    category: str  # "sft" or "dpo"
    training_type: str  # "sft" or "dpo"
    data_format: str  # Expected format: "chat", "completions", "text", "preference"
    recommended_models: list[str]  # Model IDs, ordered by preference
    config_template: dict  # Partial TrainingConfig
    auto_rules: list[str] = field(default_factory=list)  # Rule descriptions
    icon: str = ""  # Emoji or icon name for UI

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "training_type": self.training_type,
            "data_format": self.data_format,
            "recommended_models": self.recommended_models,
            "config_template": self.config_template,
            "auto_rules": self.auto_rules,
            "icon": self.icon,
        }


# Built-in recipe directory
_BUILT_IN_DIR = Path(__file__).parent / "built_in"

# Recipe cache
_recipes: dict[str, Recipe] = {}


def _load_built_in_recipes():
    """Load all built-in recipe YAML files."""
    global _recipes
    if _recipes:
        return

    for yaml_path in sorted(_BUILT_IN_DIR.glob("*.yaml")):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            recipe = Recipe(
                id=data["id"],
                name=data["name"],
                description=data["description"],
                category=data.get("category", "sft"),
                training_type=data.get("training_type", "sft"),
                data_format=data.get("data_format", "chat"),
                recommended_models=data.get("recommended_models", []),
                config_template=data.get("config_template", {}),
                auto_rules=data.get("auto_rules", []),
                icon=data.get("icon", ""),
            )
            _recipes[recipe.id] = recipe
        except Exception as e:
            print(f"Warning: Failed to load recipe {yaml_path.name}: {e}")


def list_recipes() -> list[Recipe]:
    """Return all available recipes."""
    _load_built_in_recipes()
    return list(_recipes.values())


def get_recipe(recipe_id: str) -> Optional[Recipe]:
    """Get a recipe by ID."""
    _load_built_in_recipes()
    return _recipes.get(recipe_id)
