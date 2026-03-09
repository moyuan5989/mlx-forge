"""Recipe Service — recipe operations for Studio."""

from __future__ import annotations

from typing import Optional

from cortexlab.recipes.auto_config import resolve_config
from cortexlab.recipes.registry import get_recipe, list_recipes


class RecipeService:
    """Provides recipe operations for the Studio API."""

    def list_recipes(self) -> list[dict]:
        """List all available recipes."""
        return [r.to_dict() for r in list_recipes()]

    def get_recipe(self, recipe_id: str) -> Optional[dict]:
        """Get a recipe by ID."""
        recipe = get_recipe(recipe_id)
        if recipe is None:
            return None
        return recipe.to_dict()

    def resolve(
        self,
        recipe_id: str,
        model_id: str,
        train_path: str,
        valid_path: str,
        *,
        dataset_samples: Optional[int] = None,
        overrides: Optional[dict] = None,
    ) -> dict:
        """Resolve a recipe into a full training config."""
        recipe = get_recipe(recipe_id)
        if recipe is None:
            raise ValueError(f"Recipe '{recipe_id}' not found")

        return resolve_config(
            recipe,
            model_id=model_id,
            train_path=train_path,
            valid_path=valid_path,
            dataset_samples=dataset_samples,
            overrides=overrides,
        )
