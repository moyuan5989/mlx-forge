"""Recipes API — browse and apply training recipes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mlx_forge.studio.security import validate_safe_name

router = APIRouter(prefix="/api/v2/recipes", tags=["recipes"])


@router.get("")
def list_recipes():
    """List all available training recipes."""
    from mlx_forge.recipes.registry import list_recipes as _list_recipes
    return [r.to_dict() for r in _list_recipes()]


@router.get("/{recipe_id}")
def get_recipe(recipe_id: str):
    """Get a specific recipe by ID."""
    try:
        validate_safe_name(recipe_id, "recipe_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid recipe_id: {recipe_id!r}")
    from mlx_forge.recipes.registry import get_recipe as _get_recipe
    recipe = _get_recipe(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")
    return recipe.to_dict()


@router.post("/{recipe_id}/resolve")
def resolve_recipe(recipe_id: str, body: dict):
    """Resolve a recipe into a full training config.

    Body should contain:
    - model_id: HuggingFace model ID
    - train_path: Path to training data
    - valid_path: Path to validation data
    - dataset_samples: (optional) Number of samples
    - overrides: (optional) Config overrides
    """
    try:
        validate_safe_name(recipe_id, "recipe_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid recipe_id: {recipe_id!r}")
    from mlx_forge.recipes.auto_config import resolve_config
    from mlx_forge.recipes.registry import get_recipe as _get_recipe

    recipe = _get_recipe(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")

    model_id = body.get("model_id")
    train_path = body.get("train_path")
    valid_path = body.get("valid_path")

    if not model_id:
        raise HTTPException(status_code=400, detail="'model_id' is required")
    if not train_path:
        raise HTTPException(status_code=400, detail="'train_path' is required")
    if not valid_path:
        raise HTTPException(status_code=400, detail="'valid_path' is required")

    try:
        config = resolve_config(
            recipe,
            model_id=model_id,
            train_path=train_path,
            valid_path=valid_path,
            dataset_samples=body.get("dataset_samples"),
            overrides=body.get("overrides"),
        )
        return config
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
