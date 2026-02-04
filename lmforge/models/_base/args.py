"""Base model arguments class."""

from __future__ import annotations

import inspect
from dataclasses import dataclass


@dataclass
class BaseModelArgs:
    """
    Base class for model configuration arguments.

    Provides from_dict() to create ModelArgs from config.json,
    filtering to only fields defined in the dataclass.
    """

    @classmethod
    def from_dict(cls, params: dict) -> "BaseModelArgs":
        """
        Create ModelArgs from a config dict, filtering to valid fields.

        This allows config.json to have extra fields that aren't used
        by the model implementation.

        Args:
            params: Dictionary of parameters (typically from config.json)

        Returns:
            Instance of the ModelArgs subclass
        """
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
