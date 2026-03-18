"""LoRA adapter modules for MLX Forge v0."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear and nn.QuantizedLinear.

    Implements low-rank adaptation: W' = W + (scale/r) * B * A
    where A is (r, in_features), B is (out_features, r).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        scale: float = 20.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scale = scale
        self.dropout_prob = dropout

        # LoRA parameters: A (r x in), B (out x r)
        # Initialize A with Kaiming uniform, B with zeros (standard LoRA init)
        scale_init = 1 / mx.sqrt(mx.array(in_features, dtype=mx.float32))
        self.lora_a = mx.random.uniform(
            low=-scale_init.item(),
            high=scale_init.item(),
            shape=(r, in_features),
        )
        self.lora_b = mx.zeros((out_features, r))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def __call__(self, x):
        """Forward pass: y = base(x) + (scale/r) * (x @ A.T) @ B.T"""
        # Base layer output (frozen weights)
        base_out = self.base_layer(x)

        # LoRA delta: (scale/r) * (x @ A.T) @ B.T
        lora_out = x @ self.lora_a.T  # (batch, r)
        if self.dropout is not None and self.training:
            lora_out = self.dropout(lora_out)
        lora_out = lora_out @ self.lora_b.T  # (batch, out_features)
        lora_out = lora_out * (self.scale / self.r)

        return base_out + lora_out

    @classmethod
    def from_base(
        cls, base_linear, *, r: int, scale: float, dropout: float = 0.0
    ) -> LoRALinear:
        """Create a LoRALinear from an existing Linear or QuantizedLinear module.

        The base linear's weight is frozen. The LoRALinear adds trainable low-rank
        parameters on top.
        """
        # Get dimensions from base linear
        if isinstance(base_linear, nn.QuantizedLinear):
            # QuantizedLinear weight is packed: shape (out, packed_in)
            # Actual in_features = packed_in * (32 // bits)
            out_features = base_linear.weight.shape[0]
            in_features = base_linear.weight.shape[1] * (32 // base_linear.bits)
            bias = hasattr(base_linear, "bias") and base_linear.bias is not None
        elif isinstance(base_linear, nn.Linear):
            out_features, in_features = base_linear.weight.shape
            # MLX Linear without bias doesn't have a bias attribute at all
            bias = hasattr(base_linear, "bias") and base_linear.bias is not None
        elif hasattr(base_linear, "weight"):
            # Other linear-like module
            out_features, in_features = base_linear.weight.shape
            bias = hasattr(base_linear, "bias") and base_linear.bias is not None
        else:
            raise ValueError(
                f"Cannot create LoRALinear from {type(base_linear).__name__}. "
                "Expected nn.Linear or a module with a 'weight' attribute."
            )

        # Create LoRA module
        lora = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            scale=scale,
            dropout=dropout,
            bias=bias,
        )

        # Store reference to base module (frozen)
        lora.base_layer = base_linear

        # Freeze base layer parameters
        base_linear.freeze()

        return lora

    def fuse(self):
        """Merge LoRA weights back into the base weight and return a plain Linear.

        W_fused = W_base + (scale/r) * B * A
        """
        if not hasattr(self, "base_layer"):
            raise ValueError(
                "Cannot fuse: this LoRALinear was not created from a base layer."
            )

        # Compute fused weight: W + (scale/r) * B @ A
        lora_weight = self.lora_b @ self.lora_a  # (out_features, in_features)
        lora_weight = lora_weight * (self.scale / self.r)

        fused_weight = self.base_layer.weight + lora_weight

        # Create new Linear with fused weight
        fused_linear = nn.Linear(self.in_features, self.out_features)
        fused_linear.weight = fused_weight

        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            fused_linear.bias = self.base_layer.bias

        return fused_linear


class LoRAEmbedding(nn.Module):
    """LoRA wrapper for nn.Embedding and nn.QuantizedEmbedding.

    Implements low-rank adaptation for embedding tables.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        scale: float = 20.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.r = r
        self.scale = scale
        self.dropout_prob = dropout

        # LoRA parameters for embedding
        # A: (num_embeddings, r), B: (r, embedding_dim)
        scale_init = 1 / mx.sqrt(mx.array(embedding_dim, dtype=mx.float32))
        self.lora_a = mx.random.uniform(
            low=-scale_init.item(),
            high=scale_init.item(),
            shape=(num_embeddings, r),
        )
        self.lora_b = mx.zeros((r, embedding_dim))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else None

    def __call__(self, x):
        """Forward pass: base_embedding(x) + LoRA delta."""
        # Base embedding output (frozen weights)
        base_out = self.base_layer(x)

        # LoRA delta: (lora_a[x] @ lora_b) * (scale/r)
        lora_a_embed = self.lora_a[x]  # (batch, seq, r)
        if self.dropout is not None and self.training:
            lora_a_embed = self.dropout(lora_a_embed)

        lora_out = lora_a_embed @ self.lora_b  # (batch, seq, embedding_dim)
        lora_out = lora_out * (self.scale / self.r)

        return base_out + lora_out

    @classmethod
    def from_base(
        cls, base_embedding, *, r: int, scale: float, dropout: float = 0.0
    ) -> LoRAEmbedding:
        """Create a LoRAEmbedding from an existing Embedding module."""
        if isinstance(base_embedding, nn.Embedding):
            num_embeddings, embedding_dim = base_embedding.weight.shape
        elif hasattr(base_embedding, "weight"):
            num_embeddings, embedding_dim = base_embedding.weight.shape
        else:
            raise ValueError(
                f"Cannot create LoRAEmbedding from {type(base_embedding).__name__}. "
                "Expected nn.Embedding or a module with a 'weight' attribute."
            )

        lora = cls(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            r=r,
            scale=scale,
            dropout=dropout,
        )

        lora.base_layer = base_embedding
        base_embedding.freeze()

        return lora

    def fuse(self):
        """Merge LoRA weights back into the base embedding weight."""
        if not hasattr(self, "base_layer"):
            raise ValueError(
                "Cannot fuse: this LoRAEmbedding was not created from a base layer."
            )

        # Compute fused weight: W + (scale/r) * A @ B
        lora_weight = self.lora_a @ self.lora_b  # (num_embeddings, embedding_dim)
        lora_weight = lora_weight * (self.scale / self.r)

        fused_weight = self.base_layer.weight + lora_weight

        # Create new Embedding with fused weight
        fused_embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        fused_embedding.weight = fused_weight

        return fused_embedding


def apply_lora(model, targets: list[tuple[str, object]], config) -> object:
    """Apply LoRA adapters to matched modules in-place.

    Returns the modified model.
    """
    lora_layers = []

    # Import DoRA if needed
    use_dora = getattr(config, "method", "lora") == "dora"
    if use_dora:
        from mlx_forge.adapters.dora import DoRALinear

    for name, module in targets:
        # Determine module type and create appropriate LoRA wrapper
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)) or (
            hasattr(module, "weight") and len(getattr(module, "weight").shape) == 2
        ):
            # Linear-like module — use DoRA or LoRA based on config
            if use_dora:
                lora_module = DoRALinear.from_base(
                    module,
                    r=config.rank,
                    scale=config.scale,
                    dropout=config.dropout,
                )
            else:
                lora_module = LoRALinear.from_base(
                    module,
                    r=config.rank,
                    scale=config.scale,
                    dropout=config.dropout,
                )
            lora_layers.append((name, lora_module))

        elif isinstance(module, (nn.Embedding, nn.QuantizedEmbedding)) or (
            hasattr(module, "weight")
            and hasattr(module, "__call__")
            and "embedding" in type(module).__name__.lower()
        ):
            # Embedding-like module
            lora_module = LoRAEmbedding.from_base(
                module,
                r=config.rank,
                scale=config.scale,
                dropout=config.dropout,
            )
            lora_layers.append((name, lora_module))

        else:
            raise ValueError(
                f"Cannot apply LoRA to {type(module).__name__} at '{name}'. "
                "LoRA only supports Linear and Embedding modules."
            )

    # Update model with LoRA layers
    if lora_layers:
        model.update_modules(tree_unflatten(lora_layers))

    # Log what was converted
    print(f"Applied LoRA to {len(lora_layers)} modules:")
    for name, _ in lora_layers[:10]:  # Show first 10
        print(f"  - {name}")
    if len(lora_layers) > 10:
        print(f"  ... and {len(lora_layers) - 10} more")

    # Count trainable parameters
    trainable_params = sum(
        p.size for _, p in tree_flatten(model.trainable_parameters())
    )
    print(f"Trainable parameters: {trainable_params:,}")

    return model
