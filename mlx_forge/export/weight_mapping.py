"""Architecture-specific weight name translations for GGUF export.

Maps MLX Forge weight names to GGUF tensor names.
"""

from __future__ import annotations

# Llama/Mistral weight mapping
LLAMA_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    # Per-layer patterns (use .format(i=layer_index))
    "model.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "model.layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
}

# Phi-3 weight mapping
PHI3_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{i}.self_attn.qkv_proj.weight": "blk.{i}.attn_qkv.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "model.layers.{i}.mlp.gate_up_proj.weight": "blk.{i}.ffn_gate_up.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
}

# Qwen2 uses the same structure as Llama
QWEN2_WEIGHT_MAP = LLAMA_WEIGHT_MAP.copy()

# Cohere weight mapping (parallel attention, LayerNorm)
COHERE_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "model.norm.bias": "output_norm.bias",
    "lm_head.weight": "output.weight",
    "model.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "model.layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.input_layernorm.bias": "blk.{i}.attn_norm.bias",
}

# Mixtral MoE weight mapping (base attention layers + expert routing)
MIXTRAL_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "model.layers.{i}.block_sparse_moe.gate.weight": "blk.{i}.ffn_gate_inp.weight",
    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
}

# InternLM2 weight mapping (fused wqkv)
INTERNLM2_WEIGHT_MAP = {
    "model.tok_embeddings.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "output.weight": "output.weight",
    "model.layers.{i}.attention.wqkv.weight": "blk.{i}.attn_qkv.weight",
    "model.layers.{i}.attention.wo.weight": "blk.{i}.attn_output.weight",
    "model.layers.{i}.feed_forward.w1.weight": "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.feed_forward.w3.weight": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.feed_forward.w2.weight": "blk.{i}.ffn_down.weight",
    "model.layers.{i}.attention_norm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.ffn_norm.weight": "blk.{i}.ffn_norm.weight",
}

# GLM4 weight mapping (fused QKV)
GLM4_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.{i}.self_attn.qkv_proj.weight": "blk.{i}.attn_qkv.weight",
    "model.layers.{i}.self_attn.qkv_proj.bias": "blk.{i}.attn_qkv.bias",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "model.layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
}

# StarCoder2 weight mapping (LayerNorm, bias in attention)
STARCODER2_WEIGHT_MAP = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "model.norm.bias": "output_norm.bias",
    "lm_head.weight": "output.weight",
    "model.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.q_proj.bias": "blk.{i}.attn_q.bias",
    "model.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.k_proj.bias": "blk.{i}.attn_k.bias",
    "model.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.v_proj.bias": "blk.{i}.attn_v.bias",
    "model.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "model.layers.{i}.self_attn.o_proj.bias": "blk.{i}.attn_output.bias",
    "model.layers.{i}.mlp.c_fc.weight": "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.c_fc.bias": "blk.{i}.ffn_up.bias",
    "model.layers.{i}.mlp.c_proj.weight": "blk.{i}.ffn_down.weight",
    "model.layers.{i}.mlp.c_proj.bias": "blk.{i}.ffn_down.bias",
    "model.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "model.layers.{i}.input_layernorm.bias": "blk.{i}.attn_norm.bias",
    "model.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.bias": "blk.{i}.ffn_norm.bias",
}

# Architecture name to weight map
WEIGHT_MAPS = {
    "llama": LLAMA_WEIGHT_MAP,
    "mistral": LLAMA_WEIGHT_MAP,
    "qwen2": QWEN2_WEIGHT_MAP,
    "phi3": PHI3_WEIGHT_MAP,
    # New architectures - Llama-compatible base layers
    "mixtral": MIXTRAL_WEIGHT_MAP,
    "olmo2": LLAMA_WEIGHT_MAP,
    "granite": LLAMA_WEIGHT_MAP,
    "stablelm": LLAMA_WEIGHT_MAP,
    "llama4": LLAMA_WEIGHT_MAP,
    # Architecture-specific maps
    "cohere": COHERE_WEIGHT_MAP,
    "cohere2": COHERE_WEIGHT_MAP,
    "internlm2": INTERNLM2_WEIGHT_MAP,
    "starcoder2": STARCODER2_WEIGHT_MAP,
    "glm4": GLM4_WEIGHT_MAP,
}

# Architecture name mapping for GGUF metadata
GGUF_ARCH_NAMES = {
    "llama": "llama",
    "mistral": "llama",  # Mistral uses llama architecture in GGUF
    "qwen2": "llama",    # Qwen2 is llama-compatible
    "phi3": "phi3",
    "mixtral": "llama",
    "olmo2": "llama",
    "granite": "llama",
    "stablelm": "llama",
    "llama4": "llama",
    "cohere": "command-r",
    "cohere2": "command-r",
    "internlm2": "internlm2",
    "starcoder2": "starcoder2",
    "glm4": "llama",
}

SUPPORTED_GGUF_ARCHITECTURES = list(WEIGHT_MAPS.keys())


def get_weight_map(model_type: str) -> dict[str, str]:
    """Get the weight name mapping for a given model architecture.

    Args:
        model_type: Architecture name (llama, mistral, qwen2, phi3)

    Returns:
        Dictionary mapping MLX weight names to GGUF tensor names.

    Raises:
        ValueError: If architecture is not supported.
    """
    model_type = model_type.lower()
    if model_type not in WEIGHT_MAPS:
        raise ValueError(
            f"GGUF export not supported for architecture '{model_type}'. "
            f"Supported: {SUPPORTED_GGUF_ARCHITECTURES}"
        )
    return WEIGHT_MAPS[model_type]


def translate_weight_name(mlx_name: str, model_type: str) -> str | None:
    """Translate a single MLX weight name to GGUF tensor name.

    Args:
        mlx_name: MLX Forge weight name
        model_type: Architecture type

    Returns:
        GGUF tensor name, or None if no mapping found.
    """
    weight_map = get_weight_map(model_type)

    # Try direct match first
    if mlx_name in weight_map:
        return weight_map[mlx_name]

    # Try pattern match (for layer-indexed weights)
    import re
    for mlx_pattern, gguf_pattern in weight_map.items():
        if "{i}" not in mlx_pattern:
            continue
        # Convert pattern to regex
        regex = re.escape(mlx_pattern).replace(r"\{i\}", r"(\d+)")
        match = re.fullmatch(regex, mlx_name)
        if match:
            layer_idx = match.group(1)
            return gguf_pattern.format(i=layer_idx)

    return None
