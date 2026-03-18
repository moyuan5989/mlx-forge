"""GGUF file writer -- converts fused safetensors models to GGUF v3 format.

No external dependency -- writes binary format directly via struct.pack().
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from mlx_forge.export.gguf_constants import (
    GGUF_MAGIC,
    GGUF_VERSION,
    GGMLType,
    GGUFValueType,
)
from mlx_forge.export.weight_mapping import (
    GGUF_ARCH_NAMES,
    SUPPORTED_GGUF_ARCHITECTURES,
    translate_weight_name,
)


def convert_to_gguf(
    model_dir: str | Path,
    output_path: str | Path,
    *,
    quantization: str = "f16",
) -> Path:
    """Convert a fused safetensors model to GGUF format.

    Args:
        model_dir: Directory containing model.safetensors and config.json
        output_path: Output GGUF file path
        quantization: Quantization type ("f16" or "f32")

    Returns:
        Path to the output GGUF file.
    """
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    # Determine architecture
    model_type = config.get("model_type", "").lower()
    if model_type not in SUPPORTED_GGUF_ARCHITECTURES:
        raise ValueError(
            f"GGUF export not supported for architecture '{model_type}'. "
            f"Supported: {SUPPORTED_GGUF_ARCHITECTURES}"
        )

    gguf_arch = GGUF_ARCH_NAMES[model_type]

    # Determine tensor type
    if quantization == "f16":
        ggml_type = GGMLType.F16
        np_dtype = np.float16
    elif quantization == "f32":
        ggml_type = GGMLType.F32
        np_dtype = np.float32
    else:
        raise ValueError(f"Unsupported quantization: {quantization}. Use 'f16' or 'f32'.")

    # Load weights
    from safetensors import safe_open
    weights = {}
    model_file = model_dir / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")

    with safe_open(str(model_file), framework="numpy") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    # Translate weight names and prepare tensors
    tensors = []
    for mlx_name, array in weights.items():
        gguf_name = translate_weight_name(mlx_name, model_type)
        if gguf_name is None:
            print(f"  Warning: skipping unmapped weight '{mlx_name}'")
            continue

        # Convert to target dtype
        data = array.astype(np_dtype)
        tensors.append((gguf_name, data, ggml_type))

    # Build metadata
    metadata = _build_metadata(config, gguf_arch, quantization)

    # Load tokenizer for vocab
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.exists():
        with open(tokenizer_path) as f:
            tokenizer_data = json.load(f)
        vocab = _extract_vocab(tokenizer_data)
        if vocab:
            metadata.append(("tokenizer.ggml.model", GGUFValueType.STRING, "gpt2"))
            metadata.append(("tokenizer.ggml.tokens", GGUFValueType.ARRAY, vocab))

    # Write GGUF file
    _write_gguf(output_path, metadata, tensors)

    print(f"Wrote GGUF file: {output_path}")
    print(f"  Architecture: {gguf_arch}")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Quantization: {quantization}")

    return output_path


def _build_metadata(config: dict, arch: str, quantization: str) -> list[tuple]:
    """Build GGUF metadata key-value pairs."""
    metadata = []

    metadata.append(("general.architecture", GGUFValueType.STRING, arch))
    metadata.append(("general.name", GGUFValueType.STRING, config.get("_name_or_path", "mlx-forge-model")))

    # File type
    file_type = 1 if quantization == "f16" else 0  # 0=F32, 1=F16
    metadata.append(("general.file_type", GGUFValueType.UINT32, file_type))

    # Architecture-specific metadata
    prefix = arch

    if "hidden_size" in config:
        metadata.append((f"{prefix}.embedding_length", GGUFValueType.UINT32, config["hidden_size"]))

    if "num_hidden_layers" in config:
        metadata.append((f"{prefix}.block_count", GGUFValueType.UINT32, config["num_hidden_layers"]))

    if "num_attention_heads" in config:
        metadata.append((f"{prefix}.attention.head_count", GGUFValueType.UINT32, config["num_attention_heads"]))

    if "num_key_value_heads" in config:
        metadata.append((f"{prefix}.attention.head_count_kv", GGUFValueType.UINT32, config["num_key_value_heads"]))

    if "max_position_embeddings" in config:
        metadata.append((f"{prefix}.context_length", GGUFValueType.UINT32, config["max_position_embeddings"]))

    if "intermediate_size" in config:
        metadata.append((f"{prefix}.feed_forward_length", GGUFValueType.UINT32, config["intermediate_size"]))

    if "rms_norm_eps" in config:
        metadata.append((f"{prefix}.attention.layer_norm_rms_epsilon", GGUFValueType.FLOAT32, config["rms_norm_eps"]))

    if "vocab_size" in config:
        metadata.append((f"{prefix}.vocab_size", GGUFValueType.UINT32, config["vocab_size"]))

    return metadata


def _extract_vocab(tokenizer_data: dict) -> list[str] | None:
    """Extract vocabulary from tokenizer.json."""
    model = tokenizer_data.get("model", {})
    vocab = model.get("vocab", {})
    if not vocab:
        return None

    # Sort by token ID and return token strings
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])
    return [token for token, _ in sorted_tokens]


def _write_gguf(output_path: Path, metadata: list[tuple], tensors: list[tuple]) -> None:
    """Write GGUF v3 binary file.

    Format:
    1. Header: magic, version, tensor_count, metadata_kv_count
    2. Metadata key-value pairs
    3. Tensor info (name, ndims, dims, type, offset)
    4. Padding to 32-byte alignment
    5. Tensor data (each tensor 32-byte aligned)
    """
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))

        # Metadata KV pairs
        for key, vtype, value in metadata:
            _write_string(f, key)
            f.write(struct.pack("<I", vtype))
            _write_value(f, vtype, value)

        # Tensor info
        data_offset = 0
        tensor_infos = []
        for name, data, ggml_type in tensors:
            _write_string(f, name)
            ndims = len(data.shape)
            f.write(struct.pack("<I", ndims))
            for dim in data.shape:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", data_offset))

            # Calculate data size with alignment
            data_size = data.nbytes
            tensor_infos.append((data, data_size))
            # Next tensor offset (aligned to 32 bytes)
            data_offset += data_size
            padding = (32 - data_offset % 32) % 32
            data_offset += padding

        # Pad to 32-byte alignment before tensor data
        current_pos = f.tell()
        padding = (32 - current_pos % 32) % 32
        f.write(b"\x00" * padding)

        # Tensor data
        for data, data_size in tensor_infos:
            f.write(data.tobytes())
            # Pad to 32-byte alignment
            padding = (32 - data_size % 32) % 32
            f.write(b"\x00" * padding)


def _write_string(f, s: str) -> None:
    """Write a GGUF string (length-prefixed UTF-8)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_value(f, vtype: int, value) -> None:
    """Write a typed GGUF metadata value."""
    if vtype == GGUFValueType.UINT8:
        f.write(struct.pack("<B", value))
    elif vtype == GGUFValueType.INT8:
        f.write(struct.pack("<b", value))
    elif vtype == GGUFValueType.UINT16:
        f.write(struct.pack("<H", value))
    elif vtype == GGUFValueType.INT16:
        f.write(struct.pack("<h", value))
    elif vtype == GGUFValueType.UINT32:
        f.write(struct.pack("<I", value))
    elif vtype == GGUFValueType.INT32:
        f.write(struct.pack("<i", value))
    elif vtype == GGUFValueType.FLOAT32:
        f.write(struct.pack("<f", value))
    elif vtype == GGUFValueType.BOOL:
        f.write(struct.pack("<B", 1 if value else 0))
    elif vtype == GGUFValueType.STRING:
        _write_string(f, value)
    elif vtype == GGUFValueType.ARRAY:
        # Array of strings
        f.write(struct.pack("<I", GGUFValueType.STRING))
        f.write(struct.pack("<Q", len(value)))
        for item in value:
            _write_string(f, item)
    elif vtype == GGUFValueType.UINT64:
        f.write(struct.pack("<Q", value))
    elif vtype == GGUFValueType.INT64:
        f.write(struct.pack("<q", value))
    elif vtype == GGUFValueType.FLOAT64:
        f.write(struct.pack("<d", value))
