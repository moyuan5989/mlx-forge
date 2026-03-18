"""Block quantization for GGUF export — Q4_0 and Q8_0 formats.

Implements block quantization in pure Python/NumPy. For advanced k-quants
(Q4_K_M, Q5_K_S, Q6_K), users should post-process with llama-quantize.
"""

from __future__ import annotations

import struct

import numpy as np

from mlx_forge.export.gguf_constants import GGMLType

# Block size for Q4_0 and Q8_0
BLOCK_SIZE = 32


def quantize_tensor_q8_0(data: np.ndarray) -> bytes:
    """Quantize float tensor to Q8_0 block format.

    Q8_0 layout per block (34 bytes):
    - 1 × fp16 scale (2 bytes)
    - 32 × int8 values (32 bytes)

    Args:
        data: Float tensor (flattened)

    Returns:
        Raw bytes for GGUF tensor data section.
    """
    data = data.astype(np.float32).ravel()

    # Pad to multiple of BLOCK_SIZE
    remainder = len(data) % BLOCK_SIZE
    if remainder:
        data = np.pad(data, (0, BLOCK_SIZE - remainder))

    n_blocks = len(data) // BLOCK_SIZE
    blocks = data.reshape(n_blocks, BLOCK_SIZE)

    result = bytearray()
    for block in blocks:
        # Scale = max(abs(block)) / 127
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax != 0 else 0.0

        # Quantize
        if scale != 0:
            quantized = np.round(block / scale).astype(np.int8)
        else:
            quantized = np.zeros(BLOCK_SIZE, dtype=np.int8)

        # Write: fp16 scale + int8 values
        result.extend(struct.pack("<e", np.float16(scale)))
        result.extend(quantized.tobytes())

    return bytes(result)


def quantize_tensor_q4_0(data: np.ndarray) -> bytes:
    """Quantize float tensor to Q4_0 block format.

    Q4_0 layout per block (18 bytes):
    - 1 × fp16 scale (2 bytes)
    - 16 bytes of packed 4-bit values (32 × 4-bit, low nibble first)

    Args:
        data: Float tensor (flattened)

    Returns:
        Raw bytes for GGUF tensor data section.
    """
    data = data.astype(np.float32).ravel()

    # Pad to multiple of BLOCK_SIZE
    remainder = len(data) % BLOCK_SIZE
    if remainder:
        data = np.pad(data, (0, BLOCK_SIZE - remainder))

    n_blocks = len(data) // BLOCK_SIZE
    blocks = data.reshape(n_blocks, BLOCK_SIZE)

    result = bytearray()
    for block in blocks:
        # Scale = max(abs(block)) / 7
        amax = np.max(np.abs(block))
        scale = amax / 7.0 if amax != 0 else 0.0

        # Quantize to [-8, 7] range then offset to [0, 15]
        if scale != 0:
            quantized = np.round(block / scale).astype(np.int32)
            quantized = np.clip(quantized, -8, 7)
            quantized = (quantized + 8).astype(np.uint8)  # offset to 0-15
        else:
            quantized = np.full(BLOCK_SIZE, 8, dtype=np.uint8)  # zero point

        # Pack two 4-bit values per byte (low nibble first)
        packed = np.zeros(BLOCK_SIZE // 2, dtype=np.uint8)
        for j in range(BLOCK_SIZE // 2):
            lo = quantized[2 * j] & 0x0F
            hi = quantized[2 * j + 1] & 0x0F
            packed[j] = lo | (hi << 4)

        # Write: fp16 scale + packed data
        result.extend(struct.pack("<e", np.float16(scale)))
        result.extend(packed.tobytes())

    return bytes(result)


def dequantize_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q8_0 block data back to float32 (for testing/verification)."""
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    result = np.zeros(n_blocks * BLOCK_SIZE, dtype=np.float32)

    offset = 0
    for i in range(n_blocks):
        scale = struct.unpack("<e", data[offset:offset + 2])[0]
        offset += 2
        values = np.frombuffer(data[offset:offset + BLOCK_SIZE], dtype=np.int8)
        offset += BLOCK_SIZE
        result[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = values.astype(np.float32) * scale

    return result[:n_elements]


def dequantize_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    """Dequantize Q4_0 block data back to float32 (for testing/verification)."""
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    result = np.zeros(n_blocks * BLOCK_SIZE, dtype=np.float32)

    offset = 0
    for i in range(n_blocks):
        scale = struct.unpack("<e", data[offset:offset + 2])[0]
        offset += 2
        packed = np.frombuffer(data[offset:offset + BLOCK_SIZE // 2], dtype=np.uint8)
        offset += BLOCK_SIZE // 2

        # Unpack
        values = np.zeros(BLOCK_SIZE, dtype=np.float32)
        for j in range(BLOCK_SIZE // 2):
            lo = (packed[j] & 0x0F).astype(np.int32) - 8
            hi = ((packed[j] >> 4) & 0x0F).astype(np.int32) - 8
            values[2 * j] = lo * scale
            values[2 * j + 1] = hi * scale

        result[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE] = values

    return result[:n_elements]


# Quantization dispatch table
QUANTIZATION_TYPES = {
    "f16": (GGMLType.F16, None),
    "f32": (GGMLType.F32, None),
    "q8_0": (GGMLType.Q8_0, quantize_tensor_q8_0),
    "q4_0": (GGMLType.Q4_0, quantize_tensor_q4_0),
}

# Bytes per block for quantized types
GGML_BLOCK_SIZE = {
    GGMLType.Q4_0: BLOCK_SIZE,
    GGMLType.Q8_0: BLOCK_SIZE,
}

GGML_TYPE_BYTES_PER_BLOCK = {
    GGMLType.Q4_0: 18,   # 2 (scale) + 16 (packed data)
    GGMLType.Q8_0: 34,   # 2 (scale) + 32 (int8 data)
}
