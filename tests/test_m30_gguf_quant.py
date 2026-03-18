"""Tests for M30: GGUF quantization — Q4_0 and Q8_0 block quantization.

Tests cover:
- Roundtrip accuracy (quantize -> dequantize)
- Output byte sizes
- Block alignment / padding
- Zero tensors
- Scale computation
- QUANTIZATION_TYPES dispatch table
- CLI flags
- Large and negative tensors
- Dequantize function availability
- Block size constants
"""

from __future__ import annotations

import struct

import numpy as np

# ── Q8_0 Tests ──────────────────────────────────────────────────────────────

class TestQ8_0:
    def test_q8_0_roundtrip_accuracy(self):
        """Quantize -> dequantize within +/- 1/127 tolerance."""
        from mlx_forge.export.quantize import dequantize_q8_0, quantize_tensor_q8_0

        data = np.random.randn(64).astype(np.float32)
        encoded = quantize_tensor_q8_0(data)
        decoded = dequantize_q8_0(encoded, 64)
        # Tolerance: scale * 1 where scale = max(abs) / 127
        max_abs = np.max(np.abs(data))
        tol = max_abs / 127.0 + 1e-6
        np.testing.assert_allclose(decoded, data, atol=tol)

    def test_q8_0_output_size(self):
        """Correct output size: 34 bytes per block of 32 elements."""
        from mlx_forge.export.quantize import quantize_tensor_q8_0

        data = np.random.randn(64).astype(np.float32)
        encoded = quantize_tensor_q8_0(data)
        # 64 elements = 2 blocks, 34 bytes each
        assert len(encoded) == 2 * 34

    def test_q8_0_block_alignment(self):
        """Non-32-multiple sizes get padded correctly."""
        from mlx_forge.export.quantize import quantize_tensor_q8_0

        data = np.random.randn(40).astype(np.float32)  # 40 -> padded to 64
        encoded = quantize_tensor_q8_0(data)
        # 40 padded to 64 = 2 blocks
        assert len(encoded) == 2 * 34

    def test_q8_0_zeros(self):
        """All-zero tensor quantizes correctly."""
        from mlx_forge.export.quantize import dequantize_q8_0, quantize_tensor_q8_0

        data = np.zeros(32, dtype=np.float32)
        encoded = quantize_tensor_q8_0(data)
        decoded = dequantize_q8_0(encoded, 32)
        np.testing.assert_allclose(decoded, data, atol=1e-7)

    def test_q8_0_scale_computation(self):
        """Scale = max(abs) / 127."""
        from mlx_forge.export.quantize import quantize_tensor_q8_0

        data = np.array([1.27] * 32, dtype=np.float32)  # max_abs = 1.27
        encoded = quantize_tensor_q8_0(data)
        # First 2 bytes are fp16 scale
        scale = struct.unpack("<e", encoded[:2])[0]
        expected_scale = 1.27 / 127.0
        assert abs(scale - expected_scale) < 0.01

    def test_q8_0_large_tensor(self):
        """Works on 1024-element tensor."""
        from mlx_forge.export.quantize import dequantize_q8_0, quantize_tensor_q8_0

        data = np.random.randn(1024).astype(np.float32)
        encoded = quantize_tensor_q8_0(data)
        decoded = dequantize_q8_0(encoded, 1024)
        assert decoded.shape == (1024,)
        max_abs = np.max(np.abs(data))
        tol = max_abs / 127.0 + 1e-6
        np.testing.assert_allclose(decoded, data, atol=tol)

    def test_q8_0_negative_values(self):
        """Handles negative values correctly."""
        from mlx_forge.export.quantize import dequantize_q8_0, quantize_tensor_q8_0

        data = np.array([-1.0] * 32, dtype=np.float32)
        encoded = quantize_tensor_q8_0(data)
        decoded = dequantize_q8_0(encoded, 32)
        np.testing.assert_allclose(decoded, data, atol=1.0 / 127.0 + 1e-6)


# ── Q4_0 Tests ──────────────────────────────────────────────────────────────

class TestQ4_0:
    def test_q4_0_roundtrip_accuracy(self):
        """Quantize -> dequantize within +/- 1/7 tolerance."""
        from mlx_forge.export.quantize import dequantize_q4_0, quantize_tensor_q4_0

        data = np.random.randn(64).astype(np.float32)
        encoded = quantize_tensor_q4_0(data)
        decoded = dequantize_q4_0(encoded, 64)
        max_abs = np.max(np.abs(data))
        tol = max_abs / 7.0 + 0.1  # Q4 is lower precision
        np.testing.assert_allclose(decoded, data, atol=tol)

    def test_q4_0_output_size(self):
        """Correct output size: 18 bytes per block of 32 elements."""
        from mlx_forge.export.quantize import quantize_tensor_q4_0

        data = np.random.randn(64).astype(np.float32)
        encoded = quantize_tensor_q4_0(data)
        # 64 elements = 2 blocks, 18 bytes each
        assert len(encoded) == 2 * 18

    def test_q4_0_block_alignment(self):
        """Non-32-multiple sizes get padded correctly."""
        from mlx_forge.export.quantize import quantize_tensor_q4_0

        data = np.random.randn(40).astype(np.float32)  # padded to 64
        encoded = quantize_tensor_q4_0(data)
        assert len(encoded) == 2 * 18

    def test_q4_0_zeros(self):
        """All-zero tensor quantizes correctly."""
        from mlx_forge.export.quantize import dequantize_q4_0, quantize_tensor_q4_0

        data = np.zeros(32, dtype=np.float32)
        encoded = quantize_tensor_q4_0(data)
        decoded = dequantize_q4_0(encoded, 32)
        np.testing.assert_allclose(decoded, data, atol=1e-7)

    def test_q4_0_scale_computation(self):
        """Scale = max(abs) / 7."""
        from mlx_forge.export.quantize import quantize_tensor_q4_0

        data = np.array([0.7] * 32, dtype=np.float32)
        encoded = quantize_tensor_q4_0(data)
        scale = struct.unpack("<e", encoded[:2])[0]
        expected_scale = 0.7 / 7.0
        assert abs(scale - expected_scale) < 0.02

    def test_q4_0_large_tensor(self):
        """Works on 1024-element tensor."""
        from mlx_forge.export.quantize import dequantize_q4_0, quantize_tensor_q4_0

        data = np.random.randn(1024).astype(np.float32)
        encoded = quantize_tensor_q4_0(data)
        decoded = dequantize_q4_0(encoded, 1024)
        assert decoded.shape == (1024,)

    def test_q4_0_negative_values(self):
        """Handles negative values correctly."""
        from mlx_forge.export.quantize import dequantize_q4_0, quantize_tensor_q4_0

        data = np.array([-0.5] * 32, dtype=np.float32)
        encoded = quantize_tensor_q4_0(data)
        decoded = dequantize_q4_0(encoded, 32)
        # Q4 has limited range, should be roughly correct
        np.testing.assert_allclose(decoded, data, atol=0.5 / 7.0 + 0.1)


# ── Dispatch Table Tests ────────────────────────────────────────────────────

class TestQuantizationTypes:
    def test_quantization_types_dict(self):
        from mlx_forge.export.quantize import QUANTIZATION_TYPES
        assert "f16" in QUANTIZATION_TYPES
        assert "f32" in QUANTIZATION_TYPES
        assert "q8_0" in QUANTIZATION_TYPES
        assert "q4_0" in QUANTIZATION_TYPES

    def test_quantization_f16_no_fn(self):
        from mlx_forge.export.quantize import QUANTIZATION_TYPES
        _, fn = QUANTIZATION_TYPES["f16"]
        assert fn is None

    def test_quantization_f32_no_fn(self):
        from mlx_forge.export.quantize import QUANTIZATION_TYPES
        _, fn = QUANTIZATION_TYPES["f32"]
        assert fn is None

    def test_quantization_q8_0_has_fn(self):
        from mlx_forge.export.quantize import QUANTIZATION_TYPES
        _, fn = QUANTIZATION_TYPES["q8_0"]
        assert fn is not None and callable(fn)

    def test_quantization_q4_0_has_fn(self):
        from mlx_forge.export.quantize import QUANTIZATION_TYPES
        _, fn = QUANTIZATION_TYPES["q4_0"]
        assert fn is not None and callable(fn)


# ── CLI Tests ───────────────────────────────────────────────────────────────

class TestCLIQuantize:
    def test_cli_quantize_flag(self):
        """build_parser recognizes --quantize."""
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["export", "--run", "test-run", "--quantize", "q8_0"])
        assert args.quantize == "q8_0"

    def test_cli_quantize_choices(self):
        """--quantize accepts q4_0, q8_0, f16, f32."""
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        for choice in ["q4_0", "q8_0", "f16", "f32"]:
            args = parser.parse_args(["export", "--run", "test-run", "--quantize", choice])
            assert args.quantize == choice


# ── GGUF Writer Integration ─────────────────────────────────────────────────

class TestGGUFWriter:
    def test_gguf_writer_quantization_param(self):
        """convert_to_gguf accepts quantization param."""
        import inspect

        from mlx_forge.export.gguf_writer import convert_to_gguf
        sig = inspect.signature(convert_to_gguf)
        assert "quantization" in sig.parameters


# ── Dequantize Function Availability ────────────────────────────────────────

class TestDequantize:
    def test_dequantize_q8_0_exists(self):
        from mlx_forge.export.quantize import dequantize_q8_0
        assert callable(dequantize_q8_0)

    def test_dequantize_q4_0_exists(self):
        from mlx_forge.export.quantize import dequantize_q4_0
        assert callable(dequantize_q4_0)


# ── Block Constants ─────────────────────────────────────────────────────────

class TestBlockConstants:
    def test_ggml_block_constants(self):
        """Block size constants are correct."""
        from mlx_forge.export.gguf_constants import GGMLType
        from mlx_forge.export.quantize import (
            BLOCK_SIZE,
            GGML_TYPE_BYTES_PER_BLOCK,
        )

        assert BLOCK_SIZE == 32
        assert GGML_TYPE_BYTES_PER_BLOCK[GGMLType.Q8_0] == 34
        assert GGML_TYPE_BYTES_PER_BLOCK[GGMLType.Q4_0] == 18
