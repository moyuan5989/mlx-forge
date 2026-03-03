"""Tests for V2 memory profiler and auto-configuration."""

from __future__ import annotations

import pytest

from lmforge.models.memory import (
    HardwareProfile,
    MemoryEstimate,
    ModelProfile,
    MODEL_PROFILES,
    auto_configure,
    estimate_memory,
    get_compatible_models,
)


class TestHardwareProfile:
    """Test hardware detection."""

    def test_detect_returns_profile(self):
        """HardwareProfile.detect() returns valid profile."""
        hw = HardwareProfile.detect()
        assert hw.total_memory_gb > 0
        assert hw.training_budget_gb > 0
        assert hw.training_budget_gb < hw.total_memory_gb

    def test_training_budget_is_75_percent(self):
        """Training budget is ~75% of total memory."""
        hw = HardwareProfile(total_memory_gb=32.0, chip_name="Test")
        hw.training_budget_gb = hw.total_memory_gb * 0.75
        assert abs(hw.training_budget_gb - 24.0) < 0.1


class TestModelProfiles:
    """Test model profile database."""

    def test_known_models_exist(self):
        """Key models have profiles."""
        expected = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-4B",
            "meta-llama/Llama-3.1-8B",
            "google/gemma-2-2b",
            "microsoft/phi-3-mini-4k-instruct",
        ]
        for model_id in expected:
            assert model_id in MODEL_PROFILES, f"Missing profile for {model_id}"

    def test_profiles_have_required_fields(self):
        """All profiles have required fields."""
        for model_id, profile in MODEL_PROFILES.items():
            assert profile.num_params > 0, f"{model_id} missing num_params"
            assert profile.hidden_dim > 0, f"{model_id} missing hidden_dim"
            assert profile.num_layers > 0, f"{model_id} missing num_layers"

    def test_v2_models_in_profiles(self):
        """V2 models (Qwen2.5, DeepSeek-R1, Phi-4) have profiles."""
        v2_models = [
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "microsoft/Phi-4-mini-instruct",
        ]
        for model_id in v2_models:
            assert model_id in MODEL_PROFILES, f"Missing V2 profile for {model_id}"


class TestMemoryEstimation:
    """Test memory estimation accuracy."""

    def test_estimate_known_model(self):
        """Estimation for known model returns valid result."""
        hw = HardwareProfile(total_memory_gb=36.0, training_budget_gb=27.0)
        est = estimate_memory("Qwen/Qwen3-0.6B", hardware=hw)

        assert est.base_weights_gb > 0
        assert est.total_gb > 0
        assert est.fits is True  # 0.6B should fit on 36GB

    def test_estimate_qlora_reduces_memory(self):
        """QLoRA reduces base weights memory significantly."""
        hw = HardwareProfile(total_memory_gb=36.0, training_budget_gb=27.0)

        fp16 = estimate_memory("meta-llama/Llama-3.1-8B", hardware=hw)
        qlora = estimate_memory("meta-llama/Llama-3.1-8B", quantization_bits=4, hardware=hw)

        assert qlora.base_weights_gb < fp16.base_weights_gb
        assert qlora.base_weights_gb < fp16.base_weights_gb * 0.4  # ~4x reduction

    def test_estimate_gradient_checkpointing_reduces_activations(self):
        """Gradient checkpointing reduces peak activation memory."""
        hw = HardwareProfile(total_memory_gb=36.0, training_budget_gb=27.0)

        without = estimate_memory("Qwen/Qwen3-4B", hardware=hw)
        with_ckpt = estimate_memory(
            "Qwen/Qwen3-4B", gradient_checkpointing=True, hardware=hw)

        assert with_ckpt.peak_activations_gb < without.peak_activations_gb

    def test_estimate_unknown_model_raises(self):
        """Unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            estimate_memory("nonexistent/model-99B")

    def test_memory_bar_segments(self):
        """Memory estimate produces valid bar segments."""
        hw = HardwareProfile(total_memory_gb=36.0, training_budget_gb=27.0)
        est = estimate_memory("Qwen/Qwen3-0.6B", hardware=hw)

        segments = est.bar_segments()
        assert len(segments) == 4
        assert all("label" in s and "gb" in s and "color" in s for s in segments)
        total_from_segments = sum(s["gb"] for s in segments)
        assert abs(total_from_segments - est.total_gb) < 0.1

    def test_estimate_batch_size_affects_activations(self):
        """Larger batch size increases activation memory."""
        hw = HardwareProfile(total_memory_gb=64.0, training_budget_gb=48.0)

        small = estimate_memory("Qwen/Qwen3-4B", batch_size=1, hardware=hw)
        large = estimate_memory("Qwen/Qwen3-4B", batch_size=8, hardware=hw)

        assert large.peak_activations_gb > small.peak_activations_gb

    def test_fits_flag(self):
        """fits flag correctly reflects budget comparison."""
        hw_small = HardwareProfile(total_memory_gb=8.0, training_budget_gb=6.0)
        hw_big = HardwareProfile(total_memory_gb=128.0, training_budget_gb=96.0)

        est_small = estimate_memory("meta-llama/Llama-3.1-8B", hardware=hw_small)
        est_big = estimate_memory("meta-llama/Llama-3.1-8B", hardware=hw_big)

        assert est_small.fits is False  # 8B won't fit in 6GB
        assert est_big.fits is True  # 8B will fit in 96GB


class TestCompatibleModels:
    """Test model compatibility listing."""

    def test_returns_models(self):
        """get_compatible_models returns a non-empty list."""
        hw = HardwareProfile(total_memory_gb=36.0, training_budget_gb=27.0)
        models = get_compatible_models(hw)
        assert len(models) > 0

    def test_models_sorted_by_size(self):
        """Models are sorted by parameter count."""
        hw = HardwareProfile(total_memory_gb=64.0, training_budget_gb=48.0)
        models = get_compatible_models(hw)

        sizes = [m["num_params_b"] for m in models]
        assert sizes == sorted(sizes)

    def test_models_have_estimates(self):
        """Each model has fp16 and qlora estimates."""
        hw = HardwareProfile(total_memory_gb=36.0, training_budget_gb=27.0)
        models = get_compatible_models(hw)

        for m in models:
            assert "fp16" in m
            assert "qlora_4bit" in m
            assert "total_gb" in m["fp16"]
            assert "fits" in m["fp16"]

    def test_some_models_recommended(self):
        """At least one model is recommended for 36GB."""
        hw = HardwareProfile(total_memory_gb=36.0, training_budget_gb=27.0)
        models = get_compatible_models(hw)
        recommended = [m for m in models if m["recommended"]]
        assert len(recommended) > 0


class TestAutoConfiguration:
    """Test auto-configuration rules."""

    def test_low_memory_enables_qlora(self):
        """< 16GB triggers QLoRA."""
        overrides = auto_configure("Qwen/Qwen3-4B", system_memory_gb=12)
        assert "model.quantization" in overrides
        assert overrides["model.quantization"]["bits"] == 4

    def test_low_memory_enables_grad_checkpoint(self):
        """< 16GB enables gradient checkpointing."""
        overrides = auto_configure("Qwen/Qwen3-4B", system_memory_gb=12)
        assert overrides.get("training.gradient_checkpointing") is True

    def test_medium_memory_qlora_only(self):
        """16-32GB enables QLoRA but not checkpointing."""
        overrides = auto_configure("Qwen/Qwen3-4B", system_memory_gb=24)
        assert "model.quantization" in overrides
        assert "training.gradient_checkpointing" not in overrides

    def test_high_memory_no_qlora(self):
        """> 32GB doesn't trigger QLoRA."""
        overrides = auto_configure("Qwen/Qwen3-4B", system_memory_gb=64)
        assert "model.quantization" not in overrides

    def test_small_dataset_reduces_iters(self):
        """< 500 samples reduces num_iters."""
        overrides = auto_configure("Qwen/Qwen3-4B", dataset_samples=100)
        assert overrides.get("training.num_iters") == 500

    def test_large_dataset_enables_packing(self):
        """> 10000 samples enables packing."""
        overrides = auto_configure("Qwen/Qwen3-4B", dataset_samples=20000)
        assert overrides.get("data.packing") is True

    def test_normal_dataset_no_changes(self):
        """500-10000 samples doesn't trigger dataset rules."""
        overrides = auto_configure(
            "Qwen/Qwen3-4B", system_memory_gb=64, dataset_samples=5000)
        assert "training.num_iters" not in overrides
        assert "data.packing" not in overrides
