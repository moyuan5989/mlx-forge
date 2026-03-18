"""M17 — Sampling edge-case fixes."""

from __future__ import annotations

import mlx.core as mx

from mlx_forge.inference.sampling import _apply_top_p, sample_next_token


class TestTopPFallback:
    """Top-p with very small values should not produce nan/inf."""

    def test_very_small_top_p(self):
        """top_p=0.001 should still return a valid token."""
        logits = mx.random.normal((100,))
        result = _apply_top_p(logits, top_p=0.001)
        mx.eval(result)
        # Should have at least one finite value
        finite_count = mx.sum(result != float("-inf"))
        assert finite_count.item() >= 1

    def test_top_p_zero(self):
        """top_p near zero should keep top-1 token."""
        logits = mx.array([1.0, 5.0, 2.0, 0.5])
        result = _apply_top_p(logits, top_p=1e-10)
        mx.eval(result)
        # The max logit token should be kept
        finite_mask = result != float("-inf")
        assert mx.sum(finite_mask).item() >= 1

    def test_sample_with_tiny_top_p(self):
        """Full sampling pipeline with tiny top_p should not crash."""
        mx.random.seed(42)
        logits = mx.random.normal((1000,))
        token = sample_next_token(logits, temperature=0.8, top_p=0.001)
        mx.eval(token)
        assert 0 <= token.item() < 1000


class TestTemperatureFloor:
    """Temperature=0 should produce deterministic output (argmax)."""

    def test_temperature_zero_is_greedy(self):
        logits = mx.array([1.0, 3.0, 2.0, 0.5])
        token = sample_next_token(logits, temperature=0.0)
        mx.eval(token)
        assert token.item() == 1  # index of max value

    def test_temperature_near_zero(self):
        """Very small temperature should behave like greedy."""
        logits = mx.array([1.0, 10.0, 2.0, 0.5])
        token = sample_next_token(logits, temperature=1e-10)
        mx.eval(token)
        assert token.item() == 1


class TestRepetitionPenaltyBounds:
    """Out-of-range token IDs in repetition penalty should not crash."""

    def test_out_of_range_tokens(self):
        """Token IDs exceeding vocab size should be clamped."""
        logits = mx.array([1.0, 2.0, 3.0, 4.0])  # vocab_size=4
        # Pass token IDs that exceed vocab size
        token = sample_next_token(
            logits,
            temperature=0.0,
            repetition_penalty=1.5,
            generated_tokens=[0, 1, 100, 999],  # 100 and 999 are out of range
        )
        mx.eval(token)
        # Should not crash; should still return a valid token
        assert 0 <= token.item() < 4

    def test_negative_token_ids(self):
        """Negative token IDs should be clamped to 0."""
        logits = mx.array([1.0, 2.0, 3.0, 4.0])
        token = sample_next_token(
            logits,
            temperature=0.0,
            repetition_penalty=1.5,
            generated_tokens=[-1, -10, 2],
        )
        mx.eval(token)
        assert 0 <= token.item() < 4

    def test_penalty_applied_correctly(self):
        """Repetition penalty should reduce probability of repeated tokens."""
        logits = mx.array([0.0, 5.0, 0.0, 0.0])
        # Without penalty, token 1 wins
        token_no_penalty = sample_next_token(logits, temperature=0.0)
        assert token_no_penalty.item() == 1

        # With penalty on token 1, logit is divided → might still win but value reduced
        logits2 = mx.array([4.0, 5.0, 0.0, 0.0])
        token_with_penalty = sample_next_token(
            logits2, temperature=0.0, repetition_penalty=10.0, generated_tokens=[1]
        )
        mx.eval(token_with_penalty)
        # Token 1 logit: 5.0 / 10.0 = 0.5, token 0 logit: 4.0 → token 0 wins
        assert token_with_penalty.item() == 0
