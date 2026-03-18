"""Tests for M26: GRPO Training."""

import mlx.core as mx
import pytest

from mlx_forge.losses.grpo import GRPOLoss, compute_group_advantages
from mlx_forge.trainer.reward import (
    REWARD_FUNCTIONS,
    LengthReward,
    RuleReward,
    get_reward_function,
)


class TestGRPOLoss:
    """Test GRPO loss computation."""

    def test_loss_shape(self):
        loss_fn = GRPOLoss(beta=0.1, clip_range=0.2)
        policy_lp = mx.zeros((2, 10))
        ref_lp = mx.zeros((2, 10))
        advantages = mx.array([1.0, -1.0])
        loss = loss_fn(policy_lp, ref_lp, advantages)
        assert loss.shape == ()

    def test_loss_zero_when_equal(self):
        """When policy == ref and advantages are zero, loss should be near zero."""
        loss_fn = GRPOLoss(beta=0.0, clip_range=0.2)
        lp = mx.zeros((2, 5))
        advantages = mx.array([0.0, 0.0])
        loss = loss_fn(lp, lp, advantages)
        mx.eval(loss)
        assert abs(loss.item()) < 1e-6

    def test_loss_positive_advantage_reduces_loss(self):
        """Positive advantage should encourage the policy action."""
        loss_fn = GRPOLoss(beta=0.0, clip_range=0.2)
        policy_lp = mx.array([[0.0, 0.0]])
        ref_lp = mx.array([[0.0, 0.0]])
        pos_loss = loss_fn(policy_lp, ref_lp, mx.array([1.0]))
        neg_loss = loss_fn(policy_lp, ref_lp, mx.array([-1.0]))
        mx.eval(pos_loss, neg_loss)
        # With ratio=1 (equal policies), loss = -advantage for unclipped
        assert pos_loss.item() < neg_loss.item()

    def test_kl_penalty(self):
        """Non-zero beta should add KL penalty."""
        loss_no_kl = GRPOLoss(beta=0.0, clip_range=0.2)
        loss_with_kl = GRPOLoss(beta=1.0, clip_range=0.2)

        policy_lp = mx.array([[0.1, 0.2]])
        ref_lp = mx.array([[0.0, 0.0]])
        adv = mx.array([0.0])

        l1 = loss_no_kl(policy_lp, ref_lp, adv)
        l2 = loss_with_kl(policy_lp, ref_lp, adv)
        mx.eval(l1, l2)
        # KL penalty should make loss different
        assert l1.item() != l2.item()

    def test_clipping(self):
        """Ratio outside clip range should be clipped."""
        loss_fn = GRPOLoss(beta=0.0, clip_range=0.2)
        # Large difference in log probs -> large ratio
        policy_lp = mx.array([[2.0]])
        ref_lp = mx.array([[0.0]])
        adv = mx.array([1.0])
        loss = loss_fn(policy_lp, ref_lp, adv)
        mx.eval(loss)
        # Should be clipped to 1.2 * advantage
        assert abs(loss.item() + 1.2) < 0.1  # -1.2 (clipped)

    def test_mask_handling(self):
        loss_fn = GRPOLoss(beta=0.0, clip_range=0.2)
        policy_lp = mx.zeros((1, 5))
        ref_lp = mx.zeros((1, 5))
        adv = mx.array([1.0])
        mask = mx.array([[1.0, 1.0, 0.0, 0.0, 0.0]])

        loss_masked = loss_fn(policy_lp, ref_lp, adv, mask=mask)
        mx.eval(loss_masked)
        # Masked tokens shouldn't contribute
        assert loss_masked.shape == ()


class TestGroupAdvantages:
    """Test group advantage normalization."""

    def test_normalized_mean_zero(self):
        rewards = mx.array([1.0, 2.0, 3.0, 4.0])
        advantages = compute_group_advantages(rewards)
        mx.eval(advantages)
        assert abs(mx.mean(advantages).item()) < 1e-5

    def test_normalized_unit_variance(self):
        rewards = mx.array([1.0, 2.0, 3.0, 4.0])
        advantages = compute_group_advantages(rewards)
        mx.eval(advantages)
        var = mx.mean(advantages * advantages).item()
        assert abs(var - 1.0) < 0.2  # Approximate unit variance

    def test_single_reward(self):
        rewards = mx.array([5.0])
        advantages = compute_group_advantages(rewards)
        mx.eval(advantages)
        assert abs(advantages[0].item()) < 1e-3  # Zero advantage for single item

    def test_equal_rewards(self):
        rewards = mx.array([3.0, 3.0, 3.0])
        advantages = compute_group_advantages(rewards)
        mx.eval(advantages)
        for i in range(3):
            assert abs(advantages[i].item()) < 1e-3


class TestRewardFunctions:
    """Test reward functions."""

    def test_length_reward_at_target(self):
        reward = LengthReward(target_length=10, penalty_scale=0.1)
        score = reward("prompt", "x" * 10)
        assert score == 1.0

    def test_length_reward_deviation(self):
        reward = LengthReward(target_length=100, penalty_scale=0.01)
        score = reward("prompt", "x" * 50)
        assert 0.0 < score < 1.0

    def test_length_reward_min_zero(self):
        reward = LengthReward(target_length=10, penalty_scale=1.0)
        score = reward("prompt", "x" * 1000)
        assert score == 0.0

    def test_rule_reward_base_score(self):
        reward = RuleReward()
        score = reward("prompt", "x" * 20)
        assert score > 0.0

    def test_rule_reward_required_keyword(self):
        reward = RuleReward(required_keywords=["hello"])
        score_with = reward("prompt", "hello world and more text here")
        score_without = reward("prompt", "goodbye world and more text here")
        assert score_with > score_without

    def test_rule_reward_forbidden_keyword(self):
        reward = RuleReward(forbidden_keywords=["bad"])
        score_clean = reward("prompt", "good response with enough text")
        score_bad = reward("prompt", "bad response with enough text")
        assert score_clean > score_bad

    def test_get_reward_function(self):
        fn = get_reward_function("length")
        assert isinstance(fn, LengthReward)

    def test_get_reward_function_unknown(self):
        with pytest.raises(ValueError, match="Unknown reward"):
            get_reward_function("nonexistent")

    def test_reward_functions_registry(self):
        assert "length" in REWARD_FUNCTIONS
        assert "rule" in REWARD_FUNCTIONS
        assert "external" in REWARD_FUNCTIONS


class TestGRPOConfig:
    """Test GRPO config integration."""

    def test_training_type_grpo(self):
        from mlx_forge.config import TrainingParams
        params = TrainingParams(
            training_type="grpo",
            grpo_num_generations=4,
            grpo_beta=0.1,
            steps_per_save=100,
        )
        assert params.training_type == "grpo"
        assert params.grpo_num_generations == 4

    def test_grpo_defaults(self):
        from mlx_forge.config import TrainingParams
        params = TrainingParams(training_type="grpo", steps_per_save=100)
        assert params.grpo_num_generations == 4
        assert params.grpo_beta == 0.1
        assert params.grpo_clip_range == 0.2
        assert params.grpo_max_completion_length == 256
        assert params.grpo_reward_function == "length"

    def test_training_type_sft_still_works(self):
        from mlx_forge.config import TrainingParams
        params = TrainingParams(steps_per_save=100)
        assert params.training_type == "sft"
