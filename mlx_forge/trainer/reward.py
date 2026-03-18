"""Reward functions for GRPO training.

Provides built-in reward functions and a protocol for custom rewards.
"""

from __future__ import annotations

from typing import Protocol


class RewardFunction(Protocol):
    """Protocol for reward functions."""

    def __call__(self, prompt: str, completion: str) -> float:
        """Score a completion given a prompt.

        Returns:
            Scalar reward value (higher is better).
        """
        ...


class LengthReward:
    """Reward based on completion length.

    Encourages generating responses within a target length range.
    """

    def __init__(self, target_length: int = 200, penalty_scale: float = 0.001):
        self.target_length = target_length
        self.penalty_scale = penalty_scale

    def __call__(self, prompt: str, completion: str) -> float:
        length = len(completion)
        deviation = abs(length - self.target_length)
        return max(0.0, 1.0 - self.penalty_scale * deviation)


class RuleReward:
    """Rule-based reward for format/content checking.

    Checks for required keywords, format patterns, etc.
    """

    def __init__(
        self,
        required_keywords: list[str] | None = None,
        forbidden_keywords: list[str] | None = None,
        min_length: int = 10,
    ):
        self.required_keywords = required_keywords or []
        self.forbidden_keywords = forbidden_keywords or []
        self.min_length = min_length

    def __call__(self, prompt: str, completion: str) -> float:
        score = 0.5  # Base score

        # Length check
        if len(completion) >= self.min_length:
            score += 0.2

        # Required keywords
        for kw in self.required_keywords:
            if kw.lower() in completion.lower():
                score += 0.15 / max(len(self.required_keywords), 1)

        # Forbidden keywords
        for kw in self.forbidden_keywords:
            if kw.lower() in completion.lower():
                score -= 0.3

        return max(0.0, min(1.0, score))


class ExternalReward:
    """Reward from an external HTTP API.

    Calls an endpoint with prompt/completion and expects a float score.
    """

    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url
        self.timeout = timeout

    def __call__(self, prompt: str, completion: str) -> float:
        import json
        import urllib.request

        payload = json.dumps({"prompt": prompt, "completion": completion}).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
                return float(result.get("score", 0.0))
        except Exception:
            return 0.0


REWARD_FUNCTIONS = {
    "length": LengthReward,
    "rule": RuleReward,
    "external": ExternalReward,
}


def get_reward_function(name: str, **kwargs) -> RewardFunction:
    """Get a reward function by name."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Available: {list(REWARD_FUNCTIONS.keys())}"
        )
    return REWARD_FUNCTIONS[name](**kwargs)
