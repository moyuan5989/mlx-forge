"""Tests for per-token label construction across all formats."""

from __future__ import annotations

import pytest


class MockTokenizer:
    """Mock tokenizer that simulates apply_chat_template behavior."""

    def __init__(self):
        self.eos_token_id = 2
        self.chat_template = "mock"
        self._vocab = {"<s>": 0, "</s>": 2, "hello": 10, "world": 11}

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=True):
        # Simple mock: each word becomes a token ID based on length
        words = text.split()
        tokens = []
        if add_special_tokens:
            tokens.append(0)  # BOS
        for w in words:
            tokens.append(hash(w) % 1000 + 100)
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False
    ):
        """Mock chat template: each message becomes a few tokens.

        Format: [BOS] [ROLE_TOKEN] [content_tokens...] [END_TURN] ...
        Role tokens: user=50, assistant=51, system=52
        End turn: 99
        """
        role_map = {"user": 50, "assistant": 51, "system": 52}
        tokens = [0]  # BOS

        for msg in messages:
            role_token = role_map.get(msg["role"], 50)
            tokens.append(role_token)
            # Tokenize content: each char becomes a token for simplicity
            for ch in msg["content"]:
                tokens.append(ord(ch))
            tokens.append(99)  # end turn

        if add_generation_prompt:
            tokens.append(51)  # assistant role token (prompt to generate)

        return tokens


class TestChatLabels:
    """Test per-token labels for chat format."""

    def test_single_turn_masks_user(self):
        """Single user+assistant: user tokens masked, assistant tokens have labels."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "chat", mask_prompt=True)

        assert len(result) == 1
        item = result[0]
        assert "input_ids" in item
        assert "labels" in item
        assert len(item["input_ids"]) == len(item["labels"])

        # Check that some labels are -100 (user tokens) and some are real (assistant tokens)
        labels = item["labels"]
        assert any(l == -100 for l in labels), "Expected some masked tokens"
        assert any(l != -100 for l in labels), "Expected some trainable tokens"

    def test_single_turn_no_mask(self):
        """With mask_prompt=False, all tokens should have real labels."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "chat", mask_prompt=False)

        item = result[0]
        labels = item["labels"]
        # No masking: labels == input_ids
        assert item["labels"] == item["input_ids"]

    def test_multi_turn_all_assistant_turns_trained(self):
        """Multi-turn chat: ALL assistant turns should have real labels."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "messages": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                    {"role": "user", "content": "C"},
                    {"role": "assistant", "content": "D"},
                ]
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "chat", mask_prompt=True)

        item = result[0]
        labels = item["labels"]

        # Count trainable tokens (non -100)
        trainable = [l for l in labels if l != -100]

        # Both assistant turns ("B" and "D") should contribute trainable tokens
        # With mock tokenizer, each content char is 1 token + end_turn token
        # So we should have tokens for both "B" and "D" as trainable
        assert len(trainable) >= 2, (
            f"Expected at least 2 trainable tokens for 2 assistant turns, got {len(trainable)}"
        )

    def test_system_message_masked(self):
        """System message tokens should be masked with -100."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "messages": [
                    {"role": "system", "content": "X"},
                    {"role": "user", "content": "Y"},
                    {"role": "assistant", "content": "Z"},
                ]
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "chat", mask_prompt=True)

        item = result[0]
        # System and user tokens should be masked, only assistant tokens trainable
        labels = item["labels"]
        assert any(l != -100 for l in labels), "Assistant tokens should be trainable"

    def test_labels_same_length_as_input_ids(self):
        """Labels should always be same length as input_ids."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "messages": [
                    {"role": "user", "content": "Hello world"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you"},
                    {"role": "assistant", "content": "Good thanks"},
                ]
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "chat", mask_prompt=True)

        item = result[0]
        assert len(item["input_ids"]) == len(item["labels"])

    def test_truncation(self):
        """Sequences longer than max_seq_length should be truncated."""
        from lmforge.data.preprocessing import tokenize_dataset

        # Create a sample with many tokens
        long_content = "a" * 100
        samples = [
            {
                "messages": [
                    {"role": "user", "content": long_content},
                    {"role": "assistant", "content": long_content},
                ]
            }
        ]

        tokenizer = MockTokenizer()
        max_len = 50
        result = tokenize_dataset(
            samples, tokenizer, "chat", max_seq_length=max_len
        )

        item = result[0]
        assert len(item["input_ids"]) <= max_len
        assert len(item["labels"]) <= max_len
        assert len(item["input_ids"]) == len(item["labels"])


class TestCompletionsLabels:
    """Test per-token labels for completions format."""

    def test_prompt_masked_completion_trained(self):
        """Prompt tokens masked, completion tokens trained."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [{"prompt": "Question", "completion": "Answer"}]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(
            samples, tokenizer, "completions", mask_prompt=True
        )

        item = result[0]
        assert "input_ids" in item
        assert "labels" in item
        assert len(item["input_ids"]) == len(item["labels"])

        labels = item["labels"]
        assert any(l == -100 for l in labels), "Prompt tokens should be masked"
        assert any(l != -100 for l in labels), "Completion tokens should be trainable"

    def test_completions_no_mask(self):
        """With mask_prompt=False, all tokens should be trained."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [{"prompt": "Q", "completion": "A"}]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(
            samples, tokenizer, "completions", mask_prompt=False
        )

        item = result[0]
        assert item["labels"] == item["input_ids"]

    def test_completions_output_format(self):
        """Completions should produce input_ids and labels keys."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [{"prompt": "Hello", "completion": "World"}]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "completions")

        assert len(result) == 1
        assert set(result[0].keys()) == {"input_ids", "labels"}


class TestTextLabels:
    """Test per-token labels for text format."""

    def test_text_all_trained(self):
        """Text format: labels == input_ids (train on everything)."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [{"text": "Hello world foo bar"}]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "text")

        item = result[0]
        assert item["input_ids"] == item["labels"]
        assert len(item["input_ids"]) > 0

    def test_text_eos_appended(self):
        """Text format should append EOS if not already present."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [{"text": "Hello"}]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "text")

        item = result[0]
        # Last token should be EOS
        assert item["input_ids"][-1] == tokenizer.eos_token_id

    def test_text_truncation(self):
        """Text format respects max_seq_length."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [{"text": " ".join(["word"] * 200)}]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "text", max_seq_length=32)

        item = result[0]
        assert len(item["input_ids"]) <= 32
        assert len(item["labels"]) <= 32

    def test_text_output_format(self):
        """Text should produce input_ids and labels keys."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [{"text": "Sample"}]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(samples, tokenizer, "text")

        assert len(result) == 1
        assert set(result[0].keys()) == {"input_ids", "labels"}


class TestPreferenceLabels:
    """Test per-token labels for preference (DPO) format."""

    def test_preference_output_keys(self):
        """Preference format should produce 4 keys."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "chosen": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Good"},
                ],
                "rejected": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Bad"},
                ],
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(
            samples, tokenizer, "preference", mask_prompt=True
        )

        assert len(result) == 1
        item = result[0]
        assert "chosen_input_ids" in item
        assert "chosen_labels" in item
        assert "rejected_input_ids" in item
        assert "rejected_labels" in item

    def test_preference_labels_mask_prompt(self):
        """Preference: user tokens masked, assistant tokens trained."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "chosen": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Good"},
                ],
                "rejected": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Bad"},
                ],
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(
            samples, tokenizer, "preference", mask_prompt=True
        )

        item = result[0]

        # Chosen labels should have some masked and some trainable
        chosen_labels = item["chosen_labels"]
        assert any(l == -100 for l in chosen_labels)
        assert any(l != -100 for l in chosen_labels)

        # Rejected labels should have some masked and some trainable
        rejected_labels = item["rejected_labels"]
        assert any(l == -100 for l in rejected_labels)
        assert any(l != -100 for l in rejected_labels)

    def test_preference_lengths_consistent(self):
        """Chosen/rejected input_ids and labels should have matching lengths."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "chosen": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "World"},
                ],
                "rejected": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Bad response"},
                ],
            }
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(
            samples, tokenizer, "preference", mask_prompt=True
        )

        item = result[0]
        assert len(item["chosen_input_ids"]) == len(item["chosen_labels"])
        assert len(item["rejected_input_ids"]) == len(item["rejected_labels"])

    def test_preference_multiple_samples(self):
        """Multiple preference samples should all be tokenized."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {
                "chosen": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "B"},
                ],
                "rejected": [
                    {"role": "user", "content": "A"},
                    {"role": "assistant", "content": "C"},
                ],
            },
            {
                "chosen": [
                    {"role": "user", "content": "X"},
                    {"role": "assistant", "content": "Y"},
                ],
                "rejected": [
                    {"role": "user", "content": "X"},
                    {"role": "assistant", "content": "Z"},
                ],
            },
        ]

        tokenizer = MockTokenizer()
        result = tokenize_dataset(
            samples, tokenizer, "preference", mask_prompt=True
        )

        assert len(result) == 2
        for item in result:
            assert "chosen_input_ids" in item
            assert "rejected_input_ids" in item


class TestUnknownFormat:
    """Test error handling for unknown formats."""

    def test_unknown_format_raises(self):
        """Unknown format should raise ValueError."""
        from lmforge.data.preprocessing import tokenize_dataset

        with pytest.raises(ValueError, match="Unknown format"):
            tokenize_dataset([{"text": "hi"}], MockTokenizer(), "unknown_format")


class TestTokenizeDataset:
    """Test the main tokenize_dataset function."""

    def test_empty_input(self):
        """Empty sample list should return empty result."""
        from lmforge.data.preprocessing import tokenize_dataset

        result = tokenize_dataset([], MockTokenizer(), "text")
        assert result == []

    def test_multiple_samples(self):
        """Multiple samples should all be tokenized."""
        from lmforge.data.preprocessing import tokenize_dataset

        samples = [
            {"text": "Hello"},
            {"text": "World"},
            {"text": "Foo"},
        ]

        result = tokenize_dataset(samples, MockTokenizer(), "text")
        assert len(result) == 3
        for item in result:
            assert "input_ids" in item
            assert "labels" in item
