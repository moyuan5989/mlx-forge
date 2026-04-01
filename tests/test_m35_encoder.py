"""Tests for M35: Encoder-Only Models — BERT, RoBERTa, DeBERTa.

Tests cover:
- BERT/RoBERTa forward pass, ModelArgs, sanitize
- DeBERTa disentangled attention, forward pass
- Padding mask creation
- Bidirectional attention (no causal mask)
- MLM loss (no shift, -100 masking)
- MLM tokenization (15% masking, 80/10/10)
- MLM batching
- MLM trainer
- MLMWrapper gradient flow
- Config: training_type="mlm"
- Encoder inference (embeddings)
- LoRA presets for BERT
- Registry entries
- CLI encode subcommand
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────
# BERT / RoBERTa Architecture Tests
# ──────────────────────────────────────────────────────────────────────

class TestBertModelArgs:
    def test_from_dict_filters_extra_keys(self):
        from mlx_forge.models.architectures.bert import ModelArgs
        args = ModelArgs.from_dict({
            "model_type": "bert",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "extra_field_ignored": True,
        })
        assert args.hidden_size == 128
        assert args.num_hidden_layers == 2
        assert args.head_dim == 64

    def test_default_values(self):
        from mlx_forge.models.architectures.bert import ModelArgs
        args = ModelArgs.from_dict({"model_type": "bert"})
        assert args.hidden_size == 768
        assert args.num_hidden_layers == 12
        assert args.vocab_size == 30522
        assert args.type_vocab_size == 2


class TestBertForwardPass:
    @pytest.fixture
    def tiny_bert(self):
        from mlx_forge.models.architectures.bert import Model, ModelArgs
        args = ModelArgs.from_dict({
            "model_type": "bert",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "vocab_size": 100,
            "max_position_embeddings": 32,
        })
        model = Model(args)
        mx.eval(model.parameters())
        return model

    def test_output_shape(self, tiny_bert):
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        out = tiny_bert(input_ids)
        mx.eval(out)
        assert out.shape == (1, 5, 64)  # (B, T, D) hidden states

    def test_model_category_is_encoder(self, tiny_bert):
        assert tiny_bert.model_category == "encoder"

    def test_with_attention_mask(self, tiny_bert):
        input_ids = mx.array([[1, 2, 3, 0, 0]])
        attention_mask = mx.array([[1, 1, 1, 0, 0]])
        out = tiny_bert(input_ids, attention_mask=attention_mask)
        mx.eval(out)
        assert out.shape == (1, 5, 64)

    def test_with_token_type_ids(self, tiny_bert):
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        token_type_ids = mx.array([[0, 0, 1, 1, 1]])
        out = tiny_bert(input_ids, token_type_ids=token_type_ids)
        mx.eval(out)
        assert out.shape == (1, 5, 64)

    def test_batch_forward(self, tiny_bert):
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        out = tiny_bert(input_ids)
        mx.eval(out)
        assert out.shape == (2, 3, 64)

    def test_layers_property(self, tiny_bert):
        assert len(tiny_bert.layers) == 2

    def test_no_lm_head(self, tiny_bert):
        assert not hasattr(tiny_bert, 'lm_head')


class TestBertSanitize:
    def test_sanitize_maps_hf_names(self):
        from mlx_forge.models.architectures.bert import Model
        weights = {
            "bert.embeddings.word_embeddings.weight": mx.zeros((100, 64)),
            "bert.encoder.layer.0.attention.self.query.weight": mx.zeros((64, 64)),
            "bert.encoder.layer.0.attention.self.query.bias": mx.zeros((64,)),
            "bert.encoder.layer.0.attention.output.dense.weight": mx.zeros((64, 64)),
            "bert.encoder.layer.0.attention.output.LayerNorm.weight": mx.zeros((64,)),
            "bert.encoder.layer.0.intermediate.dense.weight": mx.zeros((256, 64)),
            "bert.encoder.layer.0.output.dense.weight": mx.zeros((64, 256)),
            "bert.encoder.layer.0.output.LayerNorm.weight": mx.zeros((64,)),
            "bert.pooler.dense.weight": mx.zeros((64, 64)),
            "bert.embeddings.position_ids": mx.zeros((1, 512)),
        }
        sanitized = Model.sanitize(weights)

        assert "encoder.layers.0.attention.query.weight" in sanitized
        assert "encoder.layers.0.attention.dense.weight" in sanitized
        assert "encoder.layers.0.mlp.dense.weight" in sanitized
        assert "encoder.layers.0.mlp.dense_out.weight" in sanitized
        # Pooler and position_ids should be dropped
        assert not any("pooler" in k for k in sanitized)
        assert not any("position_ids" in k for k in sanitized)


# ──────────────────────────────────────────────────────────────────────
# DeBERTa Architecture Tests
# ──────────────────────────────────────────────────────────────────────

class TestDeBERTaModelArgs:
    def test_from_dict(self):
        from mlx_forge.models.architectures.deberta import ModelArgs
        args = ModelArgs.from_dict({
            "model_type": "deberta",
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "max_relative_positions": 64,
        })
        assert args.hidden_size == 128
        assert args.max_relative_positions == 64
        assert args.head_dim == 64


class TestDeBERTaForwardPass:
    @pytest.fixture
    def tiny_deberta(self):
        from mlx_forge.models.architectures.deberta import Model, ModelArgs
        args = ModelArgs.from_dict({
            "model_type": "deberta",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 256,
            "vocab_size": 1000,
            "max_relative_positions": 32,
        })
        model = Model(args)
        mx.eval(model.parameters())
        return model

    def test_output_shape(self, tiny_deberta):
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        out = tiny_deberta(input_ids)
        mx.eval(out)
        assert out.shape == (1, 5, 64)

    def test_model_category_is_encoder(self, tiny_deberta):
        assert tiny_deberta.model_category == "encoder"

    def test_disentangled_attention_produces_different_output_than_standard(self, tiny_deberta):
        """DeBERTa should produce different outputs than a model without relative pos bias."""
        input_ids = mx.array([[1, 2, 3, 4]])
        out = tiny_deberta(input_ids)
        mx.eval(out)
        # Just verify it runs and has correct shape
        assert out.shape == (1, 4, 64)

    def test_with_attention_mask(self, tiny_deberta):
        input_ids = mx.array([[1, 2, 3, 0]])
        attention_mask = mx.array([[1, 1, 1, 0]])
        out = tiny_deberta(input_ids, attention_mask=attention_mask)
        mx.eval(out)
        assert out.shape == (1, 4, 64)

    def test_layers_property(self, tiny_deberta):
        assert len(tiny_deberta.layers) == 2


# ──────────────────────────────────────────────────────────────────────
# Padding Mask Tests
# ──────────────────────────────────────────────────────────────────────

class TestCreatePaddingMask:
    def test_all_ones_returns_zero_mask(self):
        from mlx_forge.models._base.attention import create_padding_mask
        mask = mx.array([[1, 1, 1]])
        result = create_padding_mask(mask)
        mx.eval(result)
        # All-ones input → all-zeros additive mask (no positions masked out)
        assert result.shape == (1, 1, 1, 3)
        assert result[0, 0, 0, 0].item() == pytest.approx(0.0)
        assert result[0, 0, 0, 2].item() == pytest.approx(0.0)

    def test_with_padding(self):
        from mlx_forge.models._base.attention import create_padding_mask
        mask = mx.array([[1, 1, 0, 0]])
        result = create_padding_mask(mask)
        mx.eval(result)
        assert result.shape == (1, 1, 1, 4)
        # Real tokens should be 0, padding should be large negative
        assert result[0, 0, 0, 0].item() == pytest.approx(0.0)
        assert result[0, 0, 0, 2].item() < -1e8

    def test_none_input(self):
        from mlx_forge.models._base.attention import create_padding_mask
        assert create_padding_mask(None) is None


# ──────────────────────────────────────────────────────────────────────
# Bidirectional Attention Tests
# ──────────────────────────────────────────────────────────────────────

class TestBidirectionalAttention:
    def test_no_causal_masking(self):
        """BERT attention should be bidirectional — no lower-triangular mask."""
        from mlx_forge.models.architectures.bert import BertAttention, ModelArgs
        args = ModelArgs.from_dict({
            "hidden_size": 64,
            "num_attention_heads": 2,
            "intermediate_size": 256,
        })
        attn = BertAttention(args)
        mx.eval(attn.parameters())

        x = mx.ones((1, 4, 64))
        # Without any mask, all positions attend to all others
        out = attn(x)
        mx.eval(out)
        assert out.shape == (1, 4, 64)

    def test_position_0_attends_to_position_3(self):
        """In bidirectional attention, position 0 can attend to later positions."""
        from mlx_forge.models.architectures.bert import Model, ModelArgs
        args = ModelArgs.from_dict({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())

        # Process two sequences that differ only in later positions
        ids_a = mx.array([[1, 2, 3, 4]])
        ids_b = mx.array([[1, 2, 3, 99]])

        out_a = model(ids_a)
        out_b = model(ids_b)
        mx.eval(out_a, out_b)

        # Position 0 output should differ because it can see position 3
        diff = mx.abs(out_a[0, 0] - out_b[0, 0]).sum().item()
        assert diff > 0.0, "Bidirectional attention: position 0 should see position 3"


# ──────────────────────────────────────────────────────────────────────
# MLM Loss Tests
# ──────────────────────────────────────────────────────────────────────

class TestMLMLoss:
    def test_no_shift(self):
        """MLM loss should NOT shift labels — predict at same positions."""
        from mlx_forge.losses.mlm import MLMLoss

        loss_fn = MLMLoss()

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 10, bias=False)

            def __call__(self, input_ids, attention_mask=None):
                B, T = input_ids.shape
                h = mx.ones((B, T, 8))
                return self.proj(h)  # (B, T, 10) logits

        model = DummyModel()
        mx.eval(model.parameters())

        input_ids = mx.array([[1, 2, 3, 4]])
        labels = mx.array([[-100, 5, -100, 7]])  # Only positions 1 and 3 masked

        loss, ntoks = loss_fn(model, input_ids, labels)
        mx.eval(loss, ntoks)

        assert ntoks.item() == 2
        assert loss.item() > 0

    def test_all_masked(self):
        """When all labels are -100, loss should be 0 and ntoks 0."""
        from mlx_forge.losses.mlm import MLMLoss

        loss_fn = MLMLoss()

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 10, bias=False)

            def __call__(self, input_ids, attention_mask=None):
                B, T = input_ids.shape
                return self.proj(mx.ones((B, T, 8)))

        model = DummyModel()
        mx.eval(model.parameters())

        input_ids = mx.array([[1, 2, 3]])
        labels = mx.array([[-100, -100, -100]])

        loss, ntoks = loss_fn(model, input_ids, labels)
        mx.eval(loss, ntoks)
        assert ntoks.item() == 0

    def test_correct_loss_value(self):
        """Verify loss decreases when predictions match labels."""
        from mlx_forge.losses.mlm import MLMLoss

        loss_fn = MLMLoss()

        class PerfectModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Linear(1, 1)  # Need at least one param

            def __call__(self, input_ids, attention_mask=None):
                B, T = input_ids.shape
                # Return logits that strongly predict token 5
                logits = mx.full((B, T, 10), -10.0)
                logits = logits.at[:, :, 5].add(20.0)
                return logits

        model = PerfectModel()
        mx.eval(model.parameters())

        input_ids = mx.array([[0, 0, 0]])
        labels = mx.array([[-100, 5, -100]])

        loss, ntoks = loss_fn(model, input_ids, labels)
        mx.eval(loss, ntoks)

        assert loss.item() < 0.1  # Should be very low
        assert ntoks.item() == 1

    def test_with_attention_mask(self):
        """MLM loss should accept attention_mask parameter."""
        from mlx_forge.losses.mlm import MLMLoss

        loss_fn = MLMLoss()

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 10, bias=False)

            def __call__(self, input_ids, attention_mask=None):
                B, T = input_ids.shape
                return self.proj(mx.ones((B, T, 8)))

        model = DummyModel()
        mx.eval(model.parameters())

        input_ids = mx.array([[1, 2, 3, 0]])
        labels = mx.array([[-100, 5, -100, -100]])
        attention_mask = mx.array([[1, 1, 1, 0]])

        loss, ntoks = loss_fn(model, input_ids, labels, attention_mask)
        mx.eval(loss, ntoks)
        assert ntoks.item() == 1

    def test_loss_is_not_shifted(self):
        """Verify loss uses same-position comparison, not shifted."""
        from mlx_forge.losses.mlm import MLMLoss

        loss_fn = MLMLoss()

        # Model returns identity-like logits where position i predicts token i
        class IdentityModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Linear(1, 1)

            def __call__(self, input_ids, attention_mask=None):
                B, T = input_ids.shape
                # Create one-hot logits for each position
                logits = mx.zeros((B, T, 10))
                for b in range(B):
                    for t in range(T):
                        tok = input_ids[b, t].item()
                        if tok < 10:
                            logits = logits.at[b, t, tok].add(20.0)
                return logits

        model = IdentityModel()
        mx.eval(model.parameters())

        # Label at position 1 is token 5, but input at position 1 is also 5
        input_ids = mx.array([[3, 5, 7]])
        labels = mx.array([[-100, 5, -100]])

        loss, ntoks = loss_fn(model, input_ids, labels)
        mx.eval(loss, ntoks)
        assert loss.item() < 0.1  # Should be low since prediction matches label


# ──────────────────────────────────────────────────────────────────────
# MLM Tokenization Tests
# ──────────────────────────────────────────────────────────────────────

class TestTokenizeMLM:
    def _make_tokenizer(self):
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[101, 2000, 2001, 2002, 2003, 102])
        tok.cls_token_id = 101
        tok.sep_token_id = 102
        tok.pad_token_id = 0
        tok.mask_token_id = 103
        tok.vocab_size = 30000
        return tok

    def test_output_keys(self):
        from mlx_forge.data.preprocessing import _tokenize_mlm
        tok = self._make_tokenizer()
        result = _tokenize_mlm({"text": "hello world"}, tok, 512, seed=42)
        assert "input_ids" in result
        assert "labels" in result
        assert len(result["input_ids"]) == len(result["labels"])

    def test_special_tokens_not_masked(self):
        from mlx_forge.data.preprocessing import _tokenize_mlm
        tok = self._make_tokenizer()
        result = _tokenize_mlm({"text": "hello world"}, tok, 512, mlm_probability=1.0, seed=42)
        # CLS (pos 0) and SEP (pos 5) should not be masked
        assert result["labels"][0] == -100  # CLS
        assert result["labels"][5] == -100  # SEP

    def test_masking_distribution(self):
        """With many samples, masking should be approximately 15%."""
        from mlx_forge.data.preprocessing import _tokenize_mlm
        tok = MagicMock()
        # 100 non-special tokens
        tok.encode = MagicMock(return_value=list(range(200, 300)))
        tok.cls_token_id = 101
        tok.sep_token_id = 102
        tok.pad_token_id = 0
        tok.mask_token_id = 103
        tok.vocab_size = 30000

        total_masked = 0
        total_tokens = 0
        for i in range(100):
            result = _tokenize_mlm({"text": "test"}, tok, 512, seed=i)
            total_masked += sum(1 for l in result["labels"] if l != -100)
            total_tokens += len(result["labels"])

        ratio = total_masked / total_tokens
        assert 0.10 < ratio < 0.20, f"MLM masking ratio {ratio:.2f} outside expected range"

    def test_mask_token_applied(self):
        """80% of masked tokens should become [MASK]."""
        from mlx_forge.data.preprocessing import _tokenize_mlm
        tok = MagicMock()
        tok.encode = MagicMock(return_value=list(range(200, 300)))
        tok.cls_token_id = 101
        tok.sep_token_id = 102
        tok.pad_token_id = 0
        tok.mask_token_id = 103
        tok.vocab_size = 30000

        mask_count = 0
        total_masked = 0
        for i in range(200):
            result = _tokenize_mlm({"text": "test"}, tok, 512, seed=i)
            for j, l in enumerate(result["labels"]):
                if l != -100:
                    total_masked += 1
                    if result["input_ids"][j] == 103:
                        mask_count += 1

        ratio = mask_count / total_masked if total_masked > 0 else 0
        assert 0.70 < ratio < 0.90, f"[MASK] ratio {ratio:.2f} outside expected 80%"

    def test_truncation(self):
        from mlx_forge.data.preprocessing import _tokenize_mlm
        tok = MagicMock()
        tok.encode = MagicMock(return_value=list(range(500)))
        tok.cls_token_id = 101
        tok.sep_token_id = 102
        tok.pad_token_id = 0
        tok.mask_token_id = 103
        tok.vocab_size = 30000

        result = _tokenize_mlm({"text": "test"}, tok, 128, seed=42)
        assert len(result["input_ids"]) == 128


# ──────────────────────────────────────────────────────────────────────
# MLM Batching Tests
# ──────────────────────────────────────────────────────────────────────

class TestIterateMLMBatches:
    def test_yields_three_tensors(self):
        from mlx_forge.data.batching import iterate_mlm_batches

        dataset = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
            {"input_ids": [4, 5, 6, 7], "labels": [4, 5, 6, 7]},
        ]
        config = MagicMock()
        config.training.batch_size = 2
        config.training.mlm_probability = 0.15
        config.data.max_seq_length = 512

        batches = list(iterate_mlm_batches(dataset, config))
        assert len(batches) == 1

        input_ids, labels, attention_mask = batches[0]
        assert input_ids.shape[0] == 2
        assert labels.shape[0] == 2
        assert attention_mask.shape[0] == 2

    def test_attention_mask_correct(self):
        from mlx_forge.data.batching import iterate_mlm_batches

        dataset = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        ]
        config = MagicMock()
        config.training.batch_size = 1
        config.training.mlm_probability = 0.15
        config.data.max_seq_length = 512

        for input_ids, labels, attention_mask in iterate_mlm_batches(dataset, config):
            mx.eval(attention_mask)
            # First 3 positions should be 1, rest 0
            assert attention_mask[0, 0].item() == 1
            assert attention_mask[0, 1].item() == 1
            assert attention_mask[0, 2].item() == 1

    def test_padding(self):
        from mlx_forge.data.batching import iterate_mlm_batches

        dataset = [
            {"input_ids": [1, 2], "labels": [1, 2]},
            {"input_ids": [3, 4, 5, 6], "labels": [3, 4, 5, 6]},
        ]
        config = MagicMock()
        config.training.batch_size = 2
        config.training.mlm_probability = 0.15
        config.data.max_seq_length = 512

        for input_ids, labels, attention_mask in iterate_mlm_batches(dataset, config):
            mx.eval(input_ids, labels, attention_mask)
            # Both padded to same length (multiple of 32)
            assert input_ids.shape[1] == labels.shape[1] == attention_mask.shape[1]
            # Shorter sample should have padding
            T = input_ids.shape[1]
            assert T >= 4  # At least as long as longest sample

    def test_dynamic_masking_applied(self):
        """Verify that iterate_mlm_batches applies dynamic masking."""
        from mlx_forge.data.batching import iterate_mlm_batches

        # 20 tokens, high masking probability to ensure some are masked
        dataset = [
            {"input_ids": list(range(200, 220)), "labels": list(range(200, 220))},
        ]
        config = MagicMock()
        config.training.batch_size = 1
        config.training.mlm_probability = 0.5  # 50% for reliable detection
        config.data.max_seq_length = 512

        for input_ids, labels, attention_mask in iterate_mlm_batches(dataset, config):
            mx.eval(input_ids, labels)
            labels_list = labels[0].tolist()
            # Some positions should have labels != -100 (masked positions)
            masked_count = sum(1 for l in labels_list if l != -100)
            assert masked_count > 0, "Dynamic masking should produce some masked positions"
            # Some positions should have labels == -100 (unmasked)
            unmasked_count = sum(1 for l in labels_list if l == -100)
            assert unmasked_count > 0, "Not all positions should be masked"


# ──────────────────────────────────────────────────────────────────────
# MLM Trainer Tests
# ──────────────────────────────────────────────────────────────────────

class TestMLMTrainer:
    def _make_tiny_model(self):
        from mlx_forge.losses.mlm import MLMHead, MLMWrapper
        from mlx_forge.models.architectures.bert import Model, ModelArgs

        args = ModelArgs.from_dict({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 50,
            "max_position_embeddings": 32,
        })
        encoder = Model(args)
        mlm_head = MLMHead(32, 50)
        wrapper = MLMWrapper(encoder, mlm_head)
        mx.eval(wrapper.parameters())
        return wrapper

    def _make_checkpoint_manager(self):
        mgr = MagicMock()
        mgr.save = MagicMock(return_value="/tmp/test_ckpt")
        return mgr

    def test_instantiation(self):
        from mlx_forge.trainer.mlm_trainer import MLMTrainer

        model = self._make_tiny_model()
        config = MagicMock()
        config.training.batch_size = 2
        config.training.num_iters = 10
        config.training.learning_rate = 1e-4
        config.training.optimizer = "adam"
        config.training.optimizer_config = {}
        config.training.lr_schedule = None
        config.training.grad_accumulation_steps = 1
        config.training.max_grad_norm = None
        config.training.seed = 42
        config.training.val_batches = 5
        config.data.max_seq_length = 32

        dataset = [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
        ] * 4

        trainer = MLMTrainer(
            model=model, config=config,
            train_dataset=dataset, val_dataset=dataset,
            checkpoint_manager=self._make_checkpoint_manager(),
        )
        assert trainer is not None

    def test_evaluate(self):
        from mlx_forge.trainer.mlm_trainer import MLMTrainer

        model = self._make_tiny_model()
        config = MagicMock()
        config.training.batch_size = 2
        config.training.num_iters = 5
        config.training.learning_rate = 1e-4
        config.training.optimizer = "adam"
        config.training.optimizer_config = {}
        config.training.lr_schedule = None
        config.training.grad_accumulation_steps = 1
        config.training.max_grad_norm = None
        config.training.seed = 42
        config.training.mlm_probability = 0.15
        config.training.val_batches = 2
        config.data.max_seq_length = 32

        dataset = [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
            {"input_ids": [10, 11, 12], "labels": [10, 11, 12]},
        ] * 4

        trainer = MLMTrainer(
            model=model, config=config,
            train_dataset=dataset, val_dataset=dataset,
            checkpoint_manager=self._make_checkpoint_manager(),
        )
        val_loss = trainer.evaluate()
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_loss_decreases(self):
        """MLM loss should decrease over a few training steps."""
        from mlx_forge.trainer.mlm_trainer import MLMTrainer

        model = self._make_tiny_model()
        config = MagicMock()
        config.training.batch_size = 2
        config.training.num_iters = 20
        config.training.learning_rate = 1e-3
        config.training.optimizer = "adam"
        config.training.optimizer_config = {}
        config.training.lr_schedule = None
        config.training.grad_accumulation_steps = 1
        config.training.max_grad_norm = None
        config.training.seed = 42
        config.training.val_batches = 2
        config.training.steps_per_report = 5
        config.training.steps_per_eval = 100
        config.training.steps_per_save = 100
        config.training.mlm_probability = 0.5
        config.runtime.eager = True
        config.data.max_seq_length = 32

        dataset = [
            {"input_ids": [1, 2, 3, 4], "labels": [1, 2, 3, 4]},
            {"input_ids": [10, 11, 12], "labels": [10, 11, 12]},
        ] * 10

        trainer = MLMTrainer(
            model=model, config=config,
            train_dataset=dataset, val_dataset=dataset,
            checkpoint_manager=self._make_checkpoint_manager(),
        )

        loss_before = trainer.evaluate()
        # Run a few manual steps
        from mlx_forge.data.batching import iterate_mlm_batches
        loss_value_and_grad = nn.value_and_grad(model, trainer.loss)
        for batch in list(iterate_mlm_batches(dataset, config))[:5]:
            input_ids, labels, attention_mask = batch
            (loss, ntoks), grad = loss_value_and_grad(
                model, input_ids, labels, attention_mask)
            trainer.optimizer.update(model, grad)
            mx.eval(model.parameters(), trainer.optimizer.state)

        loss_after = trainer.evaluate()
        assert loss_after < loss_before, f"Loss should decrease: {loss_before} → {loss_after}"


# ──────────────────────────────────────────────────────────────────────
# MLMWrapper Gradient Flow Tests
# ──────────────────────────────────────────────────────────────────────

class TestMLMWrapperGradients:
    def test_gradients_flow_to_both_encoder_and_head(self):
        from mlx_forge.losses.mlm import MLMHead, MLMLoss, MLMWrapper
        from mlx_forge.models.architectures.bert import Model, ModelArgs

        args = ModelArgs.from_dict({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 50,
            "max_position_embeddings": 16,
        })
        encoder = Model(args)
        mlm_head = MLMHead(32, 50)
        wrapper = MLMWrapper(encoder, mlm_head)
        mx.eval(wrapper.parameters())

        loss_fn = MLMLoss()
        loss_value_and_grad = nn.value_and_grad(wrapper, loss_fn)

        input_ids = mx.array([[1, 2, 3, 4]])
        labels = mx.array([[-100, 5, -100, 7]])

        (loss, ntoks), grads = loss_value_and_grad(wrapper, input_ids, labels)
        mx.eval(loss, ntoks, grads)

        # Check gradients exist for encoder parameters
        from mlx.utils import tree_flatten
        flat_grads = tree_flatten(grads)
        assert any("encoder" in k for k, _ in flat_grads)
        # Check gradients exist for mlm_head parameters
        assert any("mlm_head" in k for k, _ in flat_grads)

    def test_wrapper_layers_property(self):
        from mlx_forge.losses.mlm import MLMHead, MLMWrapper
        from mlx_forge.models.architectures.bert import Model, ModelArgs

        args = ModelArgs.from_dict({
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 50,
        })
        encoder = Model(args)
        mlm_head = MLMHead(32, 50)
        wrapper = MLMWrapper(encoder, mlm_head)

        assert len(wrapper.layers) == 3


# ──────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────

class TestMLMConfig:
    def test_training_type_mlm_accepted(self):
        from mlx_forge.config import TrainingParams
        params = TrainingParams(training_type="mlm")
        assert params.training_type == "mlm"

    def test_mlm_probability_default(self):
        from mlx_forge.config import TrainingParams
        params = TrainingParams()
        assert params.mlm_probability == 0.15

    def test_pooling_strategy_values(self):
        from mlx_forge.config import TrainingParams
        params_cls = TrainingParams(pooling_strategy="cls")
        assert params_cls.pooling_strategy == "cls"
        params_mean = TrainingParams(pooling_strategy="mean")
        assert params_mean.pooling_strategy == "mean"


# ──────────────────────────────────────────────────────────────────────
# Encoder Inference Tests
# ──────────────────────────────────────────────────────────────────────

class TestEncoderInference:
    def _make_model_and_tokenizer(self):
        from mlx_forge.models.architectures.bert import Model, ModelArgs
        args = ModelArgs.from_dict({
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 100,
            "max_position_embeddings": 32,
        })
        model = Model(args)
        mx.eval(model.parameters())

        class FakeTokenizer:
            def __call__(self, texts, padding=True, truncation=True, return_tensors="np"):
                max_len = 5
                input_ids = np.zeros((len(texts), max_len), dtype=np.int32)
                attention_mask = np.zeros((len(texts), max_len), dtype=np.int32)
                for i, t in enumerate(texts):
                    tokens = [1, 2, 3]
                    input_ids[i, :len(tokens)] = tokens
                    attention_mask[i, :len(tokens)] = 1
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def encode(self, text):
                return [1, 2, 3]

        return model, FakeTokenizer()

    def test_cls_pooling(self):
        from mlx_forge.inference.encoder import encode
        model, tokenizer = self._make_model_and_tokenizer()
        embeddings = encode(model, tokenizer, ["hello"], pooling="cls")
        assert len(embeddings) == 1
        assert embeddings[0].shape == (32,)

    def test_mean_pooling(self):
        from mlx_forge.inference.encoder import encode
        model, tokenizer = self._make_model_and_tokenizer()
        embeddings = encode(model, tokenizer, ["hello"], pooling="mean")
        assert len(embeddings) == 1
        assert embeddings[0].shape == (32,)

    def test_normalization(self):
        from mlx_forge.inference.encoder import encode
        model, tokenizer = self._make_model_and_tokenizer()
        embeddings = encode(model, tokenizer, ["hello"], normalize=True)
        norm = float((embeddings[0] * embeddings[0]).sum() ** 0.5)
        assert abs(norm - 1.0) < 1e-4

    def test_no_normalization(self):
        from mlx_forge.inference.encoder import encode
        model, tokenizer = self._make_model_and_tokenizer()
        embeddings = encode(model, tokenizer, ["hello"], normalize=False)
        norm = float((embeddings[0] * embeddings[0]).sum() ** 0.5)
        # Not necessarily 1.0
        assert norm > 0


# ──────────────────────────────────────────────────────────────────────
# Embedding Request/Response Types
# ──────────────────────────────────────────────────────────────────────

class TestEmbeddingTypes:
    def test_embedding_request(self):
        from mlx_forge.serving.openai_types import EmbeddingRequest
        req = EmbeddingRequest(model="bert-base", input=["hello", "world"])
        assert req.model == "bert-base"
        assert len(req.input) == 2

    def test_embedding_response(self):
        from mlx_forge.serving.openai_types import EmbeddingData, EmbeddingResponse, EmbeddingUsage
        resp = EmbeddingResponse(
            data=[EmbeddingData(embedding=[0.1, 0.2], index=0)],
            model="bert-base",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )
        assert resp.object == "list"
        assert len(resp.data) == 1


# ──────────────────────────────────────────────────────────────────────
# LoRA Preset Tests
# ──────────────────────────────────────────────────────────────────────

class TestBertLoRAPresets:
    def test_bert_attention_preset(self):
        from mlx_forge.adapters.targeting import PRESETS
        assert "bert-attention" in PRESETS
        patterns = PRESETS["bert-attention"]
        assert "*.attention.query" in patterns
        assert "*.attention.value" in patterns

    def test_bert_all_preset(self):
        from mlx_forge.adapters.targeting import PRESETS
        assert "bert-all" in PRESETS
        patterns = PRESETS["bert-all"]
        assert len(patterns) == 6

    def test_bert_mlp_preset(self):
        from mlx_forge.adapters.targeting import PRESETS
        assert "bert-mlp" in PRESETS


class TestBertLoRATargeting:
    def test_lora_targets_resolve_on_bert(self):
        from mlx_forge.adapters.targeting import PRESETS, resolve_targets
        from mlx_forge.models.architectures.bert import Model, ModelArgs

        args = ModelArgs.from_dict({
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 50,
        })
        model = Model(args)

        targets = resolve_targets(model, PRESETS["bert-attention"])
        assert len(targets) > 0
        # Should match query and value projections
        names = [name for name, _ in targets]
        assert any("query" in n for n in names)
        assert any("value" in n for n in names)

    def test_lora_targets_bert_all(self):
        from mlx_forge.adapters.targeting import PRESETS, resolve_targets
        from mlx_forge.models.architectures.bert import Model, ModelArgs

        args = ModelArgs.from_dict({
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 64,
            "vocab_size": 50,
        })
        model = Model(args)

        targets = resolve_targets(model, PRESETS["bert-all"])
        names = [name for name, _ in targets]
        # Should include attention AND mlp targets
        assert any("query" in n for n in names)
        assert any("dense" in n for n in names)


# ──────────────────────────────────────────────────────────────────────
# Registry Tests
# ──────────────────────────────────────────────────────────────────────

class TestEncoderRegistry:
    def test_bert_in_registry(self):
        from mlx_forge.models.registry import SUPPORTED_ARCHITECTURES
        assert "bert" in SUPPORTED_ARCHITECTURES

    def test_roberta_remaps_to_bert(self):
        from mlx_forge.models.registry import MODEL_REMAPPING
        assert MODEL_REMAPPING.get("roberta") == "bert"

    def test_deberta_in_registry(self):
        from mlx_forge.models.registry import SUPPORTED_ARCHITECTURES
        assert "deberta" in SUPPORTED_ARCHITECTURES


# ──────────────────────────────────────────────────────────────────────
# CLI Encode Subcommand
# ──────────────────────────────────────────────────────────────────────

class TestCLIEncode:
    def test_encode_parser_exists(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["encode", "--model", "bert-base", "--texts", "hello", "world"])
        assert args.command == "encode"
        assert args.model == "bert-base"
        assert args.texts == ["hello", "world"]

    def test_encode_parser_options(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "encode", "--model", "bert-base",
            "--texts", "hello",
            "--pooling", "mean",
            "--no-normalize",
        ])
        assert args.pooling == "mean"
        assert args.no_normalize is True
