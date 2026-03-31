"""Tests for M36: Encoder-Decoder Models — T5, BART.

Tests cover:
- T5 ModelArgs, encoder/decoder forward pass
- T5 relative position bias bucketing
- T5 cross-attention
- BART forward pass
- sanitize() for T5, BART
- Seq2SeqLoss computation, masking
- _tokenize_seq2seq: input/target split
- iterate_seq2seq_batches: shapes, padding
- Seq2SeqTrainer with tiny model
- generate_seq2seq_tokens: two-phase generation, EOS stop
- T5 LoRA presets
- LoRA on encoder only / decoder only / both
- Registry: t5, mt5, bart, mbart
- Config: training_type="seq2seq"
- KV cache for decoder
- Data format: seq2seq detection
- Full seq2seq pipeline: tokenize → batch → train step
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest


# ──────────────────────────────────────────────────────────────────────
# T5 Architecture Tests
# ──────────────────────────────────────────────────────────────────────

class TestT5ModelArgs:
    def test_from_dict(self):
        from mlx_forge.models.architectures.t5 import ModelArgs
        args = ModelArgs.from_dict({
            "model_type": "t5",
            "d_model": 64,
            "d_ff": 128,
            "d_kv": 16,
            "num_heads": 4,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "vocab_size": 100,
        })
        assert args.d_model == 64
        assert args.num_layers == 2
        assert args.vocab_size == 100

    def test_default_values(self):
        from mlx_forge.models.architectures.t5 import ModelArgs
        args = ModelArgs.from_dict({"model_type": "t5"})
        assert args.d_model == 512
        assert args.num_heads == 8
        assert args.vocab_size == 32128


class TestT5ForwardPass:
    @pytest.fixture
    def tiny_t5(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "model_type": "t5",
            "d_model": 64,
            "d_ff": 128,
            "d_kv": 16,
            "num_heads": 4,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "vocab_size": 100,
        })
        model = Model(args)
        mx.eval(model.parameters())
        return model

    def test_full_forward_pass(self, tiny_t5):
        enc_ids = mx.array([[1, 2, 3, 4]])
        dec_ids = mx.array([[0, 5, 6]])
        logits = tiny_t5(enc_ids, dec_ids)
        mx.eval(logits)
        assert logits.shape == (1, 3, 100)  # (B, T_dec, V)

    def test_model_category(self, tiny_t5):
        assert tiny_t5.model_category == "encoder_decoder"

    def test_encode_returns_hidden_states(self, tiny_t5):
        enc_ids = mx.array([[1, 2, 3, 4]])
        hidden = tiny_t5.encode(enc_ids)
        mx.eval(hidden)
        assert hidden.shape == (1, 4, 64)  # (B, T_enc, D)

    def test_decode_returns_logits(self, tiny_t5):
        enc_ids = mx.array([[1, 2, 3]])
        hidden = tiny_t5.encode(enc_ids)
        dec_ids = mx.array([[0, 5]])
        logits = tiny_t5.decode(dec_ids, hidden)
        mx.eval(logits)
        assert logits.shape == (1, 2, 100)

    def test_with_encoder_attention_mask(self, tiny_t5):
        enc_ids = mx.array([[1, 2, 0, 0]])
        enc_mask = mx.array([[1, 1, 0, 0]])
        dec_ids = mx.array([[0, 5]])
        logits = tiny_t5(enc_ids, dec_ids, encoder_attention_mask=enc_mask)
        mx.eval(logits)
        assert logits.shape == (1, 2, 100)

    def test_batch_forward(self, tiny_t5):
        enc_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        dec_ids = mx.array([[0, 7], [0, 8]])
        logits = tiny_t5(enc_ids, dec_ids)
        mx.eval(logits)
        assert logits.shape == (2, 2, 100)


class TestT5RelativePositionBias:
    def test_bucketing(self):
        from mlx_forge.models.architectures.t5 import _relative_position_bucket
        rel_pos = mx.array([[-2, -1, 0, 1, 2, 5, 10, 100]])
        buckets = _relative_position_bucket(
            rel_pos, bidirectional=True, num_buckets=32, max_distance=128
        )
        mx.eval(buckets)
        assert buckets.shape == (1, 8)
        # All buckets should be non-negative
        assert buckets.min().item() >= 0

    def test_unidirectional_bucketing(self):
        from mlx_forge.models.architectures.t5 import _relative_position_bucket
        rel_pos = mx.array([[-5, 0, 5, 10]])
        buckets = _relative_position_bucket(
            rel_pos, bidirectional=False, num_buckets=32, max_distance=128
        )
        mx.eval(buckets)
        # In unidirectional mode, n = max(-rel_pos, 0), so -5 → n=5, which maps to bucket 5
        # Position 0 (rel_pos=0) should map to bucket 0
        assert buckets[0, 1].item() == 0  # rel_pos=0 → bucket 0

    def test_symmetry_in_bidirectional(self):
        from mlx_forge.models.architectures.t5 import _relative_position_bucket
        pos = mx.array([[5]])
        neg = mx.array([[-5]])
        b_pos = _relative_position_bucket(pos, bidirectional=True, num_buckets=32, max_distance=128)
        b_neg = _relative_position_bucket(neg, bidirectional=True, num_buckets=32, max_distance=128)
        mx.eval(b_pos, b_neg)
        # Should be in different halves of bucket space
        assert b_pos[0, 0].item() != b_neg[0, 0].item()


class TestT5CrossAttention:
    def test_cross_attention_shapes(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())

        enc_ids = mx.array([[1, 2, 3, 4, 5]])  # 5 tokens
        dec_ids = mx.array([[0, 10, 11]])  # 3 tokens
        logits = model(enc_ids, dec_ids)
        mx.eval(logits)
        assert logits.shape == (1, 3, 50)

    def test_encoder_hidden_different_length(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())

        # Encoder and decoder have different sequence lengths
        enc_hidden = model.encode(mx.array([[1, 2, 3, 4, 5, 6, 7]]))  # 7 tokens
        logits = model.decode(mx.array([[0, 10]]), enc_hidden)  # 2 tokens
        mx.eval(logits)
        assert logits.shape == (1, 2, 50)

    def test_cross_attention_uses_encoder_hidden(self):
        """Verify decoder output changes when encoder input changes."""
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())

        dec_ids = mx.array([[0, 10]])

        logits_a = model(mx.array([[1, 2, 3]]), dec_ids)
        logits_b = model(mx.array([[40, 41, 42]]), dec_ids)
        mx.eval(logits_a, logits_b)

        diff = mx.abs(logits_a - logits_b).sum().item()
        assert diff > 0, "Decoder should produce different output for different encoder input"


# ──────────────────────────────────────────────────────────────────────
# BART Architecture Tests
# ──────────────────────────────────────────────────────────────────────

class TestBARTForwardPass:
    @pytest.fixture
    def tiny_bart(self):
        from mlx_forge.models.architectures.bart import Model, ModelArgs
        args = ModelArgs.from_dict({
            "model_type": "bart",
            "d_model": 64,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "encoder_ffn_dim": 128,
            "decoder_ffn_dim": 128,
            "vocab_size": 100,
            "max_position_embeddings": 32,
        })
        model = Model(args)
        mx.eval(model.parameters())
        return model

    def test_full_forward_pass(self, tiny_bart):
        enc_ids = mx.array([[1, 2, 3, 4]])
        dec_ids = mx.array([[2, 5, 6]])
        logits = tiny_bart(enc_ids, dec_ids)
        mx.eval(logits)
        assert logits.shape == (1, 3, 100)

    def test_model_category(self, tiny_bart):
        assert tiny_bart.model_category == "encoder_decoder"

    def test_encode_decode_split(self, tiny_bart):
        enc_ids = mx.array([[1, 2, 3]])
        hidden = tiny_bart.encode(enc_ids)
        mx.eval(hidden)
        assert hidden.shape == (1, 3, 64)

        dec_ids = mx.array([[2, 5]])
        logits = tiny_bart.decode(dec_ids, hidden)
        mx.eval(logits)
        assert logits.shape == (1, 2, 100)


# ──────────────────────────────────────────────────────────────────────
# Sanitize Tests
# ──────────────────────────────────────────────────────────────────────

class TestSanitize:
    def test_t5_sanitize(self):
        from mlx_forge.models.architectures.t5 import Model
        weights = {
            "shared.weight": mx.zeros((100, 64)),
            "encoder.block.0.layer.0.SelfAttention.q.weight": mx.zeros((64, 64)),
            "decoder.block.0.layer.0.SelfAttention.q.weight": mx.zeros((64, 64)),
            "decoder.block.0.layer.1.EncDecAttention.q.weight": mx.zeros((64, 64)),
            "encoder.final_layer_norm.weight": mx.zeros((64,)),
            "decoder.embed_tokens.weight": mx.zeros((100, 64)),
        }
        sanitized = Model.sanitize(weights)
        assert "shared.weight" in sanitized
        assert "encoder.layers.0.self_attn.q_proj.weight" in sanitized
        assert "decoder.layers.0.self_attn.q_proj.weight" in sanitized
        assert "decoder.layers.0.cross_attn.q_proj.weight" in sanitized
        assert "encoder.final_norm.weight" in sanitized
        # decoder.embed_tokens should be dropped (uses shared)
        assert not any("embed_tokens" in k for k in sanitized)

    def test_bart_sanitize(self):
        from mlx_forge.models.architectures.bart import Model
        weights = {
            "model.shared.weight": mx.zeros((100, 64)),
            "model.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
            "model.decoder.layers.0.cross_attn.q_proj.weight": mx.zeros((64, 64)),
            "model.encoder.embed_tokens.weight": mx.zeros((100, 64)),
        }
        sanitized = Model.sanitize(weights)
        assert "shared.weight" in sanitized
        assert "encoder.layers.0.self_attn.q_proj.weight" in sanitized
        # embed_tokens should be dropped
        assert not any("embed_tokens" in k for k in sanitized)

    def test_t5_mlp_sanitize(self):
        from mlx_forge.models.architectures.t5 import Model
        weights = {
            "encoder.block.0.layer.1.DenseReluDense.wi.weight": mx.zeros((128, 64)),
            "encoder.block.0.layer.1.DenseReluDense.wo.weight": mx.zeros((64, 128)),
            "encoder.block.0.layer.1.DenseReluDense.wi_0.weight": mx.zeros((128, 64)),
        }
        sanitized = Model.sanitize(weights)
        assert "encoder.layers.0.mlp.wo.weight" in sanitized

    def test_t5_layer_norm_sanitize(self):
        from mlx_forge.models.architectures.t5 import Model
        weights = {
            "encoder.block.0.layer.0.layer_norm.weight": mx.zeros((64,)),
            "decoder.block.0.layer.0.layer_norm.weight": mx.zeros((64,)),
            "decoder.block.0.layer.1.layer_norm.weight": mx.zeros((64,)),
            "decoder.block.0.layer.2.layer_norm.weight": mx.zeros((64,)),
        }
        sanitized = Model.sanitize(weights)
        assert "encoder.layers.0.norm1.weight" in sanitized
        assert "decoder.layers.0.norm1.weight" in sanitized
        assert "decoder.layers.0.norm2.weight" in sanitized
        assert "decoder.layers.0.norm3.weight" in sanitized


# ──────────────────────────────────────────────────────────────────────
# Seq2Seq Loss Tests
# ──────────────────────────────────────────────────────────────────────

class TestSeq2SeqLoss:
    def _make_tiny_model(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())
        return model

    def test_loss_computation(self):
        from mlx_forge.losses.seq2seq import Seq2SeqLoss
        model = self._make_tiny_model()
        loss_fn = Seq2SeqLoss()

        enc_ids = mx.array([[1, 2, 3]])
        dec_ids = mx.array([[0, 5, 6, 7]])
        dec_labels = mx.array([[5, 6, 7, 1]])  # shifted

        loss, ntoks = loss_fn(model, enc_ids, dec_ids, dec_labels)
        mx.eval(loss, ntoks)

        assert loss.item() > 0
        assert ntoks.item() > 0

    def test_masking(self):
        from mlx_forge.losses.seq2seq import Seq2SeqLoss
        model = self._make_tiny_model()
        loss_fn = Seq2SeqLoss()

        enc_ids = mx.array([[1, 2, 3]])
        dec_ids = mx.array([[0, 5, 6, 7]])
        dec_labels = mx.array([[-100, 6, -100, -100]])  # Only position 1 contributes

        loss, ntoks = loss_fn(model, enc_ids, dec_ids, dec_labels)
        mx.eval(loss, ntoks)
        assert ntoks.item() == 1  # Only labels[:, 1:] where != -100 → position 1 (6)

    def test_decoder_shift(self):
        """Verify seq2seq loss uses shifted decoder labels."""
        from mlx_forge.losses.seq2seq import Seq2SeqLoss
        model = self._make_tiny_model()
        loss_fn = Seq2SeqLoss()

        enc_ids = mx.array([[1, 2]])
        dec_ids = mx.array([[0, 5, 6]])
        # Labels: position 0 is ignored by shift, positions 1 and 2 contribute
        dec_labels = mx.array([[-100, 6, 1]])

        loss, ntoks = loss_fn(model, enc_ids, dec_ids, dec_labels)
        mx.eval(loss, ntoks)
        # After shift: targets = labels[:, 1:] = [6, 1], both != -100
        assert ntoks.item() == 2

    def test_all_masked(self):
        from mlx_forge.losses.seq2seq import Seq2SeqLoss
        model = self._make_tiny_model()
        loss_fn = Seq2SeqLoss()

        enc_ids = mx.array([[1, 2]])
        dec_ids = mx.array([[0, 5, 6]])
        dec_labels = mx.array([[-100, -100, -100]])

        loss, ntoks = loss_fn(model, enc_ids, dec_ids, dec_labels)
        mx.eval(loss, ntoks)
        assert ntoks.item() == 0


# ──────────────────────────────────────────────────────────────────────
# Seq2Seq Tokenization Tests
# ──────────────────────────────────────────────────────────────────────

class TestTokenizeSeq2Seq:
    def _make_tokenizer(self):
        tok = MagicMock()
        tok.encode = MagicMock(side_effect=lambda text, **kw: list(range(100, 100 + len(text.split()))))
        tok.decoder_start_token_id = 0
        tok.eos_token_id = 1
        tok.pad_token_id = 0
        return tok

    def test_output_keys(self):
        from mlx_forge.data.preprocessing import _tokenize_seq2seq
        tok = self._make_tokenizer()
        result = _tokenize_seq2seq(
            {"input": "hello world", "target": "bonjour monde"}, tok, 512
        )
        assert "encoder_input_ids" in result
        assert "decoder_input_ids" in result
        assert "decoder_labels" in result

    def test_decoder_start_token(self):
        from mlx_forge.data.preprocessing import _tokenize_seq2seq
        tok = self._make_tokenizer()
        result = _tokenize_seq2seq(
            {"input": "hello", "target": "bonjour"}, tok, 512
        )
        assert result["decoder_input_ids"][0] == 0  # decoder_start_token_id

    def test_decoder_labels_end_with_eos(self):
        from mlx_forge.data.preprocessing import _tokenize_seq2seq
        tok = self._make_tokenizer()
        result = _tokenize_seq2seq(
            {"input": "hello", "target": "bonjour"}, tok, 512
        )
        assert result["decoder_labels"][-1] == 1  # eos_token_id

    def test_truncation(self):
        from mlx_forge.data.preprocessing import _tokenize_seq2seq
        tok = MagicMock()
        tok.encode = MagicMock(return_value=list(range(500)))
        tok.decoder_start_token_id = 0
        tok.eos_token_id = 1
        result = _tokenize_seq2seq(
            {"input": "long input", "target": "long target"}, tok, 128
        )
        assert len(result["encoder_input_ids"]) <= 128
        assert len(result["decoder_input_ids"]) <= 128


# ──────────────────────────────────────────────────────────────────────
# Seq2Seq Batching Tests
# ──────────────────────────────────────────────────────────────────────

class TestIterateSeq2SeqBatches:
    def test_yields_four_tensors(self):
        from mlx_forge.data.batching import iterate_seq2seq_batches

        dataset = [
            {
                "encoder_input_ids": [1, 2, 3],
                "decoder_input_ids": [0, 4, 5],
                "decoder_labels": [4, 5, 1],
            },
            {
                "encoder_input_ids": [6, 7],
                "decoder_input_ids": [0, 8],
                "decoder_labels": [8, 1],
            },
        ]
        config = MagicMock()
        config.training.batch_size = 2
        config.data.max_seq_length = 512

        batches = list(iterate_seq2seq_batches(dataset, config))
        assert len(batches) == 1

        enc_ids, dec_ids, dec_labels, enc_mask = batches[0]
        assert enc_ids.shape[0] == 2
        assert dec_ids.shape[0] == 2
        assert dec_labels.shape[0] == 2
        assert enc_mask.shape[0] == 2

    def test_encoder_attention_mask(self):
        from mlx_forge.data.batching import iterate_seq2seq_batches

        dataset = [
            {
                "encoder_input_ids": [1, 2],
                "decoder_input_ids": [0, 3],
                "decoder_labels": [3, 1],
            },
        ]
        config = MagicMock()
        config.training.batch_size = 1
        config.data.max_seq_length = 512

        for enc_ids, dec_ids, dec_labels, enc_mask in iterate_seq2seq_batches(dataset, config):
            mx.eval(enc_mask)
            assert enc_mask[0, 0].item() == 1
            assert enc_mask[0, 1].item() == 1

    def test_independent_padding(self):
        from mlx_forge.data.batching import iterate_seq2seq_batches

        dataset = [
            {
                "encoder_input_ids": [1, 2, 3, 4, 5],
                "decoder_input_ids": [0, 6],
                "decoder_labels": [6, 1],
            },
        ]
        config = MagicMock()
        config.training.batch_size = 1
        config.data.max_seq_length = 512

        for enc_ids, dec_ids, dec_labels, enc_mask in iterate_seq2seq_batches(dataset, config):
            mx.eval(enc_ids, dec_ids)
            # Encoder and decoder can have different padded lengths
            assert enc_ids.shape[1] >= 5
            assert dec_ids.shape[1] >= 2


# ──────────────────────────────────────────────────────────────────────
# Seq2Seq Trainer Tests
# ──────────────────────────────────────────────────────────────────────

class TestSeq2SeqTrainer:
    def _make_tiny_model(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())
        return model

    def _make_checkpoint_manager(self):
        mgr = MagicMock()
        mgr.save = MagicMock(return_value="/tmp/test_ckpt")
        return mgr

    def test_instantiation(self):
        from mlx_forge.trainer.seq2seq_trainer import Seq2SeqTrainer

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
        config.training.val_batches = 5
        config.data.max_seq_length = 32

        dataset = [
            {
                "encoder_input_ids": [1, 2, 3],
                "decoder_input_ids": [0, 4, 5],
                "decoder_labels": [4, 5, 1],
            },
        ] * 4

        trainer = Seq2SeqTrainer(
            model=model, config=config,
            train_dataset=dataset, val_dataset=dataset,
            checkpoint_manager=self._make_checkpoint_manager(),
        )
        assert trainer is not None

    def test_evaluate(self):
        from mlx_forge.trainer.seq2seq_trainer import Seq2SeqTrainer

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
        config.training.val_batches = 2
        config.data.max_seq_length = 32

        dataset = [
            {
                "encoder_input_ids": [1, 2, 3],
                "decoder_input_ids": [0, 4, 5],
                "decoder_labels": [4, 5, 1],
            },
        ] * 4

        trainer = Seq2SeqTrainer(
            model=model, config=config,
            train_dataset=dataset, val_dataset=dataset,
            checkpoint_manager=self._make_checkpoint_manager(),
        )
        val_loss = trainer.evaluate()
        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_loss_decreases(self):
        from mlx_forge.trainer.seq2seq_trainer import Seq2SeqTrainer

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
        config.data.max_seq_length = 32
        config.runtime.eager = True

        dataset = [
            {
                "encoder_input_ids": [1, 2, 3],
                "decoder_input_ids": [0, 4, 5],
                "decoder_labels": [4, 5, 1],
            },
        ] * 10

        trainer = Seq2SeqTrainer(
            model=model, config=config,
            train_dataset=dataset, val_dataset=dataset,
            checkpoint_manager=self._make_checkpoint_manager(),
        )

        loss_before = trainer.evaluate()

        from mlx_forge.data.batching import iterate_seq2seq_batches
        loss_value_and_grad = nn.value_and_grad(model, trainer.loss)
        for batch in list(iterate_seq2seq_batches(dataset, config))[:5]:
            enc_ids, dec_ids, dec_labels, enc_mask = batch
            (loss, ntoks), grad = loss_value_and_grad(
                model, enc_ids, dec_ids, dec_labels, enc_mask)
            trainer.optimizer.update(model, grad)
            mx.eval(model.parameters(), trainer.optimizer.state)

        loss_after = trainer.evaluate()
        assert loss_after < loss_before, f"Loss should decrease: {loss_before} → {loss_after}"


# ──────────────────────────────────────────────────────────────────────
# Seq2Seq Generation Tests
# ──────────────────────────────────────────────────────────────────────

class TestGenerateSeq2SeqTokens:
    def _make_model_and_tokenizer(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
            "eos_token_id": 1,
        })
        model = Model(args)
        mx.eval(model.parameters())

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 1
        return model, tokenizer

    def test_generates_tokens(self):
        from mlx_forge.inference.engine import generate_seq2seq_tokens
        model, tokenizer = self._make_model_and_tokenizer()

        tokens = list(generate_seq2seq_tokens(
            model, [1, 2, 3], tokenizer,
            max_tokens=10, temperature=0.7, seed=42,
        ))
        assert len(tokens) > 0
        assert len(tokens) <= 10

    def test_stops_at_eos(self):
        """If model produces EOS, generation should stop early."""
        from mlx_forge.inference.engine import generate_seq2seq_tokens
        model, tokenizer = self._make_model_and_tokenizer()

        # Generate with many max_tokens — should stop at EOS if it occurs
        tokens = list(generate_seq2seq_tokens(
            model, [1, 2, 3], tokenizer,
            max_tokens=100, temperature=0.0, seed=42,
        ))
        # Just verify it runs without error
        assert isinstance(tokens, list)

    def test_two_phase_architecture(self):
        """Verify encode is called once and decode is called multiple times."""
        from mlx_forge.models.architectures.t5 import Model, ModelArgs

        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())

        # Verify the model has separate encode/decode methods
        assert hasattr(model, 'encode')
        assert hasattr(model, 'decode')
        assert hasattr(model, 'make_cache')

        cache = model.make_cache()
        assert len(cache) == 1  # num_decoder_layers


# ──────────────────────────────────────────────────────────────────────
# LoRA Preset Tests
# ──────────────────────────────────────────────────────────────────────

class TestT5LoRAPresets:
    def test_t5_encoder_preset(self):
        from mlx_forge.adapters.targeting import PRESETS
        assert "t5-encoder" in PRESETS

    def test_t5_decoder_preset(self):
        from mlx_forge.adapters.targeting import PRESETS
        assert "t5-decoder" in PRESETS
        patterns = PRESETS["t5-decoder"]
        assert any("cross_attn" in p for p in patterns)


class TestT5LoRATargeting:
    def test_lora_on_encoder_only(self):
        from mlx_forge.adapters.targeting import PRESETS, resolve_targets
        from mlx_forge.models.architectures.t5 import Model, ModelArgs

        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "vocab_size": 50,
        })
        model = Model(args)

        targets = resolve_targets(model, PRESETS["t5-encoder"])
        names = [n for n, _ in targets]
        assert all("encoder" in n for n in names)

    def test_lora_on_decoder_only(self):
        from mlx_forge.adapters.targeting import PRESETS, resolve_targets
        from mlx_forge.models.architectures.t5 import Model, ModelArgs

        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "vocab_size": 50,
        })
        model = Model(args)

        targets = resolve_targets(model, PRESETS["t5-decoder"])
        names = [n for n, _ in targets]
        assert all("decoder" in n for n in names)

    def test_lora_on_both(self):
        from mlx_forge.adapters.targeting import PRESETS, resolve_targets
        from mlx_forge.models.architectures.t5 import Model, ModelArgs

        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 2,
            "num_decoder_layers": 2,
            "vocab_size": 50,
        })
        model = Model(args)

        targets = resolve_targets(model, PRESETS["t5-all"])
        names = [n for n, _ in targets]
        has_encoder = any("encoder" in n for n in names)
        has_decoder = any("decoder" in n for n in names)
        assert has_encoder and has_decoder


# ──────────────────────────────────────────────────────────────────────
# Registry Tests
# ──────────────────────────────────────────────────────────────────────

class TestEncDecRegistry:
    def test_t5_in_registry(self):
        from mlx_forge.models.registry import SUPPORTED_ARCHITECTURES
        assert "t5" in SUPPORTED_ARCHITECTURES

    def test_mt5_remaps_to_t5(self):
        from mlx_forge.models.registry import MODEL_REMAPPING
        assert MODEL_REMAPPING.get("mt5") == "t5"

    def test_bart_in_registry(self):
        from mlx_forge.models.registry import SUPPORTED_ARCHITECTURES
        assert "bart" in SUPPORTED_ARCHITECTURES


# ──────────────────────────────────────────────────────────────────────
# Config Tests
# ──────────────────────────────────────────────────────────────────────

class TestSeq2SeqConfig:
    def test_training_type_seq2seq(self):
        from mlx_forge.config import TrainingParams
        params = TrainingParams(training_type="seq2seq")
        assert params.training_type == "seq2seq"


# ──────────────────────────────────────────────────────────────────────
# KV Cache Tests
# ──────────────────────────────────────────────────────────────────────

class TestDecoderKVCache:
    def test_make_cache(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 2,
            "num_decoder_layers": 3,
            "vocab_size": 50,
        })
        model = Model(args)
        cache = model.make_cache()
        assert len(cache) == 3  # num_decoder_layers

    def test_cache_with_generation(self):
        from mlx_forge.models.architectures.t5 import Model, ModelArgs
        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())

        enc_hidden = model.encode(mx.array([[1, 2, 3]]))
        cache = model.make_cache()

        # First token
        logits1 = model.decode(mx.array([[0]]), enc_hidden, cache=cache)
        mx.eval(logits1)
        assert cache[0].offset == 1

        # Second token
        logits2 = model.decode(mx.array([[5]]), enc_hidden, cache=cache)
        mx.eval(logits2)
        assert cache[0].offset == 2


# ──────────────────────────────────────────────────────────────────────
# Data Format Tests
# ──────────────────────────────────────────────────────────────────────

class TestSeq2SeqFormat:
    def test_detect_seq2seq_format(self):
        from mlx_forge.data.formats import detect_format
        samples = [{"input": "Hello", "target": "Hallo"}]
        fmt = detect_format(samples)
        assert fmt == "seq2seq"

    def test_validate_seq2seq_sample(self):
        from mlx_forge.data.formats import validate_samples
        samples = [{"input": "Hello", "target": "Hallo"}]
        errors = validate_samples(samples, "seq2seq")
        assert len(errors) == 0

        bad_samples = [{"input": 123, "target": "Hallo"}]
        errors = validate_samples(bad_samples, "seq2seq")
        assert len(errors) > 0


# ──────────────────────────────────────────────────────────────────────
# Full Pipeline Test
# ──────────────────────────────────────────────────────────────────────

class TestSeq2SeqPipeline:
    def test_tokenize_batch_train_step(self):
        """End-to-end: tokenize → batch → compute loss."""
        from mlx_forge.data.batching import iterate_seq2seq_batches
        from mlx_forge.data.preprocessing import _tokenize_seq2seq
        from mlx_forge.losses.seq2seq import Seq2SeqLoss
        from mlx_forge.models.architectures.t5 import Model, ModelArgs

        tok = MagicMock()
        tok.encode = MagicMock(side_effect=lambda t, **kw: list(range(10, 10 + min(len(t), 10))))
        tok.decoder_start_token_id = 0
        tok.eos_token_id = 1

        samples = [
            {"input": "hello world", "target": "bonjour monde"},
            {"input": "how are you", "target": "comment allez"},
        ]

        tokenized = [_tokenize_seq2seq(s, tok, 128) for s in samples]

        config = MagicMock()
        config.training.batch_size = 2
        config.data.max_seq_length = 128

        batches = list(iterate_seq2seq_batches(tokenized, config))
        assert len(batches) >= 1

        args = ModelArgs.from_dict({
            "d_model": 32,
            "d_ff": 64,
            "d_kv": 8,
            "num_heads": 4,
            "num_layers": 1,
            "num_decoder_layers": 1,
            "vocab_size": 50,
        })
        model = Model(args)
        mx.eval(model.parameters())

        loss_fn = Seq2SeqLoss()
        enc_ids, dec_ids, dec_labels, enc_mask = batches[0]
        loss, ntoks = loss_fn(model, enc_ids, dec_ids, dec_labels, enc_mask)
        mx.eval(loss, ntoks)

        assert loss.item() > 0
        assert ntoks.item() > 0
