"""Seq2Seq Trainer for MLX Forge.

Extends BaseTrainer for encoder-decoder training (T5, BART).
"""

from __future__ import annotations

import itertools
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.data.batching import iterate_seq2seq_batches
from mlx_forge.losses.seq2seq import Seq2SeqLoss
from mlx_forge.trainer.trainer import BaseTrainer


class Seq2SeqTrainer(BaseTrainer):
    """Seq2Seq trainer for encoder-decoder models."""

    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.loss = Seq2SeqLoss()

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build compiled seq2seq step functions."""
        loss_value_and_grad = nn.value_and_grad(self.model, self.loss)

        if not self.config.runtime.eager:
            @partial(mx.compile, inputs=compile_state, outputs=compile_state)
            def step(encoder_ids, decoder_ids, decoder_labels, encoder_mask,
                     prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, encoder_ids, decoder_ids, decoder_labels, encoder_mask)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad
        else:
            def step(encoder_ids, decoder_ids, decoder_labels, encoder_mask,
                     prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, encoder_ids, decoder_ids, decoder_labels, encoder_mask)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

        return {"seq2seq": step}

    def _build_batch_iterator(self):
        """Build the seq2seq batch iterator."""
        return itertools.cycle(
            iterate_seq2seq_batches(self.train_dataset, self.config))

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one seq2seq training step."""
        encoder_ids, decoder_ids, decoder_labels, encoder_mask = batch_data
        loss, toks, grad_accum = step_fns["seq2seq"](
            encoder_ids, decoder_ids, decoder_labels, encoder_mask,
            grad_accum, do_update)
        return loss, toks, grad_accum

    def evaluate(self) -> float:
        """Run seq2seq evaluation on the validation set."""
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        max_batches = self.config.training.val_batches

        for encoder_ids, decoder_ids, decoder_labels, encoder_mask in iterate_seq2seq_batches(
            self.val_dataset, self.config
        ):
            loss, ntoks = self.loss(
                self.model, encoder_ids, decoder_ids, decoder_labels, encoder_mask)
            mx.eval(loss, ntoks)

            ntoks_val = ntoks.item()
            total_loss += loss.item() * ntoks_val
            total_tokens += ntoks_val
            num_batches += 1
            if num_batches >= max_batches:
                break

        return total_loss / total_tokens if total_tokens > 0 else float("inf")
