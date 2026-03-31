"""MLM Trainer for MLX Forge.

Extends BaseTrainer for Masked Language Modeling training
on encoder models (BERT, RoBERTa, DeBERTa).
"""

from __future__ import annotations

import itertools
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.data.batching import iterate_mlm_batches
from mlx_forge.losses.mlm import MLMLoss
from mlx_forge.trainer.trainer import BaseTrainer


class MLMTrainer(BaseTrainer):
    """MLM trainer — Masked Language Modeling for encoder models."""

    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.loss = MLMLoss()

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build compiled MLM step functions."""
        loss_value_and_grad = nn.value_and_grad(self.model, self.loss)

        if not self.config.runtime.eager:
            @partial(mx.compile, inputs=compile_state, outputs=compile_state)
            def step(input_ids, labels, attention_mask, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, input_ids, labels, attention_mask)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad
        else:
            def step(input_ids, labels, attention_mask, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, input_ids, labels, attention_mask)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

        return {"mlm": step}

    def _build_batch_iterator(self):
        """Build the MLM batch iterator."""
        return itertools.cycle(
            iterate_mlm_batches(self.train_dataset, self.config))

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one MLM training step."""
        input_ids, labels, attention_mask = batch_data
        loss, toks, grad_accum = step_fns["mlm"](
            input_ids, labels, attention_mask, grad_accum, do_update)
        return loss, toks, grad_accum

    def evaluate(self) -> float:
        """Run MLM evaluation on the validation set."""
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        max_batches = self.config.training.val_batches

        for input_ids, labels, attention_mask in iterate_mlm_batches(
            self.val_dataset, self.config
        ):
            loss, ntoks = self.loss(self.model, input_ids, labels, attention_mask)
            mx.eval(loss, ntoks)

            ntoks_val = ntoks.item()
            total_loss += loss.item() * ntoks_val
            total_tokens += ntoks_val
            num_batches += 1
            if num_batches >= max_batches:
                break

        return total_loss / total_tokens if total_tokens > 0 else float("inf")
