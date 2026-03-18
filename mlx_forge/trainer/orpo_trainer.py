"""ORPO Trainer for MLX Forge.

Odds Ratio Preference Optimization: SFT + odds ratio preference signal.
Same data format as DPO (paired chosen/rejected).
"""

from __future__ import annotations

import itertools
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.data.batching import iterate_batches, iterate_preference_batches
from mlx_forge.losses.sft import SFTLoss
from mlx_forge.trainer.trainer import BaseTrainer


class ORPOTrainer(BaseTrainer):
    """ORPO trainer — SFT + odds ratio preference optimization."""

    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.beta = config.training.orpo_beta
        self._sft_loss = SFTLoss()

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build compiled ORPO step functions."""
        from mlx_forge.losses.preference import orpo_loss

        def _orpo_loss_fn(model, chosen_ids, chosen_labels,
                          rejected_ids, rejected_labels):
            # Compute lengths from labels (count non -100 tokens)
            chosen_lengths = (chosen_labels != -100).sum(axis=1).astype(mx.float32)
            rejected_lengths = (rejected_labels != -100).sum(axis=1).astype(mx.float32)
            return orpo_loss(
                model, chosen_ids, rejected_ids,
                chosen_lengths, rejected_lengths,
                beta=self.beta,
            )

        loss_value_and_grad = nn.value_and_grad(self.model, _orpo_loss_fn)

        if not self.config.runtime.eager:
            @partial(mx.compile, inputs=compile_state, outputs=compile_state)
            def step(chosen_ids, chosen_labels, rejected_ids, rejected_labels,
                     prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, chosen_ids, chosen_labels,
                    rejected_ids, rejected_labels)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad
        else:
            def step(chosen_ids, chosen_labels, rejected_ids, rejected_labels,
                     prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, chosen_ids, chosen_labels,
                    rejected_ids, rejected_labels)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

        return {"orpo": step}

    def _build_batch_iterator(self):
        """Build the preference pair batch iterator."""
        return itertools.cycle(
            iterate_preference_batches(self.train_dataset, self.config))

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one ORPO training step."""
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = batch_data
        loss, toks, grad_accum = step_fns["orpo"](
            chosen_ids, chosen_labels, rejected_ids, rejected_labels,
            grad_accum, do_update)
        return loss, toks, grad_accum

    def evaluate(self) -> float:
        """Run evaluation on the validation set."""
        if self.val_dataset and "chosen_input_ids" in self.val_dataset[0]:
            return self._evaluate_preference()
        return self._evaluate_sft()

    def _evaluate_preference(self) -> float:
        """Evaluate using ORPO loss on preference pairs."""
        from mlx_forge.losses.preference import orpo_loss

        total_loss = 0.0
        total_tokens = 0

        for batch_data in iterate_preference_batches(self.val_dataset, self.config):
            chosen_ids, chosen_labels, rejected_ids, rejected_labels = batch_data
            chosen_lengths = (chosen_labels != -100).sum(axis=1).astype(mx.float32)
            rejected_lengths = (rejected_labels != -100).sum(axis=1).astype(mx.float32)
            loss, ntoks = orpo_loss(
                self.model, chosen_ids, rejected_ids,
                chosen_lengths, rejected_lengths, beta=self.beta)
            mx.eval(loss, ntoks)
            ntoks_val = ntoks.item()
            total_loss += loss.item() * ntoks_val
            total_tokens += ntoks_val

        return total_loss / total_tokens if total_tokens > 0 else float("inf")

    def _evaluate_sft(self) -> float:
        """Fall back to SFT evaluation."""
        total_loss = 0.0
        total_tokens = 0

        for input_ids, labels in iterate_batches(self.val_dataset, self.config):
            loss, ntoks = self._sft_loss(self.model, input_ids, labels)
            mx.eval(loss, ntoks)
            ntoks_val = ntoks.item()
            total_loss += loss.item() * ntoks_val
            total_tokens += ntoks_val

        return total_loss / total_tokens if total_tokens > 0 else float("inf")
