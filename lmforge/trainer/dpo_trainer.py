"""DPO Trainer for LMForge V2.

Extends BaseTrainer for Direct Preference Optimization training
with chosen/rejected preference pairs using per-token labels.
"""

from __future__ import annotations

import itertools
from functools import partial

import mlx.core as mx
import mlx.nn as nn

from lmforge.data.batching import iterate_batches, iterate_preference_batches
from lmforge.losses.dpo import DPOLoss
from lmforge.losses.sft import SFTLoss
from lmforge.trainer.trainer import BaseTrainer


class DPOTrainer(BaseTrainer):
    """DPO trainer — Direct Preference Optimization.

    Supports both standard DPO and reference-free DPO (SimPO).
    """

    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.loss = DPOLoss(
            beta=config.training.dpo_beta,
            reference_free=config.training.dpo_reference_free,
        )
        # SFT loss for evaluation (val set may be standard format)
        self._sft_loss = SFTLoss()

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build compiled DPO step functions."""
        loss_value_and_grad = nn.value_and_grad(self.model, self.loss)

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

        return {"dpo": step}

    def _build_batch_iterator(self):
        """Build the DPO preference pair batch iterator."""
        return itertools.cycle(
            iterate_preference_batches(self.train_dataset, self.config))

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one DPO training step."""
        chosen_ids, chosen_labels, rejected_ids, rejected_labels = batch_data
        loss, toks, grad_accum = step_fns["dpo"](
            chosen_ids, chosen_labels, rejected_ids, rejected_labels,
            grad_accum, do_update)
        return loss, toks, grad_accum

    def evaluate(self) -> float:
        """Run DPO evaluation on the validation set.

        If val set is preference format, compute DPO loss.
        Otherwise fall back to SFT cross-entropy loss.
        """
        # Check if validation data is preference format
        if self.val_dataset and "chosen_input_ids" in self.val_dataset[0]:
            return self._evaluate_preference()
        else:
            return self._evaluate_sft()

    def _evaluate_preference(self) -> float:
        """Evaluate using DPO loss on preference pairs."""
        total_loss = 0.0
        total_tokens = 0

        for batch_data in iterate_preference_batches(self.val_dataset, self.config):
            chosen_ids, chosen_labels, rejected_ids, rejected_labels = batch_data
            loss, ntoks = self.loss(
                self.model, chosen_ids, chosen_labels,
                rejected_ids, rejected_labels)
            mx.eval(loss, ntoks)

            total_loss += loss.item() * ntoks.item()
            total_tokens += ntoks.item()

        return total_loss / total_tokens if total_tokens > 0 else float("inf")

    def _evaluate_sft(self) -> float:
        """Fall back to SFT evaluation if val set is not preference format."""
        total_loss = 0.0
        total_tokens = 0

        for input_ids, labels in iterate_batches(self.val_dataset, self.config):
            loss, ntoks = self._sft_loss(self.model, input_ids, labels)
            mx.eval(loss, ntoks)

            total_loss += loss.item() * ntoks.item()
            total_tokens += ntoks.item()

        return total_loss / total_tokens if total_tokens > 0 else float("inf")
