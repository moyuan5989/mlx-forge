"""KTO Trainer for MLX Forge.

Kahneman-Tversky Optimization for unpaired preference data.
Each sample has text + binary label (desirable/undesirable).
"""

from __future__ import annotations

import itertools

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.data.batching import iterate_batches
from mlx_forge.losses.sft import SFTLoss
from mlx_forge.trainer.trainer import BaseTrainer


class KTOTrainer(BaseTrainer):
    """KTO trainer — unpaired preference optimization."""

    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.beta = config.training.kto_beta
        self._sft_loss = SFTLoss()

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build compiled KTO step functions."""
        from mlx_forge.losses.preference import kto_loss

        def _kto_loss_fn(model, input_ids, lengths, labels):
            return kto_loss(model, input_ids, lengths, labels, beta=self.beta)

        loss_value_and_grad = nn.value_and_grad(self.model, _kto_loss_fn)

        # KTO uses custom step due to different batch format
        return {"kto_grad": loss_value_and_grad}

    def _build_batch_iterator(self):
        """Build the KTO batch iterator for unpaired preference data."""
        return itertools.cycle(self._iterate_kto_batches())

    def _iterate_kto_batches(self):
        """Yield (input_ids, lengths, labels) tuples for KTO training."""
        import numpy as np
        batch_size = self.config.training.batch_size
        max_seq_length = self.config.data.max_seq_length

        samples = list(self.train_dataset)

        for batch_start in range(0, len(samples), batch_size):
            batch_samples = samples[batch_start:batch_start + batch_size]
            if not batch_samples:
                continue

            # Find max length
            max_len = max(len(s["input_ids"]) for s in batch_samples)
            padded_length = min(((max_len + 31) // 32) * 32, max_seq_length)

            batch_input_ids = np.zeros((batch_size, padded_length), dtype=np.int32)
            batch_lengths = np.zeros(batch_size, dtype=np.float32)
            batch_labels = np.zeros(batch_size, dtype=np.float32)

            for i, sample in enumerate(batch_samples):
                ids = sample["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                tlen = min(len(ids), max_seq_length)
                batch_input_ids[i, :tlen] = ids[:tlen]
                batch_lengths[i] = tlen
                batch_labels[i] = float(sample.get("label", 1))

            yield (
                mx.array(batch_input_ids, dtype=mx.int32),
                mx.array(batch_lengths, dtype=mx.float32),
                mx.array(batch_labels, dtype=mx.float32),
            )

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one KTO training step."""
        input_ids, lengths, labels = batch_data

        loss_value_and_grad = step_fns["kto_grad"]
        (loss, ntoks), grads = loss_value_and_grad(
            self.model, input_ids, lengths, labels)

        if do_update:
            max_grad_norm = self.config.training.max_grad_norm
            if max_grad_norm:
                from mlx_forge.trainer.trainer import clip_grad_norm
                grads = clip_grad_norm(grads, max_grad_norm)
            self.optimizer.update(self.model, grads)

        return loss, ntoks, None

    def evaluate(self) -> float:
        """Run evaluation on validation set (SFT loss)."""
        total_loss = 0.0
        total_tokens = 0

        for input_ids, labels in iterate_batches(self.val_dataset, self.config):
            loss, ntoks = self._sft_loss(self.model, input_ids, labels)
            mx.eval(loss, ntoks)
            ntoks_val = ntoks.item()
            total_loss += loss.item() * ntoks_val
            total_tokens += ntoks_val

        return total_loss / total_tokens if total_tokens > 0 else float("inf")
