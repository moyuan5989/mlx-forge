"""VLM SFT Trainer for vision-language model fine-tuning.

Key differences from text-only SFT:
- Batch size forced to 1 (variable image sizes → variable vision token counts)
- Forward pass includes pixel_values from image processor
- Loss masking: train on text completions only (mask vision tokens + prompt)
- Gradient accumulation compensates for batch_size=1
"""

from __future__ import annotations

import itertools

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.losses.sft import SFTLoss
from mlx_forge.trainer.trainer import BaseTrainer


class VLMSFTTrainer(BaseTrainer):
    """Vision-Language Model SFT trainer."""

    def __init__(self, model, processor, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.processor = processor
        self.loss = SFTLoss()
        from mlx_forge.vision.data import VisionDataCollator
        self.collator = VisionDataCollator(
            processor, max_seq_length=config.data.max_seq_length)

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build VLM step functions (not compiled due to variable input shapes)."""
        loss_value_and_grad = nn.value_and_grad(self.model, self.loss)
        return {"vlm_grad": loss_value_and_grad}

    def _build_batch_iterator(self):
        """Build iterator yielding VLM training samples (batch_size=1)."""
        def _vlm_iterator():
            for sample in itertools.cycle(self.train_dataset):
                processed = self.collator(sample)
                if processed is not None:
                    yield processed
        return _vlm_iterator()

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one VLM training step."""
        input_ids = mx.array([batch_data["input_ids"]], dtype=mx.int32)
        labels = mx.array([batch_data["labels"]], dtype=mx.int32)

        loss_value_and_grad = step_fns["vlm_grad"]
        (loss, ntoks), grads = loss_value_and_grad(
            self.model, input_ids, labels)

        if do_update:
            max_grad_norm = self.config.training.max_grad_norm
            if max_grad_norm:
                from mlx_forge.trainer.trainer import clip_grad_norm
                grads = clip_grad_norm(grads, max_grad_norm)
            self.optimizer.update(self.model, grads)

        return loss, ntoks, None

    def evaluate(self) -> float:
        """Run evaluation using SFT loss."""
        total_loss = 0.0
        total_tokens = 0
        count = 0
        max_eval = self.config.training.val_batches

        for sample in self.val_dataset:
            if count >= max_eval:
                break

            processed = self.collator(sample)
            if processed is None:
                continue

            input_ids = mx.array([processed["input_ids"]], dtype=mx.int32)
            labels = mx.array([processed["labels"]], dtype=mx.int32)

            loss, ntoks = self.loss(self.model, input_ids, labels)
            mx.eval(loss, ntoks)

            ntoks_val = ntoks.item()
            total_loss += loss.item() * ntoks_val
            total_tokens += ntoks_val
            count += 1

        return total_loss / total_tokens if total_tokens > 0 else float("inf")
