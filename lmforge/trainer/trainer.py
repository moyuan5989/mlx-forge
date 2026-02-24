"""Trainer class for LMForge v0."""

from __future__ import annotations

import itertools
import time
from functools import partial
from threading import Event

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from lmforge.data.batching import iterate_batches, iterate_packed_batches
from lmforge.trainer.callbacks import CallbackList
from lmforge.trainer.checkpoint import CheckpointManager
from lmforge.trainer.optimizer import build_optimizer
from lmforge.trainer.state import TrainState


def loss_fn(model, batch, lengths):
    """Compute cross-entropy loss with prompt masking.

    Args:
        model: The model to evaluate
        batch: Token IDs array of shape (B, T)
        lengths: Array of shape (B, 2) where [:, 0] is prompt offset and [:, 1] is total length

    Returns:
        (loss, ntoks): Average loss per token and number of tokens used
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)

    # Create mask: compute loss only on tokens in [offset, length)
    # Note: steps is 1-indexed (starts at 1, not 0)
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = (steps >= lengths[:, 0:1]) & (steps < lengths[:, 1:2])

    # Cross-entropy with masking
    ce = nn.losses.cross_entropy(logits, targets, reduction="none") * mask
    ntoks = mask.sum()

    return ce.sum() / ntoks, ntoks


def loss_fn_packed(model, batch, segment_ids, offsets):
    """Compute cross-entropy loss for packed sequences.

    Loss is computed only on completion tokens within each segment,
    respecting segment boundaries so no cross-sequence loss leaks.

    Args:
        model: The model to evaluate
        batch: Token IDs array of shape (B, T)
        segment_ids: Segment membership per token, shape (B, T). -1 = padding
        offsets: Per-segment (prompt_end, seq_end), shape (B, max_segments, 2)
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    seg_in = segment_ids[:, :-1]
    seg_out = segment_ids[:, 1:]
    logits = model(inputs)

    # Loss only where consecutive tokens are same segment AND not padding
    same_seg = (seg_in == seg_out) & (seg_out >= 0)

    # Build prompt mask: for each position, check if it's past the prompt
    # offset for its segment. We build this from the offsets array.
    B, T_out = targets.shape
    positions = mx.broadcast_to(mx.arange(T_out)[None, :], (B, T_out))
    # For each position, get the prompt_end of its segment
    # seg_out contains the segment id; we look up offsets[b, seg_id, 0]
    # To avoid fancy indexing issues in MLX, compute per-segment masks
    n_segs = offsets.shape[1]
    past_prompt = mx.zeros((B, T_out), dtype=mx.bool_)
    for s in range(n_segs):
        seg_mask = (seg_out == s)
        prompt_end = offsets[:, s, 0:1]  # (B, 1)
        # Position is 0-indexed in the packed row; prompt_end is absolute position
        # We need: position+1 >= prompt_end (since targets are shifted by 1)
        past = (positions + 1) >= prompt_end
        past_prompt = past_prompt | (seg_mask & past)

    mask = same_seg & past_prompt

    ce = nn.losses.cross_entropy(logits, targets, reduction="none") * mask
    ntoks = mask.sum()

    return ce.sum() / ntoks, ntoks


def loss_value_and_grad(model, batch, lengths):
    """Compute loss and gradients.

    Uses nn.value_and_grad to only compute gradients for trainable parameters,
    which is required for QLoRA (avoids QuantizedMatmul VJP errors).
    """
    return nn.value_and_grad(model, loss_fn)(model, batch, lengths)


def loss_value_and_grad_packed(model, batch, segment_ids, offsets):
    """Compute loss and gradients for packed sequences.

    Uses nn.value_and_grad to only compute gradients for trainable parameters.
    """
    return nn.value_and_grad(model, loss_fn_packed)(model, batch, segment_ids, offsets)


def clip_grad_norm(grads, max_norm: float):
    """Clip gradients by global norm."""
    # Compute global norm using tree_flatten to iterate over leaf values
    grad_norm = mx.sqrt(sum((g * g).sum() for _, g in tree_flatten(grads)))

    # Clip if needed
    scale = max_norm / (grad_norm + 1e-6)
    scale = mx.minimum(scale, 1.0)

    return tree_map(lambda g: g * scale, grads)


class Trainer:
    """Runs LoRA SFT training with compiled step, callbacks, and checkpointing."""

    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = CallbackList(callbacks or [])

        self.optimizer = build_optimizer(config.training, model)
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(config)

        self.state = state or TrainState(
            step=0,
            epoch=0,
            trained_tokens=0,
            best_val_loss=float("inf"),
            rng_seed=config.training.seed,
        )

        # Cooperative pause support (optional)
        self._pause_requested = Event()
        self._paused = Event()

    def fit(self) -> TrainState:
        """Run the full training loop. Returns final TrainState."""
        # Set memory limit if on Metal
        if mx.metal.is_available():
            device_info = mx.metal.device_info()
            if "max_recommended_working_set_size" in device_info:
                mx.set_wired_limit(device_info["max_recommended_working_set_size"])

        # Seed RNG
        mx.random.seed(self.config.training.seed)

        # Callbacks
        self.callbacks.on_train_begin(self.state)

        # Build compiled step function
        grad_accum_steps = self.config.training.grad_accumulation_steps
        max_grad_norm = self.config.training.max_grad_norm
        use_packing = self.config.data.packing

        # compile_state tracks the mutable state for mx.compile
        compile_state = [self.model.state, self.optimizer.state, mx.random.state]

        def _apply_grad_update(grad, prev_grad, do_update):
            """Shared gradient accumulation and update logic."""
            if prev_grad is not None:
                grad = tree_map(lambda a, b: a + b, grad, prev_grad)
            if do_update:
                if grad_accum_steps > 1:
                    grad = tree_map(lambda g: g / grad_accum_steps, grad)
                if max_grad_norm is not None:
                    grad = clip_grad_norm(grad, max_grad_norm)
                self.optimizer.update(self.model, grad)
                grad = None
            return grad

        if not self.config.runtime.eager:
            @partial(mx.compile, inputs=compile_state, outputs=compile_state)
            def step(batch, lengths, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(self.model, batch, lengths)
                grad = _apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

            @partial(mx.compile, inputs=compile_state, outputs=compile_state)
            def step_packed(batch, segment_ids, offsets, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad_packed(
                    self.model, batch, segment_ids, offsets)
                grad = _apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad
        else:
            def step(batch, lengths, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(self.model, batch, lengths)
                grad = _apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

            def step_packed(batch, segment_ids, offsets, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad_packed(
                    self.model, batch, segment_ids, offsets)
                grad = _apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

        # Training loop state
        grad_accum = None
        losses = 0.0
        n_tokens = 0
        steps_since_report = 0
        report_start_time = time.perf_counter()

        # Calculate batches per epoch for epoch tracking
        num_samples = len(self.train_dataset)
        batches_per_epoch = max(1, num_samples // self.config.training.batch_size)

        # Determine start step (for resume support)
        start_step = self.state.step + 1

        # Main training loop - cycle infinitely through batches
        if use_packing:
            batch_iterator = itertools.cycle(
                iterate_packed_batches(self.train_dataset, self.config))
        else:
            batch_iterator = itertools.cycle(
                iterate_batches(self.train_dataset, self.config))

        # Skip batches for resumed training (Tier-1 resume: re-iterate from start)
        if start_step > 1:
            print(f"Resuming from step {start_step - 1}, skipping {start_step - 1} batches...")
            for _ in range(start_step - 1):
                next(batch_iterator)

        for it, batch_data in zip(
            range(start_step, self.config.training.num_iters + 1),
            batch_iterator,
        ):
            # Update epoch count
            self.state.epoch = (it - 1) // batches_per_epoch
            do_update = (it % grad_accum_steps == 0)

            # Evaluate at step 1, every steps_per_eval, and final step
            if (it == 1 or
                it % self.config.training.steps_per_eval == 0 or
                it == self.config.training.num_iters):
                val_loss = self.evaluate()
                self.callbacks.on_eval_end(self.state, {"val_loss": val_loss})
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss

            # Training step (packed or standard)
            if use_packing:
                batch, segment_ids, offsets = batch_data
                loss, toks, grad_accum = step_packed(
                    batch, segment_ids, offsets, grad_accum, do_update)
            else:
                batch, lengths = batch_data
                loss, toks, grad_accum = step(batch, lengths, grad_accum, do_update)
            mx.eval(compile_state, loss, toks, grad_accum)  # SAFE POINT

            losses += loss.item()
            n_tokens += toks.item()
            steps_since_report += 1
            self.state.step = it
            self.state.trained_tokens += toks.item()

            # Reporting
            if (it % self.config.training.steps_per_report == 0 or
                it == self.config.training.num_iters):
                elapsed = time.perf_counter() - report_start_time
                metrics = {
                    "step": it,
                    "train_loss": losses / steps_since_report,
                    "learning_rate": float(self.optimizer.learning_rate),
                    "tokens_per_second": n_tokens / elapsed if elapsed > 0 else 0.0,
                    "trained_tokens": self.state.trained_tokens,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
                self.callbacks.on_step_end(self.state, metrics)
                losses = 0.0
                n_tokens = 0
                steps_since_report = 0
                report_start_time = time.perf_counter()

            # Checkpointing
            if (it % self.config.training.steps_per_save == 0 or
                it == self.config.training.num_iters):
                ckpt_dir = self.checkpoint_manager.save(self.state, self.model, self.optimizer)
                self.callbacks.on_save(self.state, ckpt_dir)

            # Cooperative pause check (optional feature)
            if self._pause_requested.is_set():
                self.checkpoint_manager.save(self.state, self.model, self.optimizer)
                self._paused.set()
                self._pause_requested.wait()  # Block until resume

        self.callbacks.on_train_end(self.state)
        return self.state

    def evaluate(self) -> float:
        """Run evaluation on the validation set. Returns val_loss."""
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch, lengths in iterate_batches(self.val_dataset, self.config):
            loss, ntoks = loss_fn(self.model, batch, lengths)
            mx.eval(loss, ntoks)

            total_loss += loss.item() * ntoks.item()
            total_tokens += ntoks.item()
            num_batches += 1

        return total_loss / total_tokens if total_tokens > 0 else float("inf")
