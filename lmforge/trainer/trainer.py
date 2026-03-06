"""Trainer classes for LMForge.

V2 multi-loss architecture with per-token labels:
- BaseTrainer: owns training loop, checkpointing, callbacks, evaluation
- SFTTrainer(BaseTrainer): standard supervised fine-tuning
- Trainer: alias for SFTTrainer (backward compatibility)
"""

from __future__ import annotations

import itertools
import time
from functools import partial
from threading import Event

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from lmforge.data.batching import iterate_batches, iterate_packed_batches
from lmforge.losses.sft import SFTLoss
from lmforge.trainer.callbacks import CallbackList
from lmforge.trainer.checkpoint import CheckpointManager
from lmforge.trainer.optimizer import build_optimizer
from lmforge.trainer.state import TrainState


def clip_grad_norm(grads, max_norm: float):
    """Clip gradients by global norm."""
    grad_norm = mx.sqrt(sum((g * g).sum() for _, g in tree_flatten(grads)))
    scale = max_norm / (grad_norm + 1e-6)
    scale = mx.minimum(scale, 1.0)
    return tree_map(lambda g: g * scale, grads)


class BaseTrainer:
    """Base trainer owning the training loop, checkpointing, and callbacks.

    Subclasses must implement:
    - _build_step_functions(): return step function dict
    - _build_batch_iterator(): return the batch iterator
    - _execute_step(step_fn, batch_data, grad_accum, do_update): run one step
    - evaluate(): run validation
    """

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

        self._pause_requested = Event()
        self._paused = Event()

    def _build_grad_update_fn(self):
        """Build the shared gradient accumulation and update logic."""
        grad_accum_steps = self.config.training.grad_accumulation_steps
        max_grad_norm = self.config.training.max_grad_norm

        def _apply_grad_update(grad, prev_grad, do_update):
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

        return _apply_grad_update

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build compiled step functions. Override in subclasses."""
        raise NotImplementedError

    def _build_batch_iterator(self):
        """Build the batch iterator. Override in subclasses."""
        raise NotImplementedError

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one training step. Override in subclasses.

        Returns:
            (loss, toks, grad_accum): loss scalar, token count, accumulated gradients.
        """
        raise NotImplementedError

    def evaluate(self) -> float:
        """Run evaluation. Override in subclasses."""
        raise NotImplementedError

    def fit(self) -> TrainState:
        """Run the full training loop. Returns final TrainState."""
        if mx.metal.is_available():
            device_info = mx.metal.device_info()
            if "max_recommended_working_set_size" in device_info:
                mx.set_wired_limit(device_info["max_recommended_working_set_size"])

        mx.random.seed(self.config.training.seed)
        self.callbacks.on_train_begin(self.state)

        grad_accum_steps = self.config.training.grad_accumulation_steps

        compile_state = [self.model.state, self.optimizer.state, mx.random.state]
        apply_grad_update = self._build_grad_update_fn()
        step_fns = self._build_step_functions(compile_state, apply_grad_update)
        batch_iterator = self._build_batch_iterator()

        # Training loop state
        grad_accum = None
        losses = 0.0
        n_tokens = 0
        steps_since_report = 0
        report_start_time = time.perf_counter()

        if hasattr(self.train_dataset, '__len__'):
            num_samples = len(self.train_dataset)
            batches_per_epoch = max(1, num_samples // self.config.training.batch_size)
        else:
            # Streaming/mixed dataset — epoch tracking is approximate
            num_samples = 0
            batches_per_epoch = self.config.training.num_iters

        start_step = self.state.step + 1

        # Skip batches for resumed training
        if start_step > 1:
            print(f"Resuming from step {start_step - 1}, skipping {start_step - 1} batches...")
            for _ in range(start_step - 1):
                next(batch_iterator)

        for it, batch_data in zip(
            range(start_step, self.config.training.num_iters + 1),
            batch_iterator,
        ):
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

            # Training step
            loss, toks, grad_accum = self._execute_step(
                step_fns, batch_data, grad_accum, do_update, compile_state)
            mx.eval(compile_state, loss, toks, grad_accum)

            toks_val = toks.item()
            losses += loss.item()
            n_tokens += toks_val
            steps_since_report += 1
            self.state.step = it
            self.state.trained_tokens += toks_val

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

            # Cooperative pause
            if self._pause_requested.is_set():
                self.checkpoint_manager.save(self.state, self.model, self.optimizer)
                self._paused.set()
                self._pause_requested.wait()

        self.callbacks.on_train_end(self.state)
        return self.state


class SFTTrainer(BaseTrainer):
    """SFT trainer — standard supervised fine-tuning with per-token labels."""

    def __init__(self, model, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.loss = SFTLoss()

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build compiled SFT step functions."""
        loss_value_and_grad = nn.value_and_grad(self.model, self.loss)
        loss_value_and_grad_packed = nn.value_and_grad(
            self.model, self.loss.packed)

        if not self.config.runtime.eager:
            @partial(mx.compile, inputs=compile_state, outputs=compile_state)
            def step(input_ids, labels, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, input_ids, labels)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

            @partial(mx.compile, inputs=compile_state, outputs=compile_state)
            def step_packed(input_ids, labels, segment_ids, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad_packed(
                    self.model, input_ids, labels, segment_ids)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad
        else:
            def step(input_ids, labels, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad(
                    self.model, input_ids, labels)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

            def step_packed(input_ids, labels, segment_ids, prev_grad, do_update):
                (loss, ntoks), grad = loss_value_and_grad_packed(
                    self.model, input_ids, labels, segment_ids)
                grad = apply_grad_update(grad, prev_grad, do_update)
                return loss, ntoks, grad

        return {"standard": step, "packed": step_packed}

    def _build_batch_iterator(self):
        """Build the SFT batch iterator (standard or packed)."""
        if self.config.data.packing:
            return itertools.cycle(
                iterate_packed_batches(self.train_dataset, self.config))
        else:
            return itertools.cycle(
                iterate_batches(self.train_dataset, self.config))

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one SFT training step."""
        if self.config.data.packing:
            input_ids, labels, segment_ids = batch_data
            loss, toks, grad_accum = step_fns["packed"](
                input_ids, labels, segment_ids, grad_accum, do_update)
        else:
            input_ids, labels = batch_data
            loss, toks, grad_accum = step_fns["standard"](
                input_ids, labels, grad_accum, do_update)
        return loss, toks, grad_accum

    def evaluate(self) -> float:
        """Run SFT evaluation on the validation set. Returns val_loss."""
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        max_batches = self.config.training.val_batches

        for input_ids, labels in iterate_batches(self.val_dataset, self.config):
            loss, ntoks = self.loss(self.model, input_ids, labels)
            mx.eval(loss, ntoks)

            ntoks_val = ntoks.item()
            total_loss += loss.item() * ntoks_val
            total_tokens += ntoks_val
            num_batches += 1
            if num_batches >= max_batches:
                break

        return total_loss / total_tokens if total_tokens > 0 else float("inf")


# Backward compatibility: Trainer is an alias for SFTTrainer
Trainer = SFTTrainer


# Backward-compatible module-level functions
def loss_fn(model, input_ids, labels):
    """Compute cross-entropy loss with label masking."""
    from lmforge.losses.sft import loss_fn as _loss_fn
    return _loss_fn(model, input_ids, labels)


def loss_fn_packed(model, input_ids, labels, segment_ids):
    """Compute cross-entropy loss for packed sequences."""
    from lmforge.losses.sft import loss_fn_packed as _loss_fn_packed
    return _loss_fn_packed(model, input_ids, labels, segment_ids)


def loss_value_and_grad(model, input_ids, labels):
    """Compute loss and gradients."""
    return nn.value_and_grad(model, loss_fn)(model, input_ids, labels)


def loss_value_and_grad_packed(model, input_ids, labels, segment_ids):
    """Compute loss and gradients for packed sequences."""
    return nn.value_and_grad(model, loss_fn_packed)(model, input_ids, labels, segment_ids)
