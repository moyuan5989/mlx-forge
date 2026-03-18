"""GRPO Trainer for MLX Forge.

Implements Group Relative Policy Optimization:
1. For each prompt, generate G completions
2. Score completions with reward function
3. Compute group-normalized advantages
4. Update LoRA params with clipped surrogate + KL penalty
"""

from __future__ import annotations

import itertools

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_forge.losses.grpo import GRPOLoss, compute_group_advantages
from mlx_forge.trainer.reward import get_reward_function
from mlx_forge.trainer.trainer import BaseTrainer


class GRPOTrainer(BaseTrainer):
    """GRPO trainer — RL-based fine-tuning with group relative policy optimization."""

    def __init__(self, model, tokenizer, config, train_dataset, val_dataset,
                 callbacks=None, state=None, checkpoint_manager=None):
        super().__init__(model, config, train_dataset, val_dataset,
                         callbacks, state, checkpoint_manager)
        self.tokenizer = tokenizer
        self.grpo_loss = GRPOLoss(
            beta=config.training.grpo_beta,
            clip_range=config.training.grpo_clip_range,
        )

        # GRPO params from config
        self.num_generations = config.training.grpo_num_generations
        self.max_completion_length = config.training.grpo_max_completion_length

        # Reward function
        self.reward_fn = get_reward_function(config.training.grpo_reward_function)

        # Save frozen copy of initial trainable parameters for reference policy
        self._ref_weights = {
            k: mx.array(v) for k, v in
            dict(tree_flatten(model.trainable_parameters())).items()
        }

    def _compute_ref_log_probs(self, input_ids, labels):
        """Swap in frozen weights, compute log probs, swap back."""
        current = {
            k: v for k, v in
            dict(tree_flatten(self.model.trainable_parameters())).items()
        }
        self.model.load_weights(list(self._ref_weights.items()), strict=False)
        mx.eval(self.model.parameters())

        logits = self.model(input_ids)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        ref_lps = mx.take_along_axis(
            log_probs[0], labels[0, :, None], axis=-1
        ).squeeze(-1)

        self.model.load_weights(list(current.items()), strict=False)
        mx.eval(self.model.parameters())
        return mx.stop_gradient(ref_lps)

    def _build_step_functions(self, compile_state, apply_grad_update):
        """Build GRPO step function."""
        # GRPO uses custom step logic, return empty dict
        return {}

    def _build_batch_iterator(self):
        """Build iterator that yields prompts."""
        # For GRPO, we iterate over prompts
        def _prompt_iterator():
            for sample in itertools.cycle(self.train_dataset):
                if isinstance(sample, dict):
                    if "messages" in sample:
                        # Extract user message as prompt
                        for msg in sample["messages"]:
                            if msg.get("role") == "user":
                                yield msg["content"]
                                break
                    elif "text" in sample:
                        yield sample["text"]
                    elif "prompt" in sample:
                        yield sample["prompt"]
                else:
                    yield str(sample)
        return _prompt_iterator()

    def _execute_step(self, step_fns, batch_data, grad_accum, do_update, compile_state):
        """Execute one GRPO training step.

        For each prompt:
        1. Generate G completions
        2. Score with reward function
        3. Compute advantages
        4. Compute GRPO loss and gradients
        5. Update parameters
        """
        prompt = batch_data

        # Generate completions
        prompt_tokens = self.tokenizer.encode(prompt)

        completions = []
        completion_tokens_list = []
        for _ in range(self.num_generations):
            tokens = list(prompt_tokens)
            for token_id in self._generate_tokens(tokens):
                tokens.append(token_id)
                if len(tokens) - len(prompt_tokens) >= self.max_completion_length:
                    break
            completion_tokens_list.append(tokens)
            completions.append(self.tokenizer.decode(tokens[len(prompt_tokens):]))

        # Score completions
        rewards = mx.array([self.reward_fn(prompt, c) for c in completions])

        # Compute group advantages
        advantages = compute_group_advantages(rewards)

        # Compute loss and gradients
        def grpo_loss_fn(model):
            total_loss = mx.array(0.0)
            for i, tokens in enumerate(completion_tokens_list):
                input_ids = mx.array(tokens[:-1])[None]  # (1, T)
                labels = mx.array(tokens[1:])[None]       # (1, T)

                # Current policy log probs
                logits = model(input_ids)
                log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

                # Gather token log probs
                token_lps = mx.take_along_axis(
                    log_probs[0], labels[0, :, None], axis=-1
                ).squeeze(-1)

                # Reference log probs from frozen reference model
                ref_lps = self._compute_ref_log_probs(input_ids, labels)

                # Per-token GRPO loss
                ratio = mx.exp(token_lps - ref_lps)
                adv = advantages[i]
                clipped = mx.clip(ratio, 1 - self.grpo_loss.clip_range, 1 + self.grpo_loss.clip_range)
                surrogate = mx.minimum(ratio * adv, clipped * adv)
                kl = token_lps - ref_lps
                loss_i = (-surrogate + self.grpo_loss.beta * kl).mean()
                total_loss = total_loss + loss_i

            return total_loss / len(completion_tokens_list), mx.array(sum(len(t) for t in completion_tokens_list))

        loss_and_grad = nn.value_and_grad(self.model, grpo_loss_fn)
        (loss, toks), grads = loss_and_grad(self.model)

        # Apply gradient update
        if do_update:
            max_grad_norm = self.config.training.max_grad_norm
            if max_grad_norm:
                from mlx_forge.trainer.trainer import clip_grad_norm
                grads = clip_grad_norm(grads, max_grad_norm)
            self.optimizer.update(self.model, grads)

        return loss, toks, None

    def _generate_tokens(self, prompt_tokens: list[int]):
        """Generate tokens from the model (simplified for GRPO)."""
        from mlx_forge.inference.sampling import sample_next_token

        tokens = mx.array(prompt_tokens)[None]  # (1, T)
        logits = self.model(tokens)

        next_token = sample_next_token(
            logits[0, -1, :],
            temperature=0.8,
            top_p=0.9,
        )
        mx.eval(next_token)

        for _ in range(self.max_completion_length):
            token_id = next_token.item()
            if token_id == self.tokenizer.eos_token_id:
                return
            yield token_id

            next_input = next_token.reshape(1, 1)
            logits = self.model(next_input)
            next_token = sample_next_token(
                logits[0, -1, :],
                temperature=0.8,
                top_p=0.9,
            )
            mx.eval(next_token)

    def evaluate(self) -> float:
        """Evaluate by computing average reward on validation prompts."""
        total_reward = 0.0
        count = 0
        max_eval = min(self.config.training.val_batches, 10)

        for sample in self.val_dataset:
            if count >= max_eval:
                break

            if isinstance(sample, dict):
                if "messages" in sample:
                    prompt = next((m["content"] for m in sample["messages"] if m.get("role") == "user"), "")
                elif "prompt" in sample:
                    prompt = sample["prompt"]
                else:
                    continue
            else:
                prompt = str(sample)

            if not prompt:
                continue

            # Generate one completion and score
            prompt_tokens = self.tokenizer.encode(prompt)
            tokens = list(prompt_tokens)
            for token_id in self._generate_tokens(tokens):
                tokens.append(token_id)
                if len(tokens) - len(prompt_tokens) >= self.max_completion_length:
                    break

            completion = self.tokenizer.decode(tokens[len(prompt_tokens):])
            reward = self.reward_fn(prompt, completion)
            total_reward += reward
            count += 1

        return -total_reward / max(count, 1)  # Negative because lower is "better" in val_loss convention
