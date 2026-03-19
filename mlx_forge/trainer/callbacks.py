"""Callback system for MLX Forge v0.

Callbacks execute OUTSIDE the compiled region, after mx.eval() safe points.
See V0_DESIGN_FREEZE.md §5 for callback boundary semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mlx_forge.trainer.state import TrainState


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, state: TrainState) -> None:
        pass

    def on_train_end(self, state: TrainState) -> None:
        pass

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        pass

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        pass

    def on_save(self, state: TrainState, checkpoint_dir: Path) -> None:
        pass


class CallbackList:
    """Container that dispatches callback events to a list of callbacks."""

    def __init__(self, callbacks: Optional[list[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        for cb in self.callbacks:
            cb.on_step_end(state, metrics)

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        for cb in self.callbacks:
            cb.on_eval_end(state, metrics)

    def on_save(self, state: TrainState, checkpoint_dir: Path) -> None:
        for cb in self.callbacks:
            cb.on_save(state, checkpoint_dir)


class MetricsLoggerCallback(Callback):
    """Writes JSONL metrics to logs/metrics.jsonl."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        """Write training metrics to JSONL."""
        from mlx_forge.logging.metrics import write_metrics_line

        metrics_with_event = {"event": "train", **metrics}
        write_metrics_line(self.log_path, metrics_with_event)

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        """Write evaluation metrics to JSONL."""
        from mlx_forge.logging.metrics import write_metrics_line

        metrics_with_event = {
            "event": "eval",
            "step": state.step,
            **metrics,
        }
        write_metrics_line(self.log_path, metrics_with_event)


class ConsoleCallback(Callback):
    """Prints human-readable training progress to stdout."""

    def __init__(self, num_iters: int):
        self.num_iters = num_iters

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        """Print training metrics to console."""
        from mlx_forge.logging.metrics import format_console_line

        line = format_console_line(metrics, self.num_iters)
        print(line)

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        """Print evaluation metrics to console."""
        from mlx_forge.logging.metrics import format_console_line

        line = format_console_line({**metrics, "step": state.step, "event": "eval"}, self.num_iters)
        print(line)

    def on_save(self, state: TrainState, checkpoint_dir: Path) -> None:
        """Print checkpoint save notification."""
        print(f"Step {state.step}/{self.num_iters} | Saved checkpoint to {checkpoint_dir}")


class HeartbeatCallback(Callback):
    """Touches a .heartbeat file in the run directory on every step.

    This allows the Studio to reliably detect that training is alive
    even when metrics aren't written (during compilation, eval, etc.).
    """

    def __init__(self, run_dir: Path):
        self.heartbeat_path = Path(run_dir) / ".heartbeat"

    def on_train_begin(self, state: TrainState) -> None:
        self._touch()

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        self._touch()

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        self._touch()

    def on_train_end(self, state: TrainState) -> None:
        # Remove heartbeat on clean exit so status transitions to completed/stopped
        try:
            self.heartbeat_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _touch(self):
        try:
            self.heartbeat_path.touch()
        except OSError:
            pass


class WandBCallback(Callback):
    """Optional Weights & Biases integration (try/except import)."""

    def __init__(self, project: str, run_name: str, config: dict):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )

        self.wandb = wandb
        self.run = None
        self.project = project
        self.run_name = run_name
        self.config = config

    def on_train_begin(self, state: TrainState) -> None:
        """Initialize wandb run."""
        self.run = self.wandb.init(
            project=self.project,
            name=self.run_name,
            config=self.config,
        )

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        """Log training metrics to wandb."""
        if self.run is not None:
            self.wandb.log(metrics, step=state.step)

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        """Log evaluation metrics to wandb."""
        if self.run is not None:
            self.wandb.log(metrics, step=state.step)

    def on_train_end(self, state: TrainState) -> None:
        """Finish wandb run."""
        if self.run is not None:
            self.run.finish()
