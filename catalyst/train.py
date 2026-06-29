"""Training loop and startup runtime estimation for Catalyst.

This module owns the startup runtime estimate the Trainer prints before
training begins (Requirement 9.4, 9.5, 12.1, 12.2): it times a single
forward-plus-backward step, projects the total time for all five experiments,
prints both, and warns when the projection exceeds the 45-minute budget.

The estimate is split into small, independently testable pieces. ``timed_step``
performs the measurement, ``estimate_runtime`` is a pure projection, and
``runtime_warning`` is the pure warning predicate (warns if and only if the
projection exceeds the threshold). ``report_runtime`` is the only piece that
prints, composing the other three, so the projection and warning logic stay
free of side effects for unit and property testing.
"""

import itertools
import math
import os
import time

import psutil
import torch

from .data import make_batches
from .model import AdapterManager

# The runtime budget from Requirement 9.5 / 12.2 expressed in seconds. The
# warning fires strictly above this threshold, so a projection of exactly
# 45 minutes does not warn while anything larger does.
WARNING_THRESHOLD_SECONDS = 45 * 60

# Where the trainer writes artifacts (Req 12.4, 12.5). Checkpoints are full
# model snapshots; results hold the reloadable adapter-only directories.
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# Across the five experiments only experiments 2-5 train; experiment 1 merely
# measures parameter counts and runs no training step. The trained variants
# are: three ranks (experiment 2), three target-module sets (experiment 3),
# the positive and negative named adapters (experiment 4), and three per-layer
# rank schedules (experiment 5). That is eleven training runs, each running
# ``max_steps`` steps, which is how the projected total step count is derived
# from the configuration.
TRAINING_RUNS = 11


def timed_step(model, batch) -> float:
    """Run one forward-plus-backward step and return its wall time in seconds.

    The batch carries ``input_ids``, ``attention_mask``, and ``labels`` so the
    causal-LM model returns a loss directly. Gradients are cleared before and
    after so the timing step leaves no accumulated gradient behind and measures
    only the single step's forward and backward passes.
    """
    model.train()

    # Clear any pre-existing gradients so the backward pass below times a clean
    # accumulation rather than adding onto stale state.
    for param in model.parameters():
        param.grad = None

    start = time.perf_counter()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    elapsed = time.perf_counter() - start

    # Drop the gradients the timing step produced so it has no lasting effect
    # on the model the real training loop will use.
    for param in model.parameters():
        param.grad = None

    return elapsed


def estimate_runtime(step_seconds: float, total_steps: int) -> float:
    """Project total runtime as ``step_seconds * total_steps`` (seconds).

    Kept pure (no printing) so the projection is unit-testable. ``total_steps``
    is supplied by the caller; ``total_experiment_steps`` derives it from the
    configuration for the all-five-experiments projection.
    """
    return step_seconds * total_steps


def total_experiment_steps(config: dict) -> int:
    """Total training steps projected across all five experiments.

    Derived from ``max_steps`` in the configuration multiplied by the number of
    trained variants across experiments 2-5 (see ``TRAINING_RUNS``).
    """
    max_steps = int((config or {}).get("max_steps", 0))
    return TRAINING_RUNS * max_steps


def runtime_warning(
    projected_seconds: float,
    threshold_seconds: float = WARNING_THRESHOLD_SECONDS,
) -> str:
    """Return a warning string when the projection exceeds the threshold.

    Returns ``None`` when the projected runtime is within budget. The condition
    is strictly greater than the threshold, so the warning fires if and only if
    the projection exceeds the 45-minute budget (Property 12).
    """
    if projected_seconds > threshold_seconds:
        projected_minutes = projected_seconds / 60.0
        budget_minutes = threshold_seconds / 60.0
        return (
            f"Projected runtime {projected_minutes:.1f} min exceeds the "
            f"{budget_minutes:.0f}-minute budget; consider reducing max_steps "
            f"or switching to a smaller model."
        )
    return None


def report_runtime(
    step_seconds: float,
    total_steps: int,
    threshold_seconds: float = WARNING_THRESHOLD_SECONDS,
) -> bool:
    """Print the measured step time and projected total, warning if over budget.

    Composes the pure pieces: projects with ``estimate_runtime``, prints the
    measured step time and projection, then prints the ``runtime_warning`` when
    one applies. Returns whether a warning was emitted so callers (and tests)
    can observe the threshold decision without parsing printed output.
    """
    projected = estimate_runtime(step_seconds, total_steps)
    print(f"Measured step time: {step_seconds:.4f} s")
    print(
        f"Projected total runtime for all five experiments: "
        f"{projected / 60.0:.1f} min ({projected:.1f} s) over "
        f"{total_steps} steps."
    )
    warning = runtime_warning(projected, threshold_seconds)
    if warning is not None:
        print(warning)
    return warning is not None


def perplexity(loss_value: float) -> float:
    """Perplexity is ``exp(loss)``; an overflowing loss reports infinity."""
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")


def trainable_percent(trainable: int, total: int) -> float:
    """Percentage of parameters that are trainable: ``trainable / total * 100``.

    Pure helper shared with Experiment 1's parameter-count reporting (Req 14).
    Returns ``0.0`` when ``total`` is zero rather than dividing by zero, so a
    model with no parameters reports a well-defined (empty) percentage.
    """
    if total == 0:
        return 0.0
    return trainable / total * 100


def _batch_stream(data, config, max_steps):
    """Yield up to ``max_steps`` batches, cycling the source as needed.

    ``data`` is either a ``(dataset, tokenizer)`` pair the trainer batches
    itself with the configured batch size, or an already-prepared iterable of
    batches. Either source is restarted until ``max_steps`` batches have been
    produced so a short dataset still fills the requested step count.
    """
    if isinstance(data, tuple) and len(data) == 2:
        dataset, tokenizer = data
        batch_size = int((config or {}).get("batch_size", 8))

        def restart():
            return make_batches(dataset, tokenizer, batch_size)

    else:
        # An iterator cannot be replayed, so materialize once and re-iterate
        # the cached batches when cycling.
        cached = list(data)

        def restart():
            return iter(cached)

    produced = 0
    while produced < max_steps:
        empty = True
        for batch in restart():
            empty = False
            yield batch
            produced += 1
            if produced >= max_steps:
                return
        if empty:
            raise ValueError("training data produced no batches.")


def save_checkpoint(model, step: int) -> str:
    """Write a full model checkpoint into ``checkpoints/`` (Req 12.4)."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{step}.pt")
    torch.save(model.state_dict(), path)
    return path


def _save_adapters(model, config) -> str:
    """Save adapter-only artifacts into ``results/`` via ``save_adapter``.

    Reuses ``AdapterManager.save_adapter_set`` so each adapted module is written
    under its module-path stem, leaving the directory reloadable by experiments
    (Req 12.5).
    """
    config = config or {}
    results_dir = config.get("results_dir", RESULTS_DIR)
    name = config.get("adapter_name", "adapter")
    manager = AdapterManager(model)
    return manager.save_adapter_set(name, os.path.join(results_dir, name))


def train(model, data, config) -> dict:
    """Train the adapters and persist a checkpoint and adapter-only artifacts.

    Runs one ``timed_step`` and prints the runtime estimate before the loop
    (Req 12.1, 12.2), then optimizes only the trainable adapter parameters for
    ``max_steps`` steps, logging loss, perplexity, throughput, and RAM each step
    (Req 12.3). ``data`` is a ``(dataset, tokenizer)`` pair or a prepared
    iterable of batches. Returns a summary of the run.
    """
    config = config or {}
    max_steps = int(config.get("max_steps", 0))

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(config.get("learning_rate", 5e-4)),
    )

    batches = _batch_stream(data, config, max_steps)
    try:
        first_batch = next(batches)
    except StopIteration:
        raise ValueError("training data produced no batches.")

    # Time one step and project the total before committing to training so an
    # over-budget run is flagged up front (Req 12.1, 12.2). timed_step clears
    # its own gradients, so first_batch is still trained on below.
    step_seconds = timed_step(model, first_batch)
    report_runtime(step_seconds, total_experiment_steps(config))

    process = psutil.Process()
    model.train()

    step_logs = []
    peak_ram_mb = 0.0
    ram_sum = 0.0
    last_loss = float("nan")

    for step, batch in enumerate(itertools.chain([first_batch], batches), start=1):
        start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        step_time = time.perf_counter() - start

        last_loss = float(loss.detach())
        batch_size = int(batch["input_ids"].shape[0])
        step_perplexity = perplexity(last_loss)
        throughput = batch_size / step_time if step_time > 0 else float("inf")

        # psutil RSS sampling stands in for CUDA memory stats on CPU; track the
        # running peak and average across steps.
        ram_mb = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = max(peak_ram_mb, ram_mb)
        ram_sum += ram_mb
        avg_ram_mb = ram_sum / step

        log = {
            "step": step,
            "loss": last_loss,
            "perplexity": step_perplexity,
            "throughput_samples_per_sec": throughput,
            "peak_ram_mb": peak_ram_mb,
            "avg_ram_mb": avg_ram_mb,
        }
        step_logs.append(log)
        print(
            f"step {step}/{max_steps} loss={last_loss:.4f} "
            f"ppl={step_perplexity:.4f} "
            f"throughput={throughput:.2f} samples/s "
            f"peak_ram={peak_ram_mb:.1f} MB avg_ram={avg_ram_mb:.1f} MB"
        )

    steps_run = len(step_logs)
    checkpoint_path = save_checkpoint(model, steps_run)
    adapter_dir = _save_adapters(model, config)

    avg_loss = (
        sum(entry["loss"] for entry in step_logs) / steps_run
        if steps_run
        else float("nan")
    )
    avg_throughput = (
        sum(entry["throughput_samples_per_sec"] for entry in step_logs) / steps_run
        if steps_run
        else 0.0
    )

    return {
        "steps": steps_run,
        "final_loss": last_loss,
        "avg_loss": avg_loss,
        "final_perplexity": perplexity(last_loss),
        "avg_throughput_samples_per_sec": avg_throughput,
        "peak_ram_mb": peak_ram_mb,
        "avg_ram_mb": ram_sum / steps_run if steps_run else 0.0,
        "checkpoint_path": checkpoint_path,
        "adapter_dir": adapter_dir,
        "step_logs": step_logs,
    }
