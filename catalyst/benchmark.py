"""Standalone inference benchmark for Catalyst.

Measures cold-start latency, warm-start latency, tokens per second, and adapter
swap time for the configured model, then writes a single
``results/benchmark.json``. This is a standalone artifact distinct from the
per-experiment JSONs written by ``experiments.py``.

Decoding stays deterministic and the warm-up-then-timed methodology is reused
from ``evaluate.py`` rather than reimplemented: warm-start latency and tokens
per second come straight from ``evaluate.measure_latency`` (Req 13.3, 13.4).
"""

import os
import tempfile
import time
from datetime import datetime, timezone

import torch

from .evaluate import generate_text, measure_latency
from .experiments import _write_json, load_config, set_seeds
from .model import AdapterManager, inject_lora, load_model, resolve_targets

# Where the standalone benchmark artifact is written. Overridable via
# ``config["results_dir"]`` so verification can target a temporary directory and
# leave the real ``results/`` untouched.
RESULTS_DIR = "results"
BENCHMARK_FILENAME = "benchmark.json"

# Default prompt used when the config supplies none; a short, neutral phrase the
# model can continue under greedy decoding.
DEFAULT_PROMPT = "The movie was"


def _resolve_model(config: dict):
    """Return ``(model, tokenizer)`` for the benchmark, honoring test overrides.

    The real benchmark loads the configured model and tokenizer with
    ``load_model`` (gpt2 by default, Req 13). Verification sets
    ``config["model_factory"]`` to a zero-argument callable returning a tiny
    model and ``config["tokenizer"]`` to a matching tokenizer, keeping the run
    fast and offline. A live ``config["tokenizer"]`` always wins so a fake
    tokenizer can pair with a real ``load_model``.
    """
    factory = config.get("model_factory")
    tokenizer = config.get("tokenizer")
    if factory is not None:
        if tokenizer is None:
            raise ValueError(
                "config['model_factory'] requires config['tokenizer'] for the "
                "benchmark."
            )
        return factory(), tokenizer
    model, loaded_tokenizer = load_model(config)
    return model, (tokenizer if tokenizer is not None else loaded_tokenizer)


def _eval_config(config: dict, tokenizer) -> dict:
    """Build the config the evaluator's generation helpers read.

    Carries the tokenizer, the deterministic decoding settings, and the seed so
    ``generate_text`` and ``measure_latency`` decode identically to the rest of
    the pipeline (Req 13.4).
    """
    return {
        "tokenizer": tokenizer,
        "decoding": config.get("decoding"),
        "seed": config.get("seed"),
    }


def _time_generation(model, prompt: str, eval_config: dict) -> float:
    """Wall-clock seconds for a single deterministic generation.

    Used for the cold-start measurement, which times the very first generation
    with no warm-up so it captures the one-time setup cost of the first decode.
    """
    start = time.perf_counter()
    generate_text(model, prompt, eval_config)
    return time.perf_counter() - start


def _perturb_adapters(manager: AdapterManager) -> None:
    """Add small noise to every managed adapter so a saved set differs.

    The swap measurement needs two distinct adapter sets over one base. After
    saving the first set this nudges each module's ``A`` and ``B`` so the second
    saved set carries different weights, making the timed swap fold and unfold a
    non-zero delta rather than a trivial no-op.
    """
    with torch.no_grad():
        for module in manager.modules.values():
            module.A.add_(torch.randn_like(module.A) * 0.01)
            module.B.add_(torch.randn_like(module.B) * 0.01)


def _measure_swap(model, config: dict):
    """Time one adapter swap, returning ``(seconds, reason)``.

    Injects LoRA, registers two saved adapter sets over the shared base through
    ``AdapterManager``, activates the first, then times activating the second.
    ``AdapterManager.activate`` routes through ``LoRA_Module.swap_adapter`` so
    this measures the real cost of switching adapters (Req 13.3). When no
    adapter can be measured (for example a model with no adaptable projections)
    it returns ``(None, reason)`` so the artifact records null-with-reason rather
    than a fabricated number (Req 11.5).
    """
    targets = config.get("targets") or ["q", "v"]
    alpha = float(config.get("alpha", 16.0))
    default_rank = int(config.get("default_rank", 8))
    rank_by_layer = config.get("rank_by_layer")

    mapping = resolve_targets(model, targets)
    if not mapping:
        return None, "model exposes no adaptable projections for the targets"

    if rank_by_layer:
        # Config keys may arrive as strings from YAML; coerce so layer indices
        # match the integer indices the resolver returns.
        ranks = {int(index): int(rank) for index, rank in rank_by_layer.items()}
    else:
        ranks = {index: default_rank for index in mapping}

    inject_lora(model, ranks, targets, alpha)
    manager = AdapterManager(model)

    with tempfile.TemporaryDirectory() as workdir:
        first_dir = os.path.join(workdir, "first")
        second_dir = os.path.join(workdir, "second")
        manager.save_adapter_set("first", first_dir)
        _perturb_adapters(manager)
        manager.save_adapter_set("second", second_dir)

        # Activate the first set so a merged adapter is in place; swap_adapter
        # requires a currently merged adapter, so this is the pre-swap state.
        manager.activate("first")

        start = time.perf_counter()
        manager.activate("second")
        return time.perf_counter() - start, None


def _metadata(config: dict, tokenizer, warmup: int, timed: int) -> dict:
    """Small provenance stub recorded alongside the measurements."""
    decoding = config.get("decoding") or {}
    model_name = getattr(tokenizer, "name_or_path", None) or config.get(
        "model_name"
    )
    return {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decoding": decoding,
        "warmup": warmup,
        "timed": timed,
    }


def run_benchmark(config: dict = None) -> dict:
    """Measure inference performance and write a single ``benchmark.json``.

    Loads the configuration (from ``configs/default.yaml`` when none is given),
    applies the configured seed, then measures, over the configured model:

    - ``cold_start_seconds``: the first generation with no warm-up.
    - ``warm_start``: mean/median/p95/stddev of the timed generations after the
      warm-up generations are discarded, via ``evaluate.measure_latency``.
    - ``tokens_per_second``: aggregate throughput of those timed generations.
    - ``adapter_swap_seconds``: the cost of switching adapters, or ``null`` with
      a reason when it cannot be measured.

    The result, plus a small metadata stub, is written to
    ``results/benchmark.json`` (overridable via ``config["results_dir"]``) and
    returned.
    """
    config = load_config(config)
    set_seeds(int(config.get("seed", 42)))

    model, tokenizer = _resolve_model(config)
    eval_config = _eval_config(config, tokenizer)

    prompts = config.get("prompts") or [DEFAULT_PROMPT]
    if isinstance(prompts, str):
        prompts = [prompts]

    latency = config.get("latency") or {}
    warmup = int(latency.get("warmup", 5))
    timed = int(latency.get("timed", 50))

    # Cold start is the first generation, timed before any warm-up runs.
    cold_start = _time_generation(model, prompts[0], eval_config)

    # Warm start reuses the evaluator's warm-up-then-timed methodology and its
    # statistics, so the discard and the stats are computed in exactly one place.
    stats = measure_latency(model, prompts, warmup, timed, eval_config)
    warm_start = {
        key: stats[key] for key in ("mean", "median", "p95", "stddev")
    }
    tokens_per_second = stats["tokens_per_second"]

    # Swap timing mutates the model in place (injection then merges), so it runs
    # after the latency measurements that need the un-adapted model.
    swap_seconds, swap_reason = _measure_swap(model, config)

    results = {
        "cold_start_seconds": cold_start,
        "warm_start": warm_start,
        "tokens_per_second": tokens_per_second,
        "adapter_swap_seconds": swap_seconds,
        "metadata": _metadata(config, tokenizer, warmup, timed),
    }
    if swap_seconds is None:
        results["adapter_swap_reason"] = swap_reason

    results_dir = config.get("results_dir", RESULTS_DIR)
    _write_json(os.path.join(results_dir, BENCHMARK_FILENAME), results)
    return results


def main() -> None:
    """Run the benchmark against the configured model and write the artifact."""
    run_benchmark()


if __name__ == "__main__":
    main()
