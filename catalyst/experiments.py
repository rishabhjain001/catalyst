"""Experiment orchestration for Catalyst.

This module will run the five experiments strictly in order, persisting real
measured numbers to ``results/experiment{N}.json`` before each next experiment
begins, and emit ``results/manifest.json`` (Requirements 14-19). The first
piece built here is reproducibility metadata capture (Requirement 10): every
experiment records the configuration snapshot, the seeds actually applied to
torch, numpy, and the Python ``random`` module (plus CUDA when available), the
git commit, the Python and PyTorch versions, the hardware, an ISO 8601
timestamp, the tokenizer name and vocab size, and the SHA-256 of
``configs/default.yaml`` so identically configured runs are recognizable.

``capture_metadata`` both sets the seeds and records them, so calling it at the
start of an experiment is what makes that experiment reproducible (Req 10.3,
10.7). The remaining orchestration pieces (write_results, write_manifest, the
five experiments, and run_all) are added in later tasks; the helpers here are
deliberately small and side-effect-light so those pieces can reuse them.
"""

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import psutil
import torch

from .data import load_sst2, make_batches
from .evaluate import (
    evaluate,
    generate_text,
    measure_latency_conditions,
    profile_memory_conditions,
    _sentiment_label,
)
from .lora import LoRA_Module
from .model import (
    AdapterManager,
    _decoder_blocks,
    inject_lora,
    load_model,
    resolve_targets,
)
from .train import train, trainable_percent

# Where the runner writes every experiment artifact (Req 19.2, 19.4). Matches
# train.RESULTS_DIR so adapter artifacts and result JSONs share one directory;
# overridable per call via ``config["results_dir"]`` so tests redirect output.
RESULTS_DIR = "results"

# The keys that belong in ``results/experiment{N}.json`` (the design's results
# schema). ``run_experiment`` may carry extra bookkeeping in its returned dict,
# but only these are persisted so the JSON stays faithful to the schema.
RESULTS_SCHEMA_KEYS = (
    "experiment",
    "status",
    "start_time",
    "end_time",
    "duration_seconds",
    "metrics",
    "failures",
)

# Experiment bodies (tasks 9.3-9.7) register themselves here keyed by experiment
# number. ``run_experiment`` looks a body up when the caller does not pass one
# explicitly, which is how the generic timing-and-graceful-failure wrapper stays
# decoupled from the specific experiments that plug into it.
EXPERIMENT_REGISTRY: dict[int, Callable[[dict], dict]] = {}

# The single configuration file whose contents are hashed into the metadata so
# identically configured runs share a ``config_sha256`` (Req 10.6). Resolved
# relative to the project root (the parent of this ``catalyst`` package) and
# overridable so the hash is testable against an arbitrary file.
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "default.yaml",
)

# The seed applied to every random source when a config omits ``seed`` (Req
# 10.3). Matches the default in ``configs/default.yaml``.
DEFAULT_SEED = 42


def config_sha256(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    """Return the SHA-256 hex digest of the configuration file's bytes.

    Hashes the raw file bytes (not the parsed mapping) so the digest matches a
    plain ``hashlib.sha256`` of the same file and is stable regardless of how
    the YAML is later loaded (Req 10.6).
    """
    with open(config_path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def git_commit() -> str:
    """Return the current git commit hash, or ``"unavailable"`` on any failure.

    Runs ``git rev-parse HEAD``. Any failure (not a repository, git missing, a
    non-zero exit) yields ``"unavailable"`` rather than raising, so metadata
    capture never blocks an experiment (Req 10.4).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

    commit = result.stdout.strip()
    return commit if commit else "unavailable"


def set_seeds(seed: int) -> dict:
    """Apply ``seed`` to every random source and return the recorded seeds.

    Sets the Python ``random``, numpy, and torch seeds, and ``cuda`` via
    ``torch.cuda.manual_seed_all`` when CUDA is available (Req 10.3, 10.7). The
    returned mapping carries the ``cuda`` entry only when CUDA is present, which
    is exactly what the metadata records.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    seeds = {"torch": seed, "numpy": seed, "random": seed}
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        seeds["cuda"] = seed
    return seeds


def _hardware_info() -> dict:
    """Return CPU model and total RAM in bytes.

    ``psutil`` exposes total RAM directly but not a CPU model string, so the
    model comes from ``platform`` (``processor`` then the ``uname`` processor
    field) and falls back to ``"unknown"`` when the platform reports neither
    (Req 10.5).
    """
    cpu_model = platform.processor() or platform.uname().processor or "unknown"
    return {
        "cpu_model": cpu_model,
        "total_ram_bytes": psutil.virtual_memory().total,
    }


def _tokenizer_info(config: dict) -> dict:
    """Return the tokenizer name and vocab size for the metadata record.

    Prefers a live tokenizer object under ``config["tokenizer"]``, reading its
    ``name_or_path`` and vocab size (``vocab_size``, falling back to
    ``len(tokenizer)`` for tokenizers that only expose a length). When no
    tokenizer object is present the name comes from ``config["model_name"]`` and
    the vocab size is ``None`` because it cannot be known without loading the
    tokenizer; recording ``None`` avoids fabricating a count (Req 10.8).
    """
    tokenizer = config.get("tokenizer")
    if tokenizer is not None:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = len(tokenizer)
            except TypeError:
                vocab_size = None
        return {
            "tokenizer_name": getattr(tokenizer, "name_or_path", None),
            "tokenizer_vocab_size": vocab_size,
        }

    return {
        "tokenizer_name": config.get("model_name"),
        "tokenizer_vocab_size": None,
    }


def _serializable_config(config: dict) -> dict:
    """Return a JSON-serializable snapshot of the configuration.

    Drops a live ``tokenizer`` object (kept in config for downstream use but not
    serializable) and coerces any remaining non-JSON values to strings, so the
    snapshot always round-trips through ``json`` while preserving the parsed
    YAML fields such as ``model_name`` (Req 10.2).
    """
    snapshot = {key: value for key, value in config.items() if key != "tokenizer"}
    return json.loads(json.dumps(snapshot, default=str))


def capture_metadata(
    config: dict,
    n: int,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict:
    """Set the seeds and return the reproducibility metadata for experiment ``n``.

    Applies the configured seed to every random source so the experiment that
    follows this call is reproducible (Req 10.3, 10.7), then records the full
    config snapshot, those seeds, the git commit (or ``"unavailable"``), the
    Python and PyTorch versions, the hardware, an ISO 8601 timestamp, the
    tokenizer name and vocab size, and the SHA-256 of the configuration file
    (Req 10.2, 10.4, 10.5, 10.6, 10.8). ``n`` identifies the experiment the
    metadata belongs to; the caller writes it to
    ``results/experiment{n}_metadata.json``.
    """
    config = config or {}
    seed = int(config.get("seed", DEFAULT_SEED))

    metadata = {
        "experiment": n,
        "config_snapshot": _serializable_config(config),
        "config_sha256": config_sha256(config_path),
        "seeds": set_seeds(seed),
        "git_commit": git_commit(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "hardware": _hardware_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metadata.update(_tokenizer_info(config))
    return metadata


class PartialExperiment(Exception):
    """Raised by an experiment body that produced some usable results before failing.

    Carries the partial ``metrics`` already gathered, the ``failures`` recorded
    for the metrics that could not be measured, and a ``reason`` for the abort.
    ``run_experiment`` catches this to mark the experiment ``"partial"`` rather
    than ``"failed"`` (Req 11.3), preserving the partial numbers instead of
    discarding them. A body that has no usable results raises an ordinary
    exception (or ``RuntimeError``/``MemoryError``) and is marked ``"failed"``.
    """

    def __init__(self, reason: str, metrics: dict = None, failures: list = None):
        super().__init__(reason)
        self.reason = reason
        self.metrics = metrics or {}
        self.failures = failures or []


def register_experiment(n: int) -> Callable[[Callable], Callable]:
    """Register an experiment body under number ``n`` and return it unchanged.

    Used as a decorator by the five experiment implementations (tasks 9.3-9.7)
    so ``run_experiment`` can dispatch to them by number without importing each
    one directly. The body has signature ``body(config) -> dict`` returning at
    least a ``metrics`` mapping and optionally ``failures`` and ``status``.
    """

    def decorator(body: Callable[[dict], dict]) -> Callable[[dict], dict]:
        EXPERIMENT_REGISTRY[n] = body
        return body

    return decorator


def null_metric(metrics: dict, failures: list, metric: str, reason: str) -> None:
    """Record ``metric`` as ``null`` with a ``reason``, never a fabricated value.

    Mirrors the ``{"metric", "reason"}`` failure convention used by
    ``evaluate.py`` so unmeasurable metrics are reported uniformly across the
    pipeline (Req 11.5).
    """
    metrics[metric] = None
    failures.append({"metric": metric, "reason": reason})


def _results_dir(config: dict) -> str:
    """Return the output directory, honoring ``config["results_dir"]``."""
    config = config or {}
    return config.get("results_dir", RESULTS_DIR)


def _results_filename(n: int) -> str:
    return f"experiment{n}.json"


def _metadata_filename(n: int) -> str:
    return f"experiment{n}_metadata.json"


def _manifest_path(results_dir: str) -> str:
    return os.path.join(results_dir, "manifest.json")


def _relative_path(results_dir: str, filename: str) -> str:
    """Join with forward slashes so recorded paths are stable across platforms.

    The manifest and result records store ``results/experiment{N}.json``-style
    paths (the design schema). Using ``os.path.join`` would emit backslashes on
    Windows, so paths are normalized to forward slashes for the recorded value
    while real filesystem writes use ``os.path.join`` separately.
    """
    return "/".join([results_dir.replace(os.sep, "/").rstrip("/"), filename])


def _write_json(path: str, payload: dict) -> None:
    """Write ``payload`` as pretty UTF-8 JSON, creating the parent directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def write_results(n: int, results: dict, config: dict = None) -> str:
    """Write ``results/experiment{n}.json`` and return the path written.

    Persists only the results-schema keys (Req 19.2, 19.5) so any bookkeeping
    the caller keeps alongside the record (such as the metadata path) never
    leaks into the JSON. The directory comes from ``config["results_dir"]`` or
    the ``RESULTS_DIR`` default.
    """
    results_dir = _results_dir(config)
    payload = {key: results[key] for key in RESULTS_SCHEMA_KEYS if key in results}
    path = os.path.join(results_dir, _results_filename(n))
    _write_json(path, payload)
    return path


def write_metadata(n: int, metadata: dict, config: dict = None) -> str:
    """Write ``results/experiment{n}_metadata.json`` and return the path written."""
    results_dir = _results_dir(config)
    path = os.path.join(results_dir, _metadata_filename(n))
    _write_json(path, metadata)
    return path


def write_manifest(records: list, config: dict = None) -> str:
    """Write ``results/manifest.json`` with one entry per experiment record.

    The manifest fixes ``manifest_version`` at ``1`` and lists every experiment
    with its status, ISO 8601 start and end timestamps, and the result and
    metadata output paths (Req 19.4). Result and metadata paths are derived from
    each record's experiment number so the manifest stays consistent with what
    ``write_results`` and ``write_metadata`` wrote.
    """
    results_dir = _results_dir(config)
    experiments = []
    for record in records:
        n = record["experiment"]
        experiments.append(
            {
                "experiment": n,
                "status": record["status"],
                "start_time": record["start_time"],
                "end_time": record["end_time"],
                "results_path": _relative_path(results_dir, _results_filename(n)),
                "metadata_path": _relative_path(results_dir, _metadata_filename(n)),
            }
        )

    payload = {"manifest_version": 1, "experiments": experiments}
    path = _manifest_path(results_dir)
    _write_json(path, payload)
    return path


def _finalize_status(metrics: dict, failures: list, declared: Optional[str]) -> str:
    """Decide the success status of an experiment that did not raise.

    A body may declare its own status; otherwise the status is ``"partial"``
    when some metrics could not be measured (non-empty ``failures``) and
    ``"completed"`` when everything was measured (Req 11.3).
    """
    if declared:
        return declared
    return "partial" if failures else "completed"


def run_experiment(
    n: int,
    config: dict,
    body: Callable[[dict], dict] = None,
) -> dict:
    """Run experiment ``n`` with timing, metadata capture, and graceful failure.

    This is the generic wrapper the five concrete experiments (tasks 9.3-9.7)
    plug into. It captures reproducibility metadata first (which also sets the
    seeds, Req 10.3), records an ISO 8601 ``start_time``, runs the experiment
    body, then records ``end_time`` and ``duration_seconds`` equal to their
    difference (Req 19.5).

    The body is the callable passed in, or the one registered under ``n`` via
    ``register_experiment``; with neither, the experiment fails gracefully with a
    "no body" reason rather than raising. On any runtime or memory failure the
    wrapper records ``"failed"`` (no usable results) or ``"partial"`` (a
    ``PartialExperiment`` carrying usable metrics), sets unmeasurable metrics to
    ``null`` with a reason, still writes the JSON, and returns the record so the
    caller can continue with the remaining experiments (Req 11.1-11.5). It never
    re-raises.

    Returns the results record (the persisted schema) so ``run_all`` can pass
    the collected records to ``write_manifest``.
    """
    config = config or {}

    # Capture metadata first so seeds are set before the body runs and the
    # metadata file exists even if the body fails (Req 10, 11.4).
    metadata = capture_metadata(config, n)
    write_metadata(n, metadata, config)

    start = datetime.now(timezone.utc)

    selected = body if body is not None else EXPERIMENT_REGISTRY.get(n)
    metrics: dict = {}
    failures: list = []
    status = "failed"

    try:
        if selected is None:
            raise RuntimeError(f"no experiment body registered for experiment {n}")
        outcome = selected(config) or {}
        metrics = outcome.get("metrics", {}) or {}
        failures = list(outcome.get("failures", []) or [])
        status = _finalize_status(metrics, failures, outcome.get("status"))
    except PartialExperiment as partial:
        # Some usable results were produced before the abort; keep them and mark
        # the experiment partial (Req 11.3).
        metrics = dict(partial.metrics)
        failures = list(partial.failures)
        failures.append({"metric": "experiment", "reason": partial.reason})
        status = "partial"
    except Exception as error:
        # No usable results: a runtime or memory failure aborts this experiment
        # gracefully without blocking the rest (Req 11.1, 11.2, 11.4).
        failures.append({"metric": "experiment", "reason": str(error)})
        status = "failed"

    end = datetime.now(timezone.utc)

    record = {
        "experiment": n,
        "status": status,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": (end - start).total_seconds(),
        "metrics": metrics,
        "failures": failures,
    }

    write_results(n, record, config)
    return record


# Ranks compared by Experiment 1 (Req 14.1). Overridable via
# ``config["experiment1_ranks"]`` so a faster check can probe fewer ranks.
EXPERIMENT1_RANKS = (4, 8, 16)


def _build_model(config: dict):
    """Return a fresh model to adapt, honoring a test ``model_factory`` override.

    Each rank needs its own un-adapted model because ``inject_lora`` mutates the
    module tree in place. The real run loads the configured model with
    ``load_model`` (gpt2 by default, Req 14). Tests may set
    ``config["model_factory"]`` to a zero-argument callable returning a small
    model, keeping the measurement fast and offline; the factory returns only
    the model since Experiment 1 counts parameters and needs no tokenizer.
    """
    factory = (config or {}).get("model_factory")
    if factory is not None:
        return factory()
    model, _tokenizer = load_model(config)
    return model


def _count_parameters(model) -> tuple:
    """Return ``(total_params, trainable_params)`` for an adapted model.

    After ``inject_lora`` only the adapter ``A``/``B`` matrices carry
    ``requires_grad=True`` (every base weight is frozen), so the trainable count
    is exactly the LoRA parameter count and the total is every parameter in the
    model (Req 14.1).
    """
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    return total, trainable


@register_experiment(1)
def experiment1(config: dict) -> dict:
    """Measure LoRA parameter efficiency at ranks 4, 8, and 16 (Req 14).

    For each rank the configured model is loaded fresh, LoRA is injected at a
    uniform rank across every adaptable layer using the configured targets, and
    the total and trainable parameter counts plus the percentage trained
    (``trainable / total * 100``) are recorded (Req 14.1, 14.2). This experiment
    only counts parameters and never trains, so it is fast.

    Results are grouped under ``metrics["by_rank"]`` keyed by the rank as a
    string, each holding ``total_params``, ``trainable_params``, and
    ``trainable_percent``::

        {"by_rank": {"4": {"total_params": ..., "trainable_params": ...,
                           "trainable_percent": ...}, "8": {...}, "16": {...}}}

    Model loading can fail without a network or cache; rather than fabricate
    numbers (Req 11.5) a rank that cannot be measured is recorded as ``null``
    with a reason. If every rank fails this raises so the wrapper marks the
    experiment ``failed``; if some ranks succeed it raises ``PartialExperiment``
    so the wrapper keeps the measured ranks and marks the experiment
    ``partial``. ``run_experiment`` writes ``results/experiment1.json`` and the
    metadata record (Req 14.3, 19.2).
    """
    config = config or {}
    ranks = list(config.get("experiment1_ranks", EXPERIMENT1_RANKS))
    targets = config.get("targets") or ["q", "v"]
    alpha = float(config.get("alpha", 16.0))

    by_rank: dict = {}
    failures: list = []
    measured_any = False

    for rank in ranks:
        key = str(rank)
        try:
            model = _build_model(config)
            # Uniform rank across every layer that exposes the requested
            # targets; the resolver's layer indices are the layers to adapt.
            rank_by_layer = {
                index: rank for index in resolve_targets(model, targets)
            }
            inject_lora(model, rank_by_layer, targets, alpha)
            total, trainable = _count_parameters(model)
            by_rank[key] = {
                "total_params": total,
                "trainable_params": trainable,
                "trainable_percent": trainable_percent(trainable, total),
            }
            measured_any = True
        except Exception as error:
            by_rank[key] = None
            failures.append(
                {"metric": f"by_rank.{key}", "reason": str(error)}
            )

    metrics = {"by_rank": by_rank}

    if not measured_any:
        # No rank could be measured: surface a runtime failure so the wrapper
        # records the experiment as failed without any fabricated numbers.
        reasons = "; ".join(entry["reason"] for entry in failures)
        raise RuntimeError(
            f"Experiment 1 could not measure any rank: {reasons}"
        )

    if failures:
        # Some ranks measured, others did not: keep the real numbers and mark
        # the experiment partial (Req 11.3).
        raise PartialExperiment(
            "Experiment 1 measured some ranks but not all.",
            metrics=metrics,
            failures=failures,
        )

    return {"metrics": metrics, "failures": failures}


# Ranks compared by Experiment 2 (Req 15.1). Overridable via
# ``config["experiment2_ranks"]`` so a fast offline check can train fewer,
# smaller adapters.
EXPERIMENT2_RANKS = (4, 8, 16)

# SST-2 rows expose the review text under one of these keys depending on the
# dataset revision; the first present key wins. Mirrors ``data._TEXT_KEYS`` so
# Experiment 2 can read a real SST-2 row or a plain ``{sentence, label}`` dict
# used by tests without importing a private helper.
_TEXT_KEYS = ("sentence", "text")


def _load_base(config: dict):
    """Return a fresh ``(model, tokenizer)`` honoring the test override hooks.

    Each rank needs its own un-adapted model because ``inject_lora`` mutates the
    module tree in place, so this is called once per rank. The real run loads
    the configured model and tokenizer with ``load_model`` (gpt2 by default, Req
    15). Tests set ``config["model_factory"]`` to a zero-argument callable
    returning a small model and ``config["tokenizer"]`` to a matching tokenizer,
    keeping the run fast and offline. A live ``config["tokenizer"]`` always wins
    over the loaded one so a fake tokenizer can pair with a real ``load_model``.
    """
    config = config or {}
    factory = config.get("model_factory")
    tokenizer = config.get("tokenizer")
    if factory is not None:
        if tokenizer is None:
            raise ValueError(
                "config['model_factory'] requires config['tokenizer'] for "
                "Experiment 2."
            )
        return factory(), tokenizer
    model, loaded_tokenizer = load_model(config)
    return model, (tokenizer if tokenizer is not None else loaded_tokenizer)


def _sentence(example: dict) -> str:
    """Return the text field of a SST-2 row or a fake ``{sentence, label}`` dict."""
    for key in _TEXT_KEYS:
        if key in example:
            return example[key]
    raise KeyError(
        f"Dataset example has no text field; expected one of {_TEXT_KEYS}."
    )


def _prompt_prefix(sentence: str, words: int = 4) -> str:
    """First few words of a sentence, used as a generation prompt.

    A short prefix gives the model something to continue while leaving the rest
    of the sentence as the reference for BLEU/ROUGE-L.
    """
    return " ".join(sentence.split()[:words]) or sentence


def _experiment2_train_data(config: dict):
    """Return the SST-2 training split, or the in-memory ``config["dataset"]``.

    The real run loads ``load_sst2("train")`` (Req 15.1); tests inject a tiny
    list of ``{sentence, label}`` dicts via ``config["dataset"]`` so no network
    or cache is touched.
    """
    dataset = (config or {}).get("dataset")
    if dataset is None:
        dataset = load_sst2("train")
    return dataset


def _experiment2_eval_data(config: dict, tokenizer) -> dict:
    """Assemble the evaluator input dict for one rank.

    Uses ``config["val_dataset"]`` when supplied (tests) or ``load_sst2(
    "validation")`` (real run), capped to ``config["eval_examples"]`` rows so
    evaluation stays bounded. From those rows it builds the validation batches
    (loss/perplexity), generation prompts and references (BLEU/ROUGE-L),
    ``(prompt, label)`` sentiment pairs, and the Gemini sample texts. Each input
    can be overridden directly via ``config["eval_prompts"]``,
    ``config["eval_references"]``, ``config["eval_sentiment"]``, and
    ``config["gemini_samples"]`` for tests that want fixed inputs.
    """
    config = config or {}
    val_dataset = config.get("val_dataset")
    if val_dataset is None:
        val_dataset = load_sst2("validation")

    limit = int(config.get("eval_examples", 8))
    count = min(limit, len(val_dataset))
    examples = [val_dataset[i] for i in range(count)]

    sentences = [_sentence(example) for example in examples]
    labels = [int(example["label"]) for example in examples]

    batch_size = int(config.get("batch_size", 8))
    validation = list(make_batches(examples, tokenizer, batch_size))

    prompts = config.get("eval_prompts") or [
        _prompt_prefix(sentence) for sentence in sentences
    ]
    references = config.get("eval_references") or sentences
    sentiment = config.get("eval_sentiment") or list(zip(prompts, labels))
    # Supplying Gemini samples means the judge returns neutral scores with no
    # API key rather than being skipped, so the rank still reports a Gemini
    # block (Req 15.2); the real key path is honored when configured.
    gemini_samples = config.get("gemini_samples") or sentences[:10]

    return {
        "validation": validation,
        "prompts": prompts,
        "references": references,
        "sentiment": sentiment,
        "gemini_samples": gemini_samples,
    }


def _file_size(path: str) -> Optional[int]:
    """Size of a file in bytes, or ``None`` when it cannot be read."""
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _dir_size(directory: str) -> Optional[int]:
    """Total size in bytes of the files directly inside ``directory``.

    The adapter-only artifact is a directory of per-module files, so its
    on-disk size is the sum of those files (Req 13.1 adapter size). Returns
    ``None`` when the directory cannot be read.
    """
    try:
        total = 0
        for entry in os.scandir(directory):
            if entry.is_file():
                total += entry.stat().st_size
        return total
    except OSError:
        return None


def _experiment2_eval_config(config: dict, tokenizer, summary: dict) -> dict:
    """Build the evaluator config for one rank from the training summary.

    Carries the tokenizer and decoding settings the evaluator needs, the
    latency/memory sample counts (small in tests, the defaults in a real run),
    and the already-measured training-run values so the evaluator reports them
    instead of nulling them: peak and average RAM and the throughput come from
    the ``train`` summary (Req 15.2), and the checkpoint and adapter artifact
    sizes come from the files ``train`` wrote.
    """
    config = config or {}
    return {
        "tokenizer": tokenizer,
        "decoding": config.get("decoding"),
        "seed": config.get("seed"),
        "gemini_api_key": config.get("gemini_api_key"),
        "latency_warmup": int(config.get("latency_warmup", 5)),
        "latency_timed": int(config.get("latency_timed", 50)),
        "memory_samples": int(config.get("memory_samples", 5)),
        "peak_ram_mb": summary.get("peak_ram_mb"),
        "avg_ram_mb": summary.get("avg_ram_mb"),
        "checkpoint_size_bytes": _file_size(summary.get("checkpoint_path", "")),
        "adapter_size_bytes": _dir_size(summary.get("adapter_dir", "")),
    }


def _build_condition_models(
    config: dict, targets, alpha: float, rank: int, trained_model, adapter_dir: str
) -> dict:
    """Build the base/unmerged/merged/post-swap models for inference profiling.

    Returns one real model per condition so the evaluator can measure inference
    latency and memory across them (Req 13.2, 13.3, 15.2). ``base`` is a fresh
    un-adapted model; ``unmerged`` is the trained model as it stands;
    ``merged`` is a copy with every adapter folded into its base; ``post_swap``
    is a copy taken through an ``AdapterManager`` swap to the saved adapter.
    Each condition is the model in the named state, never a fabricated stand-in.

    Copies keep the conditions independent so measuring one does not mutate
    another. The caller wraps this in a guard: if the conditions cannot be built
    (for example a copy exhausts memory) the evaluator records the inference
    metrics as ``null`` with a reason rather than failing the rank (Req 11.5).
    """
    base, _tokenizer = _load_base(config)

    merged = copy.deepcopy(trained_model)
    for module in merged.modules():
        if isinstance(module, LoRA_Module) and not module.merged:
            module.merge_adapter()

    post_swap = copy.deepcopy(trained_model)
    swap_manager = AdapterManager(post_swap)
    swap_manager.add_adapter("self", adapter_dir)
    swap_manager.activate("self")

    return {
        "base": base,
        "unmerged": trained_model,
        "merged": merged,
        "post_swap": post_swap,
    }


def _run_condition(
    config: dict,
    rank: int,
    targets,
    alpha: float,
    train_dataset,
    adapter_name: str,
    rank_by_layer: dict = None,
) -> dict:
    """Train and evaluate one adapted model and return its metric block.

    The shared core of Experiment 2 (vary rank), Experiment 3 (vary target
    set), and Experiment 5 (vary per-layer rank schedule): loads a fresh model
    and tokenizer, injects LoRA across every adaptable layer using ``targets``,
    trains it on the SST-2 training data via ``train``, then evaluates the
    trained model with ``evaluate`` to gather validation loss, perplexity, BLEU,
    ROUGE-L, sentiment accuracy, Gemini scores, trainable parameters, and
    inference latency/memory (Req 15.2, 16.2, 18.2). Peak/average RAM and
    throughput come from the training summary.

    The injected ranks come from ``rank_by_layer`` when supplied (Experiment 5's
    per-layer schedules, Req 18.1) and otherwise from a uniform ``rank`` applied
    to every adaptable layer (Experiments 2 and 3). ``adapter_name`` names the
    saved artifact so concurrent conditions never overwrite each other under the
    shared results directory. The returned block is the evaluator's metrics dict
    plus a ``throughput`` entry and the per-condition ``failures`` the evaluator
    recorded.
    """
    model, tokenizer = _load_base(config)
    if rank_by_layer is None:
        rank_by_layer = {index: rank for index in resolve_targets(model, targets)}
    inject_lora(model, rank_by_layer, targets, alpha)

    train_config = dict(config)
    train_config["adapter_name"] = adapter_name
    summary = train(model, (train_dataset, tokenizer), train_config)

    eval_data = _experiment2_eval_data(config, tokenizer)
    eval_config = _experiment2_eval_config(config, tokenizer, summary)
    try:
        eval_config["condition_models"] = _build_condition_models(
            config, targets, alpha, rank, model, summary["adapter_dir"]
        )
    except Exception:
        # Inference profiling is the most memory-hungry step; if the condition
        # models cannot be built the evaluator nulls latency/memory with a
        # reason rather than losing the condition's other metrics (Req 11.5).
        eval_config["condition_models"] = None

    result = evaluate(model, eval_data, eval_config)
    condition_metrics = dict(result["metrics"])
    condition_metrics["throughput"] = summary.get(
        "avg_throughput_samples_per_sec"
    )
    condition_metrics["failures"] = result["failures"]
    return condition_metrics


def _run_rank(
    config: dict, rank: int, targets, alpha: float, train_dataset
) -> dict:
    """Train and evaluate one Experiment 2 adapter at ``rank`` (Req 15).

    Thin wrapper over ``_run_condition`` that names the saved adapter per rank
    so the three Experiment 2 runs never overwrite each other's artifacts under
    the shared results directory.
    """
    return _run_condition(
        config,
        rank,
        targets,
        alpha,
        train_dataset,
        adapter_name=f"experiment2_rank{rank}",
    )


@register_experiment(2)
def experiment2(config: dict) -> dict:
    """Train and compare adapters at ranks 4, 8, and 16 (Experiment 2, Req 15).

    For each rank a fresh model is adapted at that uniform rank, trained on
    SST-2, and evaluated, so the three ranks can be compared on validation loss,
    perplexity, BLEU, ROUGE-L, sentiment accuracy, Gemini scores, trainable
    parameters, peak RAM, throughput, and inference latency (Req 15.1, 15.2).

    Results are grouped under ``metrics["by_rank"]`` keyed by the rank as a
    string; each block is the evaluator's metric dict plus ``throughput`` from
    the training summary::

        {"by_rank": {"4": {"validation_loss": ..., "perplexity": ...,
                           "bleu": ..., "rouge_l": ..., "sentiment_accuracy": ...,
                           "trainable_params": ..., "peak_ram_mb": ...,
                           "throughput": ..., "inference_latency": {...},
                           "gemini": {...}, ...},
                     "8": {...}, "16": {...}}}

    Override hooks for fast offline verification (used by tests, ignored by the
    real run): ``config["model_factory"]`` and ``config["tokenizer"]`` supply a
    tiny model and tokenizer; ``config["dataset"]`` and ``config["val_dataset"]``
    supply in-memory train/validation rows instead of SST-2;
    ``config["experiment2_ranks"]`` shrinks the rank set; ``config["max_steps"]``,
    ``config["latency_warmup"]``, ``config["latency_timed"]``,
    ``config["memory_samples"]``, and ``config["eval_examples"]`` bound the cost;
    and ``config["results_dir"]`` redirects every artifact to a temp directory.

    A rank that fails (an out-of-memory abort, a runtime error) is recorded as
    ``null`` with a reason and never fabricated (Req 11.5). If every rank fails
    this raises so the wrapper marks the experiment ``failed``; if some ranks
    succeed it raises ``PartialExperiment`` so the wrapper keeps the measured
    ranks and marks the experiment ``partial`` (Req 11.3). ``run_experiment``
    writes ``results/experiment2.json`` and the metadata before Experiment 3
    begins (Req 15.3, 19.2).
    """
    config = config or {}
    ranks = list(config.get("experiment2_ranks", EXPERIMENT2_RANKS))
    targets = config.get("targets") or ["q", "v"]
    alpha = float(config.get("alpha", 16.0))

    train_dataset = _experiment2_train_data(config)

    by_rank: dict = {}
    failures: list = []
    measured_any = False

    for rank in ranks:
        key = str(rank)
        try:
            by_rank[key] = _run_rank(
                config, rank, targets, alpha, train_dataset
            )
            measured_any = True
        except Exception as error:
            by_rank[key] = None
            failures.append({"metric": f"by_rank.{key}", "reason": str(error)})

    metrics = {"by_rank": by_rank}

    if not measured_any:
        reasons = "; ".join(entry["reason"] for entry in failures)
        raise RuntimeError(
            f"Experiment 2 could not train any rank: {reasons}"
        )

    if failures:
        raise PartialExperiment(
            "Experiment 2 trained some ranks but not all.",
            metrics=metrics,
            failures=failures,
        )

    return {"metrics": metrics, "failures": failures}


# The target-module sets compared by Experiment 3, all at the fixed rank below
# (Req 16.1). Each entry is a list of logical targets the injector resolves to
# concrete projections. Overridable via ``config["experiment3_target_sets"]`` so
# a fast offline check can probe fewer/smaller sets.
EXPERIMENT3_TARGET_SETS = (
    ["q", "v"],
    ["q", "k", "v", "o"],
    ["q", "k", "v", "o", "mlp"],
)

# The single rank every Experiment 3 target set is trained at (Req 16.1).
# Overridable via ``config["experiment3_rank"]`` so a tiny test model whose
# hidden size is below 8 can use a smaller rank.
EXPERIMENT3_RANK = 8


def _targets_key(targets) -> str:
    """Stable string key for a target set: the targets joined by commas.

    Produces ``"q,v"``, ``"q,k,v,o"``, ``"q,k,v,o,mlp"`` so the results JSON is
    keyed by a human-readable, order-preserving identifier of the set.
    """
    return ",".join(targets)


def _targets_adapter_name(targets) -> str:
    """Filesystem-safe adapter name for a target set.

    The comma-joined key is a poor directory name, so the saved artifact uses
    underscores (``experiment3_q_v``); this keeps each target set's adapter from
    overwriting another under the shared results directory.
    """
    return "experiment3_" + "_".join(targets)


@register_experiment(3)
def experiment3(config: dict) -> dict:
    """Compare target-module sets at a fixed rank (Experiment 3, Req 16).

    For each target set in [q,v], [q,k,v,o], [q,k,v,o,mlp] a fresh model is
    adapted at the same uniform rank 8, trained on SST-2, and evaluated, so the
    three sets can be compared on validation loss, parameter count, throughput,
    and memory (Req 16.1, 16.2). Adapting more modules adds more trainable
    parameters, so the sets are ordered by increasing capacity.

    Results are grouped under ``metrics["by_targets"]`` keyed by the target set
    as a comma-joined string (see ``_targets_key``); each block is the
    evaluator's metric dict plus ``throughput`` from the training summary::

        {"by_targets": {"q,v": {"validation_loss": ..., "perplexity": ...,
                                "trainable_params": ..., "trainable_percent": ...,
                                "peak_ram_mb": ..., "avg_ram_mb": ...,
                                "throughput": ..., "inference_memory": {...},
                                "inference_latency": {...}, ...},
                        "q,k,v,o": {...}, "q,k,v,o,mlp": {...}}}

    Each block carries the fuller ``evaluate`` suite, which at minimum includes
    the Req 16.2 metrics: ``validation_loss`` (validation loss), the parameter
    counts ``trainable_params``/``trainable_percent`` (parameter count),
    ``throughput`` (throughput), and ``peak_ram_mb``/``avg_ram_mb`` plus the
    per-condition ``inference_memory`` block (memory).

    Override hooks for fast offline verification (used by tests, ignored by the
    real run), mirroring Experiment 2: ``config["model_factory"]`` and
    ``config["tokenizer"]`` supply a tiny model and tokenizer;
    ``config["dataset"]`` and ``config["val_dataset"]`` supply in-memory
    train/validation rows instead of SST-2; ``config["experiment3_target_sets"]``
    and ``config["experiment3_rank"]`` shrink the comparison;
    ``config["max_steps"]``, ``config["latency_warmup"]``,
    ``config["latency_timed"]``, ``config["memory_samples"]``, and
    ``config["eval_examples"]`` bound the cost; and ``config["results_dir"]``
    redirects every artifact to a temp directory.

    A target set that fails (an out-of-memory abort, a runtime error) is
    recorded as ``null`` with a reason and never fabricated (Req 11.5). If every
    set fails this raises so the wrapper marks the experiment ``failed``; if some
    sets succeed it raises ``PartialExperiment`` so the wrapper keeps the
    measured sets and marks the experiment ``partial`` (Req 11.3).
    ``run_experiment`` writes ``results/experiment3.json`` and the metadata
    before Experiment 4 begins (Req 16.3, 19.2).
    """
    config = config or {}
    target_sets = [
        list(targets)
        for targets in config.get(
            "experiment3_target_sets", EXPERIMENT3_TARGET_SETS
        )
    ]
    rank = int(config.get("experiment3_rank", EXPERIMENT3_RANK))
    alpha = float(config.get("alpha", 16.0))

    train_dataset = _experiment2_train_data(config)

    by_targets: dict = {}
    failures: list = []
    measured_any = False

    for targets in target_sets:
        key = _targets_key(targets)
        try:
            by_targets[key] = _run_condition(
                config,
                rank,
                targets,
                alpha,
                train_dataset,
                adapter_name=_targets_adapter_name(targets),
            )
            measured_any = True
        except Exception as error:
            by_targets[key] = None
            failures.append(
                {"metric": f"by_targets.{key}", "reason": str(error)}
            )

    metrics = {"by_targets": by_targets}

    if not measured_any:
        reasons = "; ".join(entry["reason"] for entry in failures)
        raise RuntimeError(
            f"Experiment 3 could not train any target set: {reasons}"
        )

    if failures:
        raise PartialExperiment(
            "Experiment 3 trained some target sets but not all.",
            metrics=metrics,
            failures=failures,
        )

    return {"metrics": metrics, "failures": failures}


# The rank both named adapters in Experiment 4 are trained at (Req 17.1).
# Overridable via ``config["experiment4_rank"]`` so a tiny test model whose
# hidden size is small can use a smaller rank.
EXPERIMENT4_RANK = 8

# The mixing factors swept by the interpolation demonstration (Req 17.5). Alpha
# runs the named blend from pure positive (0.0) to pure negative (1.0) so the
# sentiment-versus-alpha curve has the five points the requirement names.
# Overridable via ``config["experiment4_alphas"]``.
EXPERIMENT4_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _label_subset(dataset, label: int) -> list:
    """Return the rows of ``dataset`` whose label matches ``label``.

    A Positive_Adapter trains only on positive rows and a Negative_Adapter only
    on negative rows (Req 7.1); ``make_batches`` then conditions each kept row
    with its own sentiment tag, so the positive subset becomes ``[POSITIVE]``-
    prefixed data and the negative subset becomes ``[NEGATIVE]``-prefixed data
    (``data.condition_text``).
    """
    return [row for row in dataset if int(row["label"]) == label]


def _train_named_adapter(
    config: dict, targets, alpha: float, rank: int, train_dataset, adapter_name: str
):
    """Train one named adapter on ``train_dataset`` and persist its artifact.

    Loads a fresh model and tokenizer (each adapter needs its own un-adapted
    model because ``inject_lora`` mutates in place), injects LoRA at the uniform
    ``rank`` across every adaptable layer, trains via ``train`` on the
    sentiment-filtered rows, and saves the adapter set under ``adapter_name`` so
    the positive and negative adapters live as sibling directories over one
    shared base (Req 7.1, 7.2, 12.5). Returns ``(model, tokenizer, summary)``.
    """
    model, tokenizer = _load_base(config)
    rank_by_layer = {index: rank for index in resolve_targets(model, targets)}
    inject_lora(model, rank_by_layer, targets, alpha)

    train_config = dict(config)
    train_config["adapter_name"] = adapter_name
    summary = train(model, (train_dataset, tokenizer), train_config)
    return model, tokenizer, summary


def _fixed_batch(dataset, tokenizer, count: int = 2) -> dict:
    """Build one tokenized batch from the first ``count`` rows of ``dataset``.

    The merge and swap demonstrations push a single fixed batch through the
    model so a before/after comparison uses identical input; reusing
    ``make_batches`` keeps the conditioning identical to training.
    """
    rows = [dataset[i] for i in range(min(count, len(dataset)))]
    return next(iter(make_batches(rows, tokenizer, len(rows))))


def _first_lora(model) -> LoRA_Module:
    """Return the first ``LoRA_Module`` in ``model`` for single-module demos."""
    for module in model.modules():
        if isinstance(module, LoRA_Module):
            return module
    raise ValueError("Model has no LoRA_Module to demonstrate persistence with.")


def _max_abs_diff(model, batch) -> float:
    """Max absolute difference between unmerged and merged forward outputs.

    Operates on a copy so the live model keeps its unmerged state for the later
    inference-condition profiling: pushes ``batch`` through the unmerged copy,
    merges every adapter, pushes the same batch again, and returns the maximum
    absolute logit difference. The merge identity holds within ``1e-5`` (Req
    2.3), so this headline number is the measured equivalence evidence (Req
    17.2).
    """
    probe = copy.deepcopy(model)
    probe.eval()
    with torch.no_grad():
        before = probe(**batch).logits
    for module in probe.modules():
        if isinstance(module, LoRA_Module) and not module.merged:
            module.merge_adapter()
    with torch.no_grad():
        after = probe(**batch).logits
    return float((before - after).abs().max().item())


def _swap_demonstration(trained_model, positive_dir: str, negative_dir: str, batch) -> dict:
    """Switch between named adapters via ``swap_adapter`` and report the change.

    Builds an ``AdapterManager`` over a copy of a trained model, registers the
    positive and negative adapter directories, activates each in turn (which
    routes through ``swap_adapter``, Req 7.3), and compares the forward outputs
    of the two active adapters on a fixed batch. Returns the active adapter name
    after each activation and the maximum absolute output difference between
    them, which is the evidence the swap actually changed the model's behavior
    (Req 17.3).
    """
    swap_model = copy.deepcopy(trained_model)
    manager = AdapterManager(swap_model)
    manager.add_adapter("positive", positive_dir)
    manager.add_adapter("negative", negative_dir)

    manager.activate("positive")
    active_after_positive = manager.active
    swap_model.eval()
    with torch.no_grad():
        positive_logits = swap_model(**batch).logits

    manager.activate("negative")
    active_after_negative = manager.active
    with torch.no_grad():
        negative_logits = swap_model(**batch).logits

    return {
        "active_after_positive": active_after_positive,
        "active_after_negative": active_after_negative,
        "output_max_abs_diff": float(
            (positive_logits - negative_logits).abs().max().item()
        ),
    }


def _persistence_round_trip(
    positive_model, negative_model, directory: str
) -> dict:
    """Save, load, and list named adapters and report the round trip (Req 17.4).

    Writes the positive and negative adapters as single-file artifacts into one
    directory, loads the positive artifact back into a shape-matching module and
    confirms its forward output is unchanged within ``1e-5`` (the persistence
    round trip, Property 2), then lists the directory and confirms the listed
    identifiers include both saved names (Property 3). Returns
    ``{"saved", "listed", "round_trip_ok"}`` where ``round_trip_ok`` is true
    only when both the load preserved the output and every saved identifier was
    listed.
    """
    os.makedirs(directory, exist_ok=True)
    positive_module = _first_lora(positive_model)
    negative_module = _first_lora(negative_model)

    positive_path = os.path.join(directory, "positive.pt")
    negative_path = os.path.join(directory, "negative.pt")
    positive_module.save_adapter(positive_path)
    negative_module.save_adapter(negative_path)
    saved = ["positive", "negative"]

    probe = torch.randn(2, positive_module.in_features)
    positive_module.eval()
    with torch.no_grad():
        before = positive_module(probe)
    # Reload into a perturbed copy so a no-op load cannot mask a broken restore.
    reloaded = copy.deepcopy(positive_module)
    with torch.no_grad():
        reloaded.A.zero_()
        reloaded.B.zero_()
    reloaded.load_adapter(positive_path)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(probe)
    load_ok = float((before - after).abs().max().item()) <= 1e-5

    listed = LoRA_Module.list_adapters(directory)
    round_trip_ok = load_ok and set(saved) <= set(listed)
    return {"saved": saved, "listed": listed, "round_trip_ok": round_trip_ok}


def _sentiment_score(text: str) -> float:
    """Scalar sentiment of a generation in ``[0.0, 1.0]`` for the alpha curve.

    Reuses the evaluator's documented tag proxy (``_sentiment_label``): a
    generation that expresses the positive tag scores ``1.0``, one that
    expresses the negative tag scores ``0.0``, and an absent or contradictory
    signal scores the neutral midpoint ``0.5``. Plotting this scalar against the
    blend alpha shows whether sentiment shifts smoothly or in discrete jumps
    (Req 17.5).
    """
    label = _sentiment_label(text)
    if label == "positive":
        return 1.0
    if label == "negative":
        return 0.0
    return 0.5


def _generate_blended(
    blend_model, manager, name_a: str, name_b: str, alpha: float, prompt: str, eval_config: dict
) -> str:
    """Generate text with two named adapters blended at ``alpha``.

    ``AdapterManager.blend`` returns each module's interpolated effective weight
    without mutating the live base (Req 7.4); to actually generate, each
    module's base weight is temporarily set to that effective weight with the
    adapter path disabled, then restored after generation. The blend model is
    kept un-activated so its base weights stay pristine ``W`` and every blend is
    read against the same baseline.
    """
    effective = manager.blend(name_a, name_b, alpha)

    restore = {}
    for path, module in manager.modules.items():
        weight = effective[path]
        # Conv1D stores its weight transposed relative to the logical d x k.
        stored = weight.t() if module.layer_kind == "conv1d" else weight
        restore[path] = (module.base.weight.detach().clone(), module.merged)
        with torch.no_grad():
            module.base.weight.copy_(stored)
        # Disable the adapter path so forward uses only the blended base weight.
        module.merged = True

    try:
        return generate_text(blend_model, prompt, eval_config)
    finally:
        for path, module in manager.modules.items():
            original, merged = restore[path]
            with torch.no_grad():
                module.base.weight.copy_(original)
            module.merged = merged


def _interpolation_curve(
    config: dict, targets, alpha: float, rank: int, positive_dir: str, negative_dir: str, prompt: str, eval_config: dict, alphas
) -> list:
    """Sweep the named blend across ``alphas`` and score each generation.

    Builds a fresh adapted model whose base weights are pristine (zero-init
    ``B``), registers the positive and negative adapters, and for each alpha
    generates a continuation of ``prompt`` from the blended weights and scores
    its sentiment. Returns the ``sentiment_by_alpha`` curve as a list of
    ``{"alpha", "sentiment_score", "text"}`` points so a continuous
    sentiment-versus-alpha curve can be plotted (Req 17.5).
    """
    blend_model, _tokenizer = _load_base(config)
    rank_by_layer = {index: rank for index in resolve_targets(blend_model, targets)}
    inject_lora(blend_model, rank_by_layer, targets, alpha)

    manager = AdapterManager(blend_model)
    manager.add_adapter("positive", positive_dir)
    manager.add_adapter("negative", negative_dir)

    curve = []
    for value in alphas:
        text = _generate_blended(
            blend_model, manager, "positive", "negative", float(value), prompt, eval_config
        )
        curve.append(
            {
                "alpha": float(value),
                "sentiment_score": _sentiment_score(text),
                "text": text,
            }
        )
    return curve


def _experiment4_eval_config(config: dict, tokenizer) -> dict:
    """Minimal evaluator config for the four-condition latency/memory profiling.

    Carries the tokenizer and deterministic decoding settings generation needs
    plus the small latency/memory sample counts a fast offline run sets; the
    real run falls back to the defaults the evaluator already documents.
    """
    config = config or {}
    return {
        "tokenizer": tokenizer,
        "decoding": config.get("decoding"),
        "seed": config.get("seed"),
        "latency_warmup": int(config.get("latency_warmup", 5)),
        "latency_timed": int(config.get("latency_timed", 50)),
        "memory_samples": int(config.get("memory_samples", 5)),
    }


@register_experiment(4)
def experiment4(config: dict) -> dict:
    """Demonstrate adapter composition, merging, swapping, persistence, and
    interpolation (Experiment 4, Req 17).

    Trains a Positive_Adapter on ``[POSITIVE]``-conditioned rows and a
    Negative_Adapter on ``[NEGATIVE]``-conditioned rows (Req 17.1), then runs
    five demonstrations over the two named adapters and records the measured
    evidence for each. The metrics block has this structure::

        {"merge_max_abs_diff": float,        # max |unmerged - merged| logit diff (Req 17.2)
         "swap_demonstrated": bool,          # the named swap changed the active adapter (Req 17.3)
         "swap_detail": {"active_after_positive": "positive",
                         "active_after_negative": "negative",
                         "output_max_abs_diff": float},
         "persistence": {"saved": [...], "listed": [...],
                         "round_trip_ok": bool},                      # save/load/list (Req 17.4)
         "sentiment_by_alpha": [{"alpha": a, "sentiment_score": s,
                                 "text": str}, ...],                  # the continuous curve (Req 17.5)
         "inference_latency": {"base": {...}, "unmerged": {...},
                               "merged": {...}, "post_swap": {...}},
         "inference_memory":  {"base": {...}, "unmerged": {...},
                               "merged": {...}, "post_swap": {...}}}

    The sentiment score is the documented tag proxy reused from the evaluator:
    a positive-tagged generation scores ``1.0``, a negative-tagged one ``0.0``,
    and an absent or contradictory signal the neutral midpoint ``0.5`` (see
    ``_sentiment_score``); plotting it against the blend alpha shows whether the
    sentiment shift is a smooth gradient or discrete jumps.

    Override hooks for fast offline verification (used by tests, ignored by the
    real run): ``config["model_factory"]`` and ``config["tokenizer"]`` supply a
    tiny model and tokenizer; ``config["dataset"]`` supplies in-memory training
    rows (which must include both positive and negative labels);
    ``config["experiment4_rank"]`` and ``config["experiment4_alphas"]`` shrink
    the demonstration; ``config["max_steps"]``, ``config["latency_warmup"]``,
    ``config["latency_timed"]``, and ``config["memory_samples"]`` bound the
    cost; ``config["eval_prompts"]`` fixes the generation prompt; and
    ``config["results_dir"]`` redirects every artifact to a temp directory.

    Training both adapters is foundational, so a failure to train either aborts
    the experiment as ``failed`` (Req 11.2). Once both adapters exist, each
    individual demonstration is guarded: a demonstration that cannot be measured
    is recorded as ``null`` with a reason and never fabricated (Req 11.5), and
    the wrapper marks the experiment ``partial`` when any demonstration failed.
    ``run_experiment`` writes ``results/experiment4.json`` and the metadata
    before Experiment 5 begins (Req 17.5, 19.2).
    """
    config = config or {}
    targets = config.get("targets") or ["q", "v"]
    alpha = float(config.get("alpha", 16.0))
    rank = int(config.get("experiment4_rank", EXPERIMENT4_RANK))
    alphas = list(config.get("experiment4_alphas", EXPERIMENT4_ALPHAS))

    train_dataset = _experiment2_train_data(config)
    positive_rows = _label_subset(train_dataset, 1)
    negative_rows = _label_subset(train_dataset, 0)
    if not positive_rows or not negative_rows:
        raise RuntimeError(
            "Experiment 4 needs both positive and negative rows to train the "
            "named adapters; the dataset is missing one sentiment."
        )

    # Train both named adapters first; without them no demonstration can run.
    positive_model, tokenizer, positive_summary = _train_named_adapter(
        config, targets, alpha, rank, positive_rows, "experiment4_positive"
    )
    negative_model, _negative_tokenizer, negative_summary = _train_named_adapter(
        config, targets, alpha, rank, negative_rows, "experiment4_negative"
    )
    positive_dir = positive_summary["adapter_dir"]
    negative_dir = negative_summary["adapter_dir"]

    metrics: dict = {}
    failures: list = []

    batch = _fixed_batch(train_dataset, tokenizer)

    # Merge equivalence: the headline measured number (Req 17.2).
    try:
        metrics["merge_max_abs_diff"] = _max_abs_diff(positive_model, batch)
    except Exception as error:
        null_metric(metrics, failures, "merge_max_abs_diff", str(error))

    # Instant named-adapter swap (Req 17.3).
    try:
        detail = _swap_demonstration(positive_model, positive_dir, negative_dir, batch)
        metrics["swap_detail"] = detail
        metrics["swap_demonstrated"] = (
            detail["active_after_positive"] == "positive"
            and detail["active_after_negative"] == "negative"
        )
    except Exception as error:
        metrics["swap_demonstrated"] = False
        null_metric(metrics, failures, "swap_detail", str(error))

    # Save/load/list round trip (Req 17.4).
    try:
        metrics["persistence"] = _persistence_round_trip(
            positive_model,
            negative_model,
            os.path.join(_results_dir(config), "experiment4_adapters"),
        )
    except Exception as error:
        null_metric(metrics, failures, "persistence", str(error))

    eval_config = _experiment4_eval_config(config, tokenizer)
    prompts = config.get("eval_prompts") or ["The movie was"]

    # Sentiment interpolation curve across alpha 0.0..1.0 (Req 17.5).
    try:
        metrics["sentiment_by_alpha"] = _interpolation_curve(
            config, targets, alpha, rank, positive_dir, negative_dir,
            prompts[0], eval_config, alphas,
        )
    except Exception as error:
        null_metric(metrics, failures, "sentiment_by_alpha", str(error))

    # Inference latency and memory for the four conditions (Req 17.5).
    try:
        condition_models = _build_condition_models(
            config, targets, alpha, rank, positive_model, positive_dir
        )
        warmup = int(config.get("latency_warmup", 5))
        timed = int(config.get("latency_timed", 50))
        samples = int(config.get("memory_samples", 5))
        metrics["inference_latency"] = measure_latency_conditions(
            condition_models, prompts, warmup, timed, eval_config
        )
        metrics["inference_memory"] = profile_memory_conditions(
            condition_models, prompts, samples, eval_config
        )
    except Exception as error:
        null_metric(metrics, failures, "inference_latency", str(error))
        null_metric(metrics, failures, "inference_memory", str(error))

    return {"metrics": metrics, "failures": failures}


# Experiment 5 rank schedules (Req 18.1). The requirement fixes the Uniform
# schedule at rank 8 everywhere, the Progressive schedule at rank 4 shallow / 8
# middle / 16 deep, and the Inverse schedule at rank 16 shallow / 4 deep. The
# Inverse middle rank is set to 8 so Progressive and Inverse share the same
# average rank ((4 + 8 + 16) / 3 == (16 + 8 + 4) / 3 == 9.33), which leaves the
# Uniform average (8) about 16.7% below them. Because a layer's LoRA parameter
# count scales linearly with its rank and the decoder layers are equal width,
# equal average rank means an approximately equal total trainable budget, so the
# three budgets fall inside the tolerance band below (Req 18.1, Property 19).
EXPERIMENT5_UNIFORM_RANK = 8
EXPERIMENT5_PROGRESSIVE_RANKS = (4, 8, 16)
EXPERIMENT5_INVERSE_RANKS = (16, 8, 4)

# The fraction by which the three schedule budgets may differ and still count as
# "approximately equal" (Req 18.1, Property 19). Set to 0.20: the Uniform
# average rank (8) sits about 16.7% below the Progressive/Inverse average (9.33)
# and that gap is fixed by the requirement's prescribed rank choices, so the
# band must admit roughly that spread. Overridable via
# ``config["experiment5_budget_tolerance"]``.
EXPERIMENT5_BUDGET_TOLERANCE = 0.20


def _split_thirds(layer_indices) -> tuple:
    """Split sorted layer indices into ``(shallow, middle, deep)`` thirds.

    The shallow third is the first ``len // 3`` layers, the deep third is the
    last ``len // 3``, and the middle takes whatever remains (so a count not
    divisible by three widens the middle). When there are fewer than three
    layers the split cannot form three groups, so every layer is treated as
    middle and the schedules gracefully collapse onto the middle rank rather
    than failing on a tiny model (a 2-layer test model, for example).
    """
    ordered = sorted(layer_indices)
    count = len(ordered)
    cut = count // 3
    if cut == 0:
        return [], ordered, []
    return ordered[:cut], ordered[cut : count - cut], ordered[count - cut :]


def rank_schedules(layer_indices, config: dict = None) -> dict:
    """Return the uniform, progressive, and inverse per-layer rank schedules.

    Each schedule is a ``rank_by_layer`` dict keyed by the adaptable decoder
    layer indices (Req 18.1). The layers are split into shallow/middle/deep
    thirds; ``uniform`` places one rank at every layer, ``progressive`` places
    increasing ranks (shallow < middle < deep), and ``inverse`` mirrors it
    (shallow > deep) while keeping the same middle rank so progressive and
    inverse carry the same average rank, and therefore the same approximate
    parameter budget, as documented at the schedule constants. The ranks are
    overridable via ``config`` so a tiny model with a small hidden size can use
    proportionally smaller ranks.
    """
    config = config or {}
    uniform_rank = int(
        config.get("experiment5_uniform_rank", EXPERIMENT5_UNIFORM_RANK)
    )
    progressive_ranks = tuple(
        config.get("experiment5_progressive_ranks", EXPERIMENT5_PROGRESSIVE_RANKS)
    )
    inverse_ranks = tuple(
        config.get("experiment5_inverse_ranks", EXPERIMENT5_INVERSE_RANKS)
    )

    shallow, middle, deep = _split_thirds(layer_indices)

    uniform = {index: uniform_rank for index in layer_indices}
    progressive: dict = {}
    inverse: dict = {}
    for group, progressive_rank, inverse_rank in (
        (shallow, progressive_ranks[0], inverse_ranks[0]),
        (middle, progressive_ranks[1], inverse_ranks[1]),
        (deep, progressive_ranks[2], inverse_ranks[2]),
    ):
        for index in group:
            progressive[index] = int(progressive_rank)
            inverse[index] = int(inverse_rank)

    return {"uniform": uniform, "progressive": progressive, "inverse": inverse}


def _schedule_param_counts(
    config: dict, targets, alpha: float, rank_by_layer: dict
) -> dict:
    """Per-layer trainable parameter counts for one schedule (Req 18.3).

    Builds a fresh model, injects LoRA at ``rank_by_layer``, and returns a dict
    mapping each adapted decoder layer index to the trainable adapter (``A`` and
    ``B``) parameter count at that layer. No training happens, so this is cheap
    and deterministic; the sum across layers is the schedule's total trainable
    budget used for the equal-budget comparison (Req 18.1, Property 19), and the
    per-layer breakdown is stored alongside each schedule's ``rank_by_layer`` so
    the parameter distribution can later be visualized (Req 18.3).
    """
    model, _tokenizer = _load_base(config)
    inject_lora(model, rank_by_layer, targets, alpha)

    counts: dict = {}
    for index, block in _decoder_blocks(model):
        if index not in rank_by_layer:
            continue
        layer_total = 0
        for module in block.modules():
            if isinstance(module, LoRA_Module):
                layer_total += int(module.A.numel() + module.B.numel())
        counts[index] = layer_total
    return counts


def _budgets_within_tolerance(budgets: dict, tolerance: float) -> bool:
    """Whether the schedule budgets sit inside the tolerance band of each other.

    Compares the largest and smallest measured budgets: they are
    "approximately equal" when ``(max - min) / min`` is at most ``tolerance``
    (Req 18.1, Property 19). Budgets recorded as ``None`` (a schedule whose
    counts could not be measured) are ignored, and fewer than two measured
    budgets cannot be out of band so the check passes.
    """
    measured = [value for value in budgets.values() if value]
    if len(measured) < 2:
        return True
    smallest = min(measured)
    largest = max(measured)
    if smallest <= 0:
        return False
    return (largest - smallest) / smallest <= tolerance


@register_experiment(5)
def experiment5(config: dict) -> dict:
    """Compare per-layer rank schedules at an equal budget (Experiment 5, Req 18).

    Builds three ``rank_by_layer`` schedules over the model's adaptable decoder
    layers — uniform (rank 8 everywhere), progressive (4 shallow / 8 middle / 16
    deep), and inverse (16 shallow / 8 middle / 4 deep) — whose total trainable
    parameter budgets are approximately equal because they share an equal (or
    near-equal) average rank over equal-width layers (Req 18.1). Each schedule's
    adapted model is trained on SST-2 and evaluated so the three can be compared
    on validation loss, perplexity, BLEU, ROUGE-L, sentiment accuracy, trainable
    parameters, peak RAM, throughput, and inference latency (Req 18.2).

    The metrics block has this structure::

        {"by_schedule": {"uniform": {<evaluate metrics>, "throughput": ...,
                                     "rank_by_layer": {"0": 8, ...},
                                     "per_layer_param_counts": {"0": 1024, ...}},
                         "progressive": {...}, "inverse": {...}},
         "schedule_budgets": {"uniform": int, "progressive": int,
                              "inverse": int},          # total trainable per schedule
         "budget_tolerance": 0.20,
         "budgets_within_tolerance": bool}              # the equal-budget check

    The per-schedule ``rank_by_layer`` and ``per_layer_param_counts`` plus the
    top-level ``schedule_budgets`` are stored so the per-layer rank allocation
    diagram, the parameter distribution chart, and the performance comparison
    table can later be drawn from this JSON (Req 18.3); the diagrams themselves
    live in the visualizer (task 11), not here.

    Override hooks for fast offline verification (used by tests, ignored by the
    real run), mirroring Experiments 2 and 3: ``config["model_factory"]`` and
    ``config["tokenizer"]`` supply a tiny model and tokenizer;
    ``config["dataset"]`` and ``config["val_dataset"]`` supply in-memory
    train/validation rows instead of SST-2;
    ``config["experiment5_uniform_rank"]``,
    ``config["experiment5_progressive_ranks"]``, and
    ``config["experiment5_inverse_ranks"]`` rescale the schedules for a small
    hidden size; ``config["experiment5_budget_tolerance"]`` sets the band;
    ``config["max_steps"]``, ``config["latency_warmup"]``,
    ``config["latency_timed"]``, ``config["memory_samples"]``, and
    ``config["eval_examples"]`` bound the cost; and ``config["results_dir"]``
    redirects every artifact to a temp directory.

    The deterministic per-schedule budgets are computed first (a fresh inject
    and count, no training) so the equal-budget data is complete even if a
    schedule's training later fails. A schedule whose training fails is recorded
    as ``null`` with a reason and never fabricated (Req 11.5). If every schedule
    fails this raises so the wrapper marks the experiment ``failed``; if some
    succeed it raises ``PartialExperiment`` so the wrapper keeps the measured
    schedules and marks the experiment ``partial`` (Req 11.3).
    ``run_experiment`` writes ``results/experiment5.json`` and the metadata
    (Req 18.4, 19.2).
    """
    config = config or {}
    targets = config.get("targets") or ["q", "v"]
    alpha = float(config.get("alpha", 16.0))
    tolerance = float(
        config.get("experiment5_budget_tolerance", EXPERIMENT5_BUDGET_TOLERANCE)
    )

    train_dataset = _experiment2_train_data(config)

    # The adaptable layer indices come from the architecture, so the schedules
    # adapt to GPT-2's 12 layers or a tiny test model's few layers alike.
    probe, _tokenizer = _load_base(config)
    layer_indices = sorted(resolve_targets(probe, targets).keys())
    schedules = rank_schedules(layer_indices, config)

    # Equal-budget data first: deterministic per-layer counts and totals are
    # recorded for every schedule regardless of whether its training succeeds
    # (Req 18.1, 18.3).
    per_layer_counts: dict = {}
    budgets: dict = {}
    for name, rank_by_layer in schedules.items():
        layer_counts = _schedule_param_counts(config, targets, alpha, rank_by_layer)
        per_layer_counts[name] = layer_counts
        budgets[name] = sum(layer_counts.values())

    by_schedule: dict = {}
    failures: list = []
    measured_any = False

    for name, rank_by_layer in schedules.items():
        try:
            block = _run_condition(
                config,
                None,
                targets,
                alpha,
                train_dataset,
                adapter_name=f"experiment5_{name}",
                rank_by_layer=rank_by_layer,
            )
            block["rank_by_layer"] = {
                str(index): rank for index, rank in rank_by_layer.items()
            }
            block["per_layer_param_counts"] = {
                str(index): count
                for index, count in per_layer_counts[name].items()
            }
            by_schedule[name] = block
            measured_any = True
        except Exception as error:
            by_schedule[name] = None
            failures.append(
                {"metric": f"by_schedule.{name}", "reason": str(error)}
            )

    metrics = {
        "by_schedule": by_schedule,
        "schedule_budgets": budgets,
        "budget_tolerance": tolerance,
        "budgets_within_tolerance": _budgets_within_tolerance(budgets, tolerance),
    }

    if not measured_any:
        reasons = "; ".join(entry["reason"] for entry in failures)
        raise RuntimeError(
            f"Experiment 5 could not train any schedule: {reasons}"
        )

    if failures:
        raise PartialExperiment(
            "Experiment 5 trained some schedules but not all.",
            metrics=metrics,
            failures=failures,
        )

    return {"metrics": metrics, "failures": failures}


# The experiments are always run in this strict order so each JSON is written
# before the next begins and the manifest reflects 1→5 progress (Req 19.1).
EXPERIMENT_ORDER = (1, 2, 3, 4, 5)


def load_config(config: dict = None, config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Return the caller's config, or load ``configs/default.yaml`` when none given.

    ``run_all`` accepts either an explicit config dict (the real entry points and
    the offline tests pass one) or ``None``, in which case the single source
    config is read from ``configs/default.yaml`` (Req 9.1). The YAML import is
    deferred so importing this module never hard-fails when pyyaml is absent; a
    genuine read failure surfaces rather than being masked.
    """
    if config is not None:
        return config

    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _startup_model_batch(config: dict):
    """Return ``(model, batch)`` for the startup timed step, honoring overrides.

    Mirrors the experiment bodies' override hooks so the startup estimate runs
    offline in tests: ``config["model_factory"]`` and ``config["tokenizer"]``
    supply a tiny model and matching tokenizer, and ``config["dataset"]`` an
    in-memory SST-2 stand-in. The real run loads the configured model and
    tokenizer with ``load_model`` (gpt2 by default) and the SST-2 training split
    (Req 9.4). One conditioned batch is built with ``make_batches`` so the timed
    step measures a representative forward-plus-backward over real data.
    """
    config = config or {}
    factory = config.get("model_factory")
    tokenizer = config.get("tokenizer")
    if factory is not None:
        if tokenizer is None:
            raise ValueError(
                "config['model_factory'] requires config['tokenizer'] for the "
                "startup runtime estimate."
            )
        model = factory()
    else:
        model, loaded_tokenizer = load_model(config)
        tokenizer = tokenizer if tokenizer is not None else loaded_tokenizer

    dataset = config.get("dataset")
    if dataset is None:
        dataset = load_sst2("train")

    batch_size = int(config.get("batch_size", 8))
    batch = next(iter(make_batches(dataset, tokenizer, batch_size)))
    return model, batch


def run_startup_estimate(config: dict) -> Optional[float]:
    """Time one startup step and print the runtime estimate; degrade gracefully.

    Runs a single ``timed_step`` over a representative batch, then prints the
    measured step time and the projected total for all five experiments,
    warning when the projection exceeds 45 minutes (Req 9.4, 9.5) via
    ``report_runtime`` and ``total_experiment_steps``. Building the step can fail
    without a network or model cache; rather than aborting the whole run the
    failure is logged and the estimate is skipped (the experiments themselves
    still run and fail gracefully if model load fails). Returns the measured step
    time, or ``None`` when the estimate was skipped.
    """
    try:
        model, batch = _startup_model_batch(config)
        step_seconds = timed_step(model, batch)
    except Exception as error:
        print(f"Skipping startup runtime estimate: {error}")
        return None

    report_runtime(step_seconds, total_experiment_steps(config))
    return step_seconds


def _log_config_sha256(config: dict) -> None:
    """Print the configuration SHA-256, degrading gracefully if unreadable.

    Surfaces the ``config_sha256`` of the source config file (Req 10.6) at the
    start of the run so identically configured runs are recognizable from the
    console alone; ``capture_metadata`` separately records it into each
    metadata file. The path is overridable via ``config["config_path"]``.
    """
    config_path = (config or {}).get("config_path", DEFAULT_CONFIG_PATH)
    try:
        print(f"Configuration SHA-256: {config_sha256(config_path)}")
    except OSError as error:
        print(f"Configuration SHA-256 unavailable: {error}")


def run_all(config: dict = None) -> None:
    """Run all five experiments strictly 1→5, persisting after each (Req 19).

    Loads the configuration (from ``configs/default.yaml`` when none is passed)
    and surfaces its ``config_sha256``, runs one startup timed step printing the
    runtime estimate and the 45-minute warning (Req 9.4, 9.5), then runs the
    experiments in strict order. Each ``run_experiment`` call writes
    ``results/experiment{n}.json`` and its metadata before the next experiment
    begins (Req 19.1, 19.2) and never raises, so a failed or partial experiment
    does not stop the run (Req 19.3). The manifest is rewritten after every
    experiment so it reflects progress incrementally and once more at the end
    (Req 19.4).
    """
    config = load_config(config)

    _log_config_sha256(config)
    run_startup_estimate(config)

    records: list = []
    for n in EXPERIMENT_ORDER:
        record = run_experiment(n, config)
        records.append(record)
        # Update the manifest after each experiment so it reflects progress even
        # if a later experiment is interrupted (Req 19.4).
        write_manifest(records, config)

    # Final manifest write after all five complete (Req 19.4).
    write_manifest(records, config)
