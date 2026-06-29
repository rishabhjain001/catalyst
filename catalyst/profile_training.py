"""cProfile entry point for one Catalyst training run.

Profiles a single ``train`` run and writes two artifacts under ``logs/``: the
raw ``cProfile`` stats (``training_profile.prof``) and a human-readable report
(``training_profile.txt``) listing the top functions by cumulative and total
time, so it is visible whether tokenization, data loading, or matmul dominates.

The real run loads the configured model and SST-2; for fast offline use the
config may supply ``model_factory``, ``tokenizer``, and ``dataset`` overrides
exactly like the experiment runner. A flame graph is rendered only when a
converter such as ``flameprof`` is installed; otherwise that step is skipped.
"""

import cProfile
import io
import os
import pstats

from .data import load_sst2
from .model import _decoder_blocks, inject_lora, load_model
from .train import train

# Artifact stems written into the output directory. The flame graph keeps the
# same stem with an ``.svg`` suffix when a converter is available.
PROFILE_STEM = "training_profile"

# How many rows each sorted section of the text report lists. Enough to surface
# the dominant tokenization, data-loading, and matmul frames without burying
# them under thousands of trivial calls.
TOP_ROWS = 30


def _build_components(config: dict):
    """Return ``(model, tokenizer, dataset)``, honoring offline overrides.

    The real run loads the configured model and tokenizer with ``load_model``
    and the SST-2 train split. Tests set ``model_factory`` (a zero-argument
    callable returning an un-adapted model), ``tokenizer``, and ``dataset`` to
    keep the run small and offline.
    """
    config = config or {}
    factory = config.get("model_factory")
    tokenizer = config.get("tokenizer")
    dataset = config.get("dataset")

    if factory is not None:
        model = factory()
        if tokenizer is None:
            _model, tokenizer = load_model(config)
    else:
        model, loaded_tokenizer = load_model(config)
        if tokenizer is None:
            tokenizer = loaded_tokenizer

    if dataset is None:
        dataset = load_sst2("train")

    return model, tokenizer, dataset


def _inject_adapters(model, config: dict):
    """Inject LoRA at a uniform rank across every decoder block.

    Mirrors the real training setup: the layer indices come from the model's
    own decoder blocks, each gets ``default_rank``, and the configured logical
    targets and ``alpha`` apply. The model is adapted in place and returned.
    """
    config = config or {}
    default_rank = int(config.get("default_rank", 8))
    targets = config.get("targets") or ["q", "v"]
    alpha = float(config.get("alpha", 16.0))
    rank_by_layer = {index: default_rank for index, _ in _decoder_blocks(model)}
    return inject_lora(model, rank_by_layer, targets, alpha)


def _write_report(profiler: cProfile.Profile, text_path: str, top_rows: int) -> None:
    """Write a top-functions report sorted by cumulative then total time.

    Cumulative time exposes the expensive call trees (data loading, the forward
    pass), while total time exposes the leaf hot spots (matmul, tokenization),
    so both sections together reveal what dominates one training run.
    """
    stream = io.StringIO()
    stream.write("Catalyst training profile\n")
    stream.write("Top functions by cumulative time (call-tree cost):\n\n")
    cumulative = pstats.Stats(profiler, stream=stream)
    cumulative.sort_stats("cumulative").print_stats(top_rows)

    stream.write("\nTop functions by total time (leaf hot spots):\n\n")
    total = pstats.Stats(profiler, stream=stream)
    total.sort_stats("tottime").print_stats(top_rows)

    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(stream.getvalue())


def _render_flame_graph(prof_path: str, svg_path: str):
    """Best-effort flame graph; return its path or ``None`` if unavailable.

    Rendering is optional by design: when no converter (``flameprof``) is
    installed the step is skipped gracefully so profiling never hard-depends on
    a flame-graph tool.
    """
    try:
        from flameprof import render
    except Exception:
        print("flame graph skipped: install 'flameprof' to render one.")
        return None

    try:
        stats = pstats.Stats(prof_path)
        with open(svg_path, "w", encoding="utf-8") as handle:
            render(stats.stats, handle)
    except Exception as error:
        print(f"flame graph skipped: rendering failed ({error}).")
        return None
    return svg_path


def run_profile(config: dict = None, output_dir: str = "logs") -> str:
    """Profile one training run and write the stats and report under ``output_dir``.

    Returns the path to the human-readable text report. A ``.prof`` stats file
    is written alongside it, and a flame-graph ``.svg`` is added when a
    converter is installed.
    """
    config = config or {}
    os.makedirs(output_dir, exist_ok=True)
    prof_path = os.path.join(output_dir, f"{PROFILE_STEM}.prof")
    text_path = os.path.join(output_dir, f"{PROFILE_STEM}.txt")

    model, tokenizer, dataset = _build_components(config)
    _inject_adapters(model, config)

    profiler = cProfile.Profile()
    profiler.enable()
    train(model, (dataset, tokenizer), config)
    profiler.disable()

    profiler.dump_stats(prof_path)
    _write_report(profiler, text_path, TOP_ROWS)
    _render_flame_graph(prof_path, os.path.join(output_dir, f"{PROFILE_STEM}.svg"))

    print(f"Wrote profile stats to {prof_path} and report to {text_path}.")
    return text_path


def main() -> None:
    """Profile a real training run with the default configuration."""
    from .experiments import load_config

    run_profile(load_config())


if __name__ == "__main__":
    main()
