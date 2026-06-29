"""Visualizations rendered from the persisted experiment JSONs (Req 21).

The visualizer runs only after all five ``results/experiment{N}.json`` files
exist, each carrying real measured numbers or a documented failure (Req 21.1).
It then renders every plot the requirements name (Req 21.2, 21.3) into
``plots_dir`` as PNGs. Where a JSON records a failed or partial experiment, or a
particular metric is missing, the corresponding panel renders the data that is
present and marks the missing values rather than fabricating numbers (Req 21.4).

The experiment JSONs store metrics, not raw adapter matrices, so the A/B
heatmaps and the ``B @ A`` singular-value spectra source their matrices from the
adapter-only artifacts the trainer wrote under ``results/`` (Req 12.5): each
rank in Experiment 2 saves to ``results/experiment2_rank{r}/`` a directory of
per-module ``.pt`` files holding ``A`` and ``B``. When those artifacts are
absent (a failed run, or a results directory without trained adapters) the
heatmap panels render a clearly-marked "data unavailable" placeholder instead of
inventing matrices.

Every per-plot function accepts already-loaded results dicts so it is testable
in isolation, and ``check_preconditions`` isolates the five-JSON gate so the
precondition can be tested without rendering anything.
"""

import glob
import json
import os

import matplotlib

# Render headlessly: the visualizer writes PNG files and never opens a window,
# so the non-interactive Agg backend must be selected before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np  # noqa: E402

# The five experiments whose JSONs must all exist before any plot is drawn
# (Req 21.1).
REQUIRED_EXPERIMENTS = (1, 2, 3, 4, 5)

# The ranks whose adapter A/B matrices and B@A spectra are visualized (Req
# 21.2). These mirror Experiment 2's trained ranks, whose adapter artifacts are
# the matrix source.
HEATMAP_RANKS = (4, 8, 16)

# The four inference conditions reported for memory and latency, matching the
# evaluator's condition order (Req 13.2, 13.3, 21.2).
CONDITIONS = ("base", "unmerged", "merged", "post_swap")

# A muted red used consistently to mark a missing or unmeasurable value so the
# absence is obvious at a glance rather than mistaken for a real datum.
MISSING_COLOR = "#c0392b"


def _results_filename(n: int) -> str:
    return f"experiment{n}.json"


def check_preconditions(results_dir: str) -> dict:
    """Return the five loaded experiment records, or raise if any JSON is missing.

    Enforces the Req 21.1 gate: every ``results/experiment{N}.json`` for N in
    1..5 must exist before any plot is drawn. A present JSON whose ``status`` is
    ``failed`` or ``partial`` is accepted here (its missing values are marked
    later, Req 21.4); only an absent file is refused, with a descriptive error
    naming the missing experiments.
    """
    missing = []
    results: dict = {}
    for n in REQUIRED_EXPERIMENTS:
        path = os.path.join(results_dir, _results_filename(n))
        if not os.path.isfile(path):
            missing.append(n)
            continue
        with open(path, "r", encoding="utf-8") as handle:
            results[n] = json.load(handle)

    if missing:
        names = ", ".join(_results_filename(n) for n in missing)
        raise FileNotFoundError(
            "Visualizer requires all five experiment JSONs before running; "
            f"missing: {names}. Run experiments.run_all first (Req 21.1)."
        )
    return results


def _metrics(results: dict, n: int) -> dict:
    """The metrics mapping for experiment ``n``, or an empty dict when absent."""
    record = results.get(n) or {}
    return record.get("metrics") or {}


def _status(results: dict, n: int) -> str:
    """The recorded status string for experiment ``n`` (``"unknown"`` if absent)."""
    record = results.get(n) or {}
    return record.get("status", "unknown")


def _annotate_unavailable(ax, message: str) -> None:
    """Mark an axes as having no data, rather than leaving it blank or faked.

    Used for any panel whose source metric is missing or whose experiment
    failed, so the figure shows the gap explicitly (Req 21.4).
    """
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=MISSING_COLOR,
        fontsize=10,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig, plots_dir: str, filename: str) -> str:
    """Write ``fig`` as a PNG into ``plots_dir`` and return the path written."""
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_adapter_matrices(results_dir: str, rank: int):
    """Return ``(A, B, source)`` for one layer's rank-``rank`` adapter, or ``None``.

    Sources the matrices from the adapter artifacts Experiment 2 saved under
    ``results/experiment2_rank{rank}/`` (Req 12.5): the first ``.pt`` file in
    that directory (sorted for determinism) stands in for "one layer". Returns
    the ``A`` and ``B`` arrays as numpy plus the file the matrices came from, or
    ``None`` when torch is unavailable, the directory is absent, or the file
    carries no ``A``/``B`` so the caller can draw a marked placeholder rather
    than fabricate matrices.
    """
    directory = os.path.join(results_dir, f"experiment2_rank{rank}")
    if not os.path.isdir(directory):
        return None
    files = sorted(glob.glob(os.path.join(directory, "*.pt")))
    if not files:
        return None

    try:
        import torch
    except ImportError:
        return None

    try:
        artifact = torch.load(files[0], map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError):
        return None

    if not isinstance(artifact, dict) or "A" not in artifact or "B" not in artifact:
        return None

    matrix_a = np.asarray(artifact["A"], dtype=float)
    matrix_b = np.asarray(artifact["B"], dtype=float)
    return matrix_a, matrix_b, os.path.basename(files[0])


def plot_heatmaps(results: dict, plots_dir: str, results_dir: str) -> str:
    """A/B matrix heatmaps and the ``B @ A`` singular-value spectrum per rank.

    Draws one row per rank in :data:`HEATMAP_RANKS` with three panels — the
    ``A`` heatmap, the ``B`` heatmap, and the singular-value spectrum of
    ``B @ A`` (Req 21.2). Matrices come from the saved adapter artifacts via
    :func:`_load_adapter_matrices`; a rank whose artifact is absent renders
    marked "data unavailable" placeholders for its three panels rather than
    fabricating matrices (Req 21.4).
    """
    ranks = list(HEATMAP_RANKS)
    fig, axes = plt.subplots(
        len(ranks), 3, figsize=(12, 3.2 * len(ranks)), squeeze=False
    )

    for row, rank in enumerate(ranks):
        ax_a, ax_b, ax_s = axes[row]
        loaded = _load_adapter_matrices(results_dir, rank)
        if loaded is None:
            for ax in (ax_a, ax_b, ax_s):
                _annotate_unavailable(ax, f"rank {rank}\nadapter data unavailable")
            ax_a.set_title(f"A (rank {rank}) — unavailable")
            ax_b.set_title(f"B (rank {rank}) — unavailable")
            ax_s.set_title(f"B@A spectrum (rank {rank}) — unavailable")
            continue

        matrix_a, matrix_b, source = loaded
        image_a = ax_a.imshow(matrix_a, aspect="auto", cmap="viridis")
        ax_a.set_title(f"A (rank {rank})  [{source}]")
        fig.colorbar(image_a, ax=ax_a, fraction=0.046, pad=0.04)

        image_b = ax_b.imshow(matrix_b, aspect="auto", cmap="viridis")
        ax_b.set_title(f"B (rank {rank})")
        fig.colorbar(image_b, ax=ax_b, fraction=0.046, pad=0.04)

        # B @ A is the logical low-rank update whose spectrum exposes the
        # effective rank: at most ``rank`` singular values are non-zero.
        delta = matrix_b @ matrix_a
        singular = np.linalg.svd(delta, compute_uv=False)
        ax_s.plot(range(1, len(singular) + 1), singular, marker="o")
        ax_s.set_title(f"B@A singular values (rank {rank})")
        ax_s.set_xlabel("index")
        ax_s.set_ylabel("singular value")
        ax_s.grid(True, alpha=0.3)

    fig.suptitle("Adapter A/B heatmaps and B@A spectra", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, plots_dir, "adapter_heatmaps.png")


def _trainable_percent_by_rank(results: dict) -> dict:
    """Map rank label to trainable percent from Experiment 1's ``by_rank``."""
    by_rank = _metrics(results, 1).get("by_rank") or {}
    percents = {}
    for key, block in by_rank.items():
        if isinstance(block, dict) and block.get("trainable_percent") is not None:
            percents[key] = block["trainable_percent"]
    return percents


def _trainable_percent_by_targets(results: dict) -> dict:
    """Map target-set label to trainable percent from Experiment 3's ``by_targets``."""
    by_targets = _metrics(results, 3).get("by_targets") or {}
    percents = {}
    for key, block in by_targets.items():
        if isinstance(block, dict) and block.get("trainable_percent") is not None:
            percents[key] = block["trainable_percent"]
    return percents


def plot_trainable_percent(results: dict, plots_dir: str) -> str:
    """Trainable-parameter-percentage bar charts for Experiments 1 and 3 (Req 21.2)."""
    fig, (ax_rank, ax_targets) = plt.subplots(1, 2, figsize=(12, 5))

    by_rank = _trainable_percent_by_rank(results)
    if by_rank:
        labels = list(by_rank.keys())
        ax_rank.bar(labels, [by_rank[k] for k in labels], color="#2980b9")
        ax_rank.set_title("Experiment 1 — trainable % by rank")
        ax_rank.set_xlabel("rank")
        ax_rank.set_ylabel("trainable parameters (%)")
    else:
        _annotate_unavailable(ax_rank, "Experiment 1\ntrainable % unavailable")
        ax_rank.set_title("Experiment 1 — trainable % by rank")

    by_targets = _trainable_percent_by_targets(results)
    if by_targets:
        labels = list(by_targets.keys())
        ax_targets.bar(labels, [by_targets[k] for k in labels], color="#27ae60")
        ax_targets.set_title("Experiment 3 — trainable % by target set")
        ax_targets.set_xlabel("target modules")
        ax_targets.set_ylabel("trainable parameters (%)")
        ax_targets.tick_params(axis="x", rotation=20)
    else:
        _annotate_unavailable(ax_targets, "Experiment 3\ntrainable % unavailable")
        ax_targets.set_title("Experiment 3 — trainable % by target set")

    fig.tight_layout()
    return _save(fig, plots_dir, "trainable_percent.png")


def _memory_conditions(results: dict) -> dict:
    """Per-condition ``{peak_ram_mb, avg_ram_mb}`` from Experiment 4, then 2.

    Experiment 4 profiles the four conditions directly. When its block is
    missing, the first measured Experiment 2 rank with an ``inference_memory``
    block is used as a fallback so a memory comparison can still be drawn.
    """
    memory = _metrics(results, 4).get("inference_memory")
    if isinstance(memory, dict) and memory:
        return memory

    by_rank = _metrics(results, 2).get("by_rank") or {}
    for block in by_rank.values():
        if isinstance(block, dict) and isinstance(block.get("inference_memory"), dict):
            return block["inference_memory"]
    return {}


def plot_memory_comparison(results: dict, plots_dir: str) -> str:
    """Peak/average RAM across base, unmerged, merged, and post-swap (Req 21.2)."""
    memory = _memory_conditions(results)
    fig, ax = plt.subplots(figsize=(9, 5))

    peak = []
    avg = []
    present = []
    for condition in CONDITIONS:
        block = memory.get(condition) if isinstance(memory, dict) else None
        if isinstance(block, dict) and block.get("peak_ram_mb") is not None:
            present.append(condition)
            peak.append(block.get("peak_ram_mb"))
            avg.append(block.get("avg_ram_mb"))

    if present:
        positions = np.arange(len(present))
        width = 0.4
        ax.bar(
            positions - width / 2, peak, width, label="peak RAM (MB)", color="#8e44ad"
        )
        ax.bar(positions + width / 2, avg, width, label="avg RAM (MB)", color="#16a085")
        ax.set_xticks(positions)
        ax.set_xticklabels(present)
        ax.set_ylabel("RAM (MB)")
        ax.legend()
        missing = [c for c in CONDITIONS if c not in present]
        title = "Inference memory by condition"
        if missing:
            title += f"  (missing: {', '.join(missing)})"
        ax.set_title(title)
    else:
        _annotate_unavailable(ax, "inference memory\nunavailable for all conditions")
        ax.set_title("Inference memory by condition")

    fig.tight_layout()
    return _save(fig, plots_dir, "memory_comparison.png")


def plot_loss_curves(results: dict, plots_dir: str) -> str:
    """Rank-ablation training loss view from Experiment 2 (Req 21.2).

    Per-step training loss is not persisted in the results JSON, so the
    rank-ablation comparison uses each rank's ``validation_loss`` as the loss
    measure and labels the axis accordingly; ranks whose value is missing are
    marked rather than dropped silently (Req 21.4).
    """
    by_rank = _metrics(results, 2).get("by_rank") or {}
    fig, ax = plt.subplots(figsize=(9, 5))

    labels = []
    losses = []
    missing = []
    for key in sorted(
        by_rank, key=lambda value: int(value) if value.isdigit() else value
    ):
        block = by_rank[key]
        if isinstance(block, dict) and block.get("validation_loss") is not None:
            labels.append(key)
            losses.append(block["validation_loss"])
        else:
            missing.append(key)

    if labels:
        ax.plot(labels, losses, marker="o", color="#d35400")
        ax.set_title("Rank ablation — validation loss per rank")
        ax.set_xlabel("rank")
        ax.set_ylabel("validation loss (per-step loss not stored)")
        ax.grid(True, alpha=0.3)
        if missing:
            ax.text(
                0.99,
                0.01,
                f"missing ranks: {', '.join(missing)}",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                color=MISSING_COLOR,
                fontsize=9,
            )
    else:
        _annotate_unavailable(ax, "rank-ablation loss\nunavailable")
        ax.set_title("Rank ablation — validation loss per rank")

    fig.tight_layout()
    return _save(fig, plots_dir, "rank_ablation_loss.png")


def _latency_conditions(results: dict) -> dict:
    """Per-condition latency stats from Experiment 4, then Experiment 2."""
    latency = _metrics(results, 4).get("inference_latency")
    if isinstance(latency, dict) and latency:
        return latency

    by_rank = _metrics(results, 2).get("by_rank") or {}
    for block in by_rank.values():
        if isinstance(block, dict) and isinstance(block.get("inference_latency"), dict):
            return block["inference_latency"]
    return {}


def plot_latency(results: dict, plots_dir: str) -> str:
    """Mean inference latency across the four conditions (Req 21.2)."""
    latency = _latency_conditions(results)
    fig, ax = plt.subplots(figsize=(9, 5))

    present = []
    means = []
    for condition in CONDITIONS:
        block = latency.get(condition) if isinstance(latency, dict) else None
        if isinstance(block, dict) and block.get("mean") is not None:
            present.append(condition)
            means.append(block["mean"])

    if present:
        ax.bar(present, means, color="#2c3e50")
        ax.set_title("Inference latency (mean) by condition")
        ax.set_xlabel("condition")
        ax.set_ylabel("latency (seconds)")
        missing = [c for c in CONDITIONS if c not in present]
        if missing:
            ax.text(
                0.99,
                0.99,
                f"missing: {', '.join(missing)}",
                ha="right",
                va="top",
                transform=ax.transAxes,
                color=MISSING_COLOR,
                fontsize=9,
            )
    else:
        _annotate_unavailable(ax, "inference latency\nunavailable for all conditions")
        ax.set_title("Inference latency (mean) by condition")

    fig.tight_layout()
    return _save(fig, plots_dir, "inference_latency.png")


def _adapter_and_checkpoint_size(results: dict):
    """Return ``(adapter_bytes, checkpoint_bytes)`` from the first source carrying them.

    Experiment 2's per-rank blocks carry ``adapter_size_bytes`` and
    ``checkpoint_size_bytes`` (the evaluator records both); the first measured
    rank with either value supplies the comparison. Experiment 4's top-level
    metrics are used as a fallback. A value that is absent stays ``None`` so the
    chart marks it.
    """
    by_rank = _metrics(results, 2).get("by_rank") or {}
    for block in by_rank.values():
        if isinstance(block, dict) and (
            block.get("adapter_size_bytes") is not None
            or block.get("checkpoint_size_bytes") is not None
        ):
            return block.get("adapter_size_bytes"), block.get("checkpoint_size_bytes")

    metrics4 = _metrics(results, 4)
    return metrics4.get("adapter_size_bytes"), metrics4.get("checkpoint_size_bytes")


def plot_size_comparison(results: dict, plots_dir: str) -> str:
    """Adapter artifact size versus full checkpoint size (Req 21.2)."""
    adapter_bytes, checkpoint_bytes = _adapter_and_checkpoint_size(results)
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = []
    values = []
    colors = []
    if adapter_bytes is not None:
        labels.append("adapter artifact")
        values.append(adapter_bytes / 1e6)
        colors.append("#27ae60")
    if checkpoint_bytes is not None:
        labels.append("full checkpoint")
        values.append(checkpoint_bytes / 1e6)
        colors.append("#c0392b")

    if values:
        ax.bar(labels, values, color=colors)
        ax.set_title("Adapter artifact vs full checkpoint size")
        ax.set_ylabel("size (MB)")
        missing = []
        if adapter_bytes is None:
            missing.append("adapter size")
        if checkpoint_bytes is None:
            missing.append("checkpoint size")
        if missing:
            ax.text(
                0.5,
                0.95,
                f"missing: {', '.join(missing)}",
                ha="center",
                va="top",
                transform=ax.transAxes,
                color=MISSING_COLOR,
                fontsize=9,
            )
    else:
        _annotate_unavailable(ax, "adapter and checkpoint\nsizes unavailable")
        ax.set_title("Adapter artifact vs full checkpoint size")

    fig.tight_layout()
    return _save(fig, plots_dir, "size_comparison.png")


def plot_interpolation_curve(results: dict, plots_dir: str) -> str:
    """Continuous sentiment-versus-alpha curve from Experiment 4 (Req 21.3, 17.5)."""
    curve = _metrics(results, 4).get("sentiment_by_alpha")
    fig, ax = plt.subplots(figsize=(8, 5))

    points = []
    if isinstance(curve, list):
        for point in curve:
            if (
                isinstance(point, dict)
                and point.get("alpha") is not None
                and point.get("sentiment_score") is not None
            ):
                points.append((point["alpha"], point["sentiment_score"]))

    if points:
        points.sort(key=lambda pair: pair[0])
        alphas = [pair[0] for pair in points]
        scores = [pair[1] for pair in points]
        ax.plot(alphas, scores, marker="o", color="#2980b9")
        ax.set_title("Sentiment vs interpolation alpha (Experiment 4)")
        ax.set_xlabel("alpha (0 = negative adapter, 1 = positive adapter)")
        ax.set_ylabel("sentiment score")
        ax.grid(True, alpha=0.3)
    else:
        _annotate_unavailable(ax, "sentiment-by-alpha\ncurve unavailable")
        ax.set_title("Sentiment vs interpolation alpha (Experiment 4)")

    fig.tight_layout()
    return _save(fig, plots_dir, "interpolation_curve.png")


def _schedule_blocks(results: dict) -> dict:
    """The measured ``by_schedule`` blocks from Experiment 5 (empty when absent)."""
    by_schedule = _metrics(results, 5).get("by_schedule") or {}
    return {
        name: block for name, block in by_schedule.items() if isinstance(block, dict)
    }


def plot_rank_allocation(results: dict, plots_dir: str) -> str:
    """Per-layer rank allocation diagram for each Experiment 5 schedule (Req 21.3)."""
    schedules = _schedule_blocks(results)
    fig, ax = plt.subplots(figsize=(10, 5))

    drawn = False
    for name, block in schedules.items():
        rank_by_layer = block.get("rank_by_layer")
        if not isinstance(rank_by_layer, dict) or not rank_by_layer:
            continue
        layers = sorted(rank_by_layer, key=lambda value: int(value))
        ranks = [rank_by_layer[layer] for layer in layers]
        ax.plot([int(layer) for layer in layers], ranks, marker="o", label=name)
        drawn = True

    if drawn:
        ax.set_title("Experiment 5 — per-layer rank allocation")
        ax.set_xlabel("layer index")
        ax.set_ylabel("rank")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        _annotate_unavailable(ax, "per-layer rank\nallocation unavailable")
        ax.set_title("Experiment 5 — per-layer rank allocation")

    fig.tight_layout()
    return _save(fig, plots_dir, "rank_allocation.png")


def plot_param_distribution(results: dict, plots_dir: str) -> str:
    """Per-layer parameter distribution across schedules for Experiment 5 (Req 21.3)."""
    schedules = _schedule_blocks(results)
    fig, ax = plt.subplots(figsize=(11, 5))

    series = {}
    layer_set = set()
    for name, block in schedules.items():
        counts = block.get("per_layer_param_counts")
        if isinstance(counts, dict) and counts:
            series[name] = counts
            layer_set.update(int(layer) for layer in counts)

    if series:
        layers = sorted(layer_set)
        names = list(series.keys())
        width = 0.8 / max(len(names), 1)
        positions = np.arange(len(layers))
        for offset, name in enumerate(names):
            counts = series[name]
            values = [counts.get(str(layer), 0) for layer in layers]
            ax.bar(positions + offset * width, values, width, label=name)
        ax.set_xticks(positions + width * (len(names) - 1) / 2)
        ax.set_xticklabels([str(layer) for layer in layers])
        ax.set_title("Experiment 5 — per-layer trainable parameters by schedule")
        ax.set_xlabel("layer index")
        ax.set_ylabel("trainable parameters")
        ax.legend()
    else:
        _annotate_unavailable(ax, "per-layer parameter\ndistribution unavailable")
        ax.set_title("Experiment 5 — per-layer trainable parameters by schedule")

    fig.tight_layout()
    return _save(fig, plots_dir, "param_distribution.png")


# The performance metrics tabulated for the Experiment 5 schedule comparison,
# each paired with the column header shown in the rendered table (Req 21.3).
PERFORMANCE_FIELDS = (
    ("validation_loss", "val loss"),
    ("perplexity", "perplexity"),
    ("bleu", "BLEU"),
    ("rouge_l", "ROUGE-L"),
    ("sentiment_accuracy", "sentiment acc"),
    ("trainable_params", "trainable params"),
    ("peak_ram_mb", "peak RAM (MB)"),
    ("throughput", "throughput"),
)


def _format_cell(value) -> str:
    """Render a metric value for the table, marking a missing one as 'missing'."""
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def plot_performance_table(results: dict, plots_dir: str) -> str:
    """Performance comparison table across Experiment 5 schedules (Req 21.3)."""
    schedules = _schedule_blocks(results)
    fig, ax = plt.subplots(figsize=(11, 0.6 * (len(PERFORMANCE_FIELDS) + 2) + 1))
    ax.axis("off")

    if schedules:
        names = list(schedules.keys())
        column_labels = ["metric"] + names
        rows = []
        for field, header in PERFORMANCE_FIELDS:
            row = [header]
            for name in names:
                row.append(_format_cell(schedules[name].get(field)))
            rows.append(row)

        table = ax.table(
            cellText=rows,
            colLabels=column_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        # Highlight any cell that reports a missing value so the gap is obvious.
        for (row_index, col_index), cell in table.get_celld().items():
            if row_index > 0 and cell.get_text().get_text() == "missing":
                cell.get_text().set_color(MISSING_COLOR)
        ax.set_title("Experiment 5 — performance comparison", pad=20)
    else:
        _annotate_unavailable(ax, "performance comparison\ntable unavailable")
        ax.set_title("Experiment 5 — performance comparison")

    return _save(fig, plots_dir, "performance_table.png")


def visualize_all(results_dir: str, plots_dir: str) -> None:
    """Render every required plot from the five experiment JSONs (Req 21).

    Refuses to run unless all five ``experiment{N}.json`` files exist (Req 21.1);
    on success it loads them and draws each plot into ``plots_dir``. Each plot
    function renders the data present and marks anything missing (Req 21.4), so a
    failed or partial experiment still yields annotated figures rather than a
    crash or fabricated numbers.
    """
    results = check_preconditions(results_dir)
    os.makedirs(plots_dir, exist_ok=True)

    plot_heatmaps(results, plots_dir, results_dir)
    plot_trainable_percent(results, plots_dir)
    plot_memory_comparison(results, plots_dir)
    plot_loss_curves(results, plots_dir)
    plot_latency(results, plots_dir)
    plot_size_comparison(results, plots_dir)
    plot_interpolation_curve(results, plots_dir)
    plot_rank_allocation(results, plots_dir)
    plot_param_distribution(results, plots_dir)
    plot_performance_table(results, plots_dir)
