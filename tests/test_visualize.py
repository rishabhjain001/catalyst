"""Unit tests for the visualizer's precondition gate and failure rendering.

Covers two acceptance criteria of Requirement 21:

* Req 21.1 -- the visualizer runs only after all five
  ``results/experiment{N}.json`` files exist. Fewer than five must be refused
  with a descriptive error naming the missing experiments; all five present
  must let it run and produce PNGs.
* Req 21.4 -- where an experiment JSON records a failure or partial run, the
  visualizer renders the data that is present and marks the missing values
  rather than crashing or fabricating numbers.

These are example-based unit tests (not property tests). Every file written
goes under pytest's ``tmp_path`` so the real ``results/`` and ``plots/``
directories are never touched.
"""

import json
import os

import pytest

from catalyst import visualize
from catalyst.visualize import (
    check_preconditions,
    plot_interpolation_curve,
    plot_performance_table,
    plot_rank_allocation,
    visualize_all,
)


# The PNG filenames visualize_all writes, one per plot function (Req 21.2, 21.3).
EXPECTED_PNGS = (
    "adapter_heatmaps.png",
    "trainable_percent.png",
    "memory_comparison.png",
    "rank_ablation_loss.png",
    "inference_latency.png",
    "size_comparison.png",
    "interpolation_curve.png",
    "rank_allocation.png",
    "param_distribution.png",
    "performance_table.png",
)


def _write_result(results_dir, n, record):
    """Persist one experiment record as ``results_dir/experiment{n}.json``."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"experiment{n}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    return path


def _completed_record(n, metrics):
    """A minimal completed experiment record matching the results schema."""
    return {
        "experiment": n,
        "status": "completed",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:10:00",
        "duration_seconds": 600.0,
        "metrics": metrics,
        "failures": [],
    }


def _experiment1_metrics():
    return {"by_rank": {"4": {"trainable_percent": 0.5}, "8": {"trainable_percent": 0.9}}}


def _experiment2_metrics():
    return {
        "by_rank": {
            "4": {
                "validation_loss": 1.5,
                "adapter_size_bytes": 100000,
                "checkpoint_size_bytes": 5000000,
                "inference_memory": {
                    "base": {"peak_ram_mb": 100.0, "avg_ram_mb": 90.0},
                    "merged": {"peak_ram_mb": 110.0, "avg_ram_mb": 95.0},
                },
                "inference_latency": {
                    "base": {"mean": 0.2},
                    "merged": {"mean": 0.25},
                },
            },
            "8": {"validation_loss": 1.3},
        }
    }


def _experiment3_metrics():
    return {
        "by_targets": {
            "attn_only": {"trainable_percent": 0.4},
            "attn_mlp": {"trainable_percent": 0.8},
        }
    }


def _experiment4_metrics():
    return {
        "inference_memory": {
            "base": {"peak_ram_mb": 100.0, "avg_ram_mb": 90.0},
            "unmerged": {"peak_ram_mb": 120.0, "avg_ram_mb": 100.0},
            "merged": {"peak_ram_mb": 110.0, "avg_ram_mb": 95.0},
            "post_swap": {"peak_ram_mb": 115.0, "avg_ram_mb": 98.0},
        },
        "inference_latency": {
            "base": {"mean": 0.20},
            "unmerged": {"mean": 0.30},
            "merged": {"mean": 0.25},
            "post_swap": {"mean": 0.27},
        },
        "sentiment_by_alpha": [
            {"alpha": 0.0, "sentiment_score": 0.1},
            {"alpha": 0.5, "sentiment_score": 0.5},
            {"alpha": 1.0, "sentiment_score": 0.9},
        ],
    }


def _experiment5_metrics():
    return {
        "by_schedule": {
            "progressive": {
                "validation_loss": 1.1,
                "perplexity": 3.0,
                "bleu": 0.25,
                "rouge_l": 0.4,
                "sentiment_accuracy": 0.8,
                "trainable_params": 100000,
                "peak_ram_mb": 130.0,
                "throughput": 12.0,
                "rank_by_layer": {"0": 8, "1": 4, "2": 2},
                "per_layer_param_counts": {"0": 5000, "1": 3000, "2": 1000},
            },
            "uniform": {
                "validation_loss": 1.2,
                "perplexity": 3.2,
                "bleu": 0.22,
                "rouge_l": 0.38,
                "sentiment_accuracy": 0.78,
                "trainable_params": 110000,
                "peak_ram_mb": 132.0,
                "throughput": 11.5,
                "rank_by_layer": {"0": 4, "1": 4, "2": 4},
                "per_layer_param_counts": {"0": 3000, "1": 3000, "2": 3000},
            },
        }
    }


def _write_all_completed(results_dir):
    """Write all five JSONs as completed runs carrying realistic small metrics."""
    _write_result(results_dir, 1, _completed_record(1, _experiment1_metrics()))
    _write_result(results_dir, 2, _completed_record(2, _experiment2_metrics()))
    _write_result(results_dir, 3, _completed_record(3, _experiment3_metrics()))
    _write_result(results_dir, 4, _completed_record(4, _experiment4_metrics()))
    _write_result(results_dir, 5, _completed_record(5, _experiment5_metrics()))


# --------------------------------------------------------------------------- #
# Req 21.1 -- precondition gate
# --------------------------------------------------------------------------- #


def test_check_preconditions_refuses_when_jsons_missing(tmp_path):
    """Fewer than five JSONs is refused with an error naming the missing ones."""
    results_dir = str(tmp_path / "results")
    # Only experiments 1 and 2 present; 3, 4, 5 are missing.
    _write_result(results_dir, 1, _completed_record(1, _experiment1_metrics()))
    _write_result(results_dir, 2, _completed_record(2, _experiment2_metrics()))

    with pytest.raises(FileNotFoundError) as excinfo:
        check_preconditions(results_dir)

    message = str(excinfo.value)
    # The descriptive error names each missing experiment and refuses to run.
    assert "experiment3.json" in message
    assert "experiment4.json" in message
    assert "experiment5.json" in message


def test_visualize_all_refuses_when_jsons_missing(tmp_path):
    """visualize_all refuses and writes no PNGs when a JSON is absent."""
    results_dir = str(tmp_path / "results")
    plots_dir = str(tmp_path / "plots")
    # Four of five present -- experiment5.json is missing.
    _write_result(results_dir, 1, _completed_record(1, _experiment1_metrics()))
    _write_result(results_dir, 2, _completed_record(2, _experiment2_metrics()))
    _write_result(results_dir, 3, _completed_record(3, _experiment3_metrics()))
    _write_result(results_dir, 4, _completed_record(4, _experiment4_metrics()))

    with pytest.raises(FileNotFoundError) as excinfo:
        visualize_all(results_dir, plots_dir)

    assert "experiment5.json" in str(excinfo.value)
    # No plots are produced when the precondition fails.
    if os.path.isdir(plots_dir):
        assert os.listdir(plots_dir) == []


def test_check_preconditions_loads_all_five_when_present(tmp_path):
    """With all five present, the records load and no error is raised."""
    results_dir = str(tmp_path / "results")
    _write_all_completed(results_dir)

    results = check_preconditions(results_dir)

    assert set(results.keys()) == {1, 2, 3, 4, 5}
    assert results[4]["metrics"]["sentiment_by_alpha"][0]["alpha"] == 0.0


def test_visualize_all_produces_pngs_when_all_present(tmp_path):
    """All five present lets visualize_all run and write every expected PNG."""
    results_dir = str(tmp_path / "results")
    plots_dir = str(tmp_path / "plots")
    _write_all_completed(results_dir)

    visualize_all(results_dir, plots_dir)

    for filename in EXPECTED_PNGS:
        path = os.path.join(plots_dir, filename)
        assert os.path.isfile(path), f"expected {filename} to be produced"
        assert os.path.getsize(path) > 0


# --------------------------------------------------------------------------- #
# Req 21.4 -- failed / partial JSONs render available data, mark missing
# --------------------------------------------------------------------------- #


def test_visualize_all_handles_failed_and_partial_inputs(tmp_path):
    """A failed and a partial JSON still yield every PNG without crashing.

    Experiments 1-3 complete with data; Experiment 4 is ``failed`` with its
    sentiment_by_alpha and inference metrics missing; Experiment 5 is
    ``partial`` with some by_schedule entries null. The visualizer must render
    the available data, mark the gaps, and never raise.
    """
    results_dir = str(tmp_path / "results")
    plots_dir = str(tmp_path / "plots")

    _write_result(results_dir, 1, _completed_record(1, _experiment1_metrics()))
    _write_result(results_dir, 2, _completed_record(2, _experiment2_metrics()))
    _write_result(results_dir, 3, _completed_record(3, _experiment3_metrics()))

    # Experiment 4 failed: metrics present but the key panels' source data is null.
    failed4 = {
        "experiment": 4,
        "status": "failed",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:01:00",
        "duration_seconds": 60.0,
        "metrics": {
            "sentiment_by_alpha": None,
            "inference_memory": None,
            "inference_latency": None,
        },
        "failures": [
            {"metric": "sentiment_by_alpha", "reason": "interpolation sweep crashed"}
        ],
    }
    _write_result(results_dir, 4, failed4)

    # Experiment 5 partial: one schedule has data, another is null.
    partial5 = {
        "experiment": 5,
        "status": "partial",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T00:05:00",
        "duration_seconds": 300.0,
        "metrics": {
            "by_schedule": {
                "progressive": {
                    "validation_loss": 1.1,
                    "perplexity": None,
                    "bleu": None,
                    "rouge_l": None,
                    "sentiment_accuracy": 0.8,
                    "trainable_params": 100000,
                    "peak_ram_mb": None,
                    "throughput": None,
                    "rank_by_layer": {"0": 8, "1": 4, "2": 2},
                    "per_layer_param_counts": {"0": 5000, "1": 3000, "2": 1000},
                },
                "inverse": None,
                "uniform": None,
            }
        },
        "failures": [
            {"metric": "by_schedule.uniform", "reason": "schedule run failed"}
        ],
    }
    _write_result(results_dir, 5, partial5)

    # The whole render must not raise for failed/partial inputs.
    visualize_all(results_dir, plots_dir)

    # Every expected PNG is still produced, including those whose source data
    # was missing (interpolation_curve, performance_table, rank_allocation).
    for filename in EXPECTED_PNGS:
        path = os.path.join(plots_dir, filename)
        assert os.path.isfile(path), f"expected {filename} despite missing data"
        assert os.path.getsize(path) > 0


def test_plot_interpolation_curve_handles_missing_data(tmp_path):
    """plot_interpolation_curve still returns a path and writes the file when empty.

    With Experiment 4's sentiment_by_alpha missing entirely, the helper marks
    the gap rather than fabricating a curve, but still produces its PNG.
    """
    plots_dir = str(tmp_path / "plots")
    results = {4: {"status": "failed", "metrics": {}}}

    path = plot_interpolation_curve(results, plots_dir)

    assert path.endswith("interpolation_curve.png")
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_plot_performance_table_handles_missing_schedules(tmp_path):
    """plot_performance_table writes a marked placeholder when Experiment 5 is empty."""
    plots_dir = str(tmp_path / "plots")
    results = {5: {"status": "failed", "metrics": {}}}

    path = plot_performance_table(results, plots_dir)

    assert path.endswith("performance_table.png")
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_plot_rank_allocation_handles_missing_schedules(tmp_path):
    """plot_rank_allocation writes a marked placeholder when no schedule has data."""
    plots_dir = str(tmp_path / "plots")
    results = {5: {"status": "partial", "metrics": {"by_schedule": {"inverse": None}}}}

    path = plot_rank_allocation(results, plots_dir)

    assert path.endswith("rank_allocation.png")
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 0


def test_format_cell_marks_missing_value():
    """A None metric value renders as the literal 'missing' marker, not a number."""
    assert visualize._format_cell(None) == "missing"
    assert visualize._format_cell(1.23456) == "1.235"
    assert visualize._format_cell(42) == "42"
