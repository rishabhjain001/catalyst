"""Unit tests for the generic experiment wrapper and the result writers.

Covers task 9.2: ``run_experiment`` times an experiment body, captures metadata,
and degrades gracefully on failure; ``write_results`` persists the results
schema; ``write_manifest`` emits ``manifest_version: 1`` with one entry per
record. All output is redirected to a temp directory so the real ``results/``
stays clean.
"""

import json
import os
from datetime import datetime

import pytest

from catalyst.experiments import (
    PartialExperiment,
    RESULTS_SCHEMA_KEYS,
    null_metric,
    run_experiment,
    write_manifest,
    write_results,
)


def _config(tmp_path):
    return {"model_name": "gpt2", "seed": 7, "results_dir": str(tmp_path)}


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _is_iso8601(value):
    # ``fromisoformat`` accepts exactly the ISO 8601 strings ``isoformat`` emits.
    datetime.fromisoformat(value)
    return True


def test_successful_body_records_completed_and_writes_json(tmp_path):
    config = _config(tmp_path)

    def body(_config):
        return {"metrics": {"trainable_params": 1234, "trainable_percent": 0.5}}

    record = run_experiment(1, config, body=body)

    assert record["status"] == "completed"
    assert record["metrics"]["trainable_params"] == 1234
    assert record["failures"] == []
    assert _is_iso8601(record["start_time"])
    assert _is_iso8601(record["end_time"])

    # The JSON and the metadata file were both written under the temp dir.
    results_file = tmp_path / "experiment1.json"
    metadata_file = tmp_path / "experiment1_metadata.json"
    assert results_file.exists()
    assert metadata_file.exists()

    written = _read(results_file)
    assert set(written) == set(RESULTS_SCHEMA_KEYS)
    assert written["status"] == "completed"


def test_failing_body_records_failed_and_still_writes_json(tmp_path):
    config = _config(tmp_path)

    def body(_config):
        raise MemoryError("ran out of memory")

    record = run_experiment(2, config, body=body)

    # The wrapper never re-raises; it records failed and writes the JSON.
    assert record["status"] == "failed"
    reasons = [entry["reason"] for entry in record["failures"]]
    assert any("out of memory" in reason for reason in reasons)

    written = _read(tmp_path / "experiment2.json")
    assert written["status"] == "failed"
    assert _is_iso8601(written["start_time"])


def test_partial_body_keeps_usable_metrics(tmp_path):
    config = _config(tmp_path)

    def body(_config):
        raise PartialExperiment(
            "runtime budget exceeded before latency was measured",
            metrics={"validation_loss": 2.5, "inference_latency": None},
            failures=[{"metric": "inference_latency", "reason": "budget exceeded"}],
        )

    record = run_experiment(3, config, body=body)

    assert record["status"] == "partial"
    assert record["metrics"]["validation_loss"] == 2.5
    assert record["metrics"]["inference_latency"] is None
    reasons = [entry["reason"] for entry in record["failures"]]
    assert "budget exceeded" in reasons


def test_body_with_failures_is_partial(tmp_path):
    config = _config(tmp_path)

    def body(_config):
        metrics = {}
        failures = []
        metrics["validation_loss"] = 2.0
        null_metric(metrics, failures, "gemini", "no samples provided")
        return {"metrics": metrics, "failures": failures}

    record = run_experiment(4, config, body=body)

    assert record["status"] == "partial"
    assert record["metrics"]["gemini"] is None


def test_missing_body_fails_gracefully(tmp_path):
    record = run_experiment(99, _config(tmp_path), body=None)

    assert record["status"] == "failed"
    assert (tmp_path / "experiment99.json").exists()


def test_duration_equals_end_minus_start(tmp_path):
    record = run_experiment(1, _config(tmp_path), body=lambda c: {"metrics": {}})

    start = datetime.fromisoformat(record["start_time"])
    end = datetime.fromisoformat(record["end_time"])
    expected = (end - start).total_seconds()
    assert record["duration_seconds"] == pytest.approx(expected, abs=1e-9)
    assert record["duration_seconds"] >= 0.0


def test_write_results_persists_only_schema_keys(tmp_path):
    config = _config(tmp_path)
    record = {key: None for key in RESULTS_SCHEMA_KEYS}
    record["experiment"] = 1
    record["extra_bookkeeping"] = "should not be written"

    path = write_results(1, record, config)
    written = _read(path)

    assert "extra_bookkeeping" not in written
    assert set(written) == set(RESULTS_SCHEMA_KEYS)


def test_write_manifest_has_version_one_and_one_entry_per_record(tmp_path):
    config = _config(tmp_path)
    records = [
        run_experiment(1, config, body=lambda c: {"metrics": {"a": 1}}),
        run_experiment(2, config, body=lambda c: (_ for _ in ()).throw(RuntimeError("x"))),
    ]

    path = write_manifest(records, config)
    manifest = _read(path)

    assert manifest["manifest_version"] == 1
    assert len(manifest["experiments"]) == 2

    first = manifest["experiments"][0]
    assert first["experiment"] == 1
    assert first["status"] == "completed"
    assert first["results_path"].endswith("experiment1.json")
    assert first["metadata_path"].endswith("experiment1_metadata.json")
    assert _is_iso8601(first["start_time"])
    assert _is_iso8601(first["end_time"])

    second = manifest["experiments"][1]
    assert second["status"] == "failed"
