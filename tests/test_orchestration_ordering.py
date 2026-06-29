"""Unit tests for ``run_all`` orchestration ordering and continuation.

Covers task 9.14: the runner drives the five experiments strictly 1→5, writes
each experiment's JSON before the next begins, continues past a failed
experiment, and maps body outcomes onto the ``failed``/``partial``/``completed``
statuses (Req 11.2, 11.3, 11.4, 19.1, 19.2).

These are plain unit tests, not property tests. ``run_all`` is isolated from the
real, heavy experiment bodies two ways: the ``EXPERIMENT_REGISTRY`` entries for
1..5 are replaced with fast fakes (via ``monkeypatch.setitem``) that record their
call order and can be made to succeed/partial/fail, and
``run_startup_estimate`` is monkeypatched to a no-op so no model is loaded and no
network is touched. Output is redirected to ``tmp_path`` through
``config["results_dir"]`` so the real ``results/`` stays clean. Metadata capture
still runs against the real ``configs/default.yaml`` (a fast local file read).
"""

import json

import pytest

from catalyst import experiments
from catalyst.experiments import (
    EXPERIMENT_ORDER,
    PartialExperiment,
    run_all,
)


def _config(tmp_path):
    return {"model_name": "gpt2", "seed": 7, "results_dir": str(tmp_path)}


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def isolate_startup(monkeypatch):
    """Replace the startup runtime estimate with a no-op (no model load/network)."""
    monkeypatch.setattr(experiments, "run_startup_estimate", lambda config: None)


def _install_bodies(monkeypatch, bodies):
    """Register the given ``{n: body}`` fakes in the experiment registry."""
    for n, body in bodies.items():
        monkeypatch.setitem(experiments.EXPERIMENT_REGISTRY, n, body)


# Feature: catalyst-lora, Task 9.14
# Validates: Requirements 19.1
def test_runs_experiments_in_strict_1_to_5_order(tmp_path, monkeypatch, isolate_startup):
    """The runner calls the five bodies in exactly the order 1, 2, 3, 4, 5."""
    call_order = []

    def make_body(n):
        def body(_config):
            call_order.append(n)
            return {"metrics": {"n": n}}

        return body

    _install_bodies(monkeypatch, {n: make_body(n) for n in EXPERIMENT_ORDER})

    run_all(_config(tmp_path))

    assert call_order == [1, 2, 3, 4, 5]


# Feature: catalyst-lora, Task 9.14
# Validates: Requirements 19.2
def test_each_json_written_before_next_experiment_starts(
    tmp_path, monkeypatch, isolate_startup
):
    """When body ``n`` runs, the previous experiment's JSON already exists on disk.

    ``run_experiment`` writes ``experiment{n}.json`` after its body returns, so by
    the time body ``n`` runs, ``experiment{n-1}.json`` must be present. Each fake
    records whether the prior file existed; raising inside a body would be
    swallowed by the graceful-failure wrapper, so the observation is recorded and
    asserted afterward instead.
    """
    prior_exists = {}

    def make_body(n):
        def body(_config):
            prev = tmp_path / f"experiment{n - 1}.json"
            prior_exists[n] = prev.exists()
            return {"metrics": {"n": n}}

        return body

    _install_bodies(monkeypatch, {n: make_body(n) for n in EXPERIMENT_ORDER})

    run_all(_config(tmp_path))

    # Experiment 1 has no predecessor; each later experiment saw the prior JSON.
    assert prior_exists[1] is False
    for n in (2, 3, 4, 5):
        assert prior_exists[n] is True, f"experiment{n - 1}.json missing when {n} ran"

    # All five JSONs are present once the run finishes.
    for n in EXPERIMENT_ORDER:
        assert (tmp_path / f"experiment{n}.json").exists()


# Feature: catalyst-lora, Task 9.14
# Validates: Requirements 11.4, 19.2
def test_continues_after_a_failed_experiment(tmp_path, monkeypatch, isolate_startup):
    """A raising experiment is marked ``failed`` but the remaining ones still run."""
    call_order = []

    def make_body(n, raises=False):
        def body(_config):
            call_order.append(n)
            if raises:
                raise RuntimeError(f"experiment {n} blew up")
            return {"metrics": {"n": n}}

        return body

    bodies = {n: make_body(n, raises=(n == 2)) for n in EXPERIMENT_ORDER}
    _install_bodies(monkeypatch, bodies)

    run_all(_config(tmp_path))

    # Every experiment ran despite experiment 2 failing.
    assert call_order == [1, 2, 3, 4, 5]

    # The failed experiment's JSON records status "failed" with the reason.
    exp2 = _read(tmp_path / "experiment2.json")
    assert exp2["status"] == "failed"
    assert any("blew up" in entry["reason"] for entry in exp2["failures"])

    # Experiments after the failure were persisted and completed.
    for n in (3, 4, 5):
        record = _read(tmp_path / f"experiment{n}.json")
        assert record["status"] == "completed"
        assert (tmp_path / f"experiment{n}.json").exists()


# Feature: catalyst-lora, Task 9.14
# Validates: Requirements 11.2, 11.3
def test_status_mapping_failed_partial_completed(
    tmp_path, monkeypatch, isolate_startup
):
    """Body outcomes map onto failed / partial / completed in the written JSON.

    - A body that raises with no usable results → ``failed`` (Req 11.2).
    - A body that raises ``PartialExperiment`` carrying metrics → ``partial`` (Req 11.3).
    - A clean body → ``completed``.
    """

    def clean_body(_config):
        return {"metrics": {"value": 1}}

    def failing_body(_config):
        raise RuntimeError("no usable results")

    def partial_body(_config):
        raise PartialExperiment(
            "aborted after some metrics",
            metrics={"validation_loss": 2.5, "latency": None},
            failures=[{"metric": "latency", "reason": "budget exceeded"}],
        )

    _install_bodies(
        monkeypatch,
        {
            1: clean_body,
            2: failing_body,
            3: partial_body,
            4: clean_body,
            5: clean_body,
        },
    )

    run_all(_config(tmp_path))

    assert _read(tmp_path / "experiment1.json")["status"] == "completed"

    exp2 = _read(tmp_path / "experiment2.json")
    assert exp2["status"] == "failed"
    assert any("no usable results" in e["reason"] for e in exp2["failures"])

    exp3 = _read(tmp_path / "experiment3.json")
    assert exp3["status"] == "partial"
    assert exp3["metrics"]["validation_loss"] == 2.5
    assert exp3["metrics"]["latency"] is None
    assert any("budget exceeded" in e["reason"] for e in exp3["failures"])

    assert _read(tmp_path / "experiment4.json")["status"] == "completed"
    assert _read(tmp_path / "experiment5.json")["status"] == "completed"


# Feature: catalyst-lora, Task 9.14
# Validates: Requirements 11.2, 11.3, 19.1
def test_manifest_reflects_all_five_with_statuses(
    tmp_path, monkeypatch, isolate_startup
):
    """The final manifest lists all five experiments in order with their statuses."""

    def clean_body(_config):
        return {"metrics": {}}

    def failing_body(_config):
        raise RuntimeError("boom")

    def partial_body(_config):
        raise PartialExperiment("partial", metrics={"x": 1}, failures=[])

    _install_bodies(
        monkeypatch,
        {
            1: clean_body,
            2: failing_body,
            3: partial_body,
            4: clean_body,
            5: clean_body,
        },
    )

    run_all(_config(tmp_path))

    manifest = _read(tmp_path / "manifest.json")
    assert manifest["manifest_version"] == 1

    experiments_list = manifest["experiments"]
    assert [e["experiment"] for e in experiments_list] == [1, 2, 3, 4, 5]
    assert [e["status"] for e in experiments_list] == [
        "completed",
        "failed",
        "partial",
        "completed",
        "completed",
    ]
