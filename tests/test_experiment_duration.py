"""Property test for the experiment duration relationship.

# Feature: catalyst-lora, Property 18

Property 18: ``run_experiment`` records ``start_time`` and ``end_time`` as ISO
8601 strings and ``duration_seconds`` equal to ``end - start`` in seconds, and
that duration is never negative (Validates: Requirements 19.5).

The experiment bodies here return immediately (optionally after a sub-10ms
sleep) and never load a model, so each example is fast and offline. Output is
redirected to a fresh temp directory per example so the real ``results/`` stays
clean; ``capture_metadata`` still runs (setting seeds and hashing
``configs/default.yaml``), which is fast and touches no network.
"""

import math
import tempfile
import time
from datetime import datetime

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from catalyst.experiments import run_experiment


def _quick_body(sleep_seconds):
    """Return an experiment body that optionally sleeps, then returns metrics."""

    def body(_config):
        if sleep_seconds:
            time.sleep(sleep_seconds)
        return {"metrics": {}}

    return body


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    sleep_seconds=st.floats(min_value=0.0, max_value=0.005),
)
def test_duration_equals_end_minus_start_and_timestamps_are_iso8601(
    n, seed, sleep_seconds
):
    """duration_seconds == end - start (ISO 8601) and is non-negative.

    Validates: Requirements 19.5
    """
    with tempfile.TemporaryDirectory() as tmp:
        config = {"results_dir": tmp, "model_name": "gpt2", "seed": seed}
        record = run_experiment(n, config, body=_quick_body(sleep_seconds))

    # Both timestamps parse as ISO 8601 (``fromisoformat`` accepts exactly the
    # strings ``isoformat`` emits).
    start = datetime.fromisoformat(record["start_time"])
    end = datetime.fromisoformat(record["end_time"])

    # duration_seconds equals the recomputed difference of the parsed timestamps.
    expected = (end - start).total_seconds()
    assert math.isclose(record["duration_seconds"], expected, rel_tol=0.0, abs_tol=1e-6)

    # Time only moves forward, so the duration is never negative.
    assert record["duration_seconds"] >= 0.0
