"""Property tests for the evaluator's latency statistics.

# Feature: catalyst-lora, Property 14

The evaluator reports latency as mean, median, p95, and (sample) standard
deviation, and excludes the first ``warmup`` timings before summarizing
(Req 13.3). These are pure numeric reductions, so they are checked directly
against numpy references over synthetic latency lists rather than by running
real (slow, nondeterministic) generation.
"""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.evaluate import _latency_stats, _summarize_latency


# Positive, finite, well-conditioned latencies (seconds). The lower bound keeps
# total timed time away from zero so tokens/sec stays well-defined, and the
# upper bound keeps numpy reductions away from float overflow.
latency_values = st.floats(
    min_value=1e-4,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
)


# Feature: catalyst-lora, Property 14
@settings(max_examples=200)
@given(samples=st.lists(latency_values, min_size=1, max_size=64))
def test_latency_stats_match_numpy(samples):
    """Validates: Requirements 13.3"""
    stats = _latency_stats(samples)
    arr = np.asarray(samples, dtype=float)

    assert math.isclose(stats["mean"], float(np.mean(arr)), rel_tol=1e-9,
                        abs_tol=1e-12)
    assert math.isclose(stats["median"], float(np.median(arr)), rel_tol=1e-9,
                        abs_tol=1e-12)
    assert math.isclose(stats["p95"], float(np.percentile(arr, 95)),
                        rel_tol=1e-9, abs_tol=1e-12)
    if arr.size > 1:
        assert math.isclose(stats["stddev"], float(arr.std(ddof=1)),
                            rel_tol=1e-9, abs_tol=1e-12)
    else:
        # A single sample has no spread and reports 0.0 rather than NaN.
        assert stats["stddev"] == 0.0


# Feature: catalyst-lora, Property 14
@settings(max_examples=200)
@given(data=st.data())
def test_summarize_latency_excludes_warmup(data):
    """Validates: Requirements 13.3"""
    # Need at least one timed sample remaining after warm-up, so size >= 2 and
    # warmup in [0, len - 1].
    latencies = data.draw(st.lists(latency_values, min_size=2, max_size=64))
    n = len(latencies)
    warmup = data.draw(st.integers(min_value=0, max_value=n - 1))
    token_counts = data.draw(
        st.lists(st.integers(min_value=1, max_value=10_000),
                 min_size=n, max_size=n)
    )

    summary = _summarize_latency(latencies, warmup, token_counts)
    expected = _latency_stats(latencies[warmup:])

    for key in ("mean", "median", "p95", "stddev"):
        assert math.isclose(summary[key], expected[key], rel_tol=1e-9,
                            abs_tol=1e-12)

    expected_tps = sum(token_counts[warmup:]) / sum(latencies[warmup:])
    assert math.isclose(summary["tokens_per_second"], expected_tps,
                        rel_tol=1e-9, abs_tol=1e-12)
