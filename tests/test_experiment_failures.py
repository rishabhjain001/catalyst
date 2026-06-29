"""Property tests for failed metrics being recorded as ``null`` with a reason.

Covers Property 16 (Req 11.5): any metric that cannot be measured is recorded
as ``null`` together with a non-empty reason string and is never assigned a
fabricated value.

Two complementary properties are exercised:

1. ``null_metric`` pure behavior -- nulling an arbitrary set of metric names
   leaves each metric ``None`` and records a matching ``{"metric", "reason"}``
   failure whose reason is the exact non-empty string supplied.
2. ``run_experiment`` graceful-failure path -- a body that raises records
   status ``"failed"`` and a failure with a non-empty reason string, while no
   metric value is ever fabricated (the metrics map is empty or all ``None``).
"""

import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.experiments import null_metric, run_experiment


# Non-empty metric names and reason strings. ``null_metric``/``run_experiment``
# place no constraints on the characters, so the full text space is fair game;
# the ``map(str.strip).filter`` keeps reasons non-blank so "non-empty reason"
# is a meaningful assertion rather than trivially-true whitespace.
metric_names = st.text(min_size=1, max_size=40)
reason_text = st.text(min_size=1, max_size=80).map(str.strip).filter(lambda s: s)


# Feature: catalyst-lora, Property 16
# Validates: Requirements 11.5
@settings(max_examples=200)
@given(
    entries=st.lists(
        st.tuples(metric_names, reason_text),
        min_size=1,
        max_size=12,
        unique_by=lambda pair: pair[0],
    )
)
def test_null_metric_records_null_with_reason(entries):
    """Every nulled metric is ``None`` and has a matching non-empty reason.

    Builds up a metrics dict by nulling several distinct metric names, then
    asserts each name maps to ``None`` (never a fabricated value) and that the
    failures list carries exactly one ``{"metric", "reason"}`` entry per nulled
    metric whose reason is the exact non-empty string supplied.
    """
    metrics: dict = {}
    failures: list = []

    for name, reason in entries:
        null_metric(metrics, failures, name, reason)

    # Every nulled metric is recorded as null -- never a fabricated value.
    for name, _reason in entries:
        assert name in metrics
        assert metrics[name] is None

    # One failure entry per nulled metric, in order, each a non-empty string
    # reason matching what was supplied. No metric is silently dropped.
    assert len(failures) == len(entries)
    for (name, reason), entry in zip(entries, failures):
        assert entry["metric"] == name
        assert entry["reason"] == reason
        assert isinstance(entry["reason"], str)
        assert entry["reason"] != ""

    # Every null metric appears in the failures list with a reason.
    nulled = [name for name, value in metrics.items() if value is None]
    failure_metrics = {entry["metric"] for entry in failures}
    for name in nulled:
        assert name in failure_metrics


# Feature: catalyst-lora, Property 16
# Validates: Requirements 11.5
@settings(max_examples=100, deadline=None)
@given(message=reason_text)
def test_run_experiment_failure_records_reason_without_fabrication(message):
    """A raising body yields status ``failed`` with a reason and no fabricated metrics.

    ``run_experiment`` must never re-raise: it records the experiment as
    ``"failed"``, attaches a failure entry whose reason is a non-empty string
    carrying the body's error message, and fabricates no metric values (the
    metrics map is empty or made entirely of ``None`` entries).
    """

    def raising_body(_config):
        raise RuntimeError(message)

    with tempfile.TemporaryDirectory() as tmp:
        config = {"model_name": "gpt2", "seed": 7, "results_dir": tmp}
        record = run_experiment(1, config, body=raising_body)

    # Graceful failure: marked failed, never re-raised.
    assert record["status"] == "failed"

    # At least one failure entry, each with a metric name and a non-empty string
    # reason -- nothing is fabricated in place of a real reason.
    assert len(record["failures"]) >= 1
    for entry in record["failures"]:
        assert "metric" in entry
        assert isinstance(entry["reason"], str)
        assert entry["reason"] != ""

    # The body's message survives in a reason; reasons are real, not invented.
    assert any(message in entry["reason"] for entry in record["failures"])

    # No metric value is fabricated: the metrics map is empty or all None.
    for value in record["metrics"].values():
        assert value is None
