# Feature: catalyst-lora, Property 17

"""Property test for manifest completeness.

``write_manifest`` emits ``results/manifest.json`` fixing ``manifest_version``
at ``1`` and listing one entry per experiment record, each carrying the
experiment number, status, ISO 8601 start/end timestamps, and the result and
metadata output paths (Req 19.4). This drives the writer over many synthetic
record lists to confirm every entry is complete and faithful to its input.
"""

import json
import tempfile
from datetime import datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.experiments import write_manifest


# A base instant the generated timestamps are offset from; keeping it fixed
# makes the produced ISO strings deterministic per drawn offset.
_BASE = datetime(2024, 1, 1, 0, 0, 0)

_STATUSES = ("completed", "partial", "failed")


@st.composite
def experiment_records(draw):
    """Draw a list of 1..N synthetic experiment records.

    Experiment numbers are unique within a list (the manifest keys paths off
    them), and each record carries a sampled status plus ISO 8601 start/end
    timestamps built from the base instant plus a drawn offset so the strings
    round-trip through ``datetime.fromisoformat``.
    """
    numbers = draw(
        st.lists(
            st.integers(min_value=1, max_value=50),
            min_size=1,
            max_size=8,
            unique=True,
        )
    )
    records = []
    for n in numbers:
        status = draw(st.sampled_from(_STATUSES))
        start_offset = draw(st.integers(min_value=0, max_value=10_000))
        duration = draw(st.integers(min_value=0, max_value=10_000))
        start = (_BASE + timedelta(seconds=start_offset)).isoformat()
        end = (_BASE + timedelta(seconds=start_offset + duration)).isoformat()
        records.append(
            {
                "experiment": n,
                "status": status,
                "start_time": start,
                "end_time": end,
            }
        )
    return records


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# Feature: catalyst-lora, Property 17
@settings(max_examples=100, deadline=None)
@given(records=experiment_records())
def test_manifest_is_complete_for_every_record(records):
    """Validates: Requirements 19.4"""
    with tempfile.TemporaryDirectory() as tmp:
        path = write_manifest(records, {"results_dir": tmp})
        manifest = _read(path)

        # The manifest version is fixed at 1.
        assert manifest["manifest_version"] == 1

        experiments = manifest["experiments"]
        # One entry per record.
        assert len(experiments) == len(records)

        for record, entry in zip(records, experiments):
            n = record["experiment"]

            # Every required field is present.
            for key in (
                "experiment",
                "status",
                "start_time",
                "end_time",
                "results_path",
                "metadata_path",
            ):
                assert key in entry

            # The entry is faithful to its input record.
            assert entry["experiment"] == n
            assert entry["status"] == record["status"]
            assert entry["status"] in _STATUSES
            assert entry["start_time"] == record["start_time"]
            assert entry["end_time"] == record["end_time"]

            # Output paths key off the record's experiment number.
            assert entry["results_path"].endswith(f"experiment{n}.json")
            assert entry["metadata_path"].endswith(f"experiment{n}_metadata.json")

            # Timestamps are valid ISO 8601 strings.
            datetime.fromisoformat(entry["start_time"])
            datetime.fromisoformat(entry["end_time"])
