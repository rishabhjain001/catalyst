"""Integration tests for SST-2 loading.

Covers the two halves of Requirement 8.1 and 8.4: a genuine load of a small
split returns a usable dataset, and a genuine load failure surfaces as a
descriptive ``RuntimeError``. The success path skips gracefully when no
network is available so a CI runner without connectivity does not hard-fail,
while the failure path is forced deterministically and never touches the
network.
"""

import pytest

import catalyst.data as data
from catalyst.data import load_sst2

# SST-2 rows expose the review under this column and the binary label under
# this one; the success test asserts both are present.
EXPECTED_TEXT_COLUMN = "sentence"
EXPECTED_LABEL_COLUMN = "label"


def test_load_validation_split_returns_usable_dataset():
    """Validates: Requirements 8.1"""
    # The point of this assertion is the success path; without network the
    # load cannot run, so skip rather than fail on a connection error.
    try:
        dataset = load_sst2("validation")
    except RuntimeError as error:
        pytest.skip(f"network unavailable: {error}")

    assert len(dataset) > 0
    assert EXPECTED_TEXT_COLUMN in dataset.column_names
    assert EXPECTED_LABEL_COLUMN in dataset.column_names


def test_genuine_load_failure_raises_descriptive_error(monkeypatch):
    """Validates: Requirements 8.4"""
    import datasets

    failure_reason = "simulated dataset hub outage"

    def raise_failure(*args, **kwargs):
        raise OSError(failure_reason)

    # Force the underlying load to fail deterministically so this test never
    # depends on network state.
    monkeypatch.setattr(datasets, "load_dataset", raise_failure)

    with pytest.raises(RuntimeError) as excinfo:
        load_sst2("validation")

    message = str(excinfo.value)
    # The error must describe the failure: name the split and carry the
    # underlying reason rather than surfacing a bare exception.
    assert "validation" in message
    assert failure_reason in message
