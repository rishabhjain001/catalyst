"""Tests for the consolidated environment snapshot (Req 10.5)."""

import json

from catalyst.environment import write_environment


def test_write_environment_creates_file_with_expected_fields(tmp_path):
    """write_environment writes a snapshot carrying every required field."""
    target = tmp_path / "environment.json"

    snapshot = write_environment(str(target))

    assert target.exists()
    with open(target, "r", encoding="utf-8") as handle:
        written = json.load(handle)
    assert written == snapshot

    expected_keys = {
        "os",
        "cpu_model",
        "total_ram_bytes",
        "python_version",
        "torch_version",
        "transformers_version",
    }
    assert expected_keys.issubset(written.keys())


def test_environment_values_are_sensible(tmp_path):
    """The recorded values have the expected types and plausible content."""
    snapshot = write_environment(str(tmp_path / "environment.json"))

    assert isinstance(snapshot["os"], str) and snapshot["os"]
    assert isinstance(snapshot["cpu_model"], str) and snapshot["cpu_model"]
    assert isinstance(snapshot["total_ram_bytes"], int)
    assert snapshot["total_ram_bytes"] > 0
    assert isinstance(snapshot["python_version"], str) and snapshot["python_version"]
    assert isinstance(snapshot["torch_version"], str) and snapshot["torch_version"]
    # Transformers is installed in this environment, so the version is a string;
    # the field is allowed to be null only when the library is absent.
    assert snapshot["transformers_version"] is None or isinstance(
        snapshot["transformers_version"], str
    )
