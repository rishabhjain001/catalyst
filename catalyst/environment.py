"""Consolidated environment snapshot for Catalyst.

The per-experiment metadata already records hardware and version fields, but
they are scattered across one JSON per experiment. This module writes a single
``results/environment.json`` that gathers the OS, CPU model, total RAM, Python
version, Torch version, and Transformers version into one easy-to-diff file
(Req 10.5).
"""

import os
import platform

import torch

from .experiments import _hardware_info, _write_json

# Transformers is imported guardedly: the snapshot still writes when the library
# is absent, recording the version as null rather than crashing.
try:
    import transformers

    _TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    _TRANSFORMERS_VERSION = None

# Default location of the consolidated snapshot. Overridable so verification can
# target a temporary path and leave the real artifact untouched.
DEFAULT_PATH = os.path.join("results", "environment.json")


def snapshot_environment() -> dict:
    """Return the environment fields gathered into one dictionary.

    Reuses ``experiments._hardware_info`` for the CPU model and total RAM so the
    snapshot reports the same hardware values the per-experiment metadata does.
    """
    hardware = _hardware_info()
    return {
        "os": platform.platform(),
        "cpu_model": hardware["cpu_model"],
        "total_ram_bytes": hardware["total_ram_bytes"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": _TRANSFORMERS_VERSION,
    }


def write_environment(path: str = DEFAULT_PATH) -> dict:
    """Write the environment snapshot to ``path`` and return it."""
    snapshot = snapshot_environment()
    _write_json(path, snapshot)
    return snapshot


def main() -> None:
    """Write the consolidated environment snapshot to the default path."""
    write_environment()


if __name__ == "__main__":
    main()
