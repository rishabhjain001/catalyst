"""Project layout smoke test.

Asserts the artifact structure mandated by Requirement 23: the ``catalyst/``
package holds every source module, the output and config directories exist,
``configs/default.yaml`` is present, ``requirements.txt`` and ``README.md`` sit
at the root, and the comparison CLI entry point ``compare.py`` lives at the
project root rather than under ``catalyst/`` (Requirement 23.3).

The project root is resolved relative to this test file so the assertions hold
regardless of the directory pytest is invoked from.
"""

from pathlib import Path

# This file lives in ``<root>/tests/``; the parent of ``tests`` is the project
# root, so resolving one level up keeps the test location-independent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Source modules that must live under the ``catalyst/`` package (Req 23.1).
# ``compare.py`` is intentionally excluded here: it belongs at the root.
CATALYST_MODULES = [
    "lora.py",
    "model.py",
    "data.py",
    "train.py",
    "evaluate.py",
    "experiments.py",
    "visualize.py",
]

# Output and config directories that must exist (Req 23.2).
EXPECTED_DIRECTORIES = [
    "configs",
    "results",
    "plots",
    "checkpoints",
    "logs",
]

# Root-level files required by Req 23.2 and 23.3.
EXPECTED_ROOT_FILES = [
    "requirements.txt",
    "README.md",
    "compare.py",
]


def test_catalyst_package_contains_source_modules():
    """Validates: Requirements 23.1"""
    catalyst_dir = PROJECT_ROOT / "catalyst"
    assert catalyst_dir.is_dir(), f"missing package directory: {catalyst_dir}"
    for module in CATALYST_MODULES:
        module_path = catalyst_dir / module
        assert module_path.is_file(), f"missing catalyst module: {module_path}"


def test_output_and_config_directories_exist():
    """Validates: Requirements 23.2"""
    for directory in EXPECTED_DIRECTORIES:
        directory_path = PROJECT_ROOT / directory
        assert directory_path.is_dir(), f"missing directory: {directory_path}"


def test_default_config_present():
    """Validates: Requirements 23.2"""
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    assert config_path.is_file(), f"missing config file: {config_path}"


def test_root_level_files_present():
    """Validates: Requirements 23.2, 23.3"""
    for filename in EXPECTED_ROOT_FILES:
        file_path = PROJECT_ROOT / filename
        assert file_path.is_file(), f"missing root file: {file_path}"


def test_compare_cli_at_root_not_under_catalyst():
    """Validates: Requirements 23.3"""
    # The CLI entry point must be the root ``compare.py`` and must not live
    # inside the package, matching the Req 23.3 placement.
    assert (PROJECT_ROOT / "compare.py").is_file()
    assert not (PROJECT_ROOT / "catalyst" / "compare.py").exists()
