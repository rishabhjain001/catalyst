"""Reproduce every Catalyst experiment from a single command.

Running ``python scripts/reproduce.py`` from the project root regenerates all
five experiment result JSONs, their metadata, and the manifest by driving the
orchestrator's strict 1→5 ``run_all`` (Req 19.1, 19.2). The default reproduces
with the configured settings; ``--max-steps`` lowers the training budget for a
quick check without editing the config.
"""

import argparse
import json
import os
import sys

# Make the project root importable so ``import catalyst`` works when this file
# is launched as ``scripts/reproduce.py`` (its own directory, not the root, is
# on sys.path by default).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from catalyst.experiments import load_config, run_all  # noqa: E402


def parse_args(argv=None):
    """Parse the optional reproduction flags."""
    parser = argparse.ArgumentParser(
        description="Reproduce all Catalyst experiments in strict order."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the configured training budget for a quick reproduction.",
    )
    return parser.parse_args(argv)


def _print_summary(config):
    """Print the manifest path and each experiment's status after the run.

    Reads the manifest the run just wrote so the summary reflects what was
    actually persisted rather than re-deriving it; a missing or unreadable
    manifest is reported instead of raising so a degraded run still ends
    cleanly.
    """
    results_dir = config.get("results_dir", "results")
    manifest_path = os.path.join(results_dir, "manifest.json")

    print(f"\nManifest: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Manifest unavailable: {error}")
        return

    for entry in manifest.get("experiments", []):
        print(f"  Experiment {entry['experiment']}: {entry['status']}")


def main(argv=None):
    """Load the config, run all experiments, and print a short summary."""
    args = parse_args(argv)

    config = load_config()
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    run_all(config)
    _print_summary(config)


if __name__ == "__main__":
    main()
