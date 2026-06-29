"""Property test for LoRA_Module.list_adapters completeness (Property 3).

Verifies that for any set of adapters saved into a directory, ``list_adapters``
returns exactly that set of identifiers (the file stems), ignoring any
non-adapter files that happen to share the directory.
"""

import tempfile
from pathlib import Path

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from catalyst.lora import LoRA_Module

# Safe filename stems: lowercase alphanumerics plus a few path-safe separators,
# length bounded so generation stays cheap and never collides with the reserved
# extension. Uppercase letters are intentionally excluded so that set-distinct
# stems are also distinct filenames on case-insensitive filesystems (e.g.
# Windows/macOS), where "X.pt" and "x.pt" would otherwise collide. The property
# under test is listing completeness, not case sensitivity, so this keeps the
# test filesystem-agnostic. Stems must be distinct as a set, so st.sets is used
# directly.
_STEM_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_-"

safe_stems = st.text(alphabet=_STEM_ALPHABET, min_size=1, max_size=12)
identifier_sets = st.sets(safe_stems, min_size=0, max_size=8)


def _save_small_adapter(directory: Path, stem: str) -> None:
    # A minimal LoRA_Module wrapping a tiny nn.Linear is enough to produce a
    # real adapter artifact; the contents are irrelevant to listing.
    module = LoRA_Module(nn.Linear(4, 3), rank=2, alpha=8.0)
    module.save_adapter(str(directory / f"{stem}.pt"))


# Feature: catalyst-lora, Property 3
# deadline is disabled because each example does real disk I/O (torch.save plus
# torch.load per identifier), whose timing is too variable for a per-example
# deadline; correctness here is about the returned set, not latency.
@settings(max_examples=100, deadline=None)
@given(identifiers=identifier_sets)
def test_list_adapters_returns_exactly_saved_set(identifiers):
    """Validates: Requirements 3.7"""
    with tempfile.TemporaryDirectory() as raw_dir:
        directory = Path(raw_dir)

        for stem in identifiers:
            _save_small_adapter(directory, stem)

        # A non-adapter file must be ignored so the returned set is exactly the
        # adapters that were saved.
        (directory / "notes.txt").write_text("not an adapter", encoding="utf-8")

        listed = LoRA_Module.list_adapters(str(directory))

        assert set(listed) == identifiers
