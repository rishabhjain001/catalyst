"""Headline merged/unmerged equivalence verification (Experiment 4 evidence).

This is the explicit equivalence check reported as evidence in Experiment 4:
pushing a fixed batch through the unmerged adapter and through the merged
adapter must agree within ``1e-5``, and ``unmerge_adapter`` must restore the
original base weight elementwise within the same tolerance. It is intentionally
kept separate from the hypothesis-driven Property 1 test in
``tests/test_lora_merge.py``: the deterministic case below (fixed seed) is the
reproducible headline number, and a small hypothesis check guards the same
invariant across random adapters and both base layouts.

Convention (matches ``catalyst/lora.py``): logical weight ``W`` is ``d x k``,
``A`` is ``r x k``, ``B`` is ``d x r``. Modest alpha and ~unit-scale ``A``/``B``/
``x`` keep float32 comfortably inside the ``1e-5`` tolerance.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.lora import LoRA_Module
from tests.strategies import base_kinds, make_base

TOLERANCE = 1e-5

# Headline regime: modest alpha and ~unit-scale A/B/x keep float32 comfortably
# inside the 1e-5 absolute tolerance, matching the Experiment 4 evidence setup.
HEADLINE_ALPHA_MAX = 16.0
HEADLINE_SCALE = 1.0


@st.composite
def headline_cases(draw):
    """Draw a self-consistent case in the modest-alpha, unit-scale regime."""
    d = draw(st.integers(min_value=1, max_value=8))
    k = draw(st.integers(min_value=1, max_value=8))
    rank = draw(st.integers(min_value=1, max_value=min(d, k)))
    alpha = draw(
        st.floats(min_value=1.0, max_value=HEADLINE_ALPHA_MAX,
                  allow_nan=False, allow_infinity=False)
    )
    kind = draw(base_kinds())
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    generator = torch.Generator().manual_seed(seed)
    base = make_base(d, k, kind)
    A = torch.randn(rank, k, generator=generator) * HEADLINE_SCALE
    B = torch.randn(d, rank, generator=generator) * HEADLINE_SCALE
    x = torch.randn(4, k, generator=generator) * HEADLINE_SCALE
    return base, rank, alpha, A, B, x


def _logical_weight(module: LoRA_Module) -> torch.Tensor:
    """Return the base weight in logical ``d x k`` form for either layout."""
    if module.layer_kind == "conv1d":
        return module.base.weight.t()
    return module.base.weight


# Feature: catalyst-lora, Property 1
@pytest.mark.parametrize("kind", ["linear", "conv1d"])
def test_merged_unmerged_equivalence_deterministic(kind):
    """Validates: Requirements 2.3, 17.2"""
    # Fixed seed makes this the reproducible headline equivalence number.
    torch.manual_seed(20240517)

    d, k, rank, alpha = 8, 6, 4, 8.0
    base = make_base(d, k, kind)
    module = LoRA_Module(base, rank, alpha)

    # Random non-zero adapter at ~unit scale; B starts at zero, so without this
    # the delta would vanish and the comparison would be vacuous.
    with torch.no_grad():
        module.A.copy_(torch.randn(rank, k))
        module.B.copy_(torch.randn(d, rank))

    # Snapshot the original base weight before any folding so the unmerge
    # restoration can be checked elementwise.
    original_weight = module.base.weight.detach().clone()

    # Fixed batch reused for both the unmerged and merged passes.
    x = torch.randn(4, k)

    out_unmerged = module.forward(x)

    module.merge_adapter()
    out_merged = module.forward(x)

    # Headline check: the merged weight reproduces the unmerged forward.
    max_diff = torch.max(torch.abs(out_unmerged - out_merged)).item()
    assert max_diff <= TOLERANCE, f"merged/unmerged max diff {max_diff} exceeds {TOLERANCE}"

    module.unmerge_adapter()

    # Unmerging must restore the original base weight elementwise.
    restore_diff = torch.max(torch.abs(module.base.weight - original_weight)).item()
    assert restore_diff <= TOLERANCE, (
        f"unmerge restoration max diff {restore_diff} exceeds {TOLERANCE}"
    )


# Feature: catalyst-lora, Property 1
@settings(max_examples=100)
@given(case=headline_cases())
def test_merged_unmerged_equivalence_random(case):
    """Validates: Requirements 2.3, 17.2"""
    base, rank, alpha, A, B, x = case
    module = LoRA_Module(base, rank, alpha)

    # Non-zero adapter values so the folded delta is non-trivial.
    with torch.no_grad():
        module.A.copy_(A)
        module.B.copy_(B)

    original_weight = module.base.weight.detach().clone()

    out_unmerged = module.forward(x)

    module.merge_adapter()
    out_merged = module.forward(x)

    assert out_merged.shape == out_unmerged.shape
    # Magnitude-aware float32 comparison: merging reorders the arithmetic, so
    # the absolute error grows with output magnitude (ULP scaling). A combined
    # atol/rtol check verifies equivalence without false alarms from rounding.
    assert torch.allclose(out_unmerged, out_merged, atol=TOLERANCE, rtol=1e-4)

    module.unmerge_adapter()
    assert torch.allclose(module.base.weight, original_weight, atol=TOLERANCE, rtol=1e-4)
