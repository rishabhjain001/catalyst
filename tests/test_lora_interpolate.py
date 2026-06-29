"""Property test for LoRA_Module adapter interpolation (Property 6).

Verifies the interpolation contract of ``interpolate_adapters``:

* Formula: the returned effective weight equals
  ``W_logical + (1 - alpha) * delta_A + alpha * delta_B`` within ``1e-5``,
  where each ``delta`` is ``(adapter_alpha / adapter_rank) * (B @ A)`` for the
  corresponding saved adapter and ``W_logical`` is the live base weight in the
  logical ``d x k`` convention (``base.weight`` for ``nn.Linear``,
  ``base.weight.T`` for ``Conv1D``).
* Endpoints: ``alpha == 0.0`` yields the pure adapter_a effective weight and
  ``alpha == 1.0`` yields the pure adapter_b effective weight, both within
  ``1e-5``.
* Out-of-range: any ``alpha`` outside the inclusive ``[0.0, 1.0]`` range
  (including ``nan``/``+-inf``) raises a descriptive error.

``interpolate_adapters`` reads its two adapters from disk paths, so each example
builds two adapter modules, sets known ``A``/``B`` matrices, and saves them to a
``tempfile.TemporaryDirectory`` inside the test body (hypothesis discourages
function-scoped fixtures). All three modules (the live base plus the two saved
adapters) share the same logical shape and rank so the loaded deltas match.

Magnitudes are deliberately bounded (unit-scale ``A``/``B``, adapter alpha in
``[1, 16]``) so the independently recomputed float32 reference satisfies the
``1e-5`` tolerance without altering production code or the tolerance.
"""

import os
import tempfile

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.lora import LoRA_Module
from tests.strategies import (
    base_kinds,
    dims,
    make_base,
    out_of_range_mixing_factors,
    tensors,
    valid_ranks,
)

TOLERANCE = 1e-5

MODEL_NAME = "gpt2"
TARGET_MODULES = ["q", "v"]


def _logical_weight(base, kind):
    """Return the live base weight in logical ``d x k`` convention."""
    weight = base.weight.detach()
    return weight.t() if kind == "conv1d" else weight


def _save_adapter(directory, name, base, rank, adapter_alpha, A, B):
    """Build a LoRA_Module, install ``A``/``B``, save it, and return its path.

    Returns the on-disk path plus the logical ``d x k`` delta the saved adapter
    contributes, computed independently as ``(adapter_alpha / rank) * (B @ A)``.
    """
    module = LoRA_Module(
        base,
        rank,
        adapter_alpha,
        model_name=MODEL_NAME,
        target_modules=TARGET_MODULES,
    )
    with torch.no_grad():
        module.A.copy_(A)
        module.B.copy_(B)
    path = os.path.join(directory, f"{name}.pt")
    module.save_adapter(path)
    delta = (adapter_alpha / rank) * (B @ A)
    return path, delta


@st.composite
def interp_cases(draw):
    """Draw a coherent interpolation scenario.

    Bundles a shared logical shape and rank, a live base of one layout, two
    sets of unit-scale ``A``/``B`` matrices, two modest adapter alphas, and a
    mixing factor in ``[0.0, 1.0]``. The shared shape/rank guarantees the two
    saved deltas match the live module so ``interpolate_adapters`` can load
    them.
    """
    d, k = draw(dims())
    rank = draw(valid_ranks(d, k))
    kind = draw(base_kinds())
    # Unit-scale A/B and modest adapter alphas keep float32 deltas small enough
    # for the 1e-5 reference comparison.
    a_a = draw(tensors((rank, k), scale_min=0.1, scale_max=1.0))
    b_a = draw(tensors((d, rank), scale_min=0.1, scale_max=1.0))
    a_b = draw(tensors((rank, k), scale_min=0.1, scale_max=1.0))
    b_b = draw(tensors((d, rank), scale_min=0.1, scale_max=1.0))
    alpha_a = draw(
        st.floats(min_value=1.0, max_value=16.0, allow_nan=False, allow_infinity=False)
    )
    alpha_b = draw(
        st.floats(min_value=1.0, max_value=16.0, allow_nan=False, allow_infinity=False)
    )
    factor = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return {
        "d": d,
        "k": k,
        "rank": rank,
        "kind": kind,
        "a_a": a_a,
        "b_a": b_a,
        "a_b": a_b,
        "b_b": b_b,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "factor": factor,
    }


# Feature: catalyst-lora, Property 6
# Disk I/O per example makes timing variable, so the per-example deadline is
# disabled; the 100-example minimum still holds.
@settings(max_examples=100, deadline=None)
@given(case=interp_cases())
def test_interpolation_formula(case):
    """Validates: Requirements 5.1, 5.4"""
    live_base = make_base(case["d"], case["k"], case["kind"])
    live = LoRA_Module(
        live_base,
        case["rank"],
        case["alpha_a"],
        model_name=MODEL_NAME,
        target_modules=TARGET_MODULES,
    )
    w_logical = _logical_weight(live_base, case["kind"])

    with tempfile.TemporaryDirectory() as directory:
        path_a, delta_a = _save_adapter(
            directory,
            "adapter_a",
            make_base(case["d"], case["k"], case["kind"]),
            case["rank"],
            case["alpha_a"],
            case["a_a"],
            case["b_a"],
        )
        path_b, delta_b = _save_adapter(
            directory,
            "adapter_b",
            make_base(case["d"], case["k"], case["kind"]),
            case["rank"],
            case["alpha_b"],
            case["a_b"],
            case["b_b"],
        )

        alpha = case["factor"]
        result = live.interpolate_adapters(path_a, path_b, alpha)

    reference = w_logical + (1.0 - alpha) * delta_a + alpha * delta_b

    assert result.shape == reference.shape
    assert torch.max(torch.abs(result - reference)).item() <= TOLERANCE


# Feature: catalyst-lora, Property 6
@settings(max_examples=100, deadline=None)
@given(case=interp_cases())
def test_interpolation_endpoints(case):
    """Validates: Requirements 5.2, 5.3"""
    live_base = make_base(case["d"], case["k"], case["kind"])
    live = LoRA_Module(
        live_base,
        case["rank"],
        case["alpha_a"],
        model_name=MODEL_NAME,
        target_modules=TARGET_MODULES,
    )
    w_logical = _logical_weight(live_base, case["kind"])

    with tempfile.TemporaryDirectory() as directory:
        path_a, delta_a = _save_adapter(
            directory,
            "adapter_a",
            make_base(case["d"], case["k"], case["kind"]),
            case["rank"],
            case["alpha_a"],
            case["a_a"],
            case["b_a"],
        )
        path_b, delta_b = _save_adapter(
            directory,
            "adapter_b",
            make_base(case["d"], case["k"], case["kind"]),
            case["rank"],
            case["alpha_b"],
            case["a_b"],
            case["b_b"],
        )

        # alpha 0.0 collapses to the pure adapter_a effective weight.
        at_zero = live.interpolate_adapters(path_a, path_b, 0.0)
        # alpha 1.0 collapses to the pure adapter_b effective weight.
        at_one = live.interpolate_adapters(path_a, path_b, 1.0)

    pure_a = w_logical + delta_a
    pure_b = w_logical + delta_b

    assert torch.max(torch.abs(at_zero - pure_a)).item() <= TOLERANCE
    assert torch.max(torch.abs(at_one - pure_b)).item() <= TOLERANCE


# Feature: catalyst-lora, Property 6
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_interpolation_out_of_range_raises(data):
    """Validates: Requirements 5.5"""
    d, k = data.draw(dims())
    rank = data.draw(valid_ranks(d, k))
    kind = data.draw(base_kinds())
    bad_alpha = data.draw(out_of_range_mixing_factors())

    live = LoRA_Module(
        make_base(d, k, kind),
        rank,
        1.0,
        model_name=MODEL_NAME,
        target_modules=TARGET_MODULES,
    )

    with tempfile.TemporaryDirectory() as directory:
        path_a, _ = _save_adapter(
            directory,
            "adapter_a",
            make_base(d, k, kind),
            rank,
            1.0,
            data.draw(tensors((rank, k), scale_min=0.1, scale_max=1.0)),
            data.draw(tensors((d, rank), scale_min=0.1, scale_max=1.0)),
        )
        path_b, _ = _save_adapter(
            directory,
            "adapter_b",
            make_base(d, k, kind),
            rank,
            1.0,
            data.draw(tensors((rank, k), scale_min=0.1, scale_max=1.0)),
            data.draw(tensors((d, rank), scale_min=0.1, scale_max=1.0)),
        )

        with pytest.raises((ValueError, RuntimeError)):
            live.interpolate_adapters(path_a, path_b, bad_alpha)
