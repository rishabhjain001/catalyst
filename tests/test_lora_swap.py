"""Property test for swap consistency and failure restoration (Property 7).

Two behaviors are verified over both base layouts (``nn.Linear`` and the
GPT-2-style ``Conv1D``):

1. Consistency: starting from a merged adapter, ``swap_adapter`` to a new
   shape-matching adapter produces the same forward output as a reference
   module that directly loaded and merged the new adapter onto the same frozen
   base, within the float32 ``1e-5`` tolerance (Req 4.1, 4.2).
2. Failure restoration: a swap to an adapter whose shape does not match raises
   a descriptive error and leaves the module in its exact pre-swap merged
   state, so the forward output is unchanged within ``1e-5`` (Req 4.3).

Magnitudes are deliberately bounded so float32 satisfies the ``1e-5`` absolute
tolerance: alpha is held to a modest ``[1.0, 16.0]`` range and the ``A``/``B``
matrices are drawn at roughly unit scale. The tolerance itself stays at the
spec convention of ``1e-5``.

hypothesis discourages function-scoped fixtures, so a
``tempfile.TemporaryDirectory`` is created inside the test body rather than
using ``tmp_path``; disk I/O makes per-example timing variable, so the deadline
is disabled.
"""

import os
import tempfile

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.lora import LoRA_Module
from tests.strategies import MAX_DIM, base_kinds, make_base, tensors

TOLERANCE = 1e-5
MODEL_NAME = "gpt2"
TARGET_MODULES = ["q", "v"]


@st.composite
def swap_cases(draw):
    """Draw a coherent swap scenario over one base layout.

    Bundles a logical shape ``d x k`` (at least 2 per dimension so a
    guaranteed-distinct mismatched shape exists), a valid rank, a bounded
    alpha, and unit-scale ``A``/``B`` matrices for both the originally merged
    adapter and the new adapter that is swapped in.
    """
    d = draw(st.integers(min_value=2, max_value=MAX_DIM))
    k = draw(st.integers(min_value=2, max_value=MAX_DIM))
    rank = draw(st.integers(min_value=1, max_value=min(d, k)))
    kind = draw(base_kinds())
    alpha = draw(
        st.floats(
            min_value=1.0, max_value=16.0, allow_nan=False, allow_infinity=False
        )
    )
    a_old = draw(tensors((rank, k), scale_min=0.1, scale_max=1.0))
    b_old = draw(tensors((d, rank), scale_min=0.1, scale_max=1.0))
    a_new = draw(tensors((rank, k), scale_min=0.1, scale_max=1.0))
    b_new = draw(tensors((d, rank), scale_min=0.1, scale_max=1.0))
    # Unit-scale input keeps the merged forward magnitudes well within range.
    batch = draw(st.integers(min_value=1, max_value=4))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, k, generator=generator)
    return {
        "d": d,
        "k": k,
        "rank": rank,
        "kind": kind,
        "alpha": alpha,
        "a_old": a_old,
        "b_old": b_old,
        "a_new": a_new,
        "b_new": b_new,
        "x": x,
    }


def _build_module(base, rank, alpha, a, b):
    module = LoRA_Module(
        base,
        rank,
        alpha,
        model_name=MODEL_NAME,
        target_modules=TARGET_MODULES,
    )
    with torch.no_grad():
        module.A.copy_(a)
        module.B.copy_(b)
    return module


def _clone_base(reference, d, k, kind):
    # Build a fresh base of the same layout carrying the same frozen weight and
    # bias as ``reference``, so a reference module shares the exact base state.
    fresh = make_base(d, k, kind)
    with torch.no_grad():
        fresh.weight.copy_(reference.weight)
        if getattr(fresh, "bias", None) is not None:
            fresh.bias.copy_(reference.bias)
    return fresh


def _save_adapter(d, k, rank, alpha, kind, a, b, path):
    saver = _build_module(make_base(d, k, kind), rank, alpha, a, b)
    saver.save_adapter(path)


# Feature: catalyst-lora, Property 7
@settings(max_examples=100, deadline=None)
@given(case=swap_cases())
def test_swap_consistency(case):
    """Validates: Requirements 4.1, 4.2, 17.3"""
    d, k, rank, alpha, kind = (
        case["d"], case["k"], case["rank"], case["alpha"], case["kind"]
    )
    x = case["x"]

    base = make_base(d, k, kind)
    # Capture the original frozen weight before any merge so the reference
    # module can merge the new adapter onto the same W.
    original_base = _clone_base(base, d, k, kind)

    module = _build_module(base, rank, alpha, case["a_old"], case["b_old"])
    module.merge_adapter()

    with tempfile.TemporaryDirectory() as directory:
        path_new = os.path.join(directory, "adapter_new.pt")
        _save_adapter(
            d, k, rank, alpha, kind, case["a_new"], case["b_new"], path_new
        )

        module.swap_adapter(path_new)
        swapped_out = module.forward(x)

        # Reference: load and merge the new adapter directly onto the original
        # frozen base (Req 4.1, 4.2).
        reference = LoRA_Module(
            original_base,
            rank,
            alpha,
            model_name=MODEL_NAME,
            target_modules=TARGET_MODULES,
        )
        reference.load_adapter(path_new)
        reference.merge_adapter()
        reference_out = reference.forward(x)

    assert swapped_out.shape == reference_out.shape
    assert torch.max(torch.abs(swapped_out - reference_out)).item() <= TOLERANCE


# Feature: catalyst-lora, Property 7
@settings(max_examples=100, deadline=None)
@given(case=swap_cases())
def test_swap_failure_restoration(case):
    """Validates: Requirements 4.3, 17.3"""
    d, k, rank, alpha, kind = (
        case["d"], case["k"], case["rank"], case["alpha"], case["kind"]
    )
    x = case["x"]

    base = make_base(d, k, kind)
    module = _build_module(base, rank, alpha, case["a_old"], case["b_old"])
    module.merge_adapter()

    pre_swap_out = module.forward(x)

    with tempfile.TemporaryDirectory() as directory:
        path_mismatch = os.path.join(directory, "adapter_mismatch.pt")
        # A strictly larger shape guarantees an A/B shape mismatch on load while
        # keeping format_version and model_name compatible, so the shape check
        # is what triggers the failure.
        mismatch_d, mismatch_k = d + 1, k + 1
        mismatch_rank = min(mismatch_d, mismatch_k, rank + 1)
        a_mismatch = torch.randn(mismatch_rank, mismatch_k)
        b_mismatch = torch.randn(mismatch_d, mismatch_rank)
        _save_adapter(
            mismatch_d,
            mismatch_k,
            mismatch_rank,
            alpha,
            kind,
            a_mismatch,
            b_mismatch,
            path_mismatch,
        )

        with pytest.raises(ValueError):
            module.swap_adapter(path_mismatch)

    # The failed swap must leave the prior adapter merged and the forward
    # output restored to its pre-swap value (Req 4.3).
    assert module.merged is True
    post_swap_out = module.forward(x)
    assert post_swap_out.shape == pre_swap_out.shape
    assert torch.max(torch.abs(post_swap_out - pre_swap_out)).item() <= TOLERANCE
