"""Property test for the LoRA_Module rank bound (Property 4).

Verifies two behaviors of construction. For a valid rank ``r <= min(d, k)`` the
adapter matrices have the implemented shapes ``A`` is ``r x k`` and ``B`` is
``d x r`` with ``B`` initialized to zero (so the adapter is identity at start).
For an out-of-range rank ``r > min(d, k)`` construction emits a descriptive
warning, clamps the effective rank to ``min(d, k)``, and leaves the module
usable with correctly shaped matrices.

Shape convention matches ``catalyst/lora.py`` and the Hu et al. formulation
behind the forward identity ``x @ W.T + (alpha/r) * x @ A.T @ B.T``: ``A`` is
``r x k`` (Gaussian) and ``B`` is ``d x r`` (zero). This is the user-approved
convention; it intentionally differs from the older "A of d x r, B of r x k"
wording.
"""

import warnings

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.lora import LoRA_Module
from tests.strategies import (
    alphas,
    base_kinds,
    dims,
    make_base,
    out_of_range_ranks,
    valid_ranks,
)


# Feature: catalyst-lora, Property 4
@settings(max_examples=150)
@given(data=st.data(), shape=dims(), kind=base_kinds(), alpha=alphas())
def test_valid_rank_gives_correct_shapes(data, shape, kind, alpha):
    """Validates: Requirements 1.1"""
    d, k = shape
    rank = data.draw(valid_ranks(d, k))
    base = make_base(d, k, kind)

    module = LoRA_Module(base, rank, alpha)

    # A is r x k, B is d x r in the implemented convention.
    assert tuple(module.A.shape) == (rank, k)
    assert tuple(module.B.shape) == (d, rank)
    # B is zero-initialized so the adapter is identity before any training.
    assert torch.count_nonzero(module.B).item() == 0
    # The valid rank is preserved unchanged (no clamping).
    assert module.rank == rank


# Feature: catalyst-lora, Property 4
@settings(max_examples=150)
@given(data=st.data(), shape=dims(), kind=base_kinds(), alpha=alphas())
def test_out_of_range_rank_warns_and_clamps(data, shape, kind, alpha):
    """Validates: Requirements 1.6"""
    d, k = shape
    limit = min(d, k)
    rank = data.draw(out_of_range_ranks(d, k))
    base = make_base(d, k, kind)

    # Construction must emit a descriptive warning rather than raise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = LoRA_Module(base, rank, alpha)

    assert any(issubclass(w.category, Warning) for w in caught)

    # Effective rank is clamped to min(d, k) and the matrices follow it.
    assert module.rank == limit
    assert tuple(module.A.shape) == (limit, k)
    assert tuple(module.B.shape) == (d, limit)
    assert torch.count_nonzero(module.B).item() == 0

    # The clamped module remains usable: a forward pass runs and is finite.
    x = torch.randn(2, k)
    output = module.forward(x)
    assert output.shape == (2, d)
    assert torch.isfinite(output).all().item()
