"""Property test for merge/unmerge forward equivalence (Property 1).

Verifies that folding the adapter into the base weight does not change the
forward output, and that unmerging restores the original behavior, both within
the float32 ``1e-5`` tolerance. Covers both an ``nn.Linear`` base and a
GPT-2-style ``Conv1D`` base whose stored weight is transposed, since the merge
arithmetic differs between the two layouts.

The ``1e-5`` *absolute* tolerance from the design is a float32 convention that
is only meaningful for reasonably-scaled values: at large magnitudes float32
rounding alone exceeds it even though the merge math is exact to a relative
~1e-7. So this test deliberately bounds the magnitude of the generated inputs
(a modest ``alpha`` and unit-normalized ``A``/``B``/``x``) so the forward
output stays at a sane scale where the fixed absolute tolerance is the right
yardstick. This calibrates the *test inputs*, not the property or the
production arithmetic.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.lora import LoRA_Module
from tests.strategies import base_kinds, make_base

TOLERANCE = 1e-5

# Keep the LoRA scaling (alpha / rank) modest so the folded delta stays at a
# magnitude where the fixed 1e-5 absolute tolerance is meaningful for float32.
MIN_ALPHA = 1.0
MAX_ALPHA = 16.0

MIN_DIM = 1
MAX_DIM = 16
MAX_BATCH = 4
MAX_SEQ = 4


def _unit(t: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to unit Frobenius norm (leave an all-zero tensor)."""
    norm = torch.linalg.vector_norm(t)
    if norm.item() == 0.0:
        return t
    return t / norm


@st.composite
def bounded_lora_cases(draw):
    """Draw a self-consistent LoRA case with bounded forward magnitude.

    Mirrors ``strategies.lora_cases`` (valid shape, valid rank, both base
    layouts, matching ``A``/``B`` and an input batch) but constrains the
    numeric scale: ``alpha`` is kept in ``[1, 16]`` and ``A``/``B``/``x`` are
    unit-normalized so ``x @ A.T @ B.T`` and the folded ``B @ A`` delta remain
    O(1). This keeps the forward output well within the range where the
    design's 1e-5 absolute tolerance holds for float32.
    """
    d = draw(st.integers(min_value=MIN_DIM, max_value=MAX_DIM))
    k = draw(st.integers(min_value=MIN_DIM, max_value=MAX_DIM))
    rank = draw(st.integers(min_value=1, max_value=min(d, k)))
    alpha = draw(
        st.floats(
            min_value=MIN_ALPHA,
            max_value=MAX_ALPHA,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    kind = draw(base_kinds())
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    generator = torch.Generator().manual_seed(seed)

    # Unit-normalized adapter matrices keep the folded delta B @ A bounded.
    A = _unit(torch.randn(rank, k, generator=generator))
    B = _unit(torch.randn(d, rank, generator=generator))

    batch = draw(st.integers(min_value=1, max_value=MAX_BATCH))
    if draw(st.booleans()):
        seq = draw(st.integers(min_value=1, max_value=MAX_SEQ))
        x = torch.randn(batch, seq, k, generator=generator)
    else:
        x = torch.randn(batch, k, generator=generator)
    x = _unit(x)

    return make_base(d, k, kind), rank, alpha, A, B, x


# Feature: catalyst-lora, Property 1
@settings(max_examples=200)
@given(case=bounded_lora_cases())
def test_merge_unmerge_forward_equivalence(case):
    """Validates: Requirements 2.1, 2.2, 2.3, 17.2"""
    base, rank, alpha, A, B, x = case
    module = LoRA_Module(base, rank, alpha)

    # Copy in non-zero adapter matrices so the delta is non-trivial; with B at
    # its zero init the merge would be a no-op and the property would be empty.
    with torch.no_grad():
        module.A.copy_(A)
        module.B.copy_(B)

    before_merge = module.forward(x)

    module.merge_adapter()
    after_merge = module.forward(x)

    # Folding ΔW into the base weight must leave the forward output unchanged.
    assert after_merge.shape == before_merge.shape
    assert torch.max(torch.abs(after_merge - before_merge)).item() <= TOLERANCE

    module.unmerge_adapter()
    after_unmerge = module.forward(x)

    # Unmerging restores the original weight, so forward matches the pre-merge
    # output elementwise within tolerance.
    assert after_unmerge.shape == before_merge.shape
    assert torch.max(torch.abs(after_unmerge - before_merge)).item() <= TOLERANCE
