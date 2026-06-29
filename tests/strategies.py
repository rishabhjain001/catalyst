"""Shared hypothesis strategies for the LoRA core property tests.

These generators are consumed by the property tests covering the
``LoRA_Module`` math (forward, rank bound, merge/unmerge, persistence, swap,
interpolation). They produce coherent ``d x k`` weights, valid and
out-of-range ranks, alphas, mixing factors, random ``A``/``B`` pairs, and input
batches, plus a base-module generator that wraps both ``nn.Linear`` and a
GPT-2-style ``Conv1D`` so the transposed-layout path is exercised.

Convention (matches ``catalyst/lora.py``): logical weight ``W`` has shape
``d x k``; ``A`` has shape ``r x k`` and ``B`` has shape ``d x r``.
"""

from dataclasses import dataclass

import torch
from hypothesis import strategies as st
from torch import nn

# Dimensions are kept modest so each property test can run its 100+ examples
# quickly; the LoRA math is shape-general, so small weights still exercise it.
MIN_DIM = 1
MAX_DIM = 16

# Batch and sequence bounds for generated input tensors.
MAX_BATCH = 4
MAX_SEQ = 4


class Conv1D(nn.Module):
    """Minimal GPT-2-style 1D convolution used as a transposed-layout base.

    Unlike ``nn.Linear`` (weight ``(out, in)``, computes ``x @ W.T``), GPT-2's
    ``Conv1D`` stores its weight as ``(in, out)`` and computes ``x @ weight``.
    Named ``Conv1D`` on purpose: when ``transformers`` is absent, the LoRA
    core detects this layout by class name and applies the transposed merge
    arithmetic, so this stub keeps that path under test.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias


@dataclass
class LoraCase:
    """A coherent bundle for exercising one ``LoRA_Module`` instance."""

    d: int
    k: int
    rank: int
    alpha: float
    kind: str
    base: nn.Module
    A: torch.Tensor
    B: torch.Tensor
    x: torch.Tensor


def make_base(d: int, k: int, kind: str) -> nn.Module:
    """Build a base module of logical shape ``d x k`` of the requested kind."""
    if kind == "linear":
        # nn.Linear(in=k, out=d) stores weight as (d, k) -> logical W is d x k.
        return nn.Linear(k, d)
    if kind == "conv1d":
        # Conv1D stores weight as (in=k, out=d); logical W is its transpose.
        return Conv1D(k, d)
    raise ValueError(f"Unknown base kind {kind!r}; expected 'linear' or 'conv1d'.")


def base_kinds() -> st.SearchStrategy:
    """Strategy over the supported base-module layouts."""
    return st.sampled_from(["linear", "conv1d"])


@st.composite
def dims(draw, min_dim: int = MIN_DIM, max_dim: int = MAX_DIM):
    """Draw a valid weight shape ``(d, k)``.

    The lower bound of 1 deliberately includes single-element ``1 x 1`` weights
    as an edge case.
    """
    d = draw(st.integers(min_value=min_dim, max_value=max_dim))
    k = draw(st.integers(min_value=min_dim, max_value=max_dim))
    return d, k


def valid_ranks(d: int, k: int) -> st.SearchStrategy:
    """Strategy over ranks satisfying ``1 <= r <= min(d, k)``."""
    return st.integers(min_value=1, max_value=min(d, k))


def out_of_range_ranks(d: int, k: int) -> st.SearchStrategy:
    """Strategy over ranks strictly greater than ``min(d, k)``."""
    limit = min(d, k)
    return st.integers(min_value=limit + 1, max_value=limit + 8)


def alphas() -> st.SearchStrategy:
    """Positive, finite LoRA scaling values kept in a numerically safe range."""
    return st.floats(
        min_value=0.1, max_value=64.0, allow_nan=False, allow_infinity=False
    )


def mixing_factors() -> st.SearchStrategy:
    """Interpolation factors within the inclusive range ``[0.0, 1.0]``."""
    return st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    )


def out_of_range_mixing_factors() -> st.SearchStrategy:
    """Mixing factors outside ``[0.0, 1.0]``, including special floats.

    Covers finite values just past each boundary, larger magnitudes, and the
    special values ``nan``/``+inf``/``-inf`` that interpolation must also
    reject.
    """
    specials = st.sampled_from(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.0001,
            1.0001,
            2.0,
            -5.0,
        ]
    )
    finite_out = st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ).filter(lambda a: a < 0.0 or a > 1.0)
    return st.one_of(specials, finite_out)


@st.composite
def tensors(draw, shape, scale_min: float = 0.1, scale_max: float = 3.0):
    """Draw a finite float32 tensor of ``shape`` with a bounded random scale.

    A drawn integer seed keeps generation reproducible under hypothesis while
    still ranging over many distinct matrices; the scale bound keeps values
    well-conditioned for the ``1e-5`` equivalence comparisons.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    scale = draw(
        st.floats(
            min_value=scale_min,
            max_value=scale_max,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator) * scale


def adapter_matrices(d: int, k: int, rank: int) -> st.SearchStrategy:
    """Strategy over ``(A, B)`` pairs with ``A`` shape ``r x k``, ``B`` ``d x r``."""
    return st.tuples(tensors((rank, k)), tensors((d, rank)))


@st.composite
def input_batches(draw, k: int, with_seq=None, max_batch: int = MAX_BATCH,
                  max_seq: int = MAX_SEQ):
    """Draw an input tensor whose last dimension is ``k``.

    Shape is ``(batch, k)`` or ``(batch, seq, k)`` so both the 2D and the
    sequence-bearing forward paths are exercised. Pass ``with_seq`` to force a
    particular rank.
    """
    batch = draw(st.integers(min_value=1, max_value=max_batch))
    if with_seq is None:
        with_seq = draw(st.booleans())
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    generator = torch.Generator().manual_seed(seed)
    if with_seq:
        seq = draw(st.integers(min_value=1, max_value=max_seq))
        return torch.randn(batch, seq, k, generator=generator)
    return torch.randn(batch, k, generator=generator)


@st.composite
def lora_cases(draw, min_dim: int = MIN_DIM, max_dim: int = MAX_DIM,
               with_seq=None):
    """Draw a complete, self-consistent ``LoraCase``.

    Bundles a valid shape, a valid rank, an alpha, a base module of one layout,
    matching random ``A``/``B`` matrices, and an input batch, so a property
    test can construct a module and immediately drive forward/merge/persistence
    behavior over both base layouts.
    """
    d = draw(st.integers(min_value=min_dim, max_value=max_dim))
    k = draw(st.integers(min_value=min_dim, max_value=max_dim))
    rank = draw(st.integers(min_value=1, max_value=min(d, k)))
    alpha = draw(alphas())
    kind = draw(base_kinds())
    base = make_base(d, k, kind)
    A = draw(tensors((rank, k)))
    B = draw(tensors((d, rank)))
    x = draw(input_batches(k, with_seq=with_seq))
    return LoraCase(d=d, k=k, rank=rank, alpha=alpha, kind=kind, base=base,
                    A=A, B=B, x=x)
