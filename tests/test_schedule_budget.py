"""Property test for the equal-budget schedule invariant (Property 19).

Experiment 5 compares three per-layer rank schedules — ``uniform`` (rank 8 at
every layer), ``progressive`` (4 shallow / 8 middle / 16 deep), and ``inverse``
(16 shallow / 8 middle / 4 deep) — and the comparison is only meaningful if the
schedules train an approximately equal number of parameters (Req 18.1). Because
every adapted layer shares the same target set and projection width, the per
layer trainable-parameter count is proportional to that layer's rank, so a
schedule's total rank-unit budget (``sum(rank_by_layer.values())``) stands in
for its parameter budget.

This module verifies the invariant in two ways:

* a property over generated decoder-layer counts, asserting the three default
  schedules always land inside the default tolerance band (0.20); and
* explicit checks of the tolerance helper itself (equal, in-band, just over).

The invariant holds for every layer count. ``_split_thirds`` gives the shallow
and deep thirds an equal size ``cut`` and assigns the leftover (the widened
middle) the same rank in both progressive and inverse, so progressive and
inverse always carry the identical total ``20*cut + 8*middle``. The uniform
total is ``8*count = 16*cut + 8*middle``, never larger, and the relative gap
``4*cut / (16*cut + 8*middle)`` peaks at ``1/6 ≈ 0.167`` when the thirds are
even (count divisible by three) — comfortably under 0.20. For fewer than three
layers the schedules collapse onto the middle rank and the three budgets are
exactly equal. The generated domain below therefore spans tiny counts, even
thirds, and widened-middle counts.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.experiments import (
    EXPERIMENT5_BUDGET_TOLERANCE,
    _budgets_within_tolerance,
    rank_schedules,
)


def _schedule_budgets(layer_indices):
    """Total rank-unit budget per schedule for the default rank schedules.

    Rank units are proportional to trainable parameters because every adapted
    layer uses the same target set and width, so summing the per-layer ranks is
    a faithful stand-in for the parameter budget the tolerance band guards.
    """
    schedules = rank_schedules(layer_indices)
    return {name: sum(ranks.values()) for name, ranks in schedules.items()}


# Feature: catalyst-lora, Property 19
@settings(max_examples=200)
@given(count=st.integers(min_value=1, max_value=48))
def test_default_schedules_share_equal_budget(count):
    """Validates: Requirements 18.1"""
    layer_indices = list(range(count))

    budgets = _schedule_budgets(layer_indices)

    # All three default schedules are populated for the generated layer set.
    assert set(budgets) == {"uniform", "progressive", "inverse"}
    # The equal-budget invariant: the three totals sit inside the default band.
    assert _budgets_within_tolerance(budgets, EXPERIMENT5_BUDGET_TOLERANCE)


# Feature: catalyst-lora, Property 19
@settings(max_examples=200)
@given(thirds=st.integers(min_value=1, max_value=16))
def test_even_thirds_progressive_equals_inverse(thirds):
    """Validates: Requirements 18.1"""
    # A layer count divisible by three gives even shallow/middle/deep thirds.
    count = 3 * thirds
    layer_indices = list(range(count))

    budgets = _schedule_budgets(layer_indices)

    # Progressive and inverse are mirror permutations of the same rank
    # multiset, so their totals are identical by construction.
    assert budgets["progressive"] == budgets["inverse"]
    # Uniform is the smallest budget and trails the others by exactly 1/6,
    # which is within the default 0.20 tolerance band.
    assert budgets["uniform"] <= budgets["progressive"]
    assert _budgets_within_tolerance(budgets, EXPERIMENT5_BUDGET_TOLERANCE)


def test_gpt2_twelve_layers_within_tolerance():
    """Validates: Requirements 18.1"""
    # The real GPT-2 has 12 decoder layers (even thirds of four each).
    budgets = _schedule_budgets(list(range(12)))

    # uniform = 8*12 = 96; progressive = inverse = (4+8+16)*4 = 112.
    assert budgets == {"uniform": 96, "progressive": 112, "inverse": 112}
    # (112 - 96) / 96 = 0.1667 <= 0.20.
    assert _budgets_within_tolerance(budgets, EXPERIMENT5_BUDGET_TOLERANCE)


def test_fewer_than_three_layers_collapse_to_equal_budgets():
    """Validates: Requirements 18.1"""
    # With one or two layers the thirds cannot form, so every layer is treated
    # as middle and all three schedules use the same middle rank (8) -> equal.
    for count in (1, 2):
        budgets = _schedule_budgets(list(range(count)))
        assert len(set(budgets.values())) == 1
        assert _budgets_within_tolerance(budgets, EXPERIMENT5_BUDGET_TOLERANCE)


def test_tolerance_helper_explicit_cases():
    """Validates: Requirements 18.1"""
    # Exactly equal budgets are trivially within any non-negative tolerance.
    assert _budgets_within_tolerance(
        {"uniform": 100, "progressive": 100, "inverse": 100}, 0.20
    )
    # A 0.20 spread sits exactly on the band edge -> within tolerance.
    assert _budgets_within_tolerance(
        {"uniform": 100, "progressive": 120, "inverse": 110}, 0.20
    )
    # Just over the band edge -> rejected.
    assert not _budgets_within_tolerance(
        {"uniform": 100, "progressive": 121, "inverse": 110}, 0.20
    )
    # Fewer than two measured budgets cannot be out of band -> within tolerance.
    assert _budgets_within_tolerance({"uniform": 100}, 0.20)
    assert _budgets_within_tolerance({"uniform": None, "progressive": None}, 0.20)
