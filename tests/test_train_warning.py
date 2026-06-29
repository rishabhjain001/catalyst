"""Property test for the 45-minute runtime warning threshold (task 6.3).

Property 12: the startup estimator warns if and only if the projected total
runtime exceeds the 45-minute budget. This is the dedicated hypothesis property
test; the example-based boundary checks live in test_train_runtime.py.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.train import (
    WARNING_THRESHOLD_SECONDS,
    estimate_runtime,
    runtime_warning,
)

# Draw projections across a wide, finite range spanning well under to well over
# the budget (0 to 10x threshold) so the boundary is exercised from both sides.
projected_seconds = st.floats(
    min_value=0.0,
    max_value=10.0 * WARNING_THRESHOLD_SECONDS,
    allow_nan=False,
    allow_infinity=False,
)

# Non-negative factors for the composed estimate_runtime path. step_seconds is
# bounded so step_seconds * total_steps stays in a sensible finite range.
step_seconds = st.floats(
    min_value=0.0,
    max_value=600.0,
    allow_nan=False,
    allow_infinity=False,
)
total_steps = st.integers(min_value=0, max_value=100_000)


# Feature: catalyst-lora, Property 12
@settings(max_examples=200)
@given(projected=projected_seconds)
def test_warning_iff_projection_exceeds_threshold(projected):
    """Validates: Requirements 9.4, 9.5, 12.1, 12.2"""
    warned = runtime_warning(projected) is not None
    assert warned == (projected > WARNING_THRESHOLD_SECONDS)


# Feature: catalyst-lora, Property 12
@settings(max_examples=200)
@given(step=step_seconds, steps=total_steps)
def test_warning_iff_composed_estimate_exceeds_threshold(step, steps):
    """Validates: Requirements 9.4, 9.5, 12.1, 12.2"""
    projected = estimate_runtime(step, steps)
    warned = runtime_warning(projected) is not None
    assert warned == (projected > WARNING_THRESHOLD_SECONDS)


# Feature: catalyst-lora, Property 12
def test_threshold_boundary_exact_and_epsilon():
    """Validates: Requirements 9.4, 9.5, 12.1, 12.2"""
    threshold = WARNING_THRESHOLD_SECONDS

    # Exactly at the budget: no warning (the condition is strictly greater).
    assert runtime_warning(threshold) is None

    # One ULP above the budget: a warning fires.
    just_over = math.nextafter(threshold, math.inf)
    assert runtime_warning(just_over) is not None

    # One ULP below the budget: no warning.
    just_under = math.nextafter(threshold, -math.inf)
    assert runtime_warning(just_under) is None
