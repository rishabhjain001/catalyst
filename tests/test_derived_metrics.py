"""Property tests for the derived training/evaluation metrics.

# Feature: catalyst-lora, Property 13

The Trainer logs perplexity as ``exp(loss)`` each step (Req 12.3) and the
Evaluator reports the trainable-parameter percentage as ``trainable / total *
100`` (Req 14). Both are pure arithmetic over their inputs, so they are checked
here across the input space rather than at a few fixed points.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.train import perplexity, trainable_percent


# exp() stays finite and precise across this loss range, so the identity
# perplexity(loss) == exp(loss) can be checked exactly (within float tolerance)
# without tripping the overflow guard.
finite_losses = st.floats(
    min_value=-50.0,
    max_value=50.0,
    allow_nan=False,
    allow_infinity=False,
)


# Feature: catalyst-lora, Property 13
@settings(max_examples=200)
@given(loss=finite_losses)
def test_perplexity_equals_exp_loss(loss):
    """Validates: Requirements 12.3, 14.2"""
    assert math.isclose(perplexity(loss), math.exp(loss), rel_tol=1e-9)


# Feature: catalyst-lora, Property 13
@settings(max_examples=200)
@given(
    loss=st.floats(
        min_value=710.0, max_value=1e6, allow_nan=False, allow_infinity=False
    )
)
def test_perplexity_overflows_to_inf(loss):
    """Validates: Requirements 12.3, 14.2"""
    # math.exp overflows above ~709.78, where perplexity reports infinity
    # rather than raising.
    assert perplexity(loss) == float("inf")


# Feature: catalyst-lora, Property 13
@settings(max_examples=200)
@given(data=st.data())
def test_trainable_percent_matches_formula_and_bounds(data):
    """Validates: Requirements 12.3, 14.2"""
    total = data.draw(st.integers(min_value=1, max_value=10**12))
    trainable = data.draw(st.integers(min_value=0, max_value=total))

    percent = trainable_percent(trainable, total)

    assert math.isclose(percent, trainable / total * 100, rel_tol=1e-12)
    # With trainable in [0, total] the percentage stays within [0, 100].
    assert 0.0 <= percent <= 100.0


# Feature: catalyst-lora, Property 13
@settings(max_examples=50)
@given(trainable=st.integers(min_value=0, max_value=10**9))
def test_trainable_percent_zero_total_is_zero(trainable):
    """Validates: Requirements 12.3, 14.2"""
    # A model with no parameters has a well-defined (empty) percentage rather
    # than a division-by-zero error.
    assert trainable_percent(trainable, 0) == 0.0
