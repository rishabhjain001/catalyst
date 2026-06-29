"""Property test for the LoRA_Module forward formula (Property 5).

Verifies that the forward output matches the documented identity
``x @ W.T + (alpha/r) * x @ A.T @ B.T`` within the float32 ``1e-5`` tolerance,
for both an ``nn.Linear`` base and a GPT-2-style ``Conv1D`` base whose stored
weight is transposed. Because every supported base may carry a bias and the
wrapped base path computes ``x @ W.T + bias`` for both layouts, the reference
is expressed as ``base(x) + (alpha/r) * x @ A.T @ B.T``.
"""

import torch
from hypothesis import given, settings

from catalyst.lora import LoRA_Module
from tests.strategies import lora_cases

TOLERANCE = 1e-5


# Feature: catalyst-lora, Property 5
@settings(max_examples=200)
@given(case=lora_cases())
def test_forward_matches_formula(case):
    """Validates: Requirements 1.4"""
    module = LoRA_Module(case.base, case.rank, case.alpha)

    # Drive the module with known adapter matrices so the forward output can be
    # compared against an independently computed reference.
    with torch.no_grad():
        module.A.copy_(case.A)
        module.B.copy_(case.B)

    actual = module.forward(case.x)

    # base(x) computes x @ W.T + bias for both nn.Linear and Conv1D (which
    # stores W transposed), so it stands in for the x @ W.T term plus any bias.
    scaling = case.alpha / case.rank
    update = (case.x @ case.A.t()) @ case.B.t()
    expected = case.base(case.x) + scaling * update

    assert actual.shape == expected.shape
    assert torch.max(torch.abs(actual - expected)).item() <= TOLERANCE
