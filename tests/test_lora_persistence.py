"""Property test for the LoRA_Module persistence round-trip (Property 2).

Verifies that saving an adapter and loading it into a shape-matching module
preserves the forward output within the float32 ``1e-5`` tolerance, and that the
saved artifact carries every required field (``A``, ``B``, ``rank``, ``alpha``,
``format_version``, ``model_name``, ``target_modules``).

A second, independently constructed module of the same logical shape, rank,
alpha, and ``model_name`` is the load target, so the test exercises a genuine
save-on-one / load-into-another round trip rather than reloading into the
source module.

hypothesis discourages function-scoped fixtures, so a ``tempfile.TemporaryDirectory``
is created inside the test body instead of the ``tmp_path`` fixture to avoid the
fixture health-check warning.
"""

import os
import tempfile

import torch
from hypothesis import given, settings

from catalyst.lora import LoRA_Module
from tests.strategies import lora_cases, make_base

TOLERANCE = 1e-5

REQUIRED_KEYS = {
    "A",
    "B",
    "rank",
    "alpha",
    "format_version",
    "model_name",
    "target_modules",
}


# Feature: catalyst-lora, Property 2
# Each example performs real disk I/O (save then load), whose timing varies, so
# the per-example deadline is disabled; the 100-example minimum still holds.
@settings(max_examples=100, deadline=None)
@given(case=lora_cases())
def test_persistence_round_trip(case):
    """Validates: Requirements 3.1, 3.2, 3.3, 17.4"""
    model_name = "gpt2"
    target_modules = ["q", "v"]

    # Source module: stamp provenance and drive it with known adapter matrices
    # so the pre-save forward output is well defined.
    source = LoRA_Module(
        case.base,
        case.rank,
        case.alpha,
        model_name=model_name,
        target_modules=target_modules,
    )
    with torch.no_grad():
        source.A.copy_(case.A)
        source.B.copy_(case.B)

    before = source.forward(case.x)

    # A second shape-matching module (same d, k, rank, alpha, model_name) is the
    # load target. Its adapter starts from a different random A so a successful
    # round trip must overwrite it with the saved values. The saved artifact
    # carries only A and B, never the base weight, so the target must share the
    # same frozen base for forward equivalence to reflect the A/B round trip.
    target_base = make_base(case.d, case.k, case.kind)
    with torch.no_grad():
        target_base.weight.copy_(case.base.weight)
        if getattr(target_base, "bias", None) is not None:
            target_base.bias.copy_(case.base.bias)
    target = LoRA_Module(
        target_base,
        case.rank,
        case.alpha,
        model_name=model_name,
        target_modules=target_modules,
    )

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "adapter.pt")
        source.save_adapter(path)

        # The saved artifact must contain every required field (Req 3.1, 17.4).
        artifact = torch.load(path, weights_only=True)
        assert REQUIRED_KEYS <= set(artifact.keys())

        target.load_adapter(path)

    after = target.forward(case.x)

    # Round-trip forward equivalence within tolerance (Req 3.2, 3.3).
    assert after.shape == before.shape
    assert torch.max(torch.abs(after - before)).item() <= TOLERANCE
