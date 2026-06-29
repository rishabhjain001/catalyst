# Feature: catalyst-lora, Property 8
"""Property test for gradient isolation after LoRA injection.

Property 8: Gradient isolation
    For any model with LoRA injected under any valid per-layer rank allocation,
    every adapter parameter A and B has requires_grad set to True and no base
    weight parameter has requires_grad set to True; if any base parameter would
    leak a gradient, injection fails loudly naming the offending parameter.

Validates: Requirements 1.2, 1.3, 6.4, 6.5, 6.6

Building a real GPT-2 per example is far too slow for 100+ examples, so a tiny
GPT-2 is built once and deep-copied per example. Each copy is injected fresh so
the property is exercised over many rank allocations and target subsets without
paying the construction cost repeatedly.
"""

import copy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import GPT2Config, GPT2LMHeadModel

from catalyst.lora import LoRA_Module
from catalyst.model import check_grad_leak, inject_lora

# A small decoder-only GPT-2 with real Conv1D projections (c_attn fused qkv,
# c_proj attention output, mlp.c_fc expand, mlp.c_proj contract). Built once at
# module import; every example deep-copies it so injection always starts fresh.
_CONFIG = GPT2Config(
    n_layer=2,
    n_embd=16,
    n_head=2,
    vocab_size=50,
    bos_token_id=None,
    eos_token_id=None,
)
_BASE_MODEL = GPT2LMHeadModel(_CONFIG)

# Layer indices available in the tiny model and the logical targets the injector
# understands. Ranks stay small but valid for the 16-wide hidden size.
_LAYER_INDICES = list(range(_CONFIG.n_layer))
_TARGETS = ["q", "k", "v", "o", "mlp"]


def _fresh_model():
    return copy.deepcopy(_BASE_MODEL)


# A non-empty per-layer rank allocation over the available layer indices, with
# small valid ranks, paired with a non-empty subset of logical targets.
rank_by_layer_strategy = st.dictionaries(
    keys=st.sampled_from(_LAYER_INDICES),
    values=st.integers(min_value=1, max_value=4),
    min_size=1,
    max_size=len(_LAYER_INDICES),
)

targets_strategy = st.lists(
    st.sampled_from(_TARGETS), min_size=1, max_size=len(_TARGETS), unique=True
)


@settings(max_examples=120, deadline=None)
@given(rank_by_layer=rank_by_layer_strategy, targets=targets_strategy)
def test_gradient_isolation_after_injection(rank_by_layer, targets):
    """Every A/B is trainable and no base weight is, across rank allocations."""
    model = _fresh_model()

    # inject_lora itself runs check_grad_leak; reaching here means it passed.
    inject_lora(model, rank_by_layer, targets, alpha=8.0)

    # Direct iteration: every adapter A/B trainable.
    adapter_param_ids = set()
    saw_adapter = False
    for module in model.modules():
        if isinstance(module, LoRA_Module):
            saw_adapter = True
            assert module.A.requires_grad is True
            assert module.B.requires_grad is True
            adapter_param_ids.add(id(module.A))
            adapter_param_ids.add(id(module.B))

    assert saw_adapter, "Injection produced no LoRA adapters for a valid config."

    # Direct iteration: no base weight is trainable.
    for name, param in model.named_parameters():
        if id(param) not in adapter_param_ids:
            assert param.requires_grad is False, (
                f"Base weight '{name}' leaked a gradient after injection."
            )

    # The dedicated assertion agrees with direct iteration.
    check_grad_leak(model)


@settings(max_examples=120, deadline=None)
@given(rank_by_layer=rank_by_layer_strategy, targets=targets_strategy)
def test_forced_base_leak_fails_loudly(rank_by_layer, targets):
    """A forced base-weight leak makes check_grad_leak raise, naming the param."""
    model = _fresh_model()
    inject_lora(model, rank_by_layer, targets, alpha=8.0)

    adapter_param_ids = set()
    for module in model.modules():
        if isinstance(module, LoRA_Module):
            adapter_param_ids.add(id(module.A))
            adapter_param_ids.add(id(module.B))

    # Force exactly one base weight to leak a gradient.
    leaked_name = None
    for name, param in model.named_parameters():
        if id(param) not in adapter_param_ids:
            param.requires_grad = True
            leaked_name = name
            break

    assert leaked_name is not None, "No base parameter available to force a leak."

    with pytest.raises(RuntimeError) as excinfo:
        check_grad_leak(model)

    # The error must name the offending parameter (Requirement 6.5).
    assert leaked_name in str(excinfo.value)
