"""Property test for per-layer rank application (Property 9).

Verifies that ``inject_lora`` honors the per-layer rank allocation: for any
``rank_by_layer`` dictionary, every ``LoRA_Module`` injected into a given
decoder layer carries the rank specified for that layer's index
(Requirement 6.1).

A single small uninitialized GPT-2 model is built once and deep-copied per
example so the property runs its 100+ cases quickly without any network
download. The GPT-2 ``Conv1D`` projections exercise the transposed-layout path
of the injector.
"""

import copy

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import GPT2Config, GPT2LMHeadModel

from catalyst.lora import LoRA_Module
from catalyst.model import _decoder_blocks, inject_lora

# A tiny GPT-2 keeps every projection's smaller dimension at 16, so ranks up to
# 8 are always valid and never trigger clamping; building it once and copying
# per example avoids repeated construction cost.
N_LAYER = 3
_BASE_CONFIG = GPT2Config(n_layer=N_LAYER, n_embd=16, n_head=2, vocab_size=50)
_BASE_MODEL = GPT2LMHeadModel(_BASE_CONFIG)

# Target sets span attention-only and attention-plus-MLP so the resolved
# projection count per layer varies across examples.
TARGET_SETS = (
    ["q", "v"],
    ["q", "k", "v", "o"],
    ["q", "k", "v", "o", "mlp"],
)


def rank_by_layer_dicts() -> st.SearchStrategy:
    """Non-empty maps from a subset of layer indices to a small valid rank."""
    return st.dictionaries(
        keys=st.integers(min_value=0, max_value=N_LAYER - 1),
        values=st.integers(min_value=1, max_value=8),
        min_size=1,
    )


def _block_adapters(block) -> list:
    """All ``LoRA_Module`` instances injected within one decoder block."""
    return [m for m in block.modules() if isinstance(m, LoRA_Module)]


# Feature: catalyst-lora, Property 9
@settings(max_examples=120, deadline=None)
@given(
    rank_by_layer=rank_by_layer_dicts(),
    targets=st.sampled_from(TARGET_SETS),
    alpha=st.floats(min_value=0.5, max_value=32.0, allow_nan=False,
                    allow_infinity=False),
)
def test_each_injected_module_carries_layer_rank(rank_by_layer, targets, alpha):
    """Validates: Requirements 6.1"""
    model = copy.deepcopy(_BASE_MODEL)

    inject_lora(model, rank_by_layer, targets, alpha)

    blocks = dict(_decoder_blocks(model))

    # At least one layer must have received adapters for the assertion to mean
    # anything; the chosen targets always resolve in GPT-2, so every requested
    # layer is adapted.
    total_checked = 0
    for index, expected_rank in rank_by_layer.items():
        adapters = _block_adapters(blocks[index])
        assert adapters, (
            f"Layer {index} was in rank_by_layer but received no adapters."
        )
        for adapter in adapters:
            assert adapter.rank == expected_rank, (
                f"Layer {index} adapter has rank {adapter.rank}, "
                f"expected {expected_rank}."
            )
        total_checked += len(adapters)

    assert total_checked > 0

    # Layers absent from rank_by_layer must not have been adapted, confirming
    # the rank map alone drives which layers carry adapters.
    for index, block in blocks.items():
        if index not in rank_by_layer:
            assert not _block_adapters(block), (
                f"Layer {index} was not in rank_by_layer but received adapters."
            )


# Feature: catalyst-lora, Property 9
def test_distinct_ranks_per_layer_are_independent():
    """Validates: Requirements 6.1"""
    # A concrete example pinning a different rank to each layer guards against a
    # single shared rank being applied to every layer.
    model = copy.deepcopy(_BASE_MODEL)
    rank_by_layer = {0: 2, 1: 5, 2: 7}

    inject_lora(model, rank_by_layer, ["q", "v"], alpha=8.0)

    blocks = dict(_decoder_blocks(model))
    for index, expected_rank in rank_by_layer.items():
        adapters = _block_adapters(blocks[index])
        assert adapters
        assert all(a.rank == expected_rank for a in adapters)
