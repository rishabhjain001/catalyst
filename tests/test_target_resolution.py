"""Unit tests for architecture-derived target resolution (Task 4.6).

These tests resolve logical adapter targets against the real GPT-2 module tree
to confirm the resolver derives concrete projections from architecture rather
than hardcoded names (Requirements 6.2, 6.7) and defaults to ``[q, v]`` when no
targets are configured (Requirement 6.3).

A real GPT-2 architecture is built from ``GPT2Config`` without downloading any
weights, which gives the genuine module tree (real ``Conv1D`` projections:
fused ``c_attn``, attention output ``c_proj``, MLP ``c_fc``/``c_proj``)
deterministically and without network access.
"""

import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from catalyst.model import SLICE_SEP, resolve_targets


def build_gpt2(n_layer: int = 2):
    """Construct a small real GPT-2 with no pretrained-weight download.

    Using a tiny config keeps the tree identical in structure to GPT-2 124M
    (Conv1D projections, fused ``c_attn``) while constructing instantly.
    """
    config = GPT2Config(
        n_layer=n_layer,
        n_embd=64,
        n_head=2,
        vocab_size=50,
    )
    return GPT2LMHeadModel(config)


def test_default_targets_are_q_and_v():
    """None and [] both default to logical [q, v] (Requirement 6.3)."""
    model = build_gpt2(n_layer=2)

    expected = {
        0: ["attn.c_attn#q", "attn.c_attn#v"],
        1: ["attn.c_attn#q", "attn.c_attn#v"],
    }
    assert resolve_targets(model, None) == expected
    assert resolve_targets(model, []) == expected


def test_fused_qkv_split_into_three_slices():
    """[q, k, v] yields three slice tags over the one fused c_attn (Req 6.2)."""
    model = build_gpt2(n_layer=2)

    mapping = resolve_targets(model, ["q", "k", "v"])

    for index in (0, 1):
        assert mapping[index] == [
            "attn.c_attn#q",
            "attn.c_attn#k",
            "attn.c_attn#v",
        ]
        # All three slice tags reference the same physical c_attn projection:
        # their relpaths (before the slice separator) are identical.
        relpaths = {ident.split(SLICE_SEP, 1)[0] for ident in mapping[index]}
        assert relpaths == {"attn.c_attn"}


def test_full_target_set_recognizes_conv1d_projections():
    """Full [q,k,v,o,mlp] resolves attention and MLP for every layer (Req 6.7).

    Confirms Conv1D recognition end to end: the fused c_attn yields three q/k/v
    slices, the attention output is c_proj, and the MLP contributes its
    expand/contract pair c_fc and c_proj.
    """
    n_layer = 2
    model = build_gpt2(n_layer=n_layer)

    mapping = resolve_targets(model, ["q", "k", "v", "o", "mlp"])

    assert set(mapping.keys()) == set(range(n_layer))
    for index in range(n_layer):
        identifiers = mapping[index]
        assert identifiers == [
            "attn.c_attn#q",
            "attn.c_attn#k",
            "attn.c_attn#v",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
        ]
        # Six identifiers but only three distinct physical modules at this
        # layer: the fused c_attn, the attention output c_proj, and the two MLP
        # projections.
        relpaths = {ident.split(SLICE_SEP, 1)[0] for ident in identifiers}
        assert relpaths == {"attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"}


def test_attention_output_distinct_from_mlp_output():
    """The o target maps to attention c_proj only, not the MLP c_proj."""
    model = build_gpt2(n_layer=1)

    only_o = resolve_targets(model, ["o"])
    assert only_o == {0: ["attn.c_proj"]}

    only_mlp = resolve_targets(model, ["mlp"])
    assert only_mlp == {0: ["mlp.c_fc", "mlp.c_proj"]}


def test_unknown_target_raises_value_error():
    """An unsupported logical target is rejected with a descriptive error."""
    model = build_gpt2(n_layer=1)

    with pytest.raises(ValueError, match="Unknown logical targets"):
        resolve_targets(model, ["q", "bogus"])
