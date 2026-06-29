"""Property test for deterministic decoding reproducibility.

# Feature: catalyst-lora, Property 11

``generate_text`` defaults to greedy decoding (temperature 0.0 maps to
``do_sample=False``), so two generations of the same prompt must be
byte-identical (Req 13.4, 20.3). The property is checked across many prompts
rather than at a few fixed points: determinism should hold for every input, not
just the ones we happened to pick.

The model and tokenizer are built once at module scope (a tiny 2-layer GPT-2
plus the real ``gpt2`` tokenizer) so the 100+ examples stay fast while still
exercising the full tokenize -> generate -> decode round trip.
"""

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from catalyst.evaluate import generate_text

# The real gpt2 tokenizer; its vocabulary fixes the model's embedding size so
# every real token id has a valid embedding row.
_TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

# Built once: a tiny GPT-2 keeps generation cheap, and a fixed manual seed makes
# the randomly initialised weights stable across the test session so the
# determinism we assert is about decoding, not about model construction. The
# vocab size matches the tokenizer so any drawn prompt tokenizes into ids the
# embedding can look up.
torch.manual_seed(0)
_MODEL = GPT2LMHeadModel(
    GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(_TOKENIZER))
).eval()

# Greedy decoding with a small token budget for speed.
_CONFIG = {
    "tokenizer": _TOKENIZER,
    "decoding": {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "max_new_tokens": 8,
    },
}

# Constrain prompts to printable ascii letters/spaces so the property targets
# decoding determinism rather than pathological tokenizer inputs.
_prompts = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu"), whitelist_characters=" "),
    min_size=1,
    max_size=40,
)


# Feature: catalyst-lora, Property 11
@settings(max_examples=100, deadline=None)
@given(prompt=_prompts)
def test_deterministic_decoding_is_reproducible(prompt):
    """Validates: Requirements 13.4, 20.3"""
    first = generate_text(_MODEL, prompt, _CONFIG)
    second = generate_text(_MODEL, prompt, _CONFIG)
    third = generate_text(_MODEL, prompt, _CONFIG)
    assert first == second == third
