"""Unit tests for evaluator metric ordering and the four inference conditions.

These tests pin two structural guarantees of ``catalyst.evaluate.evaluate``:

* the metric keys are returned in the exact order fixed by Requirement 13.1
  (quantitative metrics first, the supplementary Gemini judge last); and
* inference memory and inference latency each report the base, unmerged,
  merged, and post-swap conditions with the expected per-condition fields
  (Requirement 13.2, 13.3).

A tiny real GPT-2 (built from ``GPT2Config`` with no weight download) is paired
with a minimal in-range tokenizer. The tiny model has ``vocab_size=50``, so a
real GPT-2 tokenizer (vocab ~50k) would emit out-of-range token ids and break
the embedding; a small char-level tokenizer keeps every id inside the model's
vocabulary and keeps the test fast and network-free. The four conditions reuse
the same tiny model object, since these tests assert structure and presence,
not distinct measured values.
"""

import math

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from catalyst import evaluate


class TinyTokenizer:
    """A minimal char-level tokenizer with ids inside the tiny model vocab.

    Implements just the surface ``catalyst.evaluate`` uses: callable encoding
    (with and without ``return_tensors``), ``decode``, and the pad/eos ids. Token
    ids are kept in ``[1, vocab_size)`` so they always index the tiny model's
    embedding table safely; id ``0`` is reserved as the pad/eos token.
    """

    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 0

    def _encode(self, text: str):
        # Map each character into [1, vocab_size); 0 stays reserved for padding.
        return [(ord(char) % (self.vocab_size - 1)) + 1 for char in text] or [1]

    def __call__(self, text, return_tensors=None):
        ids = self._encode(text)
        if return_tensors == "pt":
            input_ids = torch.tensor([ids], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens=True):
        values = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return "".join(chr(int(value)) for value in values if int(value) != 0)


def build_tiny_model():
    """Construct a tiny real GPT-2 with no pretrained-weight download."""
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=50)
    return GPT2LMHeadModel(config)


def make_batch(tokenizer, text):
    """Build one validation batch (input_ids/attention_mask/labels) from text."""
    encoded = tokenizer(text, return_tensors="pt")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": encoded["input_ids"].clone(),
    }


def build_data_and_config():
    """Assemble the tiny model, fake data, and a fast evaluator config.

    The config reuses the one tiny model for all four conditions and shrinks the
    decoding/latency/memory counts so the evaluator runs quickly.
    """
    torch.manual_seed(0)
    model = build_tiny_model()
    tokenizer = TinyTokenizer(vocab_size=50)

    data = {
        "validation": [
            make_batch(tokenizer, "the movie was good"),
            make_batch(tokenizer, "the film was dull"),
        ],
        "references": ["a fine film", "a dull film"],
        "prompts": ["the movie was", "the film was"],
        "sentiment": [("the movie was", 1), ("the film was", 0)],
    }

    config = {
        "tokenizer": tokenizer,
        "decoding": {"max_new_tokens": 4},
        "latency_warmup": 1,
        "latency_timed": 2,
        "memory_samples": 2,
        "condition_models": {
            "base": model,
            "unmerged": model,
            "merged": model,
            "post_swap": model,
        },
    }
    return model, data, config


def test_metric_order_matches_requirement_13_1():
    """metrics keys appear in the exact Req 13.1 order and nothing is missing."""
    model, data, config = build_data_and_config()

    result = evaluate.evaluate(model, data, config)

    assert list(result["metrics"].keys()) == list(evaluate._METRIC_ORDER)


def test_memory_and_latency_cover_four_conditions():
    """inference_memory and inference_latency report the four conditions (Req 13.2)."""
    model, data, config = build_data_and_config()

    metrics = evaluate.evaluate(model, data, config)["metrics"]

    expected = {"base", "unmerged", "merged", "post_swap"}
    assert set(metrics["inference_memory"].keys()) == expected
    assert set(metrics["inference_latency"].keys()) == expected


def test_each_condition_dict_has_expected_fields():
    """Each latency/memory condition carries its documented fields (Req 13.2, 13.3)."""
    model, data, config = build_data_and_config()

    metrics = evaluate.evaluate(model, data, config)["metrics"]

    for condition, stats in metrics["inference_latency"].items():
        assert set(stats.keys()) >= {
            "mean",
            "median",
            "p95",
            "stddev",
            "tokens_per_second",
        }, condition

    for condition, stats in metrics["inference_memory"].items():
        assert set(stats.keys()) == {"condition", "peak_ram_mb", "avg_ram_mb"}
        assert stats["condition"] == condition


def test_perplexity_is_exp_of_validation_loss():
    """Sanity: reported perplexity equals exp(validation_loss)."""
    model, data, config = build_data_and_config()

    metrics = evaluate.evaluate(model, data, config)["metrics"]

    assert math.isclose(
        metrics["perplexity"],
        math.exp(metrics["validation_loss"]),
        rel_tol=1e-9,
    )


def test_trainable_percent_within_bounds():
    """Sanity: trainable percentage stays within [0, 100]."""
    model, data, config = build_data_and_config()

    metrics = evaluate.evaluate(model, data, config)["metrics"]

    assert 0.0 <= metrics["trainable_percent"] <= 100.0
