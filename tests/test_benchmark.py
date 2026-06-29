"""Verification for the standalone inference benchmark (task 15.1).

Confirms ``benchmark.run_benchmark`` measures cold-start latency, warm-start
latency, tokens per second, and adapter swap time, then writes a single
``benchmark.json`` with the expected schema. The benchmark runs fully offline
on a tiny GPT-2 built from ``GPT2Config`` (no weight download) paired with a
minimal in-range tokenizer, using the override hooks the entry point exposes,
with tiny warmup/timed/max_new_tokens counts so it stays fast.
"""

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from catalyst.benchmark import run_benchmark


class FakeTokenizer:
    """Minimal tokenizer whose ids stay inside the tiny model's vocab."""

    name_or_path = "fake-tokenizer"

    def __init__(self, vocab_size: int = 50):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token_id = 0

    def __len__(self):
        return self.vocab_size

    def _encode(self, text: str):
        return [(ord(char) % (self.vocab_size - 1)) + 1 for char in text] or [1]

    def __call__(
        self, text, padding=False, truncation=False, return_tensors=None, **_kwargs
    ):
        is_batch = isinstance(text, (list, tuple))
        texts = list(text) if is_batch else [text]
        seqs = [self._encode(item) for item in texts]
        if padding and len(seqs) > 1:
            width = max(len(seq) for seq in seqs)
            seqs = [seq + [self.pad_token_id] * (width - len(seq)) for seq in seqs]
        masks = [[1] * len(seq) for seq in seqs]

        if return_tensors == "pt":
            input_ids = torch.tensor(seqs, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        if is_batch:
            return {"input_ids": seqs, "attention_mask": masks}
        return {"input_ids": seqs[0], "attention_mask": masks[0]}

    def decode(self, ids, skip_special_tokens=True):
        values = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return "".join(chr(int(value)) for value in values if int(value) != 0)


def _tiny_model():
    """A tiny real GPT-2 with Conv1D projections and no weight download."""
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=50)
    return GPT2LMHeadModel(config)


def _config(tmp_path):
    return {
        "model_factory": _tiny_model,
        "tokenizer": FakeTokenizer(vocab_size=50),
        "targets": ["q", "v"],
        "alpha": 8.0,
        "default_rank": 2,
        "rank_by_layer": {0: 2, 1: 2},
        "prompts": ["the movie was"],
        "latency": {"warmup": 1, "timed": 3},
        "decoding": {
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "max_new_tokens": 3,
        },
        "seed": 7,
        "results_dir": str(tmp_path),
    }


def test_benchmark_writes_expected_schema(tmp_path):
    """run_benchmark writes benchmark.json with the documented keys."""
    results = run_benchmark(_config(tmp_path))

    artifact = tmp_path / "benchmark.json"
    assert artifact.is_file()
    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written == results

    for key in (
        "cold_start_seconds",
        "warm_start",
        "tokens_per_second",
        "adapter_swap_seconds",
        "metadata",
    ):
        assert key in written

    for stat in ("mean", "median", "p95", "stddev"):
        assert stat in written["warm_start"]

    metadata = written["metadata"]
    assert metadata["model_name"] == "fake-tokenizer"
    assert metadata["warmup"] == 1
    assert metadata["timed"] == 3
    assert "timestamp" in metadata
    assert metadata["decoding"]["max_new_tokens"] == 3


def test_benchmark_numbers_are_positive(tmp_path):
    """Latency, throughput, and swap measurements are real positive numbers."""
    results = run_benchmark(_config(tmp_path))

    assert results["cold_start_seconds"] > 0.0
    assert results["warm_start"]["mean"] > 0.0
    assert results["warm_start"]["median"] > 0.0
    assert results["warm_start"]["p95"] > 0.0
    assert results["warm_start"]["stddev"] >= 0.0
    assert results["tokens_per_second"] > 0.0

    # Swap time is measured on the tiny model (it exposes q/v projections), so
    # it is a real positive number rather than null-with-reason.
    assert results["adapter_swap_seconds"] is not None
    assert results["adapter_swap_seconds"] > 0.0
    assert "adapter_swap_reason" not in results
