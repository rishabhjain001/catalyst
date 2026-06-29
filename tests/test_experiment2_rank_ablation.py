"""Verification for Experiment 2 — rank ablation (task 9.4).

These tests confirm the registered Experiment 2 body trains and evaluates one
adapter per rank and that ``run_experiment(2, config)`` persists
``results/experiment2.json`` with a ``by_rank`` block for each rank. They run
fully offline on a tiny GPT-2 built from ``GPT2Config`` (no weight download)
paired with a minimal in-range tokenizer, with the override hooks the body
exposes for testing: ``model_factory``/``tokenizer`` supply the tiny model and
tokenizer, ``dataset``/``val_dataset`` supply in-memory rows instead of SST-2,
and ``results_dir``/checkpoint monkeypatching keep the real output dirs clean.
"""

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import catalyst.train as train_module
from catalyst.experiments import run_experiment


class FakeTokenizer:
    """Minimal tokenizer whose ids stay inside the tiny model's vocab.

    Implements the surface the data loader, trainer, and evaluator use: callable
    encoding for single strings and lists (with padding for batches), ``decode``,
    a ``pad_token``/``eos_token`` pair, and a length. Ids live in
    ``[1, vocab_size)`` so every token indexes the tiny embedding table; id 0 is
    the pad/eos token.
    """

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


_DATASET = [
    {"sentence": "the movie was great", "label": 1},
    {"sentence": "the film was awful", "label": 0},
    {"sentence": "a wonderful delight", "label": 1},
    {"sentence": "a boring mess", "label": 0},
]


def _config(tmp_path):
    return {
        "model_factory": _tiny_model,
        "tokenizer": FakeTokenizer(vocab_size=50),
        "dataset": _DATASET,
        "val_dataset": _DATASET,
        "experiment2_ranks": [2, 4],
        "targets": ["q", "v"],
        "alpha": 8.0,
        "max_steps": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "decoding": {"max_new_tokens": 3},
        "latency_warmup": 1,
        "latency_timed": 2,
        "memory_samples": 1,
        "eval_examples": 4,
        "results_dir": str(tmp_path),
        "seed": 7,
    }


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_experiment2_writes_by_rank_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    config = _config(tmp_path)

    record = run_experiment(2, config)

    assert record["status"] == "completed"
    by_rank = record["metrics"]["by_rank"]
    assert set(by_rank.keys()) == {"2", "4"}

    for key, block in by_rank.items():
        # Every compared metric is present for each rank (Req 15.2).
        for metric in (
            "validation_loss",
            "perplexity",
            "bleu",
            "rouge_l",
            "sentiment_accuracy",
            "gemini",
            "trainable_params",
            "peak_ram_mb",
            "throughput",
            "inference_latency",
        ):
            assert metric in block, (key, metric)

    # Higher rank trains strictly more parameters than lower rank.
    assert by_rank["4"]["trainable_params"] > by_rank["2"]["trainable_params"]

    # The result JSON was persisted under the temp results dir (Req 15.3).
    written = _read(tmp_path / "experiment2.json")
    assert written["experiment"] == 2
    assert written["status"] == "completed"
    assert set(written["metrics"]["by_rank"].keys()) == {"2", "4"}


def test_experiment2_inference_latency_covers_four_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

    record = run_experiment(2, _config(tmp_path))

    latency = record["metrics"]["by_rank"]["2"]["inference_latency"]
    assert set(latency.keys()) == {"base", "unmerged", "merged", "post_swap"}
    for stats in latency.values():
        assert {"mean", "median", "p95", "stddev", "tokens_per_second"} <= set(
            stats.keys()
        )


def test_experiment2_partial_when_one_rank_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    config = _config(tmp_path)

    calls = {"n": 0}
    good_factory = config["model_factory"]

    def flaky_factory():
        # Fail the very first model build so the first rank aborts while the
        # second rank still trains, yielding a partial experiment (Req 11.3).
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced model build failure")
        return good_factory()

    config["model_factory"] = flaky_factory

    record = run_experiment(2, config)

    assert record["status"] == "partial"
    by_rank = record["metrics"]["by_rank"]
    # One rank is null with a recorded reason; the other carries real metrics.
    assert by_rank["2"] is None
    assert by_rank["4"] is not None
    reasons = [entry["reason"] for entry in record["failures"]]
    assert any("forced model build failure" in reason for reason in reasons)
