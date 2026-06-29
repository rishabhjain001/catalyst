"""Verification for Experiment 4 — composition, merge, swap, persistence,
interpolation (task 9.6).

These tests confirm the registered Experiment 4 body trains a positive and a
negative adapter and records the five demonstrations the requirement names:
the measured unmerged-vs-merged difference, an instant named-adapter swap, a
save/load/list round trip, the continuous ``sentiment_by_alpha`` curve, and
inference latency/memory across the four conditions. They run fully offline on
a tiny GPT-2 built from ``GPT2Config`` (no weight download) paired with a
minimal in-range tokenizer, using the override hooks the body exposes.
"""

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import catalyst.train as train_module
from catalyst.experiments import run_experiment


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
        "targets": ["q", "v"],
        "alpha": 8.0,
        "experiment4_rank": 2,
        "experiment4_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "max_steps": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "decoding": {"max_new_tokens": 3},
        "latency_warmup": 1,
        "latency_timed": 2,
        "memory_samples": 1,
        "eval_prompts": ["the"],
        "results_dir": str(tmp_path),
        "seed": 7,
    }


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def test_experiment4_records_all_demonstrations(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

    record = run_experiment(4, _config(tmp_path))

    assert record["status"] == "completed", record["failures"]
    metrics = record["metrics"]

    # Merge equivalence is a small measured float (Req 17.2).
    assert metrics["merge_max_abs_diff"] is not None
    assert metrics["merge_max_abs_diff"] <= 1e-4

    # The named swap changed the active adapter (Req 17.3).
    assert metrics["swap_demonstrated"] is True
    assert metrics["swap_detail"]["active_after_positive"] == "positive"
    assert metrics["swap_detail"]["active_after_negative"] == "negative"

    # Persistence round trip: listed identifiers include the saved ones (Req 17.4).
    persistence = metrics["persistence"]
    assert set(persistence["saved"]) <= set(persistence["listed"])
    assert persistence["round_trip_ok"] is True

    # Continuous sentiment-versus-alpha curve covering alpha 0..1 (Req 17.5).
    curve = metrics["sentiment_by_alpha"]
    assert [point["alpha"] for point in curve] == [0.0, 0.25, 0.5, 0.75, 1.0]
    for point in curve:
        assert 0.0 <= point["sentiment_score"] <= 1.0


def test_experiment4_latency_and_memory_cover_four_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

    record = run_experiment(4, _config(tmp_path))

    conditions = {"base", "unmerged", "merged", "post_swap"}
    latency = record["metrics"]["inference_latency"]
    memory = record["metrics"]["inference_memory"]
    assert set(latency.keys()) == conditions
    assert set(memory.keys()) == conditions
    for stats in latency.values():
        assert {"mean", "median", "p95", "stddev", "tokens_per_second"} <= set(
            stats.keys()
        )
    for stats in memory.values():
        assert {"peak_ram_mb", "avg_ram_mb"} <= set(stats.keys())


def test_experiment4_persists_results_json(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

    run_experiment(4, _config(tmp_path))

    written = _read(tmp_path / "experiment4.json")
    assert written["experiment"] == 4
    assert written["status"] == "completed"
    assert "merge_max_abs_diff" in written["metrics"]
    assert len(written["metrics"]["sentiment_by_alpha"]) == 5
