"""Unit tests for the training loop and artifact saving in catalyst.train.

Covers task 6.2: a tiny LoRA-injected GPT-2 trains for a few steps, each step
logs loss, perplexity, throughput, and RAM, a checkpoint is written to the
checkpoint directory, and adapter-only artifacts land in the results directory.
Checkpoint and results directories are redirected to a temp location so the
real output dirs stay clean.
"""

import math

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import catalyst.train as train_module
from catalyst.lora import LoRA_Module
from catalyst.model import inject_lora
from catalyst.train import perplexity, train


def _injected_model():
    config = GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=50)
    model = GPT2LMHeadModel(config)
    return inject_lora(model, {0: 2, 1: 2}, ["q", "v"], alpha=8.0)


def _fake_batches(count=3, vocab_size=50, batch=2, seq=8):
    batches = []
    for _ in range(count):
        input_ids = torch.randint(0, vocab_size, (batch, seq))
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones(batch, seq, dtype=torch.long),
                "labels": input_ids.clone(),
            }
        )
    return batches


def test_perplexity_is_exp_of_loss():
    assert math.isclose(perplexity(0.0), 1.0, rel_tol=1e-9)
    assert math.isclose(perplexity(1.5), math.exp(1.5), rel_tol=1e-9)
    # An overflowing loss reports infinity rather than raising.
    assert perplexity(10_000.0) == float("inf")


def test_train_loop_logs_metrics_and_saves_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    results_dir = str(tmp_path / "results")

    model = _injected_model()
    batches = _fake_batches(count=3)
    config = {"max_steps": 4, "results_dir": results_dir, "adapter_name": "demo"}

    summary = train(model, batches, config)

    # The loop ran exactly max_steps and logged one entry per step.
    assert summary["steps"] == 4
    assert len(summary["step_logs"]) == 4

    for entry in summary["step_logs"]:
        # Perplexity is exp(loss) and throughput/RAM are recorded positives.
        assert math.isclose(entry["perplexity"], math.exp(entry["loss"]), rel_tol=1e-6)
        assert entry["throughput_samples_per_sec"] > 0.0
        assert entry["peak_ram_mb"] > 0.0
        assert entry["avg_ram_mb"] > 0.0

    # A checkpoint file was written to the checkpoint directory.
    checkpoint = summary["checkpoint_path"]
    assert checkpoint.startswith(str(tmp_path / "checkpoints"))
    assert (tmp_path / "checkpoints").exists()
    assert torch.load(checkpoint, weights_only=True)

    # Adapter-only artifacts (one per LoRA module) landed under results/.
    adapter_dir = summary["adapter_dir"]
    assert adapter_dir.startswith(results_dir)
    module_count = sum(
        1 for _ in (m for m in model.modules() if isinstance(m, LoRA_Module))
    )
    saved = list((tmp_path / "results" / "demo").glob("*.pt"))
    assert len(saved) == module_count


def test_train_accepts_dataset_tokenizer_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

    class FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<pad>"

        def __call__(self, texts, **kwargs):
            seq = 6
            input_ids = torch.randint(0, 50, (len(texts), seq))
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones(len(texts), seq, dtype=torch.long),
            }

    dataset = [{"sentence": f"review {i}", "label": i % 2} for i in range(4)]
    model = _injected_model()
    config = {
        "max_steps": 3,
        "batch_size": 2,
        "results_dir": str(tmp_path / "results"),
        "adapter_name": "pair",
    }

    summary = train(model, (dataset, FakeTokenizer()), config)

    assert summary["steps"] == 3
    assert summary["final_perplexity"] == math.exp(summary["final_loss"])
