"""Offline test for the training profiling entry point (task 15.5).

Runs ``run_profile`` with a tiny GPT-2 built via ``model_factory``, a fake
tokenizer, and a small in-memory dataset for two steps, writing into a temp
output directory. Asserts the ``.prof`` and ``.txt`` artifacts are created and
the report carries real profiling rows, so nothing touches the real ``logs/``.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import catalyst.train as train_module
from catalyst.profile_training import PROFILE_STEM, run_profile


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


def _tiny_model():
    config = GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=50)
    return GPT2LMHeadModel(config)


def test_run_profile_writes_stats_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    output_dir = tmp_path / "logs"

    dataset = [{"sentence": f"review {i}", "label": i % 2} for i in range(4)]
    config = {
        "model_factory": _tiny_model,
        "tokenizer": FakeTokenizer(),
        "dataset": dataset,
        "default_rank": 2,
        "targets": ["q", "v"],
        "alpha": 8.0,
        "max_steps": 2,
        "batch_size": 2,
        "results_dir": str(tmp_path / "results"),
        "adapter_name": "profile",
    }

    report_path = run_profile(config, output_dir=str(output_dir))

    prof_path = output_dir / f"{PROFILE_STEM}.prof"
    text_path = output_dir / f"{PROFILE_STEM}.txt"

    assert prof_path.is_file()
    assert text_path.is_file()
    assert str(text_path) == report_path

    report = text_path.read_text(encoding="utf-8")
    # The report names both sorted sections and lists real call rows (the
    # pstats header reports the number of function calls profiled).
    assert "cumulative time" in report
    assert "total time" in report
    assert "function calls" in report
    assert "train.py" in report
