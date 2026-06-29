"""Unit tests for startup runtime estimation in catalyst.train.

Covers the three pieces of task 6.1: ``timed_step`` measures one
forward-plus-backward step on a tiny GPT-2 model, ``estimate_runtime`` is a
pure projection, and the warning fires exactly at the 45-minute boundary
(just below, at, and just above) via ``runtime_warning`` and ``report_runtime``.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from catalyst.train import (
    WARNING_THRESHOLD_SECONDS,
    estimate_runtime,
    report_runtime,
    runtime_warning,
    timed_step,
    total_experiment_steps,
)


def _tiny_model():
    # A deliberately small causal-LM so a real forward+backward is fast.
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=50)
    return GPT2LMHeadModel(config)


def _fake_batch(vocab_size=50, batch=2, seq=8):
    input_ids = torch.randint(0, vocab_size, (batch, seq))
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
        "labels": input_ids.clone(),
    }


def test_timed_step_returns_positive_time_and_leaves_no_grad():
    model = _tiny_model()
    batch = _fake_batch()

    elapsed = timed_step(model, batch)

    assert isinstance(elapsed, float)
    assert elapsed > 0.0
    # The timing step must not leave gradients behind for the real loop.
    assert all(param.grad is None for param in model.parameters())


def test_estimate_runtime_is_product():
    assert estimate_runtime(0.5, 100) == 50.0
    assert estimate_runtime(0.0, 100) == 0.0
    assert estimate_runtime(2.0, 0) == 0.0


def test_total_experiment_steps_scales_max_steps():
    assert total_experiment_steps({"max_steps": 200}) == 11 * 200
    assert total_experiment_steps({}) == 0


def test_runtime_warning_boundary():
    threshold = WARNING_THRESHOLD_SECONDS

    # Just below the budget: no warning.
    assert runtime_warning(threshold - 1.0) is None
    # Exactly at the budget: still no warning (strictly greater than).
    assert runtime_warning(threshold) is None
    # Just above the budget: a descriptive warning string.
    warning = runtime_warning(threshold + 1.0)
    assert warning is not None
    assert "exceeds" in warning


def test_report_runtime_returns_whether_it_warned():
    # step_seconds * total_steps stays under the 45-minute budget.
    assert report_runtime(0.1, 10) is False
    # A step time that projects well over 45 minutes warns.
    over_budget_step = (WARNING_THRESHOLD_SECONDS + 60.0) / 10.0
    assert report_runtime(over_budget_step, 10) is True
