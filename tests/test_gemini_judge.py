"""Integration tests for the supplementary Gemini judge (Req 13.5, 13.6).

The judge scores three dimensions (sentiment_coherence, fluency,
prompt_adherence) on 0-10 across up to 10 samples per condition, returns the
clamped per-dimension mean, and labels the result supplementary so downstream
reporting never treats it as primary evidence. These are example-based
integration/unit tests: the real-API path is mocked (or bypassed) so the
network is never touched, and the pure aggregation is exercised directly.
"""

import numpy as np

import catalyst.evaluate as evaluate
from catalyst.evaluate import _GEMINI_DIMENSIONS, gemini_judge


def _bounded_scores(n):
    """Build ``n`` per-sample score dicts with bounded (in-range) values."""
    return [
        {
            "sentiment_coherence": float((i % 11)),
            "fluency": float((i * 2) % 11),
            "prompt_adherence": float((i * 3) % 11),
        }
        for i in range(n)
    ]


def test_mocked_scores_aggregate_as_clamped_mean(monkeypatch):
    """Validates: Requirements 13.5, 13.6

    With a mock returning bounded per-sample scores, each dimension is the mean
    of the mocked scores and lands within ``[0, 10]``.
    """
    per_sample = _bounded_scores(10)
    monkeypatch.setattr(
        evaluate, "_score_with_gemini", lambda samples, config: per_sample
    )

    samples = [f"generated text {i}" for i in range(10)]
    result = gemini_judge(samples)

    for dimension in _GEMINI_DIMENSIONS:
        expected = float(np.mean([s[dimension] for s in per_sample]))
        assert result[dimension] == expected
        assert 0.0 <= result[dimension] <= 10.0


def test_result_is_labelled_and_supplementary(monkeypatch):
    """Validates: Requirements 13.5, 13.6"""
    monkeypatch.setattr(
        evaluate,
        "_score_with_gemini",
        lambda samples, config: _bounded_scores(5),
    )

    result = gemini_judge(["a", "b", "c", "d", "e"])

    assert result["label"] == "LLM-based qualitative evaluation"
    assert result["supplementary"] is True


def test_out_of_range_scores_are_clamped(monkeypatch):
    """Validates: Requirements 13.6

    Raw scores above 10 or below 0 are clamped into ``[0, 10]`` before the mean
    is taken, so a value of 12 contributes 10 and -3 contributes 0.
    """
    per_sample = [
        {"sentiment_coherence": 12, "fluency": -3, "prompt_adherence": 5},
        {"sentiment_coherence": 8, "fluency": 100, "prompt_adherence": -50},
    ]
    monkeypatch.setattr(
        evaluate, "_score_with_gemini", lambda samples, config: per_sample
    )

    result = gemini_judge(["x", "y"])

    # sentiment_coherence: clamp(12)=10, clamp(8)=8 -> mean 9
    assert result["sentiment_coherence"] == 9.0
    # fluency: clamp(-3)=0, clamp(100)=10 -> mean 5
    assert result["fluency"] == 5.0
    # prompt_adherence: clamp(5)=5, clamp(-50)=0 -> mean 2.5
    assert result["prompt_adherence"] == 2.5
    for dimension in _GEMINI_DIMENSIONS:
        assert 0.0 <= result[dimension] <= 10.0


def test_sample_count_caps_at_ten(monkeypatch):
    """Validates: Requirements 13.5

    No more than 10 samples per condition are scored, so ``sample_count`` caps
    at 10 even when more samples are passed.
    """
    monkeypatch.setattr(
        evaluate,
        "_score_with_gemini",
        lambda samples, config: _bounded_scores(len(samples)),
    )

    samples = [f"sample {i}" for i in range(25)]
    result = gemini_judge(samples)

    assert result["sample_count"] == 10


def test_config_scores_drive_aggregation_without_api():
    """Validates: Requirements 13.5, 13.6

    Passing precomputed ``config['scores']`` drives the pure aggregation path
    directly, bypassing any API entirely.
    """
    per_sample = _bounded_scores(4)
    result = gemini_judge(["p", "q", "r", "s"], config={"scores": per_sample})

    for dimension in _GEMINI_DIMENSIONS:
        expected = float(np.mean([s[dimension] for s in per_sample]))
        assert result[dimension] == expected
    assert result["supplementary"] is True
    assert result["sample_count"] == 4


def test_no_api_path_returns_neutral_supplementary(monkeypatch):
    """Validates: Requirements 13.5, 13.6

    Without an API key (and no supplied scores) the judge never touches the
    network: it routes through ``_score_with_gemini``, which returns ``None``,
    and the judge reports neutral, clearly supplementary scores.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    calls = {"count": 0}
    real = evaluate._score_with_gemini

    def spy(samples, config):
        calls["count"] += 1
        return real(samples, config)

    monkeypatch.setattr(evaluate, "_score_with_gemini", spy)

    result = gemini_judge(["only sample"])

    # The real-API path was routed through _score_with_gemini exactly once and,
    # with no key, returned None (no network contact) -> neutral scores.
    assert calls["count"] == 1
    for dimension in _GEMINI_DIMENSIONS:
        assert result[dimension] == 0.0
    assert result["label"] == "LLM-based qualitative evaluation"
    assert result["supplementary"] is True
    assert result["sample_count"] == 1
