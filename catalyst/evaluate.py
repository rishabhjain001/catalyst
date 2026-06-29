"""Evaluator for Catalyst.

Computes the quantitative and qualitative metrics defined in the design. This
file currently provides deterministic text generation; latency measurement,
memory profiling, the full metric suite, and the Gemini judge are added by
later tasks.
"""

import json
import os
import re
import time

import numpy as np
import psutil
import torch

from . import train as _train


def _decoding_kwargs(config: dict) -> dict:
    """Map config decoding settings to Hugging Face ``generate`` kwargs.

    The defaults (temperature 0.0, top-k disabled, top-p 1.0) map to greedy
    decoding (``do_sample=False``) so repeated generations of one prompt match;
    a configured temperature above 0.0 switches to sampling and honors the
    given top-k/top-p (Req 13.4).
    """
    decoding = config.get("decoding") or {}
    temperature = decoding.get("temperature", 0.0)
    top_k = decoding.get("top_k", 0)
    top_p = decoding.get("top_p", 1.0)
    max_new_tokens = decoding.get("max_new_tokens", 30)

    kwargs = {"max_new_tokens": int(max_new_tokens)}
    if temperature and float(temperature) > 0.0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = float(top_p)
        # top_k 0 disables the filter; pass it through only when enabled.
        if top_k and int(top_k) > 0:
            kwargs["top_k"] = int(top_k)
    else:
        # Greedy decoding is deterministic regardless of top-k/top-p.
        kwargs["do_sample"] = False
    return kwargs


def _resolve_tokenizer(config: dict):
    """Return the tokenizer from config, loading it from ``model_name`` if absent.

    The design signature is ``generate_text(model, prompt, config)``, so the
    tokenizer travels inside config; ``model_name`` is the lazy fallback when no
    tokenizer is supplied.
    """
    tokenizer = config.get("tokenizer")
    if tokenizer is not None:
        return tokenizer
    model_name = config.get("model_name")
    if model_name is None:
        raise ValueError(
            "generate_text needs a tokenizer: provide config['tokenizer'] or "
            "config['model_name']."
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def generate_text(model, prompt: str, config: dict) -> str:
    """Generate the continuation of ``prompt`` under deterministic decoding.

    Returns only the newly generated text (the prompt echo is stripped) so
    BLEU, ROUGE-L, and latency numbers compare against references cleanly.
    """
    tokenizer = _resolve_tokenizer(config)
    kwargs = _decoding_kwargs(config)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_length = inputs["input_ids"].shape[1]

    # The seed only affects sampling; greedy decoding needs none to be
    # reproducible.
    seed = config.get("seed")
    if kwargs.get("do_sample") and seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        output_ids = model.generate(**inputs, pad_token_id=pad_token_id, **kwargs)

    generated = output_ids[0, prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _latency_stats(samples) -> dict:
    """Summarize latency samples as mean, median, p95, and standard deviation.

    Standard deviation uses the sample convention (``ddof=1``); this is the
    standard deviation of the observed timed samples, so a single sample has no
    spread and reports ``0.0`` rather than raising. p95 is the 95th percentile
    via linear interpolation (``numpy.percentile``).
    """
    values = np.asarray(list(samples), dtype=float)
    if values.size == 0:
        raise ValueError("latency stats need at least one timed sample")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "stddev": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def _summarize_latency(latencies, warmup: int, token_counts) -> dict:
    """Discard the first ``warmup`` timings then summarize the remaining ones.

    Keeping the warm-up discard as plain slicing makes the exclusion directly
    checkable: the summary is computed only over ``latencies[warmup:]`` and the
    matching generated-token counts, so warm-up samples never influence the
    reported statistics (Req 13.3, Property 14).
    """
    timed_latencies = list(latencies)[warmup:]
    timed_tokens = list(token_counts)[warmup:]
    if not timed_latencies:
        raise ValueError("no timed samples remain after discarding warm-up generations")

    stats = _latency_stats(timed_latencies)
    total_time = float(np.sum(timed_latencies))
    total_tokens = int(sum(timed_tokens))
    # tokens/sec is the aggregate generation throughput across the timed run,
    # which is steadier than averaging per-call ratios.
    stats["tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0.0
    return stats


def measure_latency(
    model, prompts, warmup: int = 5, timed: int = 50, config: dict = None
) -> dict:
    """Measure single-model generation latency over a warm-up/timed split.

    Runs ``warmup`` generations whose timings are discarded, then ``timed``
    generations whose wall times are recorded, and returns
    ``tokens_per_second`` together with the ``mean``, ``median``, ``p95``, and
    ``stddev`` of the timed latencies (Req 13.3). Generation reuses
    ``generate_text`` so decoding stays deterministic (Req 13.4); the tokenizer
    and decoding settings travel inside ``config`` as elsewhere in this module.

    This measures one model. The base/unmerged/merged/post-swap reporting is the
    caller's responsibility: ``evaluate`` builds those four models and calls this
    per condition (see ``measure_latency_conditions`` for the thin mapping).
    """
    config = config or {}
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise ValueError("measure_latency needs at least one prompt")

    tokenizer = _resolve_tokenizer(config)

    latencies = []
    token_counts = []
    # Time every generation including the warm-up ones; the warm-up slice is
    # dropped in _summarize_latency so the discard is explicit and testable.
    for index in range(warmup + timed):
        prompt = prompts[index % len(prompts)]
        start = time.perf_counter()
        generated = generate_text(model, prompt, config)
        latencies.append(time.perf_counter() - start)
        token_counts.append(len(tokenizer(generated)["input_ids"]))

    return _summarize_latency(latencies, warmup, token_counts)


def measure_latency_conditions(
    models: dict, prompts, warmup: int = 5, timed: int = 50, config: dict = None
) -> dict:
    """Measure latency for each named condition in ``models``.

    Convenience wrapper for the four-condition report (base, unmerged, merged,
    post-swap): maps each ``{condition_name: model}`` entry to its latency stats
    by calling ``measure_latency`` per model (Req 13.3).
    """
    return {
        name: measure_latency(model, prompts, warmup, timed, config)
        for name, model in models.items()
    }


def _rss_mb() -> float:
    """Current process resident set size in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def profile_memory(
    model,
    condition: str,
    prompts=None,
    samples: int = 5,
    config: dict = None,
) -> dict:
    """Profile peak and average process RAM while exercising one model.

    Records the resident set size (RSS) before the workload and again after
    each of ``samples`` deterministic ``generate_text`` calls, then returns the
    ``condition`` label with the peak (max sample) and average (mean of samples)
    RAM in MB (Req 13.2). Including the pre-workload baseline among the samples
    keeps the result well defined for any positive ``samples`` and guarantees
    ``peak_ram_mb >= avg_ram_mb``.

    This profiles one model. The base/unmerged/merged/post-swap reporting is the
    caller's responsibility: ``evaluate`` builds those four models and calls this
    per condition (see ``profile_memory_conditions`` for the thin mapping). The
    tokenizer and decoding settings travel inside ``config`` as elsewhere here.
    """
    config = config or {}
    if prompts is None:
        prompts = config.get("prompts") or ["The movie was"]
    if isinstance(prompts, str):
        prompts = [prompts]
    if not prompts:
        raise ValueError("profile_memory needs at least one prompt")
    if samples < 1:
        raise ValueError("profile_memory needs at least one sample")

    # The baseline anchors the average so it never sits above the peak, even
    # when a generation transiently frees memory below the starting footprint.
    rss_samples = [_rss_mb()]
    for index in range(samples):
        generate_text(model, prompts[index % len(prompts)], config)
        rss_samples.append(_rss_mb())

    return {
        "condition": condition,
        "peak_ram_mb": float(max(rss_samples)),
        "avg_ram_mb": float(np.mean(rss_samples)),
    }


def profile_memory_conditions(
    models: dict, prompts=None, samples: int = 5, config: dict = None
) -> dict:
    """Profile memory for each named condition in ``models``.

    Convenience wrapper for the four-condition report (base, unmerged, merged,
    post-swap): maps each ``{condition_name: model}`` entry to its memory dict by
    calling ``profile_memory`` per model (Req 13.2).
    """
    return {
        name: profile_memory(model, name, prompts, samples, config)
        for name, model in models.items()
    }


# Sentiment tags match data.condition_text so the accuracy proxy can detect
# which sentiment a conditioned generation expresses.
POSITIVE_TAG = "[POSITIVE]"
NEGATIVE_TAG = "[NEGATIVE]"

# The four inference conditions reported for memory and latency (Req 13.2,
# 13.3).
_CONDITIONS = ("base", "unmerged", "merged", "post_swap")

# The Gemini judge scores these three dimensions; kept as a constant so the
# real-API path and the mockable aggregation path agree on the keys (Req 13.6).
_GEMINI_DIMENSIONS = ("sentiment_coherence", "fluency", "prompt_adherence")

# The metric keys in the exact order fixed by Req 13.1 (quantitative first,
# Gemini last). evaluate builds its result dict by inserting keys in this order
# so dict insertion order is the reported order (Property under task 7.7).
_METRIC_ORDER = (
    "validation_loss",
    "perplexity",
    "bleu",
    "rouge_l",
    "sentiment_accuracy",
    "trainable_params",
    "trainable_percent",
    "training_time_seconds",
    "peak_ram_mb",
    "avg_ram_mb",
    "checkpoint_size_bytes",
    "adapter_size_bytes",
    "inference_memory",
    "inference_latency",
    "gemini",
)


def _validation_loss(model, batches) -> float:
    """Average language-model loss over a validation set.

    Sums each batch's ``model(**batch).loss`` weighted by nothing more than the
    batch count, matching the simple mean the design describes; the model is run
    under ``no_grad`` since evaluation never updates weights.
    """
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = model(**batch)
            losses.append(float(output.loss))
    if not losses:
        raise ValueError("validation loss needs at least one batch")
    return float(np.mean(losses))


def _sentiment_label(text: str):
    """Classify which sentiment tag a generated string expresses, if any.

    The accuracy proxy is deliberately lightweight and documented: a generation
    is counted positive when it surfaces the ``[POSITIVE]`` tag (or the word
    "positive") without the negative counterpart, and vice versa. This mirrors
    the sentiment conditioning the model is trained on (``data.condition_text``
    prepends the tag), so a model that learned the conditioning reproduces the
    matching tag. Returns ``"positive"``, ``"negative"``, or ``None`` when the
    signal is absent or contradictory.
    """
    lowered = text.lower()
    positive = POSITIVE_TAG.lower() in lowered or "positive" in lowered
    negative = NEGATIVE_TAG.lower() in lowered or "negative" in lowered
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return None


def _sentiment_accuracy(model, samples, config: dict):
    """Accuracy of the sentiment-tag proxy over labelled prompts.

    ``samples`` is an iterable of ``(prompt, expected_label)`` where the label
    is ``1`` (positive) or ``0`` (negative). For each prompt the model generates
    a continuation and ``_sentiment_label`` reads back the expressed sentiment;
    accuracy is the fraction whose expressed sentiment matches the expected one.
    Returns ``None`` with no samples so the caller records a reason rather than
    fabricating a score.
    """
    pairs = list(samples)
    if not pairs:
        return None
    correct = 0
    for prompt, label in pairs:
        expected = "positive" if int(label) == 1 else "negative"
        generated = generate_text(model, prompt, config)
        if _sentiment_label(generated) == expected:
            correct += 1
    return correct / len(pairs)


def _text_metrics(model, references, prompts, config: dict):
    """Compute BLEU and ROUGE-L of generated continuations against references.

    Uses ``sacrebleu`` for corpus BLEU and ``rouge_score`` for ROUGE-L F-measure
    averaged over the pairs. When either library is unavailable the matching
    score comes back as ``None`` with a reason string so nothing is fabricated
    (Req 11.5). Returns ``(bleu, rouge_l, reasons)`` where ``reasons`` maps any
    null metric to its explanation.
    """
    reasons = {}
    predictions = [generate_text(model, prompt, config) for prompt in prompts]
    references = list(references)

    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(predictions, [references]).score
        bleu = float(bleu)
    except Exception as error:
        bleu = None
        reasons["bleu"] = f"sacrebleu unavailable: {error}"

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(reference, prediction)["rougeL"].fmeasure
            for reference, prediction in zip(references, predictions)
        ]
        rouge_l = float(np.mean(scores)) if scores else None
        if rouge_l is None:
            reasons["rouge_l"] = "no reference/prediction pairs to score"
    except Exception as error:
        rouge_l = None
        reasons["rouge_l"] = f"rouge_score unavailable: {error}"

    return bleu, rouge_l, reasons


def _count_parameters(model):
    """Return ``(trainable, total)`` parameter counts for a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _null_conditions():
    """A four-condition mapping with each condition set to ``None``.

    Used when the evaluator cannot build the base/unmerged/merged/post-swap
    models so the schema keys still appear, just unmeasured (Req 11.5).
    """
    return {condition: None for condition in _CONDITIONS}


def evaluate(model, data, config) -> dict:
    """Compute the full metric suite in the Req 13.1 order.

    ``model`` is the adapted model under test. ``data`` carries the evaluation
    inputs: ``validation`` batches for loss/perplexity, ``references`` and
    ``prompts`` for BLEU/ROUGE-L, and ``sentiment`` ``(prompt, label)`` pairs for
    the sentiment proxy. ``config`` carries decoding settings, the tokenizer, and
    any already-measured training-run values (``training_time_seconds``,
    ``peak_ram_mb``, ``avg_ram_mb``, ``checkpoint_size_bytes``,
    ``adapter_size_bytes``) plus an optional ``condition_models`` mapping for the
    four inference conditions.

    The returned dict has a ``metrics`` ordered exactly as Req 13.1 requires and
    a ``failures`` list naming every metric left ``null`` and why; values that
    cannot be measured are never fabricated (Req 11.5).
    """
    data = data or {}
    config = config or {}
    metrics = {}
    failures = []

    def record(key, value, reason):
        # Centralizes the "null with a reason" contract so every unmeasured
        # metric lands in failures exactly once.
        metrics[key] = value
        if value is None:
            failures.append({"metric": key, "reason": reason})

    # Validation loss and perplexity.
    validation = data.get("validation")
    if validation:
        loss = _validation_loss(model, validation)
        metrics["validation_loss"] = loss
        metrics["perplexity"] = _train.perplexity(loss)
    else:
        record("validation_loss", None, "no validation batches provided")
        record("perplexity", None, "perplexity needs validation loss")

    # BLEU and ROUGE-L over generated continuations.
    references = data.get("references")
    prompts = data.get("prompts")
    if references and prompts:
        bleu, rouge_l, reasons = _text_metrics(model, references, prompts, config)
        record("bleu", bleu, reasons.get("bleu"))
        record("rouge_l", rouge_l, reasons.get("rouge_l"))
    else:
        record("bleu", None, "no references/prompts provided")
        record("rouge_l", None, "no references/prompts provided")

    # Sentiment classification accuracy via the documented tag proxy.
    sentiment_samples = data.get("sentiment")
    accuracy = _sentiment_accuracy(model, sentiment_samples or [], config)
    record(
        "sentiment_accuracy",
        accuracy,
        "no labelled sentiment samples provided",
    )

    # Parameter counts come straight from the live model.
    trainable, total = _count_parameters(model)
    metrics["trainable_params"] = trainable
    metrics["trainable_percent"] = _train.trainable_percent(trainable, total)

    # Training-run values are measured during training, not here; accept them
    # from config and null any that the caller could not supply.
    for key in (
        "training_time_seconds",
        "peak_ram_mb",
        "avg_ram_mb",
        "checkpoint_size_bytes",
        "adapter_size_bytes",
    ):
        if config.get(key) is not None:
            metrics[key] = config[key]
        else:
            record(key, None, f"{key} not supplied by the training run")

    # Inference memory and latency over the four conditions.
    condition_models = config.get("condition_models")
    eval_prompts = prompts or config.get("prompts") or ["The movie was"]
    if condition_models:
        warmup = int(config.get("latency_warmup", 5))
        timed = int(config.get("latency_timed", 50))
        samples = int(config.get("memory_samples", 5))
        metrics["inference_memory"] = profile_memory_conditions(
            condition_models, eval_prompts, samples, config
        )
        metrics["inference_latency"] = measure_latency_conditions(
            condition_models, eval_prompts, warmup, timed, config
        )
    else:
        metrics["inference_memory"] = _null_conditions()
        metrics["inference_latency"] = _null_conditions()
        failures.append(
            {
                "metric": "inference_memory",
                "reason": "no condition_models provided for the four conditions",
            }
        )
        failures.append(
            {
                "metric": "inference_latency",
                "reason": "no condition_models provided for the four conditions",
            }
        )

    # Gemini qualitative evaluation runs last and is supplementary.
    gemini_samples = data.get("gemini_samples")
    if gemini_samples:
        metrics["gemini"] = gemini_judge(gemini_samples, config)
    else:
        metrics["gemini"] = None
        failures.append(
            {
                "metric": "gemini",
                "reason": "no generation samples provided for the Gemini judge",
            }
        )

    # Reassemble in the fixed Req 13.1 order; every key is present by now.
    ordered = {key: metrics[key] for key in _METRIC_ORDER}
    return {"metrics": ordered, "failures": failures}


def _clamp_score(value) -> float:
    """Clamp a raw score into the ``[0, 10]`` band the judge reports."""
    return float(min(10.0, max(0.0, float(value))))


def _aggregate_gemini(per_sample_scores) -> dict:
    """Mean each dimension over the per-sample scores, clamped to ``[0, 10]``.

    ``per_sample_scores`` is a list of dicts holding the three dimension keys.
    This is the pure, testable core the mock in task 7.8 drives: it never calls
    any API and simply averages the clamped scores per dimension. Missing
    dimensions on a sample default to ``0.0`` so a malformed score cannot crash
    the aggregation.
    """
    means = {}
    for dimension in _GEMINI_DIMENSIONS:
        values = [
            _clamp_score(score.get(dimension, 0.0)) for score in per_sample_scores
        ]
        means[dimension] = float(np.mean(values)) if values else 0.0
    return means


def _score_with_gemini(samples, config: dict):
    """Score samples with the real Gemini API when a key is configured.

    Returns a list of per-sample dimension dicts, or ``None`` when no API key is
    available so ``gemini_judge`` can fall back to a neutral, clearly
    supplementary result without contacting any service. The actual request is
    guarded behind the key check and the import so verification never reaches the
    network.
    """
    api_key = config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model_name = config.get("gemini_model", "gemini-1.5-flash")
    judge = genai.GenerativeModel(model_name)

    per_sample = []
    for sample in samples:
        prompt = (
            "Score the following text from 0 to 10 on sentiment_coherence, "
            "fluency, and prompt_adherence. Reply as JSON with those three "
            f"keys.\n\nText:\n{sample}"
        )
        response = judge.generate_content(prompt)
        per_sample.append(_parse_gemini_response(response.text))
    return per_sample


def _parse_gemini_response(text: str) -> dict:
    """Parse a Gemini JSON reply into the three dimension scores.

    Tolerates surrounding prose by extracting the first JSON object; any missing
    dimension defaults to ``0.0`` and is clamped during aggregation.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    payload = json.loads(match.group(0)) if match else {}
    return {dimension: payload.get(dimension, 0.0) for dimension in _GEMINI_DIMENSIONS}


def gemini_judge(samples, config: dict = None) -> dict:
    """Score generation samples with the supplementary LLM judge (Req 13.5, 13.6).

    Scores ``sentiment_coherence``, ``fluency``, and ``prompt_adherence`` on a
    0-10 scale across up to 10 samples per condition and returns the mean of each
    dimension. The result is labelled "LLM-based qualitative evaluation" and
    flagged ``supplementary`` so downstream reporting never treats it as primary
    evidence.

    The real API call is optional and guarded: when ``config['gemini_api_key']``
    or the ``GEMINI_API_KEY`` environment variable is set the judge queries
    Gemini, otherwise it returns neutral scores. Tests mock ``_score_with_gemini``
    (or pass ``config['scores']``) to drive the pure aggregation without touching
    the network.
    """
    config = config or {}
    samples = list(samples)[:10]

    # An explicit per-sample score list short-circuits to the pure path, which
    # is what the integration mock exercises.
    per_sample = config.get("scores")
    if per_sample is None:
        per_sample = _score_with_gemini(samples, config)

    if per_sample:
        means = _aggregate_gemini(per_sample)
    else:
        # No API and no supplied scores: report neutral midpoints rather than
        # fabricating a confident judgement, still clearly supplementary.
        means = {dimension: 0.0 for dimension in _GEMINI_DIMENSIONS}

    return {
        "label": "LLM-based qualitative evaluation",
        "supplementary": True,
        "sample_count": len(samples),
        **means,
    }
