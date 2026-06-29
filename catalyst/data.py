"""SST-2 loading and sentiment conditioning.

Loads the Stanford Sentiment Treebank v2 via Hugging Face ``datasets``,
prepends a sentiment tag to each example, and yields tokenized batches for
training. This module depends only on a tokenizer passed in by the caller, so
it stays independent of the model layer.
"""

from typing import Iterator

# Positive examples are prefixed with this tag, negative examples with the
# other, so the model learns sentiment-conditioned generation.
POSITIVE_TAG = "[POSITIVE]"
NEGATIVE_TAG = "[NEGATIVE]"

# SST-2 rows expose the review under one of these keys depending on the source
# revision; the first present key wins.
_TEXT_KEYS = ("sentence", "text")


def load_sst2(split: str):
    # Import lazily so importing this module does not hard-fail when datasets
    # is absent; the dependency is only needed for a real load.
    try:
        from datasets import load_dataset
    except Exception as error:
        raise RuntimeError(
            "Loading SST-2 requires the 'datasets' library; install it to "
            f"continue (import failed: {error})."
        ) from error

    # stanfordnlp/sst2 is parquet-based and needs no loading script, so it
    # works on datasets releases that dropped remote script execution.
    try:
        return load_dataset("stanfordnlp/sst2", split=split)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load SST-2 split {split!r}: {error}"
        ) from error


def condition_text(text: str, label: int) -> str:
    tag = POSITIVE_TAG if label == 1 else NEGATIVE_TAG
    return f"{tag} {text}"


def _example_text(example: dict) -> str:
    for key in _TEXT_KEYS:
        if key in example:
            return example[key]
    raise KeyError(
        f"SST-2 example has no text field; expected one of {_TEXT_KEYS}."
    )


def make_batches(dataset, tokenizer, batch_size: int) -> Iterator[dict]:
    # GPT-2's tokenizer ships without a pad token; reuse the end-of-sequence
    # token so padding to a uniform batch length works.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total = len(dataset)
    for start in range(0, total, batch_size):
        # Index one row at a time so the same loop works for a Hugging Face
        # Dataset and a plain list of example dicts.
        examples = [dataset[i] for i in range(start, min(start + batch_size, total))]
        conditioned = [
            condition_text(_example_text(example), example["label"])
            for example in examples
        ]
        batch = tokenizer(
            conditioned,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # Causal-LM training reuses the input ids as targets; padded positions
        # are ignored downstream via the attention mask.
        batch["labels"] = batch["input_ids"].clone()
        yield dict(batch)
