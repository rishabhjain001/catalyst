"""Unit tests for the interactive comparison CLI (task 12.2).

These cover the variant coverage the CLI promises (Req 20.1, 20.2), the
side-by-side terminal rendering (Req 20.4), and the no-server / no-dashboard
guarantee (Req 20.6). Everything runs fully offline: a tiny GPT-2 built from
``GPT2Config`` (no weight download) is paired with a minimal in-range
tokenizer and passed into ``run_variants`` through the ``config`` injection
hooks, and all artifacts are written under ``tmp_path``.
"""

import copy
import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel

import compare
from compare import BLEND_ALPHAS, run_variants, _print_side_by_side
from catalyst.model import AdapterManager, inject_lora


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


def _offline_config(model, tokenizer):
    """Config that keeps run_variants offline via preloaded model/tokenizer."""
    return {
        "model": model,
        "tokenizer": tokenizer,
        "targets": ["q", "v"],
        "alpha": 8.0,
        "rank_by_layer": {0: 2, 1: 2},
        "decoding": {"max_new_tokens": 3},
    }


def _save_named_adapter(base_model, config, name, directory):
    """Create a compatible adapter-set directory by injecting LoRA + saving.

    Builds an injected copy with the same rank_by_layer/targets/alpha as the
    CLI uses, wraps it in an AdapterManager, and writes one per-module artifact
    per managed module so the directory is a complete, reloadable named adapter
    that ``_discover_adapters`` will accept.
    """
    injected = copy.deepcopy(base_model)
    inject_lora(injected, config["rank_by_layer"], config["targets"], config["alpha"])
    manager = AdapterManager(injected)
    manager.save_adapter_set(name, str(directory))


# --- Variant coverage (Req 20.1, 20.2) -------------------------------------


def test_run_variants_no_adapters_runs_base_unmerged_merged(tmp_path):
    base = _tiny_model()
    tokenizer = FakeTokenizer()
    config = _offline_config(base, tokenizer)
    results_dir = tmp_path / "empty_results"
    results_dir.mkdir()

    outputs = run_variants(
        prompt="the", max_tokens=3, gemini=False, config=config,
        results_dir=str(results_dir),
    )

    # Base and the two zero-delta LoRA variants always run (Req 20.1).
    assert {"base", "unmerged", "merged"} <= set(outputs)
    # No named adapters means no adapter:/blend: entries.
    assert not [name for name in outputs if name.startswith("adapter:")]
    assert not [name for name in outputs if name.startswith("blend:")]
    # The absence of named adapters is reported, not treated as a failure.
    assert "_notes" in outputs
    assert "no saved named adapters" in outputs["_notes"]["text"]


def test_run_variants_two_adapters_covers_adapters_and_blends(tmp_path):
    base = _tiny_model()
    tokenizer = FakeTokenizer()
    config = _offline_config(base, tokenizer)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    _save_named_adapter(base, config, "positive", results_dir / "positive")
    _save_named_adapter(base, config, "negative", results_dir / "negative")

    outputs = run_variants(
        prompt="the", max_tokens=3, gemini=False, config=config,
        results_dir=str(results_dir),
    )

    # Core variants present (Req 20.2).
    assert {"base", "unmerged", "merged"} <= set(outputs)

    # One adapter:<name> entry per discovered adapter.
    adapter_keys = {name for name in outputs if name.startswith("adapter:")}
    assert adapter_keys == {"adapter:negative", "adapter:positive"}

    # Three blend entries covering exactly alpha 0.25, 0.5, 0.75 (Req 20.2).
    blend_keys = [name for name in outputs if name.startswith("blend:")]
    assert len(blend_keys) == 3
    blend_alphas = {float(name.rsplit("@", 1)[1]) for name in blend_keys}
    assert blend_alphas == set(BLEND_ALPHAS) == {0.25, 0.5, 0.75}

    # Every variant carries a text payload.
    for value in outputs.values():
        assert "text" in value


# --- Side-by-side printing (Req 20.4) --------------------------------------


def test_print_side_by_side_emits_variant_labels(capsys):
    outputs = {
        "base": {"text": "hello world", "gemini": None},
        "unmerged": {"text": "foo bar", "gemini": None},
        "merged": {"text": "baz qux", "gemini": None},
    }

    _print_side_by_side(outputs)

    captured = capsys.readouterr().out
    header = captured.splitlines()[0]
    # The header row lists every variant label side by side (Req 20.4).
    for label in ("base", "unmerged", "merged"):
        assert label in header
    # The cell text appears in the body so columns sit beside each other.
    assert "hello world" in captured
    assert "foo bar" in captured


def test_print_side_by_side_renders_notes(capsys):
    outputs = {
        "base": {"text": "hi", "gemini": None},
        "_notes": {"text": "no saved named adapters found", "gemini": None},
    }

    _print_side_by_side(outputs)

    captured = capsys.readouterr().out
    assert "no saved named adapters found" in captured
    # Notes are not rendered as a variant column.
    assert "_notes" not in captured.splitlines()[0]


# --- No-server / no-dashboard guarantee (Req 20.6) -------------------------


SERVER_MODULES = (
    "flask",
    "fastapi",
    "streamlit",
    "gradio",
    "aiohttp",
    "http.server",
    "tornado",
    "django",
)


def test_compare_import_starts_no_server():
    # Importing compare must not pull in any web/dashboard server (Req 20.6).
    assert compare  # already imported at module top
    for module_name in SERVER_MODULES:
        assert module_name not in sys.modules, (
            f"compare import unexpectedly loaded server module {module_name!r}"
        )


def test_compare_source_imports_no_server():
    # The source itself imports none of the server frameworks (Req 20.6).
    source = (compare.PROJECT_ROOT / "compare.py").read_text(encoding="utf-8")
    for module_name in SERVER_MODULES:
        root = module_name.split(".", 1)[0]
        assert f"import {root}" not in source, (
            f"compare.py unexpectedly imports {root!r}"
        )
