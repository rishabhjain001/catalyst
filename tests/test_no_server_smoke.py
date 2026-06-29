"""No-auto-switch and no-server smoke test (task 14.2).

Two guarantees are asserted here, both fully offline:

* **No automatic model switching (Req 9.2, 9.3).** The model identifier comes
  solely from ``config["model_name"]`` (defaulting to ``"gpt2"``) and is never
  changed by the system on its own. This is checked behaviorally by
  monkeypatching the transformers loaders so ``load_model`` records the exact
  name it requests without touching the network, and structurally by confirming
  ``load_model`` derives the identifier only from the config / ``"gpt2"`` default
  and never reassigns it.

* **No web server or dashboard (Req 20.6).** Neither the root ``compare.py`` nor
  any ``catalyst/*.py`` module imports a web/dashboard framework, and importing
  ``compare`` and the ``catalyst`` package pulls none of them into
  ``sys.modules``.

The project root is resolved relative to this file so the assertions hold
regardless of the directory pytest runs from.
"""

import subprocess
import sys

from pathlib import Path

# This file lives in ``<root>/tests/``; one level up is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALYST_DIR = PROJECT_ROOT / "catalyst"

# Web/dashboard frameworks that a pure terminal tool must never import (Req
# 20.6). Each entry is checked both as a source import and as a loaded module.
SERVER_MODULES = (
    "flask",
    "fastapi",
    "streamlit",
    "gradio",
    "aiohttp",
    "tornado",
    "django",
    "uvicorn",
    "bottle",
    "dash",
    "http.server",
)


# --- No automatic model switching (Req 9.2, 9.3) ---------------------------


class _DummyModel:
    """Stand-in returned by the patched model loader; no network, no weights."""


class _DummyTokenizer:
    """Stand-in tokenizer with a pad token so load_model's eos branch is skipped."""

    pad_token = "<pad>"
    eos_token = "<pad>"


def _patch_loaders(monkeypatch):
    """Patch the transformers loaders to record the requested name offline.

    ``load_model`` does ``from transformers import AutoModelForCausalLM,
    AutoTokenizer`` inside the function, so patching the classmethods on the
    transformers classes intercepts the exact identifier passed through without
    any download.
    """
    import transformers

    recorded = {}

    def fake_model(name, *args, **kwargs):
        recorded["model"] = name
        return _DummyModel()

    def fake_tokenizer(name, *args, **kwargs):
        recorded["tokenizer"] = name
        return _DummyTokenizer()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", fake_model
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", fake_tokenizer
    )
    return recorded


def test_load_model_uses_configured_name_only(monkeypatch):
    """Validates: Requirements 9.2"""
    recorded = _patch_loaders(monkeypatch)
    from catalyst.model import load_model

    # An explicitly configured model_name is the exact identifier requested for
    # both the model and the tokenizer; nothing rewrites it.
    load_model({"model_name": "distilgpt2"})
    assert recorded["model"] == "distilgpt2"
    assert recorded["tokenizer"] == "distilgpt2"


def test_load_model_defaults_to_gpt2(monkeypatch):
    """Validates: Requirements 9.2"""
    recorded = _patch_loaders(monkeypatch)
    from catalyst.model import load_model

    # With no model_name configured, the sole fallback is the documented "gpt2"
    # default, never some other model chosen at runtime.
    load_model({})
    assert recorded["model"] == "gpt2"
    assert recorded["tokenizer"] == "gpt2"


def test_load_model_does_not_switch_models_automatically(monkeypatch):
    """Validates: Requirements 9.3"""
    recorded = _patch_loaders(monkeypatch)
    from catalyst.model import load_model

    # The same configured name is honored regardless of runtime conditions: the
    # system never substitutes a different model on its own. Loading twice with
    # the same config yields the same requested identifier both times.
    load_model({"model_name": "gpt2-medium"})
    first = recorded["model"]
    load_model({"model_name": "gpt2-medium"})
    second = recorded["model"]
    assert first == second == "gpt2-medium"


def test_model_identifier_source_is_only_config(monkeypatch):
    """Validates: Requirements 9.2, 9.3

    Structural guard: the base-model identifier in ``load_model`` is read solely
    from ``config["model_name"]`` with the ``"gpt2"`` default and is never
    reassigned afterward, so there is no code path that switches models
    automatically.
    """
    source = (CATALYST_DIR / "model.py").read_text(encoding="utf-8")

    # Isolate the load_model function body so the check is scoped to where the
    # base model is actually loaded.
    start = source.index("def load_model(")
    end = source.index("\ndef ", start)
    body = source[start:end]

    # The identifier comes from the config with the documented default, and that
    # is the only assignment to model_name in the function (no conditional
    # reassignment to a different model).
    assert '(config or {}).get("model_name", "gpt2")' in body
    assignments = [
        line for line in body.splitlines()
        if line.strip().startswith("model_name =")
    ]
    assert len(assignments) == 1, (
        f"load_model reassigns model_name {len(assignments)} times; the "
        f"identifier must come solely from config['model_name']."
    )


# --- No web server or dashboard (Req 20.6) ---------------------------------


def _python_sources():
    """Yield (label, source_text) for compare.py and every catalyst module."""
    yield "compare.py", (PROJECT_ROOT / "compare.py").read_text(encoding="utf-8")
    for path in sorted(CATALYST_DIR.glob("*.py")):
        yield f"catalyst/{path.name}", path.read_text(encoding="utf-8")


def test_sources_import_no_web_or_dashboard_framework():
    """Validates: Requirements 20.6"""
    for label, source in _python_sources():
        for module_name in SERVER_MODULES:
            root = module_name.split(".", 1)[0]
            assert f"import {root}" not in source, (
                f"{label} unexpectedly imports web/dashboard module {root!r}"
            )
            assert f"from {root}" not in source, (
                f"{label} unexpectedly imports from web/dashboard module {root!r}"
            )


def _server_modules_loaded_by_import(target):
    """Import ``target`` in a clean interpreter; return loaded SERVER_MODULES.

    Checking ``sys.modules`` in-process is unreliable here: other tests import
    the Gemini client and HuggingFace ``datasets``, both of which transitively
    load the ``aiohttp`` HTTP client, leaving it resident regardless of what
    ``compare``/``catalyst`` themselves pull in. Re-importing in a fresh
    subprocess measures the real transitive closure of the import under test.
    """
    probe = (
        "import importlib, sys\n"
        f"importlib.import_module({target!r})\n"
        f"server = {SERVER_MODULES!r}\n"
        "print(','.join(name for name in server if name in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"probe importing {target!r} failed: {completed.stderr}"
    )
    loaded = completed.stdout.strip()
    return [name for name in loaded.split(",") if name]


def test_importing_compare_loads_no_server():
    """Validates: Requirements 20.6"""
    # Import the root CLI fresh and confirm it dragged in no server framework.
    loaded = _server_modules_loaded_by_import("compare")
    assert not loaded, f"importing compare loaded server modules {loaded!r}"


def test_importing_catalyst_package_loads_no_server():
    """Validates: Requirements 20.6"""
    loaded = _server_modules_loaded_by_import("catalyst")
    assert not loaded, f"importing catalyst loaded server modules {loaded!r}"
