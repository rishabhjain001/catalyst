"""Interactive comparison CLI for Catalyst (Requirement 20, 23.3).

Root-level terminal entry point invoked as::

    python compare.py --prompt "The movie was" --max_tokens 30

It loads the base model and every saved adapter, runs one prompt through the
base model, the LoRA unmerged and merged variants, each named adapter, and the
interpolated blends at alpha 0.25/0.5/0.75, then prints the outputs side by
side. Gemini scoring is optional. The tool is pure terminal: it starts no web
server and renders no dashboard (Req 20.6), so it imports only argparse, the
standard library, torch, and the Catalyst package.
"""

import argparse
import copy
import textwrap
from pathlib import Path

import yaml

from catalyst.evaluate import gemini_judge, generate_text
from catalyst.model import AdapterManager, inject_lora, load_model

# The blends the CLI sweeps between the first two named adapters (Req 20.2).
BLEND_ALPHAS = (0.25, 0.5, 0.75)

# Default prompt so the CLI is runnable with no arguments while still honoring
# an explicit --prompt (Req 20.1).
DEFAULT_PROMPT = "The movie was"

# Where trained adapter-set directories live, matching train.RESULTS_DIR.
DEFAULT_RESULTS_DIR = "results"

# The project root is this file's directory; the config and results paths are
# resolved relative to it so the CLI works regardless of the caller's cwd.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _load_config(config_path=None) -> dict:
    """Read ``configs/default.yaml`` into a dict (Req 9.1, 20.3)."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _rank_by_layer(config: dict) -> dict:
    """Per-layer rank allocation with int keys (matches injection contract)."""
    raw = config.get("rank_by_layer") or {}
    return {int(layer): int(rank) for layer, rank in raw.items()}


def _decoding_config(config: dict, tokenizer, max_tokens: int) -> dict:
    """Build the per-generation config reused for every variant.

    Carries the tokenizer and the deterministic decoding settings (temperature
    0.0, top-k off, top-p 1.0) from the file, overriding only the generated
    length with the CLI ``max_tokens`` so all variants decode identically
    (Req 20.3).
    """
    gen_config = dict(config)
    gen_config["tokenizer"] = tokenizer
    decoding = dict(config.get("decoding") or {})
    decoding["max_new_tokens"] = int(max_tokens)
    gen_config["decoding"] = decoding
    return gen_config


def _load_base(config: dict):
    """Return ``(model, tokenizer)``, reusing a preloaded pair when supplied.

    A caller (or a test) may pass an already-loaded model and tokenizer through
    ``config["model"]`` / ``config["tokenizer"]`` to stay offline; otherwise the
    base model and tokenizer are loaded from ``model_name`` (Req 20.1).
    """
    model = config.get("model")
    tokenizer = config.get("tokenizer")
    if model is not None and tokenizer is not None:
        return model, tokenizer
    return load_model(config)


def _injected_copy(base_model, config: dict):
    """Deep-copy the base model and inject LoRA per the configured targets.

    A copy is used so the pristine base model stays available for the ``base``
    variant while the injected copy carries the adapters (Req 20.2).
    """
    model = copy.deepcopy(base_model)
    targets = config.get("targets") or ["q", "v"]
    alpha = float(config.get("alpha", 16.0))
    inject_lora(model, _rank_by_layer(config), targets, alpha)
    return model


def _discover_adapters(results_dir: str, manager: AdapterManager) -> list:
    """Return the names of adapter-set directories compatible with ``manager``.

    Each trained named adapter is a directory of per-module artifacts under
    ``results/`` (see ``AdapterManager.save_adapter_set``). A directory is
    compatible when it holds an artifact for every module the manager adapts;
    ``add_adapter`` enforces that, so a directory it rejects (a result JSON's
    sibling folder, an adapter trained on a different target set) is skipped
    rather than aborting discovery (Req 20.1).
    """
    base = Path(results_dir)
    if not base.is_dir():
        return []
    discovered = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        try:
            manager.add_adapter(entry.name, str(entry))
        except (FileNotFoundError, NotADirectoryError):
            continue
        discovered.append(entry.name)
    return discovered


def _safe_generate(model, prompt: str, gen_config: dict) -> str:
    """Generate text, returning a readable error string instead of raising.

    The CLI compares many variants in one run; one variant failing (an adapter
    whose rank does not match the configured injection, say) should annotate
    that cell rather than abort the whole comparison.
    """
    try:
        return generate_text(model, prompt, gen_config)
    except Exception as error:
        return f"<error: {error}>"


def _load_unmerged(manager: AdapterManager, name: str) -> bool:
    """Load one named adapter into every managed module without merging.

    Returns True on success. Used to produce the ``unmerged`` variant: each
    module loads its saved ``A``/``B`` while the adapter forward path stays
    active (nothing is folded into the base weight).
    """
    directory = Path(manager.adapters[name])
    for path, module in manager.modules.items():
        module.load_adapter(str(directory / manager._module_file(path)))
    return True


def _apply_blend(manager: AdapterManager, effective: dict):
    """Write blended effective weights into the bases and disable the adapters.

    Returns a restore callback that puts every touched module back exactly as it
    was. ``interpolate_adapters`` returns each module's logical ``d x k``
    effective weight without mutating the live base, so the CLI installs those
    weights (transposed for a GPT-2 ``Conv1D`` layout) and sets ``merged`` so the
    base path alone produces the blended output (Req 20.2).
    """
    import torch

    saved = []
    with torch.no_grad():
        for path, value in effective.items():
            module = manager.modules[path]
            stored = value.t() if module.layer_kind == "conv1d" else value
            saved.append((module, module.base.weight.detach().clone(), module.merged))
            module.base.weight.copy_(stored)
            module.merged = True

    def restore():
        with torch.no_grad():
            for module, weight, merged in saved:
                module.base.weight.copy_(weight)
                module.merged = merged

    return restore


def run_variants(prompt: str, max_tokens: int, gemini: bool, config: dict = None,
                 results_dir: str = None) -> dict:
    """Run ``prompt`` through every variant and return their outputs.

    The result maps each variant name to ``{"text": str, "gemini": dict | None}``.
    Variants are the base model, the LoRA ``unmerged`` and ``merged`` variants,
    one ``adapter:<name>`` entry per discovered named adapter, and one
    ``blend:<a>+<b>@<alpha>`` entry per alpha in ``BLEND_ALPHAS`` between the
    first two named adapters (Req 20.2). All variants decode with the same
    deterministic settings (Req 20.3). When no compatible adapters are found the
    base and the (zero-delta) unmerged/merged variants still run, and the absence
    of named adapters is reported rather than treated as a failure (Req 20.1).

    ``config`` and ``results_dir`` are optional injection points: ``config`` may
    carry a preloaded ``model``/``tokenizer`` to stay offline, and ``results_dir``
    overrides where adapter directories are discovered. Gemini scoring runs per
    variant only when ``gemini`` is set (Req 20.5); it never starts a server
    (Req 20.6).
    """
    config = config or _load_config()
    results_dir = results_dir or config.get("results_dir", DEFAULT_RESULTS_DIR)

    base_model, tokenizer = _load_base(config)
    gen_config = _decoding_config(config, tokenizer, max_tokens)

    outputs = {}
    notes = []

    def emit(name, text):
        score = gemini_judge([text], gen_config) if gemini else None
        outputs[name] = {"text": text, "gemini": score}

    base_model.eval()
    emit("base", _safe_generate(base_model, prompt, gen_config))

    # One injected copy serves discovery, the unmerged/merged demonstration, and
    # the per-adapter activations; a separate clean copy serves the blends so
    # their base weights are pristine when interpolation reads them.
    injected = _injected_copy(base_model, config)
    injected.eval()
    manager = AdapterManager(injected)
    adapter_names = _discover_adapters(results_dir, manager)

    # The unmerged and merged variants share one adapter so the two code paths
    # are compared on equal footing; with no adapters the zero-init injection
    # still exercises both paths.
    demo_name = adapter_names[0] if adapter_names else None
    if demo_name is not None:
        try:
            _load_unmerged(manager, demo_name)
        except Exception as error:
            notes.append(f"unmerged/merged demo adapter {demo_name!r} skipped: {error}")
            demo_name = None
    emit("unmerged", _safe_generate(injected, prompt, gen_config))
    for module in manager.modules.values():
        if not module.merged:
            module.merge_adapter()
    emit("merged", _safe_generate(injected, prompt, gen_config))

    if not adapter_names:
        notes.append("no saved named adapters found; ran base/unmerged/merged only")

    # Each named adapter, activated through swap_adapter (Req 7.3). The modules
    # are merged from the demonstration above, so activate can swap directly.
    for name in adapter_names:
        try:
            manager.activate(name)
            text = _safe_generate(injected, prompt, gen_config)
        except Exception as error:
            text = f"<error: {error}>"
        emit(f"adapter:{name}", text)

    # Blends between the first two named adapters across the configured alphas.
    if len(adapter_names) >= 2:
        name_a, name_b = adapter_names[0], adapter_names[1]
        blend_model = _injected_copy(base_model, config)
        blend_model.eval()
        blend_manager = AdapterManager(blend_model)
        blend_manager.add_adapter(name_a, str(Path(results_dir) / name_a))
        blend_manager.add_adapter(name_b, str(Path(results_dir) / name_b))
        for alpha in BLEND_ALPHAS:
            label = f"blend:{name_a}+{name_b}@{alpha}"
            try:
                effective = blend_manager.blend(name_a, name_b, alpha)
                restore = _apply_blend(blend_manager, effective)
                try:
                    text = _safe_generate(blend_model, prompt, gen_config)
                finally:
                    restore()
            except Exception as error:
                text = f"<error: {error}>"
            emit(label, text)
    else:
        notes.append("need two named adapters to blend; blends skipped")

    if notes:
        outputs["_notes"] = {"text": " | ".join(notes), "gemini": None}
    return outputs


def _print_side_by_side(outputs: dict, width: int = 32) -> None:
    """Print variant outputs in aligned columns (Req 20.4).

    Each variant becomes a fixed-width column; the text is wrapped to the column
    width and the columns are printed row by row so the outputs sit side by side
    in the terminal. Notes, when present, are printed beneath the table.
    """
    notes = outputs.get("_notes")
    variants = [(name, value) for name, value in outputs.items() if name != "_notes"]

    labels = [name for name, _ in variants]
    columns = [
        textwrap.wrap(value["text"] or "", width=width) or [""]
        for _, value in variants
    ]

    header = " | ".join(label[:width].ljust(width) for label in labels)
    separator = "-+-".join("-" * width for _ in labels)
    print(header)
    print(separator)

    height = max(len(column) for column in columns)
    for row in range(height):
        cells = [
            (column[row] if row < len(column) else "").ljust(width)
            for column in columns
        ]
        print(" | ".join(cells))

    has_gemini = any(value["gemini"] is not None for _, value in variants)
    if has_gemini:
        print()
        print("Gemini (LLM-based qualitative evaluation, supplementary):")
        for name, value in variants:
            if value["gemini"] is not None:
                score = value["gemini"]
                print(
                    f"  {name}: coherence={score['sentiment_coherence']:.1f} "
                    f"fluency={score['fluency']:.1f} "
                    f"adherence={score['prompt_adherence']:.1f}"
                )

    if notes is not None:
        print()
        print(f"note: {notes['text']}")


def main() -> None:
    """Parse arguments, run the variants, and print them side by side."""
    parser = argparse.ArgumentParser(
        description="Compare Catalyst model variants on a prompt."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt fed to every variant.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=30,
        help="Number of new tokens each variant generates.",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Score each output with the supplementary Gemini judge.",
    )
    args = parser.parse_args()

    outputs = run_variants(args.prompt, args.max_tokens, args.gemini)
    _print_side_by_side(outputs)


if __name__ == "__main__":
    main()
