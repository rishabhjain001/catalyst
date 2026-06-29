"""Architecture-agnostic LoRA injector for Catalyst.

Loads the configured Hugging Face decoder-only transformer and resolves
logical adapter targets (``q``, ``k``, ``v``, ``o``, ``mlp``) to the concrete
projection modules of the loaded architecture. Target resolution inspects the
model's module tree and dimensions rather than matching hardcoded names, so a
new architecture is supported by its structure alone (Requirement 6.2, 6.7).

The transformers import is deferred into ``load_model`` so that importing this
module never hard-fails in an environment where transformers is unavailable;
``resolve_targets`` operates on an already-loaded model and needs only torch.
"""

from pathlib import Path

from torch import nn

from .lora import LoRA_Module

# Logical targets the resolver understands, in the canonical order used for the
# returned identifier lists.
LOGICAL_TARGETS = ("q", "k", "v", "o", "mlp")

# Separates a concrete module's relative path from the slice of a fused
# projection it refers to, e.g. "attn.c_attn#q" names the q slice of the fused
# qkv projection. Non-fused targets carry no suffix.
SLICE_SEP = "#"


def load_model(config: dict):
    """Load the configured model and tokenizer from ``model_name``.

    Returns ``(model, tokenizer)``. The model identifier comes from
    ``config["model_name"]`` and defaults to ``"gpt2"`` (Requirement 9.1).
    """
    model_name = (config or {}).get("model_name", "gpt2")

    # Deferred so module import succeeds even when transformers is absent.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPT-2 has no pad token; reuse eos so batched tokenization has a pad id.
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        for field in ("hidden_size", "n_embd", "d_model"):
            value = getattr(config, field, None)
            if value:
                return int(value)
    raise ValueError("Cannot determine hidden size from the model config.")


def _num_layers(model: nn.Module):
    config = getattr(model, "config", None)
    if config is not None:
        for field in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(config, field, None)
            if value:
                return int(value)
    return None


def _decoder_blocks(model: nn.Module):
    """Return ``[(index, block_module), ...]`` for the repeated decoder blocks.

    The block container is found by locating the ``nn.ModuleList`` whose length
    matches the configured layer count; falling back to the longest
    ``nn.ModuleList`` of submodules. Indices match the keys callers use in a
    per-layer rank allocation.
    """
    target_count = _num_layers(model)
    candidates = [
        module
        for module in model.modules()
        if isinstance(module, nn.ModuleList) and len(module) > 0
    ]
    if not candidates:
        return []

    chosen = None
    if target_count is not None:
        for module in candidates:
            if len(module) == target_count:
                chosen = module
                break
    if chosen is None:
        chosen = max(candidates, key=len)

    return list(enumerate(chosen))


def _logical_dims(module: nn.Module):
    """Return logical ``(out_features, in_features)`` for a projection module.

    ``nn.Linear`` stores its weight as ``(out, in)``; a GPT-2 ``Conv1D`` stores
    it transposed as ``(in, out)``. Recognizing both is how the resolver stays
    architecture-derived without keying off module names. Returns ``None`` for
    modules that are not weight projections.
    """
    if isinstance(module, nn.Linear):
        out_dim, in_dim = module.weight.shape
        return int(out_dim), int(in_dim)
    if type(module).__name__ == "Conv1D":
        in_dim, out_dim = module.weight.shape
        return int(out_dim), int(in_dim)
    return None


def _block_projections(block: nn.Module):
    """Collect projection leaves of one block as ``(relpath, parent, dims)``."""
    projections = []
    for relpath, module in block.named_modules():
        if not relpath:
            continue
        dims = _logical_dims(module)
        if dims is None:
            continue
        parent = relpath.rsplit(".", 1)[0] if "." in relpath else ""
        projections.append((relpath, parent, dims))
    return projections


def _classify_block(block: nn.Module, hidden: int):
    """Map logical roles to concrete projection identifiers for one block.

    Classification is by dimension and structural grouping, not by name:
    a fused qkv projection has output ``3 * hidden``; an attention output is
    square ``hidden x hidden``; an MLP input expands from ``hidden``; an MLP
    output contracts back to ``hidden``. Returns a dict keyed by logical target
    whose values are lists of concrete identifiers.
    """
    projections = _block_projections(block)

    roles = {target: [] for target in LOGICAL_TARGETS}

    # Group projections by their parent submodule so the attention block and
    # the MLP block can be told apart structurally.
    by_parent = {}
    for relpath, parent, dims in projections:
        by_parent.setdefault(parent, []).append((relpath, dims))

    # Classify each parent group as a whole: a group holding a fused qkv (or
    # separate q/k/v/o square projections) is the attention sublayer; a group
    # holding an expand-then-contract pair is the MLP sublayer. Classifying the
    # group as one or the other avoids mistaking the fused qkv projection
    # (which also expands from hidden) for an MLP input.
    for members in by_parent.values():
        fused = [rel for rel, (o, i) in members if o == 3 * hidden and i == hidden]
        square = [rel for rel, (o, i) in members if o == hidden and i == hidden]
        contract = [rel for rel, (o, i) in members if o == hidden and i > hidden]

        if fused:
            # GPT-2 style: one projection produces q, k, v as three equal
            # output slices, concatenated in that order.
            base = fused[0]
            roles["q"].append(f"{base}{SLICE_SEP}q")
            roles["k"].append(f"{base}{SLICE_SEP}k")
            roles["v"].append(f"{base}{SLICE_SEP}v")
            # A square projection alongside the fused qkv is the attention
            # output projection (o).
            for rel in square:
                roles["o"].append(rel)
        elif contract:
            # An expand/contract pair marks the MLP sublayer. The input
            # (expand) projection and the output (contract) projection are both
            # adapted under the logical ``mlp`` target.
            expand = [rel for rel, (o, i) in members if i == hidden and o > hidden]
            for rel in expand:
                roles["mlp"].append(rel)
            for rel in contract:
                roles["mlp"].append(rel)
        elif square:
            # Separate, equally shaped q/k/v/o projections (e.g. Llama). Their
            # dimensions are identical, so names are the only remaining signal
            # and are consulted strictly as a fallback for these architectures.
            _assign_separate(square, roles)

    return roles


def _assign_separate(square_paths, roles):
    # Used only for architectures with separate, equally shaped q/k/v/o
    # projections, where dimensions cannot disambiguate. Names are the only
    # remaining signal here, so they are consulted strictly as a fallback.
    hints = {"q": "q", "k": "k", "v": "v", "o": "o"}
    for rel in square_paths:
        lower = rel.lower()
        for target, hint in hints.items():
            if f"{hint}_proj" in lower or lower.endswith(f".{hint}"):
                roles[target].append(rel)
                break


def resolve_targets(model: nn.Module, targets) -> dict:
    """Map each decoder layer index to the concrete projections to adapt.

    ``targets`` are logical names from ``LOGICAL_TARGETS``; an empty or missing
    list defaults to ``["q", "v"]`` (Requirement 6.3). The returned dict maps a
    layer index to the list of concrete projection identifiers present at that
    layer for the requested targets, derived from the model architecture
    (Requirement 6.2, 6.7). A fused qkv projection yields slice-tagged
    identifiers such as ``"attn.c_attn#q"``.
    """
    requested = [t.lower() for t in targets] if targets else ["q", "v"]
    unknown = [t for t in requested if t not in LOGICAL_TARGETS]
    if unknown:
        raise ValueError(
            f"Unknown logical targets {unknown}; supported targets are "
            f"{list(LOGICAL_TARGETS)}."
        )

    hidden = _hidden_size(model)
    mapping = {}
    for index, block in _decoder_blocks(model):
        roles = _classify_block(block, hidden)
        # Preserve canonical target order and drop duplicates while keeping the
        # first occurrence.
        present = []
        seen = set()
        for target in LOGICAL_TARGETS:
            if target not in requested:
                continue
            for identifier in roles[target]:
                if identifier not in seen:
                    seen.add(identifier)
                    present.append(identifier)
        if present:
            mapping[index] = present
    return mapping


def _locate_module(root: nn.Module, relpath: str):
    """Walk ``relpath`` from ``root`` to the leaf, returning its parent.

    Returns ``(parent, attr, module)`` so the caller can swap the leaf in place
    with ``setattr(parent, attr, replacement)``.
    """
    parts = relpath.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    return parent, attr, getattr(parent, attr)


def _adapter_param_ids(model: nn.Module) -> set:
    # Identity of every adapter A/B tensor, used to tell trainable adapter
    # parameters apart from frozen base parameters by object identity rather
    # than by fragile name matching.
    ids = set()
    for module in model.modules():
        if isinstance(module, LoRA_Module):
            ids.add(id(module.A))
            ids.add(id(module.B))
    return ids


def inject_lora(
    model: nn.Module, rank_by_layer: dict, targets, alpha: float
) -> nn.Module:
    """Replace resolved projections with ``LoRA_Module`` adapters in place.

    For each layer index present in ``rank_by_layer`` (per-layer rank,
    Requirement 6.1) the resolved targets at that layer are wrapped with a
    ``LoRA_Module`` at the layer's rank. After wrapping, every non-adapter
    parameter is frozen and ``check_grad_leak`` runs so training never begins
    with a leaking base weight (Requirement 6.4).

    Fused projections (GPT-2's ``c_attn`` carrying ``#q``/``#k``/``#v`` slice
    tags) name one physical module three times. ``LoRA_Module`` adapts an entire
    weight, not an output slice, so a fused projection is wrapped once if any of
    its slices are requested; slice-level selectivity collapses to adapting the
    whole fused projection.
    """
    mapping = resolve_targets(model, targets)
    blocks = dict(_decoder_blocks(model))
    model_name = getattr(getattr(model, "config", None), "name_or_path", None)
    requested = [t.lower() for t in targets] if targets else ["q", "v"]

    for index, rank in rank_by_layer.items():
        identifiers = mapping.get(index)
        block = blocks.get(index)
        if not identifiers or block is None:
            continue
        wrapped = set()
        for identifier in identifiers:
            relpath = identifier.split(SLICE_SEP, 1)[0]
            # Distinct slices of one fused projection share a relpath; wrapping
            # it once avoids double-wrapping the same physical module.
            if relpath in wrapped:
                continue
            parent, attr, base = _locate_module(block, relpath)
            adapter = LoRA_Module(
                base,
                rank=rank,
                alpha=alpha,
                model_name=model_name,
                target_modules=requested,
            )
            setattr(parent, attr, adapter)
            wrapped.add(relpath)

    # Freeze everything that is not an adapter matrix so the adapters are the
    # only trainable parameters before the gradient-leak assertion runs.
    adapter_ids = _adapter_param_ids(model)
    for param in model.parameters():
        if id(param) not in adapter_ids:
            param.requires_grad = False

    check_grad_leak(model)
    return model


def check_grad_leak(model: nn.Module) -> None:
    """Assert adapters are trainable and every base weight is frozen.

    Every ``LoRA_Module`` ``A``/``B`` must have ``requires_grad=True`` and no
    other parameter may, so gradients flow only into the adapters. On the first
    violation this raises immediately, naming the offending parameter
    (Requirement 6.5, 6.6).
    """
    adapter_ids = _adapter_param_ids(model)
    for name, param in model.named_parameters():
        if id(param) in adapter_ids:
            if not param.requires_grad:
                raise RuntimeError(
                    f"LoRA adapter parameter '{name}' must be trainable but "
                    f"has requires_grad=False."
                )
        elif param.requires_grad:
            raise RuntimeError(
                f"Base weight '{name}' leaks gradients: requires_grad=True "
                f"after injection."
            )


class AdapterManager:
    """Coordinates named adapters that share one frozen base (Req 7).

    A model adapted by ``inject_lora`` holds many ``LoRA_Module`` instances, one
    per resolved projection per layer, all wrapping the same frozen base. A
    named adapter such as ``"positive"`` or ``"negative"`` (Req 7.1) is therefore
    not a single artifact but a *set* of per-module ``save_adapter`` artifacts.
    This manager represents each named adapter as a directory holding one file
    per adapted module, keyed by the module's qualified path, so the positive
    and negative adapters live as sibling directories over the one shared base
    (Req 7.2).

    ``activate`` routes every managed module through ``swap_adapter`` (Req 7.3)
    and ``blend`` routes every managed module through ``interpolate_adapters``
    over ``[0.0, 1.0]`` (Req 7.4). Because the model has many modules, ``blend``
    generalizes the single-tensor result of one ``interpolate_adapters`` call to
    a mapping from module path to that module's effective weight, leaving every
    live base weight untouched so the shared base is preserved during blending.
    """

    def __init__(self, model: nn.Module):
        # Order is the model's own module order so per-module files line up
        # deterministically across save, activate, and blend.
        self.modules = {
            path: module
            for path, module in model.named_modules()
            if isinstance(module, LoRA_Module)
        }
        if not self.modules:
            raise ValueError(
                "AdapterManager requires a model with injected LoRA_Module "
                "adapters; none were found."
            )
        self.adapters = {}
        self.active = None

    @staticmethod
    def _module_file(path: str) -> str:
        # Dots in a module path would split the file stem, so flatten them into
        # a single unambiguous stem per module.
        return f"{path.replace('.', '__')}.pt"

    def _resolve_dir(self, name: str) -> Path:
        if name not in self.adapters:
            raise KeyError(
                f"No adapter named {name!r}; register it with add_adapter first."
            )
        return Path(self.adapters[name])

    def save_adapter_set(self, name: str, directory: str) -> str:
        """Write every managed module's adapter into ``directory`` and register.

        This is the counterpart the trainer uses to persist a named adapter:
        each module's current ``A``/``B`` is saved under its module-path stem so
        the directory becomes a complete, reloadable named adapter.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for path, module in self.modules.items():
            module.save_adapter(str(target / self._module_file(path)))
        self.adapters[name] = str(target)
        return str(target)

    def add_adapter(self, name: str, path: str) -> None:
        """Register a named adapter by the directory holding its module files.

        The directory must contain one artifact per managed module so the named
        adapter can be activated as a whole; a missing file is reported rather
        than discovered mid-activation.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Adapter {name!r} path {path!r} is not a directory of "
                f"per-module artifacts."
            )
        missing = [
            self._module_file(module_path)
            for module_path in self.modules
            if not (directory / self._module_file(module_path)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Adapter {name!r} directory {path!r} is missing artifacts for "
                f"{len(missing)} module(s): {missing[:3]}."
            )
        self.adapters[name] = str(directory)

    def activate(self, name: str) -> None:
        """Make ``name`` the active adapter, routing through ``swap_adapter``.

        ``swap_adapter`` requires a currently merged adapter, so a freshly
        injected module (still unmerged, with zero-init ``B``) is merged first;
        that initial merge folds a zero delta and leaves the base unchanged.
        Every module then swaps to its file from the named adapter's directory.
        """
        directory = self._resolve_dir(name)
        for path, module in self.modules.items():
            if not module.merged:
                module.merge_adapter()
            module.swap_adapter(str(directory / self._module_file(path)))
        self.active = name

    def blend(self, name_a: str, name_b: str, alpha: float) -> dict:
        """Blend two named adapters at ``alpha`` without mutating the base.

        Each managed module computes its own effective weight through
        ``interpolate_adapters`` (which validates ``alpha`` is in ``[0.0, 1.0]``
        and never touches the live base). The result maps each module path to
        that module's effective weight, so callers can read the full blended
        state while the shared frozen base stays intact (Req 7.2, 7.4).
        """
        dir_a = self._resolve_dir(name_a)
        dir_b = self._resolve_dir(name_b)
        effective = {}
        for path, module in self.modules.items():
            file_a = str(dir_a / self._module_file(path))
            file_b = str(dir_b / self._module_file(path))
            effective[path] = module.interpolate_adapters(file_a, file_b, alpha)
        return effective
