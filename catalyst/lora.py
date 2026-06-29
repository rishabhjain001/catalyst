"""From-scratch LoRA adapter core.

Wraps one frozen base weight (``nn.Linear`` or GPT-2 ``Conv1D``) with a
trainable low-rank pair and exposes the forward composition documented in the
Catalyst design. The module is intentionally free of project dependencies so
the mathematical core can be validated in isolation.
"""

import warnings
from pathlib import Path

import torch
from torch import nn

# Bumped only when the on-disk adapter layout changes incompatibly; load
# rejects any artifact whose stored version differs.
FORMAT_VERSION = 1

# GPT-2 stores its projections as transformers ``Conv1D`` whose weight is
# transposed relative to ``nn.Linear``. Import it when transformers is present
# so the layer kind can be detected by type; fall back to a name check when it
# is not installed (the core must work without the model layer available).
try:
    from transformers.pytorch_utils import Conv1D as _HFConv1D
except Exception:
    _HFConv1D = None


def _is_conv1d(module):
    if _HFConv1D is not None and isinstance(module, _HFConv1D):
        return True
    # Name fallback covers environments where transformers is absent but a
    # Conv1D-shaped base (weight stored as (in, out)) is supplied for testing.
    return type(module).__name__ == "Conv1D"


class LoRA_Module(nn.Module):
    """A frozen base weight adapted by a low-rank ``A``/``B`` pair.

    ``W`` has logical shape ``d x k`` (``nn.Linear`` convention: ``d`` outputs,
    ``k`` inputs). ``A`` is ``r x k`` (Gaussian), ``B`` is ``d x r`` (zero), so
    the update term equals ``(alpha / r) * x @ A.T @ B.T`` and folds to
    ``(alpha / r) * B @ A`` in weight space.
    """

    def __init__(
        self,
        base: nn.Module,
        rank: int,
        alpha: float,
        model_name: str = None,
        target_modules: list = None,
    ):
        super().__init__()

        if isinstance(base, nn.Linear):
            self.layer_kind = "linear"
            out_dim, in_dim = base.weight.shape
        elif _is_conv1d(base):
            self.layer_kind = "conv1d"
            # Conv1D stores weight as (in, out); the logical W is its transpose.
            in_dim, out_dim = base.weight.shape
        else:
            raise TypeError(
                f"LoRA_Module supports nn.Linear and GPT-2 Conv1D bases, "
                f"got {type(base).__name__}"
            )

        self.base = base
        self.in_features = in_dim
        self.out_features = out_dim
        self.alpha = float(alpha)

        # Recorded so save_adapter can stamp provenance and load_adapter can
        # reject artifacts trained against a different model or target set.
        self.model_name = model_name
        self.target_modules = target_modules

        limit = min(out_dim, in_dim)
        if rank > limit:
            # Clamp rather than abort so a misconfigured layer stays usable,
            # but signal the misconfiguration loudly.
            warnings.warn(
                f"Requested rank {rank} exceeds min(d, k) = {limit} for a "
                f"{out_dim}x{in_dim} weight; clamping effective rank to {limit}.",
                stacklevel=2,
            )
            rank = limit
        self.rank = rank
        self.scaling = self.alpha / rank

        # Freeze the wrapped base (weight and any bias) so only the adapter
        # trains.
        for param in self.base.parameters():
            param.requires_grad = False

        weight = base.weight
        self.A = nn.Parameter(
            torch.randn(rank, in_dim, dtype=weight.dtype, device=weight.device)
        )
        self.B = nn.Parameter(
            torch.zeros(out_dim, rank, dtype=weight.dtype, device=weight.device)
        )

        # When True the delta has been folded into the base weight, so the
        # adapter forward path is disabled to avoid double counting.
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The wrapped module supplies the base path for both layer kinds:
        # nn.Linear computes x @ W.T (+ bias), Conv1D computes x @ stored, which
        # is the same as x @ W.T because Conv1D stores the transposed weight.
        base_out = self.base(x)
        if self.merged:
            # The base weight already carries the delta; adding the adapter term
            # again would apply it twice.
            return base_out
        update = (x @ self.A.t()) @ self.B.t()
        return base_out + self.scaling * update

    def _delta_weight(self) -> torch.Tensor:
        # Logical update in nn.Linear convention: B (d x r) @ A (r x k) -> d x k.
        return self.scaling * (self.B @ self.A)

    def merge_adapter(self) -> None:
        if self.merged:
            raise RuntimeError(
                "Adapter is already merged; unmerge before merging again."
            )
        delta = self._delta_weight()
        # Conv1D stores the weight transposed (k x d), so the delta is applied
        # transposed to match the stored layout; nn.Linear stores W as d x k.
        if self.layer_kind == "conv1d":
            delta = delta.t()
        with torch.no_grad():
            self.base.weight.add_(delta)
        self.merged = True

    def unmerge_adapter(self) -> None:
        if not self.merged:
            raise RuntimeError(
                "Adapter is not merged; nothing to unmerge."
            )
        delta = self._delta_weight()
        if self.layer_kind == "conv1d":
            delta = delta.t()
        with torch.no_grad():
            self.base.weight.sub_(delta)
        self.merged = False

    def save_adapter(self, path: str) -> None:
        artifact = {
            "format_version": FORMAT_VERSION,
            "model_name": self.model_name,
            "target_modules": self.target_modules,
            "rank": self.rank,
            "alpha": self.alpha,
            "A": self.A.detach().cpu().clone(),
            "B": self.B.detach().cpu().clone(),
        }
        torch.save(artifact, path)

    def load_adapter(self, path: str) -> None:
        artifact = torch.load(path, map_location="cpu", weights_only=True)

        # Validate everything before touching A/B so a rejected artifact leaves
        # the live adapter unchanged (Req 3.4, 3.6).
        stored_version = artifact.get("format_version")
        if stored_version != FORMAT_VERSION:
            raise ValueError(
                f"Incompatible adapter format_version {stored_version!r}; "
                f"this module expects {FORMAT_VERSION}."
            )

        stored_model = artifact.get("model_name")
        if stored_model != self.model_name:
            raise ValueError(
                f"Adapter model_name {stored_model!r} does not match this "
                f"module's model_name {self.model_name!r}."
            )

        stored_a = artifact["A"]
        stored_b = artifact["B"]
        if tuple(stored_a.shape) != tuple(self.A.shape):
            raise ValueError(
                f"Adapter A shape {tuple(stored_a.shape)} does not match this "
                f"module's A shape {tuple(self.A.shape)}."
            )
        if tuple(stored_b.shape) != tuple(self.B.shape):
            raise ValueError(
                f"Adapter B shape {tuple(stored_b.shape)} does not match this "
                f"module's B shape {tuple(self.B.shape)}."
            )

        with torch.no_grad():
            self.A.copy_(stored_a.to(device=self.A.device, dtype=self.A.dtype))
            self.B.copy_(stored_b.to(device=self.B.device, dtype=self.B.dtype))

    def swap_adapter(self, path: str) -> None:
        # Precondition: an adapter is currently merged into the base weight.
        # Swapping means replacing that merged adapter with a new one in a
        # single call (Req 4.1).
        if not self.merged:
            raise RuntimeError(
                "swap_adapter requires a currently merged adapter; "
                "merge an adapter before swapping."
            )

        # Restore the original base weight so the new adapter merges onto W
        # rather than onto the previous adapter's already-folded delta.
        self.unmerge_adapter()
        try:
            self.load_adapter(path)
        except Exception:
            # load_adapter validates the artifact before mutating A/B, so a
            # rejected swap leaves the previous adapter's matrices intact.
            # Re-merge them to restore the exact pre-swap merged state, then
            # surface the original descriptive error (Req 4.3).
            self.merge_adapter()
            raise
        self.merge_adapter()

    def _load_delta(self, path: str) -> torch.Tensor:
        # Load a saved adapter and return its logical d x k delta without
        # touching this module's live A/B. Validation mirrors load_adapter so a
        # malformed or mismatched artifact raises a descriptive error rather
        # than producing a silently wrong effective weight.
        artifact = torch.load(path, map_location="cpu", weights_only=True)

        stored_version = artifact.get("format_version")
        if stored_version != FORMAT_VERSION:
            raise ValueError(
                f"Incompatible adapter format_version {stored_version!r}; "
                f"this module expects {FORMAT_VERSION}."
            )

        stored_model = artifact.get("model_name")
        if stored_model != self.model_name:
            raise ValueError(
                f"Adapter model_name {stored_model!r} does not match this "
                f"module's model_name {self.model_name!r}."
            )

        stored_a = artifact["A"]
        stored_b = artifact["B"]
        if tuple(stored_a.shape) != tuple(self.A.shape):
            raise ValueError(
                f"Adapter A shape {tuple(stored_a.shape)} does not match this "
                f"module's A shape {tuple(self.A.shape)}."
            )
        if tuple(stored_b.shape) != tuple(self.B.shape):
            raise ValueError(
                f"Adapter B shape {tuple(stored_b.shape)} does not match this "
                f"module's B shape {tuple(self.B.shape)}."
            )

        weight = self.base.weight
        stored_a = stored_a.to(device=weight.device, dtype=weight.dtype)
        stored_b = stored_b.to(device=weight.device, dtype=weight.dtype)
        scaling = artifact["alpha"] / artifact["rank"]
        # Logical d x k delta: B (d x r) @ A (r x k).
        return scaling * (stored_b @ stored_a)

    def interpolate_adapters(
        self, adapter_a: str, adapter_b: str, alpha: float
    ) -> torch.Tensor:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"interpolate_adapters requires alpha in [0.0, 1.0], "
                f"got {alpha}."
            )

        delta_a = self._load_delta(adapter_a)
        delta_b = self._load_delta(adapter_b)

        # The live base is never mutated: read W logically (Conv1D stores it
        # transposed) and blend into a fresh tensor so callers receive the
        # effective weight while the module keeps its current state.
        if self.layer_kind == "conv1d":
            base_weight = self.base.weight.t()
        else:
            base_weight = self.base.weight
        base_weight = base_weight.detach()

        return base_weight + (1.0 - alpha) * delta_a + alpha * delta_b

    @staticmethod
    def list_adapters(directory: str) -> list:
        # An identifier is a file stem whose contents load as an adapter
        # artifact; non-adapter files in the directory are ignored so the
        # returned set is exactly the adapters that were saved.
        found = []
        for entry in sorted(Path(directory).iterdir()):
            if not entry.is_file():
                continue
            try:
                artifact = torch.load(
                    entry, map_location="cpu", weights_only=True
                )
            except Exception:
                continue
            if isinstance(artifact, dict) and {
                "format_version",
                "A",
                "B",
            } <= artifact.keys():
                found.append(entry.stem)
        return found
