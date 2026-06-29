"""Property test for reproducibility metadata completeness (Property 15).

``capture_metadata`` both sets the random seeds and returns the metadata record
that makes an experiment reproducible (Req 10). This test checks that for any
configuration the returned record carries every required field with a sound
value: a JSON-serializable config snapshot, the torch/numpy/random seeds (all
equal to the configured seed, plus ``cuda`` exactly when CUDA is available), a
git commit that is either a 40-char hex string or the literal ``"unavailable"``,
non-empty Python and PyTorch versions, hardware info (CPU model and a positive
total RAM), an ISO 8601 timestamp, the tokenizer name and vocab size, and a
``config_sha256`` equal to the SHA-256 of ``configs/default.yaml``. It also
checks that the seeds are actually applied, not merely recorded, by confirming
that re-capturing with the same seed reproduces a sequence of ``torch.randn``
draws.

Shape/field convention matches ``catalyst/experiments.capture_metadata``.
"""

import hashlib
import json
import re
from datetime import datetime

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.experiments import DEFAULT_CONFIG_PATH, capture_metadata


# A git commit hash is exactly 40 lowercase hex characters; anything else must
# be the documented sentinel ``"unavailable"`` (Req 10.4).
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

# The reference digest the metadata's ``config_sha256`` must match (Req 10.6):
# the SHA-256 of the raw bytes of the same config file the function hashes.
with open(DEFAULT_CONFIG_PATH, "rb") as _handle:
    _REFERENCE_CONFIG_SHA256 = hashlib.sha256(_handle.read()).hexdigest()


class _FakeTokenizer:
    """A minimal tokenizer stub exposing ``name_or_path`` and ``vocab_size``.

    Mirrors the two attributes ``capture_metadata`` reads from a live tokenizer
    so the tokenizer-present branch is exercised without loading a real model.
    """

    def __init__(self, name_or_path: str, vocab_size: int):
        self.name_or_path = name_or_path
        self.vocab_size = vocab_size


def _model_names() -> st.SearchStrategy:
    """Strategy over varied, JSON-safe model-name strings."""
    return st.text(
        alphabet=st.characters(
            min_codepoint=0x20, max_codepoint=0x7E, blacklist_characters="\\"
        ),
        min_size=0,
        max_size=24,
    )


@st.composite
def configs(draw):
    """Draw a varied experiment configuration.

    Always carries a ``model_name`` and a ``seed``; sometimes carries a fake
    tokenizer (with its own name and vocab size) and some extra JSON-serializable
    fields, so both the tokenizer-present and tokenizer-absent branches and a
    range of snapshots are exercised.
    """
    config = {
        "model_name": draw(_model_names()),
        "seed": draw(st.integers(min_value=0, max_value=2**31 - 1)),
    }

    if draw(st.booleans()):
        config["tokenizer"] = _FakeTokenizer(
            draw(_model_names()),
            draw(st.integers(min_value=1, max_value=300000)),
        )

    # A handful of extra JSON-serializable fields to vary the snapshot.
    if draw(st.booleans()):
        config["alpha"] = draw(
            st.floats(
                min_value=0.1, max_value=64.0, allow_nan=False, allow_infinity=False
            )
        )
    if draw(st.booleans()):
        config["targets"] = draw(
            st.lists(st.sampled_from(["q", "v", "k", "o"]), max_size=4, unique=True)
        )

    return config


# Feature: catalyst-lora, Property 15
@settings(max_examples=100, deadline=None)
@given(config=configs(), n=st.integers(min_value=1, max_value=5))
def test_metadata_is_complete_and_well_formed(config, n):
    """Validates: Requirements 10.2, 10.3, 10.4, 10.5, 10.6, 10.8"""
    seed = config["seed"]
    has_tokenizer = "tokenizer" in config

    metadata = capture_metadata(config, n)

    # Identifies the experiment it belongs to.
    assert metadata["experiment"] == n

    # Config snapshot is present, a dict, and JSON-serializable (Req 10.2).
    snapshot = metadata["config_snapshot"]
    assert isinstance(snapshot, dict)
    json.dumps(snapshot)  # raises if not serializable
    # The live tokenizer object is never carried into the serializable snapshot.
    assert "tokenizer" not in snapshot

    # All three seeds are set to the configured seed; cuda iff CUDA available
    # (Req 10.3, 10.7).
    seeds = metadata["seeds"]
    assert seeds["torch"] == seed
    assert seeds["numpy"] == seed
    assert seeds["random"] == seed
    assert ("cuda" in seeds) == torch.cuda.is_available()
    if torch.cuda.is_available():
        assert seeds["cuda"] == seed

    # Git commit is a 40-char hex string or the literal sentinel (Req 10.4).
    git = metadata["git_commit"]
    assert git == "unavailable" or _HEX40.match(git)

    # Python and PyTorch versions are non-empty strings.
    assert isinstance(metadata["python_version"], str) and metadata["python_version"]
    assert isinstance(metadata["pytorch_version"], str) and metadata["pytorch_version"]

    # Hardware info: a CPU model string and a positive total RAM (Req 10.5).
    hardware = metadata["hardware"]
    assert isinstance(hardware["cpu_model"], str)
    assert isinstance(hardware["total_ram_bytes"], int)
    assert hardware["total_ram_bytes"] > 0

    # Timestamp parses as ISO 8601.
    assert isinstance(datetime.fromisoformat(metadata["timestamp"]), datetime)

    # Tokenizer name and vocab size are present (Req 10.8). With a live
    # tokenizer they reflect it; without one the name falls back to model_name
    # and the vocab size is None (never fabricated).
    assert "tokenizer_name" in metadata
    assert "tokenizer_vocab_size" in metadata
    if has_tokenizer:
        tokenizer = config["tokenizer"]
        assert metadata["tokenizer_name"] == tokenizer.name_or_path
        assert metadata["tokenizer_vocab_size"] == tokenizer.vocab_size
    else:
        assert metadata["tokenizer_name"] == config["model_name"]
        assert metadata["tokenizer_vocab_size"] is None

    # config_sha256 equals the SHA-256 of the default config file (Req 10.6).
    assert metadata["config_sha256"] == _REFERENCE_CONFIG_SHA256


# Feature: catalyst-lora, Property 15
@settings(max_examples=100, deadline=None)
@given(config=configs(), n=st.integers(min_value=1, max_value=5))
def test_seeds_are_actually_applied_not_just_recorded(config, n):
    """Validates: Requirements 10.3"""
    # Capturing metadata sets the seed, so a fixed sequence of draws afterward
    # is reproducible across two captures with the same seed (Req 10.3, 10.7).
    capture_metadata(config, n)
    first_a = torch.randn(4)
    first_b = torch.randn(4)

    capture_metadata(config, n)
    second_a = torch.randn(4)
    second_b = torch.randn(4)

    assert torch.equal(first_a, second_a)
    assert torch.equal(first_b, second_b)


def test_config_sha256_matches_an_arbitrary_passed_config_file(tmp_path):
    """Validates: Requirements 10.6

    When a ``config_path`` is supplied, the recorded digest is the SHA-256 of
    that file's bytes, not the default config's.
    """
    custom = tmp_path / "custom.yaml"
    custom.write_bytes(b"model_name: gpt2\nseed: 123\n")
    expected = hashlib.sha256(custom.read_bytes()).hexdigest()

    metadata = capture_metadata(
        {"model_name": "gpt2", "seed": 123}, 1, config_path=str(custom)
    )

    assert metadata["config_sha256"] == expected
    assert metadata["config_sha256"] != _REFERENCE_CONFIG_SHA256
