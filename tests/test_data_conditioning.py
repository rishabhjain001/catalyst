"""Property test for sentiment conditioning (Property 10).

Verifies that ``condition_text`` prepends ``[POSITIVE]`` for positive labels
and ``[NEGATIVE]`` for negative labels while preserving the original text
exactly after the tag and its single-space separator.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from catalyst.data import NEGATIVE_TAG, POSITIVE_TAG, condition_text

# Arbitrary review text, including empty, whitespace, and unicode, since the
# conditioner must preserve any string verbatim after the tag.
texts = st.text()
# SST-2 is binary: label 1 is positive, label 0 is negative.
labels = st.sampled_from([0, 1])


# Feature: catalyst-lora, Property 10
@settings(max_examples=200)
@given(text=texts, label=labels)
def test_conditioning_tags_and_preserves_text(text, label):
    """Validates: Requirements 8.2, 8.3"""
    result = condition_text(text, label)

    expected_tag = POSITIVE_TAG if label == 1 else NEGATIVE_TAG

    # The output begins with the sentiment tag matching the label.
    assert result.startswith(expected_tag)

    # The original text is preserved exactly after the tag and single space.
    assert result == f"{expected_tag} {text}"
    assert result[len(expected_tag) + 1:] == text
