"""Tests utility functions."""


import pytest

from fastllm.utils import get_logit_bias


@pytest.mark.parametrize(
    "model_name, patterns, bias, expected_logit_bias, expected_tokens",
    [
        (
            "gpt-4",
            ["Hello World!"],
            100,
            {"9906": 100, "4435": 100, "0": 100},
            ["!", "Hello", " World"],
        ),
        (
            "gpt-4",
            [r"(1|2)", r"(2|3)"],
            -100,
            {"16": -100, "17": -100, "18": -100},
            ["1", "2", "3"],
        ),
        (
            "gpt-4",
            [r"(H(e(l(l(o?)?)?)?)?)?"],
            -100,
            {"39": -100, "1548": -100, "9906": -100, "81394": -100, "33813": -100},
            ["H", "He", "Hello", "Hell", "Hel"],
        ),
    ],
)
def test_logit_bias(model_name, patterns, bias, expected_logit_bias, expected_tokens):
    """Test logit_bias function."""

    logit_bias, tokens, _ = get_logit_bias(model_name, patterns, bias)

    assert logit_bias == expected_logit_bias
    assert tokens == expected_tokens


def test_logit_bias_errors():
    """Test logit_bias function."""

    with pytest.raises(ValueError):
        get_logit_bias("gpt-4", [r"[0-9]{5}"])
