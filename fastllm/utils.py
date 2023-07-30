"""Utility functions for fastllm package."""

import re
from itertools import chain

import exrex
import tiktoken


def get_logit_bias(
    model_name: str, pattern: list[str] | str, bias: int = 100, limit: int = 10000
) -> tuple[dict[str, int], list[str]]:
    """Given a model name, a pattern and a bias, returns a logit bias dict."""

    tokenizer = tiktoken.encoding_for_model(model_name)

    encoded_strings = []
    for p in [pattern] if isinstance(pattern, str) else pattern:
        try:
            re.compile(p)

            is_valid_regex = True
        except re.error:
            is_valid_regex = False

        if is_valid_regex:
            count = exrex.count(p, limit=limit + 1)
            if count > limit:
                raise ValueError(
                    f"Exhausted limit of {limit} strings for pattern {p} \
with {count} strings."
                )

            strings = exrex.generate(p, limit=limit)
        else:
            strings = [p]

        encoded_strings += tokenizer.encode_batch(strings)

    tokens = set(chain.from_iterable(encoded_strings))

    return {f"{t}": bias for t in tokens}, [tokenizer.decode([t]) for t in tokens]
