"""Tests utilities."""


from fastllm.utils import Functions


def test_describe():
    """Tests the function describe method."""

    def add_two_numbers(a: int, b: int) -> int:
        """Adds two integer numbers together."""

        return a + b

    description = Functions.describe(add_two_numbers)

    assert description["description"] == "Adds two integer numbers together."
