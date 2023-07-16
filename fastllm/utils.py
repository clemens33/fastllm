"""Utility functions."""


import ast
import inspect
from typing import Any, Callable, get_type_hints


class Functions:
    """Utility functions for working with function calls in LLMs."""

    TYPE_MAP = {int: "integer", str: "string", bool: "boolean", float: "number"}

    @classmethod
    def _parameters(cls, fn: Callable[..., Any]) -> dict:
        """Returns a description of the function as well as its parameters."""

        return {
            "type": "object",
            "properties": cls._extract_properties(fn),
            "required": cls._extract_required(fn),
        }

    @classmethod
    def _extract_properties(cls, func):
        return {
            name: {"type": cls.TYPE_MAP[type_hint]}
            for name, type_hint in get_type_hints(func).items()
        }

    @classmethod
    def _extract_required(cls, func):
        return [
            name
            for name, parameter in inspect.signature(func).parameters.items()
            if parameter.default == inspect.Parameter.empty
        ]

    @classmethod
    def describe(cls, fn: Callable[..., Any]) -> dict:
        """Returns a description of the function."""

        docstring = fn.__doc__
        docstring = docstring.strip() if docstring else fn.__name__

        return {
            "name": fn.__name__,
            "description": docstring,
            "parameters": cls._parameters(fn),
        }

    @classmethod
    def call(
        cls, functions: list[Callable[..., Any]], function_call: dict[str, str]
    ) -> Any:
        """Calls a function."""

        for function in functions:
            if function.__name__ == function_call["name"]:
                kwargs = ast.literal_eval(function_call["arguments"])

                return function(**kwargs)
