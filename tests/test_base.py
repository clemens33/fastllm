"""Tests for the Message class."""

import logging
import re
from typing import Any, Callable, List, Literal

import pytest
from jsonschema import ValidationError

from fastllm.base import (
    Agent,
    Conversation,
    Function,
    FunctionCall,
    Functions,
    Message,
    Model,
    Prompt,
    Role,
)

logger = logging.getLogger(__name__)


class TestFunction:
    """Tests the Function class."""

    def test_basic(self):
        """Tests basic functionality."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        function = Function(add_numbers)
        assert function.name == "add_numbers"
        assert function.description == "Adds two integer numbers."
        assert function.parameters == {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        assert function(a=3, b=4) == 7

    def test_basic_string(self):
        """Tests basic functionality."""

        def revert_string(value: str) -> str:
            """Reverts a string."""

            return value[::-1]

        function = Function(revert_string)

        assert function(value="Hello") == "olleH"

    def test_basic_from_function(self):
        """Tests basic functionality."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        _function = Function(add_numbers)
        function = Function(_function)

        assert function.name == "add_numbers"
        assert function.description == "Adds two integer numbers."
        assert function.parameters == {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        assert function(a=3, b=4) == 7

    def test_basic_no_hints(self):
        """Tests basic functionality without type hints."""

        def add_numbers(a, b):
            """Adds two integer numbers."""

            return a + b

        function = Function(add_numbers)
        assert function.name == "add_numbers"
        assert function.description == "Adds two integer numbers."
        assert function.parameters == {
            "type": "object",
            "properties": {"a": {}, "b": {}},
            "required": ["a", "b"],
        }
        assert function(a=3, b=4) == 7

    def test_literal_type_hints(self):
        """Tests functions with literal type hints."""

        def calculator(
            a: int,
            b: int,
            op: Literal[
                "+",
                "-",
            ],
        ):
            """Calculator function."""

            match op:
                case "+":
                    return a + b
                case "-":
                    return a - b
                case _:
                    ValueError(f"Operator {op} not supported.")

        function = Function(calculator)
        assert function.parameters == {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
                "op": {"type": "string", "enum": ["+", "-"]},
            },
            "required": ["a", "b", "op"],
        }
        assert function(a=3, b=4, op="+") == 7

    def test_list_type_hints(self):
        """Tests functions with list type hints."""

        def add_numbers(a: list[int], b: List[float]):
            """Adds all numbers in a list."""

            return sum(a) + sum(b)

        function = Function(add_numbers)
        assert function.parameters == {
            "type": "object",
            "properties": {
                "a": {"type": "array", "items": {"type": "integer"}},
                "b": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["a", "b"],
        }
        assert function(a=[3, 4], b=[5.0, 6.0]) == 18

    def test_custom_init(self):
        """Tests custom init."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        function = Function(
            add_numbers,
            "my name",
            "my description",
        )
        assert function.name == "my name"
        assert function.description == "my description"
        assert function(a=3, b=4) == 7

    def test_schema(self):
        """Tests that a Function can be converted to a schema/dictionary."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        function = Function(add_numbers)
        assert function.schema == {
            "name": "add_numbers",
            "description": "Adds two integer numbers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        }

    def test_unsupported_type_hint(self):
        """Tests that a Function raises an error.

        When a function with unsupported type hints is used.
        """

        def unsupported_type_hint(a: tuple[int]):
            return sum(a)

        with pytest.raises(TypeError):
            Function(unsupported_type_hint)

    def test_call(self):
        """Test the function call."""

        def add_numbers(a, b):
            return a + b

        function = Function(add_numbers)

        assert function.call('{\n  "a": 1234,\n  "b": 9876\n}') == 11110

    def test_call_json_parse_error(self):
        """Test the function call expecting a json parse error."""

        def add_numbers(a, b):
            return a + b

        function = Function(add_numbers)

        with pytest.raises(ValueError):
            function.call('{\n  "a": 1234,\n  "b": 9876\n')

    def test_call_json_schema_error(self):
        """Test the function call expecting a json parse error."""

        def add_numbers(a, b):
            return a + b

        function = Function(add_numbers)

        with pytest.raises(ValidationError):
            function.call('{\n  "a": 1234,\n  "c": 9876\n}')


class TestFunctions:
    """Tests the functions mixin."""

    def test_mixin(self):
        """Tests mixin."""

        class Agent(Functions):
            """Test class."""

            def __init__(self, functions):
                """Initializes the class."""

                super().__init__(functions=functions)

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        agent = Agent(functions=[add_numbers])

        assert len(agent._functions) == 1
        assert agent.function_call("add_numbers", a=1, b=2) == 3

    def test_mixin_basic_decorator(self):
        """Tests mixin with basic decorator."""

        class Agent(Functions):
            """Test class."""

            def __init__(self, functions: list[Callable[..., Any]] | None = None):
                """Initializes the class."""

                super().__init__(functions=functions)

        agent = Agent()

        @agent.function
        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        result1 = add_numbers(a=1, b=2)
        result2 = add_numbers(1, 2)
        result3 = agent.function_call("add_numbers", a=1, b=2)

        assert len(agent._functions) == 1
        assert result1 == result2 == result3 == 3

    def test_mixin_decorator_custom(self):
        """Tests basic mixin."""

        class Agent(Functions):
            """Test class."""

            def __init__(self, functions: list[Callable[..., Any]] | None = None):
                """Initializes the class."""

                super().__init__(functions=functions)

        agent = Agent()

        @agent.function
        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        @agent.function(name="my_subtraction")
        def subtract_numbers(a: int, b: int):
            """Subtracts two integer numbers."""

            return a - b

        @agent.function(name="my_multiplication")
        def multiply_numbers(a: int, b: int):
            """Multiplies two integer numbers."""

            return a * b

        assert len(agent._functions) == 3
        assert agent.function_call("my_subtraction", a=5, b=2) == 3
        assert agent.function_call("my_multiplication", a=1, b=0) == 0


class TestFunctionCall:
    """Tests the FunctionCall method."""

    def test_add(self):
        """Tests basic functionality."""

        fc1 = FunctionCall("add_numbers", arguments="{\n  ")
        fc2 = FunctionCall("subtract_numbers", arguments='"a": 1234,\n  "b": 9876\n}')

        fc = fc1 + fc2
        assert fc.name == "add_numbers"
        assert fc.arguments == '{\n  "a": 1234,\n  "b": 9876\n}'

    def test_add_fails(self):
        """Tests basic functionality."""

        fc1 = FunctionCall("add_numbers", arguments="{\n  ")

        with pytest.raises(TypeError):
            fc1 + "abc"  # type: ignore


class TestMessage:
    """Tests for Message class."""

    def test_from_str(self):
        """Tests initialization with a string."""

        message = Message("Hello world")
        assert message.role == Role.USER
        assert message.content == "Hello world"

    def test_from_message(self):
        """Tests initialization with a Message instance."""

        original_message = Message("Hello world")
        new_message = Message(original_message)
        assert new_message.role == Role.USER
        assert new_message.content == "Hello world"

    def test_from_dict_response(self):
        """Tests initialization with a dictionary."""

        message_dict = {
            "choices": [{"message": {"content": "Hello world", "role": "assistant"}}]
        }
        message = Message(message_dict)
        assert message.role == Role.ASSISTANT
        assert message.content == "Hello world"

    def test_from_dict_delta_response(self):
        """Tests initialization with a dictionary."""

        pass

    def test_from_dict_function_response(self):
        """Tests initialization with a dictionary."""

        message_dict = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "add_numbers",
                            "arguments": '{\n  "a": 1234,\n  "b": 9876\n}',
                        },
                    },
                    "finish_reason": "function_call",
                }
            ],
            "usage": {
                "prompt_tokens": 96,
                "completion_tokens": 24,
                "total_tokens": 120,
            },
        }
        message = Message(message_dict)
        assert message.role == Role.ASSISTANT
        assert message.content == ""
        assert message.function_call is not None
        assert message.function_call.name == "add_numbers"

    def test_str(self):
        """Tests the string representation of a Message."""

        message = Message("Hello world")
        assert str(message) == "user: Hello world"

    def test_add_str(self):
        """Tests concatenation of a Message and a string."""

        message = Message("Hello")
        message += " world"
        assert message.content == "Hello world"

    def test_add_message(self):
        """Tests concatenation of two Messages."""

        message1 = Message("Hello")
        message2 = Message(" world")
        message1 += message2
        assert message1.content == "Hello world"

    def test_to_dict(self):
        """Tests conversationersion of a Message to a dictionary."""

        message = Message("Hello world")
        assert message.to_dict() == {"role": "user", "content": "Hello world"}


class TestConversation:
    """Test suite for the Conversation class."""

    def test_init_with_messages(self):
        """Tests initialization with Message instances."""

        message1 = Message("Hello")
        message2 = Message(" world")
        conversation = Conversation(message1, message2, "!")
        assert len(conversation) == 3
        assert conversation[0].content == "Hello"
        assert conversation[1].content == " world"
        assert conversation[2].content == "!"

    def test_init_with_conversation(self):
        """Tests initialization with a Conversation instance."""

        message1 = Message("Hello")
        message2 = Message(" world")
        conversation1 = Conversation(message1, message2)
        conversation2 = Conversation(conversation1, "!")
        assert len(conversation2) == 3
        assert conversation2[2].content == "!"

    def test_add(self):
        """Tests concatenation of Conversations."""

        message1 = Message("Hello")
        message2 = Message(" world")
        conversation1 = Conversation(message1)
        conversation2 = Conversation(message2)
        conversation = conversation1 + conversation2
        assert len(conversation) == 2
        assert conversation[0].content == "Hello"
        assert conversation[1].content == " world"

    def test_str(self):
        """Tests string representation of a Conversation."""
        message1 = Message("Hello")
        message2 = Message(" world")
        conversation = Conversation(message1, message2)
        assert str(conversation) == "user: Hello\nuser:  world"

    def test_to_list(self):
        """Tests conversationersion of a Conversation to a list of dictionaries."""
        message1 = Message("Hello")
        message2 = Message(" world")
        conversation = Conversation(message1, message2)
        assert conversation.to_list() == [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": " world"},
        ]


class TestPrompt:
    """Test suite for the Prompt class."""

    def test_init(self):
        """Tests initialization of a Prompt."""

        template_str = "Hello, {{ name }}!"
        prompt = Prompt(template_str)

        assert prompt.role == Role.USER
        assert prompt.model_params == {}

    def test_call(self):
        """Tests generating a Message from a Prompt."""

        template_str = "Hello, {{ name }}!"
        prompt = Prompt(template_str)
        message = prompt(name="world")
        assert isinstance(message, Message)
        assert message.role == Role.USER
        assert message.content == "Hello, world!"

    def test_template_no_values_given(self):
        """Tests a prompt template."""

        prompt = Prompt("Find {{ nr_names }} short names")

        message = prompt()

        assert message.content == "Find  short names"

    def test_no_template(self):
        """Tests a prompt."""

        prompt = Prompt("Find short names")

        assert prompt() == Message("Find short names")

    def test_functions(self):
        """Tests a prompts with a function."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        prompt = Prompt(
            "Calculate the result for task: {{ task }}",
            functions=[add_numbers],
        )

        @prompt.function
        def subtract_numbers(a: int, b: int):
            """Subtracts two integer numbers."""

            return a - b

        assert prompt.function_call("add_numbers", a=1, b=2) == 3
        assert prompt.function_call("subtract_numbers", a=3, b=2) == 1


class TestModel:
    """Tests Model class."""

    def test_seed_from_string(self):
        """Tests creating a model from a conversation."""

        model = Model("You are a friendly assistant.")

        assert model.conversation[0] == Message(
            "You are a friendly assistant.", Role.SYSTEM
        )

    def test_seed_from_message(self):
        """Tests creating a model from a conversation."""

        model = Model(Message("You are a friendly assistant.", Role.USER))

        assert model.conversation[0] == Message(
            "You are a friendly assistant.", Role.USER
        )

    def test_seed_from_conversation(self):
        """Tests creating a model from a conversation."""

        model = Model(
            Conversation(
                Message("You are a friendly assistant.", Role.SYSTEM),
                Message("I am a friendly user."),
                Message("I am a friendly assistant.", Role.ASSISTANT),
            )
        )

        assert model.conversation == Conversation(
            Message("You are a friendly assistant.", Role.SYSTEM),
            Message("I am a friendly user."),
            Message("I am a friendly assistant.", Role.ASSISTANT),
        )

    def test_create_no_seed(self):
        """Tests creating a model without a seed."""

        model = Model(name="gpt-4")

        assert model.conversation == Conversation()

    @pytest.mark.openai
    @pytest.mark.parametrize("model_name", ["gpt-4", "gpt-3.5-turbo"])
    def test_string(self, model_name):
        """Tests creating a model with a single message."""

        response = Model(name=model_name)("Say Hello!")

        logger.info(response)

        assert "hello" in response.lower()

    @pytest.mark.openai
    @pytest.mark.parametrize("model_name", ["gpt-3.5-turbo"])
    def test_conversation(self, model_name):
        """Tests creating a model with a conversation."""

        response = Model(name=model_name)(
            Conversation(
                Message("You are a friendly assistant.", Role.SYSTEM),
                Message("Say Hello!"),
            )
        )

        logger.info(response)

        assert "hello" in response.lower()

    @pytest.mark.openai
    def test_stream(self):
        """Tests creating a model with a conversation."""

        model = Model(name="gpt-4")

        response = ""

        def _callback(chunk):
            nonlocal response
            response = response + chunk

        full_response = model("Say Hello! Say it 10 times!", stream_callback=_callback)

        assert full_response == response

    def test_functions(self):
        """Test function mixin in model."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        model = Model(functions=[add_numbers])

        @model.function
        def subtract_numbers(a: int, b: int):
            """Subtracts two integer numbers."""

            return a - b

        assert model.function_call("add_numbers", a=1, b=2) == 3
        assert model.function_call("subtract_numbers", a=3, b=2) == 1

    def test_functions_callback(self):
        """Test function mixin in model."""

        def stream_callback(chunk):
            logger.info(chunk)

        model = Model(stream_callback=stream_callback)

        @model.function
        def subtract_numbers(a: int, b: int):
            """Subtracts two integer numbers."""

            return a - b

        result = model("subtract 37 from 42")

        assert "5" in result


class TestAgent:
    """Tests Agent class."""

    @pytest.mark.openai
    def test_simple_single_string(self):
        """Tests a simple agent."""

        find_names = Agent(
            "Find {{ n }} short names. Return them comma separated. Nothing else!"
        )

        names = find_names(n=3).split(",")

        logger.info(names)

        assert len(names) == 3

    @pytest.mark.openai
    def test_simple_multiple_messages(self):
        """Tests a simple agent."""

        find_names = Agent(
            Message("You are a name finder.", Role.SYSTEM),
            Prompt("Find a short name.", max_tokens=6),
            Prompt("Find yet another name.", max_tokens=6),
            "This is random system message. No problem for you model no? :)",
            Prompt("Find and a third name.", max_tokens=6),
            Prompt("List the found names comma separated. Ignore everything else."),
            model=Model(stream_callback=lambda chunk: logger.info(chunk)),
        )

        names = find_names()

        logger.info(names)

        assert len(names.split(",")) == 3

    @pytest.mark.openai
    def test_nested_agent(self):
        """Tests a nested agent."""

        story_writer = Agent(
            Message("Given the following names", Role.SYSTEM),
            Agent(
                Message("You are a name finder.", Role.SYSTEM),
                Prompt("Find a short name.", max_tokens=6),
                Prompt("Find yet another name.", max_tokens=6),
                Prompt("List the found names comma separated. Ignore everything else."),
                model=Model(),
            ),
            Prompt("Write a story with the names.", max_tokens=100),
            model=Model(),
        )

        story = story_writer()

        logger.info(story)

        assert len(story) > 0

    @pytest.mark.openai
    @pytest.mark.parametrize("nr_names, seperator", [(3, ","), (5, ";")])
    def test_prompt_template(self, nr_names, seperator):
        """Tests an agent with a prompt template."""

        find_names = Agent(
            Message("You are an expert name giver!", Role.SYSTEM),
            Prompt("Find {{ nr_names }} short names", temperature=2.0, max_tokens=20),
            Prompt("Print them {{ seperator }} separated", max_tokens=20),
            model=Model(),
        )

        names = find_names(nr_names=nr_names, seperator=seperator)

        logger.info(names)

        assert len(names.split(seperator)) == nr_names

    @pytest.mark.openai
    @pytest.mark.parametrize(
        "task, expected_result",
        [
            ("add 1 and 2", "3"),
            ("add 1 and 2 and 3", "6"),
        ],
    )
    def test_function_call(self, task, expected_result):
        """Tests an agent with function calling."""

        calculator_agent = Agent(
            Message(
                "You are an expert calculater. Use provided functions!",
                Role.SYSTEM,
            ),
            Prompt("Calculate the result for task: {{ task }}"),
            Prompt("Print the result number, nothing else!"),
            model=Model("gpt-3.5-turbo-0613"),
            # model=Model("gpt-4-0613"),
        )

        @calculator_agent.function
        def calculater(a, b, operator: Literal["+", "-", "*", "/"]):
            """Calculator helper function."""

            match operator:
                case "+":
                    return a + b
                case "-":
                    return a - b
                case "*":
                    return a * b
                case "/":
                    return a / b
                case _:
                    ValueError(f"Operator {operator} not supported.")

        result = calculator_agent(task=task)

        assert result == expected_result

    @pytest.mark.openai
    def test_multiple_function_calls(self):
        """Tests an agent with a prompt template."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        def subtract_numbers(a: int, b: int):
            """Subtracts the integer b from integer a."""

            return a - b

        def multiply_numbers(a: int, b: int):
            """Multiplies two integer numbers."""

            return a * b

        calculator = Agent(
            Message(
                "You are a calculater who has access to various functions. Use them!. ",
                Role.SYSTEM,
            ),
            Prompt("Calculate the result for task: {{ task }}"),
            Prompt("Only give the result number as result without anything else!"),
            model=Model(
                "gpt-3.5-turbo-0613",
                functions=[add_numbers, subtract_numbers, multiply_numbers],
            ),
        )

        result = calculator(task="give the final result for (11 + 14) * (6 - 2)")

        assert result == "100"

    @pytest.mark.openai
    def test_multiple_function_calls_mixed(self):
        """Tests an agent with a prompt template."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        def subtract_numbers(a: int, b: int):
            """Subtracts the integer b from integer a."""

            return a - b

        calculator = Agent(
            Message(
                "You are a calculater who has access to various functions. Use them!. ",
                Role.SYSTEM,
            ),
            Prompt(
                "Calculate the result for task: {{ task }}",
                functions=[subtract_numbers],
            ),
            Prompt("Only give the result number as result without anything else!"),
            model=Model(
                "gpt-3.5-turbo-0613",
                functions=[add_numbers],
            ),
        )

        @calculator.function
        def multiply_numbers(a: int, b: int):
            """Multiplies two integer numbers."""

            return a * b

        result = calculator(task="give the final result for (11 + 14) * (6 - 2)")

        assert result == "100"

    def test_prompt_prefer(self):
        """Tests usage of pattern prompt."""

        find_number = Agent(
            Prompt("Give me a random number between 10 and 99", prefer=r"[0-9]{2}"),
        )

        number = find_number()

        assert re.match(r"[0-9]{2}", number)

    def test_prompt_avoid(self):
        """Tests usage of pattern prompt."""

        say_hello = Agent(
            Prompt(
                "Say Hello!",
                avoid=[r"([ ]?[Hh](e(l(l(o?)?)?)?)?)?"],
            ),
        )

        not_hello = say_hello()

        assert "Hello" not in not_hello
