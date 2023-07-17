"""Tests for the Message class."""

import logging
from math import log

import pytest

from fastllm.base import Agent, Conversation, Message, Model, Prompt, Role

logger = logging.getLogger(__name__)


class TestMessage:
    """Tests for Message class."""

    def test_message_from_str(self):
        """Tests initialization with a string."""

        message = Message("Hello world")
        assert message.role == Role.USER
        assert message.content == "Hello world"

    def test_message_from_message(self):
        """Tests initialization with a Message instance."""

        original_message = Message("Hello world")
        new_message = Message(original_message)
        assert new_message.role == Role.USER
        assert new_message.content == "Hello world"

    def test_message_from_dict_response(self):
        """Tests initialization with a dictionary."""

        message_dict = {
            "choices": [{"message": {"content": "Hello world", "role": "assistant"}}]
        }
        message = Message(message_dict)
        assert message.role == Role.ASSISTANT
        assert message.content == "Hello world"

    def test_message_from_dict_delta_response(self):
        """Tests initialization with a dictionary."""

        pass

    def test_message_from_dict_function_response(self):
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
        assert message.function_call["name"] == "add_numbers"

    def test_message_str(self):
        """Tests the string representation of a Message."""

        message = Message("Hello world")
        assert str(message) == "user: Hello world"

    def test_message_add_str(self):
        """Tests concatenation of a Message and a string."""

        message = Message("Hello")
        message += " world"
        assert message.content == "Hello world"

    def test_message_add_message(self):
        """Tests concatenation of two Messages."""

        message1 = Message("Hello")
        message2 = Message(" world")
        message1 += message2
        assert message1.content == "Hello world"

    def test_message_to_dict(self):
        """Tests conversationersion of a Message to a dictionary."""

        message = Message("Hello world")
        assert message.to_dict() == {"role": "user", "content": "Hello world"}


class TestConversation:
    """Test suite for the Conversation class."""

    def test_conversation_init_with_messages(self):
        """Tests initialization with Message instances."""

        message1 = Message("Hello")
        message2 = Message(" world")
        conversation = Conversation(message1, message2, "!")
        assert len(conversation) == 3
        assert conversation[0].content == "Hello"
        assert conversation[1].content == " world"
        assert conversation[2].content == "!"

    def test_conversation_init_with_conversation(self):
        """Tests initialization with a Conversation instance."""

        message1 = Message("Hello")
        message2 = Message(" world")
        conversation1 = Conversation(message1, message2)
        conversation2 = Conversation(conversation1, "!")
        assert len(conversation2) == 3
        assert conversation2[2].content == "!"

    def test_conversation_add(self):
        """Tests concatenation of Conversations."""

        message1 = Message("Hello")
        message2 = Message(" world")
        conversation1 = Conversation(message1)
        conversation2 = Conversation(message2)
        conversation = conversation1 + conversation2
        assert len(conversation) == 2
        assert conversation[0].content == "Hello"
        assert conversation[1].content == " world"

    def test_conversation_str(self):
        """Tests string representation of a Conversation."""
        message1 = Message("Hello")
        message2 = Message(" world")
        conversation = Conversation(message1, message2)
        assert str(conversation) == "user: Hello\nuser:  world"

    def test_conversation_to_list(self):
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

    def test_prompt_init(self):
        """Tests initialization of a Prompt."""
        template_str = "Hello, {{ name }}!"
        prompt = Prompt(template_str)

        assert prompt.role == Role.USER
        assert prompt.model_params == {}

    def test_prompt_call(self):
        """Tests generating a Message from a Prompt."""

        template_str = "Hello, {{ name }}!"
        prompt = Prompt(template_str)
        message = prompt(name="world")
        assert isinstance(message, Message)
        assert message.role == Role.USER
        assert message.content == "Hello, world!"

    def test_prompt_template_no_values_given(self):
        """Tests a prompt template."""

        prompt = Prompt("Find {{ nr_names }} short names")

        message = prompt()

        assert message.content == "Find  short names"

    def test_prompt_no_template(self):
        """Tests a prompt."""

        prompt = Prompt("Find short names")

        assert prompt() == Message("Find short names")


class TestModel:
    """Tests Model class."""

    def test_model_seed_from_string(self):
        """Tests creating a model from a conversation."""

        model = Model("You are a friendly assistant.")

        assert model.conversation[0] == Message(
            "You are a friendly assistant.", Role.SYSTEM
        )

    def test_model_seed_from_message(self):
        """Tests creating a model from a conversation."""

        model = Model(Message("You are a friendly assistant.", Role.USER))

        assert model.conversation[0] == Message(
            "You are a friendly assistant.", Role.USER
        )

    def test_model_seed_from_conversation(self):
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

    def test_model_create_no_seed(self):
        """Tests creating a model without a seed."""

        model = Model(name="gpt-4")

        assert model.conversation == Conversation()

    @pytest.mark.openai
    @pytest.mark.parametrize("model_name", ["gpt-4", "gpt-3.5-turbo"])
    def test_model_string(self, model_name):
        """Tests creating a model with a single message."""

        response = Model(name=model_name)("Say Hello!")

        logger.info(response)

        assert "hello" in response.lower()

    @pytest.mark.openai
    @pytest.mark.parametrize("model_name", ["gpt-3.5-turbo"])
    def test_model_conversation(self, model_name):
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
    def test_model_stream(self):
        """Tests creating a model with a conversation."""

        model = Model(name="gpt-4")

        response = ""

        def _callback(chunk):
            nonlocal response
            response = response + chunk

        full_response = model("Say Hello! Say it 10 times!", stream_callback=_callback)

        assert full_response == response


class TestAgent:
    """Tests Agent class."""

    @pytest.mark.openai
    def test_simple_agent_from_str(self):
        """Tests a simple agent."""

        find_names = Agent(
            "Find {{ n }} short names. Return them comma separated. Nothing else!"
        )

        names = find_names(n=3).split(",")

        logger.info(names)

        assert len(names) == 3

    @pytest.mark.openai
    def test_simple_agent(self):
        """Tests a simple agent."""

        find_names = Agent(
            Message("You are a name finder.", Role.SYSTEM),
            Prompt("Find a short name.", max_tokens=6),
            Prompt("Find yet another name.", max_tokens=6),
            "This is random system message. No problem for you model no? I try to confuse you.",
            Prompt("Find and a third name.", max_tokens=6),
            Prompt("List the found names comma separated. Ignore everything else."),
            model=Model(stream_callback=lambda chunk: logger.info(chunk)),
        )

        names = find_names()

        logger.info(names)

        assert len(names.split(",")) == 3
        assert len(find_names.model.conversation) == 10

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
    def test_agent_prompt_template(self, nr_names, seperator):
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
    def test_agent_function_call(self):
        """Tests an agent with a prompt template."""

        def add_numbers(a: int, b: int):
            """Adds two integer numbers."""

            return a + b

        calculator = Agent(
            Message(
                "You are a calculater who has access to various functions. Use them!. ",
                Role.SYSTEM,
            ),
            Prompt(
                "Calculate the result for task: {{ task }}", functions=[add_numbers]
            ),
            Prompt("Only give the result number as result without anything else!"),
            model=Model("gpt-3.5-turbo-0613"),
        )

        result = calculator(task="add the numbers 11111 and 22222")

        assert result == "33333"

    @pytest.mark.openai
    def test_agent_multiple_function_calls(self):
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
