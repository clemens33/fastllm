"""Base classes and functions."""


from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Callable, Generator

import backoff
import openai
from jinja2 import Template
from openai.error import RateLimitError

from fastllm.utils import Functions

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Type of model."""

    CHAT = "chat"
    COMPLETION = "completion"


class Role(Enum):
    """Represents the available roles for messages in a chat model."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Style(Enum):
    """Represents the completion style. Corresponds to the temperature parameter."""

    PRECISE = 0.0
    BALANCED = 1.0
    CREATIVE = 2.0


@dataclass
class Message:
    """Represents a message when interacting with a chat model."""

    seed: InitVar[Message | dict | str]
    role: Role = Role.USER
    name: str | None = field(default=None)
    function_call: dict[str, str] | None = field(default=None)
    content: str = field(init=False)

    def __post_init__(self, seed: Message | dict | str):
        """Initializes the message content."""

        if isinstance(seed, str):
            self.content = seed
        elif isinstance(seed, Message):
            self.content = seed.content
            self.role = seed.role
        elif isinstance(seed, dict):
            _seed = self.from_response(seed)

            if len(_seed) > 1:
                logger.warning(
                    "More than one message in response. Using first message."
                )

            self.content = _seed[0].content
            self.role = _seed[0].role
            self.function_call = _seed[0].function_call
        else:
            raise TypeError(f"Cannot create Message from {type(seed)}.")

    @classmethod
    def from_response(cls, response: dict) -> list[Message]:
        """Returns a message or a list of messages from a openai response dict."""

        messages = []
        for choice in response["choices"]:
            if "message" in choice:
                message = choice["message"]

                content = "" if message["content"] is None else message["content"]
                role = Role(message["role"])

                if "function_call" in message:
                    function_call = {
                        "name": message["function_call"]["name"],
                        "arguments": message["function_call"]["arguments"],
                    }
                    messages.append(Message(content, role, function_call=function_call))
                else:
                    messages.append(Message(content, role))
            elif "delta" in choice:
                content = (
                    choice["delta"]["content"] if "content" in choice["delta"] else ""
                )
                role = (
                    Role(choice["delta"]["role"])
                    if "role" in choice["delta"]
                    else Role.ASSISTANT
                )

                messages.append(Message(content, role))

        if len(messages) == 0:
            raise ValueError("No messages in response.")

        return messages

    def __call__(self) -> str:
        """Returns the message content."""

        return self.content

    def __str__(self) -> str:
        """Returns the message as a string."""

        return f"{self.role.value}: {self.content}"

    def __add__(self, other: Message | str) -> Message:
        """Implements the + operator."""

        if isinstance(other, Message):
            if self.role == other.role:
                return Message(self.content + other.content, self.role)
            else:
                raise ValueError("Cannot add messages with different roles.")
        elif isinstance(other, str):
            return Message(self.content + other, self.role)
        else:
            raise TypeError(f"Cannot add Message and {type(other)}.")

    def to_dict(self) -> dict:
        """Returns the message as a dict."""

        message = {
            "role": self.role.value,
            "content": self.content if self.content else None,
        }

        if self.name:
            message["name"] = self.name

        if self.function_call:
            message["function_call"] = self.function_call

        return message


@dataclass
class Conversation:
    """Represents a ordered list of messages."""

    messages: list[Message]

    def __init__(self, *args: Conversation | Message | str):
        """Initializes the conversation."""

        self.messages = []

        if len(args) > 0:
            self._add(*args)

    def __add__(self, *other: Conversation | Message | str) -> Conversation:
        """Implements the + operator."""

        return Conversation(self, *other)

    def __getitem__(self, index: int) -> Message:
        """Returns the message at the given index."""

        return self.messages[index]

    def __str__(self) -> str:
        """Returns the conversation as a string."""

        return "\n".join([str(m) for m in self.messages])

    def __len__(self) -> int:
        """Returns the number of messages in the conversation."""

        return len(self.messages)

    def _add(self, *args: Conversation | Message | str):
        """Adds the given conversations, messages or strings to the conversation."""

        for arg in args:
            if isinstance(arg, Conversation):
                self.messages += arg.messages
            elif isinstance(arg, Message):
                self.messages += [arg]
            elif isinstance(arg, str):
                self.messages += [Message(arg)]
            else:
                raise TypeError(f"Cannot add {type(arg)} to Conversation.")

    def to_list(self) -> list[dict]:
        """Returns the messages as a list of dicts."""

        return [m.to_dict() for m in self.messages]


@dataclass
class Prompt:
    """Represents a prompt."""

    template: Template
    role: Role = Role.USER
    functions: list[Callable[..., Any]] | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        template: str,
        role: Role = Role.USER,
        functions: list[Callable[..., Any]] | None = None,
        **model_params,
    ):
        """Initializes the prompt."""

        self.template = Template(template)
        self.role = role
        self.functions = functions
        self.model_params = model_params

    def __call__(self, **kwargs) -> Message:
        """Returns the message from the prompt."""

        return Message(self.template.render(**kwargs), self.role)


@dataclass
class Model:
    """Represents a model."""

    seed: InitVar[Conversation | Message | str | None] = None
    name: str = "gpt-3.5-turbo"
    nr_tokens_max: int | None = None
    token_in_price: float | None = None
    token_out_price: float | None = None
    model_type: ModelType = ModelType.CHAT
    functions: list[Callable[..., Any]] | None = None
    stream_callback: Callable[[str], Any] | None = None
    conversation: Conversation = field(init=False)

    def __post_init__(self, seed: Conversation | Message | str | None):
        """Initializes the model. Optionally pass a seed as conversation, message or string."""

        if seed is None:
            self.conversation = Conversation()
        elif isinstance(seed, str):
            self.conversation = Conversation(Message(seed, Role.SYSTEM))
        elif isinstance(seed, Message):
            self.conversation = Conversation(seed)
        elif isinstance(seed, Conversation):
            self.conversation = seed
        else:
            raise TypeError(f"Cannot create Model from {type(seed)}.")

    def __call__(
        self,
        *args: Conversation | Message | str,
        functions: list[Callable[..., Any]] | None = None,
        stream_callback: Callable[[str], Any] | None = None,
        **kwargs,
    ) -> str:
        """Returns the response of the model."""

        self.update(*args)
        functions = functions or self.functions

        if functions:
            kwargs["functions"] = [
                Functions.describe(function) for function in functions
            ]

        stream_callback = stream_callback or self.stream_callback

        if stream_callback is not None:
            response = Message("", Role.ASSISTANT)

            for chunk in self.completion(stream=True, **kwargs):
                chunk = Message(chunk)
                stream_callback(chunk.content)

                response += chunk
        else:
            response = next(self.completion(**kwargs))

        message = Message(response)

        # Call function if message contains a function call
        if functions and message.function_call:
            self.update(message)

            function_output = Functions.call(functions, message.function_call)

            # Call self with function output - TODO check that we leave a potential infinite loop
            self(
                Message(
                    str(function_output), Role.FUNCTION, message.function_call["name"]
                )
            )
        else:
            self.update(Message(response))

        return self.conversation[-1]()

    @backoff.on_exception(backoff.expo, RateLimitError, max_time=60, logger=logger)
    def completion(self, stream: bool = False, **kwargs) -> Generator[dict, None, None]:
        """Calls the model, retries on RateLimitError."""

        if len(self.conversation) > 0:
            response = openai.ChatCompletion.create(
                model=self.name,
                messages=self.conversation.to_list(),
                stream=stream,
                **kwargs,
            )
        else:
            raise ValueError("Cannot create completion from empty conversation.")

        if stream:
            for chunk in response:
                yield dict(chunk)
        else:
            yield response  # type: ignore

    def update(self, *args: Conversation | Message | str):
        """Updates the models conversation."""

        self.conversation += Conversation(*args)

    def reset(self):
        """Resets the conversation."""

        self.conversation = Conversation()


@dataclass
class Agent:
    """Represents an agent working with a model and a playbook."""

    playbook: list[Conversation | Message | Prompt | Agent | str]
    model: Model

    def __init__(
        self,
        *args: Conversation | Message | Prompt | Agent | str,
        model: Model = Model(),
    ):
        """Initializes the playbook."""

        self.playbook = []
        for arg in args:
            if isinstance(arg, str):
                self.playbook.append(Prompt(arg))
            elif isinstance(arg, Conversation | Message | Prompt | Agent):
                self.playbook.append(arg)
            else:
                raise TypeError(f"Cannot add {type(arg)} to Agent.")

        self.model = model

    def __call__(self, **inputs: Any) -> str:
        """Returns the final response of the agent."""

        final_response = ""
        for response in self.run(**inputs):
            final_response = response

        return final_response

    def run(self, **inputs: Any) -> Generator[str, None, None]:
        """Runs through the playbook and yields intermediate responses."""

        steps = []
        for step in self.playbook:
            if isinstance(step, Prompt):
                prompt = step

                yield self.model(
                    *steps,
                    prompt(**inputs),
                    functions=prompt.functions,
                    **prompt.model_params,
                )

                steps = []
            elif isinstance(step, Agent):
                agent = step

                yield self.model(*steps, agent(**inputs))
            else:
                steps.append(step)


class Models:
    """Represents the available openai models."""

    available_models = {
        "gpt-4": {
            "nr_tokens_max": 8192,
            "token_in_price": 0.03,
            "token_out_price": 0.06,
            "model_type": ModelType.CHAT,
            "functions": False,
        },
        "gpt-4-0613": {
            "nr_tokens_max": 8192,
            "token_in_price": 0.03,
            "token_out_price": 0.06,
            "model_type": ModelType.CHAT,
            "functions": True,
        },
        "gpt-3.5-turbo": {
            "nr_tokens_max": 4096,
            "token_in_price": 0.0015,
            "token_out_price": 0.002,
            "model_type": ModelType.CHAT,
            "functions": False,
        },
        "gpt-3.5-turbo-0613": {
            "nr_tokens_max": 4096,
            "token_in_price": 0.0015,
            "token_out_price": 0.002,
            "model_type": ModelType.CHAT,
            "functions": True,
        },
        "gpt-3.5-turbo-16k": {
            "nr_tokens_max": 16384,
            "token_in_price": 0.003,
            "token_out_price": 0.004,
            "model_type": ModelType.CHAT,
            "functions": False,
        },
        "gpt-3.5-turbo-16k-0613": {
            "nr_tokens_max": 16384,
            "token_in_price": 0.003,
            "token_out_price": 0.004,
            "model_type": ModelType.CHAT,
            "functions": True,
        },
    }

    @classmethod
    def create(
        cls,
        seed: Conversation | Message | str | None = None,
        name: str = "gpt-3.5-turbo",
    ) -> Model:
        """Returns a new instance of the model with the given name."""

        try:
            model_info = cls.available_models[name]
            return Model(seed, name=name, **model_info)
        except KeyError:
            raise ValueError(f"Model {name} not found.")
