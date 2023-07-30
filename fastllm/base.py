"""fastllm base classes and functions."""


from __future__ import annotations

import inspect
import json
import logging
from dataclasses import InitVar, dataclass, field
from enum import Enum
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import backoff
import openai
from jinja2 import Template
from jsonschema import ValidationError, validate
from openai.error import RateLimitError, ServiceUnavailableError

from fastllm.utils import get_logit_bias

logger = logging.getLogger(__name__)


P = ParamSpec("P")
R = TypeVar("R")


class Role(Enum):
    """Represents the available roles for messages in a chat model."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Function:
    """Represents a function that can be called from a model."""

    TYPE_MAP = {
        int: "integer",
        str: "string",
        bool: "boolean",
        float: "number",
    }

    function: Callable[..., Any] = field(init=False)
    name: str = field(init=False)
    description: str = field(init=False)
    parameters: dict = field(init=False)

    def __init__(
        self,
        function: Callable[..., Any] | Function,
        name: str | None = None,
        description: str | None = None,
    ):
        """Initializes the function."""

        if isinstance(function, Function):
            self.function = function.function
            self.name = function.name
            self.description = function.description
            self.parameters = function.parameters
        else:
            self.function = function
            self.name = name or function.__name__
            self.description = description or function.__doc__ or function.__name__
            self.parameters = self._parameters(function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the function."""

        return self.function(*args, **kwargs)

    def call(self, arguments_json_string: str) -> Any:
        """Calls function from arguments json string.

        Is validated against its parameter schema.

        """

        arguments = None

        try:
            arguments = json.loads(arguments_json_string)

            validate(instance=arguments, schema=self.parameters)
        except json.JSONDecodeError as e:
            logger.debug(f"Could not decode arguments {arguments_json_string}: {e}")

            raise e
        except ValidationError as e:
            logger.debug(
                f"Arguments {arguments} do not match schema {self.parameters}: {e}"
            )

            raise e

        return self(**arguments)

    @property
    def schema(self) -> dict:
        """Returns a schema description of the function."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def _properties(cls, function: Callable[..., Any]) -> dict:
        """Returns the parameters and types of a function."""

        params = inspect.signature(function).parameters
        type_hints = get_type_hints(function)

        properties = {param: {} for param in params}
        for name, type_hint in type_hints.items():
            if name == "return":
                continue

            array = False
            literals = None

            if get_origin(type_hint) is Literal:
                literals = get_args(type_hint)
                literal_types = {type(literal) for literal in literals}

                if len(literal_types) != 1:
                    raise TypeError(
                        f"Literal type hints must be of the same type. \
Got {literal_types}."
                    )

                type_hint = literal_types.pop()
            elif get_origin(type_hint) is list:
                list_types = get_args(type_hint)

                if len(list_types) != 1:
                    raise TypeError("Optional type hints not supported in list.")

                type_hint = list_types[0]
                array = True

            if type_hint and type_hint not in cls.TYPE_MAP:
                raise TypeError(
                    f"Type {type_hint} of argument {name} not supported. "
                    f"Supported type hints are {list(cls.TYPE_MAP.keys())}."
                )

            if array:
                properties[name]["type"] = "array"
                properties[name]["items"] = {"type": cls.TYPE_MAP[type_hint]}
            else:
                properties[name]["type"] = cls.TYPE_MAP[type_hint]

                if literals:
                    properties[name]["enum"] = list(literals)

        return properties

    @classmethod
    def _parameters(cls, fn: Callable[..., Any]) -> dict:
        """Returns a description of the functions parameters."""

        return {
            "type": "object",
            "properties": cls._properties(fn),
            "required": cls._required(fn),
        }

    @classmethod
    def _required(cls, func):
        """Returns the required parameters of a function."""

        return [
            name
            for name, parameter in inspect.signature(func).parameters.items()
            if parameter.default == inspect.Parameter.empty
        ]


@dataclass(kw_only=True)
class Functions:
    """Mixin for registering and handling functions."""

    functions: InitVar[list[Callable[..., Any]] | None] = None
    _functions: list[Function] = field(init=False, default_factory=list)

    def __post_init__(
        self,
        functions: list[Callable[..., Any]] | None = None,
    ):
        """Initializes the functions."""

        self._functions = [Function(fn) for fn in functions or []]

    @overload
    def function(self, fn: Callable[P, R]) -> Callable[P, R]:
        ...

    @overload
    def function(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        ...

    def function(
        self,
        fn: Callable[P, R] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]] | Callable[P, R]:
        """Decorator for registering a function."""

        def wrapper(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
            return fn(*args, **kwargs)

        if fn is not None:
            if not callable(fn):
                raise TypeError("Only supports optional keyword arguments.")

            self._functions.append(Function(fn, name, description))

            return wraps(fn)(partial(wrapper, fn))

        def decorator(fn: Callable[P, R]) -> Callable[P, R]:
            self._functions.append(Function(fn, name, description))

            return wraps(fn)(partial(wrapper, fn))

        return decorator

    def function_call(
        self,
        name: str,
        arguments_json_string: str | None = None,
        functions: list[Function] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Calls the function with the given name.

        If functions are passed, they are used instead of the instance functions.
        """

        functions = functions or self._functions

        for function in functions:
            if function.name == name:
                if arguments_json_string:
                    return function.call(arguments_json_string)
                else:
                    return function(**kwargs)

        raise ValueError(f"Function {name} not available. ")


@dataclass
class FunctionCall:
    """Represents a function call in a message."""

    name: str
    arguments: str

    def __add__(self, other: FunctionCall):
        """Implements the + operator."""

        if isinstance(other, FunctionCall):
            return FunctionCall(
                self.name,
                self.arguments + other.arguments,
            )
        else:
            raise TypeError(f"Cannot add FunctionCall and {type(other)}.")


@dataclass
class Message:
    """Represents a message when interacting with a chat model."""

    seed: InitVar[Message | dict | str]
    role: Role = Role.USER
    name: str | None = None
    function_call: FunctionCall | None = None
    content: str = field(init=False, default="")

    def __post_init__(self, seed: Message | dict | str):
        """Initializes the message content."""

        if isinstance(seed, str):
            self.content = seed
        elif isinstance(seed, Message):
            self.content = seed.content
            self.role = seed.role
            self.function_call = seed.function_call
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
                data = choice["message"]

                content = "" if data["content"] is None else data["content"]
                role = Role(data["role"])

            elif "delta" in choice:
                data = choice["delta"]

                content = data["content"] if "content" in data else ""
                content = content or ""

                role = Role(data["role"]) if "role" in data else Role.ASSISTANT
            else:
                raise ValueError("Cannot parse response.")

            if "function_call" in data:
                function_call = data["function_call"]

                name = function_call["name"] if "name" in function_call else ""
                function_call = FunctionCall(name, function_call["arguments"])

                messages.append(Message(content, role, function_call=function_call))
            else:
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
                function_call = None
                if self.function_call and other.function_call:
                    function_call = self.function_call + other.function_call
                elif self.function_call:
                    function_call = self.function_call
                elif other.function_call:
                    function_call = other.function_call

                return Message(
                    self.content + other.content,
                    self.role,
                    function_call=function_call,
                )
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
            message["function_call"] = {
                "name": self.function_call.name,
                "arguments": self.function_call.arguments,
            }

        return message


@dataclass
class Conversation:
    """Represents a ordered list of messages."""

    messages: list[Message] = field(init=False, default_factory=list)

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
class Prompt(Functions):
    """Represents a prompt."""

    template: Template = field(init=False)
    role: Role = field(init=False, default=Role.USER)
    prefer: list[str] | str | None = None
    avoid: list[str] | str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        template: str,
        role: Role = Role.USER,
        prefer: list[str] | str | None = None,
        avoid: list[str] | str | None = None,
        functions: list[Callable[..., Any]] | None = None,
        **model_params,
    ):
        """Initializes the prompt."""

        super().__init__(functions=functions)

        self.template = Template(template)
        self.role = role
        self.prefer = prefer
        self.avoid = avoid
        self.model_params = model_params

    def __call__(self, **kwargs) -> Message:
        """Returns the message from the prompt."""

        return Message(self.template.render(**kwargs), self.role)


@dataclass
class Model(Functions):
    """Represents a model."""

    seed: InitVar[Conversation | Message | str | None] = None
    name: str = "gpt-3.5-turbo-0613"
    stream_callback: Callable[[str], Any] | None = None
    conversation: Conversation = field(init=False, default_factory=Conversation)

    def __post_init__(
        self,
        functions: list[Callable[..., Any]] | None = None,
        seed: Conversation | Message | str | None = None,
    ):
        """Initializes the model.

        Optionally pass a seed as conversation, message or string.
        """

        super().__post_init__(functions)

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
        functions: list[Callable[..., Any]] | list[Function] | None = None,
        prefer: list[str] | str | None = None,
        avoid: list[str] | str | None = None,
        stream_callback: Callable[[str], Any] | None = None,
        **kwargs,
    ) -> str:
        """Returns the response of the model."""

        self.update(*args)

        _functions = [Function(fn) for fn in functions or []] + self._functions

        if _functions:
            kwargs["functions"] = [function.schema for function in _functions]
            # kwargs["function_call"] = "auto"

        stream_callback = stream_callback or self.stream_callback

        response = None
        if stream_callback:
            for chunk in self.completion(
                stream=True, prefer=prefer, avoid=avoid, **kwargs
            ):
                chunk = Message(chunk)
                stream_callback(chunk.content)

                response = chunk if response is None else response + chunk
        else:
            response = next(self.completion(prefer=prefer, avoid=avoid, **kwargs))

        if response is None:
            raise ValueError("No response from model.")

        message = Message(response)
        if message.function_call:
            self.update(message)

            function_output = self.function_call(
                message.function_call.name,
                message.function_call.arguments,
                functions=_functions,
            )

            function_message = Message(
                str(function_output), Role.FUNCTION, message.function_call.name
            )

            kwargs.pop("functions", None)

            # TODO might result in an infinite loop - fix this!
            self(
                function_message,
                functions=functions,
                stream_callback=stream_callback,
                **kwargs,
            )
        else:
            self.update(Message(response))

        return self.conversation[-1]()

    @backoff.on_exception(
        backoff.expo,
        [RateLimitError, ServiceUnavailableError],
        max_time=60,
        logger=logger,
    )
    def completion(
        self,
        stream: bool = False,
        prefer: list[str] | str | None = None,
        avoid: list[str] | str | None = None,
        **kwargs,
    ) -> Generator[dict, None, None]:
        """Calls the model, retries on RateLimitError."""

        name = kwargs.pop("name", self.name)
        logit_bias = kwargs.get("logit_bias", {})

        if prefer:
            _logit_bias, _ = get_logit_bias(name, prefer)

            logit_bias.update(_logit_bias)
        if avoid:
            _logit_bias, _ = get_logit_bias(name, avoid, bias=-100)

            logit_bias.update(_logit_bias)

        if logit_bias:
            kwargs["logit_bias"] = logit_bias

            # this is a hard limit by OpenAI for tokens with a logit bias
            if len(kwargs["logit_bias"]) > 300:
                raise ValueError(
                    f'Too many tokens ({len(kwargs["logit_bias"])}) in logit bias \
for model {name}.'
                )

        if len(self.conversation) > 0:
            response = openai.ChatCompletion.create(
                model=name,
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

    def function_call(
        self,
        name: str,
        arguments: str | None = None,
        functions: list[Function] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Calls the function with the given name."""

        result = None
        try:
            result = super().function_call(name, arguments, functions, **kwargs)
        except (ValidationError, json.JSONDecodeError) as e:
            result = e

        return result

    def update(self, *args: Conversation | Message | str):
        """Updates the models conversation."""

        self.conversation += Conversation(*args)

    def reset(self):
        """Resets the conversation."""

        self.conversation = Conversation()


@dataclass
class Agent(Functions):
    """Represents an agent working with a model and a playbook."""

    playbook: list[Conversation | Message | Prompt | Agent | str] = field(init=False)
    model: Model = field(default_factory=Model)

    def __init__(
        self,
        *args: Conversation | Message | Prompt | Agent | str,
        model: Model = Model(),
        functions: list[Callable[..., Any]] | None = None,
    ):
        """Initializes the playbook."""

        super().__init__(functions=functions)

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
                    prefer=prompt.prefer,
                    avoid=prompt.avoid,
                    functions=prompt._functions + self._functions,
                    **prompt.model_params,
                )

                steps = []
            elif isinstance(step, Agent):
                agent = step

                yield self.model(*steps, agent(**inputs), functions=self._functions)
            else:
                steps.append(step)
