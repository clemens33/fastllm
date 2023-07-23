# FastLLM

Fast and simple wrapper around LLMs. The package aims to be simply, precise and allows for fast prototyping of agents and applications around LLMs. At the moment focus around OpenAI's chat models.

**Warning - experimental package and subject to change.** For features and plans see the [roadmap](#roadmap).

## Samples

Require an openai api key in `OPENAI_API_KEY` environment variable or `.env` file.

```bash
export OPENAI_API_KEY=...
```

### Agents

```python
from fastllm import Agent

find_cities = Agent("List {{ n }} cities comma separated in {{ country }}.")
cities = find_cities(n=3, country="Austria").split(",")

print(cities)
```

```bash
['Vienna', 'Salzburg', 'Graz']
```

```python
from fastllm import Agent, Message, Model, Prompt, Role

s = ";"

creative_name_finder = Agent(
    Message("You are an expert name finder.", Role.SYSTEM),
    Prompt("Find {{ n }} names.", temperature=2.0),
    Prompt("Print names {{ s }} separated, nothing else!"),
    model=Model("gpt-4"),
)

names = creative_name_finder(n=3, s=s).split(s)

print(names)
```

```bash
['Ethan Gallagher, Samantha Cheng, Max Thompson']
```

#### Functions

Functions can be added to Agents, Models or Prompts. Either as initial arguments or as decorator. Functions type hints, documentation and name are inferred from the function and added to the model call.

```python
from typing import Literal

from fastllm import Agent, Prompt

calculator_agent = Agent(
    Prompt("Calculate the result for task: {{ task }}"),
    Prompt("Only give the result number as result without anything else!"),
)


@calculator_agent.function
def calculator(a, b, operator: Literal["+", "-", "*", "/"]):
    """A basic calculator using various operators."""

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
            raise ValueError(f"Unknown operator {operator}")


result = calculator_agent(task="give the final result for (11 + 14) * (6 - 2)")

print(result)

another_result = calculator_agent(
    task="if I have 114 apples and 3 children, how many apples will each child get?"
)

print(another_result)
```

```bash
100
38
```

## Roadmap

### Features

- [x] Prompts using jinja2 templates
- [x] LLM calling with backoff and retry
- [x] Able to register functions to agents, models and prompts using decorators
- [x] Possible to register functions on multiple levels (agent, model, prompt). The function call is only available on the level it was registered.
- [x] Conversation history. The Model class keeps track of the conversation history.
- [x] Function schema is inferred from python function type hints, documentation and name
- [x] Function calling is handled by the Model class itself. Meaning if a LLM response indicate a function call, the Model class will call the function and return the result back to the LLM
- [ ] Function calling can result in an infinite loop if LLM can not provide function name or arguments properly. This needs to be handled by the Model class.
- [ ] Prompts with pattern using logit bias to guide LLM completion.
- [ ] Handling of multiple response messages from LLMs in a single call. At the moment only the first response is kept.
- [ ] Supporting non chat based LLMs (e.g. OpenAI's completion LLMs).
- [ ] Supporting other LLMs over APIs except OpenAI's. (e.g. google, anthropics, etc.)
- [ ] Supporting local LLMs (e.g. llama-1, llama-2, mpt, etc.)

### Package

- [x] Basic package structure and functionality
- [x] Test cases and high test coverage
- [ ] Tests against multiple python versions
- [ ] 100% test coverage (at the moment around 90%)
- [ ] Better documentation including readthedocs site.
- [ ] Better error handling and logging
- [ ] Better samples using jupyter notebooks
- [ ] Set up of pre-commit
- [ ] CI using github actions
- [ ] Prober release and versioning

## Development

Using [poetry](https://python-poetry.org/docs/#installation).

```bash
poetry install
```

### Tests

```bash
poetry run pytest
``` 