# FastLLM

Fast and simple wrapper around LLMs. The package aims to be simply, precise and allows for fast prototyping of agents and applications around LLMs. At the moment focus around OpenAI's chat models.

**Warning - experimental package and subject to change.** For features and plans see the [roadmap](#roadmap).

## Installation

```bash
pip install fastllm
```

## [Samples](./samples)

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

creative_name_finder = Agent(
    Message("You are an expert name finder.", Role.SYSTEM),
    Prompt("Find {{ n }} names.", temperature=2.0),
    Prompt("Print names comma separated, nothing else!"),
    model=Model(name="gpt-4"),
)

names = creative_name_finder(n=3).split(",")

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
```

```bash
100
```

```python
another_result = calculator_agent(
    task="If I have 114 apples and 3 elephants, how many apples will each elephant get?"
)

print(another_result)
```

```bash
38
```

#### Avoid words/phrases 

Avoid/ban word and phrases - supports patterns. Patterns follow regex syntax but do not support all features. If the number of possible strings matching the pattern is too large, the pattern is ignored. 

For avoiding/banning words typically it is advised to put a [blank space](https://community.openai.com/t/reproducible-gpt-3-5-turbo-logit-bias-100-not-functioning/88293/8) in front of the word.

```python
cat = Agent(
    Prompt("Say Cat!"),
)

print(cat())
```

```bash
Cat!
```

No we avoid/ban the regex pattern `r"[ ]?Cat"` (e.g. " Cat" or "Cat") from the response.

```python
not_cat = Agent(
    Prompt("Say Cat!", avoid=r"[ ]?Cat"),
)

print(not_cat())
```

OpenAI is making fun of us (that really happened!) - obviously we need to be more specific (e.g. ban lowercase and uppercase)

```bash
Dog! Just kidding, cat!
```

Ok let's try again.

```python
seriously_not_a_cat = Agent(
    Prompt("Say Cat!, PLEEASSEE", avoid=r"[ ]?[Cc][aA][tT]]"),
)

print(seriously_not_a_cat())
```

Well no cat but kudos for the effort.

```bash
Sure, here you go: "Meow! "
```

#### Prefer words/phrases

Prefer words/phrases - supports patterns. Patterns follow regex syntax but do not support all features. Only supports pattern matching a limited number of strings. The max token length is set to the longest possible string in the pattern. Also the order of token can not be guaranteed. 

```python
meow = Agent(
    Prompt("Say Hi!", prefer="Meow!"),
)

print(meow())
```

```bash
Meow!
```

```python
austria_wins = Agent(
    Prompt("Predict the score for Austria against Germany.", prefer=r"Austria: [3-4], Germany: [0-1]"),
)

print(austria_wins())
```

Order of tokens is not guaranteed (and this is obviously false ;))

```bash
Austria, 1: Germany, 3
```

```python
meow = Agent(
    Prompt("Say Hi!", prefer="Meow!", max_tokens=10),
)

print(meow())
```

Our model can only say "Meow" or "!" up to 10 tokens.

```bash
Meow!!!!Meow!Meow!Meow!Meow!Meow
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
- [x] Streaming with function calling
- [ ] Function calling can result in an infinite loop if LLM can not provide function name or arguments properly. This needs to be handled by the Model class.
- [ ] Force particular function call by providing function call argument
- [ ] Option to "smartly forget" conversation history in case context length is too long.
- [x] Prompts with pattern using logit bias to guide LLM completion.
- [ ] Able to switch between models (e.g. 3.5 and 4) within one agent over different prompts.
- [ ] Handling of multiple response messages from LLMs in a single call. At the moment only the first response is kept.
- [ ] Supporting non chat based LLMs (e.g. OpenAI's completion LLMs).
- [ ] Supporting other LLMs over APIs except OpenAI's. (e.g. Google etc.)
- [ ] Supporting local LLMs (e.g. llama-1, llama-2, etc.)

### Package

- [x] Basic package structure and functionality
- [x] Test cases and high test coverage
- [ ] Mock implementation of OpenAI's API for tests
- [ ] Tests against multiple python versions
- [ ] 100% test coverage (at the moment around 90%)
- [ ] Better documentation including readthedocs site.
- [ ] Better error handling and logging
- [ ] Better samples using jupyter notebooks
- [ ] Set up of pre-commit
- [ ] CI using github actions
- [ ] Release and versioning

## Development

Using [poetry](https://python-poetry.org/docs/#installation).

```bash
poetry install
```

### Tests

```bash
poetry run pytest
``` 