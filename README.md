# FastLLM

Fast and easy wrapper around LLMs. The package aims to be simply, precise and allows for fast prototyping of agents and applications around LLMs. At the moment focus around OpenAI's models.

**Warning - very early stage of development.**

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

## Development

Using [poetry](https://python-poetry.org/docs/#installation).

```bash
poetry install
```

### Tests

```bash
poetry run pytest
``` 