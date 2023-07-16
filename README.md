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
from fastllm import Agent, Message, Model, Prompt, Role

name_finder = Agent(
    Message("You are an expert name finder.", Role.SYSTEM),
    Prompt("Find {{ n }} short names."),
    Prompt("Print names comma separated, nothing else!"),
    model=Model(),
)

names = name_finder(n=3).split(",")

print(names)
```

Output:

```bash
['John', 'Bob', 'Alice']
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