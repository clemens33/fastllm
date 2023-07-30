"""Using logit bias to guide the generation of a message."""

from fastllm import Agent, Prompt

cat = Agent(
    Prompt("Say Cat!"),
)

print(cat())

not_cat = Agent(
    Prompt("Say Cat!", avoid=r"[ ]?Cat"),
)

print(not_cat())


seriously_not_a_cat = Agent(
    Prompt("Say Cat!, PLEEASSEE", avoid=r"[ ]?[Cc][aA][tT]]"),
)

print(seriously_not_a_cat())
