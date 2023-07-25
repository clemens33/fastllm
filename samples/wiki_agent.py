"""Wiki agent sample.

Given a question, it uses Wikipedia to give short, factual and correct answers.
"""

import requests
from bs4 import BeautifulSoup
from jinja2 import Template

from fastllm import Agent, Message, Model, Prompt, Role

wiki_agent = Agent(
    Message(
        "You are an helpful assistant using Wikipedia to give short, \
        factual and correct answers! \
        If the question is unclear ask the user for clarification, \
        by calling the appropriate function!. \
        Always use provided functions to first get factual context! \
        In your final answer always mention the Wikipedia page as source!",
        Role.SYSTEM,
    ),
    Prompt("{{ query }}", Role.USER),
    model=Model(name="gpt-3.5-turbo-16k-0613"),
)


@wiki_agent.function
def query_user(model_question: str):
    """Ask user for clarification."""

    return input(model_question + ":\n")


@wiki_agent.function
def search_page(query: str):
    """Given a search query returns matching Wikipedia page titles."""

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "utf8": 1,
        "srsearch": query,
    }

    data = requests.get(url, params=params).json()

    template = Template(
        """Found wikipedia pages for query: "{{ query }}"
    {%- for i in data["query"]["search"] %}
- Title: "{{ i["title"] }}", Words: {{ i["wordcount"] }}
    {%- endfor %}"""
    )

    return template.render(query=query, data=data)


@wiki_agent.function
def get_page(title: str):
    """Get pure text content of a Wikipedia page given its page title."""

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
        "redirects": "",
    }

    response = requests.get(url, params=params)
    data = response.json()

    raw_html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(raw_html, "html.parser")
    soup.find_all("p")
    text = ""

    for p in soup.find_all("p"):
        text += p.text

    return text


while True:
    wiki_agent.model.reset()

    query = input("Ask me something:\n")

    if query == "exit":
        break

    try:
        response = wiki_agent(query=query)

        print(response)
    except Exception as e:
        print(e)

        continue
