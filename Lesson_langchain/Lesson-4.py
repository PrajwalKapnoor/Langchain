# Demonstrates LCEL (LangChain Expression Language) pipelines with RunnablePassthrough and ChatPromptTemplate.
# Takes a dynamic topic input, injects standard parameters, and outputs a formatted topic breakdown.

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)


prompt = ChatPromptTemplate.from_template(
    """
    Explain the following programming topic.

    Topic: {topic}

    Difficulty: {difficulty}

    Include:
    - Definition
    - Code example
    - Real-world use case
    """
)

parser = StrOutputParser()

chain = (
    {
        "topic": RunnablePassthrough(),
        "difficulty": lambda x: "Intermediate"
    }
    | prompt
    | llm
    | parser
)

topic = input("enter your topic : ")

response = chain.invoke(topic)

print(response)