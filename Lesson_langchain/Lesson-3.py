# Demonstrates using ChatPromptTemplate, SystemMessage, HumanMessage, and StrOutputParser with ChatGroq.
# Prompts the user for a programming topic and formats the LLM output into a structured explanation.

# ============================================
# LESSON 3 — PROMPTS + OUTPUT PARSERS + MESSAGES
# ============================================


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

system_message = SystemMessage(
    content="You are a helpful programming tutor."
)

topic = input("Enter topic : ")

human_message = HumanMessage(
    content=f"Explain {topic} in simple terms"
)

# Display the messages
print("System Message:")
print(system_message.content)

print("\nHuman Message:")
print(human_message.content)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful programming tutor."),
    ("human", """
    Explain the following topic in simple terms:

    Topic:
    {topic}

    Include:
    - Definition
    - Simple example
    - One real-world use case
    """)
])

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

parser = StrOutputParser()

chain = (
    prompt
    | llm
    | parser
)

answer = chain.invoke({
    "topic": topic
})

print("\nLLM Response:")
print(answer)