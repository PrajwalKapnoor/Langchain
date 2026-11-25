import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

conversation_history = []
conversation_summary = ""

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Summarize the conversation below.
    Preserve important facts, user preferences, names,
    projects, and relevant context.
    Keep the summary concise.
    """),
    ("human", "{conversation}")
])

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a helpful AI assistant.

    Previous conversation summary:
    {summary}

    Use the summary and recent conversation history
    to answer the user's questions.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


def summarize_history():
    global conversation_summary, conversation_history

    conversation_text = "\n".join(
        f"{message.type}: {message.content}"
        for message in conversation_history
    )

    summary_chain = summary_prompt | llm

    response = summary_chain.invoke({
        "conversation": conversation_text
    })

    conversation_summary = response.content

    conversation_history.clear()


def chat(user_input):
    global conversation_history

    chain = chat_prompt | llm

    response = chain.invoke({
        "summary": conversation_summary,
        "history": conversation_history,
        "input": user_input
    })

    conversation_history.append(
        HumanMessage(content=user_input)
    )

    conversation_history.append(
        AIMessage(content=response.content)
    )

    if len(conversation_history) >= 10:
        summarize_history()

    return response.content


while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        break

    response = chat(user_input)

    print("\nAI:", response)
    print("\n")

    if conversation_summary:
        print("\n[Conversation summary updated]")
