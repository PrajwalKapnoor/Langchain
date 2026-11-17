
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {
    "configurable": {
        "session_id": "user_1"
    }
}

response1 = chatbot.invoke(
    {"input": "My name is Prajwal."},
    config=config
)

print(response1.content)

response2 = chatbot.invoke(
    {"input": "What is my name?"},
    config=config
)

print(response2.content)

response3 = chatbot.invoke(
    {"input": "I am learning LangChain."},
    config=config
)

print(response3.content)

response4 = chatbot.invoke(
    {"input": "What am I learning?"},
    config=config
)

print(response4.content)

