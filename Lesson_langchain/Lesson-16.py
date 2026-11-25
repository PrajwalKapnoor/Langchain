import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use conversation history to answer questions."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm


def get_session_history(session_id):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"
    )


chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


session_id = "prajwal_session"

config = {
    "configurable": {
        "session_id": session_id
    }
}


while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Chat ended.")
        break

    response = chatbot.invoke(
        {"input": user_input},
        config=config
    )

    print("\nAI:", response.content)

