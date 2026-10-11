# Demonstrates conversational chat history using ChatGroq with SystemMessage, HumanMessage, and AIMessage.
# Maintains context in a multi-turn chat loop until the user types '0' to quit.

from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# Using standard model name
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)

messages = [
    SystemMessage(content="u are helpfull ai assistant"),
]

print("type 0 to quit")

while True:
    prompt = input("You : ")
    if prompt == "0":
        break
    messages.append(HumanMessage(content=prompt))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("Bot :", response.content)


