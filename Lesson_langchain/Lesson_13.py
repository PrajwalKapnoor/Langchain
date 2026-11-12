from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

history = []

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages = [
        SystemMessage(
            content="You are a helpful AI assistant."
        )
    ] + history + [
        HumanMessage(content=user_input)
    ]

    response = model.invoke(messages)

    print("AI:", response.content)

    # Store conversation
    history.append(HumanMessage(content=user_input))
    history.append(response)
