# Demonstrates basic LangChain model integration with ChatOpenAI for answering questions.
# Implements a command-line loop that takes user programming questions and outputs AI answers.

# ============================================================
# LESSON 1 — LANGCHAIN BASICS
# Project: AI Programming Assistant
# ============================================================

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    max_tokens=500
)

def ask_question(question: str) -> str:
    """
    Send a programming question to the model
    and return the generated answer.
    """

    if not question.strip():
        return "Please enter a valid question."

    try:
        response = llm.invoke(question)

        answer = response.content

        return answer

    except Exception as error:
        return f"An error occurred: {error}"


while True:

    question = input("enter your question : ").strip()

    if question.lower() in {"exit", "quit"}:
        print("Thank you for using the assistant!")
        break

    answer = ask_question(question)

    print("\nAI ANSWER:")
    print(answer)
