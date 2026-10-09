# ============================================
# LESSON 1 — LANGCHAIN BASICS
# ============================================

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

response = llm.invoke(
    "What is LangChain?"
)

print(response)

print(response.content)