# Demonstrates sequential chain composition with RunnablePassthrough to combine multiple prompts.
# First generates a summary of an input topic and then automatically constructs 3 interview quiz questions.

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)


summary_prompt = ChatPromptTemplate.from_template(
    """
    Summarize the following topic in simple terms.

    Topic: {topic}

    Include the key concepts and a practical example.
    """
)


quiz_prompt = ChatPromptTemplate.from_template(
    """
    Based on the following summary, create 3
    interview questions with answers.

    Summary:
    {summary}
    """
)


summary_chain = (
    summary_prompt
    | llm
    | StrOutputParser()
)


quiz_chain = (
    {
        "summary": summary_chain,
        "question": RunnablePassthrough()
    }
    | quiz_prompt
    | llm
    | StrOutputParser()
)


chain = (
    RunnablePassthrough()
    | quiz_chain
)


topic = input("enter your topic : ")
response = chain.invoke(
    {"topic": topic}
)

print(response)