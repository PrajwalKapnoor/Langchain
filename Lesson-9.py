# Demonstrates a complete Retrieval-Augmented Generation (RAG) pipeline with Chroma and ChatOpenAI.
# Retrieves relevant context from vector embeddings and passes it to the LLM to answer specific questions.

from dotenv import load_dotenv
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    Document(
        page_content="Python is a programming language widely used in data science.",
        metadata={"topic": "python"}
    ),
    Document(
        page_content="Pandas is a Python library used for data manipulation and analysis.",
        metadata={"topic": "pandas"}
    ),
    Document(
        page_content="NumPy is a Python library used for numerical computing.",
        metadata={"topic": "numpy"}
    ),
    Document(
        page_content="Scikit-learn provides machine learning algorithms and tools.",
        metadata={"topic": "machine_learning"}
    ),
    Document(
        page_content="Matplotlib is a Python library used for data visualization.",
        metadata={"topic": "visualization"}
    )
]


vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="lesson_9_rag"
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}

    If the answer is not available in the context,
    say "I don't know based on the provided documents."
    """
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)

def format_documents(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )


rag_chain = (
    {
        "context": retriever | format_documents,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = input("enter the question : ")

answer = rag_chain.invoke(question)

print("Question:", question)
print("\nAnswer:", answer)
