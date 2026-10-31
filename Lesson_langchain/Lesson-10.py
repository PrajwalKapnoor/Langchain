from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


documents = [
    Document(
        page_content="""
        Python is a programming language used in data science.
        Pandas supports data manipulation and analysis.
        NumPy provides efficient numerical computing.
        Scikit-learn provides machine learning algorithms.
        Matplotlib is used for data visualization.
        """,
        metadata={"source": "data_science.txt"}
    ),
    Document(
        page_content="""
        Supervised learning uses labeled training data.
        Unsupervised learning identifies patterns in unlabeled data.
        Classification predicts categories.
        Regression predicts continuous numerical values.
        """,
        metadata={"source": "machine_learning.txt"}
    ),
    Document(
        page_content="""
        RAG stands for Retrieval-Augmented Generation.
        It retrieves relevant documents before generating an answer.
        Embeddings represent text as numerical vectors.
        Vector stores enable similarity-based document retrieval.
        """,
        metadata={"source": "rag_notes.txt"}
    )
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="lesson_10_advanced_rag"
)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


prompt = ChatPromptTemplate.from_template(
    """
    You are a data science learning assistant.

    Answer using only the retrieved context.
    If the context does not contain the answer,
    say you do not know based on the provided documents.

    Context:
    {context}

    Question:
    {question}

    Provide a clear explanation with relevant examples.
    """
)


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


def format_documents(documents):
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
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


questions = [
    "What is the difference between classification and regression?",
    "How does RAG work?",
    "Which library is used for numerical computing?"
]


for question in questions:
    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    answer = rag_chain.invoke(question)

    print("\nANSWER:")
    print(answer)
