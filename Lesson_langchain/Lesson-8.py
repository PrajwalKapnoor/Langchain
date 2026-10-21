# Demonstrates vector storage and similarity retrieval using OpenAIEmbeddings and Chroma database.
# Stores document embeddings in a vector collection and queries the top matching documents by similarity.

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    Document(
        page_content="Python is a programming language used in data science.",
        metadata={"topic": "python"}
    ),
    Document(
        page_content="Pandas is used for data manipulation and analysis.",
        metadata={"topic": "pandas"}
    ),
    Document(
        page_content="NumPy is used for numerical computing and array operations.",
        metadata={"topic": "numpy"}
    ),
    Document(
        page_content="Scikit-learn provides machine learning algorithms.",
        metadata={"topic": "machine_learning"}
    ),
    Document(
        page_content="Matplotlib is a Python library for data visualization.",
        metadata={"topic": "visualization"}
    )
]


vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="data_science_knowledge"
)


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

query = "Which libraries are useful for data analysis?"

results = retriever.invoke(query)

for i, document in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)