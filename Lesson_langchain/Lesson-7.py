# Demonstrates document chunking techniques using RecursiveCharacterTextSplitter on LangChain Documents.
# Splits large text blocks into smaller overlapping chunks while tracking character indices and metadata.

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

documents = [
    Document(
        page_content="""
        Python is a programming language widely used in data science.
        Pandas is a Python library used for data manipulation and analysis.
        NumPy is used for numerical computing and array operations.
        Scikit-learn provides machine learning algorithms and tools.
        Matplotlib is used for data visualization.
        """,
        metadata={"source": "data_science_notes", "topic": "data_science"}
    ),

    Document(
        page_content="""
        Machine learning is a branch of artificial intelligence.
        Supervised learning uses labeled training data.
        Unsupervised learning discovers patterns in unlabeled data.
        Model evaluation measures the performance of machine learning models.
        """,
        metadata={"source": "ml_notes", "topic": "machine_learning"}
    )
]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    add_start_index=True
)


chunks = text_splitter.split_documents(documents)


print("Total documents:", len(documents))
print("Total chunks:", len(chunks))


for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}\n")
    print("Content:", chunk.page_content)
    print("\nMetadata:", chunk.metadata)
