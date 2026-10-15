# Demonstrates document loading capabilities using TextLoader, PyPDFLoader, and DirectoryLoader.
# Reads raw text, PDF files, and directory contents into LangChain Document objects with metadata.

from langchain_community.document_loaders import TextLoader

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from warnings import filterwarnings
filterwarnings("ignore")
loader = TextLoader(
    "Lesson_langchain\data.txt"
)

documents = loader.load()

for document in documents:
    print(document.page_content)
    print()
    print(document.metadata)
    print("\n")

pdf_loader = PyPDFLoader("Lesson_langchain\\attention-is-all-you-need-Paper.pdf")

pdf_documents = pdf_loader.load()

for document in pdf_documents:
    print(document.page_content)
    print(document.metadata)

directory_loader = DirectoryLoader(
    "Lesson_langchain\documents/",
    glob="**/*.txt",
    loader_cls=TextLoader
)

all_documents = directory_loader.load()

print("Total documents:", len(all_documents))

for document in all_documents:
    print(document.metadata)
    print(document.page_content)