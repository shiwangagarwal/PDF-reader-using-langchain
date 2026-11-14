# Imports
from langchain_community.document_loaders import PyPDFLoader

# LangChain 1.x moved the text splitters to their own package
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_document(pdf):
    """
    Load a PDF and split it into chunks for efficient retrieval.

    :param pdf: path to the PDF file to load
    :return: List of chunks of text (each keeps its page number in metadata)
    """
    loader = PyPDFLoader(pdf)
    docs = loader.load()

    # Chunks of 500 characters with 100 characters of overlap so context is not lost at boundaries
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    return chunks
