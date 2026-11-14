# Imports
import os
import tempfile

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# FastEmbed: free local embeddings on ONNX (no PyTorch needed)
from langchain_community.embeddings import FastEmbedEmbeddings

# LangChain 1.x moved the classic chains to the langchain-classic package
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

from scripts.document_loader import load_document
from scripts.secret import GROQ_KEY

# Use a HuggingFace token (if set) so the embedding model download isn't rate-limited (HTTP 429)
try:
    from scripts.secret import HF_TOKEN
    if HF_TOKEN and not HF_TOKEN.startswith("hf_your"):
        os.environ["HF_TOKEN"] = HF_TOKEN
except ImportError:
    pass

# Settings
LLM_MODEL = "openai/gpt-oss-120b"   # switch to "openai/gpt-oss-20b" if you hit rate limits
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # free, runs locally (~70 MB, downloaded once)
NUM_CHUNKS = 6  # how many chunks are retrieved and sent to the LLM per question


# Cached resources
# Streamlit reruns this whole script on every interaction. Without caching, the PDF
# would be re-loaded and re-embedded every time a question is asked.

@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~70 MB)...")
def get_embeddings():
    """Load the local embedding model once and reuse it."""
    try:
        return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception:
        st.error(
            "Could not download the embedding model from HuggingFace (usually a 429 rate limit). "
            "Add your free HuggingFace token as HF_TOKEN in scripts/secret.py "
            "(get one at https://huggingface.co/settings/tokens), then reload this page."
        )
        st.stop()


@st.cache_resource(show_spinner="Processing document... :watch:")
def build_vector_db(file_bytes: bytes):
    """
    Split the PDF into chunks, embed them and store them in FAISS.
    Cached on the file contents, so each PDF is processed only once.
    """
    # PyPDFLoader needs a file path, so write the upload to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        chunks = load_document(tmp_path)
    finally:
        os.remove(tmp_path)

    if not chunks:
        return None, 0
    return FAISS.from_documents(chunks, get_embeddings()), len(chunks)


@st.cache_resource
def get_llm():
    return ChatGroq(model=LLM_MODEL, api_key=GROQ_KEY)


# Streamlit app
st.title("AI-Powered Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    vector_db, num_chunks = build_vector_db(uploaded_file.getvalue())

    if vector_db is None:
        st.error("No text found in this PDF. It may be a scanned/image-only document.")
        st.stop()

    st.success(f"Document ready: {num_chunks} chunks indexed.")

    # Retriever: returns the NUM_CHUNKS chunks most similar to the question
    retriever = vector_db.as_retriever(search_kwargs={"k": NUM_CHUNKS})

    # System prompt: the retrieved chunks are inserted into {context}
    system_prompt = (
        "You are a helpful assistant. Use the given context to answer the question. "
        "If you don't know the answer, say you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # "Stuff" the retrieved chunks into the prompt, then build the RAG chain
    question_answer_chain = create_stuff_documents_chain(get_llm(), prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Thinking..."):
            try:
                result = chain.invoke({"input": question})
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    st.error("Groq rate limit reached. Wait a minute and try again, "
                             "or switch LLM_MODEL to a smaller model.")
                else:
                    st.error(f"Error: {e}")
                st.stop()

        st.write(result["answer"])

        # Show which pages the answer was based on
        pages = sorted({doc.metadata.get("page", 0) + 1 for doc in result["context"]})
        st.caption("Sources: page " + ", ".join(str(p) for p in pages))

        with st.expander("Show retrieved chunks"):
            for doc in result["context"]:
                st.markdown(f"**Page {doc.metadata.get('page', 0) + 1}**")
                st.text(doc.page_content)
