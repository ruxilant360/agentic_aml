"""
Builds the FAISS vector store over AML policy documents, used by the RAG
report tool to ground fraud-flag explanations in actual policy text.

Generalizes the ad-hoc single-PDF loader in the repo's original `rag.py`
prototype into a reusable, multi-document builder.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_embedding_model_name, get_policy_docs_dir

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 200


def _load_policy_documents(doc_paths: list[str]):
    documents = []
    for path in doc_paths:
        documents.extend(PyPDFLoader(path).load())
    return documents


def build_policy_vector_store(
    doc_paths: list[str] | None = None,
    embedding_model_name: str | None = None,
) -> FAISS:
    """
    Build a FAISS vector store over one or more AML policy PDFs.

    Args:
        doc_paths: Paths to policy PDF files. Defaults to every `.pdf` file
            found in `POLICY_DOCS_DIR` (config).
        embedding_model_name: HuggingFace embedding model name. Defaults to
            `EMBEDDING_MODEL_NAME` (config).

    Returns:
        A FAISS vector store ready for similarity search.

    Raises:
        FileNotFoundError: if no policy documents are found.
    """
    if doc_paths is None:
        docs_dir = Path(get_policy_docs_dir())
        doc_paths = [str(p) for p in docs_dir.glob("*.pdf")]

    if not doc_paths:
        raise FileNotFoundError(
            "No policy documents found to build the RAG vector store. "
            "Pass doc_paths explicitly or populate POLICY_DOCS_DIR."
        )

    documents = _load_policy_documents(doc_paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    splits = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name or get_embedding_model_name()
    )
    return FAISS.from_documents(splits, embeddings)
