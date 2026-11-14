"""Meeting-scoped vector retrieval.

Each build receives a fresh in-memory collection, preventing an earlier meeting's
transcript from silently appearing in a new meeting's chat. Persistence and
multi-user memory require a separate authenticated database design.
"""
from __future__ import annotations

import uuid
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vector_store(transcript: str, *, meeting_id: str | None = None) -> Chroma:
    if not transcript or not transcript.strip():
        raise ValueError("Cannot build a search index for an empty transcript")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    # A fresh ID is created for every analysis; callers should not reuse a
    # previous meeting's collection, even when rerunning a different source.
    safe_id = (meeting_id or str(uuid.uuid4())).replace("-", "_")
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i, "meeting_id": safe_id})
        for i, chunk in enumerate(chunks)
    ]
    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name="meeting_" + safe_id,
        # No persist_directory. This demo has no authentication and must not
        # retain globally searchable meeting content between sessions.
    )


def load_vector_store():
    raise RuntimeError(
        "The old shared on-disk meeting index is intentionally disabled. "
        "Call build_vector_store(transcript) for a meeting-scoped session."
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    if vector_store is None:
        raise ValueError("An explicit meeting-scoped vector store is required")
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": max(1, min(k, 10))},
    )
