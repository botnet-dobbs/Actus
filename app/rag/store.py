from __future__ import annotations

import chromadb
from app.config import get_settings

_settings = get_settings()
_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_settings.chroma_path)
    return _client


def get_collection(type_name: str) -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=type_name.lower(),
        metadata={"hnsw:space": "cosine"},
    )
