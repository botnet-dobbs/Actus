import structlog
from app.rag.embedder import embed
from app.rag.store import get_collection

log = structlog.get_logger()

_METADATA_FIELDS = {
    "id", "created_at", "updated_at", "created_by",
    "is_deleted", "deleted_at", "deleted_by",
}


def _object_to_text(type_name: str, obj) -> str:
    parts = [type_name]
    for field, value in obj.model_dump().items():
        if field not in _METADATA_FIELDS and value is not None:
            parts.append(f"{field}={value}")
    return " ".join(str(p) for p in parts)


def index_object(type_name: str, object_id: int, obj) -> None:
    try:
        text = _object_to_text(type_name, obj)
        embedding = embed(text)
        collection = get_collection(type_name)
        collection.upsert(
            ids=[str(object_id)],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"type": type_name, "object_id": object_id}],
        )
        log.info("rag_indexed", type=type_name, id=object_id)
    except Exception as e:
        log.error("rag_index_failed", type=type_name, id=object_id, error=str(e))


def delete_from_index(type_name: str, object_id: int) -> None:
    try:
        collection = get_collection(type_name)
        collection.delete(ids=[str(object_id)])
        log.info("rag_deleted", type=type_name, id=object_id)
    except Exception as e:
        log.error("rag_delete_failed", type=type_name, id=object_id, error=str(e))
