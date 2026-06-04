import structlog
from app.rag.embedder import embed
from app.rag.store import get_collection
from app.ontology.registry import list_types

log = structlog.get_logger()


def retrieve(query: str, type_name: str | None = None, top_k: int = 5) -> list[dict]:
    embedding = embed(query)
    type_names = [type_name] if type_name else list_types()

    results = []
    for tn in type_names:
        try:
            collection = get_collection(tn)
            count = collection.count()
            if count == 0:
                continue
            r = collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, count),
            )
            for doc, meta, dist in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0]
            ):
                results.append({
                    "document": doc,
                    "metadata": meta,
                    "score": round(1 - dist, 4),
                })
        except Exception as e:
            log.warning("rag_retrieval_failed", type=tn, error=str(e))
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
