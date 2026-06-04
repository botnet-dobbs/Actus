from __future__ import annotations

import structlog
from app.config import get_settings
from sentence_transformers import SentenceTransformer

log = structlog.get_logger()

_settings = get_settings()
_model: SentenceTransformer | None = None


def warmup() -> None:
    global _model
    log.info("embedding_model_loading", model=_settings.embedding_model)
    _model = SentenceTransformer(_settings.embedding_model)
    log.info("embedding_model_loaded", model=_settings.embedding_model)


def embed(text: str) -> list[float]:
    if _model is None:
        raise RuntimeError("Embedding model not loaded — call warmup() in lifespan first")
    return _model.encode(text, normalize_embeddings=True).tolist()
