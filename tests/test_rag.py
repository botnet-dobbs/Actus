import pytest
from unittest.mock import MagicMock, patch


# ── embedder ──────────────────────────────────────────────────────────────────

def test_embed_raises_before_warmup():
    from app.rag import embedder
    original = embedder._model
    embedder._model = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            embedder.embed("hello")
    finally:
        embedder._model = original


def test_embed_returns_list_of_floats():
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    with patch("app.rag.embedder._model", mock_model):
        from app.rag.embedder import embed
        result = embed("test query")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


# ── indexer ───────────────────────────────────────────────────────────────────

def test_object_to_text_excludes_metadata():
    from app.rag.indexer import _object_to_text

    class FakeObj:
        def model_dump(self):
            return {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "created_at": "2026-01-01",
                "is_deleted": False,
                "created_by": 42,
            }

    text = _object_to_text("Customer", FakeObj())
    assert "Customer" in text
    assert "Alice" in text
    assert "alice@example.com" in text
    assert "created_at" not in text
    assert "is_deleted" not in text
    assert "created_by" not in text


def test_index_object_calls_upsert():
    mock_collection = MagicMock()

    class FakeObj:
        def model_dump(self):
            return {"id": 1, "name": "Bob", "segment": "starter"}

    with patch("app.rag.indexer.embed", return_value=[0.1, 0.2, 0.3]), \
         patch("app.rag.indexer.get_collection", return_value=mock_collection):
        from app.rag.indexer import index_object
        index_object("Customer", 1, FakeObj())

    mock_collection.upsert.assert_called_once()
    call_kwargs = mock_collection.upsert.call_args.kwargs
    assert call_kwargs["ids"] == ["1"]
    assert call_kwargs["embeddings"] == [[0.1, 0.2, 0.3]]


def test_index_object_swallows_errors():
    with patch("app.rag.indexer.embed", side_effect=Exception("model error")):
        from app.rag.indexer import index_object

        class FakeObj:
            def model_dump(self):
                return {"name": "Test"}

        # Should not raise
        index_object("Customer", 1, FakeObj())


def test_delete_from_index_calls_delete():
    mock_collection = MagicMock()
    with patch("app.rag.indexer.get_collection", return_value=mock_collection):
        from app.rag.indexer import delete_from_index
        delete_from_index("Customer", 42)
    mock_collection.delete.assert_called_once_with(ids=["42"])


# ── retriever ─────────────────────────────────────────────────────────────────

def test_retrieve_returns_sorted_results():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [["Customer name=Alice", "Customer name=Bob"]],
        "metadatas": [[{"type": "Customer", "object_id": 1},
                       {"type": "Customer", "object_id": 2}]],
        "distances": [[0.1, 0.3]],  # lower distance = higher score
    }

    with patch("app.rag.retriever.embed", return_value=[0.1, 0.2]), \
         patch("app.rag.retriever.get_collection", return_value=mock_collection), \
         patch("app.rag.retriever.list_types", return_value=["Customer"]):
        from app.rag.retriever import retrieve
        results = retrieve("find alice")

    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["score"] == round(1 - 0.1, 4)


def test_retrieve_skips_empty_collections():
    mock_empty = MagicMock()
    mock_empty.count.return_value = 0

    with patch("app.rag.retriever.embed", return_value=[0.1, 0.2]), \
         patch("app.rag.retriever.get_collection", return_value=mock_empty), \
         patch("app.rag.retriever.list_types", return_value=["Customer"]):
        from app.rag.retriever import retrieve
        results = retrieve("anything")

    assert results == []


def test_retrieve_filters_by_type():
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "documents": [["Machine serial_number=SN001"]],
        "metadatas": [[{"type": "Machine", "object_id": 5}]],
        "distances": [[0.2]],
    }

    with patch("app.rag.retriever.embed", return_value=[0.1]), \
         patch("app.rag.retriever.get_collection", return_value=mock_collection):
        from app.rag.retriever import retrieve
        results = retrieve("machine in warehouse", type_name="Machine")

    assert len(results) == 1
    assert results[0]["metadata"]["type"] == "Machine"


# ── semantic_search tool ──────────────────────────────────────────────────────

def test_semantic_search_tool_registered():
    from app.agents.tools import _tool_schemas
    assert "semantic_search" in _tool_schemas
    schema = _tool_schemas["semantic_search"]
    assert "query" in schema["parameters"]
    assert "type_name" in schema["parameters"]
    assert "top_k" in schema["parameters"]


def test_semantic_search_tool_calls_retrieve():
    with patch("app.rag.retriever.embed", return_value=[0.1, 0.2]), \
         patch("app.rag.retriever.get_collection") as mock_get_col, \
         patch("app.rag.retriever.list_types", return_value=[]):
        mock_get_col.return_value.count.return_value = 0
        from app.agents.tools import _tools
        result = _tools["semantic_search"](query="test", type_name="", top_k=3)
    assert isinstance(result, list)
