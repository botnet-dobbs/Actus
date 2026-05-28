import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.orchestrator import extract_json, run_agent
from app.agents.builder import AgentConfig
from app.agents.tools import ToolResult


# ── helpers ───────────────────────────────────────────────────────────────────

def loads(s: str) -> dict:
    return json.loads(extract_json(s))


def make_config(**kwargs) -> AgentConfig:
    defaults = {
        "id": "test-agent",
        "name": "Test Agent",
        "model": "ollama/mistral",
        "max_iterations": 3,
        "tools": ["search"],
        "token_budget": 10_000,
    }
    return AgentConfig(**(defaults | kwargs))


def llm_response(content: str, total_tokens: int = 100):
    r = MagicMock()
    r.choices[0].message.content = content
    r.usage.total_tokens = total_tokens
    return r


DONE = llm_response('{"done": true, "result": "finished"}')
TOOL_CALL = llm_response('{"tool": "search", "args": {"query": "test"}}')
NON_JSON = llm_response("Here is a summary of the findings.")


# ── extract_json: no fences ───────────────────────────────────────────────────

def test_plain_json():
    assert loads('{"done": true, "result": "ok"}') == {"done": True, "result": "ok"}


def test_plain_json_with_whitespace():
    assert loads('  {"tool": "search"}  ') == {"tool": "search"}


# ── extract_json: fenced ──────────────────────────────────────────────────────

def test_json_fence():
    assert loads("```\n{\"done\": true}\n```") == {"done": True}


def test_json_fence_with_language_tag():
    assert loads("```json\n{\"tool\": \"lookup\", \"args\": {}}\n```") == {"tool": "lookup", "args": {}}


def test_fence_without_closing():
    assert loads("```json\n{\"done\": true}") == {"done": True}


def test_multiline_json_in_fence():
    raw = "```json\n{\n  \"tool\": \"query\",\n  \"args\": {\"limit\": 10}\n}\n```"
    assert loads(raw) == {"tool": "query", "args": {"limit": 10}}


def test_json_with_backtick_value():
    assert loads('{"code": "x = `hello`"}') == {"code": "x = `hello`"}


def test_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        loads("not json at all")


def test_invalid_json_in_fence_raises():
    with pytest.raises(json.JSONDecodeError):
        loads("```json\nnot json\n```")


# ── run_agent: happy path ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path_completes():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "completed"
    assert result["result"] == "finished"
    assert result["total_tokens"] == 100


# ── run_agent: failure modes ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_json_response_handled_gracefully():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=NON_JSON)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "completed"
    assert "summary" in result["result"]


@pytest.mark.asyncio
async def test_llm_call_failure_returns_error():
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=Exception("connection refused"))), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "error"
    assert "connection refused" in result["error"]


@pytest.mark.asyncio
async def test_max_iterations_returns_incomplete():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=TOOL_CALL)), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output="data"))), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=2))
    assert result["status"] == "incomplete"
    assert result["iterations"] == 2


@pytest.mark.asyncio
async def test_token_budget_exceeded_stops_run():
    over_budget = llm_response('{"tool": "search", "args": {}}', total_tokens=200)
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=over_budget)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(token_budget=50))
    assert result["status"] == "incomplete"
    assert result["total_tokens"] == 200


@pytest.mark.asyncio
async def test_tool_failure_does_not_crash():
    responses = [
        llm_response('{"tool": "search", "args": {"query": "test"}}'),
        DONE,
    ]
    failed = ToolResult(tool_name="search", success=False, output=None, error="timeout")
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(side_effect=responses)), \
         patch("app.agents.orchestrator.run_tool", AsyncMock(return_value=failed)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_unauthorised_tool_not_executed():
    responses = [
        llm_response('{"tool": "delete_all", "args": {}}'),
        DONE,
    ]
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(side_effect=responses)), \
         patch("app.agents.orchestrator.run_tool") as mock_run, \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(tools=["search"]))
    mock_run.assert_not_called()
    assert result["status"] == "completed"
