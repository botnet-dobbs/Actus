import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.orchestrator import extract_json, run_agent, _build_rag_query
from app.agents.builder import AgentConfig
from app.agents.tools import MAX_INVOKE_DEPTH, MAX_PARALLEL_AGENTS, ToolResult, _invoke_stack


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


def llm_response(content: str, total_tokens: int = 100,
                 prompt_tokens: int = 60, completion_tokens: int = 40):
    r = MagicMock()
    r.choices[0].message.content = content
    r.usage.total_tokens = total_tokens
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
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
    assert result["prompt_tokens"] == 60
    assert result["completion_tokens"] == 40
    assert result["confidence"] is None  # done signal had no confidence field


# ── run_agent: failure modes ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_json_response_triggers_recovery():
    # Non-JSON followed by a valid done signal — agent recovers
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[NON_JSON, DONE])), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=3))
    assert result["status"] == "completed"
    assert result["result"] == "finished"


@pytest.mark.asyncio
async def test_non_json_at_last_iteration_returns_incomplete():
    # Non-JSON at the final iteration — returns incomplete with raw text
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=NON_JSON)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=1))
    assert result["status"] == "incomplete"
    assert "summary" in result["result"]  # raw LLM text preserved


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


# ── Path 2: pre-loop RAG context injection ────────────────────────────────────

def test_build_rag_query_static_template():
    config = make_config(rag_query_template="overdue invoices unpaid")
    assert _build_rag_query(config, None) == "overdue invoices unpaid"


def test_build_rag_query_dynamic_template():
    config = make_config(rag_query_template="invoices for client {client}")
    assert _build_rag_query(config, {"client": "Acme"}) == "invoices for client Acme"


def test_build_rag_query_template_missing_var_uses_raw():
    config = make_config(rag_query_template="invoices for {client}")
    assert _build_rag_query(config, {}) == "invoices for {client}"


def test_build_rag_query_fallback_to_extra_context_query():
    config = make_config(rag_query_template="")
    assert _build_rag_query(config, {"query": "at-risk customers"}) == "at-risk customers"


def test_build_rag_query_no_template_no_query():
    config = make_config(rag_query_template="")
    assert _build_rag_query(config, {"region": "EU"}) is None
    assert _build_rag_query(config, None) is None


@pytest.mark.asyncio
async def test_rag_context_preloaded_into_messages():
    retrieved = [{"document": "Customer name=Alice segment=enterprise", "metadata": {"type": "Customer", "object_id": 1}, "score": 0.92}]
    with patch("app.agents.orchestrator.retrieve", return_value=retrieved), \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(
            make_config(rag_query_template="enterprise customers"),
            extra_context=None,
        )
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_rag_context_failure_is_non_fatal():
    with patch("app.agents.orchestrator.retrieve", side_effect=Exception("chroma unavailable")), \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(rag_query_template="find something"))
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_rag_no_template_skips_retrieval():
    with patch("app.agents.orchestrator.retrieve") as mock_retrieve, \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        await run_agent(make_config())
    mock_retrieve.assert_not_called()


# ── invoke_agent and invoke_agents_parallel ───────────────────────────────────

@pytest.mark.asyncio
async def test_invoke_agent_happy_path():
    child_result = {"status": "completed", "result": "analysis done",
                    "confidence": 0.9, "iterations": 1, "total_tokens": 50}
    with patch("app.agents.builder.get_agent", return_value=make_config(id="child")), \
         patch("app.agents.orchestrator.run_agent", AsyncMock(return_value=child_result)):
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="child", query="analyse this")
    assert result["status"] == "completed"
    assert result["result"] == "analysis done"
    assert result["confidence"] == 0.9
    assert "prompt_tokens" not in result  # internal fields stripped


@pytest.mark.asyncio
async def test_invoke_agent_unknown_agent_returns_error():
    with patch("app.agents.builder.get_agent", side_effect=KeyError("no-such-agent")):
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="no-such-agent")
    assert result["status"] == "error"
    assert "Unknown agent" in result["error"]


@pytest.mark.asyncio
async def test_invoke_agent_circular_detection():
    # Simulate the stack already containing the target agent
    token = _invoke_stack.set(["planner", "child-agent"])
    try:
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="child-agent", query="anything")
    finally:
        _invoke_stack.reset(token)
    assert result["status"] == "error"
    assert "Circular" in result["error"]


@pytest.mark.asyncio
async def test_invoke_agent_depth_limit():
    deep_stack = [f"agent-{i}" for i in range(MAX_INVOKE_DEPTH)]
    token = _invoke_stack.set(deep_stack)
    try:
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="one-more")
    finally:
        _invoke_stack.reset(token)
    assert result["status"] == "error"
    assert "depth" in result["error"].lower()


@pytest.mark.asyncio
async def test_invoke_agents_parallel_happy_path():
    r1 = {"status": "completed", "result": "result-a", "iterations": 1, "total_tokens": 10}
    r2 = {"status": "completed", "result": "result-b", "iterations": 1, "total_tokens": 10}
    configs = {"agent-a": make_config(id="agent-a"), "agent-b": make_config(id="agent-b")}
    with patch("app.agents.builder.get_agent", side_effect=lambda aid: configs[aid]), \
         patch("app.agents.orchestrator.run_agent", AsyncMock(side_effect=[r1, r2])):
        from app.agents.tools import invoke_agents_parallel
        results = await invoke_agents_parallel(agent_ids=["agent-a", "agent-b"], query="go")
    assert len(results) == 2
    assert all(r["status"] == "completed" for r in results)


@pytest.mark.asyncio
async def test_invoke_agents_parallel_partial_failure():
    r_ok = {"status": "completed", "result": "ok", "iterations": 1, "total_tokens": 10}

    def _get_agent(aid):
        if aid == "good":
            return make_config(id="good")
        raise KeyError(aid)

    with patch("app.agents.builder.get_agent", side_effect=_get_agent), \
         patch("app.agents.orchestrator.run_agent", AsyncMock(return_value=r_ok)):
        from app.agents.tools import invoke_agents_parallel
        results = await invoke_agents_parallel(agent_ids=["good", "missing"])
    assert len(results) == 2
    statuses = {r["status"] for r in results}
    assert "completed" in statuses
    assert "error" in statuses


@pytest.mark.asyncio
async def test_invoke_agents_parallel_empty_list():
    from app.agents.tools import invoke_agents_parallel
    results = await invoke_agents_parallel(agent_ids=[])
    assert results == []


@pytest.mark.asyncio
async def test_confidence_score_passed_through():
    done_with_confidence = llm_response('{"done": true, "result": "analysis done", "confidence": 0.92}')
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=done_with_confidence)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["status"] == "completed"
    assert result["confidence"] == 0.92


@pytest.mark.asyncio
async def test_confidence_none_when_not_provided():
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config())
    assert result["confidence"] is None


def test_extract_json_handles_text_before_json():
    mixed = 'Sure! Here is my response: {"done": true, "result": "ok"}'
    assert extract_json(mixed) == '{"done": true, "result": "ok"}'


def test_extract_json_handles_nested_args():
    mixed = 'Let me call a tool: {"tool": "search", "args": {"query": "test"}}'
    result = extract_json(mixed)
    parsed = json.loads(result)
    assert parsed["tool"] == "search"
    assert parsed["args"]["query"] == "test"


def test_extract_json_handles_text_after_json():
    mixed = '{"done": true, "result": "ok"} then I will proceed with more reasoning.'
    result = extract_json(mixed)
    parsed = json.loads(result)
    assert parsed["done"] is True


@pytest.mark.asyncio
async def test_empty_action_sends_recovery_message():
    empty_response = llm_response('{}')
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[empty_response, DONE])), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=3))
    assert result["status"] == "completed"
    assert result["iterations"] == 2  # recovered on second iteration


@pytest.mark.asyncio
async def test_invoke_agents_parallel_cap_single_rejection():
    from app.agents.tools import invoke_agents_parallel
    too_many = [f"agent-{i}" for i in range(MAX_PARALLEL_AGENTS + 1)]
    results = await invoke_agents_parallel(agent_ids=too_many)
    assert len(results) == 1  # single rejection, not N identical errors
    assert results[0]["status"] == "error"
    assert "Batch rejected" in results[0]["error"]


@pytest.mark.asyncio
async def test_non_dict_args_do_not_crash():
    null_args = llm_response('{"tool": "search", "args": null}')
    with patch("app.agents.orchestrator.call_llm_with_retry",
               AsyncMock(side_effect=[null_args, DONE])), \
         patch("app.agents.orchestrator.run_tool",
               AsyncMock(return_value=ToolResult(tool_name="search", success=True, output="ok"))), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(max_iterations=3))
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_save_context_failure_does_not_crash_run():
    from unittest.mock import patch as _patch
    with patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         _patch("app.agents.orchestrator.save_context",
                side_effect=ValueError("context too large")):
        result = await run_agent(make_config())
    assert result["status"] == "completed"
    assert result["result"] == "finished"


@pytest.mark.asyncio
async def test_invoke_agent_no_call_stack_in_error():
    token = _invoke_stack.set(["planner", "child"])
    try:
        from app.agents.tools import invoke_agent
        result = await invoke_agent(agent_id="child")
    finally:
        _invoke_stack.reset(token)
    assert result["status"] == "error"
    assert "call_stack" not in result


# ── Tool schema filtering ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_only_allowed_tools_in_system_prompt():
    fake_schemas = {
        "tool_a": {"name": "tool_a", "description": "Does A"},
        "tool_b": {"name": "tool_b", "description": "Does B"},
        "tool_c": {"name": "tool_c", "description": "Does C"},
    }
    captured: list[list] = []

    async def mock_llm(model, messages, **kwargs):
        captured.append(messages)
        return DONE

    with patch("app.agents.orchestrator._tool_schemas", fake_schemas), \
         patch("app.agents.orchestrator.call_llm_with_retry", side_effect=mock_llm), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(tools=["tool_a"]))

    assert result["status"] == "completed"
    system_content = captured[0][0]["content"]
    assert "tool_a" in system_content
    assert "tool_b" not in system_content
    assert "tool_c" not in system_content


@pytest.mark.asyncio
async def test_unregistered_tool_logged_as_warning():
    fake_schemas = {"existing_tool": {"name": "existing_tool", "description": "Real tool"}}
    with patch("app.agents.orchestrator._tool_schemas", fake_schemas), \
         patch("app.agents.orchestrator.call_llm_with_retry", AsyncMock(return_value=DONE)), \
         patch("app.agents.orchestrator.save_context"):
        result = await run_agent(make_config(tools=["existing_tool", "missing_tool"]))
    # unregistered tool is silently filtered — agent still runs
    assert result["status"] == "completed"


# ── Env interpolation (builder._interpolate_env) ──────────────────────────────

def test_env_interpolation_replaces_placeholder(monkeypatch):
    from app.agents.builder import _interpolate_env
    monkeypatch.setenv("MY_SECRET", "super-secret-value")
    result = _interpolate_env({"webhook": {"secret": "${MY_SECRET}"}})
    assert result == {"webhook": {"secret": "super-secret-value"}}


def test_env_interpolation_missing_var_raises(monkeypatch):
    from app.agents.builder import _interpolate_env
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(ValueError, match="DEFINITELY_NOT_SET"):
        _interpolate_env("${DEFINITELY_NOT_SET}")


def test_env_interpolation_passthrough_non_string():
    from app.agents.builder import _interpolate_env
    assert _interpolate_env(42) == 42
    assert _interpolate_env(True) is True
    assert _interpolate_env(None) is None
    assert _interpolate_env(3.14) == 3.14


def test_env_interpolation_nested_list(monkeypatch):
    from app.agents.builder import _interpolate_env
    monkeypatch.setenv("DB_URL", "postgres://localhost/db")
    result = _interpolate_env(["static", "${DB_URL}", 42])
    assert result == ["static", "postgres://localhost/db", 42]
