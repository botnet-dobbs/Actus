import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from sqlmodel import Session
from app.agents.builder import AgentConfig
from app.context.models import Workflow, WorkflowStatus
from tests.conftest import seed_user, get_token


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def make_agent_config() -> AgentConfig:
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        model="ollama/mistral",
        tools=[],
    )


# ── POST /automation/trigger/{agent_id} ───────────────────────────────────────

def test_trigger_requires_auth(client):
    resp = client.post("/automation/trigger/test-agent")
    assert resp.status_code == 401


def test_trigger_viewer_blocked(client, engine):
    seed_user(engine, "viewer1", "viewer")
    token = get_token(client, "viewer1")
    resp = client.post("/automation/trigger/test-agent", headers=auth_header(token))
    assert resp.status_code == 403


def test_trigger_unknown_agent_returns_404(client, engine):
    seed_user(engine, "analyst1", "analyst")
    token = get_token(client, "analyst1")
    with patch("app.automation.router.get_agent", side_effect=KeyError("test-agent")):
        resp = client.post("/automation/trigger/missing-agent", headers=auth_header(token))
    assert resp.status_code == 404


def test_trigger_analyst_queues_agent(client, engine):
    seed_user(engine, "analyst2", "analyst")
    token = get_token(client, "analyst2")
    with patch("app.automation.router.get_agent", return_value=make_agent_config()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post("/automation/trigger/test-agent", headers=auth_header(token))
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["agent_id"] == "test-agent"
    assert "workflow_id" in body


def test_outcome_map_completed_becomes_success():
    from app.automation.router import _OUTCOME_MAP
    assert _OUTCOME_MAP["completed"] == "success"
    assert _OUTCOME_MAP["incomplete"] == "incomplete"
    assert _OUTCOME_MAP["error"] == "error"
    assert _OUTCOME_MAP["timeout"] == "timeout"


def test_trigger_admin_can_trigger(client, engine):
    seed_user(engine, "admin1", "admin")
    token = get_token(client, "admin1")
    with patch("app.automation.router.get_agent", return_value=make_agent_config()), \
         patch("app.automation.router._run_workflow", new=AsyncMock(return_value=None)):
        resp = client.post("/automation/trigger/test-agent", headers=auth_header(token))
    assert resp.status_code == 202


# ── SSE streaming helpers ──────────────────────────────────────────────────────

def seed_workflow(engine, agent_id: str, status: str) -> int:
    with Session(engine) as session:
        wf = Workflow(name="Test", agent_id=agent_id,
                      status=WorkflowStatus(status), created_by=None)
        session.add(wf)
        session.commit()
        session.refresh(wf)
        return wf.id


# ── GET /automation/workflows/{id}/stream ─────────────────────────────────────

def test_stream_requires_auth(client):
    resp = client.get("/automation/workflows/999/stream")
    assert resp.status_code == 401


def test_stream_viewer_blocked(client, engine):
    seed_user(engine, "vstream", "viewer")
    token = get_token(client, "vstream")
    resp = client.get("/automation/workflows/999/stream", headers=auth_header(token))
    assert resp.status_code == 403


def test_stream_not_found(client, engine):
    seed_user(engine, "astream_nf", "analyst")
    token = get_token(client, "astream_nf")
    with client.stream("GET", "/automation/workflows/99999/stream",
                       headers=auth_header(token)) as resp:
        body = resp.read().decode()
    assert '"type": "error"' in body


def test_stream_already_completed(client, engine):
    seed_user(engine, "astream_done", "analyst")
    token = get_token(client, "astream_done")
    wf_id = seed_workflow(engine, "test-agent", "completed")
    with client.stream("GET", f"/automation/workflows/{wf_id}/stream",
                       headers=auth_header(token)) as resp:
        body = resp.read().decode()
    assert '"type": "status"' in body
    assert '"status": "completed"' in body
    # full WorkflowResponse emitted immediately — no polling
    assert f'"id":{wf_id}' in body


def test_stream_live_queue_emits_per_iteration_events(client, engine):
    from app.automation.router import _run_queues

    seed_user(engine, "astream_live", "analyst")
    token = get_token(client, "astream_live")
    wf_id = seed_workflow(engine, "test-agent", "running")

    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait({"type": "iteration_start", "run_id": "r1", "iteration": 0})
    queue.put_nowait({"type": "tool_call", "run_id": "r1", "iteration": 0,
                      "tool": "my_tool", "args": {}})
    queue.put_nowait({"type": "tool_result", "run_id": "r1", "iteration": 0,
                      "tool": "my_tool", "success": True, "preview": "{}"})
    queue.put_nowait({"type": "done", "run_id": "r1", "status": "completed",
                      "result": "all good", "iterations": 1, "total_tokens": 100})
    queue.put_nowait(None)  # sentinel
    _run_queues[wf_id] = queue

    try:
        with client.stream("GET", f"/automation/workflows/{wf_id}/stream",
                           headers=auth_header(token)) as resp:
            body = resp.read().decode()
    finally:
        _run_queues.pop(wf_id, None)

    assert "iteration_start" in body
    assert "tool_call" in body
    assert "tool_result" in body
    assert '"type": "done"' in body
    assert '"status": "completed"' in body


def test_stream_db_poll_fallback_detects_transition(client, engine):
    """DB polling fallback: detects running→completed transition and emits full payload."""
    from unittest.mock import patch, AsyncMock
    from app.automation.router import _run_queues

    seed_user(engine, "astream_poll", "analyst")
    token = get_token(client, "astream_poll")
    wf_id = seed_workflow(engine, "test-agent", "running")
    assert wf_id not in _run_queues

    flipped = [False]

    async def flip_on_first_sleep(*_args):
        if not flipped[0]:
            flipped[0] = True
            with Session(engine) as session:
                wf = session.get(Workflow, wf_id)
                wf.status = WorkflowStatus.completed
                session.add(wf)
                session.commit()

    with patch("app.automation.router.asyncio.sleep", side_effect=flip_on_first_sleep):
        with client.stream("GET", f"/automation/workflows/{wf_id}/stream",
                           headers=auth_header(token)) as resp:
            body = resp.read().decode()

    assert '"type": "status"' in body
    assert '"status": "running"' in body          # initial event (f-string template, with spaces)
    assert '"status":"completed"' in body          # WorkflowResponse payload (model_dump_json, no spaces)
