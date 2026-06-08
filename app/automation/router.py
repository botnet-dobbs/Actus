import json
from datetime import datetime, timezone
from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlmodel import Session, select
from app.agents.audit import log_agent_run
from app.agents.builder import get_agent, list_agents, reload_agents
from app.agents.orchestrator import run_agent_with_timeout
from app.auth.models import User
from app.auth.jwt import require_role, write_audit_log
from app.context.models import Workflow, WorkflowStatus
from app.database import get_engine, get_session
import structlog

log = structlog.get_logger()
router = APIRouter()

_OUTCOME_MAP = {
    "completed": "success",
    "incomplete": "incomplete",
    "error": "error",
    "timeout": "timeout",
}


class TriggerRequest(BaseModel):
    extra_context: dict | None = None


class WorkflowResponse(BaseModel):
    id: int
    name: str
    agent_id: str
    status: str
    run_id: str | None
    result_json: str | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    created_by: int | None
    extra_context_json: str | None


async def _run_workflow(workflow_id: int, triggered_by: int | None, ip_address: str | None) -> None:
    started_at = datetime.now(timezone.utc)
    with Session(get_engine()) as session:
        wf = session.get(Workflow, workflow_id)
        wf.status = WorkflowStatus.running
        wf.started_at = started_at
        session.add(wf)
        session.commit()
        session.refresh(wf)
        agent_id = wf.agent_id
        extra_context_json = wf.extra_context_json

    status = WorkflowStatus.failed
    outcome = "error"
    result = None
    error = None
    config = None
    try:
        config = get_agent(agent_id)
        extra_context = json.loads(extra_context_json) if extra_context_json else None
        result = await run_agent_with_timeout(config, extra_context=extra_context)
        status = WorkflowStatus.completed
        outcome = _OUTCOME_MAP.get(result.get("status", ""), "error")
    except Exception as e:
        error = str(e)
        log.error("workflow_failed", workflow_id=workflow_id, agent_id=agent_id, error=error)
    finally:
        with Session(get_engine()) as session:
            wf = session.get(Workflow, workflow_id)
            wf.status = status
            wf.completed_at = datetime.now(timezone.utc)
            wf.result_json = json.dumps(result) if result else None
            wf.error = error
            if result:
                wf.run_id = result.get("run_id")
            session.add(wf)
            session.commit()

        raw_result = result.get("result") if result else None
        summary = str(raw_result)[:500] if raw_result else None
        log_agent_run(
            run_id=result.get("run_id", "") if result else "",
            started_at=started_at,
            model=config.model if config else None,
            pii_detected=result.get("pii_detected", False) if result else False,
            prompt_tokens=result.get("prompt_tokens", 0) if result else 0,
            completion_tokens=result.get("completion_tokens", 0) if result else 0,
            total_tokens=result.get("total_tokens", 0) if result else 0,
            outcome=outcome,
            tool_calls=result.get("tool_calls") if result else None,
            agent_id=agent_id,
            triggered_by=triggered_by,
            result_summary=summary,
            ip_address=ip_address,
        )


# ── Trigger ───────────────────────────────────────────────────────────────────

@router.post("/trigger/{agent_id}", status_code=202)
async def trigger_agent(
    agent_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    body: TriggerRequest | None = None,
    user: User = Depends(require_role("analyst")),
):
    try:
        config = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent not found: '{agent_id}'")

    with Session(get_engine()) as session:
        wf = Workflow(
            name=config.name,
            agent_id=agent_id,
            created_by=user.id,
            extra_context_json=json.dumps(body.extra_context) if body and body.extra_context else None,
        )
        session.add(wf)
        session.commit()
        session.refresh(wf)
        workflow_id = wf.id

    ip = request.client.host if request.client else None
    background_tasks.add_task(_run_workflow, workflow_id, user.id, ip)

    write_audit_log(
        username=user.username,
        action="agent_trigger",
        resource=agent_id,
        ip=ip,
        detail=f"workflow_id={workflow_id}",
    )

    return {"status": "queued", "agent_id": agent_id, "workflow_id": workflow_id}


# ── Workflow polling ───────────────────────────────────────────────────────────

@router.get("/workflows", response_model=list[WorkflowResponse])
async def list_workflows(
    agent_id: Annotated[str | None, Query()] = None,
    status: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("analyst")),
):
    query = select(Workflow)
    if agent_id:
        query = query.where(Workflow.agent_id == agent_id)
    if status:
        query = query.where(Workflow.status == status)
    query = query.order_by(Workflow.created_at.desc()).offset(offset).limit(limit)
    return session.exec(query).all()


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    session: Session = Depends(get_session),
    _: User = Depends(require_role("analyst")),
):
    wf = session.get(Workflow, workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    return wf


# ── Reload agents ─────────────────────────────────────────────────────────────

@router.post("/reload")
async def reload(user: User = Depends(require_role("admin"))):
    reload_agents()
    agents = list_agents()
    log.info("agents_reloaded", count=len(agents), triggered_by=user.username)
    return {"loaded": len(agents), "agents": [a.id for a in agents]}
