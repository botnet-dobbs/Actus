from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from app.agents.builder import get_agent
from app.agents.orchestrator import run_agent_with_timeout
from app.auth.models import User
from app.auth.jwt import require_role, write_audit_log

router = APIRouter()


@router.post("/trigger/{agent_id}")
async def trigger_agent(
    agent_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_role("analyst")),
):
    try:
        config = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent not found: '{agent_id}'")

    background_tasks.add_task(run_agent_with_timeout, config)

    ip = request.client.host if request.client else None
    write_audit_log(username=user.username, action="agent_trigger", resource=agent_id, ip=ip)

    return {"status": "queued", "agent_id": agent_id}
