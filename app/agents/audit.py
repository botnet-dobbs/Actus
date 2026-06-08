import json
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, Session
from app.database import get_engine
import structlog

log = structlog.get_logger()


class AgentRunLog(SQLModel, table=True):
    __tablename__ = "agent_run_logs"

    id: int | None = Field(default=None, primary_key=True)
    agent_id: str | None = Field(default=None, index=True)
    run_id: str = Field(index=True)
    triggered_by: int | None = Field(default=None, index=True)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    model: str | None = None
    pii_detected: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: str = Field(default="[]")  # JSON: [{tool, success, detail}]
    outcome: str = "success"              # "success" | "incomplete" | "error" | "timeout"
    result_summary: str | None = None     # first 500 chars only, never raw output
    ip_address: str | None = None


def log_agent_run(
    run_id: str,
    started_at: datetime,
    model: str | None,
    pii_detected: bool,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    outcome: str,
    tool_calls: list[dict] | None = None,
    agent_id: str | None = None,
    triggered_by: int | None = None,
    result_summary: str | None = None,
    ip_address: str | None = None,
) -> None:
    entry = AgentRunLog(
        run_id=run_id,
        agent_id=agent_id,
        triggered_by=triggered_by,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        model=model,
        pii_detected=pii_detected,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tool_calls=json.dumps(tool_calls or []),
        outcome=outcome,
        result_summary=result_summary,
        ip_address=ip_address,
    )
    with Session(get_engine()) as session:
        try:
            session.add(entry)
            session.commit()
        except Exception as e:
            session.rollback()
            log.error("agent_run_log_write_failed", run_id=run_id, error=str(e))
