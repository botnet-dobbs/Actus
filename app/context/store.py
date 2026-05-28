from datetime import datetime, timezone, timedelta
from sqlmodel import SQLModel, Field, Session, select
from app.database import get_engine
from app.context.models import AgentContext
import structlog

log = structlog.get_logger()

MAX_CONTEXT_BYTES = 512_000  # 500 KB


class ContextSnapshot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    agent_id: str = Field(index=True)
    run_id: str = Field(unique=True, index=True)
    context_json: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: int | None = None
    ttl_expires_at: datetime | None = None
    is_deleted: bool = Field(default=False, index=True)
    deleted_at: datetime | None = None
    deleted_by: int | None = None


def save_context(context: AgentContext, created_by: int | None = None) -> ContextSnapshot:
    json_str = context.model_dump_json()
    size_bytes = len(json_str.encode("utf-8"))
    if size_bytes > MAX_CONTEXT_BYTES:
        log.error("context_too_large", run_id=context.run_id, size_bytes=size_bytes, limit=MAX_CONTEXT_BYTES)
        raise ValueError(f"Context too large: {size_bytes} bytes (limit {MAX_CONTEXT_BYTES})")
    if size_bytes > MAX_CONTEXT_BYTES * 0.8:
        log.warning("context_approaching_limit", run_id=context.run_id, size_bytes=size_bytes)

    with Session(get_engine()) as session:
        snap = ContextSnapshot(
            agent_id=context.agent_id,
            run_id=context.run_id,
            context_json=json_str,
            created_by=created_by,
            ttl_expires_at=datetime.now(timezone.utc) + timedelta(seconds=context.ttl_seconds),
        )
        try:
            session.add(snap)
            session.commit()
            session.refresh(snap)
        except Exception as e:
            session.rollback()
            log.error("context_save_failed", run_id=context.run_id, agent_id=context.agent_id, error=str(e))
            raise
        log.info("context_saved", run_id=context.run_id, agent_id=context.agent_id)
        return snap


def load_context(run_id: str) -> AgentContext | None:
    with Session(get_engine()) as session:
        snap = session.exec(
            select(ContextSnapshot).where(
                ContextSnapshot.run_id == run_id,
                ContextSnapshot.is_deleted == False,
            )
        ).first()
        if not snap:
            return None
        if snap.ttl_expires_at and datetime.now(timezone.utc) > snap.ttl_expires_at:
            log.warning("context_expired", run_id=run_id, agent_id=snap.agent_id)
            return None
        return AgentContext.model_validate_json(snap.context_json)


def delete_context(run_id: str, deleted_by: int | None = None) -> bool:
    with Session(get_engine()) as session:
        snap = session.exec(
            select(ContextSnapshot).where(
                ContextSnapshot.run_id == run_id,
                ContextSnapshot.is_deleted == False,
            )
        ).first()
        if not snap:
            return False
        snap.is_deleted = True
        snap.deleted_at = datetime.now(timezone.utc)
        snap.deleted_by = deleted_by
        session.add(snap)
        session.commit()
        log.info("context_deleted", run_id=run_id, agent_id=snap.agent_id)
        return True
