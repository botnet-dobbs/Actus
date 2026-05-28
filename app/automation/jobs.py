from datetime import datetime, timezone, timedelta
from sqlmodel import Session, select, col
from app.database import get_engine
from app.context.store import ContextSnapshot
import structlog

log = structlog.get_logger()

PURGE_AFTER_DAYS = 30


async def purge_old_context_snapshots() -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=PURGE_AFTER_DAYS)
    with Session(get_engine()) as session:
        old = session.exec(
            select(ContextSnapshot).where(
                ContextSnapshot.is_deleted == True,
                col(ContextSnapshot.deleted_at) < cutoff,
            )
        ).all()
        try:
            for snap in old:
                session.delete(snap)
            session.commit()
        except Exception as e:
            session.rollback()
            log.error("context_purge_failed", error=str(e))
            raise
        log.info("context_purge_complete", deleted=len(old))
        return len(old)
