from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlmodel import Session, select, col
from app.database import get_engine
from app.context.store import ContextSnapshot
from app.agents.builder import get_agent
from app.agents.orchestrator import run_agent
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


async def run_customer_analysis() -> None:
    try:
        config = get_agent("customer_analyst")
        result = await run_agent(config)
        log.info("scheduled_agent_complete", job="customer_analysis", run_id=result["run_id"])
    except KeyError:
        log.error("scheduled_agent_not_found", agent_id="customer_analyst")
    except Exception as e:
        log.error("scheduled_agent_failed", job="customer_analysis", error=str(e), exc_info=True)


async def heartbeat() -> None:
    try:
        with Session(get_engine()) as session:
            session.exec(select(1))
        log.info("heartbeat", db="ok")
    except Exception as e:
        log.error("heartbeat_db_failed", error=str(e), exc_info=True)


def register_all_jobs(scheduler: AsyncIOScheduler) -> None:
    scheduler.add_job(
        purge_old_context_snapshots,
        CronTrigger(hour=2, minute=0),
        id="purge_context_snapshots",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        run_customer_analysis,
        CronTrigger(day_of_week="mon", hour=8, minute=0),
        id="customer_analysis",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        heartbeat,
        IntervalTrigger(seconds=60),
        id="heartbeat",
        replace_existing=True,
    )
