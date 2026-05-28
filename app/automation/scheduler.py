from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from app.config import get_settings
from app.automation.jobs import register_all_jobs
import structlog

log = structlog.get_logger()

_settings = get_settings()

scheduler = AsyncIOScheduler(
    jobstores={"default": SQLAlchemyJobStore(url=_settings.database_url)},
    timezone="UTC",
)


def job_listener(event) -> None:
    if event.exception:
        log.error("scheduled_job_failed",
                  job_id=event.job_id,
                  error=str(event.exception),
                  traceback=str(event.traceback))
    else:
        log.info("scheduled_job_complete",
                 job_id=event.job_id,
                 retval=str(event.retval))


def start_scheduler() -> None:
    scheduler.add_listener(job_listener, EVENT_JOB_ERROR | EVENT_JOB_EXECUTED)
    register_all_jobs(scheduler)
    scheduler.start()
    log.info("scheduler_started", job_count=len(scheduler.get_jobs()))


def stop_scheduler() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        log.info("scheduler_stopped")
