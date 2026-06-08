from typing import cast
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlmodel import Session, select
from app.database import get_session
from app.ontology.models import OntologyObjectBase
from app.ontology.registry import get_type, list_types
from app.auth.models import User
from app.auth.jwt import get_current_user, write_audit_log
from app.rag.indexer import delete_from_index, index_object
import structlog

log = structlog.get_logger()
router = APIRouter()


@router.get("/types")
def get_ontology_types():
    return {"types": list_types()}


@router.get("/objects/{type_name}")
def list_objects(
    type_name: str,
    session: Session = Depends(get_session),
    limit: int = 50,
    offset: int = 0,
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    results = session.exec(
        select(cls).where(cls.is_deleted == False).offset(offset).limit(limit)
    ).all()
    return {"type": type_name, "count": len(results), "items": results}


@router.get("/objects/{type_name}/{object_id}")
def get_object(
    type_name: str,
    object_id: int,
    session: Session = Depends(get_session),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    return obj


@router.post("/objects/{type_name}", status_code=201)
def create_object(
    type_name: str,
    payload: dict,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    unknown = set(payload.keys()) - set(cls.model_fields.keys())
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown fields: {unknown}. Valid: {list(cls.model_fields.keys())}",
        )
    try:
        obj = cls(**payload)
    except (ValidationError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    obj.created_by = user.id
    try:
        session.add(obj)
        session.commit()
        session.refresh(obj)
    except Exception as e:
        session.rollback()
        log.error("ontology_create_failed", type=type_name, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    background_tasks.add_task(index_object, type_name, obj.id, obj)  # pyright: ignore[reportAttributeAccessIssue]
    ip = request.client.host if request.client else None
    write_audit_log(username=user.username, action="ontology_create", resource=f"{type_name}:{obj.id}", ip=ip)  # pyright: ignore[reportAttributeAccessIssue]
    log.info("ontology_object_created", type=type_name, id=obj.id, by=user.username)  # pyright: ignore[reportAttributeAccessIssue]
    return obj


@router.put("/objects/{type_name}/{object_id}")
def update_object(
    type_name: str,
    object_id: int,
    payload: dict,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    unknown = [k for k in payload if k not in cls.model_fields]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown fields: {unknown}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    try:
        for key, value in payload.items():
            setattr(obj, key, value)
        obj.touch()
        session.add(obj)
        session.commit()
        session.refresh(obj)
    except Exception as e:
        session.rollback()
        log.error("ontology_update_failed", type=type_name, id=object_id, error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    background_tasks.add_task(index_object, type_name, obj.id, obj)  # pyright: ignore[reportAttributeAccessIssue]
    ip = request.client.host if request.client else None
    write_audit_log(username=user.username, action="ontology_update", resource=f"{type_name}:{object_id}", ip=ip)
    log.info("ontology_object_updated", type=type_name, id=object_id, by=user.username)
    return obj


@router.delete("/objects/{type_name}/{object_id}", status_code=204)
def delete_object(
    type_name: str,
    object_id: int,
    request: Request,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
):
    try:
        cls = cast(type[OntologyObjectBase], get_type(type_name))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown type: {type_name}")
    obj = session.get(cls, object_id)
    if not obj or obj.is_deleted:
        raise HTTPException(status_code=404, detail="Object not found")
    obj.soft_delete(deleted_by=user.id)
    session.add(obj)
    session.commit()
    background_tasks.add_task(delete_from_index, type_name, object_id)
    ip = request.client.host if request.client else None
    write_audit_log(username=user.username, action="ontology_delete", resource=f"{type_name}:{object_id}", ip=ip)
    log.info("ontology_object_deleted", type=type_name, id=object_id, by=user.username)
