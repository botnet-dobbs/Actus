from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field
import uuid


class ContextualData(BaseModel):
    type: str
    object_ids: list[int]
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: list[dict] = []


class ContextualLogic(BaseModel):
    name: str
    description: str
    priority: int = 0


class ContextualAction(BaseModel):
    tool_name: str
    description: str
    parameters: dict[str, Any] = {}


class AgentContext(BaseModel):
    agent_id: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: list[ContextualData] = []
    logic: list[ContextualLogic] = []
    actions: list[ContextualAction] = []
    metadata: dict[str, Any] = {}
    assembled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600
