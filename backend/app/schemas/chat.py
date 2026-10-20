from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from app.core.message import AgentEvent, ToolExecution


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None

    @field_validator('message')
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('问题不能为空')
        return value


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tool_calls: list[ToolExecution]
    memories: dict[str, str]
    stopped_by_limit: bool
    agent_events: list[AgentEvent] = Field(default_factory=list)
    decision: dict = Field(default_factory=dict)
    verification: dict = Field(default_factory=dict)
