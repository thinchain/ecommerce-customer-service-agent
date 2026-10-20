"""Separate model protocol messages from persisted UI/audit records."""
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

MessageRole = Literal['user', 'assistant', 'system', 'tool']


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(min_length=1)
    type: Literal['function'] = 'function'
    function: FunctionCall


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True, revalidate_instances='always')
    role: MessageRole
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_role(self):
        if self.tool_calls and self.role != 'assistant':
            raise ValueError('只有 assistant 消息可以包含 tool_calls')
        if self.role == 'tool':
            if not self.tool_call_id or self.content is None:
                raise ValueError('tool 消息必须包含 tool_call_id 和 content')
        elif self.tool_call_id is not None:
            raise ValueError('只有 tool 消息可以包含 tool_call_id')
        if self.content is None and not (self.role == 'assistant' and self.tool_calls):
            raise ValueError('非工具调用消息必须包含 content')
        if len({call.id for call in self.tool_calls}) != len(self.tool_calls):
            raise ValueError('同一消息的工具调用 ID 不能重复')
        return self

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {'role': self.role, 'content': self.content}
        if self.tool_calls:
            result['tool_calls'] = [call.model_dump() for call in self.tool_calls]
        if self.tool_call_id is not None:
            result['tool_call_id'] = self.tool_call_id
        return result


def model_messages(messages: list[Message | dict]) -> list[dict]:
    """Validate complete call/result groups before sending them to a model."""
    result = []
    pending: set[str] = set()
    for item in messages:
        message = Message.model_validate(item)
        if message.role == 'tool':
            if message.tool_call_id not in pending:
                raise ValueError('工具结果没有对应调用，或已经返回过结果')
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ValueError('工具调用结果尚未完整返回')
            pending = {call.id for call in message.tool_calls}
        result.append(message.to_dict())
    if pending:
        raise ValueError('工具调用缺少结果消息')
    return result


class ToolExecution(BaseModel):
    """Execution evidence, distinct from ToolCall. Defaults support legacy records."""
    model_config = ConfigDict(extra='allow')
    step: int = Field(default=0, ge=0)
    name: str
    arguments: str = ''
    result: dict[str, Any] = Field(default_factory=dict)
    agent: str = 'supervisor'


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra='allow')
    agent: str
    task: str = ''
    status: Literal['completed', 'failed', 'partial', 'limit']
    tool_count: int = Field(default=0, ge=0)
    execution_state: str | None = None


class ConversationMessage(BaseModel):
    """In storage, historical tool_calls means evidence, not protocol requests."""
    model_config = ConfigDict(extra='allow', populate_by_name=True)
    id: str | None = None
    role: Literal['user', 'assistant']
    content: str
    timestamp: datetime | None = None
    status: str = 'complete'
    metadata: dict[str, Any] = Field(default_factory=dict)
    tool_executions: list[ToolExecution] = Field(default_factory=list, alias='tool_calls')
    agent_events: list[AgentEvent] = Field(default_factory=list)
    execution_state: str | None = None
    decision: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_storage(cls, record: dict) -> 'ConversationMessage':
        return cls.model_validate(record)

    @classmethod
    def new(cls, role: Literal['user', 'assistant'], content: str, **kwargs) -> 'ConversationMessage':
        return cls(id=str(uuid4()), timestamp=datetime.now(timezone.utc), role=role,
                   content=content, status='complete', **kwargs)

    def to_storage(self) -> dict:
        # Do not invent IDs/timestamps or change absent legacy fields on read.
        return self.model_dump(mode='json', by_alias=True, exclude_unset=True)

    def to_message(self) -> Message:
        return Message(role=self.role, content=self.content)
