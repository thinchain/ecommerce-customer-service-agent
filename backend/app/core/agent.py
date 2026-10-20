"""Stateless asynchronous agents receive explicit per-run context."""
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import AgentConfig
from app.core.llm import LLM
from app.core.message import AgentEvent, ConversationMessage, ToolExecution


class RunContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1)
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    history: list[ConversationMessage] = Field(default_factory=list)
    memories: dict[str, str] = Field(default_factory=dict)


class AgentResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reply: str = Field(min_length=1)
    trace: list[ToolExecution] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    memories: dict[str, str] = Field(default_factory=dict)
    stopped: bool = False
    decision: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)


class Agent(ABC):
    def __init__(self, name: str, llm: LLM, *, config: AgentConfig | None = None,
                 system_prompt: str | None = None):
        self.name = name
        self.llm = llm
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt

    @abstractmethod
    async def run(self, input_text: str, *, context: RunContext) -> AgentResult:
        """Execute one turn without retaining another session's context."""
        raise NotImplementedError
