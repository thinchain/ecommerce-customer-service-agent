"""Asynchronous model contract for agents."""
from typing import Protocol

from app.core.message import Message


class LLM(Protocol):
    async def __call__(self, messages: list[dict], tools: list[dict], *,
                       final: bool = False, required_tool: str | None = None) -> Message: ...
