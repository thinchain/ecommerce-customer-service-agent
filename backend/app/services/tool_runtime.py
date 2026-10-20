"""One transport session per tool batch, opened/closed within the same graph node task."""
import asyncio
import logging
import time
from contextlib import AsyncExitStack

from app.commerce.tools import BUSINESS_TOOLS
from app.ops.telemetry import TOOL_CALLS, TOOL_LATENCY
from app.core.config import Settings, get_settings


class ToolRuntime:
    def __init__(self, *, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.stack = AsyncExitStack()
        self.session = None

    async def __aenter__(self):
        await self.stack.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self.stack.__aexit__(*args)

    async def execute(self, name, arguments, memories, local_executor):
        label = name if name in BUSINESS_TOOLS or name in {'web_search','calculator','current_time','memory_set','memory_list','memory_delete'} else 'unknown'
        started = time.perf_counter()
        outcome = 'error'
        try:
            result = await self._execute(name, arguments, memories, local_executor)
            outcome = 'ok' if result.get('ok') else 'error'
            return result
        finally:
            TOOL_CALLS.labels(label,outcome).inc()
            TOOL_LATENCY.labels(label).observe(time.perf_counter()-started)

    async def _execute(self, name, arguments, memories, local_executor):
        from app.agents.policy import danger_reason
        import json
        try:
            text = json.dumps(json.loads(arguments), ensure_ascii=False)
        except ValueError:
            text = arguments
        if danger_reason(text):
            return {'ok': False, 'error': '工具请求包含不允许的操作', 'code': 'policy_blocked'}
        if self.settings.commerce_tool_transport == 'mcp' and name in BUSINESS_TOOLS:
            from app.mcp.client import connect, invoke
            try:
                if self.session is None:
                    self.session = await self.stack.enter_async_context(connect(settings=self.settings))
                return await invoke(self.session, name, arguments)
            except Exception as exc:
                # Cancellation is a BaseException and must propagate to release the turn.
                logging.getLogger(__name__).warning('MCP transport failure: %s', type(exc).__name__)
                return {'ok': False, 'error': 'MCP 业务服务暂时不可用，请重试。',
                        'code': 'dependency_unavailable', 'retryable': True}
        # The executor may resolve cached stores via get_settings in its worker thread.
        from app.core.config import override_settings
        with override_settings(self.settings):
            return await asyncio.to_thread(local_executor, name, arguments, memories)
