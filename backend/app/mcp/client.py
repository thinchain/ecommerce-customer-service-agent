"""MCP adapter for the bundled server; a fixed executable/module, never model-selected."""
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.config import PROJECT_ROOT, Settings, get_settings


@asynccontextmanager
async def connect(*, settings: Settings | None = None):
    settings = settings or get_settings()
    env = dict(os.environ)
    env['WEB_SEARCH_PROXY'] = settings.web_search_proxy or ''
    # Explicit configuration must survive the stdio process boundary.
    for name, field in Settings.model_fields.items():
        alias = field.validation_alias
        if not isinstance(alias, str):
            continue
        value = getattr(settings, name)
        if hasattr(value, 'get_secret_value'):
            value = value.get_secret_value()
        env[alias] = str(value).lower() if isinstance(value, bool) else str(value)
    params = StdioServerParameters(command=sys.executable, args=['-m', 'app.mcp.server'],
                                   cwd=str(PROJECT_ROOT / 'backend'), env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=45)) as session:
            await session.initialize()
            yield session


async def invoke(session, name, raw_arguments):
    try:
        result = await session.call_tool(name, json.loads(raw_arguments))
        payload = result.structuredContent
        if isinstance(payload, dict) and isinstance(payload.get('ok'), bool):
            return payload
        return {'ok': False, 'error': 'MCP 工具执行失败或结果格式不合法', 'code': 'mcp_error'}
    except (ValueError, TimeoutError):
        return {'ok': False, 'error': 'MCP 调用参数错误或超时', 'code': 'mcp_error'}
