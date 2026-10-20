"""Standard MCP stdio server. No HTTP port and no arbitrary SQL execution."""
import asyncio
import json

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from app.commerce.tools import BUSINESS_TOOLS
from app.services.tools import TOOL_SCHEMAS, execute_tool

server = Server('commerce-tools')
EXPOSED = set(BUSINESS_TOOLS)


@server.list_tools()
async def list_tools():
    return [types.Tool(name=item['function']['name'], description=item['function']['description'],
                       inputSchema=item['function']['parameters'],
                       annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
            for item in TOOL_SCHEMAS if item['function']['name'] in EXPOSED]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name not in EXPOSED:
        return types.CallToolResult(content=[types.TextContent(type='text', text='未知业务工具')], isError=True)
    result = await asyncio.to_thread(execute_tool, name, json.dumps(arguments, ensure_ascii=False), {})
    return types.CallToolResult(content=[types.TextContent(type='text', text=json.dumps(result, ensure_ascii=False))],
                                structuredContent=result, isError=not result['ok'])


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
