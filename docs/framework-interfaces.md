# Message、Config 与 Agent 接口

业务执行继续使用现有 LangGraph。`app.core` 提供统一消息、配置、模型及 Agent 契约，`LangGraphTeamAgent` 实现异步执行，`services/agent.py` 管理会话占用与整轮提交。

## 消息与会话记录

`Message` 表示模型协议消息，角色限定为 `system/user/assistant/tool`。函数调用使用 `ToolCall`，工具结果必须携带对应的 `tool_call_id`。助手发出调用时允许 `content=None`；普通消息和工具结果需要字符串内容。

```python
from app.core import Message
from app.core.message import model_messages

messages = model_messages([
    Message(role="user", content="计算 1+1"),
    Message(role="assistant", tool_calls=[{
        "id": "calc-1", "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
    }]),
    Message(role="tool", tool_call_id="calc-1", content='{"ok":true,"result":2}'),
])
```

`model_messages()` 校验调用与结果完整配对，拒绝孤立、重复或缺少结果的调用组。上下文仍按完整组裁剪。`Message.to_dict()` 仅输出模型协议字段，内部时间戳和元数据不会发送给模型。

`ConversationMessage` 表示持久化问答记录。其 `tool_executions` 是实际执行证据，与模型请求的 `tool_calls` 类型不同。存储及现有 API 继续使用历史键名 `tool_calls`。通过 `to_message()` 转换历史时，只提取角色和内容；工具执行轨迹不会变成新的函数调用。

新记录由 `ConversationMessage.new()` 生成 ID 和 UTC 时间戳。旧记录可缺少 ID、时间戳、协作事件及核验字段，读取时不会伪造历史时间，也会保留额外审计字段。SQLite 表结构无需迁移。

## 配置与覆盖

`Settings` 使用 `pydantic-settings`，保留 `.env.example` 中现有环境变量名，按模型、Agent、存储、检索、搜索和 Jev 分组定义字段。加载优先级为显式参数、进程环境、根目录 `.env`、默认值。部署脚本使用的其他 `.env` 字段被忽略。

```python
from app.core import Settings, get_settings, override_settings

settings = get_settings()                    # 默认配置缓存一次
limits = settings.agent                      # 独立的 AgentConfig
isolated = Settings.from_env(env_file=None)   # 不读 .env，仍读取进程环境

with override_settings(agent_max_steps=2, jev_scope_enabled=False):
    assert get_settings().agent.max_steps == 2
```

配置为不可变对象，超限、非法枚举、非有限超时及不完整 Chroma 配置会在加载时失败，不再静默截断。密钥使用 `SecretStr`，日志/JSON 默认脱敏。FastAPI 启动时验证配置；修改环境配置后需重启服务。

新增环境变量：`MODEL_TEMPERATURE`、`MODEL_MAX_TOKENS`、`AGENT_MAX_HISTORY_TURNS`、`AGENT_HANDOFF_REPORT_BYTES`、`AGENT_MAX_CALLS_PER_RESPONSE`。默认值沿用原有设置。

`override_settings()` 接受字段名或环境变量别名，在当前任务中覆盖配置，退出后恢复；覆盖也会验证。配置通过 ContextVar 传播到异步子任务和 `asyncio.to_thread()`，MCP 子进程显式接收当前配置。不同配置使用各自的模型客户端、熔断器和存储缓存。

测试使用 `ConfiguredTestCase/ConfiguredAsyncTestCase` 注入默认配置，避免开发者 `.env`、代理和 Jev 服务影响回归测试。旧常量导入继续兼容脚本，应用代码通过 `get_settings()` 或显式实例读取配置。

## 异步 Agent

```python
from app.core import Agent, AgentConfig, AgentResult, RunContext

class EchoAgent(Agent):
    async def run(self, input_text: str, *, context: RunContext) -> AgentResult:
        return AgentResult(reply=input_text, memories=dict(context.memories))

agent = EchoAgent("echo", llm, config=AgentConfig(max_history_turns=50))
result = await agent.run("你好", context=RunContext(session_id="session-id"))
```

`Agent` 定义名称、模型、提示词和执行配置。`RunContext` 提供 `session_id/run_id`、类型化历史和当前记忆；`AgentResult` 提供最终回复、执行轨迹、协作事件、记忆、预算标识、决策及核验结果。Agent 不持有共享 `_history`，实现应把每轮状态保留在调用内。

`LLM` 是异步可调用协议，接受模型消息、工具 schema 和工具选择约束，返回 `Message`。现有 `qwen.complete()` 已实现此契约。共享服务错误移至 `core/exceptions.py`，旧导入路径仍可使用。

`LangGraphTeamAgent` 包装协调图及专业子图，图在实例初始化时编译，可复用同一实例处理不同会话。提示词、模型、执行器、配置均可注入。范围分类、历史回顾和图执行共享整轮超时。

`run_agent(question, session_id, store, agent=..., settings=...)` 是应用入口。它取得会话租约、构造上下文、调用 Agent，成功后一次性提交最终历史与记忆；失败或取消不会提交。历史上限按完整问答轮次计算，租约时长覆盖注入 Agent 的执行时限。已有聊天 API 和前端字段保持兼容。

## 验证

```bash
PYTHONPATH=backend:. .venv/bin/python -m unittest discover -s backend/tests -v
```

测试覆盖协议与存储序列化、配置优先级与并发隔离、共享 Agent 并发调用、准入超时回滚、可配置调用批次预算、真实 LangGraph/MCP 和 Chroma 存储。模型 SDK 调用使用模拟 HTTP；这些测试不衡量真实模型的回答质量。
