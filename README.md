# 电商客服 Agent

基于 LangGraph、FastAPI 和 Next.js 的电商客服应用，连接支持 Function Calling 的 OpenAI 兼容模型服务，通过业务数据库和知识库回答商品、订单、物流及售后政策问题。

- 协调 Agent 按需委派订单、商品 Agent，按角色分配工具。
- SQLite 提供业务查询，FTS5 / BM25 或 Chroma 提供知识检索。
- 根据工具证据校验价格、库存和引用，由模型组织回复。
- 支持会话历史、显式记忆、执行预算和 MCP stdio 工具调用。

示例使用合成数据和固定用户 `U00001`，业务工具提供只读查询。

## 本地运行

需要 Python 3.12、Node.js 20.9+，以及自行部署的模型服务。

### 后端

在项目根目录创建虚拟环境：

```shell
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

安装依赖，将 `.env.example` 复制为 `.env`，按实际模型服务填写 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和 `QWEN_MODEL`：

```shell
python -m pip install -r backend/requirements.txt
```

初始化示例数据并启动：

```shell
python scripts/generate_mock_data.py
python scripts/ingest_knowledge.py fixtures/knowledge/demo.jsonl
python backend/run.py
```

API 默认地址为 `http://127.0.0.1:8000`，默认启用热重载。可通过 `.env` 中的 `API_HOST`、`API_PORT` 或命令行参数调整，例如 `python backend/run.py --port 8080 --no-reload`。数据默认保存在 `data/`，可通过 `DATA_DIR` 修改。

### 前端

将 `frontend/.env.example` 复制为 `frontend/.env.local`，在另一个终端执行：

```shell
cd frontend
npm ci
npm run dev:turbopack
```

打开 `http://127.0.0.1:3000`，可询问“DEMO-RUN 的价格、库存以及 Windows 兼容性”或“查询 ORD00001 的物流和商品当前价格”。后端连接地址通过 `frontend/.env.local` 中的 `FASTAPI_BASE_URL` 配置。

模型服务、服务器部署和数据接入由使用者自行配置；可选 Kev 服务默认关闭。

## 许可证

[MIT](LICENSE)
