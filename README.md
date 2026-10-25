# 电商客服 Agent

基于 LangGraph、FastAPI 和 Next.js 的电商客服应用，可连接本地 Qwen / vLLM 等支持 Function Calling 的 OpenAI 兼容模型服务，结合业务数据库与知识库回答商品、订单、物流及售后政策问题。

## 项目内容

- **Agent 协作**：协调 Agent 按需串行委派订单、商品两个 ReAct 子图，按角色限制工具权限。
- **业务查询与知识检索**：使用 SQLite 查询订单、价格和库存，通过 FTS5 / BM25 检索商品资料与政策，保留知识来源和引用。
- **回答校验**：价格库存由服务端根据工具证据校验并渲染；混合咨询中保留已成功查询的部分，说明缺失信息。
- **运行支持**：提供会话记忆、执行预算、超时与有限重试，业务工具支持本地调用或 MCP stdio。

项目使用合成业务数据和固定演示用户，业务工具仅提供查询，不执行下单、支付或退款。尚未接入真实用户认证，适合本地演示与开发。

## 快速开始

需要 Python 3.12、Node.js 20.9+，以及已启动的模型服务。模型权重不包含在仓库中；使用已有服务时，应用本身不需要 GPU。

### 1. 安装后端依赖

在项目根目录执行（以下为 Bash 命令）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c backend/constraints-py312.txt -r backend/requirements.txt
cp -n .env.example .env
```

编辑 `.env` 中的模型配置，已有配置无需覆盖：

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
OPENAI_API_KEY=local-dev-key
QWEN_MODEL=qwen3.5-4b
```

地址、密钥和模型名需与实际服务一致；`local-dev-key` 是无鉴权本地服务的占位值。模型服务须启用工具调用支持；可选 Kev 服务默认关闭，无需额外部署。

### 2. 初始化数据并启动后端

```bash
PYTHONPATH=backend:. python scripts/generate_mock_data.py
PYTHONPATH=backend:. python scripts/ingest_knowledge.py fixtures/knowledge/demo.jsonl
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

数据目录通过 `.env` 中的 `DATA_DIR` 配置，默认使用项目内的 `data/`；各数据库也可单独配置，见 `.env.example`。相对路径以项目根目录为基准，自定义私有目录请放在仓库外或加入忽略规则。

上述命令生成小型演示数据；万级合成数据可通过 `scripts/build_dataset.py` 另外构建。

### 3. 启动前端

在另一个终端执行：

```bash
cd frontend
cp -n .env.example .env.local
npm ci
npm run dev
```

打开 <http://127.0.0.1:3000>，可尝试：

- “DEMO-RUN 的价格、库存以及 Windows 兼容性。”
- “查询 ORD00001 的物流和商品当前价格。”

前端默认连接 `http://127.0.0.1:8000`；如需修改，在 `frontend/.env.local` 中设置 `FASTAPI_BASE_URL`。

## 测试

在项目根目录执行：

```bash
source .venv/bin/activate
python -B scripts/test_offline.py
```

离线测试使用临时数据库和 Mock 模型，无需 GPU 或模型服务；测试通过不代表真实模型的回答准确率。

## 许可证

[MIT](LICENSE)
