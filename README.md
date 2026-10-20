# 电商客服 Agent

基于 **FastAPI、Next.js、LangGraph 和 OpenAI 兼容模型接口**的电商客服工程样本。
使用 SQLite 合成订单、商品检索和带来源的 RAG，演示多 Agent 协作、工具权限、
证据校验、失败恢复和评测。当前为固定演示身份的只读沙盒，不执行真实交易。

## 能力与边界

- 协调 Agent 按需委派订单、商品、资讯子图，共享模型并控制执行预算。
- 商品价格 `price_cents` 和库存 `stock` 由服务端绑定工具证据并渲染，零值有效。
- 混合咨询按部分保留商品、知识摘录和原订单记录，证据不足只说明对应缺口。
- 知识保留来源与完整返回片段；来源有效不等于语义结论已获验证。
- 支持本地工具或 MCP stdio、会话记忆、超时、限流、熔断及监控。
- 尚无用户登录或生产租户认证；请保持本地访问，不直接暴露公网。

## 快速开始

要求 Python 3.12、Node.js 20.9+；运行离线测试不需要 GPU 或模型服务。
以下命令均在项目根目录执行，Windows 用户可替换虚拟环境的激活及可执行文件路径。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c backend/constraints-py312.txt -r backend/requirements-dev.txt
cp -n .env.example .env
python -B scripts/test_offline.py
```

已有 `.env` 不要覆盖。编辑它以连接自己的 OpenAI 兼容模型服务：

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
OPENAI_API_KEY=local-dev-key
QWEN_MODEL=qwen3.5-4b
```

`local-dev-key` 只是无鉴权本地服务的开发占位值。启用鉴权时填自己的凭据，
并与模型服务配置匹配。接口模型名必须与服务实际公布的名称一致。
可选 Kev 默认关闭；仅使用 Qwen 时无需部署 Kev。模型不随仓库分发。

### 初始化小型演示数据

```bash
PYTHONPATH=backend:. python scripts/generate_mock_data.py
PYTHONPATH=backend:. python scripts/ingest_knowledge.py fixtures/knowledge/demo.jsonl
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

默认数据位于项目 `data/`，已被忽略。配置自己的路径前先确认不会指向真实数据库。
大规模合成数据及可选公开语料见[数据管线](docs/public-datasets.md)与[运维说明](docs/operations.md)。

### 启动前端

在另一终端运行：

```bash
cd frontend
cp -n .env.example .env.local
npm ci
npm run dev
```

访问 `http://127.0.0.1:3000`。Next.js 将 `/api` 转发给默认的本地 FastAPI；
使用其他后端时在 `frontend/.env.local` 配置 `FASTAPI_BASE_URL`。
`NEXT_PUBLIC_SHOW_DEV_DETAILS` 控制诊断面板展示，不提供访问控制。
`NEXT_PUBLIC_*` 会进入浏览器，禁止放凭据。生产环境变更构建期配置需重新构建。

可尝试：“DEMO-RUN 价格库存以及 Windows 兼容性”“ORD00001 的物流与商品当前价格”。

## 模型部署与 Docker

应用可连接已有模型服务，不会自动启动 GPU、下载权重或终止其他进程。
自行部署 Qwen/Kev 见[模型部署](docs/kev-local.md)，需显式指定个人路径及获授权的 GPU。
Docker 使用方式见[运维说明](docs/operations.md)，容器模型地址需实际可达。

## 项目结构与改造入口

| 目录 | 用途 |
| --- | --- |
| `backend/app/agents/` | 图、角色、商品证据与混合回答组合 |
| `backend/app/commerce/` | 业务数据与工具参数模型 |
| `backend/app/knowledge/` | FTS / 可选向量检索 |
| `backend/app/core/config.py` | 集中配置与环境变量 |
| `backend/tests/` | 单元和 Mock 集成测试 |
| `frontend/` | Next.js 页面和 API 转发 |
| `fixtures/` | 小型合成资料与固定评测案例 |
| `scripts/` | 数据、部署、测试、评测及发布检查 |

添加业务工具时修改 commerce 工具注册和相应角色白名单；修改回答契约时同步验证节点和测试。
不要从模型摘要提取权威价格库存。订单和知识尚未获得完整确定性事实保证。

## 测试与评测

```bash
python -B scripts/test_offline.py
cd frontend
npx tsc --noEmit
npm run build
```

离线运行器将 SQLite 限制在临时目录并禁止主进程 TCP；MCP 使用临时数据和本地 stdio。
模型全为 Mock，离线通过数量不代表真实模型正确率。

真实 Qwen A/B 已准备 37 个固定案例、41 轮请求，覆盖多轮与部分失败。
准备数据不调用模型；实际推理需另行确认资源及授权，见[混合咨询与 A/B](docs/mixed-business-ab.md)。
[报告索引](reports/README.md)区分历史结果、原始本地记录与公开材料。

## 本地开发与公开发布

- `README.md`：随代码公开的说明。
- `README.local.md`：个人操作记录，可自行维护，Git/Docker 均忽略；新下载的代码不包含此文件。
- `.env`、`.env.model.local`、`frontend/.env.local`：本地配置；公开仅保留各目录 `.env.example`。
- `reports/` 默认忽略原始记录；仅 `README.md` 和经过审核的 `public/` 可上传。

检查或导出一份独立公开目录：

```bash
python3 -B scripts/check_public_files.py
python3 -B scripts/export_public.py /tmp/ecommerce-agent-public
```

导出目录必须不存在且位于项目外；源文件不删除，Git 历史不复制，不自动提交或推送。
检查是有限规则扫描，发布前仍需人工审核。详见[发布指南](docs/public-release.md)和[贡献说明](CONTRIBUTING.md)。

## 更多设计文档

- [多 Agent 拓扑与预算](docs/multi-agent.md)
- [业务、RAG、MCP 与记忆](docs/ecommerce-architecture.md)
- [商品结构化证据 MVP](docs/catalog-evidence-mvp.md)
- [接口与配置](docs/framework-interfaces.md)
- [身份与客服范围](docs/customer-policy.md)
- [项目论证与评测边界](docs/portfolio.md)
