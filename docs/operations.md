# 部署、评测与故障演练

## 数据初始化

从工程根目录运行，输出到新的 data/portfolio；已有数据不删除。构建后在自己的 `.env` 中配置业务/知识路径。

```bash
.venv/bin/python -m pip install -r backend/requirements-dev.txt
PYTHONPATH=backend .venv/bin/python scripts/datasets/fetch_pqa.py data/public/pqa-1000.jsonl
PYTHONPATH=backend:. .venv/bin/python scripts/build_dataset.py --public data/public/pqa-1000.jsonl
```

下载脚本默认四个品类各250条，有20MB上限，保留 URL、时间和 SHA256；不下载17GB全量语料。已有下载不会覆盖；已完成数据集相同参数重跑直接复用，参数变更需新目录。`manifest.json` 记录规模、版本和数据哈希。公开原文不提交仓库，保留许可与上游声明链接。PQA 中多个用户答案可能有冲突，不应将其视为厂商承诺。

需要配置：

```dotenv
COMMERCE_DB_PATH=/绝对路径/data/portfolio/commerce.sqlite3
KNOWLEDGE_DB_PATH=/绝对路径/data/portfolio/knowledge.sqlite3
```

会话数据默认仍使用原 MEMORY_DB_PATH，不擅自清除历史。Docker 将会话放在同一个持久卷下的 portfolio/memory.sqlite3。

## Docker

```bash
docker compose config --quiet
docker compose up --build -d
docker compose logs -f api
docker compose --profile observability up -d prometheus
```

默认 seed 生成320个新合成商品和万级订单，不在容器启动时自动联网下载公开语料。公开数据版在空卷中先执行：

```bash
docker compose run --rm --no-deps seed python scripts/datasets/fetch_pqa.py /app/data/pqa-1000.jsonl
docker compose run --rm --no-deps seed python scripts/build_dataset.py --output /app/data/portfolio-public --public /app/data/pqa-1000.jsonl
```

然后将 Compose 的 COMMERCE_DB_PATH、KNOWLEDGE_DB_PATH 指向 portfolio-public，或使用 override 文件。不要用 `down -v` 删除已有数据；命名卷在普通 down/up 后保留。

前端以非 root 用户运行 Next standalone；API 同样非 root。镜像与 CI 使用 backend/constraints-py312.txt 固定已验证的 Python 3.12 依赖版本。镜像不包含 `.env`、本机数据、node_modules 或模型权重。API/metrics 仅在 Compose 内部网络，Web 映射127.0.0.1:3000。Prometheus 可选，映射本机9090。没有登录系统，不应直接暴露公网。

模型服务独立部署，设置 `DOCKER_MODEL_BASE_URL` 为容器可访问的 OpenAI 兼容服务 `/v1` 地址。默认 host.docker.internal 在 Linux 用 host-gateway；**宿主机模型仅绑定127.0.0.1时，桥接容器通常无法访问**，需通过受控网卡地址提供服务或专门的内部模型容器，并用防火墙限制访问。容器内127.0.0.1不是宿主机。代理使用独立 DOCKER_WEB_SEARCH_PROXY，不继承本机 `.env` 的 localhost:10809。

Compose 等待初始化任务成功及 API liveness 后启动网页；模型未就绪时仍可浏览商品，但 readiness 返回503，聊天会明确失败。重启策略只处理进程退出，不等于自动修复不健康服务。镜像运行需在有 Docker 的机器或随附 CI 上验证；配置文件和前端 standalone 构建不等于镜像已验证。

## 评测与消融

```bash
PYTHONPATH=backend:. .venv/bin/python scripts/evaluate_portfolio.py --output reports/retrieval-scoped.json
PYTHONPATH=backend:. .venv/bin/python scripts/evaluate_portfolio.py --unscoped --output reports/retrieval-unscoped.json
PYTHONPATH=backend:. .venv/bin/python scripts/evaluate_agent.py --url http://127.0.0.1:8000
.venv/bin/python -B scripts/test_offline.py
```

真实模型未启动时不要拿脚本化测试当作回答准确率。当前商品分支绑定并渲染价格库存，知识分支保留完整来源片段；尚未验证片段是否语义充分支持答案，也不能检测所有自然语言幻觉。

## 监控与限流

- `/health/live`：进程存活，不探测模型。
- `/health/ready`：业务/知识库有数据且模型 `/models` 可达；不是完整推理或向量检索探测。
- `/metrics`：HTTP请求与耗时、活动聊天、模型请求/错误/token、工具次数与耗时。
- 响应 `X-Request-ID` 和无正文的结构化日志便于关联请求；标签使用路由模板，避免会话ID带来的高基数。

`MAX_ACTIVE_CHATS=2` 默认无排队准入，饱和立即429；按 GPU 能力调整。`MODEL_TIMEOUT_SECONDS=180`，`MODEL_RETRIES=1`（仅连接失败、408、429、5xx），退避加抖动。`MODEL_CIRCUIT_THRESHOLD=5` 连续瞬态失败后熔断，`MODEL_CIRCUIT_COOLDOWN=30` 秒后允许一个探针，成功恢复。400参数错误不重试。没有重放整个 Agent 回合，因此不会因一次模型重试直接重复业务工具。

## 压测

```bash
PYTHONPATH=backend:. .venv/bin/python scripts/run_load_smoke.py
.venv/bin/locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 --headless -u 2 -r 1 -t 5m --csv reports/chat ChatUser
docker compose --profile load run --rm load
```

冒烟脚本只启动自己的临时随机端口 API，压测结束关闭；不会占用8000或停止现有服务。ChatUser每个模拟用户使用独立会话；同会话并发409和准入429应分开统计。

建议阶梯：目录接口并发10/20/50；真实推理并发1/2/4，每阶段5分钟，再做30分钟稳定性测试。验收目标由实际硬件设定，不预先声称满足：例如健康错误率<1%、明确区分429、P95聊天<指定秒数、无OOM或无限排队。报告模型token吞吐与HTTP吞吐，不能互相替代。

## 备份与恢复

```bash
PYTHONPATH=backend:. .venv/bin/python scripts/backup_data.py backup data/portfolio data/backups/run-001
PYTHONPATH=backend:. .venv/bin/python scripts/backup_data.py restore data/backups/run-001 data/restore-drill/run-001
```

使用 SQLite backup API，不直接复制写入中的数据库；验证 integrity_check 和 SHA256。恢复只允许新目录，避免覆盖当前业务；恢复后切换配置并验证订单、记忆、检索。会话若在另一目录，需单独备份该目录。多个数据库的快照并非跨库事务快照；关联写入上线后需暂停写入或统一事务。Chroma 不在此备份范围，从规范化知识JSONL重建。

已提供自动测试：模型503后重试恢复、熔断半开单探针、过载429、旧租约不能覆盖新租约、备份篡改拒绝、专业Agent故障保留证据。实际恢复演练需记录备份时间、最近可恢复数据点（RPO）和恢复验证完成时间（RTO）；本项目没有宣称达到某个生产SLA。
