# 部署、评测与故障演练

## 数据初始化

从工程根目录运行，默认输出到 `DATA_DIR/portfolio`（未配置时为项目内的 `data/portfolio`）；已有数据不删除。构建后在自己的 `.env` 中配置业务/知识路径。

```bash
.venv/bin/python -m pip install -r backend/requirements-dev.txt
PYTHONPATH=backend .venv/bin/python scripts/datasets/fetch_pqa.py data/public/pqa-1000.jsonl
PYTHONPATH=backend:. .venv/bin/python scripts/build_dataset.py --public data/public/pqa-1000.jsonl
```

下载脚本默认四个品类各250条，有20MB上限，保留 URL、时间和 SHA256；不下载17GB全量语料。已有下载不会覆盖；已完成数据集相同参数重跑直接复用，参数变更需新目录。`manifest.json` 记录规模、版本和数据哈希。公开原文不提交仓库，保留许可与上游声明链接。PQA 中多个用户答案可能有冲突，不应将其视为厂商承诺。

需要配置：

```dotenv
# 以下为默认 DATA_DIR=data 时的示例；使用私有目录时仅修改本地 .env。
COMMERCE_DB_PATH=data/portfolio/commerce.sqlite3
KNOWLEDGE_DB_PATH=data/portfolio/knowledge.sqlite3
```

会话数据默认仍使用原 MEMORY_DB_PATH，不擅自清除历史。会话数据库由 MEMORY_DB_PATH 配置。

## 评测与消融

```bash
PYTHONPATH=backend:. .venv/bin/python scripts/evaluate_portfolio.py --output reports/retrieval-scoped.json
PYTHONPATH=backend:. .venv/bin/python scripts/evaluate_portfolio.py --unscoped --output reports/retrieval-unscoped.json
PYTHONPATH=backend:. .venv/bin/python scripts/evaluate_agent.py --url http://127.0.0.1:8000
.venv/bin/python -B scripts/test_offline.py
```

真实模型未启动时不要拿脚本化测试当作回答准确率。当前商品分支绑定并渲染价格库存，知识分支保留完整来源片段；尚未验证片段是否语义充分支持答案，也不能检测所有自然语言幻觉。

## 监控与限流

已安装 Prometheus 时，可在本机运行 `prometheus --config.file=deploy/prometheus.yml`。
默认抓取 `127.0.0.1:8000`；后端端口变化时同步修改配置。

- `/health/live`：进程存活，不探测模型。
- `/health/ready`：业务/知识库有数据且模型 `/models` 可达；不是完整推理或向量检索探测。
- `/metrics`：HTTP请求与耗时、活动聊天、模型请求/错误/token、工具次数与耗时。
- 响应 `X-Request-ID` 和无正文的结构化日志便于关联请求；标签使用路由模板，避免会话ID带来的高基数。

`MAX_ACTIVE_CHATS=2` 默认无排队准入，饱和立即429；按 GPU 能力调整。`MODEL_TIMEOUT_SECONDS=180`，`MODEL_RETRIES=1`（仅连接失败、408、429、5xx），退避加抖动。`MODEL_CIRCUIT_THRESHOLD=5` 连续瞬态失败后熔断，`MODEL_CIRCUIT_COOLDOWN=30` 秒后允许一个探针，成功恢复。400参数错误不重试。没有重放整个 Agent 回合，因此不会因一次模型重试直接重复业务工具。

## 压测

```bash
PYTHONPATH=backend:. .venv/bin/python scripts/run_load_smoke.py
.venv/bin/locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 --headless -u 2 -r 1 -t 5m --csv reports/chat ChatUser
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
