# 电商客服 Agent

基于 LangGraph、FastAPI 和 Next.js 的电商客服应用，通过业务数据库和知识库回答商品、订单、物流及售后政策问题。协调 Agent 按需委派订单、商品 Agent，支持工具证据校验、会话记忆和 MCP stdio。

本项目使用 **vLLM 部署本地模型**。模型权重由使用者自行下载；也可替换为支持 Function Calling 的其他 OpenAI 兼容模型。与开发配置一致的对话模型为 **Qwen3.5-4B**，决策模型为 **Kev-4B**。

## 环境与模型

应用需要 Python 3.12、Node.js 20.9+。下列模型部署命令在项目根目录的 Linux / CUDA 环境执行；Windows 用户可通过 WSL2 或远程 Linux 部署模型，前后端可在 Windows 本机运行。

| 用途 | 开发使用的模型 | 下载与引用 |
| --- | --- | --- |
| 对话、工具调用与回复生成 | `Qwen/Qwen3.5-4B` | [官方权重与模型卡](https://huggingface.co/Qwen/Qwen3.5-4B) |
| 客服范围分类和领域路由 | `jaredpalmer/kev-4b` | [权重与模型卡](https://huggingface.co/jaredpalmer/kev-4b)、[Kev 源码](https://github.com/jaredpalmer/kev) |
| Kev 的底座 | `Qwen/Qwen3.5-4B-Base` | [官方基础模型](https://huggingface.co/Qwen/Qwen3.5-4B-Base) |

Kev 发布的是基于 Base 模型的 LoRA adapter 和决策 head，需与 `Qwen3.5-4B-Base` 一起导出；对话服务使用的 `Qwen3.5-4B` 是另一份权重。

开发模型环境为 Python 3.12、vLLM `0.29.0`、PyTorch `2.13.0`（CUDA 12.9）、Transformers `5.10.4`、vllm-jev `0.3.6`。模型依赖与应用依赖分别安装。vllm-jev 使用 [上游源码](https://github.com/mode-io/vllm-jev)，具体源码版本固定在 [模型依赖文件](backend/requirements-model.txt)，安装时需要 Git。

### 下载模型

创建并激活模型环境，安装依赖：

```bash
python3.12 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -r backend/requirements-model.txt
```

根据机器的 GPU 和驱动安装对应的 vLLM / PyTorch 构建，参见 [vLLM 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)。下载对话模型；需要启用 Kev 时再下载后两项：

```shell
hf download Qwen/Qwen3.5-4B --local-dir data/models/qwen3.5-4b
hf download jaredpalmer/kev-4b --local-dir data/models/kev-4b
hf download Qwen/Qwen3.5-4B-Base --local-dir data/models/qwen3.5-4b-base
```

### 启动 Qwen3.5

在模型环境中选择 GPU 并启动对话服务：

```bash
export CUDA_VISIBLE_DEVICES=0
vllm serve data/models/qwen3.5-4b --host 127.0.0.1 --port 8001 --served-model-name qwen3.5-4b --dtype bfloat16 --language-model-only --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.40 --kv-cache-memory-bytes 1536M --enable-auto-tool-choice --tool-call-parser qwen3_coder --enforce-eager --api-key local-dev-key
```

示例沿用开发时的上下文、并发和显存预算，显存比例需按机器容量调整。Qwen3.5 的工具调用参数与纯文本模式见 [vLLM 官方部署指南](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)。更换模型时，同步调整工具解析器和应用中的模型名。

### 导出并启动 Kev（可选）

在另一个终端激活模型环境，先将 Base、adapter 和 head 导出为本地 checkpoint：

```shell
python scripts/prepare_kev.py --source data/models/kev-4b --base data/models/qwen3.5-4b-base --output data/models/kev-4b-vllm
vllm-jev serve data/models/kev-4b-vllm --served-model-name kev-4b --host 127.0.0.1 --port 8795 --gpu-memory-utilization 0.30 --max-model-len 4096 --max-num-seqs 4 --enforce-eager
```

导出在 CPU 上执行，生成新目录；启动后通过 `/v1/systemone` 提供决策服务。选择其他 GPU 时，在对应终端设置 `CUDA_VISIBLE_DEVICES`。

## 安装应用

在项目根目录创建应用环境：

```shell
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS 激活：

```bash
source .venv/bin/activate
```

安装应用依赖，将 `.env.example` 复制为 `.env`：

```shell
python -m pip install -r backend/requirements.txt
```

Qwen 服务使用上述命令时，应用配置为：

```dotenv
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
OPENAI_API_KEY=local-dev-key
QWEN_MODEL=qwen3.5-4b
RAG_BACKEND=fts
```

Kev 默认关闭。要复现开发时的分类配置，在 `.env` 中设置：

```dotenv
JEV_BASE_URL=http://127.0.0.1:8795/v1
JEV_MODEL=kev-4b
JEV_MODE=shadow
JEV_SCOPE_ENABLED=true
```

`shadow` 记录路由建议；如需让有效建议参与路由，设为 `advisory`。客服范围分类由 `JEV_SCOPE_ENABLED` 单独控制。模型在另一台机器上时，将两个服务地址改为对应机器的可访问地址。

## 数据来源与重建

`data/` 中包含生成数据、外部问答及运行产物。业务查询使用固定演示用户 `U00001`，仅提供只读查询。

| 内容 | 来源与生成方式 |
| --- | --- |
| 小型商品、用户、订单、物流、支付和退款数据 | [generate_mock_data.py](scripts/generate_mock_data.py) 生成，默认 4 个商品、10 个用户、40 笔订单 |
| 演示商品说明与商店政策 | 仓库中的 [demo.jsonl](fixtures/knowledge/demo.jsonl)，由项目编写的虚拟资料 |
| 开发使用的大型业务库与合成商品知识 | [build_dataset.py](scripts/build_dataset.py) 按固定种子生成，默认增加 320 个合成商品、1,000 个用户、10,000 笔订单 |
| 外部商品问答 | [Amazon-PQA](https://registry.opendata.aws/amazon-pqa/)，由 [fetch_pqa.py](scripts/datasets/fetch_pqa.py) 下载、[import_public.py](scripts/datasets/import_public.py) 转换并保留来源 |
| `memory.sqlite3` | 应用运行时生成的会话和记忆库 |
| `backups/`、`restore-drill/` | 本地数据库副本，可从重建后的数据库重新备份 |

外部 PQA 提供商品标识、描述及问答；对应商品的价格、库存和所有交易记录仍由项目合成。开发使用的 PQA 样本下载于 2026-10-03，取以下四类文件各前 250 条，共 1,000 条：

| 类别 | 公开下载地址 |
| --- | --- |
| Headsets | [amazon_pqa_headsets.json](https://amazon-pqa.s3.amazonaws.com/amazon_pqa_headsets.json) |
| Keyboards | [amazon_pqa_keyboards.json](https://amazon-pqa.s3.amazonaws.com/amazon_pqa_keyboards.json) |
| Earbud headphones | [amazon_pqa_earbud_headphones.json](https://amazon-pqa.s3.amazonaws.com/amazon_pqa_earbud_headphones.json) |
| Portable Bluetooth speakers | [amazon_pqa_portable_bluetooth_speakers.json](https://amazon-pqa.s3.amazonaws.com/amazon_pqa_portable_bluetooth_speakers.json) |

### 小型演示数据

仅体验应用时，执行：

```shell
python scripts/generate_mock_data.py --seed 42 --as-of 2026-10-02
python scripts/ingest_knowledge.py fixtures/knowledge/demo.jsonl
```

默认生成 `data/commerce.sqlite3` 与 `data/knowledge.sqlite3`，知识库含 5 个片段，无需下载外部数据。

### 开发使用的完整数据

下载同一 PQA 样本并校验开发样本的 SHA-256，再构建数据库：

```shell
python scripts/datasets/fetch_pqa.py data/public/pqa-1000.jsonl --per-category 250 --expected-sha256 c5cd0259a433d3719fc16e7a609734918cd972e62f9775e3448a7149fad0f03d
python scripts/build_dataset.py --output data/portfolio --products 320 --users 1000 --orders 10000 --seed 42 --as-of 2026-10-03 --public data/public/pqa-1000.jsonl
```

这会重建开发数据：418 个商品（4 个演示商品、320 个合成商品、94 个 PQA 商品）、1,000 个用户、10,000 笔订单、19,834 条订单明细和 2,088 个知识片段。下载器记录来源 URL、条数和摘要；构建器生成带参数与摘要的 `manifest.json`、知识 JSONL 和检索样例。输出目录用于一个固定版本，参数或数据源变化时使用新目录。

在 `.env` 中切换到完整数据：

```dotenv
COMMERCE_DB_PATH=data/portfolio/commerce.sqlite3
KNOWLEDGE_DB_PATH=data/portfolio/knowledge.sqlite3
```

生成脚本和演示资料随仓库公开；外部问答通过上游地址下载，SQLite 数据库、会话及模型权重在本地生成或保存。

## 启动应用

激活应用环境并完成上述任意一种数据初始化后，启动 API：

```shell
python backend/run.py
```

默认地址为 `http://127.0.0.1:8000`，启用热重载。可通过 `.env` 中的 `API_HOST`、`API_PORT` 或命令行参数调整，例如 `python backend/run.py --port 8080 --no-reload`。数据根目录通过 `DATA_DIR` 配置。

将 `frontend/.env.example` 复制为 `frontend/.env.local`，在另一个终端启动前端：

```shell
cd frontend
npm ci
npm run dev:turbopack
```

打开 `http://127.0.0.1:3000`，可询问“DEMO-RUN 的价格、库存以及 Windows 兼容性”或“查询 ORD00001 的物流和商品当前价格”。前端的后端地址通过 `FASTAPI_BASE_URL` 配置。

## 引用与许可证

- 对话模型：[Qwen3.5-4B 模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)；Kev 底座：[Qwen3.5-4B-Base 模型卡](https://huggingface.co/Qwen/Qwen3.5-4B-Base)。
- 决策模型：Jared Palmer，[Kev 源码](https://github.com/jaredpalmer/kev)及 [Kev-4B 模型卡](https://huggingface.co/jaredpalmer/kev-4b)。
- 推理服务：[vLLM](https://github.com/vllm-project/vllm) 与 [vllm-jev](https://github.com/mode-io/vllm-jev)。
- 商品问答：Rozen, Carmel, Mejer, Mirkis and Ziser，*Answering Product-Questions by Utilizing Questions from Other Contextually Similar Products*，NAACL 2021，[论文](https://aclanthology.org/2021.naacl-main.23/)、[AWS 数据登记页](https://registry.opendata.aws/amazon-pqa/)、[上游说明](https://amazon-pqa.s3.amazonaws.com/readme.txt)。

项目代码采用 [MIT](LICENSE)。模型和外部数据按各自上游许可使用；PQA 登记页列出 CDLA-Permissive-1.0，上游说明另列用途和署名要求，下载使用时一并保留。
