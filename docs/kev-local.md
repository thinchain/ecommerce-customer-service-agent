# 本地模型部署（通用配置）

连接已有模型接口时无需运行本页脚本。自行部署前需准备兼容的 vLLM 环境、
合法取得的权重及获授权的 GPU；脚本不安装依赖、不下载权重、不终止占用端口的服务。
具体 CUDA、PyTorch、vLLM 和 vllm-jev 版本应按实际模型验证，不能套用其他机器的路径。

## 配置分层

应用从项目 `.env` 读取配置。模型启动脚本读取 `.env` 和可选 `.env.model.local`，
优先级为：**已设置的 shell 变量 > .env.model.local > .env**。
`.env.model.local` 仅用于启动脚本，不改变后端 Settings 的读取规则。

启动脚本用项目 `.venv/bin/python` 解析 dotenv，再定位模型环境；也可显式设置
`MODEL_CONFIG_PYTHON` 为安装了 python-dotenv 的 Python。dotenv 仅解析字面值，
不执行 Shell 表达式，也不展开 `${VAR}`；请填写完整路径。

在被忽略的 `.env.model.local` 中填写自己的值：

```dotenv
VLLM_ENV=/absolute/path/to/vllm-environment
CUDA_VISIBLE_DEVICES=0
MODEL_PATH=/absolute/path/to/qwen-checkpoint
# KEV_SOURCE_PATH=/absolute/path/to/kev-adapter
# KEV_BASE_PATH=/absolute/path/to/kev-base
# KEV_CHECKPOINT_PATH=/absolute/path/to/new-export
```

GPU 编号只是示例，必须确认资源可用并获授权。脚本不提供个人路径和 GPU 编号默认值。
缺少路径或 GPU 选择时会报错退出。准备 Kev 使用 CPU，不要求选择 GPU。

## Qwen

从项目根目录执行：

```bash
bash scripts/serve_qwen.sh
```

识别 Qwen3.5 文本模型或支持的 Qwen3 AWQ 配置，匹配 dtype 和工具解析器。
默认监听回环地址8001，接口模型名 `qwen3.5-4b`；Qwen3.5 使用 BF16、纯文本模式、
qwen3_coder 解析器、8K 上下文、单序列、固定 KV cache 1536M。
参数可通过 `.env` 或 Shell 覆盖；显存利用率是预算，不是进程间硬隔离。
应用的 `QWEN_MODEL` 和凭据需与服务一致，修改应用配置后重启 API。

## 可选 Kev

```bash
bash scripts/prepare_kev.sh
# 确认导出完成、GPU 可用后：
bash scripts/serve_kev.sh
```

准备时从本地底座提取文本权重，合并 adapter/head，验证后发布新 checkpoint。
不覆盖原始权重；输入变化时选择新的导出目录。离线模式不会获取缺失文件。
导出依赖相应 vllm-jev 接口，需自行验证版本。

应用示例：

```dotenv
JEV_BASE_URL=http://127.0.0.1:8795/v1
JEV_MODEL=kev-4b
JEV_MODE=shadow
JEV_SCOPE_ENABLED=false
```

`JEV_MODE` 控制路由建议；`JEV_SCOPE_ENABLED` 独立控制客服范围分类。
启用范围分类后，Kev 故障可能阻止业务请求，应先验证服务。
本地端点与云端凭据要求不同，真实密钥只放在本地配置中。

## 网络与评测

不要为方便接入直接将无认证模型服务暴露公网。

健康检查和简单生成只证明服务可用，不证明业务事实正确率、路由质量或容量。
真实评测流程见 [Qwen A/B](mixed-business-ab.md)。
个人目录、共享机器端口分配和原始日志保存在被忽略的本地说明中，不随代码公开。
