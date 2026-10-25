# 第二阶段：混合咨询与本地 Qwen A/B

## 修复前审查

下列定位针对第一阶段源码逻辑，相关分支现已修改：

| 问题 | 原代码依据 | 本次处理 |
| --- | --- | --- |
| 价格库存＋兼容性只剩数值 | `catalog_evidence.render_catalog_reply` 末尾提示另行提问；`graph.verify` 在商品渲染后直接返回 | `compose_business_reply` 按 catalog/knowledge/orders 组合，草稿不参与发布 |
| 订单物流＋商品价格丢订单 | 同上，订单 trace 已在主图，却未用于最终输出 | 混合回复显示原订单工具记录；独立订单保持原链路，不新增状态真实性声明 |
| 无 ID 的名称／指代可能绕过 | `needs_catalog_render` 要求 product_context；“轻跑多少钱”“现在价格呢”可不命中；记忆工具曾直接豁免 | 查询价格词直接触发保护；仅继承最近唯一已验证实体，不继承其旧数值 |
| 空知识是否抹掉成功项 | catalog 分支直接返回时知识丢失；其他路径旧验证节点可整体拒答 | 知识空／错实体／失败只产生本节缺口，价格或订单仍显示 |
| 截断依赖 | `execute_tool` 在返回前删除超预算记录；`trace_item` 在此后捕获；`bounded_report` 再裁剪深拷贝 | 明确只信返回窗口，记录 before/retained；快照有内容哈希，拒绝变更；不恢复或发布已删除记录 |
| 开关仍有遗漏路径 | 历史回顾由 `team_agent._execute` 提前返回，未经过图 verify；意图规则也非完备 | 历史数值复用已捕获快照并标历史；当前值不能用旧值；关键词识别和纯订单旧路径仍有限制 |

## 新回复契约

- 价格和库存来自校验字段，零值有效。实体不明确时明确标候选，不把模型选的名称当作可靠实体解析。
- 知识使用完整已返回 chunk，保留否定和条件，不按字符截取到输出预算。超预算时舍弃完整片段并说明缺口。
- 引用检查、商品元数据匹配、原文一致性不等于语义支持度。未知功能可能只有候选资料，需人工判断是否充分回答。
- 知识片段若命中价格库存数值或危险指令规则，整段不发布。这是保守的有限模式过滤，
  不能证明所有语言、单位与隐喻形式均被识别；未知表达仍需人工审查。
- 混合订单输出只展示现有 `order_detail/list_orders` 原始字段的有限映射，不采用专业 summary 或协调草稿。
  原权限查询不变；`state_consistency=not_measured`，没有验证状态跨表一致性、到账、预计到达承诺。
- 各部分元信息在 `verification.parts`；`verification.catalog` 保留兼容。
- `result_sha256` 是一致性检查，不是外部真实性或恶意进程防篡改认证。
- 没有添加 Agent、图节点、推理验证调用或新数据库。生产数据、模型权重没有修改。

## 离线测试

```bash
.venv/bin/python -B scripts/test_offline.py tests.test_business_reply tests.test_qwen_ab
.venv/bin/python -B scripts/test_offline.py
```

模型全为 Mock，SQL 在临时目录，MCP 为本地 stdio。评测运行器的 HTTP 路由测试也使用
Mock 模型和 Mock `/models` 响应；这不是一次真实 Qwen 测试。临时结果退出即删除。

## 固定案例与数据

`fixtures/evaluation/qwen-ab-cases.jsonl` 包含37条固定案例、41轮请求（4个多轮 episode），涵盖目录、
名称、多商品、订单联合、指代、RAG、无答案、实体错误、部分故障。

`--prepare` 创建全新合成目录：复用演示商品和知识，生成固定种子的订单；额外添加
零值与无知识商品，固定 ORD00011 属于 U00002。不会读取项目 `data/` 或复制真实会话。
保存数据库和案例 SHA256；运行前、episode 后检查。已存在目录拒绝覆盖。

仅准备数据，不访问模型：

```bash
.venv/bin/python -B scripts/evaluate_qwen_ab.py --prepare /tmp/ecommerce-qwen-ab-v2
```

## 获得资源使用授权后的运行

运行器不启动 GPU 服务，不下载模型。只有操作者确认 GPU 资源及获得使用授权后，才执行：

```bash
.venv/bin/python -B scripts/evaluate_qwen_ab.py \
  --run /tmp/ecommerce-qwen-ab-v2 \
  --output /tmp/ecommerce-qwen-ab-v2-results \
  --base-url http://127.0.0.1:8001/v1 \
  --model qwen3.5-4b --temperature 0.3 --max-tokens 1200 \
  --repeats 3 --allow-model-requests
```

只允许回环 HTTP `/v1`，远程／付费 API 地址被拒绝；密钥从 `OPENAI_API_KEY` 读取但不记录。
已运行的服务若配置不同密钥，应先在环境中设置。`/models` 只核对接口模型名，不能证明
权重未被同名替换。需另记录实际权重、vLLM 启动参数、GPU 和负载，并在实验期间保持稳定。

A/B 为同一源码的开关对照：legacy=false，evidence=true；不是旧 commit 二进制复现。
两组同 endpoint/model/temperature/max_tokens/预算，关闭 Kev，使用 FTS、本地工具。
按 episode 交错 AB/BA，默认3次重复、串行执行；每组每案例每次重复独立数据库副本和 UUID。
同 episode 多轮共享自己的会话。模型客户端／熔断在 episode 边界重置。
工具故障通过固定标记注入，两个版本一致，不修改数据库；真实模型仍负责选择和调用工具。

该对照覆盖实际 FastAPI ASGI `/api/sessions` 和 `/api/chat`、LangGraph、工具和本地 Qwen。
延迟包括应用和推理，不含浏览器、Next 代理及网络入口。共享 GPU 上其他请求、模型随机性、
失败后的远端遗留计算会影响延迟，不能据此认定生产容量或多 Agent 优于单 Agent。

## 结果和审核

- `run.json`：共同配置、执行预算、数据和源码哈希、接口模型列表。
- `results.jsonl`：case/repeat/variant/turn 的稳定 ID、问题、回答、API 状态、工具调用、
  工具尝试（含最终API失败时的诊断）、验证信息、每次模型耗时和 usage、整轮延迟、固定 golden、人工审核字段。
- `summary.json`：两组延迟 P50/P95、逐案例配对的延迟差、HTTP 失败、可用 Token 覆盖率、已审核质量率。
- `episodes/`：独立合成数据和测试会话，便于复现 Bad Case，不含生产数据。

Token 使用 SDK 返回的 usage；缺失记 null。失败请求的远端消耗可能未知，不编造总成本。
正则断言只作筛查；数值核对从回答文本与固定 golden 比较，不读取新验证器判定。
不能解析的自然语言数值记未知；不能用解析成功子集的正确率冒充全部回答准确率。
额外事实、否定、实体指代、片段相关性、订单结论和任务完整性均需人工审核。

审核者填写 `manual_review.fact_correct/task_completed/false_refusal/reviewer/notes/unsupported_claims`；
`subtasks` 可逐项写任务与完成情况。误拒答指本可由固定资料回答的子问题被拒绝，
不是证据确实缺失时的合理拒答；混合回答中某一子问题的误拒答也计入。摘要报告已审核样本中的比例。
可复制完整结果或只写 `{id, manual_review}` 到新 reviews.jsonl。建议隐去 variant 进行盲审。
未审核时质量率为 null，而不是0或100%。审核后执行：

```bash
.venv/bin/python -B scripts/evaluate_qwen_ab.py \
  --summarize /tmp/ecommerce-qwen-ab-v2-results/results.jsonl \
  --reviews /tmp/ecommerce-qwen-ab-v2-results/reviews.jsonl
```

事实一致率按人工判断的已审核回答统计，任务完成率要求回答所有可完成子问题并准确说明缺口。
应一起报告审核覆盖率、错误放行、误拒答、子任务覆盖和 Bad Case；不能靠全拒答获得漂亮准确率。
先使用固定案例作基线，再增加人工真实风格问题与单 Agent 基线，才讨论架构收益。
