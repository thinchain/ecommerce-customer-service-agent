# 如何审核客服 Agent 评测

当前 Qwen A/B 和 Kev C/D 各 82 轮，新增多轮工作流三组各30轮，共254条原始结果。它们记录实际请求行为，
不把关键词命中、工具成功或 HTTP 200 直接当作回答正确。结果原件保持不变。

## 打开本地审阅页

当前完整文件为 `reports/evaluation-workflow-complete.html`，包含旧 A/B/C/D 的164条记录和
新 Qwen_only/Kev_route/Kev_scope 的90条记录，以及254条对应 AI 评价。
新测试有真实Qwen初审和Codex逐条复核；原始初审与修改依据均保留，不是人工盲评。
下载这个 HTML 到有浏览器的电脑后直接打开即可。
它包含必要数据，不请求模型、不需要前后端服务、不使用外部 CDN，也不上传审核内容。

从现有记录重新生成（输出文件须不存在）：

```bash
python3 -B scripts/render_evaluation_report.py \
  reports/qwen-ab-run-r1 reports/qwen-cd-run-r1 reports/workflow-live-r4 \
  --dataset reports/workflow-data-r1 \
  --expectations reports/workflow-data-r1/expectations.json \
  --reviews reports/qwen-ab-run-r1/assistant-reviews.jsonl \
  --reviews reports/qwen-cd-run-r1/assistant-reviews.jsonl \
  --reviews reports/workflow-review-r4-v2/final-reviews.jsonl \
  --output reports/evaluation-workflow-complete-v2.html
```

可省略 `--dataset`，此时不会附加原始数据库。指定数据时仅接受哈希校验通过的
合成 `--prepare` 数据，SQLite 以只读方式打开。报告复制测试数据用于离线审核，
与原始结果一样留在被忽略的 `reports/` 中。

## 每条记录的固定格式

1. **输入与上下文**：原始问题、会话 ID、案例/版本/轮次及此前实际对话。
2. **预期与测试条件**：固定案例要求、预期角色和工具依赖、故障注入和运行预算。
3. **实际流程和过程数据**：整轮延迟，各类调用次数和累计墙钟时间；入口和领域决策；
   Agent 委派任务及完成事件；每个工具的参数、完整返回、证据及匹配尝试的耗时；
   Qwen 每次调用的用量/耗时/错误；验证信息、自动检查及完整原始 JSON。
4. **最终输出**：回复原文和 HTTP 状态；单例模式可展开四个版本的回答对照。
5. **AI 评价与复核**：八个评价维度、判断依据、边界案例、来源和完整标注数据。

Kev 不只显示最终选择：每次调用均显示所有选项的概率条、原始置信度、采用标志、
实际入口放行/拦截状态、耗时和返回用量。`confidence` 和选项概率是不同字段。
部分调用已返回 `input_tokens` / `output_tokens`，但旧 `kev_tokens` 汇总不完整；
页面同时保留两者，不将缺失汇总擅自补为完整成本。

点击“连续浏览全部样例”可在一页按同一格式逐条展开，筛选在两种模式下均有效；
“展开当前筛选的全部记录”可同时展开全部案例。“展开本轮全部过程数据”显示该轮
所有已保存细节。所有记录与 AI 初审都已嵌入同一个 HTML，展开无需加载外部文件。
编辑人工复核时切回单例模式，标注仍可按批次导出或导入。

## 指标分别回答什么问题

| 指标 / 字段 | 判断内容 | 局限 |
| --- | --- | --- |
| `http_failures` | 整轮接口是否正常返回 | HTTP 200 中仍可能存在范围拒绝、专业 Agent 失败或模型调用错误 |
| `automatic_checks` | 必需和禁止表达的正则命中 | “支持”也可能出现在“不支持”中；字面匹配不能判断实体、否定或完整性 |
| `machine_catalog_fields` | 从回答中成功解析的价格库存是否符合固定 golden | 分母仅为已解析字段，缺失/未解析不在其中；不是全量回答准确率 |
| `verification` | 服务端引用、字段及证据契约检查 | 属于被测系统自身的检查；不能替代独立答案审核 |
| P50 / P95 | 中位数和约 95% 请求的完成耗时 | 包含失败/拒绝请求；少回答会更快，跨批次受 GPU 负载等影响 |
| Qwen / Kev calls | 实际模型调用次数，包括失败尝试 | 调用少不一定质量好；要结合任务完成率、失败和误拒答 |
| Token 与覆盖数量 | 服务实际返回且完整的用量 | Qwen 与 Kev 分别统计；无用量记未知，不推算显存、能耗或金额 |
| `human_fact_correctness` | 已审核回复是否与资料一致 | 当前未标注时为 null；应同时展示审核覆盖数量 |
| `human_task_success` | 已审核轮次是否完成全部可完成要求 | HTTP 失败应标 false；全体审核后才能视为该集合的任务完成率 |
| `human_false_refusal` | 已审核且适用的轮次中是否误拒绝可回答内容 | true 是坏结果；纯无答案问题可标 null，并在 notes 说明不适用 |
| `human_workflow_correctness` | 已审核执行流程是否符合业务和权限要求 | 与答案事实、完整性分开，不只看工具名称集合 |
| `human_kev_*_correctness` | Kev 范围/领域分类是否合理 | 只审核实际调用和适用问题；当前样例不是完整分类保留集 |

好坏由业务目标和对照决定，不能凭“超过某个百分比”就认定合格。优先关注误拦截、
漏答、错误实体、越权和编造，再在可比质量下比较资源。小集合成绩只能代表该集合。
开发集用来修复和调参，独立保留集用于最终评价；同一问题重复运行并不是更多独立样本。

## 逐条判断回复

1. 把用户问题拆成子任务。例如订单物流、当前售价、库存和 Windows 兼容性分别检查。
2. 对照下方基准数据：订单商品关联、当前价格库存、物流记录和知识原文。
3. 检查商品实体、分/元单位、历史/当前值、否定和前提条件；引用存在不等于支持结论。
4. 判断是否完整：仅贴一大段资料不一定回答了问题；有证据却说资料不足可能属于误拒答。
5. 无资料或工具故障时，检查缺口说明是否准确、其他成功子任务是否保留、是否编造交易成功。

例如 h04 第一轮“DEMO-KEY 支持 macOS 吗”，基准知识明确支持，D 组却按无关请求拒绝。
这是范围误分类和误拒答的明确线索，即使 HTTP 是 200。对单纯拒绝的回复，不要机械地
把“没有虚构数字”算成任务成功；事实一致性不适用时保持 null，并说明原因。

## 判断执行流程

`fixtures/evaluation/qwen-review-expectations.json` 为原始 41 轮提供审核清单，包含必要工具、
预期专业 Agent 和部分依赖顺序。它是在运行后补充的流程审查辅助，不是盲测标签。
只有案例哈希一致时才匹配；自定义案例不能仅凭同一个 case_id 继承清单。

以 o05 为例，合理流程为：

```text
协调 Agent
  → 订单 Agent：order_detail(ORD00001)
  → 从订单取得真实商品 ID
  → 商品 Agent：catalog_search + knowledge_search
  → 合并物流、价格库存与兼容性，逐项保留证据或说明缺口
```

依次审核角色权限、必要工具、数据依赖、参数/实体、故障传播和最终证据使用。
未查询订单就搜索一批耳机，不能视为已完成“订单内这款耳机”的定位。
直接商品咨询通常只需商品 Agent，不应为了固定流程无意义调用订单 Agent。
歧义问题可先澄清，不强制固定工具序列。

页面结构检查只核对角色集合、必要工具是否尝试及部分顺序；工具返回失败也算“尝试”，
是否正确处理失败由人工判断。`agent_events.status=failed` 与工具结果成功可以同时出现，
例如工具拿到数据后，专业 Agent 最后的模型报告失败。两者不能混为一个成功指标。

现有 `tool_calls`、`tool_attempts`、`agent_events` 是已保存的执行记录；step 是各角色内部
轮次，不是全局时间。旧结果没有模型完整上下文；新工作流结果保存真实请求消息、响应、角色及调用偏移。
报告不补造未记录的内部思考或逐节点时间戳。

## 标注、导出与汇总

在页面逐条标注，填写审核者和依据。布尔字段的含义：

- fact_correct / task_completed / workflow_correct / kev_*_correct：true 表示该维度符合要求。
- false_refusal：true 表示发生误拒答。
- null：未审核或该维度不适用；应在 notes 写清不适用原因。

浏览器本地存储仅作临时保存，不能直接修改服务器 JSONL。每个批次分别导出审核文件。
例如导出 `qwen-cd-run-r1-reviews.jsonl`，保存到结果目录后：

```bash
.venv/bin/python -B scripts/evaluate_qwen_ab.py \
  --summarize reports/qwen-cd-run-r1/results.jsonl \
  --reviews reports/qwen-cd-run-r1/qwen-cd-run-r1-reviews.jsonl \
  > reports/qwen-cd-run-r1/summary-reviewed.json
```

命令只读取已有数据，不调用模型。原 `summary.json` 与 `results.jsonl` 不变。
建议隐藏版本做独立审核、记录争议，再比较各版本，避免仅凭偏好判断。

## 四个版本的 AI 初审

对旧 `legacy`（A，证据关闭）和 `evidence`（B，证据开启）82 条记录，已另存
`reports/qwen-ab-run-r1/assistant-reviews.jsonl` 及方法说明、汇总与分析。
这些是运行后查看版本和结果的 AI 初审，`review_source=assistant`；不能作为独立人工盲评成绩。
每条均保留依据、需要人工确认标志及部分边界案例说明。原结果和原汇总保持不变。

C / D 的另外 82 条初审保存在 `reports/qwen-cd-run-r1/assistant-reviews.jsonl`，
方法说明为同目录 `assistant-review-method.json`，汇总为 `assistant-summary.json`。
与 A / B 使用相同的业务事实、可完成子任务、证据保留和流程审核口径；
Kev 分类是否正确独立于置信度和是否采用。初审过程只读取已有数据，不调用被测模型。

生成附带初审结论的审阅页：

```bash
python3 -B scripts/render_evaluation_report.py \
  reports/qwen-ab-run-r1 reports/qwen-cd-run-r1 \
  --dataset reports/qwen-ab-data-r1 \
  --reviews reports/qwen-ab-run-r1/assistant-reviews.jsonl \
  --reviews reports/qwen-cd-run-r1/assistant-reviews.jsonl \
  --output reports/evaluation-complete-v3.html
```

界面区分 AI 初审与人工复核，没有提供审核文件的轮次仍保持未审核。
已有浏览器标注会优先于预载初审，避免覆盖个人审核。人工逐项复核后，再修改审核来源和审核者。
字段 `evidence_supported` 单独评价本轮目标结论是否有有效证据；`citation_correct` 单独评价明确引用。
不能因为价格恰好与基准一致，就认为非法类型或历史观察足以支持当前查询。

汇总 JSON 为兼容旧格式保留 `human_*` 字段名，但应同时检查 `review_sources`。
源为 assistant 的比例只表示 AI 初审结论；无标签的维度仍为 null，不能改称人工准确率。
