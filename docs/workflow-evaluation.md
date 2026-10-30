# 真实模型多轮与流程评测

本测试只连接已经部署的 Qwen 和 Kev，不启动或下载模型，不预设模型返回，不注入
工具故障，不执行生成的代码。业务数据来自独立合成副本，不读取应用的业务数据库。
低风险边界用例包括打印 hello 的编程请求、写诗、身份冒充、他人合成订单查询和记忆污染。

## 运行

在项目根目录配置好 `.env` 的 Qwen/Kev 地址、模型名与凭据后：

```bash
.venv/bin/python -B scripts/evaluate_workflow.py --prepare reports/workflow-data-r1
.venv/bin/python -B scripts/evaluate_workflow.py \
  --run reports/workflow-data-r1 --output reports/workflow-live-r1 \
  --repeats 1 --allow-model-requests
```

数据和输出目录必须为新目录，已有结果不覆盖。端点限制为本地服务，凭据不写入报告。
准备数据不请求模型；`--run` 会实际消耗已部署服务的推理资源。
可以用 `--cases w-price w-history` 做子集冒烟；完整评测不要仅挑成功案例。

三组使用相同数据和预算、独立会话及数据库副本，按案例交替执行：

| 组 | 协调与业务流程 | Kev 范围分类 |
| --- | --- | --- |
| Qwen_only | Qwen 协调器与专业 LangGraph 子图 | 关闭 |
| Kev_route | 真实领域建议；高置信且实体/角色匹配时直接进入专业子图，否则回退协调器 | 关闭 |
| Kev_scope | 与 Kev_route 相同 | 开启，低置信拒绝回退给主流程 |

三组均启用实体状态、订单依赖保护、参数绑定、按需输出和紧凑历史回顾。
因此这组对照评估优化后流程中的 Kev 接入差异，不是未经修改代码的前后对照。
旧 A/B/C/D 结果仍保留，可作历史参考，不把不同测试集的比例直接当作准确率提升。

阈值默认仍为0.8。`JEV_ADVICE_THRESHOLD`控制建议采用，`JEV_REJECT_THRESHOLD`
控制范围拒绝。低置信回退不授予权限，工具依然按固定用户过滤。
`KEV_DIRECT_ROUTING`控制已验证路由是否绕过协调模型；关闭时仍可记录建议。

## 记录与审核

`results.jsonl` 每行是一轮，包含原始输入、会话、最终输出、角色任务/事件、工具实际
参数与完整返回、证据观察、Qwen 每次真实输入消息/工具 schema/响应、Token/耗时、
Kev 请求/选项/概率/置信度/响应/用量，以及本轮提交后的显式记忆。
参数绑定会分别保留模型请求的参数和实际执行参数，不能将绑定后的参数伪称模型原始输出。
自然表达默认开启，`verification.presentation`记录生成/复用/回退及新增模型调用；
新增调用的`agent_context`为`responder/presentation`，参与耗时和token统计。
它不是业务查询角色，结构检查仍核对订单和商品Agent；引用编号由校验后的来源附加。
没有保存的节点时间不补造，调用耗时不当作 GPU 计算时间或能耗。

`w-history` 在明确标注的轮次把独立副本 RUN 价格从299元变为309元；
历史回顾应使用已有观察，实时重查必须取得309元。变更账本随结果保存，原始基准不变。
历史回复不能从其他商品或未校验模型文字中推导价格。

关键词检查只用于初筛，不证明事实、完整性、权限或否定关系正确。
AI 初审另存为 `assistant-reviews.jsonl`，标明 source=assistant、审核依据和需人工确认，
不改原始结果，不称为人工盲评。按完整会话统计调用量/Token/完成率，再比较历史回顾收益。

使用真实本地模型审核已保存结果（另有推理开销）：

```bash
.venv/bin/python -B scripts/review_workflow.py \
  --run reports/workflow-live-r4 --dataset reports/workflow-data-r1 \
  --output reports/workflow-review-new --allow-model-requests
.venv/bin/python -B scripts/summarize_workflow.py \
  --run reports/workflow-live-r4 \
  --reviews reports/workflow-review-new/assistant-reviews.jsonl \
  --output reports/workflow-review-new/session-metrics.json
```

`judge-calls.jsonl`保留评审模型的实际输入、输出、用量与耗时。评审模型与被测模型
使用同一服务，存在自评偏差。无效评审最多重试一次，仍无效就保留null，不预设通过。
最终报告应明确区分模型初审与后续人工/其他AI复核的修订；原评审返回不覆盖。

生成包含旧结果和新结果的 HTML，可重复指定 `--reviews` 和 `--expectations`；
新的流程清单位于准备目录 `expectations.json`。输入/输出消息只包含本次合成测试数据，
原始报告保留在被 Git 忽略的 reports 中，不自动公开。

## 本次完成的结果

最终真实运行是 `reports/workflow-live-r4`，16个会话案例、每组30轮，共90轮。
前面的 smoke/r1/r2/r3 是修复期间的开发记录，不能合并充当独立保留集样本。
本地Qwen初审位于 `reports/workflow-review-r4-v2/assistant-reviews.jsonl`；
`final-reviews.jsonl` 是Codex查看实际回复、工具和基准后逐条复核的结果，
`adjudications.jsonl` 明确保留每个修改的维度和理由。二者都是AI评价，仍需人工确认。
模型原始判断和实际评审请求响应没有覆盖。

汇总为 `reports/workflow-review-r4-v2/session-metrics-v2.json`，分析为同目录 `analysis.md`。
Kev返回 `input_tokens/output_tokens`，补充汇总保留这些实际字段并单独求和，
不能因为旧版只识别 `prompt_tokens/completion_tokens` 就称Kev成本为零。
完整审阅页为 `reports/evaluation-workflow-complete.html`，含旧164轮和本次90轮。
