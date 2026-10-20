# 商品查询证据回归

问题：介绍 SYN-CAM-0015 的适用场景，并查询价格和库存。
原会话没有工具调用，回答却包含 299 元、库存 15 件及自动对焦等虚构信息；
Kev shadow 判断为 product，只记录建议。旧校验只检查 KB 引用 ID，
无法识别伪造的 `[catalog_search]` 来源。

修复：

- 根据明确的商品查询词识别必需工具，首轮指定 delegate_product。
- 商品 Agent 对尚未尝试的必需工具指定调用；失败不自动重复强制请求。
- 最终校验缺失的成功查询和伪造工具名引用，缺证据时拦截回答。
- catalog_search 的 product_ids 优先于 query 名称文本，价格和库存条件仍生效。
- FTS 对已有 product_id 的关键词漏检回退到该 ID、该 kind 的文档，仍保留来源和引用；
  回退结果 rank 为 null，不声称具有关键词相关度。

这些规则覆盖明确的商品查询，不是通用语义真实性判定；
verification.semantic_factuality 仍为 not_measured。
历史问题的指代解析、复杂请求分类和取得证据后的逐句语义一致性仍需进一步评估。

验证：69 项后端测试通过；真实 Qwen + Kev 和本地数据回归调用
knowledge_search、catalog_search，回答 1167.66 元、库存 23，
视频会议/USB/Windows/macOS/Linux、不支持自动对焦、不支持独立录像存储。
历史原始结果与日志保存在本地 `reports/product-evidence/`，不随公开仓库发布。测试会话使用临时数据库，未改写用户历史。

生效：停止原 FastAPI 进程后重新启动。建议新建会话复测，避免旧错误回答影响上下文。

```bash
# 从项目根目录运行
.venv/bin/uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```
