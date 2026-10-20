# 公开数据导入

代码不自动下载或再分发公开数据。请保留原始来源说明及许可文件，先选择小规模商品领域样本。

- ESCI：使用 [Amazon Science 官方项目](https://github.com/amazon-science/esci-data)，商品字段含 product_id/product_locale/product_title/product_description/product_bullet_point/product_brand。项目标注 Apache-2.0。不要把查询相关性标注文件当商品描述表。
- PQA：以 [AWS Registry](https://registry.opendata.aws/amazon-pqa/) 和 [上游 README](https://amazon-pqa.s3.amazonaws.com/readme.txt) 为来源。解析 asin_id/asin、question_id、question_text、answer_text 或 answers[].answer_text、item_name、brand_name。实际下载已验证 asin + answers 数组格式。Registry 标注 CDLA-Permissive-1.0，但 README 还含用途/署名声明；导入保留这些来源说明，不把它概括为无条件商用许可。公开仓库发布数据前需核对适用条款。
- Bitext：使用[发布者数据卡](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)，作为客服意图评测参考，保留 CDLA-Sharing-1.0 信息，不导入店铺事实库。

下面命令均在工程根目录执行。可选依赖仅用于 JSON 数组和 Parquet：

```bash
.venv/bin/python -m pip install -r backend/requirements-data.txt
PYTHONPATH=backend .venv/bin/python scripts/datasets/import_public.py esci /path/products.parquet data/esci-knowledge.jsonl --limit 1000
PYTHONPATH=backend .venv/bin/python scripts/ingest_knowledge.py data/esci-knowledge.jsonl
PYTHONPATH=backend .venv/bin/python scripts/datasets/import_public.py pqa /path/amazon_pqa_headsets.json data/pqa-knowledge.jsonl --limit 1000 --locale us
PYTHONPATH=backend .venv/bin/python scripts/ingest_knowledge.py data/pqa-knowledge.jsonl
PYTHONPATH=backend .venv/bin/python scripts/datasets/import_bitext.py /path/bitext.json data/intent-eval.jsonl --limit 1000
```

`--limit` 是扫描记录上限，不保证得到同样数量的独立商品。输出文件须不存在，避免覆盖已有导出；格式错误会指出记录序号。已成功导入的知识可重复 upsert。商品价格库存由商品 ID 确定性生成，并非真实售价；再次导入不修改已有运营值。

公开商品导入后，再扩展合成订单，可让新增订单引用扩展商品目录：

```bash
PYTHONPATH=backend .venv/bin/python scripts/generate_mock_data.py --users 10000 --orders 30000 --as-of 2026-10-02
```

生成器不覆盖已有订单；需要另一套独立样本时使用 `--db data/commerce-other.sqlite3`，并通过 COMMERCE_DB_PATH 选择数据库。英文/日文/西班牙文知识不应期待用中文词法查询稳定命中；跨语言需求应配置适合的多语言 Embedding 模型，并单独评测。
