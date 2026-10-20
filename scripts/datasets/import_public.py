"""Normalize local ESCI/PQA exports; import catalog + emit provenance-rich knowledge JSONL.
No download is performed. Retain the upstream license/README with your downloaded files.
"""
import argparse
import csv
import hashlib
import json
from itertools import islice
from pathlib import Path

from app.commerce.store import CommerceStore
from app.core.config import get_settings
from app.knowledge.store import Document


def rows(path):
    if path.suffix == '.parquet':
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError('Parquet 输入需安装 backend/requirements-data.txt') from exc
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    elif path.suffix == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as file:
            yield from csv.DictReader(file)
    else:
        with path.open(encoding='utf-8') as file:
            first = next((char for char in iter(lambda: file.read(1), '') if not char.isspace()), '')
            file.seek(0)
            if first == '[':
                try:
                    import ijson
                except ImportError as exc:
                    raise ValueError('JSON 数组输入需安装 backend/requirements-data.txt，或转换为 JSONL') from exc
                yield from ijson.items(file, 'item')
            else:
                for line in file:
                    if line.strip():
                        yield json.loads(line)


def text(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return '\n'.join(text(item) for item in value)
    if not isinstance(value, (str, int, float)):
        raise ValueError('不支持的字段结构，请先规范为字符串')
    return str(value).strip()


def normalize(row, dataset, locale='us'):
    source_id = text(row.get('product_id') if dataset == 'esci' else row.get('asin_id') or row.get('asin'))
    locale = text(row.get('product_locale')) or locale
    if not source_id or len(source_id) > 100 or locale not in ('us', 'es', 'jp'):
        raise ValueError('缺少商品标识，或 locale 不属于 us/es/jp')
    pid = f'amazon:{locale}:{source_id}'
    title = text(row.get('product_title') if dataset == 'esci' else row.get('item_name')) or source_id
    brand = text(row.get('product_brand') if dataset == 'esci' else row.get('brand_name'))
    if dataset == 'esci':
        body = '\n'.join(filter(None, [title, text(row.get('product_description')), text(row.get('product_bullet_point'))]))
        doc_id, kind = 'esci:' + pid, 'product'
        url, license_name = 'https://github.com/amazon-science/esci-data', 'Apache-2.0'
    else:
        question = text(row.get('question_text'))
        answer = text(row.get('answer_text'))
        if not answer and isinstance(row.get('answers'), list):
            answer = '\n'.join(text(item.get('answer_text')) for item in row['answers'] if isinstance(item, dict))
        if not question or not answer:
            raise ValueError('PQA 缺少 question_text 或 answer_text')
        qid = text(row.get('question_id')) or hashlib.sha256(question.encode()).hexdigest()[:20]
        body = f'Question: {question}\nAnswer: {answer}'
        doc_id, kind = f'pqa:{pid}:{qid}', 'product_qa'
        url = 'https://registry.opendata.aws/amazon-pqa/'
        license_name = 'CDLA-Permissive-1.0 (registry); also retain upstream README notices'
    document = Document(document_id=doc_id, product_id=pid, title=title[:500], text=body[:100000],
                        kind=kind, source=dataset, source_url=url, license=license_name)
    # Synthetic operational values, deterministic per ID, not real Amazon prices/stock.
    number = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
    product = (pid, title[:500], brand[:200], 9900 + number % 90000, number % 31, dataset, source_id, locale)
    return document, product


def convert(source, output, dataset, store, limit=1000, locale='us'):
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output avoids silently replacing an existing knowledge export.
    destination = output.open('x', encoding='utf-8')
    try:
        with destination, store.connection() as db:
            for index, row in enumerate(islice(rows(source), limit), 1):
                try:
                    doc, product = normalize(row, dataset, locale)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'第 {index} 条记录格式错误: {exc}') from exc
                db.execute('''INSERT INTO products VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(product_id) DO UPDATE SET title=excluded.title,brand=excluded.brand''', product)
                destination.write(doc.model_dump_json() + '\n')
                count += 1
            # Flush before the database transaction commits.
            destination.flush()
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {'rows': count, 'output': str(output), 'business_price_stock': 'synthetic'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['esci', 'pqa'])
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--locale', choices=['us', 'es', 'jp'], default='us')
    parser.add_argument('--db', type=Path, default=get_settings().commerce_db_path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100000:
        parser.error('limit 必须为 1..100000')
    print(json.dumps(convert(args.source, args.output, args.dataset, CommerceStore(args.db), args.limit, args.locale), ensure_ascii=False))
