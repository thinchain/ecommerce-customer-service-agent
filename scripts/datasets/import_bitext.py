"""Export supported Bitext intents as evaluation cases, never as store policy/RAG facts."""
import argparse
import json
from itertools import islice
from pathlib import Path
from import_public import rows

TOOL_EXPECTATIONS = {
    'track_order': ['list_orders', 'order_detail'],
    'check_refund_status': ['list_orders', 'order_detail'],
    'check_invoice': ['list_orders', 'order_detail'],
    'get_refund': ['knowledge_search'],
    'check_cancellation_fee': ['knowledge_search'],
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int, default=1000)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit 必须大于 0')
    kept = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        for row in islice(rows(args.source), args.limit):
            intent = row.get('intent')
            if intent not in TOOL_EXPECTATIONS:
                continue
            out.write(json.dumps({'query': row['instruction'], 'intent': intent,
                                  'acceptable_tools': TOOL_EXPECTATIONS[intent],
                                  'source': 'bitext', 'license': 'CDLA-Sharing-1.0'}, ensure_ascii=False) + '\n')
            kept += 1
    print(json.dumps({'exported_cases': kept, 'note': '工具选择评测参考；需人工审核，不是任务答案'}))
