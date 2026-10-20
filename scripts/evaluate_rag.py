"""Offline retrieval regression, not an end-to-end answer quality benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
from app.knowledge.store import get_knowledge

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=Path('fixtures/knowledge/eval.jsonl'))
    args = parser.parse_args()
    store = get_knowledge()
    total = hits = 0
    reciprocal_rank = 0.0
    failures = []
    for line in args.cases.read_text(encoding='utf-8').splitlines():
        case = json.loads(line)
        result = store.search(case['query'], case.get('product_id', ''), case.get('kind', ''), limit=4)
        expected = hashlib.sha256((case['expected_document_id']+':0').encode()).hexdigest()
        ids = [hit['chunk_id'] for hit in result['hits']]
        total += 1
        if expected in ids:
            hits += 1
            reciprocal_rank += 1/(ids.index(expected)+1)
        else:
            failures.append(case['query'])
    print(json.dumps({'backend': store.backend, 'cases': total, 'hit_at_4': hits/max(total, 1),
                      'mrr_at_4': reciprocal_rank/max(total, 1), 'failed_queries': failures}, ensure_ascii=False))
    raise SystemExit(0 if total and hits == total else 1)
