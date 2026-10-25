"""Reproducible retrieval + SQL integrity evaluation; no fabricated end-to-end LLM score."""
import argparse
import json
import platform
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.metrics import retrieval_scores
from app.core.config import get_settings
from app.knowledge.store import KnowledgeStore


def evaluate(dataset, split='test', scoped=True):
    dataset=Path(dataset)
    kb=KnowledgeStore(dataset/'knowledge.sqlite3',backend='fts')
    with kb.connection() as db: mapping={row['id']:row['document_id'] for row in db.execute('SELECT id,document_id FROM chunks')}
    scores, failures, durations = [], [], []
    for line in (dataset/'retrieval-cases.jsonl').read_text().splitlines():
        case=json.loads(line)
        if case['split']!=split:continue
        started=time.perf_counter()
        result=kb.search(case['query'],product_id=case.get('product_id','') if scoped else '',limit=4)
        durations.append((time.perf_counter()-started)*1000)
        score=retrieval_scores([mapping[hit['chunk_id']] for hit in result['hits']],case['expected_document_ids'])
        scores.append(score)
        if not score.get('hit',score.get('empty_correct')):failures.append(case['id'])
    positive=[s for s in scores if 'hit' in s]; negative=[s for s in scores if 'empty_correct' in s]
    with sqlite3.connect(dataset/'commerce.sqlite3') as db:
        checks={
            'foreign_key_violations':len(db.execute('PRAGMA foreign_key_check').fetchall()),
            'order_total_mismatches':db.execute('''SELECT count(*) FROM (SELECT o.order_id FROM orders o JOIN order_items i USING(order_id)
                GROUP BY o.order_id HAVING o.total_cents!=sum(i.quantity*i.unit_price_cents))''').fetchone()[0],
            'refund_overpayments':db.execute('''SELECT count(*) FROM (SELECT r.order_id FROM refunds r JOIN payments p USING(order_id)
                GROUP BY r.order_id HAVING sum(r.amount_cents)>p.amount_cents OR p.status!='paid')''').fetchone()[0],
            'future_deliveries':db.execute("SELECT count(*) FROM shipments WHERE delivered_at>'2026-10-03'").fetchone()[0],
        }
    durations.sort()
    return {'generated_at':datetime.now(timezone.utc).isoformat(),'environment':platform.platform(),
            'dataset':json.loads((dataset/'manifest.json').read_text()),'split':split,'scope_filter':scoped,
            'cases':len(scores),'answerable':len(positive),'unanswerable':len(negative),
            'metrics':{key:sum(s[key] for s in positive)/max(len(positive),1) for key in ('hit','recall','mrr','ndcg')},
            'empty_result_accuracy':sum(s['empty_correct'] for s in negative)/max(len(negative),1),
            'retrieval_p95_ms':durations[min(len(durations)-1,int(len(durations)*.95))] if durations else None,
            'integrity':checks,'failed_case_ids':failures,
            'limitations':['Synthetic product-disjoint queries, not human-labelled real customer accuracy.',
                           'Public PQA is in the index but these relevance labels cover synthetic products only.',
                           'No generative model was used; end-to-end task accuracy is not measured.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=get_settings().data_dir/'portfolio')
    parser.add_argument('--split',choices=['dev','test'],default='test')
    parser.add_argument('--unscoped',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('reports/retrieval.json'))
    args=parser.parse_args()
    result=evaluate(args.dataset,args.split,not args.unscoped)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ('cases','metrics','empty_result_accuracy','retrieval_p95_ms','integrity')},ensure_ascii=False))
