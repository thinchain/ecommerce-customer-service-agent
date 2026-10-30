"""Compare whole sessions and matched turns without generating new model responses."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT)]
from scripts.evaluate_qwen_ab import summarize
from scripts.render_evaluation_report import read_records


def metrics(run,reviews=None):
    run=Path(run);rows=read_records(run/'results.jsonl')
    labels={r['id']:r['manual_review'] for r in read_records(Path(reviews))} if reviews else {}
    variants=list(dict.fromkeys(r['variant'] for r in rows))
    sessions=defaultdict(list)
    for r in rows:sessions[(r['variant'],r['case_id'],r['repeat'])].append(r)
    episodes=[]
    for (variant,case,repeat),group in sessions.items():
        known=[r for r in group if r['tokens']['complete']]
        episodes.append({'variant':variant,'case_id':case,'repeat':repeat,'turns':len(group),
                         'seconds':sum(r['latency_seconds'] for r in group),
                         'qwen_calls':sum(len(r['model_calls']) for r in group),
                         'kev_calls':sum(len(r.get('kev_calls',[])) for r in group),
                         'known_token_turns':len(known),'total_tokens':sum(r['tokens']['total_tokens'] for r in known) if len(known)==len(group) else None,
                         'tasks_completed':sum(labels.get(r['id'],{}).get('task_completed') is True for r in group),
                         'tasks_reviewed':sum(type(labels.get(r['id'],{}).get('task_completed')) is bool for r in group)})
    comparisons=[]
    key=lambda r:(r['case_id'],r['turn'],r['repeat'])
    for a,b in itertools.combinations(variants,2):
        left={key(r):r for r in rows if r['variant']==a};right={key(r):r for r in rows if r['variant']==b}
        pairs=[(left[k],right[k]) for k in left.keys()&right.keys()]
        deltas=[y['latency_seconds']-x['latency_seconds'] for x,y in pairs]
        both=[(x,y) for x,y in pairs if labels.get(x['id'],{}).get('task_completed') is True and labels.get(y['id'],{}).get('task_completed') is True]
        comparisons.append({'left':a,'right':b,'pairs':len(pairs),
                            'mean_latency_delta_seconds':statistics.mean(deltas),
                            'median_latency_delta_seconds':statistics.median(deltas),
                            'qwen_call_delta':sum(len(y['model_calls'])-len(x['model_calls']) for x,y in pairs),
                            'both_task_complete_pairs':len(both),
                            'both_complete_mean_latency_delta_seconds':statistics.mean(y['latency_seconds']-x['latency_seconds'] for x,y in both) if both else None})
    recalls=[]
    for (variant,case,repeat),group in sessions.items():
        group.sort(key=lambda r:r['turn'])
        for index,row in enumerate(group):
            if row.get('decision',{}).get('execution')!='history_only':continue
            fields=(row['decision'].get('request') or {}).get('fields')
            prior=next((p for p in reversed(group[:index]) if p.get('decision',{}).get('execution')!='history_only'
                        and (p['decision'].get('request') or {}).get('fields')==fields
                        and {'price','stock'}.intersection(fields or [])),None)
            if prior and prior['tokens']['complete'] and row['tokens']['complete']:
                recalls.append({'variant':variant,'case_id':case,'source_turn':prior['turn'],
                                'recall_turn':row['turn'],'fields':fields,
                                'source_tokens':prior['tokens']['total_tokens'],'recall_tokens':row['tokens']['total_tokens'],
                                'source_qwen_calls':len(prior['model_calls']),'recall_qwen_calls':len(row['model_calls']),
                                'recall_business_tools':sum(t['name'] in ('catalog_search','knowledge_search','order_detail') for t in row['tool_calls']),
                                'source_task_completed':labels.get(prior['id'],{}).get('task_completed'),
                                'recall_task_completed':labels.get(row['id'],{}).get('task_completed')})
    meta=json.loads((run/'run.json').read_text())
    source_match=all((ROOT/p).is_file() and hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in meta['source_sha256'].items())
    # Kev reports input_tokens/output_tokens rather than Qwen's prompt_tokens/
    # completion_tokens. Retain the legacy aggregate and account for both formats
    # explicitly here; missing usage is never counted as zero.
    kev_usage={}
    for variant in variants:
        calls=[c for r in rows if r['variant']==variant for c in r.get('kev_calls',[])]
        known=[]
        for call in calls:
            usage=call.get('usage') or {}
            inp=usage.get('input_tokens',usage.get('prompt_tokens'))
            out=usage.get('output_tokens',usage.get('completion_tokens'))
            if type(inp) is int and inp>=0 and type(out) is int and out>=0:
                known.append((inp,out))
        kev_usage[variant]={'calls':len(calls),'calls_with_input_and_output_usage':len(known),
                            'input_tokens_known_calls':sum(i for i,o in known) if known else None,
                            'output_tokens_known_calls':sum(o for i,o in known) if known else None,
                            'sum_input_output_known_calls':sum(i+o for i,o in known) if known else None,
                            'note':'Sum of returned input/output counts, separately from Qwen. No calls means not applicable.'}
    return {'summary':summarize(run/'results.jsonl',reviews),'sessions':episodes,
            'kev_reported_usage':kev_usage,
            'pairwise':comparisons,'same_requested_fields_recall_pairs':recalls,
            'source_hashes_match_current_files':source_match,
            'notes':['Small development/regression suite, one run is not a stable latency claim.',
                     'Recall pairs match requested fields; historical values must not be used for current queries.',
                     'Keep per-model usage separate; wall time is not GPU compute time, energy or money.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--reviews',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a new output file')
    result=metrics(args.run,args.reviews)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'records':sum(v['turns'] for v in result['summary']['variants'].values()),'sessions':len(result['sessions'])}))


if __name__=='__main__':main()
