"""Run real model/API acceptance cases; store actual outputs, checks, latency and failures."""
import argparse
import asyncio
import json
import time
from pathlib import Path
import httpx
from app.evaluation.metrics import score_case


async def evaluate(url, cases):
    output=[]
    async with httpx.AsyncClient(base_url=url,timeout=200,trust_env=False) as client:
        for case in cases:
            sid=None; started=time.perf_counter()
            try:
                session=await client.post('/api/sessions');session.raise_for_status();sid=session.json()['session_id']
                response=await client.post('/api/chat',json={'session_id':sid,'message':case['question']})
                response.raise_for_status()
                result=response.json()
                output.append({'id':case['id'],'checks':score_case(case,result),'result':result,'duration_seconds':time.perf_counter()-started})
            except (httpx.HTTPError,ValueError,KeyError) as exc:
                output.append({'id':case['id'],'error':type(exc).__name__,'task_pass':False,'duration_seconds':time.perf_counter()-started})
            finally:
                if sid:
                    try:await client.delete('/api/sessions/'+sid)
                    except httpx.HTTPError:pass
    return {'cases':len(output),'task_pass_rate':sum(row.get('checks',{}).get('task_pass',False) for row in output)/max(len(output),1),
            'http_failure_count':sum('error' in row for row in output),'details':output,
            'note':'Rule-scored acceptance subset; not a human-rated factuality score.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--cases',type=Path,default=Path('fixtures/evaluation/agent-cases.jsonl'))
    p.add_argument('--output',type=Path,default=Path('reports/agent-live.json'));a=p.parse_args()
    result=asyncio.run(evaluate(a.url,[json.loads(line) for line in a.cases.read_text().splitlines()]))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='details'}))
