"""Opt-in: send two small concurrent requests; never starts/restarts vLLM."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from threading import Barrier
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'backend'), str(ROOT)]
from app.core.config import get_settings

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    settings = get_settings()
    url = settings.model_base_url
    key = settings.model_api_key.get_secret_value()
    model = settings.model_name
    barrier = Barrier(2)
    def request(index):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        body = json.dumps({'model':model,'messages':[{'role':'user','content':'Reply with OK only.'}],
                           'max_tokens':16,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}).encode()
        req = urllib.request.Request(url+'/chat/completions', data=body,
            headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        barrier.wait(timeout=10)
        started = time.monotonic()
        try:
            with opener.open(req, timeout=args.timeout) as response:
                result = json.load(response)
                ok = bool(result.get('choices'))
            return {'client':index,'ok':ok,'seconds':round(time.monotonic()-started,2)}
        except Exception as exc:
            return {'client':index,'ok':False,'error_type':type(exc).__name__,
                    'seconds':round(time.monotonic()-started,2)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, (1,2)))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print('Two successful responses show shared access, not simultaneous GPU execution or long-context capacity.')
    return 0 if all(item['ok'] for item in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
