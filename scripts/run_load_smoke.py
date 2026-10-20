"""Isolated TCP/Locust catalog load smoke. Never starts on the user's application port."""
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import build_opener, ProxyHandler

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as temp, socket.socket() as listener, (reports/'load-api.log').open('w') as log:
        listener.bind(('127.0.0.1',0));listener.listen(128);port=listener.getsockname()[1]
        env={**os.environ,'PYTHONPATH':str(ROOT/'backend')+':'+str(ROOT),'MEMORY_DB_PATH':temp+'/memory.sqlite3',
             'NO_PROXY':'127.0.0.1,localhost','no_proxy':'127.0.0.1,localhost'}
        opener=build_opener(ProxyHandler({}))
        server=subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--fd',str(listener.fileno())],
                                cwd=ROOT,env=env,pass_fds=(listener.fileno(),),stdout=log,stderr=log)
        try:
            for _ in range(100):
                if server.poll() is not None:raise RuntimeError('临时压测服务启动失败')
                try:
                    with opener.open(f'http://127.0.0.1:{port}/health/live',timeout=1):break
                except OSError:time.sleep(.1)
            else:raise RuntimeError('临时服务启动超时')
            run=subprocess.run([sys.executable,'-m','locust','-f','tests/load/locustfile.py','--host',f'http://127.0.0.1:{port}',
                                '--headless','-u','20','-r','5','-t','20s','--csv',str(reports/'catalog-load'),'--only-summary','BrowseUser'],
                               cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45)
            (reports/'locust.log').write_text(run.stdout)
            if run.returncode:raise RuntimeError('Locust 压测失败，见 reports/locust.log')
            with (reports/'catalog-load_stats.csv').open() as file:
                aggregate=next(row for row in csv.DictReader(file) if row['Name']=='Aggregated')
            summary={'scope':'20-second localhost TCP catalog smoke; excludes LLM inference and chat capacity',
                     'users':20,'statistics':aggregate,'return_code':run.returncode}
            (reports/'load-smoke.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
            print(json.dumps(summary,ensure_ascii=False))
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill();server.wait()
