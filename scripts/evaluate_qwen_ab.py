"""Prepare offline fixtures, or explicitly run an already authorized LOCAL Qwen.

No model process is started. A/B uses the FastAPI ASGI endpoints in this process,
real LangGraph/tools/model client, independent DB copies and sessions per episode.
Latency includes ASGI/application/model time, not browser or TCP proxy overhead.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
from decimal import Decimal
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'backend'), str(ROOT)]
sys.dont_write_bytecode = True

import httpx
from app.core.config import Settings, override_settings
from app.commerce.store import CommerceStore
from app.knowledge.store import KnowledgeStore
from app.main import app
from app.services import agent as agent_service, qwen
from app.services.tools import execute_tool
from scripts.generate_mock_data import generate
from scripts.ingest_knowledge import ingest_file

CASES = ROOT/'fixtures/evaluation/qwen-ab-cases.jsonl'
FILES = ('commerce.sqlite3', 'knowledge.sqlite3', 'cases.jsonl')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(directory, *, enabled=True, base_url='http://127.0.0.1:8001/v1', model='qwen3.5-4b',
                  temperature=.3, max_tokens=1200, timeout=180, agent_timeout=600):
    root = Path(directory)
    return Settings(_env_file=None, OPENAI_BASE_URL=base_url, QWEN_MODEL=model,
                    OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY', 'local-dev-key'),
                    MODEL_TEMPERATURE=temperature, MODEL_MAX_TOKENS=max_tokens,
                    MODEL_TIMEOUT_SECONDS=timeout, AGENT_TIMEOUT_SECONDS=agent_timeout, MODEL_RETRIES=0,
                    COMMERCE_DB_PATH=root/'commerce.sqlite3', KNOWLEDGE_DB_PATH=root/'knowledge.sqlite3',
                    MEMORY_DB_PATH=root/'memory.sqlite3', CHROMA_PATH=root/'chroma',
                    RAG_BACKEND='fts', JEV_MODE='off', JEV_SCOPE_ENABLED=False,
                    DEMO_USER_ID='U00001', COMMERCE_TOOL_TRANSPORT='local', CATALOG_EVIDENCE_ENABLED=enabled)


def load_cases(path):
    cases = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Duplicate case ID')
    for case in cases:
        if not case.get('turns'):
            raise ValueError('Each case needs turns')
        for turn in case['turns']:
            if not isinstance(turn.get('message'), str) or not turn['message'].strip():
                raise ValueError('Invalid message')
            for pattern in [*turn.get('required_patterns', []), *turn.get('forbidden_patterns', [])]:
                re.compile(pattern)
    return cases


def prepare(directory):
    """New synthetic dataset only; never reads or overwrites the project's data/."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with override_settings(configuration(directory)):
            generate(directory/'commerce.sqlite3', users=2, orders=12)
            store = CommerceStore(directory/'commerce.sqlite3')
            with store.connection() as db:
                db.executemany('INSERT INTO products VALUES (?,?,?,?,?,?,?,?)', [
                    ('DEMO-ZERO', '零值测试商品', 'Benchmark', 0, 0, 'synthetic_benchmark', 'ZERO', 'demo'),
                    ('DEMO-NOKB', '无说明测试商品', 'Benchmark', 25000, 4, 'synthetic_benchmark', 'NOKB', 'demo')])
                db.execute('UPDATE orders SET user_id=? WHERE order_id=?', ('U00002', 'ORD00011'))
            ingest_file(ROOT/'fixtures/knowledge/demo.jsonl', KnowledgeStore(directory/'knowledge.sqlite3', backend='fts'))
        shutil.copyfile(CASES, directory/'cases.jsonl')
        cases = load_cases(directory/'cases.jsonl')
        manifest = {'version': 1, 'synthetic_only': True, 'cases': len(cases),
                    'turns': sum(len(c['turns']) for c in cases),
                    'sha256': {name: digest(directory/name) for name in FILES},
                    'notes': ['ORD00011 belongs to U00002; current user is U00001.',
                              'Fault cases alter tool responses only, identically in both variants.',
                              'No live model has been run by prepare.']}
        (directory/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    except BaseException:
        shutil.rmtree(directory)
        raise


def validate_dataset(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('synthetic_only') is not True or set(manifest['sha256']) != set(FILES):
        raise ValueError('Use a dataset created by --prepare')
    if any(digest(directory/name) != manifest['sha256'][name] for name in FILES):
        raise ValueError('Dataset checksum mismatch; do not compare different data')
    return manifest, load_cases(directory/'cases.jsonl')


def local_endpoint(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip('/') != '/v1'):
        raise ValueError('Only an already-running local HTTP /v1 endpoint is allowed')
    return url.rstrip('/')


def fault_executor(faults):
    def execute(name, arguments, memories):
        fault = faults.get(name)
        if fault == 'error':
            return {'ok': False, 'error': 'Benchmark injected tool failure', 'code': 'benchmark_fault'}
        if fault == 'empty':
            return {'ok': True, 'result': {'hits': []} if name == 'knowledge_search' else []}
        if fault == 'wrong_entity':
            return execute_tool('catalog_search', '{"product_ids":["DEMO-DESK"]}', memories)
        if fault == 'invalid_fields':
            return {'ok': True, 'result': [{'product_id': 'DEMO-RUN', 'price_cents': '29900', 'stock': True}]}
        if fault is not None:
            raise ValueError('Unsupported benchmark fault')
        if name == 'web_search':
            return {'ok': False, 'error': 'External search disabled in this benchmark'}
        return execute_tool(name, arguments, memories)
    return execute


def rule_checks(turn, reply):
    # Independent fixed golden labels. These lexical checks are deliberately NOT
    # called factual accuracy or task success; unrelated text can satisfy regexes.
    return {'required': {p: bool(re.search(p, reply)) for p in turn.get('required_patterns', [])},
            'forbidden': {p: bool(re.search(p, reply)) for p in turn.get('forbidden_patterns', [])}}


def catalog_checks(turn, reply):
    """Independent text/golden check, not a consumer of verification.catalog.

Unparseable prose/multi-entity lines need human review, never count as correct.
Only these known template-shaped fields are checked; extra claims need review.
"""
    checks = []
    for golden in turn.get('expected_catalog', []):
        lines = [line for line in reply.splitlines() if golden['product_id'] in line
                 and sum(pid in line for pid in ('DEMO-RUN', 'DEMO-DESK', 'DEMO-BUDS', 'DEMO-KEY', 'DEMO-ZERO', 'DEMO-NOKB')) == 1]
        price = next((match for line in lines if (match := re.search(r'(?:¥|￥|价格\s*|售价\s*)(\d+(?:\.\d{1,2})?)', line))), None)
        stock = next((match for line in lines if (match := re.search(r'库存\s*[:：]?\s*(\d+)\s*件', line))), None)
        checks.append({'product_id': golden['product_id'],
                       'price_correct': int(Decimal(price[1])*100) == golden['price_cents'] if price else None,
                       'stock_correct': int(stock[1]) == golden['stock'] if stock else None})
    return checks


def token_summary(calls):
    usages = [call.get('usage') for call in calls]
    if not calls:
        return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'complete': True}
    complete = all(isinstance(u, dict) and all(type(u.get(k)) is int for k in
                   ('prompt_tokens', 'completion_tokens', 'total_tokens')) for u in usages)
    return {**{key: sum(u[key] for u in usages) if complete else None
               for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}, 'complete': complete}


async def run_benchmark(dataset, output, *, allow_model_requests=False, repeats=3, base_url='http://127.0.0.1:8001/v1',
                        model='qwen3.5-4b', temperature=.3, max_tokens=1200, timeout=180, agent_timeout=600):
    if not allow_model_requests:
        raise ValueError('Run requires explicit --allow-model-requests after resource/usage authorization')
    if not 1 <= repeats <= 20:
        raise ValueError('repeats must be 1..20')
    base_url = local_endpoint(base_url)
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest, cases = validate_dataset(dataset)
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    common = dict(base_url=base_url, model=model, temperature=temperature, max_tokens=max_tokens,
                  timeout=timeout, agent_timeout=agent_timeout)
    # A read-only model-list probe, never a launcher or weight download.
    async with httpx.AsyncClient(timeout=5, trust_env=False) as probe:
        response = await probe.get(base_url+'/models', headers={
            'Authorization': 'Bearer '+os.environ.get('OPENAI_API_KEY', 'local-dev-key')})
        response.raise_for_status()
        model_list = response.json()
        if model not in {item['id'] for item in model_list.get('data', [])}:
            raise ValueError('Requested model is not served by this endpoint')
    output.mkdir(parents=True, exist_ok=False)
    source_paths = sorted((ROOT/'backend/app').rglob('*.py')) + [Path(__file__), CASES]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    metadata = {'configuration': common, 'dataset': manifest, 'source_sha256': source_hashes,
                'agent_limits': configuration(output, **common).agent.model_dump(),
                'protocol': {'thinking': False, 'rag': 'fts', 'tool_transport': 'local',
                             'model_retries': 0, 'jev_scope': False},
                'repeats': repeats, 'models_response': model_list,
                'variants': {'legacy': False, 'evidence': True}, 'counterbalanced_order': True,
                'scope': 'FastAPI ASGI + LangGraph + local model + tools; excludes browser/TCP proxy',
                'human_review_required': True, 'model_weights_identity': 'must be confirmed by operator'}
    (output/'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    old_complete, old_executor = agent_service.complete, agent_service.execute_tool
    try:
        with (output/'results.jsonl').open('x') as results:
            for repeat in range(repeats):
                for case_index, case in enumerate(cases):
                    variants = [('legacy', False), ('evidence', True)]
                    if (repeat+case_index) % 2:
                        variants.reverse()
                    for variant, enabled in variants:
                        await qwen.close_model_client()  # Separate breaker/client state per episode.
                        episode = output/'episodes'/f"{repeat}-{case['id']}-{variant}"
                        episode.mkdir(parents=True)
                        for name in FILES[:2]:
                            shutil.copyfile(dataset/name, episode/name)
                        settings = configuration(episode, enabled=enabled, **common)
                        calls, attempts = [], []
                        async def measured_complete(*args, **kwargs):
                            started = time.perf_counter()
                            try:
                                answer = await old_complete(*args, **kwargs)
                                calls.append({'seconds': time.perf_counter()-started, **answer.metadata})
                                return answer
                            except Exception as exc:
                                calls.append({'seconds': time.perf_counter()-started, 'usage': None,
                                              'error_type': type(exc).__name__})
                                raise
                        # Instrumentation forwards every model call unchanged. No
                        # model outputs are mocked; only labelled tool-fault cases differ.
                        agent_service.complete = measured_complete
                        execute = fault_executor(case.get('faults', {}))
                        def measured_tool(name, arguments, memories):
                            started = time.perf_counter()
                            result = execute(name, arguments, memories)
                            attempts.append({'name': name, 'arguments': arguments, 'result': result,
                                             'seconds': time.perf_counter()-started})
                            return result
                        agent_service.execute_tool = measured_tool
                        with override_settings(settings):
                            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                                         base_url='http://benchmark') as client:
                                created = await client.post('/api/sessions')
                                created.raise_for_status()
                                sid = created.json()['session_id']
                                for turn_index, turn in enumerate(case['turns']):
                                    calls.clear()
                                    attempts.clear()
                                    started = time.perf_counter()
                                    response = await client.post('/api/chat', json={'session_id': sid, 'message': turn['message']})
                                    payload = response.json()
                                    reply = payload.get('reply', '')
                                    record = {'id': f"{repeat}:{case['id']}:{variant}:{turn_index}",
                                              'case_id': case['id'], 'category': case['category'],
                                              'repeat': repeat, 'variant': variant, 'turn': turn_index,
                                              'question': turn['message'], 'session_id': sid,
                                              'http_status': response.status_code, 'reply': reply,
                                              'tool_calls': payload.get('tool_calls', []),
                                              'tool_attempts': list(attempts),
                                              'agent_events': payload.get('agent_events', []),
                                              'verification': payload.get('verification', {}),
                                              'error': payload.get('detail'), 'latency_seconds': time.perf_counter()-started,
                                              'model_calls': list(calls), 'tokens': token_summary(calls),
                                              'faults': case.get('faults', {}), 'golden': turn,
                                              'automatic_checks': rule_checks(turn, reply),
                                              'independent_catalog_checks': catalog_checks(turn, reply),
                                              'manual_review': {'fact_correct': None, 'task_completed': None,
                                                                'false_refusal': None, 'subtasks': [],
                                                                'unsupported_claims': [], 'reviewer': None, 'notes': ''}}
                                    results.write(json.dumps(record, ensure_ascii=False)+'\n'); results.flush()
                        for name in FILES[:2]:
                            if digest(episode/name) != manifest['sha256'][name]:
                                raise ValueError('Benchmark business dataset changed during episode')
    finally:
        agent_service.complete, agent_service.execute_tool = old_complete, old_executor
        await qwen.close_model_client()
    validate_dataset(dataset)
    summary = summarize(output/'results.jsonl')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize(path, reviews=None):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    review_map = {}
    if reviews:
        for line in Path(reviews).read_text().splitlines():
            item = json.loads(line)
            if item['id'] in review_map:
                raise ValueError('Duplicate review ID')
            review_map[item['id']] = item['manual_review']
        if set(review_map) - {r['id'] for r in records}:
            raise ValueError('Unknown review ID')
    answer = {}
    for variant in ('legacy', 'evidence'):
        rows = [r for r in records if r['variant'] == variant]
        reviews_for_rows = [review_map.get(r['id'], r['manual_review']) for r in rows]
        reviewed = {}
        for key in ('fact_correct', 'task_completed', 'false_refusal'):
            values = [r.get(key) for r in reviews_for_rows if r.get(key) is not None]
            if any(type(value) is not bool for value in values):
                raise ValueError('Review decisions must be boolean or null')
            reviewed[key] = {'reviewed': len(values), 'rate': sum(values)/len(values) if values else None}
        latencies = sorted(r['latency_seconds'] for r in rows)
        known_tokens = [r['tokens']['total_tokens'] for r in rows if r['tokens']['complete']]
        checked_fields = [check[key] for r in rows for check in r.get('independent_catalog_checks', [])
                          for key in ('price_correct', 'stock_correct') if check[key] is not None]
        answer[variant] = {'turns': len(rows), 'http_failures': sum(r['http_status'] != 200 for r in rows),
                           'human_fact_correctness': reviewed['fact_correct'], 'human_task_success': reviewed['task_completed'],
                           'human_false_refusal': reviewed['false_refusal'],
                           'latency_p50_seconds': statistics.median(latencies) if latencies else None,
                           'latency_p95_seconds': latencies[min(len(latencies)-1, int(len(latencies)*.95))] if latencies else None,
                           'token_turns_with_complete_usage': len(known_tokens),
                           'mean_tokens_known_turns': statistics.mean(known_tokens) if known_tokens else None}
        answer[variant]['machine_catalog_fields'] = {'parsed': len(checked_fields),
            'correct_rate_parsed_only': sum(checked_fields)/len(checked_fields) if checked_fields else None,
            'note': 'Conditional text extraction, not overall factuality/task accuracy.'}
    paired = {}
    for record in records:
        key = (record['repeat'], record['case_id'], record['turn']) if 'case_id' in record else record['id']
        paired.setdefault(key, {})[record['variant']] = record['latency_seconds']
    deltas = [v['evidence']-v['legacy'] for v in paired.values() if set(v) == {'legacy', 'evidence'}]
    return {'variants': answer,
            'paired_latency': {'pairs': len(deltas), 'mean_evidence_minus_legacy_seconds': statistics.mean(deltas) if deltas else None,
                               'median_evidence_minus_legacy_seconds': statistics.median(deltas) if deltas else None},
            'note': 'No human labels means null quality rates, never automatic factuality claims.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', type=Path)
    mode.add_argument('--run', type=Path, help='Prepared dataset directory')
    mode.add_argument('--summarize', type=Path, help='Existing results.jsonl')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--allow-model-requests', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    parser.add_argument('--model', default='qwen3.5-4b')
    parser.add_argument('--temperature', type=float, default=.3)
    parser.add_argument('--max-tokens', type=int, default=1200)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.prepare)
    elif args.summarize:
        result = summarize(args.summarize, args.reviews)
    else:
        if args.output is None or not args.allow_model_requests:
            parser.error('--run requires a new --output and explicit --allow-model-requests after authorization')
        result = asyncio.run(run_benchmark(args.run, args.output, allow_model_requests=True,
                             repeats=args.repeats, base_url=args.base_url, model=args.model,
                             temperature=args.temperature, max_tokens=args.max_tokens))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
