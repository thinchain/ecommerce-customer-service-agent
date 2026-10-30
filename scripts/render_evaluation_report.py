"""Build an offline HTML review report from existing evaluations; never call models.

Example: python scripts/render_evaluation_report.py reports/qwen-ab-run-r1
    reports/qwen-cd-run-r1 --dataset reports/qwen-ab-data-r1
    --output reports/evaluation-review.html
"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'scripts/templates/evaluation_review.html'
EXPECTATIONS = ROOT / 'fixtures/evaluation/qwen-review-expectations.json'


def read_records(path):
    records, seen = [], set()
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in seen:
            raise ValueError(f'{path.name}:{number}: invalid or duplicate record ID')
        seen.add(row['id'])
        records.append(row)
    if not records:
        raise ValueError('The results file is empty')
    return records


def workflow_checks(record, expected):
    """Structural review aids. Passing these checks is not semantic task success."""
    trace = record.get('tool_calls', [])
    attempts = record.get('tool_attempts', [])
    calls = trace or attempts
    names = [call.get('name') for call in calls]
    checks = []
    if expected:
        missing = sorted(set(expected['required_tools']) - set(names))
        checks.append({'label': '必要工具已尝试', 'ok': not missing,
                       'detail': '缺少：' + '、'.join(missing) if missing else '已尝试标注要求的工具；是否取得正确证据仍需审核。'})
        roles = set(expected['expected_agents'] or [])
        actual = {event['agent'] for event in record.get('agent_events', [])
                  if event.get('agent') not in (None, 'supervisor', 'responder')}
        actual |= {call['agent'] for call in trace if call.get('agent') not in (None, 'supervisor', 'responder')}
        if expected['expected_agents'] is not None:
            checks.append({'label': '专业 Agent 集合匹配', 'ok': roles == actual,
                           'detail': f'预期：{sorted(roles)}；记录：{sorted(actual)}。不要求机械地调用所有 Agent。'})
        for before, after in expected.get('ordered_tools', []):
            ok = before in names and after in names and names.index(before) < names.index(after)
            checks.append({'label': f'{before} → {after}', 'ok': ok,
                           'detail': '按工具记录数组检查依赖顺序；不能证明实体交接正确。'})
    return checks


def alerts(record, checks):
    result = []
    if record.get('http_status') != 200:
        result.append({'kind': 'http', 'text': 'HTTP 请求失败'})
    scope = record.get('decision', {}).get('scope', {})
    if scope.get('status') == 'blocked':
        result.append({'kind': 'scope', 'text': '范围拦截：' + str(scope.get('reason', '未知'))})
    errors = sum('error_type' in call for call in record.get('model_calls', []))
    if errors:
        result.append({'kind': 'model', 'text': f'内部 Qwen 调用异常 {errors} 次'})
    if any(call.get('result', {}).get('status') != 'evaluated' for call in record.get('kev_calls', [])):
        result.append({'kind': 'kev', 'text': 'Kev 调用失败或结果不可用'})
    if any(event.get('status') in ('failed', 'limit') for event in record.get('agent_events', [])):
        result.append({'kind': 'worker', 'text': '专业 Agent 失败或预算耗尽'})
    if any(check['ok'] is False for check in checks):
        result.append({'kind': 'flow', 'text': '流程检查项不满足'})
    automatic = record.get('automatic_checks', {})
    if any(value is False for value in automatic.get('required', {}).values()) or any(
            value is True for value in automatic.get('forbidden', {}).values()):
        result.append({'kind': 'rules', 'text': '关键词规则存在未通过项'})
    verification = record.get('verification', {})
    if verification.get('invalid'):
        result.append({'kind': 'citation', 'text': '引用未能核验'})
    if any(call.get('result', {}).get('ok') is False for call in record.get('tool_calls', [])):
        result.append({'kind': 'tool', 'text': '存在工具错误；结合故障注入和部分回答审核'})
    return result


def reference_data(dataset):
    """Only export a checksum-verified synthetic benchmark, using read-only SQLite."""
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('synthetic_only') is not True:
        raise ValueError('Reference data must come from the synthetic --prepare benchmark')
    expected_files = ('commerce.sqlite3', 'knowledge.sqlite3', 'cases.jsonl')
    for name in expected_files:
        if hashlib.sha256((dataset/name).read_bytes()).hexdigest() != manifest.get('sha256', {}).get(name):
            raise ValueError('Reference dataset checksum mismatch')
    tables = {}
    names = ('users', 'products', 'orders', 'order_items', 'payments', 'shipments', 'refunds', 'tickets')
    with closing(sqlite3.connect((dataset/'commerce.sqlite3').as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        for name in names:
            # Names are a fixed allowlist, never SQL supplied by report data.
            rows = [dict(row) for row in db.execute(f'SELECT * FROM {name} LIMIT 1001')]
            if len(rows) > 1000:
                raise ValueError('Reference dataset is too large for the review report')
            tables[name] = rows
    with closing(sqlite3.connect((dataset/'knowledge.sqlite3').as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute('SELECT id,document_id,text,metadata FROM chunks LIMIT 1001')]
        if len(rows) > 1000:
            raise ValueError('Reference knowledge is too large for the review report')
        for row in rows:
            row['metadata'] = json.loads(row['metadata'])
        tables['knowledge_chunks'] = rows
    return {'manifest': manifest, 'tables': tables}


def build_report(inputs, output, *, dataset=None, expectations=EXPECTATIONS, reviews=()):
    output = Path(output).resolve()
    if output.suffix.lower() != '.html':
        raise ValueError('Output must be an .html file')
    if output.exists():
        raise FileExistsError('Choose a new report filename; existing files are not overwritten')
    paths = expectations if isinstance(expectations, (tuple, list)) else [expectations]
    expected_sets = [json.loads(Path(path).read_text()) for path in paths]
    expected_by_hash = {entry['cases_sha256']: entry for entry in expected_sets}
    runs, records = [], []
    reference = reference_data(dataset) if dataset else None
    for index, value in enumerate(inputs):
        path = Path(value)
        path = path/'results.jsonl' if path.is_dir() else path
        metadata_path = path.parent/'run.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        cases_sha = metadata.get('dataset', {}).get('sha256', {}).get('cases.jsonl')
        expected = expected_by_hash.get(cases_sha)
        checksum_match = expected is not None
        label = path.parent.name
        run_id = f'run{index+1}'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        runs.append({'id': run_id, 'label': label, 'results_sha256': sha,
                     'suite': metadata.get('suite', 'evidence-ab'),
                     'model': metadata.get('configuration', {}).get('model'),
                     'temperature': metadata.get('configuration', {}).get('temperature'),
                     'max_tokens': metadata.get('configuration', {}).get('max_tokens'),
                     'variant_configuration': metadata.get('variant_configuration', {}),
                     'agent_limits': metadata.get('agent_limits', {}),
                     'protocol': {key: metadata.get('protocol', {}).get(key) for key in
                                  ('thinking', 'rag', 'tool_transport', 'model_retries', 'jev_scope')},
                     'expectations_matched': checksum_match,
                     'reference_matched': reference is not None and all(
                         metadata.get('dataset', {}).get('sha256', {}).get(name) == reference['manifest']['sha256'][name]
                         for name in ('commerce.sqlite3', 'knowledge.sqlite3'))})
        for row in read_records(path):
            expectation = expected['turns'].get(f"{row.get('case_id')}:{row.get('turn', 0)}") if checksum_match else None
            checks = workflow_checks(row, expectation)
            records.append({'key': f'{run_id}:{row["id"]}', 'run': run_id, 'raw': row,
                            'expected': expectation, 'workflow_checks': checks, 'alerts': alerts(row, checks)})
    supplied = set()
    for review_path in reviews:
        for annotation in read_records(Path(review_path)):
            matches = [item for item in records if item['raw']['id'] == annotation['id']]
            if len(matches) != 1 or annotation['id'] in supplied:
                raise ValueError('Review ID is unknown, ambiguous between runs, or duplicated')
            judgment = annotation.get('manual_review')
            if not isinstance(judgment, dict):
                raise ValueError('Review must include a manual_review object')
            for name in ('fact_correct', 'task_completed', 'workflow_correct', 'false_refusal',
                         'kev_scope_correct', 'kev_route_correct', 'evidence_supported', 'citation_correct'):
                if judgment.get(name) is not None and type(judgment[name]) is not bool:
                    raise ValueError('Review decisions must be boolean or null')
            matches[0]['initial_review'] = judgment
            supplied.add(annotation['id'])
    payload = {'runs': runs, 'records': records, 'reference': reference,
               'review_key': hashlib.sha256(''.join(run['results_sha256'] for run in runs).encode()).hexdigest()[:20]}
    # Prevent untrusted model text from closing the JSON script element or adding HTML.
    encoded = json.dumps(payload, ensure_ascii=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    encoded = encoded.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    html = TEMPLATE.read_text().replace('/*__REPORT_DATA__*/', encoded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as target:
        target.write(html)
    return {'records': len(records), 'preloaded_reviews': len(supplied),
            'variants': dict(Counter(row['raw']['variant'] for row in records)),
            'alerts': sum(bool(row['alerts']) for row in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='Run directories or results.jsonl files')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', type=Path, help='Optional original synthetic --prepare dataset')
    parser.add_argument('--reviews', type=Path, action='append', default=[], help='Optional review JSONL files; IDs must be unambiguous')
    parser.add_argument('--expectations', type=Path, action='append', default=[], help='Additional case-hash-matched review checklists')
    args = parser.parse_args()
    print(json.dumps(build_report(args.inputs, args.output, dataset=args.dataset, reviews=args.reviews,
                                  expectations=[EXPECTATIONS, *args.expectations]), ensure_ascii=False))


if __name__ == '__main__':
    main()
