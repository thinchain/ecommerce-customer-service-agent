"""Real local Qwen/Kev evaluation: multi-turn memory, entity flow and low-risk boundaries.

No scripted model responses, fault injection, shell/code execution or model launch.
All model requests are forwarded to the configured services, with full audit capture.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'backend'), str(ROOT)]
sys.dont_write_bytecode = True

from app.core.config import get_settings
from scripts.evaluate_qwen_ab import prepare, run_benchmark, digest, load_cases

CASES = ROOT/'fixtures/evaluation/workflow-live-cases.jsonl'


def prepare_workflow(directory):
    directory = Path(directory)
    manifest = prepare(directory)
    cases = load_cases(CASES)
    (directory/'cases.jsonl').write_bytes(CASES.read_bytes())
    manifest.update(cases=len(cases), turns=sum(len(c['turns']) for c in cases))
    manifest['sha256']['cases.jsonl'] = digest(directory/'cases.jsonl')
    manifest['notes'] = ['Real model responses only; no fault injection or executable attack payloads.',
                         'Labelled price updates affect only per-episode synthetic database copies.']
    (directory/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    turns = {}
    for case in cases:
        for i, turn in enumerate(case['turns']):
            tools = turn.get('expected_tools', [])
            agents = list(dict.fromkeys('orders' if t == 'order_detail' else 'product' for t in tools if t in ('order_detail','catalog_search','knowledge_search')))
            turns[f"{case['id']}:{i}"] = {'required_tools': tools, 'expected_agents': None if turn.get('expected_execution') == 'history_only' else agents,
                                         'ordered_tools': [['order_detail', t] for t in ('catalog_search','knowledge_search') if 'order_detail' in tools and t in tools],
                                         'notes': turn.get('expected_execution', '逐项核对当前输入、实体、权限、所问字段与实际证据。')}
    (directory/'expectations.json').write_text(json.dumps({'cases_sha256':manifest['sha256']['cases.jsonl'],'turns':turns}, ensure_ascii=False, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', type=Path)
    modes.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-model-requests', action='store_true')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--cases', nargs='+', help='Optional smoke subset, preserving original case IDs')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_workflow(args.prepare), ensure_ascii=False, indent=2));return
    if not args.output:
        parser.error('--run requires --output')
    settings = get_settings()
    # Load configured keys without printing them or storing them in run metadata.
    os.environ['OPENAI_API_KEY'] = settings.model_api_key.get_secret_value()
    os.environ['JEV_API_KEY'] = settings.jev_api_key.get_secret_value()
    summary = asyncio.run(run_benchmark(args.run,args.output,allow_model_requests=args.allow_model_requests,
        repeats=args.repeats,base_url=settings.model_base_url,model=settings.model_name,
        temperature=settings.model_temperature,max_tokens=settings.model_max_tokens,
        timeout=settings.model_timeout_seconds,agent_timeout=settings.agent_timeout_seconds,
        suite='workflow-live',kev_base_url=settings.jev_base_url,kev_model=settings.jev_model,
        kev_timeout=settings.jev_timeout_seconds,capture_messages=True,selected_cases=args.cases))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
