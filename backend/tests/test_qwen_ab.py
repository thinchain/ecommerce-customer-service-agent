import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app.core.message import Message
from scripts.evaluate_qwen_ab import (CASES, catalog_checks, digest, load_cases, local_endpoint, prepare,
                                     rule_checks, run_benchmark, summarize, token_summary, validate_dataset)
from tests.support import ConfiguredAsyncTestCase, ConfiguredTestCase


class BenchmarkPreparationTests(ConfiguredTestCase):
    def test_fixed_cases_cover_required_categories_and_multiturn(self):
        cases = load_cases(CASES)
        self.assertGreaterEqual(len(cases), 30)
        categories = {c['category'] for c in cases}
        self.assertTrue({'catalog', 'order_catalog', 'multiturn', 'rag', 'unanswerable',
                         'entity_error', 'partial_failure'} <= categories)
        self.assertGreaterEqual(sum(len(c['turns']) > 1 for c in cases), 4)

    def test_prepare_is_offline_synthetic_and_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)/'dataset'
            with patch('socket.socket.connect', side_effect=AssertionError('No network')):
                manifest = prepare(dataset)
            self.assertTrue(manifest['synthetic_only'])
            self.assertGreaterEqual(manifest['cases'], 30)
            self.assertEqual(validate_dataset(dataset)[0], manifest)
            before = digest(dataset/'commerce.sqlite3')
            with self.assertRaises(FileExistsError): prepare(dataset)
            self.assertEqual(digest(dataset/'commerce.sqlite3'), before)

    def test_checksum_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)/'dataset'; prepare(dataset)
            with (dataset/'cases.jsonl').open('a') as file: file.write('\n')
            with self.assertRaises(ValueError): validate_dataset(dataset)

    def test_paid_or_remote_endpoint_is_rejected(self):
        for url in ('https://api.openai.com/v1', 'http://example.com/v1', 'http://127.0.0.1/v1?key=x',
                    'http://user:secret@127.0.0.1/v1', 'http://localhost:8001/other'):
            with self.subTest(url=url), self.assertRaises(ValueError): local_endpoint(url)
        self.assertEqual(local_endpoint('http://127.0.0.1:8001/v1/'), 'http://127.0.0.1:8001/v1')

    def test_missing_usage_is_unknown_not_zero(self):
        self.assertIsNone(token_summary([{'usage': None}])['total_tokens'])
        self.assertEqual(token_summary([])['total_tokens'], 0)

    def test_catalog_checks_use_golden_text_not_verifier(self):
        turn = {'expected_catalog': [{'product_id': 'DEMO-RUN', 'price_cents': 29900, 'stock': 18}]}
        wrong = catalog_checks(turn, '商品 DEMO-RUN：价格¥999.00，库存 999 件')
        self.assertFalse(wrong[0]['price_correct'])
        self.assertFalse(wrong[0]['stock_correct'])
        self.assertIsNone(catalog_checks(turn, '资料不足')[0]['price_correct'])

    def test_regex_checks_are_not_factuality_labels(self):
        checks = rule_checks({'required_patterns': ['299']}, '模型猜测299')
        self.assertTrue(checks['required']['299'])
        self.assertNotIn('task_pass', checks)

    def test_summary_requires_human_labels_and_valid_review_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); results = root/'results.jsonl'
            record = {'id': 'sample', 'variant': 'evidence', 'http_status': 200, 'latency_seconds': 1,
                      'tokens': {'total_tokens': None, 'complete': False},
                      'manual_review': {'fact_correct': None, 'task_completed': None}}
            results.write_text(json.dumps(record)+'\n')
            summary = summarize(results)['variants']['evidence']
            self.assertIsNone(summary['human_task_success']['rate'])
            self.assertIsNone(summary['human_false_refusal']['rate'])
            review = root/'reviews.jsonl'
            review.write_text(json.dumps({'id': 'sample', 'manual_review': {'fact_correct': True, 'task_completed': False}})+'\n')
            self.assertEqual(summarize(results, review)['variants']['evidence']['human_task_success']['rate'], 0)
            review.write_text(json.dumps({'id': 'unknown', 'manual_review': {}})+'\n')
            with self.assertRaises(ValueError): summarize(results, review)


class BenchmarkRunnerTests(ConfiguredAsyncTestCase):
    async def test_run_requires_authorization_before_any_network_or_file_write(self):
        with patch('httpx.AsyncClient', side_effect=AssertionError('Network forbidden')):
            with self.assertRaises(ValueError): await run_benchmark('unused', 'unused-output')

    async def test_mock_run_exercises_both_real_api_paths_in_isolated_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); dataset = root/'dataset'; prepare(dataset)
            # A tiny test-only subset; production fixed cases are separately checked.
            cases = [{'id': 'small', 'category': 'catalog', 'turns': [
                {'message': '查询 DEMO-RUN 价格库存'}, {'message': '现在价格呢'}]}]
            (dataset/'cases.jsonl').write_text(json.dumps(cases[0], ensure_ascii=False)+'\n')
            manifest = json.loads((dataset/'manifest.json').read_text())
            manifest['sha256']['cases.jsonl'] = digest(dataset/'cases.jsonl')
            (dataset/'manifest.json').write_text(json.dumps(manifest))
            original_client = httpx.AsyncClient
            def client_factory(*args, **kwargs):
                if 'transport' not in kwargs:
                    kwargs['transport'] = httpx.MockTransport(lambda request: httpx.Response(200, json={
                        'data': [{'id': 'qwen3.5-4b'}]}))
                return original_client(*args, **kwargs)
            model = AsyncMock(return_value=Message(role='assistant', content='价格999元', metadata={
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}, 'model': 'qwen3.5-4b'}))
            with patch('scripts.evaluate_qwen_ab.httpx.AsyncClient', client_factory), \
                    patch('app.services.agent.complete', model):
                summary = await run_benchmark(dataset, root/'results', repeats=1, allow_model_requests=True)
            records = [json.loads(line) for line in (root/'results/results.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 4)
            sessions = {variant: {r['session_id'] for r in records if r['variant'] == variant} for variant in ('legacy', 'evidence')}
            self.assertFalse(sessions['legacy'] & sessions['evidence'])
            self.assertEqual(len(sessions['evidence']), 1)
            self.assertTrue(all(r['tokens']['total_tokens'] == 12 for r in records))
            self.assertTrue(all('999' not in r['reply'] for r in records if r['variant'] == 'evidence'))
            self.assertIsNone(summary['variants']['evidence']['human_fact_correctness']['rate'])
            self.assertEqual(summary['paired_latency']['pairs'], 2)
            self.assertEqual(validate_dataset(dataset)[0]['sha256'], manifest['sha256'])
