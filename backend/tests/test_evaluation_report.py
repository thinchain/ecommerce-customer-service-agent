import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts.evaluate_qwen_ab import prepare, summarize
from scripts.render_evaluation_report import build_report, EXPECTATIONS, reference_data, workflow_checks
from tests.support import ConfiguredTestCase


class ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.current = {'attrs': dict(attrs), 'body': ''}
            self.scripts.append(self.current)

    def handle_endtag(self, tag):
        if tag == 'script':
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current['body'] += data


class EvaluationReportTests(ConfiguredTestCase):
    def test_multiple_runs_preserve_all_process_data_and_preload_each_review_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, annotations, original = [], [], []
            for variant in ('A', 'D'):
                run = root/variant; run.mkdir(); inputs.append(run)
                record = {'id': f'0:p01:{variant}:0', 'case_id': 'p01', 'turn': 0,
                          'variant': variant, 'reply': '原文回答', 'latency_seconds': 1.25,
                          'tool_calls': [{'name': 'catalog_search', 'arguments': '{"query":"RUN"}',
                                          'result': {'ok': True, 'result': [{'price_cents': 29900}]}}],
                          'kev_calls': [{'mode': 'scope', 'seconds': .1,
                                         'result': {'choice': 'customer_service', 'probabilities': {
                                             'customer_service': .9, 'unrelated': .1}},
                                         'usage': {'input_tokens': 10, 'output_tokens': 0}}],
                          'manual_review': {'task_completed': None}}
                original.append(record)
                (run/'results.jsonl').write_text(json.dumps(record)+'\n')
                (run/'run.json').write_text(json.dumps({
                    'configuration': {'model': 'demo', 'base_url': 'http://PRIVATE-SERVICE/v1'},
                    'models_response': {'root': '/PRIVATE-MODEL-DIRECTORY'},
                    'agent_limits': {'max_steps': 4}, 'protocol': {'rag': 'fts'}}))
                annotations.append({'id': record['id'], 'manual_review': {
                    'task_completed': True, 'review_source': 'assistant', 'notes': '按记录审核'}})
            reviews = root/'reviews.jsonl'
            reviews.write_text(''.join(json.dumps(row)+'\n' for row in annotations))
            output = root/'all.html'
            with patch('socket.socket.connect', side_effect=AssertionError('No model or network requests')):
                stats = build_report(inputs, output, reviews=[reviews])
            parser = ScriptCollector(); parser.feed(output.read_text())
            payload = json.loads(parser.scripts[0]['body'])
            self.assertEqual([item['raw'] for item in payload['records']], original)
            self.assertEqual(stats['preloaded_reviews'], 2)
            self.assertEqual(len({item['key'] for item in payload['records']}), 2)
            self.assertTrue(all(item['initial_review']['task_completed'] for item in payload['records']))
            self.assertEqual(payload['runs'][0]['agent_limits'], {'max_steps': 4})
            self.assertNotIn('PRIVATE-SERVICE', output.read_text())
            self.assertNotIn('PRIVATE-MODEL-DIRECTORY', output.read_text())
            # A wrong review file must fail instead of silently omitting a record.
            reviews.write_text(json.dumps({'id': 'unknown', 'manual_review': {}})+'\n')
            with self.assertRaises(ValueError):
                build_report(inputs, root/'wrong.html', reviews=[reviews])
            self.assertFalse((root/'wrong.html').exists())

    def test_report_preserves_raw_text_escapes_scripts_and_keeps_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root/'dataset'
            manifest = prepare(dataset)
            run = root/'run'; run.mkdir()
            (run/'run.json').write_text(json.dumps({'dataset': manifest}))
            dangerous = '</script><script>window.UNTRUSTED=true</script><img src=x onerror=alert(1)>'
            record = {'id': '0:o05:D:0', 'case_id': 'o05', 'turn': 0, 'variant': 'D', 'repeat': 0,
                      'question': '查询订单和商品', 'reply': dangerous, 'http_status': 200,
                      'tool_calls': [], 'tool_attempts': [], 'agent_events': [], 'manual_review': {}}
            results = run/'results.jsonl'; results.write_text(json.dumps(record)+'\n')
            review = root/'reviews.jsonl'
            review.write_text(json.dumps({'id': record['id'], 'manual_review': {
                'task_completed': False, 'review_source': 'assistant', 'reviewer': 'test-review'}})+'\n')
            before = hashlib.sha256(results.read_bytes()).hexdigest()
            output = root/'review.html'
            stats = build_report([run], output, dataset=dataset, reviews=[review])
            self.assertEqual(stats['records'], 1)
            parser = ScriptCollector(); parser.feed(output.read_text())
            self.assertEqual(len(parser.scripts), 2)
            payload = json.loads(parser.scripts[0]['body'])
            self.assertEqual(payload['records'][0]['raw']['reply'], dangerous)
            self.assertEqual(payload['records'][0]['raw']['manual_review'], {})
            self.assertEqual(payload['records'][0]['initial_review']['review_source'], 'assistant')
            self.assertTrue(payload['runs'][0]['reference_matched'])
            self.assertEqual(len(payload['reference']['tables']['products']), 6)
            self.assertTrue(any(c['ok'] is False for c in payload['records'][0]['workflow_checks']))
            self.assertEqual(hashlib.sha256(results.read_bytes()).hexdigest(), before)
            self.assertEqual(reference_data(dataset)['manifest'], manifest)
            with self.assertRaises(FileExistsError):
                build_report([run], output, dataset=dataset)
            review.write_text(json.dumps({'id': record['id'], 'manual_review': {'evidence_supported': 'yes'}})+'\n')
            with self.assertRaises(ValueError):
                build_report([run], root/'bad-review.html', reviews=[review])
            self.assertFalse((root/'bad-review.html').exists())

    def test_unknown_cases_do_not_inherit_workflow_expectations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            results=root/'results.jsonl'
            results.write_text(json.dumps({'id':'other', 'case_id':'o05', 'turn':0, 'variant':'C'})+'\n')
            output=root/'review.html'; build_report([results], output)
            parser=ScriptCollector(); parser.feed(output.read_text())
            payload=json.loads(parser.scripts[0]['body'])
            self.assertIsNone(payload['records'][0]['expected'])
            self.assertEqual(payload['records'][0]['workflow_checks'], [])

    def test_workflow_checks_count_failed_attempts_but_require_dependency_order(self):
        expected=json.loads(EXPECTATIONS.read_text())['turns']['o05:0']
        record={'agent_events':[{'agent':'responder','status':'completed'}],'tool_calls':[
            {'name':'catalog_search','agent':'product','result':{'ok':True}},
            {'name':'order_detail','agent':'orders','result':{'ok':False}},
            {'name':'knowledge_search','agent':'product','result':{'ok':True}}]}
        checks={item['label']:item['ok'] for item in workflow_checks(record, expected)}
        self.assertTrue(checks['必要工具已尝试'])
        self.assertTrue(checks['专业 Agent 集合匹配'])
        self.assertFalse(checks['order_detail → catalog_search'])
        self.assertTrue(checks['order_detail → knowledge_search'])

    def test_reference_rejects_non_synthetic_and_changed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); dataset=root/'dataset'; prepare(dataset)
            with (dataset/'cases.jsonl').open('a') as target: target.write('\n')
            with self.assertRaises(ValueError): reference_data(dataset)
            (dataset/'manifest.json').write_text('{"synthetic_only":false}')
            with self.assertRaises(ValueError): reference_data(dataset)

    def test_exported_workflow_review_is_summarized_with_correct_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{'id':name,'variant':'D','http_status':200,'latency_seconds':1,
                   'tokens':{'complete':False},'manual_review':{}} for name in ('one','two')]
            results=root/'results.jsonl';results.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            reviews=root/'reviews.jsonl';reviews.write_text(json.dumps({'id':'one','manual_review':{
                'workflow_correct':False,'review_source':'assistant','citation_correct':False}})+'\n')
            result=summarize(results,reviews)['variants']['D']
            quality=result['human_workflow_correctness']
            self.assertEqual(quality,{'reviewed':1,'rate':0.0})
            self.assertEqual(result['review_sources'],{'assistant':1})
            self.assertEqual(result['reviewed_citation_correctness'],{'reviewed':1,'rate':0.0})
