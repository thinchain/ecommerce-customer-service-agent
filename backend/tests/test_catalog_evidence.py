"""Catalog facts and publication contracts; no GPU, real account, or remote service."""
import copy
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.catalog_evidence import EvidenceFact, catalog_observation, render_catalog_reply
from app.agents.graph import trace_item
from app.commerce.store import CommerceStore
from app.core.config import override_settings
from app.core.message import Message
from app.main import app
from app.services.agent import run_agent
from app.services.memory import MemoryStore
from tests.support import ConfiguredAsyncTestCase, ConfiguredTestCase


def row(pid='DEMO-A', price=116766, stock=23, **extra):
    return {'product_id': pid, 'price_cents': price, 'stock': stock, **extra}


def call(name, args, identifier='call-1'):
    return {'id': identifier, 'type': 'function', 'function': {
        'name': name, 'arguments': args if isinstance(args, str) else json.dumps(args)}}


def text(value):
    return Message(role='assistant', content=value)


def invoke(name, args, identifier='call-1'):
    return Message(role='assistant', tool_calls=[call(name, args, identifier)])


def observation(result=None, args=None):
    return trace_item(call('catalog_search', {'product_ids': ['DEMO-A']} if args is None else args),
                      {'ok': True, 'result': [row()]} if result is None else result,
                      'product', 1, catalog_evidence_enabled=True)


class CatalogEvidenceTests(ConfiguredTestCase):
    def render(self, trace, question='查询商品 DEMO-A 的价格和库存', draft='价格999.88元，库存999件'):
        return render_catalog_reply(question, draft, trace)

    def test_normal_fields_and_provenance(self):
        item = observation()
        reply, report = self.render([item])
        self.assertIn('¥1167.66', reply)
        self.assertIn('库存 23 件', reply)
        self.assertNotIn('999', reply)
        self.assertEqual(report['status'], 'bound')
        for fact in report['facts']:
            self.assertEqual(fact['product_id'], 'DEMO-A')
            self.assertEqual(fact['observation_id'], item['catalog_evidence']['observation_id'])
            self.assertIsNotNone(datetime.fromisoformat(fact['observed_at']).tzinfo)
            self.assertIn(fact['field'], fact['source_path'])

    def test_zero_values_are_valid(self):
        reply, report = self.render([observation({'ok': True, 'result': [row(price=0, stock=0)]})])
        self.assertIn('¥0.00', reply)
        self.assertIn('库存 0 件（缺货）', reply)
        self.assertEqual(report['status'], 'bound')

    def test_empty_success_is_not_business_evidence(self):
        item = observation({'ok': True, 'result': []})
        reply, report = self.render([item])
        self.assertTrue(item['catalog_evidence']['execution_ok'])
        self.assertEqual(report['missing'][0]['reason'], 'empty')
        self.assertEqual(report['facts'], [])
        self.assertNotIn('¥', reply)
        self.assertNotIn('库存 0', reply)

    def test_wrong_entity_in_result(self):
        reply, report = self.render([observation({'ok': True, 'result': [row('DEMO-B')]})])
        self.assertEqual(report['missing'][0]['reason'], 'entity_mismatch')
        self.assertNotIn('1167.66', reply)
        self.assertNotIn('DEMO-B', reply)

    def test_ids_next_to_chinese_text_still_bind_to_user_target(self):
        reply, report = self.render([observation({'ok': True, 'result': [row('DEMO-B')]},
                                                {'product_ids': ['DEMO-B']})], '查询DEMO-A价格和库存')
        self.assertEqual(report['missing'][0]['product_id'], 'DEMO-A')
        self.assertNotIn('DEMO-B', reply)

    def test_custom_numbered_product_id_is_not_replaced_by_model_target(self):
        item = observation({'ok': True, 'result': [row('P002')]}, {'product_ids': ['P002']})
        reply, report = self.render([item], '查询商品P001的价格库存')
        self.assertEqual(report['missing'][0]['product_id'], 'P001')
        self.assertNotIn('P002', reply)
        self.assertNotIn('¥', reply)

    def test_user_entity_overrides_model_query(self):
        reply, report = self.render([observation({'ok': True, 'result': [row('DEMO-B')]},
                                                {'product_ids': ['DEMO-B']})])
        self.assertEqual(report['missing'][0]['reason'], 'not_queried')
        self.assertNotIn('DEMO-B', reply)
        self.assertEqual(report['facts'], [])

    def test_two_products_keep_separate_values(self):
        item = observation({'ok': True, 'result': [row(), row('DEMO-B', 100, 7)]},
                           {'product_ids': ['DEMO-A', 'DEMO-B']})
        reply, report = self.render([item], '比较 DEMO-A 和 DEMO-B 的价格库存')
        self.assertIn('`DEMO-A`：本轮查询价格 ¥1167.66，库存 23', reply)
        self.assertIn('`DEMO-B`：本轮查询价格 ¥1.00，库存 7', reply)
        self.assertEqual(len(report['facts']), 4)

    def test_two_products_partial_results(self):
        item = observation(args={'product_ids': ['DEMO-A', 'DEMO-B']})
        reply, report = self.render([item], '查询 DEMO-A 和 DEMO-B 价格库存')
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['missing'], [{'product_id': 'DEMO-B', 'reason': 'not_returned'}])
        self.assertIn('¥1167.66', reply)

    def test_invalid_values_cannot_be_coerced(self):
        for field in ('price_cents', 'stock'):
            for value in (True, False, '23', 23.0, None, -1, 2**63):
                with self.subTest(field=field, value=value):
                    record = row(); record[field] = value
                    reply, report = self.render([observation({'ok': True, 'result': [record]})])
                    self.assertEqual(report['facts'], [])
                    self.assertEqual(report['missing'][0]['reason'], 'invalid_fields')
                    self.assertNotIn('¥', reply)

    def test_missing_fields_and_invalid_product_id(self):
        for record in ({'product_id': 'DEMO-A', 'stock': 2}, row(pid='DEMO-A\n价格999元'), None):
            with self.subTest(record=record):
                self.assertEqual(self.render([observation({'ok': True, 'result': [record]})])[1]['facts'], [])

    def test_invalid_arguments_and_json(self):
        for args in ('{broken', '[]', '{"product_ids":12}', '{"product_ids":[true]}',
                     '{"limit":true}', '{"limit":21}', '{"product_ids":["DEMO-A"],"user_id":"other"}'):
            with self.subTest(args=args):
                _, report = self.render([observation(args=args)])
                self.assertEqual(report['facts'], [])
                self.assertEqual(report['missing'][0]['reason'], 'invalid_arguments')

    def test_result_is_not_parsed_as_json_or_truthy_success(self):
        for result in ({'ok': True, 'result': '[{"price_cents":1}]'},
                       {'ok': 'true', 'result': [row()]}, {'ok': 1, 'result': [row()]},
                       {'ok': True, 'result': {}}, {'ok': True, 'result': [row()], 'truncated': 'true'}):
            with self.subTest(result=result):
                self.assertEqual(self.render([observation(result)])[1]['facts'], [])

    def test_returned_rows_must_obey_query_filters(self):
        for args, record in (({'product_ids': ['DEMO-A'], 'max_price_cents': 100}, row()),
                             ({'product_ids': ['DEMO-A'], 'in_stock': True}, row(stock=0))):
            with self.subTest(args=args):
                _, report = self.render([observation({'ok': True, 'result': [record]}, args)])
                self.assertEqual(report['missing'][0]['reason'], 'filter_mismatch')
                self.assertEqual(report['facts'], [])

    def test_duplicate_conflicting_rows_are_not_selected(self):
        item = observation({'ok': True, 'result': [row(), row(stock=2)]})
        self.assertEqual(self.render([item])[1]['missing'][0]['reason'], 'conflicting_rows')

    def test_equivalent_duplicates_are_deduplicated(self):
        self.assertEqual(len(self.render([observation({'ok': True, 'result': [row(), row()]})])[1]['facts']), 2)

    def test_latest_query_replaces_both_fields_and_source(self):
        first = observation()
        second = observation({'ok': True, 'result': [row(price=123, stock=5)]})
        reply, report = self.render([first, second])
        self.assertIn('¥1.23，库存 5', reply)
        self.assertNotIn('1167.66', reply)
        self.assertEqual(report['sources']['DEMO-A'], second['catalog_evidence']['observation_id'])
        self.assertNotEqual(first['catalog_evidence']['observation_id'], second['catalog_evidence']['observation_id'])

    def test_latest_failed_empty_or_invalid_query_does_not_revive_old_value(self):
        for result in ({'ok': False, 'error': 'offline'}, {'ok': True, 'result': []},
                       {'ok': True, 'result': [row(stock='bad')]}):
            with self.subTest(result=result):
                reply, report = self.render([observation(), observation(result)])
                self.assertEqual(report['facts'], [])
                self.assertNotIn('1167.66', reply)

    def test_latest_query_for_other_product_does_not_erase_first(self):
        trace = [observation(), observation({'ok': False, 'error': 'offline'}, {'product_ids': ['DEMO-B']})]
        self.assertEqual(self.render(trace)[1]['status'], 'bound')

    def test_empty_repeated_text_query_does_not_revive_previous_candidate(self):
        args = {'query': '运动耳机'}
        trace = [observation(args=args), observation({'ok': True, 'result': []}, args)]
        reply, report = self.render(trace, '查询耳机价格库存')
        self.assertEqual(report['facts'], [])
        self.assertNotIn('1167.66', reply)

    def test_truncation_keeps_returned_rows_without_inventing_missing_ones(self):
        item = observation({'ok': True, 'result': [row()], 'truncated': True},
                           {'product_ids': ['DEMO-A', 'DEMO-B']})
        reply, report = self.render([item], '查询 DEMO-A 和 DEMO-B 价格库存')
        self.assertEqual(report['status'], 'partial')
        self.assertIn('有截断', reply)
        self.assertEqual({f['product_id'] for f in report['facts']}, {'DEMO-A'})

    def test_generic_search_is_labelled_candidates(self):
        reply, _ = self.render([observation(args={'query': '耳机'})], '找有库存的耳机')
        self.assertIn('候选商品', reply)

    def test_amazon_and_explicit_custom_id_binding(self):
        for pid, question in (('amazon:us:B123', '查询 amazon:us:B123 价格库存'),
                              ('P001', '查询商品 ID P001 价格库存')):
            with self.subTest(pid=pid):
                item = observation({'ok': True, 'result': [row(pid)]}, {'product_ids': [pid]})
                self.assertEqual(self.render([item], question)[1]['status'], 'bound')

    def test_english_query_and_unqueried_claim_fail_closed(self):
        reply, report = self.render([], 'What is the price and stock of DEMO-A?')
        self.assertEqual(report['facts'], [])
        self.assertNotIn('999', reply)

    def test_unrequested_catalog_claim_in_product_answer_is_not_published(self):
        reply, _ = self.render([], '请介绍耳机', '价格999.88元，库存999件')
        self.assertNotIn('999', reply)

    def test_unlabelled_currency_and_stock_claims_are_not_published(self):
        for draft in ('只要999元', '¥999.88，有999件', '仅剩999台'):
            with self.subTest(draft=draft):
                reply, _ = self.render([], '这款耳机支持蓝牙吗', draft)
                self.assertNotIn('999', reply)

    def test_invalid_answer_plan_and_summary_are_ignored(self):
        for draft in ('{bad plan', '{"blocks":[{"price":99988}]}', '库存999件 [catalog_search]'):
            with self.subTest(draft=draft):
                reply, report = self.render([observation()], draft=draft)
                self.assertIn('¥1167.66', reply)
                self.assertNotIn(draft, reply)
                self.assertFalse(report['draft_published'])

    def test_missing_capture_never_reconstructs_trusted_facts(self):
        item = observation(); item.pop('catalog_evidence')
        self.assertEqual(self.render([item])[1]['missing'][0]['reason'], 'missing_observation')

    def test_immutable_fact_and_no_input_mutation(self):
        item = observation(); before = copy.deepcopy(item)
        fact = EvidenceFact.model_validate(item['catalog_evidence']['facts'][0])
        with self.assertRaises(ValidationError): fact.value = 999
        self.render([item])
        self.assertEqual(item, before)

    def test_non_catalog_answers_remain_outside_scope(self):
        for question, draft in (('你好', '欢迎光临'), ('查询 ORD00001 物流', '订单运输中'),
                                ('耳机支持 Windows 吗', '支持 Windows'), ('计算 1+1', '2')):
            with self.subTest(question=question):
                self.assertIsNone(render_catalog_reply(question, draft, []))

    def test_order_amount_and_memory_budget_are_not_current_catalog_claims(self):
        cases = [('我的耳机订单到哪里了', '订单运输中，成交金额299元',
                  [{'name': 'order_detail', 'agent': 'orders'}]),
                 ('记住我的耳机预算', '已记住预算299元',
                  [{'name': 'memory_set', 'agent': 'supervisor'}])]
        for question, draft, trace in cases:
            with self.subTest(question=question):
                self.assertIsNone(render_catalog_reply(question, draft, trace))

    def test_oversized_result_and_missing_source_fields_fail_closed(self):
        oversized = observation({'ok': True, 'result': [row(), row('DEMO-B')]}, {'limit': 1})
        self.assertEqual(oversized['catalog_evidence']['status'], 'invalid_result')
        self.assertEqual(self.render([oversized])[1]['facts'], [])


class CatalogGraphTests(ConfiguredAsyncTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MemoryStore(Path(self.temp.name)/'sessions.sqlite3')

    async def run_catalog(self, result, *, enabled=True, draft='虚构价格999.88元，库存999件',
                          question='查询商品 DEMO-A 的价格库存', arguments=None, session_id=None):
        model = AsyncMock(side_effect=[invoke('delegate_product', {'task': question}),
            invoke('catalog_search', {'product_ids': ['DEMO-A']} if arguments is None else arguments),
            text('专业报告中的价格999.88元，库存999件'), text(draft)])
        executor = Mock(return_value=result)
        with override_settings(CATALOG_EVIDENCE_ENABLED=enabled), patch('app.services.agent.complete', model), \
                patch('app.services.agent.execute_tool', executor):
            answer = await run_agent(question, session_id or self.store.create(), self.store)
        return answer, model, executor

    async def test_real_graph_renders_facts_without_extra_model_calls(self):
        answer, model, executor = await self.run_catalog({'ok': True, 'result': [row()]})
        self.assertIn('¥1167.66', answer['reply'])
        self.assertNotIn('999', answer['reply'])
        self.assertEqual(model.await_count, 4)
        self.assertEqual(executor.call_count, 1)
        self.assertEqual(answer['verification']['catalog']['status'], 'bound')
        self.assertIn('catalog_evidence', answer['tool_calls'][0])

    async def test_graph_empty_mismatch_and_wrong_fields(self):
        for result in ({'ok': True, 'result': []}, {'ok': True, 'result': [row('DEMO-B')]},
                       {'ok': True, 'result': [row(price='wrong', stock=True)]}):
            with self.subTest(result=result):
                answer, _, _ = await self.run_catalog(result)
                self.assertEqual(answer['verification']['catalog']['status'], 'insufficient')
                self.assertNotIn('999', answer['reply'])
                self.assertNotIn('¥', answer['reply'])

    async def test_invalid_json_and_plan_have_safe_server_output(self):
        for arguments, draft in (('{bad arguments', '{bad answer plan'),
                                 ({'product_ids': ['DEMO-A']}, '{bad answer plan')):
            with self.subTest(arguments=arguments):
                answer, _, _ = await self.run_catalog({'ok': True, 'result': [row()]}, arguments=arguments, draft=draft)
                self.assertNotIn(draft, answer['reply'])
                self.assertEqual(bool(answer['verification']['catalog']['facts']), isinstance(arguments, dict))

    async def test_flag_off_preserves_legacy_for_ab_comparison(self):
        result = {'ok': True, 'result': [row()]}
        old, _, _ = await self.run_catalog(result, enabled=False)
        new, _, _ = await self.run_catalog(result, enabled=True)
        self.assertIn('999.88', old['reply'])
        self.assertNotIn('catalog', old['verification'])
        self.assertNotIn('catalog_evidence', old['tool_calls'][0])
        self.assertIn('¥1167.66', new['reply'])

    async def test_repeated_calls_use_latest_observation_without_extra_model_call(self):
        model = AsyncMock(side_effect=[invoke('delegate_product', {'task': '查询 DEMO-A 价格库存'}),
            invoke('catalog_search', {'product_ids': ['DEMO-A']}),
            invoke('catalog_search', {'product_ids': ['DEMO-A']}), text('999元'), text('999元')])
        executor = Mock(side_effect=[{'ok': True, 'result': [row()]},
                                    {'ok': True, 'result': [row(price=123, stock=5)]}])
        with patch('app.services.agent.complete', model), patch('app.services.agent.execute_tool', executor):
            answer = await run_agent('查询 DEMO-A 价格库存', self.store.create(), self.store)
        self.assertIn('¥1.23，库存 5', answer['reply'])
        observations = [t['catalog_evidence'] for t in answer['tool_calls']]
        self.assertNotEqual(observations[0]['observation_id'], observations[1]['observation_id'])
        self.assertEqual(answer['verification']['catalog']['sources']['DEMO-A'], observations[1]['observation_id'])
        self.assertEqual(model.await_count, 5)

    async def test_independent_knowledge_consultation_preserves_negative_answer(self):
        question = 'DEMO-A 支持双设备连接吗'
        reply = '不支持双设备同时连接 [KB-012345abcdef]'
        model = AsyncMock(side_effect=[invoke('delegate_product', {'task': question}),
            invoke('knowledge_search', {'query': '双设备', 'product_id': 'DEMO-A'}), text(reply), text(reply)])
        executor = Mock(return_value={'ok': True, 'result': {'hits': [
            {'citation': 'KB-012345abcdef', 'product_id': 'DEMO-A', 'text': '不支持双设备同时连接。'}]}})
        with patch('app.services.agent.complete', model), patch('app.services.agent.execute_tool', executor):
            answer = await run_agent(question, self.store.create(), self.store)
        self.assertIn('不支持双设备同时连接', answer['reply'])
        self.assertIn('[KB-012345abcdef]', answer['reply'])
        self.assertEqual(answer['verification']['parts']['knowledge']['semantic_factuality'], 'not_measured')
        self.assertNotIn('catalog', answer['verification'])

    async def test_mixed_product_answer_only_publishes_catalog_facts(self):
        answer, _, _ = await self.run_catalog({'ok': True, 'result': [row()]},
            question='介绍 DEMO-A 的功能，并查询价格库存', draft='价格999元，支持自动对焦，订单已签收')
        self.assertIn('¥1167.66', answer['reply'])
        self.assertNotIn('支持自动对焦', answer['reply'])
        self.assertNotIn('订单已签收', answer['reply'])
        self.assertIn('资料不足', answer['reply'])
        self.assertEqual(answer['verification']['invalid'], [])

    async def test_greeting_and_order_business_keep_existing_path(self):
        scripts = [('你好', [text('欢迎光临')], Mock()),
                   ('查询 ORD00001 物流', [invoke('delegate_orders', {'task': '查询订单'}),
                    invoke('order_detail', {'order_id': 'ORD00001'}), text('运输中'), text('订单运输中')],
                    Mock(return_value={'ok': True, 'result': {'order_id': 'ORD00001', 'status': 'in_transit'}}))]
        for question, replies, executor in scripts:
            with self.subTest(question=question), patch('app.services.agent.complete', AsyncMock(side_effect=replies)), \
                    patch('app.services.agent.execute_tool', executor):
                answer = await run_agent(question, self.store.create(), self.store)
                self.assertNotIn('catalog', answer['verification'])
                self.assertIn('欢迎' if question == '你好' else '运输中', answer['reply'])

    async def test_api_and_stored_reply_use_same_rendered_output(self):
        sid = self.store.create()
        model = AsyncMock(side_effect=[invoke('delegate_product', {'task': '查询 DEMO-A 价格库存'}),
                                      invoke('catalog_search', {'product_ids': ['DEMO-A']}), text('999元'), text('999元')])
        with patch('app.api.routes.chat.get_store', return_value=self.store), \
                patch('app.services.agent.complete', model), \
                patch('app.services.agent.execute_tool', Mock(return_value={'ok': True, 'result': [row()]})), \
                TestClient(app) as client:
            response = client.post('/api/chat', json={'session_id': sid, 'message': '查询商品 DEMO-A 价格库存'})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn('¥1167.66', payload['reply'])
        stored = self.store.get(sid)['messages'][-1]
        self.assertEqual(stored['content'], payload['reply'])
        self.assertEqual(stored['verification']['catalog'], payload['verification']['catalog'])

    async def test_real_local_and_mcp_catalog_share_validation(self):
        database = Path(self.temp.name)/'commerce.sqlite3'
        store = CommerceStore(database)
        with store.connection() as db:
            db.execute('INSERT INTO products VALUES (?,?,?,?,?,?,?,?)', ('DEMO-A', '耳机', '', 100, 0, 'synthetic', 'A', 'demo'))
        for transport in ('local', 'mcp'):
            with self.subTest(transport=transport), override_settings(COMMERCE_DB_PATH=database, COMMERCE_TOOL_TRANSPORT=transport):
                model = AsyncMock(side_effect=[invoke('delegate_product', {'task': '查询 DEMO-A 价格库存'}),
                    invoke('catalog_search', {'product_ids': ['DEMO-A']}), text('库存999件'), text('库存999件')])
                with patch('app.services.agent.complete', model):
                    answer = await run_agent('查询商品 DEMO-A 价格库存', self.store.create(), self.store)
                self.assertIn('¥1.00，库存 0 件', answer['reply'])
                self.assertEqual(answer['verification']['catalog']['status'], 'bound')
