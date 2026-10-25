"""Multi-agent graph integration with a scripted LLM and real tool/runtime boundaries."""
from tests.support import ConfiguredAsyncTestCase
from app.core.config import override_settings
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from openai.types.chat import ChatCompletionMessage

from app.agents.graph import build_team_graph, bounded_report
from app.agents.roles import HANDOFFS, ROLES, SUPERVISOR_SCHEMAS, role_schemas
from app.commerce.store import CommerceStore
from app.services.agent import run_agent
from app.services.memory import MemoryStore
from app.services.qwen import ModelServiceError
from app.services.tools import TOOL_SCHEMAS, execute_tool
from scripts.generate_mock_data import generate


def text(value):
    return ChatCompletionMessage(role='assistant', content=value)


def call(name, arguments, id='call-1'):
    return {'id': id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}


def calls(*items):
    return ChatCompletionMessage(role='assistant', content=None, tool_calls=list(items))


class MultiAgentTests(ConfiguredAsyncTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.store = MemoryStore(self.path / 'sessions.db')
        self.sid = self.store.create()

    def tearDown(self):
        self.temp.cleanup()

    async def run_script(self, replies, executor=None, question='测试任务'):
        model = AsyncMock(side_effect=replies)
        executor = executor or Mock(return_value={'ok': True, 'result': []})
        with patch('app.services.agent.complete', model), patch('app.services.agent.execute_tool', executor):
            result = await run_agent(question, self.sid, self.store)
        return result, model, executor

    async def test_chat_does_not_delegate_and_restores_events(self):
        result, model, executor = await self.run_script([text('欢迎光临')])
        self.assertEqual(model.await_count, 1)
        executor.assert_not_called()
        self.assertEqual(result['agent_events'][0]['agent'], 'supervisor')
        self.assertEqual(self.store.get(self.sid)['messages'][-1]['agent_events'], result['agent_events'])

    async def test_unqueried_product_claims_are_blocked_even_if_model_ignores_tool_choice(self):
        result, model, executor = await self.run_script(
            [text('价格299元，库存15件，支持自动对焦。来源 [catalog_search]')],
            question='请介绍摄像头 SYN-CAM-0015 的适用场景，并查询价格和库存。')
        executor.assert_not_called()
        self.assertEqual(model.call_args.kwargs['required_tool'], 'delegate_product')
        self.assertNotIn('299', result['reply'])
        self.assertIn('catalog_search', result['verification']['draft_checks']['invalid'])
        self.assertEqual(result['verification']['invalid'], [])
        self.assertEqual(result['verification']['missing_required_tools'], ['knowledge_search', 'catalog_search'])

    async def test_legacy_product_requires_both_sources_and_preserves_negative_capability(self):
        question = '介绍摄像头 SYN-CAM-0015 的适用场景，并查询价格和库存'
        def execute(name, arguments, memories):
            if name == 'knowledge_search':
                return {'ok': True, 'result': {'hits': [{'citation': 'KB-012345abcdef', 'text': '适合视频会议，不支持自动对焦。'}]}}
            return {'ok': True, 'result': [{'product_id': 'SYN-CAM-0015', 'price_cents': 116766, 'stock': 23}]}
        # A/B baseline keeps the old mixed free-text answer. The new MVP publishes
        # catalog fields only; independent knowledge replies are tested separately.
        with override_settings(CATALOG_EVIDENCE_ENABLED=False):
            result, model, _ = await self.run_script([
                calls(call('delegate_product', {'task': question})),
                calls(call('knowledge_search', {'query': '适用场景', 'product_id': 'SYN-CAM-0015'})),
                calls(call('catalog_search', {'product_ids': ['SYN-CAM-0015']})),
                text('适合视频会议，不支持自动对焦 [KB-012345abcdef]；1167.66元，库存23件'),
                text('适合视频会议，不支持自动对焦 [KB-012345abcdef]；1167.66元，库存23件'),
            ], Mock(side_effect=execute), question=question)
        self.assertEqual(model.call_args_list[1].kwargs['required_tool'], 'knowledge_search')
        self.assertEqual(model.call_args_list[2].kwargs['required_tool'], 'catalog_search')
        self.assertEqual(result['verification']['missing_required_tools'], [])
        self.assertEqual(result['verification']['invalid'], [])
        self.assertIn('不支持自动对焦', result['reply'])

    async def test_failed_catalog_query_cannot_support_fabricated_price(self):
        question = '查询商品 SYN-CAM-0015 价格库存'
        result, model, executor = await self.run_script([
            calls(call('delegate_product', {'task': question})),
            calls(call('catalog_search', {'product_ids': ['SYN-CAM-0015']})),
            text('299元，15件'), text('299元，15件'),
        ], Mock(return_value={'ok': False, 'error': 'database unavailable'}), question=question)
        self.assertEqual(executor.call_count, 1)
        self.assertNotIn('required_tool', model.call_args_list[2].kwargs)
        self.assertEqual(result['verification']['missing_required_tools'], ['catalog_search'])
        self.assertNotIn('299', result['reply'])

    async def test_batched_delegations_are_all_answered_before_supervisor(self):
        result, model, _ = await self.run_script([
            calls(call('delegate_orders', {'task': '查询订单'}, 'a'), call('delegate_product', {'task': '查询退货政策'}, 'b')),
            calls(call('list_orders', {})), text('订单记录已查'),
            calls(call('knowledge_search', {'query': '退货', 'kind': 'policy'})), text('政策已查'),
            text('已整理订单与政策'),
        ])
        messages = model.call_args.args[0]
        responses = [item for item in messages if item['role'] == 'tool']
        self.assertEqual([item['tool_call_id'] for item in responses], ['a', 'b'])
        self.assertEqual([item['agent'] for item in result['tool_calls']], ['orders', 'product'])
        self.assertIn('evidence', json.loads(responses[0]['content'])['result'])

    async def test_agent_permissions_enforced_even_for_fabricated_calls(self):
        result, model, executor = await self.run_script([
            calls(call('delegate_product', {'task': '查询耳机'})),
            calls(call('order_detail', {'order_id': 'ORD00001'}), call('memory_set', {'key': 'x', 'value': 'y'}, 'b')),
            text('没有权限访问订单'), text('请说明商品型号'),
        ])
        executor.assert_not_called()
        self.assertEqual(self.store.get(self.sid)['memories'], {})
        self.assertTrue(all(item['result']['code'] == 'tool_not_allowed' for item in result['tool_calls']))
        worker_schema = {item['function']['name'] for item in model.call_args_list[1].args[1]}
        self.assertEqual(worker_schema, ROLES['product']['tools'])

    async def test_supervisor_cannot_bypass_worker(self):
        result, _, executor = await self.run_script([calls(call('order_detail', {'order_id': 'ORD00001'})), text('请重试')])
        executor.assert_not_called()
        self.assertEqual(result['tool_calls'][0]['result']['code'], 'tool_not_allowed')

    async def test_worker_model_failure_preserves_collected_evidence(self):
        observed = {'ok': True, 'result': {'order_id': 'ORD00001', 'status': 'in_transit'}}
        result, model, _ = await self.run_script([
            calls(call('delegate_orders', {'task': '查询订单'})), calls(call('order_detail', {'order_id': 'ORD00001'})),
            ModelServiceError(502, '专业模型暂时不可用'), text('订单运输中，详细解释暂时不可用'),
        ], Mock(return_value=observed))
        self.assertEqual(result['agent_events'][0]['status'], 'failed')
        report = json.loads(model.call_args.args[0][-1]['content'])
        self.assertFalse(report['ok'])
        self.assertEqual(report['result']['evidence'][0]['observation'], observed)
        self.assertTrue(result['tool_calls'][0]['result']['ok'])

    async def test_product_does_not_receive_history_or_explicit_memory(self):
        self.store.save(self.sid, [{'role': 'user', 'content': '内部订单 SECRET-ORDER'},
                                   {'role': 'assistant', 'content': '已记下订单上下文'}], {'电话': 'PRIVATE-PHONE'})
        _, model, _ = await self.run_script([
            calls(call('delegate_product', {'task': '查询耳机兼容性'})),
            calls(call('knowledge_search', {'query': '耳机兼容性'})), text('商品知识报告'), text('已整理商品知识'),
        ])
        worker_messages = json.dumps(model.call_args_list[1].args[0], ensure_ascii=False)
        self.assertNotIn('SECRET-ORDER', worker_messages)
        self.assertNotIn('PRIVATE-PHONE', worker_messages)
        self.assertIn('耳机兼容性', worker_messages)

    async def test_removed_search_calls_are_rejected_without_executing_tools(self):
        schemas = TOOL_SCHEMAS + SUPERVISOR_SCHEMAS
        names = {item['function']['name'] for item in schemas}
        self.assertNotIn('web_search', names)
        self.assertNotIn('delegate_research', names)
        self.assertFalse(execute_tool('web_search', '{"query":"耳机新闻"}', {})['ok'])
        result, _, executor = await self.run_script([
            calls(call('delegate_research', {'task': '查询耳机新闻'}),
                  call('web_search', {'query': '耳机新闻'}, 'search')),
            text('请咨询本店商品、订单或售后问题'),
        ])
        executor.assert_not_called()
        self.assertTrue(all(item['result']['code'] == 'tool_not_allowed' for item in result['tool_calls']))

    async def test_handoff_and_global_tool_budgets(self):
        with override_settings(AGENT_MAX_HANDOFFS=1):
            result, model, executor = await self.run_script([
                calls(call('delegate_orders', {'task': '查询订单'}, 'a'), call('delegate_product', {'task': '查询政策'}, 'b')),
                calls(call('list_orders', {})), text('查到了'), text('政策查询达到上限'),
            ])
        self.assertEqual(executor.call_count, 1)
        self.assertEqual(model.await_count, 4)
        self.assertTrue(result['stopped_by_limit'])
        self.assertEqual(result['agent_events'][1]['status'], 'limit')

        with override_settings(AGENT_MAX_TOOL_CALLS=1):
            result, model, executor = await self.run_script([
                calls(call('delegate_orders', {'task': '查询订单'})),
                calls(call('list_orders', {}), call('current_time', {}, 'b')),
                text('已有订单列表'), text('列出了订单，未核验当前日期'),
            ])
        self.assertEqual(executor.call_count, 1)
        self.assertTrue(model.call_args_list[2].kwargs['final'])
        self.assertTrue(model.call_args.kwargs['final'])
        self.assertTrue(result['stopped_by_limit'])

    async def test_malformed_handoff_never_starts_worker(self):
        result, model, executor = await self.run_script([calls(call('delegate_orders', {'task': ' '})), text('请说明订单')])
        self.assertEqual(model.await_count, 2)
        executor.assert_not_called()
        self.assertEqual(result['agent_events'][0]['status'], 'failed')

    async def test_supervisor_failure_rolls_back_worker_turn_and_memory(self):
        with self.assertRaises(ModelServiceError):
            await self.run_script([
                calls(call('memory_set', {'key': '语言', 'value': '中文'})),
                calls(call('delegate_orders', {'task': '查询订单'})),
                calls(call('list_orders', {})), text('查到了'), ModelServiceError(502, '协调模型失败'),
            ])
        self.assertEqual(self.store.get(self.sid)['messages'], [])
        self.assertEqual(self.store.get(self.sid)['memories'], {})
        with self.store.edit(self.sid):
            pass

    async def test_mcp_scope_across_two_worker_tool_nodes(self):
        database = self.path / 'commerce.db'
        generate(database)
        with override_settings({'COMMERCE_TOOL_TRANSPORT': 'mcp', 'COMMERCE_DB_PATH': str(database)}):
            result, _, _ = await self.run_script([
                calls(call('delegate_orders', {'task': '查询 ORD00001'})),
                calls(call('list_orders', {})), calls(call('order_detail', {'order_id': 'ORD00001'})),
                text('ORD00001 运输中'), text('订单运输中'),
            ])
        self.assertTrue(all(item['result']['ok'] for item in result['tool_calls']))
        self.assertEqual(result['tool_calls'][1]['result']['result']['order_id'], 'ORD00001')

    def test_real_langgraph_topology(self):
        graph = build_team_graph(AsyncMock(), Mock(), max_steps=2, worker_steps=2,
                                 max_handoffs=2, max_tool_calls=8, context_bytes=12000)
        nodes = graph.get_graph().nodes
        self.assertEqual(set(nodes), {'__start__', '__end__', 'decision', 'verify', 'supervisor', 'orders', 'product', 'utility'})
        for role in ROLES:
            self.assertNotIn('memory_set', {item['function']['name'] for item in role_schemas(role)})

    def test_large_handoff_keeps_latest_evidence_without_mutating_trace(self):
        evidence = {'ok': True, 'result': {'hits': [{'citation': f'KB-{i}', 'text': '长' * 700} for i in range(4)]}}
        worker = {'reply': '报告' * 600, 'trace': [{'name': 'knowledge_search', 'arguments': '{}', 'result': evidence}]}
        report = bounded_report(worker)
        self.assertLessEqual(len(json.dumps(report, ensure_ascii=False).encode()), 6500)
        self.assertTrue(report['evidence_truncated'])
        self.assertEqual(report['evidence'][0]['observation']['result']['hits'][0]['citation'], 'KB-0')
        self.assertEqual(len(evidence['result']['hits']), 4)
