from tests.support import ConfiguredAsyncTestCase, ConfiguredTestCase
from app.core.config import get_settings, override_settings
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessage

from app.main import app
from app.services.agent import build_context, run_agent, size
from app.services.memory import MemoryStore
from app.services.qwen import ModelServiceError
from app.services.tools import execute_tool


def answer(content='完成', calls=None):
    return ChatCompletionMessage(role='assistant', content=content, tool_calls=calls)


def call(name, args, id='call_1'):
    return {'id': id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class AgentTests(ConfiguredAsyncTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / 'test.sqlite3')
        self.sid = self.store.create()

    def tearDown(self):
        self.temp.cleanup()

    async def test_tool_observations_then_final_and_followup(self):
        complete = AsyncMock(side_effect=[answer(None, [call('calculator', {'expression': '(128+256)*3/2'})]), answer('576')])
        with patch('app.services.agent.complete', complete):
            result = await run_agent('计算一下', self.sid, self.store)
        self.assertEqual(result['reply'], '576')
        self.assertEqual(result['tool_calls'][0]['result']['result'], 576)
        messages = complete.call_args.args[0]
        observation = next(m for m in messages if m['role'] == 'tool')
        self.assertEqual(observation['tool_call_id'], 'call_1')
        with patch('app.services.agent.complete', AsyncMock(return_value=answer('上一题是 576'))) as model:
            await run_agent('上一题结果？', self.sid, self.store)
            self.assertTrue(any((m.get('content') or '').startswith('576') for m in model.call_args.args[0]))

    async def test_memory_persistence_and_session_isolation(self):
        with patch('app.services.agent.complete', AsyncMock(side_effect=[
            answer(None, [call('memory_set', {'key': '语言', 'value': '中文'})]), answer('已记住')])):
            await run_agent('请记住我喜欢中文', self.sid, self.store)
        reopened = MemoryStore(self.store.path)
        self.assertEqual(reopened.get(self.sid)['memories'], {'语言': '中文'})
        self.assertEqual(reopened.get(reopened.create())['memories'], {})

    async def test_error_rolls_back_memory_and_releases_claim(self):
        with patch('app.services.agent.complete', AsyncMock(side_effect=[
            answer(None, [call('memory_set', {'key': '语言', 'value': '中文'})]), ModelServiceError(502, '失败')])):
            with self.assertRaises(ModelServiceError):
                await run_agent('记住中文', self.sid, self.store)
        self.assertEqual(self.store.get(self.sid)['memories'], {})
        self.assertEqual(self.store.get(self.sid)['messages'], [])
        with self.store.edit(self.sid):
            pass

    async def test_tool_failure_returned_to_model_and_recovered(self):
        with patch('app.services.agent.complete', AsyncMock(side_effect=[
            answer(None, [call('calculator', {'expression': '1/0'}), call('unknown', {}, 'call_2')]),
            answer(None, [call('calculator', {'expression': '1+1'})]), answer('2')])):
            result = await run_agent('计算', self.sid, self.store)
        self.assertEqual([t['result']['ok'] for t in result['tool_calls']], [False, False, True])

    async def test_step_limit_forces_final(self):
        with override_settings(AGENT_MAX_STEPS=1), patch('app.services.agent.complete', AsyncMock(side_effect=[
            answer(None, [call('current_time', {})]), answer('时间已查到')])) as model:
            result = await run_agent('时间', self.sid, self.store)
        self.assertTrue(result['stopped_by_limit'])
        self.assertTrue(model.call_args.kwargs['final'])
        self.assertEqual(model.await_count, 2)

    async def test_timeout_and_cancellation_release_session(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(10)
        with override_settings(AGENT_TIMEOUT_SECONDS=0.01), patch('app.services.agent.complete', slow):
            with self.assertRaises(ModelServiceError) as error:
                await run_agent('等待', self.sid, self.store)
            self.assertEqual(error.exception.status_code, 504)
        with patch('app.services.agent.complete', AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await run_agent('取消', self.sid, self.store)
        with self.store.edit(self.sid):
            pass

    def test_busy_session_blocks_other_writers(self):
        with self.store.edit(self.sid):
            with self.assertRaises(RuntimeError):
                with self.store.edit(self.sid):
                    pass
            with self.assertRaises(RuntimeError):
                self.store.delete(self.sid)

    def test_context_keeps_whole_pairs_within_budget(self):
        history = [{'role': role, 'content': str(i) + '长' * 200} for i in range(100) for role in ('user', 'assistant')]
        messages = build_context(history, {}, '问题')
        self.assertLessEqual(size(messages), get_settings().agent_context_bytes)
        self.assertLess(len(messages), len(history))
        self.assertEqual(messages[-2]['role'], 'assistant')
        self.assertEqual((len(messages) - 3) % 2, 0)
        self.assertTrue(messages[-2]['content'].startswith('99'))

    async def test_context_pressure_keeps_tool_protocol_intact(self):
        history = [{'role': role, 'content': '长' * 400} for _ in range(20) for role in ('user', 'assistant')]
        self.store.save(self.sid, history, {})
        recorded = []
        async def model(messages, tools, final=False):
            recorded.append(json.loads(json.dumps(messages)))
            if final:
                return answer('完成')
            return answer(None, [call('calculator', {'expression': '1+1'}, str(len(recorded)))])
        with override_settings(AGENT_CONTEXT_BYTES=4500), patch('app.services.agent.complete', model):
            await run_agent('计算', self.sid, self.store)
        for messages in recorded:
            self.assertLessEqual(size(messages), 4500)
            ids = set()
            for message in messages:
                for tool in message.get('tool_calls', []):
                    ids.add(tool['id'])
                if message['role'] == 'tool':
                    self.assertIn(message['tool_call_id'], ids)

    async def test_empty_response_does_not_save_history(self):
        with patch('app.services.agent.complete', AsyncMock(return_value=answer(''))):
            with self.assertRaises(ModelServiceError):
                await run_agent('问题', self.sid, self.store)
        self.assertEqual(self.store.get(self.sid)['messages'], [])

    def test_expired_session_claim_can_be_recovered(self):
        with self.store.connection() as db:
            db.execute('UPDATE sessions SET busy=1 WHERE id=?', (self.sid,))
        with self.store.edit(self.sid):
            pass


class ToolTests(ConfiguredTestCase):
    def test_rejects_code_and_expensive_arithmetic(self):
        for expression in ["__import__('os').system('echo unsafe')", '2**1000000', 'True+1', '1/0', '1e999', '(-1)**0.5']:
            with self.subTest(expression=expression):
                self.assertFalse(execute_tool('calculator', json.dumps({'expression': expression}), {})['ok'])

    def test_arguments_unknown_tools_and_timezones(self):
        for name, args in [('missing', '{}'), ('calculator', 'bad'), ('calculator', '{"expression": 3}'),
                           ('current_time', '{"timezone":"Not/AZone"}'), ('memory_set', '{"key":" ","value":"x"}')]:
            self.assertFalse(execute_tool(name, args, {})['ok'])
        self.assertTrue(execute_tool('current_time', '{}', {})['ok'])

    def test_memory_capacity_update_and_delete(self):
        memories = {}
        self.assertTrue(execute_tool('memory_set', '{"key":"a","value":"b"}', memories)['ok'])
        self.assertTrue(execute_tool('memory_set', '{"key":"a","value":"c"}', memories)['ok'])
        self.assertEqual(memories, {'a': 'c'})
        self.assertTrue(execute_tool('memory_delete', '{"key":"a"}', memories)['result']['deleted'])
        self.assertEqual(memories, {})
        full = {str(i): 'value' for i in range(20)}
        self.assertFalse(execute_tool('memory_set', '{"key":"extra","value":"b"}', full)['ok'])
        self.assertNotIn('extra', full)


class ApiTests(ConfiguredTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / 'api.sqlite3')
        self.patch = patch('app.api.routes.chat.get_store', return_value=self.store)
        self.patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.patch.stop()
        self.client.close()
        self.temp.cleanup()

    def test_session_memory_chat_restore_delete(self):
        sid = self.client.post('/api/sessions').json()['session_id']
        url = f'/api/sessions/{sid}'
        self.assertEqual(self.client.put(url + '/memories', json={'key': '语言', 'value': '中文'}).status_code, 200)
        with patch('app.services.agent.complete', AsyncMock(return_value=answer('你好'))):
            response = self.client.post('/api/chat', json={'message': '您好', 'session_id': sid})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['memories'], {'语言': '中文'})
        self.assertEqual(len(self.client.get(url).json()['messages']), 2)
        self.assertEqual(self.client.delete(url + '/memories/语言').json()['memories'], {})
        self.assertEqual(self.client.put(url + '/memories', json={'key': 'a/b', 'value': '测试'}).status_code, 200)
        self.assertEqual(self.client.delete(url + '/memories/a%2Fb').json()['memories'], {})
        self.assertEqual(self.client.delete(url).status_code, 200)
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.post('/api/chat', json={'message': '你好', 'session_id': sid}).status_code, 404)

    def test_validation_busy_and_legacy_request(self):
        self.assertEqual(self.client.post('/api/chat', json={'message': '  '}).status_code, 422)
        self.assertEqual(self.client.post('/api/chat', json={'message': '你好', 'session_id': '../x'}).status_code, 422)
        sid = self.store.create()
        with self.store.edit(sid):
            self.assertEqual(self.client.post('/api/chat', json={'message': '你好', 'session_id': sid}).status_code, 409)
        with patch('app.services.agent.complete', AsyncMock(return_value=answer('你好'))):
            response = self.client.post('/api/chat', json={'message': '你好'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['session_id'])


if __name__ == '__main__':
    unittest.main()
