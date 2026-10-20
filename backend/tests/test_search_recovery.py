from tests.support import ConfiguredAsyncTestCase
"""Generic historical execution handling: retries are not hardwired to web search."""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from openai.types.chat import ChatCompletionMessage

from app.services.agent import build_context, run_agent
from app.services.history import execution_state, prepare_history
from app.services.memory import MemoryStore


def answer(text, tool=None):
    return ChatCompletionMessage(role='assistant', content=text, tool_calls=None if tool is None else [
        {'id': 'tool-1', 'type': 'function', 'function': {'name': tool, 'arguments': '{"query":"耳机新闻"}'}}])


def history(name='web_search', ok=False):
    return [{'role': 'user', 'content': '查询商品资料'},
            {'role': 'assistant', 'content': '服务不可用', 'tool_calls': [{'name': name, 'result': {'ok': ok}}]}]


class RecoveryTests(ConfiguredAsyncTestCase):
    async def test_fail_fail_recovery_retains_logs_but_not_failed_prose(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.db')
            sid = store.create()
            snapshots = []
            async def model(messages, schemas, final=False):
                snapshots.append(json.loads(json.dumps(messages)))
                names = {item['function']['name'] for item in schemas}
                if 'delegate_research' in names:
                    if messages[-1]['role'] == 'tool':
                        report = json.loads(messages[-1]['content'])['result']
                        return answer('服务不可用' if report['execution_state'] == 'failed' else '已恢复')
                    return ChatCompletionMessage(role='assistant', content=None, tool_calls=[{
                        'id': 'handoff', 'type': 'function', 'function': {
                            'name': 'delegate_research', 'arguments': '{"task":"查询耳机公开新闻"}'}}])
                if messages[-1]['role'] == 'tool':
                    return answer('服务不可用' if not json.loads(messages[-1]['content'])['ok'] else '已恢复')
                return answer(None, 'web_search')
            with patch('app.services.agent.complete', model), patch('app.services.agent.execute_tool', side_effect=[
                {'ok': False, 'error': 'timeout'}, {'ok': False, 'error': 'timeout'}, {'ok': True, 'result': []}]) as tool:
                for _ in range(3):
                    result = await run_agent('查询商品资料', sid, store)
            self.assertEqual(tool.call_count, 3)
            self.assertEqual(result['reply'], '已恢复')
            self.assertEqual(store.get(sid)['messages'][1]['content'], '服务不可用')
            initial_third = snapshots[8]
            self.assertNotIn('服务不可用', json.dumps(initial_third, ensure_ascii=False))
            self.assertIn('historical_turn_only', json.dumps(initial_third))

    async def test_retry_after_normal_answer_does_not_force_any_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.db')
            sid = store.create()
            store.save(sid, [{'role': 'user', 'content': '写一句欢迎语'}, {'role': 'assistant', 'content': '欢迎光临'}], {})
            with patch('app.services.agent.complete', AsyncMock(return_value=answer('很高兴为您服务'))) as model, patch('app.services.agent.execute_tool') as tool:
                await run_agent('再试一次', sid, store)
            self.assertNotIn('required_tool', model.call_args.kwargs)
            tool.assert_not_called()

    def test_all_tools_share_failure_policy(self):
        for name in ('web_search', 'order_detail', 'knowledge_search', 'calculator'):
            original = history(name)
            context = prepare_history(original)
            self.assertNotIn('服务不可用', context[1]['content'])
            self.assertIn(name, context[1]['content'])
            self.assertEqual(original[1]['content'], '服务不可用')
            self.assertEqual(context[0]['content'], '查询商品资料')

    def test_mixed_observations_preserve_partial_answer(self):
        original = history()
        original[-1]['tool_calls'].append({'name': 'order_detail', 'result': {'ok': True}})
        self.assertEqual(execution_state(original[-1]['tool_calls']), 'mixed')
        self.assertIn('服务不可用', prepare_history(original)[-1]['content'])

    def test_legacy_no_tool_answer_not_guessed_from_words(self):
        original = history()
        original[-1]['tool_calls'] = []
        self.assertEqual(prepare_history(original)[-1]['content'], '服务不可用')

    def test_failed_specialist_without_tools_is_historical_failure(self):
        original = history()
        original[-1]['tool_calls'] = []
        original[-1]['agent_events'] = [{'agent': 'orders', 'status': 'failed'}]
        content = prepare_history(original)[-1]['content']
        self.assertNotIn('服务不可用', content)
        self.assertIn('orders', content)

    def test_memories_are_not_changed(self):
        memories = {'语言': '中文'}
        build_context(history(), memories, '你好')
        self.assertEqual(memories, {'语言': '中文'})
