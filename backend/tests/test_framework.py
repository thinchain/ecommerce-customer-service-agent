"""Framework contracts: message fidelity, settings isolation and stateless execution."""
import asyncio
from contextlib import chdir
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.agents.team_agent import LangGraphTeamAgent
from app.agents.graph import bounded_report
from app.core import Agent, AgentConfig, AgentResult, ConversationMessage, Message, RunContext, Settings
from app.core.config import PROJECT_ROOT, get_settings, override_settings
from app.core.exceptions import ModelServiceError
from app.core.message import model_messages
from app.services.agent import run_agent
from app.services.memory import MemoryStore
from app.services.qwen import complete, initialize_model_client, close_model_client
from tests.support import ConfiguredAsyncTestCase, ConfiguredTestCase


def call(identifier='a'):
    return {'id': identifier, 'type': 'function',
            'function': {'name': 'calculator', 'arguments': '{"expression":"1+1"}'}}


class MessageTests(ConfiguredTestCase):
    def test_protocol_keeps_calls_and_excludes_internal_metadata(self):
        assistant = Message(role='assistant', tool_calls=[call()], metadata={'private': 'internal'})
        tool = Message(role='tool', tool_call_id='a', content='2')
        wire = model_messages([Message(role='user', content='1+1'), assistant, tool])
        self.assertIsNone(wire[1]['content'])
        self.assertEqual(wire[1]['tool_calls'], [call()])
        self.assertEqual(wire[2]['tool_call_id'], 'a')
        self.assertNotIn('metadata', json.dumps(wire))
        self.assertNotIn('timestamp', json.dumps(wire))
        self.assertIsNotNone(assistant.timestamp.tzinfo)

    def test_invalid_roles_and_tool_fields_are_rejected(self):
        for record in ({'role': 'unknown', 'content': 'x'}, {'role': 'tool', 'content': '2'},
                       {'role': 'user', 'content': 'x', 'tool_calls': [call()]},
                       {'role': 'assistant', 'tool_calls': [call(), call()]},
                       {'role': 'user', 'content': 'x', 'tool_call_id': 'a'}):
            with self.subTest(record=record), self.assertRaises(ValidationError):
                Message.model_validate(record)

    def test_dangling_duplicate_and_orphan_tool_results_are_rejected(self):
        assistant = Message(role='assistant', tool_calls=[call(), call('b')])
        tool = Message(role='tool', tool_call_id='a', content='2')
        for records in ([tool], [assistant], [assistant, tool], [assistant, tool, tool],
                        [assistant, Message(role='user', content='interrupt')]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                model_messages(records)

    def test_ids_can_be_reused_in_separate_completed_groups(self):
        group = [Message(role='assistant', tool_calls=[call()]),
                 Message(role='tool', tool_call_id='a', content='2')]
        self.assertEqual(len(model_messages([*group, *group])), 4)

    def test_legacy_storage_roundtrip_does_not_invent_fields(self):
        legacy = {'role': 'assistant', 'content': '旧回答', 'tool_calls': [
            {'name': 'calculator', 'result': {'ok': True}}], 'custom_audit': {'version': 1}}
        record = ConversationMessage.from_storage(legacy)
        self.assertEqual(record.to_storage(), legacy)
        self.assertIsNone(record.timestamp)
        self.assertIsNone(record.id)
        self.assertEqual(record.to_message().to_dict(), {'role': 'assistant', 'content': '旧回答'})

    def test_new_records_have_ids_timestamps_and_independent_defaults(self):
        first = ConversationMessage.new('user', 'hello')
        second = ConversationMessage.new('user', 'hello')
        self.assertNotEqual(first.id, second.id)
        self.assertTrue(first.to_storage()['timestamp'].endswith('Z'))
        first.metadata['x'] = 'y'
        self.assertEqual(second.metadata, {})


class SettingsTests(ConfiguredTestCase):
    def test_storage_root_and_individual_override_follow_configuration_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dotenv = root / '.env'
            dotenv.write_text(f'DATA_DIR={root / "file-data"}\nMEMORY_DB_PATH={root / "private-memory.sqlite3"}\n')
            with patch.dict(os.environ, {'DATA_DIR': str(root/'environment-data')}, clear=True):
                settings = Settings.from_env(dotenv)
                self.assertEqual(settings.commerce_db_path, root/'environment-data/commerce.sqlite3')
                self.assertEqual(settings.knowledge_db_path, root/'environment-data/knowledge.sqlite3')
                self.assertEqual(settings.chroma_path, root/'environment-data/chroma')
                self.assertEqual(settings.memory_db_path, root/'private-memory.sqlite3')
                explicit = Settings(_env_file=dotenv, DATA_DIR=root/'explicit-data')
                self.assertEqual(explicit.commerce_db_path, root/'explicit-data/commerce.sqlite3')
            self.assertFalse((root/'environment-data').exists())

    def test_relative_storage_paths_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory, chdir(directory), patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=None, DATA_DIR='data/demo', MEMORY_DB_PATH='data/custom.sqlite3')
        self.assertEqual(settings.data_dir, PROJECT_ROOT/'data/demo')
        self.assertEqual(settings.commerce_db_path, PROJECT_ROOT/'data/demo/commerce.sqlite3')
        self.assertEqual(settings.memory_db_path, PROJECT_ROOT/'data/custom.sqlite3')

    def test_storage_root_override_rebases_defaults_and_keeps_custom_paths(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            base = Settings(_env_file=None, DATA_DIR=root/'original', MEMORY_DB_PATH=root/'custom-memory.sqlite3')
            with override_settings(base), override_settings(DATA_DIR=root/'replacement') as settings:
                self.assertEqual(settings.commerce_db_path, root/'replacement/commerce.sqlite3')
                self.assertEqual(settings.memory_db_path, root/'custom-memory.sqlite3')
                with override_settings(DATA_DIR=root/'nested', COMMERCE_DB_PATH=root/'custom-commerce.sqlite3') as nested:
                    self.assertEqual(nested.commerce_db_path, root/'custom-commerce.sqlite3')
                    self.assertEqual(nested.knowledge_db_path, root/'nested/knowledge.sqlite3')
            self.assertEqual(base.commerce_db_path, root/'original/commerce.sqlite3')

    def test_env_file_process_env_and_explicit_arguments_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / '.env'
            dotenv.write_text('QWEN_MODEL=file-model\nAGENT_MAX_STEPS=2\nEXTRA_SCRIPT_SETTING=ignored\n')
            with patch.dict(os.environ, {'QWEN_MODEL': 'env-model'}, clear=True):
                settings = Settings.from_env(dotenv)
                self.assertEqual(settings.model_name, 'env-model')
                self.assertEqual(settings.agent.max_steps, 2)
                self.assertEqual(Settings(_env_file=dotenv, model_name='explicit').model_name, 'explicit')

    def test_invalid_values_are_rejected_instead_of_clamped(self):
        for values in ({'AGENT_MAX_STEPS': 0}, {'AGENT_MAX_STEPS': 9}, {'MODEL_RETRIES': 3},
                       {'MODEL_TIMEOUT_SECONDS': 'nan'}, {'AGENT_TIMEOUT_SECONDS': 'inf'},
                       {'MODEL_TEMPERATURE': -1}, {'QWEN_MODEL': ' '}, {'OPENAI_BASE_URL': ''},
                       {'OPENAI_BASE_URL': 'http://'}, {'OPENAI_BASE_URL': 'http://localhost:99999'},
                       {'RAG_BACKEND': 'other'}, {'COMMERCE_TOOL_TRANSPORT': 'other'},
                       {'JEV_MODE': 'other'}, {'JEV_TIMEOUT_SECONDS': 0},
                       {'DATA_DIR': ''}, {'COMMERCE_DB_PATH': ''}):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                with override_settings(values):
                    self.fail('Invalid settings must not be activated')

    def test_chroma_requires_embedding_configuration(self):
        with self.assertRaises(ValidationError):
            with override_settings(rag_backend='chroma'):
                pass
        with override_settings(rag_backend='chroma', embedding_base_url='http://localhost:8002/v1/',
                               embedding_model='embedding') as settings:
            self.assertEqual(settings.embedding_base_url, 'http://localhost:8002/v1')

    def test_nested_overrides_restore_and_are_validated(self):
        original = get_settings()
        with override_settings(AGENT_MAX_STEPS=1):
            with override_settings(agent_max_steps=2):
                self.assertEqual(get_settings().agent.max_steps, 2)
            self.assertEqual(get_settings().agent.max_steps, 1)
        self.assertIs(get_settings(), original)
        with self.assertRaises(ValueError):
            with override_settings(typo_setting=1):
                pass

    def test_override_cannot_be_shadowed_by_os_aliases(self):
        with patch.dict(os.environ, {'JEV_SCOPE_ENABLED': 'true', 'QWEN_MODEL': 'outside'}):
            with override_settings(jev_scope_enabled=False, model_name='inside'):
                self.assertFalse(get_settings().jev_scope_enabled)
                self.assertEqual(get_settings().model_name, 'inside')

    def test_secrets_are_redacted_and_settings_are_immutable(self):
        with override_settings(OPENAI_API_KEY='hidden-model', JEV_API_KEY='hidden-jev') as settings:
            self.assertNotIn('hidden-model', repr(settings))
            self.assertNotIn('hidden-jev', settings.model_dump_json())
            with self.assertRaises(ValidationError):
                settings.model_name = 'changed'


class AgentInterfaceTests(ConfiguredAsyncTestCase):
    async def test_settings_overrides_are_isolated_between_tasks_and_threads(self):
        async def read(steps):
            with override_settings(agent_max_steps=steps):
                await asyncio.sleep(0)
                return get_settings().agent.max_steps, await asyncio.to_thread(lambda: get_settings().agent.max_steps)
        self.assertEqual(await asyncio.gather(read(1), read(2)), [(1, 1), (2, 2)])
        self.assertEqual(get_settings().agent.max_steps, 4)

    def test_base_agent_requires_run_implementation(self):
        with self.assertRaises(TypeError):
            Agent('abstract', AsyncMock())

    async def test_same_agent_handles_concurrent_sessions_without_context_leaks(self):
        async def model(messages, tools, **kwargs):
            await asyncio.sleep(0)
            return Message(role='assistant', content=messages[-1]['content'])
        runner = LangGraphTeamAgent('shared', model, Mock())
        first = RunContext(session_id='first', memories={'key': 'one'})
        second = RunContext(session_id='second', memories={'key': 'two'})
        results = await asyncio.gather(runner.run('hello-one', context=first), runner.run('hello-two', context=second))
        self.assertEqual([item.reply for item in results], ['hello-one', 'hello-two'])
        self.assertEqual([item.memories for item in results], [{'key': 'one'}, {'key': 'two'}])
        self.assertEqual(first.history, [])
        self.assertEqual(first.memories, {'key': 'one'})
        self.assertFalse(hasattr(runner, '_history'))
        self.assertNotEqual(first.run_id, second.run_id)

    async def test_timeout_includes_scope_classification_and_rolls_back(self):
        async def slow(*args, **kwargs):
            await asyncio.sleep(1)
        runner = LangGraphTeamAgent('slow-gate', AsyncMock(), Mock(), config=AgentConfig(timeout_seconds=.01))
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.db')
            sid = store.create()
            with patch('app.agents.team_agent.customer_gate', slow), self.assertRaises(ModelServiceError) as error:
                await run_agent('hello', sid, store, agent=runner)
            self.assertEqual(error.exception.status_code, 504)
            self.assertEqual(store.get(sid)['messages'], [])
            with store.edit(sid):
                pass
        runner.llm.assert_not_awaited()

    async def test_custom_agent_commits_typed_result_and_trims_complete_turns(self):
        class EchoAgent(Agent):
            async def run(self, input_text, *, context):
                self.seen = context
                return AgentResult(reply=input_text, memories=context.memories)
        runner = EchoAgent('echo', AsyncMock(), config=AgentConfig(max_history_turns=1))
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.db')
            sid = store.create()
            for question in ('first', 'second'):
                result = await run_agent(question, sid, store, agent=runner)
                self.assertEqual(result['reply'], question)
            messages = store.get(sid)['messages']
            self.assertEqual([item['content'] for item in messages], ['second', 'second'])
            self.assertTrue(all(item['id'] and item['timestamp'] for item in messages))
            self.assertEqual(runner.seen.history[0].content, 'first')
        runner.llm.assert_not_awaited()

    async def test_native_message_contract_and_generation_settings_reach_sdk(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload['model'], 'injected-model')
            self.assertEqual(payload['temperature'], .1)
            self.assertEqual(payload['max_tokens'], 64)
            self.assertNotIn('metadata', payload['messages'][0])
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion', 'created': 0,
                'model': 'injected-model', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'ok'}}]})
        async with AsyncOpenAI(api_key='test', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch('app.services.qwen.client', client), override_settings(
                    model_name='injected-model', model_temperature=.1, model_max_tokens=64):
                answer = await complete([Message(role='user', content='hi', metadata={'internal': 'secret'})], [])
        self.assertIsInstance(answer, Message)
        self.assertEqual(answer.content, 'ok')

    async def test_configurable_call_batch_has_sufficient_graph_recursion_budget(self):
        replies = [Message(role='assistant', tool_calls=[call(f'{round}-{index}') for index in range(16)])
                   for round in range(2)] + [Message(role='assistant', content='done')]
        model = AsyncMock(side_effect=replies)
        runner = LangGraphTeamAgent('batched', model, Mock(return_value={'ok': True, 'result': '2'}),
                                   config=AgentConfig(max_steps=2, max_calls_per_response=16,
                                                      max_tool_calls=40, context_bytes=14000))
        result = await runner.run('hello', context=RunContext(session_id='batch'))
        self.assertEqual(result.reply, 'done')
        self.assertEqual(len(result.trace), 32)
        self.assertEqual(model.await_count, 3)
        self.assertTrue(result.stopped)

    async def test_model_clients_and_breakers_are_scoped_by_endpoint(self):
        with override_settings(model_base_url='http://first.invalid/v1'):
            first = initialize_model_client()
            self.assertEqual(first, initialize_model_client())
        with override_settings(model_base_url='http://second.invalid/v1'):
            second = initialize_model_client()
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        await close_model_client()
        self.assertTrue(first[0].is_closed())
        self.assertTrue(second[0].is_closed())

    def test_small_report_budget_bounds_summary_and_error_without_evidence(self):
        for worker in ({'trace': [], 'reply': '长' * 1200},
                       {'trace': [], 'reply': '长' * 1200, 'error': '错误' * 1000}):
            report = bounded_report(worker, 1000)
            self.assertLessEqual(len(json.dumps(report, ensure_ascii=False).encode()), 1000)
            self.assertTrue(report['evidence_truncated'])
