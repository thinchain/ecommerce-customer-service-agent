from tests.support import ConfiguredAsyncTestCase
from app.core.config import override_settings
import json
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from app.services.qwen import ModelServiceError, complete
from app.services.tools import TOOL_SCHEMAS
from app.ops.resilience import CircuitBreaker


class ModelTests(ConfiguredAsyncTestCase):
    async def test_history_only_request_omits_empty_tools_for_vllm(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertNotIn('tools', body)
            self.assertNotIn('tool_choice', body)
            return httpx.Response(200, json={
                'id': 'completion-history', 'object': 'chat.completion', 'created': 0, 'model': 'qwen',
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': '历史结果'}}],
            })
        async with AsyncOpenAI(api_key='test', max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch('app.services.qwen.client', client), patch('app.services.qwen.breaker', CircuitBreaker()):
                result = await complete([{'role': 'user', 'content': '回顾上一问'}], [], final=True)
        self.assertEqual(result.content, '历史结果')

    async def test_timeout_does_not_duplicate_shared_gpu_work(self):
        attempts = []
        def handler(request):
            attempts.append(request)
            raise httpx.ReadTimeout('queued', request=request)
        async with AsyncOpenAI(api_key='test', max_retries=0, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )) as client:
            with patch('app.services.qwen.client', client), override_settings(MODEL_RETRIES=2), \
                    patch('app.services.qwen.breaker', CircuitBreaker()):
                with self.assertRaises(ModelServiceError) as error:
                    await complete([{'role': 'user', 'content': '你好'}], TOOL_SCHEMAS)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(error.exception.status_code, 504)

    async def test_retry_can_require_search_and_final_disables_tools(self):
        choices = []

        def handler(request):
            choices.append(json.loads(request.content)['tool_choice'])
            return httpx.Response(200, json={
                'id': 'completion-1', 'object': 'chat.completion', 'created': 0, 'model': 'qwen',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': '完成'}}],
            })

        async with AsyncOpenAI(api_key='test', http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch('app.services.qwen.client', client):
                await complete([{'role': 'user', 'content': '再试一次'}], TOOL_SCHEMAS, required_tool='web_search')
                await complete([{'role': 'user', 'content': '再试一次'}], TOOL_SCHEMAS, final=True, required_tool='web_search')
        self.assertEqual(choices, [{'type': 'function', 'function': {'name': 'web_search'}}, 'none'])

    async def test_sdk_wire_format_and_tool_response(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body['tool_choice'], 'auto')
            self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
            self.assertIn('calculator', {tool['function']['name'] for tool in body['tools']})
            return httpx.Response(200, json={
                'id': 'completion-1', 'object': 'chat.completion', 'created': 0, 'model': 'qwen',
                'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'call-1', 'type': 'function',
                    'function': {'name': 'calculator', 'arguments': '{"expression":"1+1"}'}}]}}],
            })
        async with AsyncOpenAI(api_key='test', http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch('app.services.qwen.client', client):
                result = await complete([{'role': 'user', 'content': '1+1'}], TOOL_SCHEMAS)
        self.assertEqual(result.tool_calls[0].function.name, 'calculator')

    async def test_http_failure_has_actionable_error(self):
        async with AsyncOpenAI(api_key='test', max_retries=0, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(400, json={'error': {'message': 'bad parser'}}))
        )) as client:
            with patch('app.services.qwen.client', client):
                with self.assertRaises(ModelServiceError) as error:
                    await complete([{'role': 'user', 'content': '你好'}], TOOL_SCHEMAS)
        self.assertEqual(error.exception.status_code, 502)
        self.assertIn('工具解析参数', error.exception.detail)
