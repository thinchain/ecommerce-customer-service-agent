from tests.support import ConfiguredAsyncTestCase
from app.core.config import override_settings
import json

import httpx

from app.agents.decisions import OPTIONS, routing_advice


class DecisionTests(ConfiguredAsyncTestCase):
    async def test_local_kev_without_key_preserves_shadow_and_redaction(self):
        def handler(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:8795/v1/systemone')
            self.assertNotIn('authorization', request.headers)
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'kev-4b')
            self.assertNotIn('13812345678', body['state']['question'])
            return httpx.Response(200, json={'model': 'kev-4b', 'answers': {'route': {
                'type': 'choice', 'choice': 'orders', 'confidence': .95,
                'probabilities': {key: 1 if key == 'orders' else 0 for key in OPTIONS}}}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with override_settings({'JEV_MODE': 'shadow', 'JEV_BASE_URL': 'http://127.0.0.1:8795/v1/',
                                        'JEV_API_KEY': '', 'JEV_MODEL': 'kev-4b'}):
                result = await routing_advice('查询ORD00001，电话13812345678', client)
        self.assertEqual(result['status'], 'evaluated')
        self.assertFalse(result['advisory_accepted'])

    async def test_timeout_falls_back_without_retry(self):
        requests = []
        def handler(request):
            requests.append(request)
            raise httpx.ReadTimeout('busy', request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with override_settings({'JEV_MODE': 'advisory', 'JEV_BASE_URL': 'http://127.0.0.1:8795/v1',
                                        'JEV_API_KEY': ''}):
                result = await routing_advice('查询订单', client)
        self.assertEqual(result, {'status': 'unavailable', 'reason': 'provider_error'})
        self.assertEqual(len(requests), 1)

    async def test_hosted_jev_still_requires_key(self):
        with override_settings({'JEV_MODE': 'shadow', 'JEV_BASE_URL': 'https://api.typesafe.ai/v1',
                                    'JEV_API_KEY': ''}):
            self.assertEqual((await routing_advice('你好'))['reason'], 'missing_key')

    def test_invalid_timeout_is_rejected_before_execution(self):
        for value in ('nan', '0', 'invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with override_settings({'JEV_TIMEOUT_SECONDS': value}):
                    self.fail('Invalid timeout must not become active settings')
