from tests.support import ConfiguredAsyncTestCase
from app.core.config import get_settings, override_settings
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.agents.decisions import OPTIONS, routing_advice
from app.evaluation.metrics import retrieval_scores, citation_check
from app.ops.resilience import CircuitBreaker, CircuitOpen, Admission, CapacityExceeded
from app.services.memory import MemoryStore
from app.services.qwen import complete, ModelServiceError
from app.services.tools import TOOL_SCHEMAS
from app.main import app
from scripts.backup_data import backup, restore
from scripts.build_dataset import build
from scripts.evaluate_portfolio import evaluate


class OperationTests(ConfiguredAsyncTestCase):
    def test_breaker_half_open_recovery(self):
        now=[10.0];breaker=CircuitBreaker(2,5,lambda:now[0])
        breaker.before();breaker.failure();breaker.before();breaker.failure()
        with self.assertRaises(CircuitOpen):breaker.before()
        now[0]=16;breaker.before()
        with self.assertRaises(CircuitOpen):breaker.before()
        breaker.success();breaker.before()

    def test_admission_recovers_after_failure(self):
        limit=Admission(1)
        with limit.enter():
            with self.assertRaises(CapacityExceeded):
                with limit.enter():pass
        with limit.enter():self.assertEqual(limit.active,1)
        self.assertEqual(limit.active,0)

    def test_saturation_returns_429_without_new_session(self):
        from app.api.routes.chat import _admission
        admission = _admission(get_settings().max_active_chats)
        with patch.object(admission,'capacity',1),admission.enter(),TestClient(app) as client:
            response=client.post('/api/chat',json={'message':'你好'})
        self.assertEqual(response.status_code,429)
        self.assertIn('retry-after',response.headers)

    def test_health_and_metrics(self):
        with TestClient(app) as client:
            self.assertEqual(client.get('/health/live').status_code,200)
            response=client.get('/metrics')
            self.assertIn('commerce_http_requests_total',response.text)
            self.assertIn('x-request-id',response.headers)

    def test_expired_owner_cannot_clear_or_overwrite_new_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            store=MemoryStore(Path(directory)/'memory.sqlite3');sid=store.create()
            with store.edit(sid):
                with store.connection() as db:db.execute('UPDATE sessions SET lease_token=?,busy=? WHERE id=?',('new-owner',9999999999,sid))
                with self.assertRaises(RuntimeError):store.save(sid,[],{})
            with store.connection() as db:self.assertEqual(db.execute('SELECT lease_token FROM sessions').fetchone()[0],'new-owner')

    def test_snapshot_restore_integrity_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';store=MemoryStore(source/'memory.sqlite3');sid=store.create()
            store.save(sid,[],{'language':'zh'})
            backup(source,root/'backup');restore(root/'backup',root/'restored')
            self.assertEqual(MemoryStore(root/'restored/memory.sqlite3').get(sid)['memories'],{'language':'zh'})
            with (root/'backup/memory.sqlite3').open('ab') as file:file.write(b'tampered')
            with self.assertRaises(ValueError):restore(root/'backup',root/'bad')
            self.assertFalse((root/'bad').exists())

    def test_dataset_idempotence_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'dataset'
            first=build(target,products=16,users=4,orders=30)
            self.assertEqual(build(target,products=16,users=4,orders=30),first)
            self.assertEqual(first['counts']['orders'],30)
            self.assertTrue(all(value==0 for value in evaluate(target)['integrity'].values()))
            with self.assertRaises(ValueError):build(target,products=24,users=4,orders=30)

    def test_metrics_known_values_and_citations(self):
        score=retrieval_scores(['x','a','a','b'],['a','b'])
        self.assertEqual(score['mrr'],.5);self.assertEqual(score['recall'],1)
        check=citation_check('来源 [KB-123456abcdef]',[])
        self.assertEqual(check['invalid'],['KB-123456abcdef'])
        self.assertIsNone(citation_check('没有引用',[])['citation_id_precision'])

    async def test_model_transient_retry_then_recovery(self):
        attempts=[]
        def handler(request):
            attempts.append(1)
            if len(attempts)==1:return httpx.Response(503,json={'error':{'message':'unavailable'}})
            return httpx.Response(200,json={'id':'x','object':'chat.completion','created':0,'model':'test',
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':'恢复'}}]})
        async with AsyncOpenAI(api_key='test',max_retries=0,http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch('app.services.qwen.client',client),patch('app.services.qwen.breaker',CircuitBreaker()),override_settings(MODEL_RETRIES=1):
                answer=await complete([{'role':'user','content':'你好'}],TOOL_SCHEMAS)
        self.assertEqual(answer.content,'恢复');self.assertEqual(len(attempts),2)

    async def test_jev_real_contract_with_mock_http_and_fallback(self):
        requests=[]
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200,json={'model':'jev-test','answers':{'route':{'type':'choice','choice':'orders',
                'confidence':.95,'probabilities':{key:1.0 if key=='orders' else 0.0 for key in OPTIONS}}}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with override_settings({'JEV_MODE':'shadow','JEV_API_KEY':'test'}):
                result=await routing_advice('查询 ORD00001，电话13812345678',client)
                self.assertFalse(result['advisory_accepted']);self.assertEqual(result['choice'],'orders')
                self.assertNotIn('13812345678',json.dumps(requests[0]))
                self.assertEqual(requests[0]['questions']['route']['type'],'choice')
            with override_settings({'JEV_MODE':'off'}):
                self.assertEqual((await routing_advice('hello',client))['status'],'disabled')
                self.assertEqual(len(requests),1)
