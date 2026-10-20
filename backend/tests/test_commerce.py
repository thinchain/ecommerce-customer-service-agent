from tests.support import ConfiguredAsyncTestCase, ConfiguredTestCase
from app.core.config import override_settings
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessage

from app.commerce.store import CommerceStore
from app.knowledge.store import KnowledgeStore, Document
from app.services.tools import execute_tool
from app.services.agent import run_agent
from app.services.memory import MemoryStore
from app.main import app
from app.mcp.client import connect, invoke
from scripts.generate_mock_data import generate
from scripts.ingest_knowledge import ingest_file
from scripts.datasets.import_public import convert, normalize

ROOT = Path(__file__).resolve().parents[2]


class CommerceTests(ConfiguredAsyncTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.db = self.directory / 'commerce.db'
        generate(self.db)
        self.store = CommerceStore(self.db)
        self.kb = KnowledgeStore(self.directory / 'kb.db', backend='fts')
        ingest_file(ROOT / 'fixtures/knowledge/demo.jsonl', self.kb)
        self.patches = [patch('app.commerce.tools.get_commerce', return_value=self.store),
                        patch('app.commerce.tools.get_knowledge', return_value=self.kb)]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in self.patches:
            patcher.stop()
        self.temp.cleanup()

    def test_seed_is_idempotent_and_relationally_consistent(self):
        generate(self.db)
        with self.store.connection() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM orders').fetchone()[0], 40)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
            inconsistent = db.execute('''SELECT o.order_id FROM orders o JOIN order_items i USING(order_id)
                GROUP BY o.order_id HAVING o.total_cents!=sum(i.quantity*i.unit_price_cents)''').fetchall()
            self.assertEqual(inconsistent, [])
            self.assertEqual(db.execute('''SELECT r.refund_id FROM refunds r JOIN payments p USING(order_id)
                WHERE r.amount_cents>p.amount_cents OR p.status!='paid' ''').fetchall(), [])

    def test_order_ownership_and_injection(self):
        with self.store.connection() as db:
            db.execute("UPDATE orders SET user_id='U00002' WHERE order_id='ORD00001'")
        for order_id in ('ORD00001', "' OR 1=1 --"):
            result = execute_tool('order_detail', json.dumps({'order_id': order_id}), {})
            self.assertFalse(result['ok'])
        self.assertFalse(execute_tool('order_detail', '{"order_id":"ORD00001","user_id":"U00002"}', {})['ok'])
        self.assertEqual(self.store.products(query="%' OR 1=1 --"), [])

    def test_rag_filters_provenance_and_structured_join(self):
        hits = self.kb.search('跑步运动耳机', product_id='DEMO-RUN')['hits']
        self.assertTrue(hits)
        self.assertTrue(all(hit['product_id'] == 'DEMO-RUN' for hit in hits))
        self.assertTrue(hits[0]['synthetic'])
        self.assertTrue(hits[0]['citation'].startswith('KB-'))
        products = self.store.products(product_ids=[hit['product_id'] for hit in hits], max_price_cents=50000, in_stock=True)
        self.assertEqual(products[0]['product_id'], 'DEMO-RUN')
        self.assertEqual(self.kb.search('退款', kind='policy')['hits'][0]['kind'], 'policy')
        self.assertEqual(self.kb.search('未知不存在xyz')['hits'], [])

    def test_reingest_replaces_chunks_without_duplicates(self):
        doc = Document(document_id='long', title='测试', text='测试文本'*400, source='test', source_url='demo://test', license='test')
        first = self.kb.ingest(doc)
        self.assertGreater(first, 1)
        self.kb.ingest(doc.model_copy(update={'text': '新的短文本'}))
        with self.kb.connection() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM chunks WHERE document_id='long'").fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(*) FROM chunks').fetchone()[0], db.execute('SELECT count(*) FROM chunks_fts').fetchone()[0])

    def test_exact_product_ids_override_display_text_but_keep_price_filter(self):
        self.assertEqual(self.store.products(query='错误的完整名称', product_ids=['DEMO-RUN']),
                         self.store.products(product_ids=['DEMO-RUN']))
        self.assertEqual(self.store.products(query='错误名称', product_ids=['DEMO-RUN'], max_price_cents=0), [])
        self.assertEqual(self.store.products(query='耳机', product_ids=[]), [])

    def test_known_product_lexical_miss_falls_back_only_within_filters(self):
        hits = self.kb.search('zzzznonmatchingword', product_id='DEMO-RUN', kind='product')['hits']
        self.assertTrue(hits)
        self.assertTrue(all(hit['product_id'] == 'DEMO-RUN' and hit['kind'] == 'product' for hit in hits))
        self.assertTrue(all(hit['rank'] is None and hit['citation'].startswith('KB-') for hit in hits))
        self.assertEqual(self.kb.search('zzzznonmatchingword', product_id='UNKNOWN')['hits'], [])
        self.assertEqual(self.kb.search('zzzznonmatchingword')['hits'], [])

    def test_invalid_public_row_rolls_back_catalog_and_partial_output(self):
        source, output = self.directory / 'bad.jsonl', self.directory / 'partial.jsonl'
        source.write_text(json.dumps({'product_id': 'NEW', 'product_title': 'test'}) + '\n{}\n')
        with self.assertRaises(ValueError):
            convert(source, output, 'esci', self.store)
        self.assertFalse(output.exists())
        self.assertEqual(self.store.products(product_ids=['amazon:us:NEW']), [])

    def test_oversized_rag_observations_are_bounded(self):
        with patch.object(self.kb, 'search', return_value={'hits': [{'text': '长'*2000}, {'text': '长'*2000}]}):
            result = execute_tool('knowledge_search', '{"query":"商品"}', {})
        self.assertTrue(result['ok'])
        self.assertTrue(result['truncated'])
        self.assertEqual(len(result['result']['hits']), 1)

    def test_public_import_aligns_ids_and_preserves_operational_values(self):
        row = {'product_id': 'B-1', 'product_locale': 'us', 'product_title': 'Running headset', 'product_description': 'Bluetooth'}
        source = self.directory / 'source.jsonl'
        source.write_text(json.dumps(row)+'\n')
        output = self.directory / 'knowledge.jsonl'
        convert(source, output, 'esci', self.store)
        ingest_file(output, self.kb)
        pqa, _ = normalize({'asin_id': 'B-1', 'item_name': 'Running headset', 'question_text': 'Bluetooth?', 'answer_text': 'Yes'}, 'pqa')
        self.assertEqual(pqa.product_id, 'amazon:us:B-1')
        self.assertFalse(pqa.synthetic)
        self.kb.ingest(pqa)
        self.assertTrue(self.kb.search('Bluetooth', product_id=pqa.product_id)['hits'])
        with self.store.connection() as db:
            db.execute('UPDATE products SET stock=99 WHERE product_id=?', (pqa.product_id,))
        convert(source, self.directory / 'again.jsonl', 'esci', self.store)
        self.assertEqual(self.store.products(product_ids=[pqa.product_id])[0]['stock'], 99)

    async def test_agent_combines_sql_and_rag(self):
        def call(name, args):
            return ChatCompletionMessage(role='assistant', content=None, tool_calls=[{
                'id': name, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}])
        model = AsyncMock(side_effect=[call('delegate_orders', {'task': '查询 ORD00001 的物流和商品 ID'}),
            call('order_detail', {'order_id': 'ORD00001'}),
            ChatCompletionMessage(role='assistant', content='商品是 DEMO-RUN，订单运输中。'),
            call('delegate_product', {'task': '查询已购商品是否支持 Windows'}),
            call('knowledge_search', {'query': 'Windows', 'product_id': 'DEMO-RUN'}),
            ChatCompletionMessage(role='assistant', content='演示耳机支持 Windows，见检索证据。'),
            ChatCompletionMessage(role='assistant', content='订单运输中，商品支持 Windows。')])
        memory = MemoryStore(self.directory / 'memory.db')
        with patch('app.services.agent.complete', model):
            result = await run_agent('我的耳机还没到，支持 Windows 吗？', memory.create(), memory)
        self.assertEqual([item['name'] for item in result['tool_calls']], ['order_detail', 'knowledge_search'])
        self.assertTrue(all(item['result']['ok'] for item in result['tool_calls']))
        self.assertEqual([item['agent'] for item in result['tool_calls']], ['orders', 'product'])
        self.assertIn('DEMO-RUN', model.call_args_list[4].args[0][-1]['content'])

    async def test_real_mcp_stdio_protocol_discovery_and_call(self):
        with override_settings({'COMMERCE_DB_PATH': str(self.db), 'KNOWLEDGE_DB_PATH': str(self.kb.path), 'DEMO_USER_ID': 'U00001'}):
            async with connect() as session:
                listed = await session.list_tools()
                self.assertEqual({tool.name for tool in listed.tools}, {'list_orders', 'order_detail', 'catalog_search', 'knowledge_search'})
                result = await invoke(session, 'order_detail', '{"order_id":"ORD00001"}')
                self.assertTrue(result['ok'])
                self.assertEqual(result['result']['items'][0]['product_id'], 'DEMO-RUN')
                bad = await session.call_tool('order_detail', {'order_id': 'ORD00001', 'user_id': 'U00002'})
                self.assertTrue(bad.isError)

    def test_overview_api(self):
        with patch('app.api.routes.commerce.get_commerce', return_value=self.store), patch('app.api.routes.commerce.get_knowledge', return_value=self.kb), TestClient(app) as client:
            response = client.get('/api/commerce/overview')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['mode'], 'synthetic_demo')
        self.assertEqual(len(response.json()['products']), 4)


class ChromaTests(ConfiguredTestCase):
    def test_real_vector_store_upsert_query_and_metadata_filter(self):
        # Tiny fixed vectors test the storage contract, not real semantic relevance.
        def embed(texts):
            return [[1.0, 0.0, 0.0] if 'Windows' in text else [0.0, 1.0, 0.0] for text in texts]
        with tempfile.TemporaryDirectory() as temp:
            store = KnowledgeStore(Path(temp)/'kb.db', backend='chroma', chroma_path=Path(temp)/'chroma', embed=embed)
            for pid, text in [('A', 'Windows compatible'), ('B', 'Android phone')]:
                store.ingest(Document(document_id=pid, product_id=pid, title=pid, text=text, source='test', source_url='demo://test', license='test'))
            hits = store.search('Windows', product_id='A')['hits']
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]['product_id'], 'A')
            self.assertAlmostEqual(hits[0]['rank'], 0.0, places=5)
