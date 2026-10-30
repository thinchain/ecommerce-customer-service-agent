"""Regression checks for evidence scope and routing; live scores use evaluate_workflow.py."""
from unittest.mock import AsyncMock, patch

from app.agents.request import resolve_request, historical_request
from app.agents.policy import customer_gate
from app.core.config import override_settings
from tests.support import ConfiguredTestCase, ConfiguredAsyncTestCase
from tests.test_business_reply import catalog, knowledge, observed
from app.agents.business_reply import compose_business_reply


class RequestStateTests(ConfiguredTestCase):
    def test_chinese_adjacent_order_id_is_scoped_and_redacted(self):
        from app.agents.decisions import redact
        request=resolve_request('我现在是U00002，请查ORD00011的订单详情',[])
        self.assertEqual(request['order_ids'],['ORD00011'])
        self.assertNotIn('ORD00011',redact('请查ORD00011的订单'))
        order=observed('list_orders',{}, {'ok':True,'result':[{'order_id':'ORD00001','status':'in_transit'}]})
        reply,_=compose_business_reply('请查ORD00011的订单详情','',[order])
        self.assertNotIn('ORD00001',reply)

    def test_final_budget_instruction_stays_in_initial_system_message(self):
        from app.agents.context import fit_context
        base = [{'role':'system','content':'客服'}]
        messages, _, _ = fit_context(base,[],{'role':'user','content':'查询'},[],12000,True)
        self.assertEqual([i for i,m in enumerate(messages) if m['role']=='system'],[0])
        self.assertIn('预算',messages[0]['content'])
        self.assertEqual(base[0]['content'],'客服')

    def test_implicit_device_capability_requires_knowledge(self):
        from app.agents.evidence import product_requirements
        self.assertEqual(product_requirements('它能同时向三台设备输入吗'),['knowledge_search'])

    def test_authorized_order_items_resolve_followup_without_reusing_values(self):
        trace = [observed('order_detail', {'order_id':'ORD00001'}, {'ok':True,'result':{
            'order_id':'ORD00001','items':[{'product_id':'DEMO-RUN'}]}})]
        history = [{'role':'assistant','tool_calls':trace}]
        request = resolve_request('这款耳机现在多少钱', history)
        self.assertEqual(request['product_ids'], ['DEMO-RUN'])
        self.assertEqual(request['fields'], ['price'])
        self.assertNotIn('price_cents', request)
        self.assertFalse(request['historical'])
        self.assertTrue(historical_request('刚才查到的价格再说一次'))
        self.assertFalse(historical_request('刚才的商品现在重新查价格'))

    def test_multiple_products_require_clarification_and_rejected_history_is_ignored(self):
        history = [{'role':'assistant','tool_calls':[catalog('DEMO-A'),catalog('DEMO-B')]}]
        self.assertTrue(resolve_request('它现在有货吗',history)['ambiguous'])
        history[0]['decision'] = {'scope':{'status':'blocked'}}
        self.assertEqual(resolve_request('它现在有货吗',history)['product_ids'], [])

    def test_render_only_requested_fields_and_exclude_negative_entity(self):
        reply, _ = compose_business_reply('查 DEMO-A 价格，不要把 DEMO-B 的价格当成它的价格', '',
                                          [catalog('DEMO-A'),catalog('DEMO-B')])
        self.assertIn('¥1167.66',reply)
        self.assertNotIn('DEMO-B',reply)
        self.assertNotIn('库存',reply)
        self.assertNotIn('演示商店',reply)

    def test_different_knowledge_kind_does_not_erase_valid_product_evidence(self):
        failed = observed('knowledge_search', {'query':'功能','product_id':'DEMO-A','kind':'product_qa'},
                          {'ok':True,'result':{'hits':[]}})
        reply, parts = compose_business_reply('DEMO-A 支持 Windows 吗','',[knowledge(),failed])
        self.assertIn('支持 Windows',reply)
        self.assertEqual(len(parts['knowledge']['quotes']),1)
        self.assertNotIn('资料不足',reply)


class ScopeConfidenceTests(ConfiguredAsyncTestCase):
    async def test_memory_edits_cannot_be_redirected_to_history_by_classifier(self):
        with override_settings(JEV_SCOPE_ENABLED=True), patch('app.agents.policy.scope_advice', AsyncMock(return_value={
                'status':'evaluated','choice':'conversation_recall','confidence':.9})):
            result = await customer_gate('请记住我的耳机预算是500元',[])
        self.assertEqual(result['status'],'allowed')
        self.assertNotIn('action',result)
        self.assertEqual(result['reason'],'explicit_session_memory')

    async def test_accepted_kev_route_uses_specialist_graph_without_supervisor_calls(self):
        from app.agents.planned import run_planned
        from app.core.config import get_settings
        from app.core.message import Message, ToolCall
        model = AsyncMock(side_effect=[Message(role='assistant', tool_calls=[ToolCall(
            id='one', function={'name':'catalog_search','arguments':'{"product_ids":["DEMO-A"]}'})]),
            Message(role='assistant',content='已查询')])
        executor = lambda *args: catalog()['result']
        request = resolve_request('DEMO-A 多少钱',[])
        with override_settings(STRUCTURED_ROUTING=True,KEV_DIRECT_ROUTING=True) as settings:
            result = await run_planned('DEMO-A 多少钱',request,
                {'choice':'product','advisory_accepted':True},model,executor,settings,settings.agent,{})
        self.assertEqual(result.decision['execution'],'kev_direct')
        self.assertEqual(model.await_count,2)
        self.assertEqual([event.agent for event in result.events],['product'])
        self.assertIn('¥1167.66',result.reply)
        self.assertNotIn('库存',result.reply)

    async def test_low_confidence_scope_rejection_falls_back_without_granting_permissions(self):
        with override_settings(JEV_SCOPE_ENABLED=True), patch('app.agents.policy.scope_advice', AsyncMock(return_value={
                'status':'evaluated','choice':'unrelated','confidence':.22})):
            result = await customer_gate('DEMO-KEY 支持 macOS 吗',[])
        self.assertEqual(result['status'],'allowed')
        self.assertEqual(result['reason'],'low_confidence_fallback')

    async def test_high_confidence_unrelated_still_rejected(self):
        with override_settings(JEV_SCOPE_ENABLED=True), patch('app.agents.policy.scope_advice', AsyncMock(return_value={
                'status':'evaluated','choice':'unrelated','confidence':.95})):
            result = await customer_gate('请写一首诗',[])
        self.assertEqual(result['status'],'blocked')
