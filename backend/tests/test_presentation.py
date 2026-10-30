from unittest.mock import AsyncMock, patch

from app.agents.business_reply import compose_business_reply
from app.agents.presentation import present_business_result
from app.core.agent import AgentResult
from app.core.exceptions import ModelServiceError
from app.core.message import Message
from tests.support import ConfiguredAsyncTestCase
from tests.test_business_reply import catalog, knowledge


def result_for(question,trace,candidate=None,historical=False):
    reply,parts=compose_business_reply(question,'',trace,historical=historical)
    result=AgentResult(reply=reply,trace=trace,verification={'parts':parts})
    if candidate is not None:result.verification['presentation_candidate']=candidate
    return result


class PresentationTests(ConfiguredAsyncTestCase):
    async def test_malformed_kb_label_cannot_bypass_citation_validation(self):
        q='DEMO-A 的价格是多少'
        result=result_for(q,[catalog()],'DEMO-A 售价1167.66元。[KB-DEMO-A]')
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',
            content='DEMO-A 售价1167.66元。')),12000)
        self.assertNotIn('[KB-DEMO-A]',actual.reply)
        self.assertIn('invalid_citation',actual.verification['presentation']['existing_draft_checks'])

    async def test_history_disclaimer_does_not_license_current_price_claim(self):
        q='刚才DEMO-A的价格是多少'
        text='以下基于历史，未重新查询。DEMO-A 当前价格为1167.66元。'
        result=result_for(q,[catalog()],text,historical=True)
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',
            content='上次查询到 DEMO-A 的价格为1167.66元。')),12000)
        self.assertNotIn('当前价格',actual.reply)
        self.assertIn('historical_value_asserted_current',actual.verification['presentation']['existing_draft_checks'])

    async def test_order_items_answer_does_not_require_unrequested_status(self):
        from tests.test_business_reply import observed
        q='这单里有哪些商品，数量多少'
        trace=[observed('order_detail',{'order_id':'ORD00001'},{'ok':True,'result':{
            'order_id':'ORD00001','status':'in_transit','items':[{'product_id':'DEMO-A','quantity':1}]}})]
        result=result_for(q,trace)
        text='订单 ORD00001 包含商品 DEMO-A，数量为1。'
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content=text)),12000)
        self.assertEqual(actual.reply,text)
        self.assertEqual(actual.verification['presentation']['status'],'generated')

    async def test_real_agent_boundary_uses_common_presenter_after_empty_lookup(self):
        from app.core.config import override_settings, get_settings
        from app.services.agent import run_agent
        from app.services.memory import MemoryStore
        from tests.test_catalog_evidence import invoke, text
        natural='暂未查到 DEMO-A 的兼容性资料，无法确认是否支持 Windows。'
        model=AsyncMock(side_effect=[invoke('delegate_product',{'task':'查询 DEMO-A 的 Windows 兼容性'}),
            invoke('knowledge_search',{'query':'Windows','product_id':'DEMO-A'}),text('资料不足'),
            text('没有找到对应商品。'),text(natural)])
        with override_settings(NATURAL_RESPONSE_ENABLED=True):
            store=MemoryStore(get_settings().memory_db_path);sid=store.create()
            with patch('app.services.agent.complete',model),patch('app.services.agent.execute_tool',
                    return_value={'ok':True,'result':{'hits':[]}}):
                response=await run_agent('DEMO-A 支持 Windows 吗',sid,store)
        self.assertEqual(response['reply'],natural)
        self.assertEqual(model.await_count,5)
        self.assertEqual(response['verification']['presentation']['status'],'generated')
        self.assertEqual(response['agent_events'][-1]['agent'],'responder')

    async def test_empty_knowledge_is_explained_by_model_not_internal_template(self):
        question='DEMO-A 支持 Windows 吗'
        result=result_for(question,[knowledge(result={'ok':True,'result':{'hits':[]}})])
        model=AsyncMock(return_value=Message(role='assistant',content='暂未查到 DEMO-A 的相关兼容性资料，无法确认是否支持 Windows。'))
        actual=await present_business_result(question,result,model,12000)
        self.assertEqual(actual.reply,model.return_value.content)
        self.assertNotIn('是否充分回答',actual.reply)
        self.assertEqual(actual.verification['presentation']['status'],'generated')
        messages,tools=model.call_args.args
        self.assertEqual(tools,[])
        self.assertNotIn('是否充分回答',messages[1]['content'])
        self.assertIn('missing_products',messages[1]['content'])

    async def test_checked_existing_model_reply_is_reused_without_extra_generation(self):
        question='DEMO-A 的价格是多少'
        natural='DEMO-A 的售价为 1167.66 元。'
        result=result_for(question,[catalog()],natural)
        model=AsyncMock()
        actual=await present_business_result(question,result,model,12000)
        self.assertEqual(actual.reply,natural)
        model.assert_not_awaited()
        self.assertEqual(actual.verification['presentation']['status'],'reused_model_reply')

    async def test_invalid_prices_entities_and_unrequested_stock_are_not_published(self):
        for text in ('DEMO-A 的价格是999元。','DEMO-B 的价格是1167.66元。',
                     'DEMO-A 的价格是1167.66元，库存23件。'):
            with self.subTest(text=text):
                q='DEMO-A 的价格是多少';result=result_for(q,[catalog()])
                actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content=text)),12000)
                self.assertEqual(actual.verification['presentation']['status'],'evidence_fallback')
                self.assertNotIn('999',actual.reply)
                self.assertNotIn('DEMO-B',actual.reply)
                self.assertNotIn('库存',actual.reply)

    async def test_two_products_cannot_swap_verified_prices(self):
        from tests.test_catalog_evidence import row
        q='DEMO-A 和 DEMO-B 的价格是多少'
        trace=[catalog(),catalog('DEMO-B',{'ok':True,'result':[row('DEMO-B',price=29900)]})]
        result=result_for(q,trace)
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content=
            'DEMO-A 售价299元，DEMO-B 售价1167.66元。')),12000)
        self.assertEqual(actual.verification['presentation']['status'],'evidence_fallback')
        self.assertIn('missing_or_misbound_price',actual.verification['presentation']['checks'])

    async def test_no_source_does_not_mean_product_does_not_exist(self):
        q='DEMO-A 支持 Windows 吗'
        result=result_for(q,[knowledge(result={'ok':True,'result':{'hits':[]}})])
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content=
            '没有找到对应商品。')),12000)
        self.assertNotEqual(actual.reply,'没有找到对应商品。')
        self.assertIn('knowledge_absence_is_not_product_absence',actual.verification['presentation']['checks'])

    async def test_paraphrase_keeps_citation_and_original_negative_source_for_audit(self):
        q='DEMO-A 支持双设备同时连接吗'
        result=result_for(q,[knowledge()])
        model=AsyncMock(return_value=Message(role='assistant',content='DEMO-A 不支持双设备同时连接。[KB-012345abcdef]'))
        actual=await present_business_result(q,result,model,12000)
        self.assertEqual(actual.reply,model.return_value.content)
        self.assertEqual(actual.verification['parts']['knowledge']['quotes'][0]['text'],
                         '支持 Windows；不支持双设备同时连接。')
        self.assertEqual(actual.verification['presentation']['semantic_entailment'],'not_measured')

    async def test_sources_are_bound_by_server_but_invented_citation_is_rejected(self):
        q='DEMO-A 支持 Windows 吗';result=result_for(q,[knowledge()])
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content='DEMO-A 支持 Windows。')),12000)
        self.assertEqual(actual.verification['presentation']['status'],'generated')
        self.assertIn('[KB-012345abcdef]',actual.reply)
        result=result_for(q,[knowledge()])
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content='DEMO-A 支持 Windows。[KB-ffffffffffff]')),12000)
        self.assertEqual(actual.verification['presentation']['status'],'evidence_fallback')
        self.assertNotIn('KB-ffffffffffff',actual.reply)

    async def test_partial_failure_does_not_erase_successful_price(self):
        q='DEMO-A 的价格是多少，支持 Windows 吗'
        result=result_for(q,[catalog(),knowledge(result={'ok':False,'error':'offline'})])
        text='DEMO-A 售价1167.66元；暂未取得兼容性资料，无法确认是否支持 Windows。'
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content=text)),12000)
        self.assertEqual(actual.reply,text)

    async def test_historical_reply_must_not_become_current_price(self):
        q='刚才DEMO-A的价格是多少';result=result_for(q,[catalog()],historical=True)
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content=
            'DEMO-A 当前售价1167.66元。')),12000)
        self.assertEqual(actual.verification['presentation']['status'],'evidence_fallback')
        self.assertIn('history_presented_as_current',actual.verification['presentation']['checks'])

    async def test_model_failure_keeps_evidence_and_reports_presentation_failure(self):
        q='DEMO-A 支持 Windows 吗';result=result_for(q,[knowledge()])
        actual=await present_business_result(q,result,AsyncMock(side_effect=ModelServiceError(502,'offline')),12000)
        self.assertIn('不支持双设备',actual.reply)
        self.assertNotIn('是否充分回答',actual.reply)
        self.assertEqual(actual.events[-1].status,'failed')
        self.assertEqual(actual.verification['presentation']['error_type'],'ModelServiceError')

    async def test_zero_price_and_stock_are_preserved(self):
        from tests.test_catalog_evidence import row
        q='DEMO-A 的价格和库存'
        result=result_for(q,[catalog(result={'ok':True,'result':[row(price=0,stock=0)]})])
        text='DEMO-A 售价0元，库存0件，目前缺货。'
        actual=await present_business_result(q,result,AsyncMock(return_value=Message(role='assistant',content=text)),12000)
        self.assertEqual(actual.reply,text)
