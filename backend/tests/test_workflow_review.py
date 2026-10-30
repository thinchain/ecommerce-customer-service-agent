import unittest

from scripts.review_workflow import judge_context


class WorkflowReviewTests(unittest.TestCase):
    def test_chinese_adjacent_order_id_keeps_reference_and_fixed_identity(self):
        row=self.row('请查ORD00011的订单详情')
        reference={'tables':{'order_items':[{'order_id':'ORD00011','product_id':'DEMO-RUN'}],
            'products':[{'product_id':'DEMO-RUN','price_cents':29900,'stock':18}],
            'orders':[{'order_id':'ORD00011','user_id':'U00002'}],'knowledge_chunks':[]}}
        context=judge_context([row],[row],reference)
        self.assertEqual(context['reference']['user_id'],'U00001')
        self.assertEqual(context['reference']['orders'][0]['user_id'],'U00002')

    def test_judge_uses_own_session_price_updates_and_full_observations(self):
        prior=self.row('查价',turn=0)
        prior.update(reply='x'*1000,tool_calls=[{'name':'catalog_search','result':{'price_cents':29900}}])
        row=self.row('上次价格',turn=1)
        row['fixture_mutation']={'price_cents':30900}
        other=self.row('查价',turn=1,variant='Kev_scope')
        other['fixture_mutation']={'price_cents':99900}
        repeated=self.row('查价',turn=1,repeat=1)
        repeated['fixture_mutation']={'price_cents':88800}
        context=judge_context([row],[prior,other,repeated,row],
            {'tables':{k:[] for k in ('order_items','products','orders','knowledge_chunks')}})
        self.assertEqual(context['price_updates'],[{'price_cents':30900}])
        history=context['records'][0]['prior_turns']
        self.assertEqual(len(history),1)
        self.assertEqual(history[0]['tools'],prior['tool_calls'])
        self.assertEqual(len(history[0]['reply']),1000)

    @staticmethod
    def row(question,turn=0,variant='Qwen_only',repeat=0):
        return {'id':f'{repeat}:test:{variant}:{turn}','case_id':'test','variant':variant,'repeat':repeat,
            'turn':turn,'question':question,'reply':'','http_status':200,'decision':{},
            'tool_calls':[],'agent_events':[],'kev_calls':[],'golden':{}}
