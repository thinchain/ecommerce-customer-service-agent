import math
import re


def retrieval_scores(retrieved, relevant, k=4):
    retrieved=list(dict.fromkeys(retrieved))[:k]
    relevant=set(relevant)
    if not relevant:return {'empty_correct':not retrieved}
    hits=[int(item in relevant) for item in retrieved]
    dcg=sum(hit/math.log2(i+2) for i,hit in enumerate(hits))
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(relevant))))
    return {'hit':int(any(hits)), 'recall':sum(hits)/len(relevant),
            'mrr':next((1/(i+1) for i,hit in enumerate(hits) if hit),0), 'ndcg':dcg/ideal}


def citation_check(reply, trace):
    cited=set(re.findall(r'KB-[a-f0-9]{12}',reply))
    available=set()
    for call in trace:
        if call['name']=='knowledge_search' and call['result'].get('ok'):
            result=call['result'].get('result',{})
            for hit in result.get('hits',[]) if isinstance(result,dict) else []:
                if isinstance(hit,dict) and isinstance(hit.get('citation'),str):available.add(hit['citation'])
    # A bracketed tool name is also a source claim, not proof that it ran.
    tool_citations = set(re.findall(r'\[(catalog_search|knowledge_search|list_orders|order_detail)\]', reply))
    used = {call['name'] for call in trace if call['result'].get('ok')}
    return {'cited':sorted(cited),'invalid':sorted((cited-available) | (tool_citations-used)),
            'citation_id_precision':len(cited&available)/len(cited) if cited else None,
            'semantic_factuality':'not_measured'}


def score_case(case, result):
    used={call['name'] for call in result.get('tool_calls',[]) if call['result'].get('ok')}
    expected=set(case.get('required_tools',[]))
    routes={event['agent'] for event in result.get('agent_events',[]) if event['agent']!='supervisor'}
    reply=result['reply']
    checks={'tools_correct':expected<=used and not (used & set(case.get('forbidden_tools',[]))),
            'routing_exact':routes==set(case.get('expected_agents',[])),
            'answer_assertions':all(re.search(pattern,reply) for pattern in case.get('required_patterns',[])) and
                                 not any(re.search(pattern,reply) for pattern in case.get('forbidden_patterns',[])),
            'citations_valid':not citation_check(reply,result.get('tool_calls',[]))['invalid']}
    checks['task_pass']=all(checks.values())
    return checks
