"""Validated Kev routing can bypass the supervisor; specialists still use real LLM tools."""
import time

from app.agents.graph import build_worker, product_context
from app.agents.roles import ROLES
from app.agents.request import task_text
from app.agents.business_reply import compose_business_reply
from app.evaluation.metrics import citation_check
from app.core.agent import AgentResult


async def run_planned(question, request, advice, model, executor, settings, config, memories):
    expected = 'mixed' if len(request['roles']) > 1 else request['roles'][0] if request['roles'] else None
    if not (settings.kev_direct_routing and request['supported'] and advice.get('advisory_accepted')
            and advice.get('choice') == expected):
        return None
    trace, events, stopped = [], [], False
    for role in request['roles']:
        if len(events) >= config.max_handoffs or len(trace) >= config.max_tool_calls:
            stopped = True
            break
        current = dict(request)
        if role == 'product' and request['order_ids']:
            current['product_ids'] = request['product_ids'] or product_context(trace)
            if not current['product_ids']:
                break  # No guessing after missing/unauthorized orders.
        task = task_text(question, current, role)
        worker = build_worker(role, model, executor, max_steps=config.worker_steps,
                              context_bytes=config.context_bytes,
                              max_calls=config.max_calls_per_response, settings=settings)
        started = time.perf_counter()
        result = await worker.ainvoke({'task': task, 'rounds': [], 'pending': [], 'steps': 0,
                                      'remaining': config.max_tool_calls-len(trace), 'trace': [], 'stopped': False})
        trace.extend(result['trace']);stopped |= result.get('stopped', False)
        events.append({'agent': role, 'task': task, 'tool_count': len(result['trace']),
                       'status': 'failed' if result.get('error') else 'completed',
                       'seconds': time.perf_counter()-started, 'source': 'kev_validated_route',
                       'error': result.get('error'), 'report': result.get('reply', '')})
    rendered = compose_business_reply(question, '', trace, target_ids=request['product_ids'])
    if rendered is None:
        return None
    reply, parts = rendered
    check = {**citation_check(reply, trace), 'parts': parts}
    if 'catalog' in parts:
        check['catalog'] = parts['catalog']
    decision = {**advice, 'execution': 'kev_direct', 'request': request}
    return AgentResult(reply=reply, trace=trace, events=events, memories=memories,
                       stopped=stopped, decision=decision, verification=check)
