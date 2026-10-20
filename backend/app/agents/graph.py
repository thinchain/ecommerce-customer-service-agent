"""LangGraph supervisor with three separately scoped, bounded ReAct subgraphs."""
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError
from app.core.config import Settings, get_settings
from app.core.exceptions import ModelServiceError

from app.agents.context import assistant_calls, fit_context, serialize_calls, size, tool_message
from app.agents.roles import Delegation, HANDOFFS, ROLES, SUPERVISOR_SCHEMAS, UTILITY_TOOLS, role_prompt, role_schemas
from app.services.history import execution_state
from app.services.tool_runtime import ToolRuntime
from app.agents.decisions import routing_advice
from app.evaluation.metrics import citation_check
from app.agents.evidence import missing_product_evidence
from app.agents.catalog_evidence import catalog_observation
from app.agents.business_reply import capture_knowledge, compose_business_reply


class WorkerState(TypedDict, total=False):
    task: str
    rounds: list[list[dict[str, Any]]]
    pending: list[dict[str, Any]]
    steps: int
    remaining: int
    trace: list[dict[str, Any]]
    reply: str
    error: str
    stopped: bool


class TeamState(TypedDict, total=False):
    product_targets: list[str]
    decision: dict[str, Any]
    verification: dict[str, Any]
    base: list[dict[str, Any]]
    history: list[dict[str, Any]]
    current: dict[str, Any]
    rounds: list[list[dict[str, Any]]]
    pending: list[dict[str, Any]]
    pending_index: int
    group: list[dict[str, Any]]
    steps: int
    handoffs: int
    trace: list[dict[str, Any]]
    events: list[dict[str, Any]]
    memories: dict[str, str]
    reply: str
    stopped: bool


def trace_item(call, result, agent, step, *, catalog_evidence_enabled=False):
    item = {'step': step, 'agent': agent, 'name': call['function']['name'],
            'arguments': call['function']['arguments'][:2000], 'result': result}
    if catalog_evidence_enabled and item['name'] == 'catalog_search':
        item['catalog_evidence'] = catalog_observation(call['function']['arguments'], result)
        item['catalog_evidence']['tool_call_id'] = call['id']
    if catalog_evidence_enabled and item['name'] == 'knowledge_search':
        item['knowledge_evidence'] = capture_knowledge(call['function']['arguments'], result)
    return item


def build_worker(role, model, executor, *, max_steps, context_bytes, max_calls=4, settings=None):
    settings = settings or get_settings()
    async def think(state: WorkerState):
        final = state['steps'] >= max_steps or len(state['trace']) >= state['remaining']
        try:
            messages, _, rounds = fit_context(
                [{'role': 'system', 'content': role_prompt(role)}], [],
                {'role': 'user', 'content': state['task']}, state['rounds'], context_bytes, final)
            missing = missing_product_evidence(state['task'], state['trace']) if role == 'product' else []
            # Do not repeat a failed query automatically; preserve the failure report.
            attempted = {item['name'] for item in state['trace']}
            required = next((name for name in missing if name not in attempted), None)
            kwargs = {'required_tool': required} if required and not final else {}
            answer = await model(messages, role_schemas(role), final=final, **kwargs)
            calls = serialize_calls(answer, max_calls)
            if final and calls:
                raise ModelServiceError(502, '专业 Agent 在预算用尽后仍请求工具，未执行。')
            if not calls and not (answer.content or '').strip():
                raise ModelServiceError(502, '专业 Agent 返回空报告。')
            return {'rounds': rounds, 'pending': calls, 'steps': state['steps'] + 1,
                    'reply': (answer.content or '').strip(), 'stopped': final}
        except ModelServiceError as exc:
            # Preserve already collected evidence when only a specialist's model fails.
            return {'pending': [], 'error': exc.detail, 'reply': '', 'stopped': final}

    async def tools(state: WorkerState):
        trace = list(state['trace'])
        group = [assistant_calls(state['pending'])]
        stopped = state.get('stopped', False)
        # MCP cancel scopes must be opened/closed in this same LangGraph node task.
        async with ToolRuntime(settings=settings) as runtime:
            for call in state['pending']:
                name, raw = call['function']['name'], call['function']['arguments']
                if len(trace) >= state['remaining']:
                    result = {'ok': False, 'error': '本轮全局工具预算已用尽', 'code': 'budget_exhausted'}
                    stopped = True
                elif name not in ROLES[role]['tools']:
                    result = {'ok': False, 'error': '该 Agent 无权调用此工具', 'code': 'tool_not_allowed'}
                    trace.append(trace_item(call, result, role, state['steps'],
                                            catalog_evidence_enabled=settings.catalog_evidence_enabled))
                else:
                    result = await runtime.execute(name, raw, {}, executor)
                    trace.append(trace_item(call, result, role, state['steps'],
                                            catalog_evidence_enabled=settings.catalog_evidence_enabled))
                group.append(tool_message(call, result))
        return {'rounds': [*state['rounds'], group], 'trace': trace, 'pending': [], 'stopped': stopped}

    graph = StateGraph(WorkerState)
    graph.add_node('model', think)
    graph.add_node('tools', tools)
    graph.add_edge(START, 'model')
    graph.add_conditional_edges('model', lambda state: 'tools' if state.get('pending') else END, ['tools', END])
    graph.add_edge('tools', 'model')
    return graph.compile(name=f'{role}_agent')


def bounded_report(result, budget=6500):
    """Handoff contains actual observations, not just an unverified worker narrative."""
    trace = result['trace']
    report = {'summary': result.get('reply', '')[:1200], 'execution_state': execution_state(trace),
              'error': result.get('error', ''), 'stopped_by_limit': result.get('stopped', False),
              'evidence': [{'tool': item['name'], 'arguments': item['arguments'],
                            'observation': json.loads(json.dumps(item['result']))} for item in trace]}
    if size(report) > budget:
        report['summary'] = report['summary'][:400]
        report['evidence_truncated'] = True
    while size(report) > budget and report['evidence']:
        report['evidence_truncated'] = True
        if len(report['evidence']) > 1:
            report['evidence'].pop(0)
            continue
        latest = report['evidence'][0]
        if 'arguments' in latest:
            del latest['arguments']
            continue
        payload = latest['observation'].get('result')
        records = payload if isinstance(payload, list) else payload.get('hits') if isinstance(payload, dict) else None
        if isinstance(records, list) and len(records) > 1:
            records.pop()
        elif report['summary']:
            report['summary'] = ''
        else:
            report['evidence'] = []
            report['error'] = '单条证据超过交接预算，请缩小查询范围后重新核验。'
    # Configurable budgets can be smaller than a summary even without evidence.
    for field in ('summary', 'error'):
        if size(report) <= budget:
            break
        report['evidence_truncated'] = True
        value = report[field]
        report[field] = ''
        remaining = max(0, budget - size(report))
        report[field] = value.encode('utf-8')[:remaining].decode('utf-8', errors='ignore')
        while size(report) > budget and report[field]:
            report[field] = report[field][:-1]
    return report


def product_context(trace):
    """Pass only verified product IDs; never copy order/customer records to other workers."""
    ids = []
    for item in trace:
        if item['name'] == 'order_detail' and item['result'].get('ok'):
            for product in item['result'].get('result', {}).get('items', []):
                if product.get('product_id') and product['product_id'] not in ids:
                    ids.append(product['product_id'])
    return ids[-20:]


def build_team_graph(model, executor, *, max_steps, worker_steps, max_handoffs, max_tool_calls, context_bytes,
                     report_bytes=6500, max_calls=4, settings: Settings | None = None):
    settings = settings or get_settings()
    workers = {role: build_worker(role, model, executor, max_steps=worker_steps, context_bytes=context_bytes,
                                  max_calls=max_calls, settings=settings)
               for role in ROLES}

    async def decide(state: TeamState):
        advice = await routing_advice(state['current']['content'], settings=settings)
        update = {'decision': advice}
        if advice.get('advisory_accepted'):
            base = [dict(item) for item in state['base']]
            base[0]['content'] += '\n可选决策模型建议领域：' + advice['choice'] + '。仅供参考，不覆盖权限、用户意图或证据要求。'
            update['base'] = base
        return update

    def verify(state: TeamState):
        check = citation_check(state['reply'], state['trace'])
        check['missing_required_tools'] = missing_product_evidence(state['current']['content'], state['trace'])
        update = {'verification': check}
        if settings.catalog_evidence_enabled:
            rendered = compose_business_reply(state['current']['content'], state['reply'], state['trace'],
                                              target_ids=state.get('product_targets'))
            if rendered is not None:
                reply, parts = rendered
                # Preserve draft diagnostics separately. Final citation checks must
                # describe what was published, including a safe partial answer.
                final_check = citation_check(reply, state['trace'])
                final_check.update(missing_required_tools=check['missing_required_tools'],
                                   draft_checks=check, parts=parts)
                if 'catalog' in parts:
                    final_check['catalog'] = parts['catalog']
                update.update(reply=reply, verification=final_check)
                return update
        if check['invalid']:
            update['reply'] = '本轮回答包含无法核验的资料引用，暂不能提供可靠结论。请缩小问题范围后重试。'
        elif check['missing_required_tools']:
            update['reply'] = '本轮未取得完整的商品查询证据，暂不能确认所请求的商品信息、价格或库存。请重试；不能将模型生成的数值或功能描述作为查询结果。'
        return update

    async def supervisor(state: TeamState):
        final = state['steps'] >= max_steps or len(state['trace']) >= max_tool_calls
        messages, history, rounds = fit_context(state['base'], state['history'], state['current'],
                                                state['rounds'], context_bytes, final)
        required = missing_product_evidence(state['current']['content'], state['trace'])
        kwargs = {'required_tool': 'delegate_product'} if required and state['steps'] == 0 and not final else {}
        answer = await model(messages, SUPERVISOR_SCHEMAS, final=final, **kwargs)
        calls = serialize_calls(answer, max_calls)
        if final and calls:
            raise ModelServiceError(502, '协调 Agent 在预算用尽后仍请求工具，未执行。')
        if not calls and not (answer.content or '').strip():
            raise ModelServiceError(502, '本地模型返回了空回复，请重试。')
        update = {'history': history, 'rounds': rounds, 'steps': state['steps'] + 1,
                  'pending': calls, 'pending_index': 0, 'group': [assistant_calls(calls)] if calls else [],
                  'stopped': state['stopped'] or final, 'reply': (answer.content or '').strip()}
        if not calls:
            update['events'] = [*state['events'], {'agent': 'supervisor', 'task': '汇总并回复',
                                                   'status': 'completed', 'tool_count': 0}]
        return update

    def route(state: TeamState):
        if not state['pending']:
            return END
        name = state['pending'][state['pending_index']]['function']['name']
        return HANDOFFS.get(name, 'utility')

    def finish_call(state, result, **updates):
        call = state['pending'][state['pending_index']]
        group = [*state['group'], tool_message(call, result)]
        index = state['pending_index'] + 1
        if index == len(state['pending']):
            updates.update({'rounds': [*state['rounds'], group], 'pending': [], 'pending_index': 0, 'group': []})
        else:
            updates.update({'group': group, 'pending_index': index})
        return updates

    async def utility(state: TeamState):
        call = state['pending'][state['pending_index']]
        name, raw = call['function']['name'], call['function']['arguments']
        stopped = state['stopped']
        if len(state['trace']) >= max_tool_calls:
            result = {'ok': False, 'error': '全局工具预算已用尽', 'code': 'budget_exhausted'}
            stopped = True
        elif name not in UTILITY_TOOLS:
            result = {'ok': False, 'error': '协调 Agent 无权直接调用该工具，请使用委派接口', 'code': 'tool_not_allowed'}
        else:
            async with ToolRuntime(settings=settings) as runtime:
                result = await runtime.execute(name, raw, state['memories'], executor)
        trace = state['trace']
        if len(trace) < max_tool_calls:
            trace = [*trace, trace_item(call, result, 'supervisor', state['steps'])]
        return finish_call(state, result, trace=trace, stopped=stopped, memories=state['memories'])

    def specialist_node(role):
        async def run(state: TeamState):
            call = state['pending'][state['pending_index']]
            raw = call['function']['arguments']
            task = ''
            handoffs = state['handoffs']
            trace = state['trace']
            stopped = state['stopped']
            event = {'agent': role, 'task': '', 'status': 'failed', 'tool_count': 0}
            if handoffs >= max_handoffs or len(trace) >= max_tool_calls:
                result = {'ok': False, 'error': '本轮委派或工具预算已用尽', 'code': 'budget_exhausted'}
                stopped = True
                event['status'] = 'limit'
            else:
                handoffs += 1
                try:
                    if len(raw) > 4000:
                        raise ValueError('委派参数过长')
                    task = Delegation.model_validate_json(raw).task
                except (ValidationError, ValueError):
                    result = {'ok': False, 'error': '委派参数不合法，需提供明确的 task', 'code': 'invalid_handoff'}
                else:
                    event['task'] = task[:800]
                    if role == 'product':
                        ids = product_context(trace)
                        if ids:
                            task += '\n本轮订单工具实际返回的商品 ID（仅为候选，按任务选择）：' + json.dumps(ids)
                    worker = await workers[role].ainvoke(
                        {'task': task, 'rounds': [], 'pending': [], 'steps': 0,
                         'remaining': max_tool_calls-len(trace), 'trace': [], 'stopped': False},
                        config={'recursion_limit': 2*worker_steps + 5})
                    report = bounded_report(worker, report_bytes)
                    # Worker errors are observations for the supervisor, not user-facing HTTP failures.
                    result = {'ok': not bool(worker.get('error')), 'result': report}
                    if worker.get('error'):
                        result['error'] = worker['error']
                    trace = [*trace, *worker['trace']]
                    stopped = stopped or worker.get('stopped', False)
                    event.update({'status': 'failed' if worker.get('error') or report['execution_state'] == 'failed'
                                  else 'partial' if report['execution_state'] == 'mixed' or worker.get('stopped') else 'completed',
                                  'tool_count': len(worker['trace']), 'execution_state': report['execution_state']})
            return finish_call(state, result, trace=trace, handoffs=handoffs, stopped=stopped,
                               events=[*state['events'], event])
        return run

    graph = StateGraph(TeamState)
    graph.add_node('decision', decide)
    graph.add_node('verify', verify)
    graph.add_node('supervisor', supervisor)
    graph.add_node('utility', utility)
    for role in ROLES:
        graph.add_node(role, specialist_node(role))
    graph.add_edge(START, 'decision')
    graph.add_edge('decision', 'supervisor')
    graph.add_conditional_edges('supervisor', lambda state: route(state) if state['pending'] else 'verify', [*ROLES, 'utility', 'verify'])
    graph.add_edge('verify', END)
    # A batched tool call is fully answered before the next supervisor model call.
    def after_dispatch(state):
        return route(state) if state['pending'] else 'supervisor'
    for name in [*ROLES, 'utility']:
        graph.add_conditional_edges(name, after_dispatch, [*ROLES, 'utility', 'supervisor'])
    return graph.compile(name='commerce_multi_agent')
