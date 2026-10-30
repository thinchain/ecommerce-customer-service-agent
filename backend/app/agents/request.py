"""Small request state: entity references and requested fields, never cached live values."""
import json
import re

from app.agents.catalog_evidence import product_ids

PRICE = re.compile(r'价格|价钱|多少钱|售价|\bprice\b', re.I)
STOCK = re.compile(r'库存|存货|余量|剩余数量|多少件|有货|缺货|\bstock\b', re.I)
KNOWLEDGE = re.compile(r'兼容|支持|功能|适合|连接|切换|续航|防水|游泳|怎么样|介绍|同时.*设备|设备.*输入|\bWindows\b|\bmacOS\b', re.I)
ORDER = re.compile(r'订单|这单|那单|物流|退款|支付|(?<![A-Za-z0-9_])ORD\d+(?![A-Za-z0-9_])', re.I)
ORDER_IDS = re.compile(r'(?<![A-Za-z0-9_])ORD\d+(?![A-Za-z0-9_])', re.I)
UNRELATED = re.compile(r'爬虫|写.{0,8}(?:代码|程序|小说|诗)|编写.{0,8}(?:代码|程序|脚本)|\b(?:python|javascript)\b', re.I)


def requested_fields(question):
    return [name for name, pattern in [('price', PRICE), ('stock', STOCK), ('knowledge', KNOWLEDGE)]
            if pattern.search(question)]


def explicit_products(question):
    # IDs mentioned only as a prohibited substitution are not query targets.
    excluded = set()
    for clause in re.split(r'[，,；;。]', question):
        if re.search(r'不要把|别把|不查|排除', clause):
            ids = product_ids(clause)
            if '当成' in clause:
                ids = product_ids(clause.split('当成')[0])
            excluded.update(ids)
    return [pid for pid in product_ids(question) if pid not in excluded]


def historical_request(question):
    return bool(re.search(r'刚才|刚刚|上次|前面|之前|问过|历史|回顾', question)
                and not re.search(r'现在|当前|重新|最新|实时|再查', question)
                and not session_memory_request(question)
                and not UNRELATED.search(question))


def session_memory_request(question):
    return bool(re.search(r'预算|偏好|请记住|请忘记|会话记忆', question))


def resolve_request(question, history):
    ids = explicit_products(question)
    orders = list(dict.fromkeys(ORDER_IDS.findall(question)))
    recent_ids, recent_order = [], None
    for message in reversed(history):
        if message.get('role') != 'assistant' or message.get('decision', {}).get('scope', {}).get('status') == 'blocked':
            continue
        plan = message.get('decision', {}).get('request') or {}
        trace = message.get('tool_calls', [])
        order_hits = [t['result']['result'] for t in trace if t.get('name') == 'order_detail'
                      and t.get('result', {}).get('ok') is True and isinstance(t['result'].get('result'), dict)]
        if not recent_order and order_hits:
            recent_order = order_hits[-1]['order_id']
        if not recent_ids:
            candidates = []
            for t in trace:
                if t.get('result', {}).get('ok') is not True:
                    continue
                value = t['result'].get('result')
                if t.get('name') == 'order_detail' and isinstance(value, dict):
                    candidates.extend(i['product_id'] for i in value.get('items', []) if i.get('product_id'))
                elif t.get('name') == 'catalog_search' and isinstance(value, list):
                    candidates.extend(i['product_id'] for i in value if isinstance(i, dict) and i.get('product_id'))
                elif t.get('name') == 'knowledge_search' and isinstance(value, dict):
                    candidates.extend(i['product_id'] for i in value.get('hits', []) if i.get('product_id'))
            recent_ids = list(dict.fromkeys(candidates)) or plan.get('product_ids', [])
        if recent_order and recent_ids:
            break
    reference = bool(re.search(r'它|这款|那款|这单|那单|现在价格|售价呢|价格呢|现在.*库存', question)) or historical_request(question)
    if not orders and recent_order and ORDER.search(question):
        orders = [recent_order]
    ambiguous = bool(not ids and not orders and reference and len(recent_ids) > 1)
    if not ids and not orders and reference and len(recent_ids) == 1:
        ids = recent_ids
    fields = requested_fields(question)
    roles = (['orders'] if orders else []) + (['product'] if fields and (ids or orders) else [])
    return {'order_ids': orders, 'product_ids': ids, 'fields': fields,
            'roles': roles, 'ambiguous': ambiguous, 'historical': historical_request(question),
            'supported': bool(roles) and not ambiguous and not UNRELATED.search(question),
            'source': 'explicit' if explicit_products(question) or re.search(r'\bORD\d+\b', question) else 'verified_history'}


def task_text(question, request, role):
    tools = []
    if {'price','stock'}.intersection(request['fields']):tools.append('catalog_search')
    if 'knowledge' in request['fields']:tools.append('knowledge_search')
    instruction = '\n本任务仅需 ' + ', '.join(tools) + '，不查询其他字段。'
    if 'knowledge' in request['fields']:
        instruction += '知识类型优先product；已有有效证据就停止。'
    return (question + '\n已解析的本轮任务（历史只继承标识，不继承当前数值）：' +
            json.dumps({k: request[k] for k in ('order_ids', 'product_ids', 'fields')}, ensure_ascii=False) +
            ('\n先查询给定本人订单，报告真实商品ID及用户所问的订单字段。' if role == 'orders' else
             instruction))
