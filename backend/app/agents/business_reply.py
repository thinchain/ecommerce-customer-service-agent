"""Independent sections from observations, never model prose. No extra I/O/LLM.

Knowledge is a complete chunk quote, not a semantic entailment check. Orders
display the existing authorized tool records, not a new state verifier.
"""
import copy
import re

from pydantic import ValidationError

from app.agents.catalog_evidence import (CatalogRow, _CATALOG_VALUE, _PRODUCT,
                                        product_ids, render_catalog_reply, result_digest)
from app.commerce.tools import KnowledgeSearch, OrderDetail

_KNOWLEDGE = re.compile(r'兼容|功能|支持|适合|适用|连接|续航|参数|政策|退货期限|几天.*退|Windows|macOS|Bluetooth', re.I)
_ORDERS = re.compile(r'订单|物流|支付|退款|\bORD\d+\b', re.I)
_CITATION = re.compile(r'KB-[a-f0-9]{12}')
_STATUS = {'in_transit': '运输中', 'delivered': '已签收', 'delayed': '物流延迟', 'paid': '已支付',
           'failed': '失败', 'void': '已作废', 'payment_failed': '支付失败', 'cancelled': '已取消',
           'refund_pending': '退款处理中', 'refunded': '已退款', 'partial_refund': '部分退款',
           'return_expired': '超过演示退货期', 'exception': '异常', 'pending': '处理中', 'succeeded': '处理成功'}


def capture_knowledge(arguments, result):
    """Snapshot post-tool-boundary records; handoff trimming cannot mutate it."""
    captured = {'arguments': None, 'status': 'invalid_arguments', 'hits': [], 'truncated': False}
    try:
        args = KnowledgeSearch.model_validate_json(arguments)
        captured['arguments'] = args.model_dump()
        captured['result_sha256'] = result_digest(result)
    except (ValueError, TypeError, ValidationError):
        return captured
    if not isinstance(result, dict) or result.get('ok') is not True:
        captured['status'] = 'tool_error'
        return captured
    body = result.get('result')
    hits = body.get('hits') if isinstance(body, dict) else None
    if not isinstance(hits, list) or len(hits) > args.limit:
        captured['status'] = 'invalid_result'
        return captured
    captured.update(status='empty' if not hits else 'available', hits=copy.deepcopy(hits),
                    truncated=result.get('truncated') is True)
    return captured


def _safe_pid(value):
    try:
        return CatalogRow(product_id=value, price_cents=0, stock=0).product_id
    except ValidationError:
        return None


def order_product_ids(trace):
    ids = []
    for item in trace:
        if item['name'] != 'order_detail' or item['result'].get('ok') is not True:
            continue
        body = item['result'].get('result', {})
        for row in body.get('items', []) if isinstance(body, dict) else []:
            pid = _safe_pid(row.get('product_id')) if isinstance(row, dict) else None
            if pid and pid not in ids:
                ids.append(pid)
    return ids


def _literal(text):
    return re.sub(r'([\\`*_{}\[\]()<>#+!|])', r'\\\1', text).replace('\n', '\n> ')


def knowledge_section(question, trace, targets):
    relevant = [item for item in trace if item['name'] == 'knowledge_search']
    if not (relevant or (_KNOWLEDGE.search(question) and (_PRODUCT.search(question) or targets or '政策' in question))):
        return None
    latest, issues, budget = {}, [], 6500
    for item in relevant:
        snapshot = item.get('knowledge_evidence')
        if not snapshot:
            issues.append('missing_observation'); continue
        args = snapshot['arguments'] or {}
        key = (args.get('product_id') or ('policy' if args.get('kind') == 'policy' else 'unscoped'),
               args.get('kind') or '')
        try:
            unchanged = snapshot.get('result_sha256') == result_digest(item['result'])
        except (ValueError, TypeError):
            unchanged = False
        latest[key] = snapshot if unchanged else {**snapshot, 'hits': [], 'status': 'source_changed'}
    quotes = []
    for (key, kind), snapshot in latest.items():
        if snapshot['status'] != 'available':
            issues.append(snapshot['status']); continue
        for hit in snapshot['hits']:
            if not isinstance(hit, dict):
                issues.append('invalid_hit'); continue
            text, citation, pid = hit.get('text'), hit.get('citation'), hit.get('product_id', '')
            args = snapshot['arguments'] or {}
            policy = hit.get('kind') == 'policy'
            if (not policy and not _safe_pid(pid)) or (policy and pid != '' and not _safe_pid(pid)):
                issues.append('unbound_product'); continue
            if key == 'unscoped' and any(k[0] == pid and k[1] in ('', hit.get('kind')) for k in latest):
                # A scoped follow-up, including failure, supersedes broad candidates.
                continue
            if (not isinstance(text, str) or not text.strip() or len(text) > 2000
                    or not isinstance(citation, str) or not _CITATION.fullmatch(citation)):
                issues.append('invalid_hit'); continue
            if ((args.get('product_id') and pid != args['product_id'])
                    or (args.get('kind') and hit.get('kind') != args['kind'])
                    or (targets and pid not in targets and not policy)):
                issues.append('entity_mismatch'); continue
            # Omit the entire chunk rather than removing negations or conditions.
            if (_CATALOG_VALUE.search(text) or re.search(r'(?:库存|stock|price_cents)\s*[:：=]?\s*[\d零一二三四五六七八九十百千]', text, re.I)
                    or re.search(r'[零一二三四五六七八九十百千万]+元', text)
                    or _CITATION.search(text)):
                issues.append('catalog_or_citation_text_omitted'); continue
            from app.agents.policy import danger_reason
            if danger_reason(text):
                issues.append('unsafe_quote'); continue
            cost = len(text.encode('utf-8'))
            if cost > budget:
                issues.append('whole_quote_budget'); continue
            budget -= cost
            quote = {'product_id': pid, 'citation': citation, 'text': text,
                     'source': hit.get('source'), 'source_url': hit.get('source_url'),
                     'synthetic': hit.get('synthetic'), 'scope': key}
            if quote not in quotes:
                quotes.append(quote)
    # This is the evidence-preserving fallback. Normal delivery is composed by
    # the language model from the structured evidence, not this diagnostic text.
    lines = []
    for quote in quotes:
        label = quote['product_id'] or '商店政策'
        source = quote['source'] if isinstance(quote['source'], str) else '知识库'
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', source):
            source = '知识库'
        lines.append(f"`{label}` [{quote['citation']}] 来源：{_literal(source)}\n\n> {_literal(quote['text'])}")
    missing = [pid for pid in targets if pid not in {q['product_id'] for q in quotes}]
    if not quotes or missing:
        lines.append('现有资料不足，暂时无法确认您询问的商品信息。')
    return '\n\n'.join(lines), {'status': 'quoted' if quotes and not missing and not issues else 'partial' if quotes else 'insufficient',
                                'quotes': quotes, 'missing_products': missing, 'issues': issues,
                                'semantic_factuality': 'not_measured', 'assurance': 'complete_chunk_quote'}


def order_section(question, trace):
    relevant = [item for item in trace if item['name'] in ('order_detail', 'list_orders')]
    if not relevant and not _ORDERS.search(question):
        return None
    records, tools, issues = {}, {}, []
    from app.agents.request import ORDER_IDS
    requested = set(ORDER_IDS.findall(question))
    for item in relevant:
        result = item['result']
        if item['name'] == 'order_detail':
            try:
                oid = OrderDetail.model_validate_json(item['arguments']).order_id
            except ValidationError:
                issues.append('invalid_order_arguments'); continue
            if requested and oid not in requested:
                issues.append('unrelated_order'); continue
            records.pop(oid, None)
            body = result.get('result')
            if result.get('ok') is True and isinstance(body, dict) and body.get('order_id') == oid:
                records[oid], tools[oid] = body, 'order_detail'
            else:
                issues.append('order_detail_unavailable')
        elif result.get('ok') is True and isinstance(result.get('result'), list):
            for body in result['result']:
                if isinstance(body, dict) and isinstance(body.get('order_id'), str):
                    oid = body['order_id']
                    if (not requested or oid in requested) and oid not in records:
                        records[oid], tools[oid] = body, 'list_orders'
    from app.agents.request import requested_fields
    detail = bool(re.search(r'详情|详细', question))
    items_requested = detail or bool(re.search(r'订单.{0,8}(?:内容|商品|买)|这单.{0,8}(?:内容|商品|买)|买了|买的|哪些商品|商品清单|里面有什么', question))
    status_requested = detail or bool(re.search(r'物流|状态|进度|还没到|退款|支付|查.{0,5}ORD', question, re.I))
    if requested_fields(question) and not re.search(r'物流|状态|进度|还没到|退款|支付|详情|内容|哪些商品|商品清单', question):
        items_requested = status_requested = False
    lines = ['订单查询记录：']
    facts = []
    for oid, body in records.items():
        if not re.fullmatch(r'ORD\d+', oid, re.I):
            continue
        status = body.get('status')
        fact = {'order_id': oid, 'source_tool':tools[oid]}
        if status_requested or not requested_fields(question):
            if status_requested or not items_requested:
                fact['status'] = _STATUS.get(status, '状态未识别') if isinstance(status, str) else '状态未识别'
            lines.append(f"订单 `{oid}`：{_STATUS.get(status, '状态未识别') if isinstance(status, str) else '状态未识别'}。[{tools[oid]}]")
        if items_requested:
            for item in body.get('items', []):
                if isinstance(item, dict) and _safe_pid(item.get('product_id')):
                    if type(item.get('quantity')) is int and item['quantity'] >= 0:
                        fact.setdefault('items', []).append({'product_id': item['product_id'], 'quantity': item['quantity']})
                    lines.append(f"商品 `{item['product_id']}`：{_literal(str(item.get('title', item['product_id'])))}，数量 {item.get('quantity', '未记录')}。[order_detail]")
        for table, label in (('shipments', '物流'), ('payments', '支付'), ('refunds', '退款')):
            if not detail and label not in question and not (label == '物流' and '还没到' in question):
                continue
            for row in body.get(table, []) if isinstance(body.get(table), list) else []:
                if isinstance(row, dict):
                    value = row.get('status')
                    fact.setdefault(table, []).append({'status': _STATUS.get(value, '状态未识别') if isinstance(value, str) else '状态未识别'})
                    lines.append(f"{label}记录：{_STATUS.get(value, '状态未识别') if isinstance(value, str) else '状态未识别'}。")
        facts.append(fact)
    if len(lines) == 1 or issues or requested - records.keys():
        lines.append('部分订单记录本轮未取得，无法确认对应进度；其他查询结果仍可参考。')
    if records and not status_requested and not items_requested and requested_fields(question):
        lines = []
    return '\n\n'.join(lines), {'status': 'records' if records else 'insufficient', 'order_ids': list(records),
                                'facts': facts,
                                'issues': issues, 'assurance': 'existing_tool_record_display',
                                'state_consistency': 'not_measured'}


def compose_business_reply(question, draft, trace, *, target_ids=None, historical=False):
    from app.agents.request import explicit_products, requested_fields
    targets = explicit_products(question) or order_product_ids(trace) or list(target_ids or [])
    catalog = render_catalog_reply(question, draft, trace, target_ids=targets, historical=historical)
    # Candidate IDs constrain knowledge too, without promoting a name match to a
    # resolved user entity in the catalog renderer.
    knowledge_targets = targets or (list(dict.fromkeys(f['product_id'] for f in catalog[1]['facts'])) if catalog else [])
    knowledge = knowledge_section(question, trace, knowledge_targets) if 'knowledge' in requested_fields(question) else None
    orders = order_section(question, trace) if not historical else None
    if catalog is None and knowledge is None and orders is None:
        return None
    parts, lines = {}, []
    for name, section in (('catalog', catalog), ('knowledge', knowledge), ('orders', orders)):
        if section is not None:
            if section[0]:
                lines.append(section[0])
            parts[name] = section[1]
    if not lines:
        return '本轮没有与当前问题匹配的可核验证据。', parts
    return '\n\n'.join(lines), parts
