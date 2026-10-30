"""Deterministic catalog evidence. No model drafts, database reads, or network calls.

Observation time means tool-return time, not a database modification timestamp.
Current queries use current observations; history recall labels captured old facts.
"""
import json
import hashlib
import re
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.commerce.tools import CatalogSearch


ProductId = Annotated[str, Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9:._/-]{0,159}$', strict=True)]
CatalogInt = Annotated[int, Field(strict=True, ge=0, le=2**63-1)]
_IDS = re.compile(r'(?<![A-Za-z0-9_:-])(?:SYN-[A-Z]+-\d+|DEMO-[A-Z0-9_-]+|amazon:[a-z]{2}:[A-Z0-9_-]+)(?![A-Za-z0-9_:-])', re.I)
_NAMED_ID = re.compile(r'(?:商品\s*(?:ID|编号)|product\s+ID)\s*[:：=]?\s*([A-Za-z0-9][A-Za-z0-9:._/-]{0,159})', re.I)
_CUSTOM_ID = re.compile(r'(?:商品|产品|product)\s*[:：#]?\s*([A-Za-z][A-Za-z0-9:._/-]*\d[A-Za-z0-9:._/-]*)', re.I)
_CATALOG = re.compile(r'价格|价钱|库存|存货|余量|多少钱|多少件|售价|有货|缺货|\b(?:price|stock|inventory|availability)\b', re.I)
_CATALOG_VALUE = re.compile(r'[¥￥$€£]\s*\d|\b(?:USD|CNY|RMB|EUR)\s*\d|\d+(?:\.\d+)?\s*(?:元|USD\b|CNY\b|RMB\b|EUR\b)|(?:还有|仅剩|剩余|剩下|有)\s*\d+\s*(?:件|台|个)|\bprice\s*[:=]?\s*\d', re.I)
_PRODUCT = re.compile(r'商品|产品|耳机|摄像头|键盘|鼠标|这款|那款|它|\b(?:product|headphones?|camera|keyboard|mouse)\b', re.I)


class EvidenceFact(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    product_id: ProductId
    field: Literal['price_cents', 'stock']
    value: CatalogInt
    observation_id: str
    observed_at: datetime
    source_path: str


class CatalogRow(BaseModel):
    # Other catalog columns are not facts in this MVP and are never rendered.
    model_config = ConfigDict(extra='ignore', strict=True)
    product_id: ProductId
    price_cents: CatalogInt
    stock: CatalogInt


def product_ids(text):
    return list(dict.fromkeys([*_IDS.findall(text), *_NAMED_ID.findall(text), *_CUSTOM_ID.findall(text)]))


def result_digest(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def catalog_observation(raw_arguments, result):
    """Normalize once, immediately after execution, independent of local/MCP transport."""
    observation = {'observation_id': str(uuid4()), 'observed_at': datetime.now(timezone.utc).isoformat(),
                   'arguments': None, 'execution_ok': isinstance(result, dict) and result.get('ok') is True,
                   'status': 'invalid_arguments', 'facts': [], 'issues': [], 'truncated': False}
    try:
        observation['result_sha256'] = result_digest(result)
    except (ValueError, TypeError):
        observation['status'] = 'invalid_result'
        return observation
    try:
        if not isinstance(raw_arguments, str) or len(raw_arguments) > 2000:
            return observation
        arguments = CatalogSearch.model_validate_json(raw_arguments)
        if arguments.product_ids is not None:
            # Validate IDs too: CatalogSearch currently only bounds the list length.
            for pid in arguments.product_ids:
                CatalogRow(product_id=pid, price_cents=0, stock=0)
        observation['arguments'] = arguments.model_dump()
    except (ValidationError, ValueError, TypeError):
        return observation
    if not isinstance(result, dict) or type(result.get('ok')) is not bool:
        observation['status'] = 'invalid_result'
        return observation
    if not result['ok']:
        observation['status'] = 'tool_error'
        return observation
    rows = result.get('result')
    if not isinstance(rows, list) or len(rows) > arguments.limit or type(result.get('truncated', False)) is not bool:
        observation['status'] = 'invalid_result'
        return observation
    observation['truncated'] = result.get('truncated', False)
    valid, conflicted = {}, set()
    for index, row in enumerate(rows):
        try:
            record = CatalogRow.model_validate(row)
        except ValidationError:
            observation['issues'].append('invalid_fields')
            # A malformed row can conceal a conflicting duplicate: reject the batch.
            observation['status'] = 'invalid_fields'
            return observation
        pid = record.product_id
        if arguments.product_ids is not None and pid not in arguments.product_ids:
            observation['issues'].append('entity_mismatch')
            observation['status'] = 'entity_mismatch'
            return observation
        if ((arguments.max_price_cents is not None and record.price_cents > arguments.max_price_cents)
                or (arguments.in_stock and record.stock == 0)):
            observation['issues'].append('filter_mismatch')
            observation['status'] = 'filter_mismatch'
            return observation
        values = (record.price_cents, record.stock)
        if pid in valid and valid[pid][0] != values:
            conflicted.add(pid)
        valid[pid] = (values, index)
    for pid in conflicted:
        valid.pop(pid)
        observation['issues'].append('conflicting_rows')
    for pid, (values, index) in valid.items():
        for field, value in zip(('price_cents', 'stock'), values):
            fact = EvidenceFact(product_id=pid, field=field, value=value,
                                observation_id=observation['observation_id'],
                                observed_at=datetime.fromisoformat(observation['observed_at']),
                                source_path=f'result[{index}].{field}')
            observation['facts'].append(fact.model_dump(mode='json'))
    observation['status'] = 'usable' if valid else observation['issues'][0] if observation['issues'] else 'empty'
    return observation


def needs_catalog_render(question, draft, trace):
    if any(item['name'] == 'catalog_search' for item in trace):
        return True
    if re.search(r'库存|stock|当前价格|售价|商品价格|\bprice(?:_cents)?\b', draft, re.I):
        return True
    product_trace = any(item.get('agent') == 'product' for item in trace)
    if not product_trace:
        # An order's historical amount or a saved budget is outside this MVP.
        # Do not reinterpret their numeric text as a current catalog assertion.
        if (any(item['name'].startswith('memory_') for item in trace)
                and re.search(r'记住|偏好|忘记', question) and not re.search(r'查询|核验|现在|当前', question)):
            return False
        if (re.search(r'订单|物流|退款|支付|\bORD\d+\b', question, re.I)
                and not _CATALOG.search(question)):
            return False
    if _CATALOG.search(question):
        return True  # Named products/references need not contain a product keyword.
    if _CATALOG.search(draft) or _CATALOG_VALUE.search(draft):
        return True  # A model must not introduce catalog values into a name-only question.
    product_context = bool(product_ids(question) or _PRODUCT.search(question)
                           or product_trace)
    return product_context and bool(_CATALOG.search(question) or _CATALOG.search(draft)
                                    or _CATALOG_VALUE.search(draft))


def render_catalog_reply(question, draft, trace, *, target_ids=None, historical=False):
    """Return None outside catalog scope. Never append an unchecked business draft.

Explicit user IDs take precedence over model-selected query IDs. Without an
explicit ID, results are labelled candidates, not a resolved user reference.
Latest relevant attempts (including failures/empty results) supersede old facts.
"""
    if not needs_catalog_render(question, draft, trace) and not (target_ids and _CATALOG_VALUE.search(draft)):
        return None
    observations = []
    for item in trace:
        if item['name'] == 'catalog_search':
            observation = item.get('catalog_evidence')
            if observation is not None:
                try:
                    unchanged = observation.get('result_sha256') == result_digest(item['result'])
                except (ValueError, TypeError):
                    unchanged = False
                if not unchanged:
                    observation = {**observation, 'facts': [], 'status': 'source_changed'}
            if observation is None:
                # Missing capture is not permission to reconstruct trusted facts later.
                observation = {'arguments': None, 'facts': [], 'status': 'missing_observation',
                               'execution_ok': False, 'truncated': False}
            observations.append(observation)
    from app.agents.request import explicit_products, requested_fields
    fields = requested_fields(question)
    explicit = explicit_products(question) or list(target_ids or [])
    targets = list(explicit)
    if not targets:
        for observation in observations:
            args = observation['arguments'] or {}
            for pid in [*(args.get('product_ids') or []), *(f['product_id'] for f in observation['facts'])]:
                if pid not in targets:
                    targets.append(pid)
    latest, scopes = {}, {}
    for observation in observations:
        args = observation['arguments']
        if args is None:
            covered = set(targets)  # Unknown scope fails closed, never revives older facts.
        else:
            signature = json.dumps({k: v for k, v in args.items() if k != 'limit'}, sort_keys=True)
            covered = set(args['product_ids'] or []) | set(product_ids(args['query']))
            covered.update(f['product_id'] for f in observation['facts'])
            covered.update(scopes.get(signature, set()))
            if args['product_ids'] is None and not args['query']:
                covered.update(targets)
            scopes[signature] = covered
        for pid in targets:
            if pid in covered:
                latest[pid] = observation
    facts, missing, sources, lines = [], [], {}, []
    if not explicit and targets:
        lines.append('以下是本次工具查询返回的候选商品，尚未确认是否为你所指的商品：')
    for pid in targets:
        observation = latest.get(pid)
        selected = [fact for fact in observation['facts'] if fact['product_id'] == pid] if observation else []
        by_field = {fact['field']: fact for fact in selected}
        if set(by_field) != {'price_cents', 'stock'}:
            reason = observation['status'] if observation else 'not_queried'
            if reason == 'usable':
                reason = 'not_returned'
            missing.append({'product_id': pid, 'reason': reason})
            if observation:
                sources[pid] = observation.get('observation_id')
            lines.append(f'商品 `{pid}`：本轮未取得足够的价格库存证据，暂不能确认。')
            continue
        cents, stock = by_field['price_cents']['value'], by_field['stock']['value']
        sources[pid] = observation['observation_id']
        facts.extend(selected)
        values = []
        if 'price' in fields:
            values.append(f'本轮查询价格 ¥{cents // 100}.{cents % 100:02d}')
        if 'stock' in fields:
            values.append(f'库存 {stock} 件' + ('（缺货）' if stock == 0 else ''))
        if values:
            lines.append(f'商品 `{pid}`：' + '，'.join(values) + '。[catalog_search]')
    if not targets:
        lines.append('本轮未取得可校验的商品价格库存，请提供明确的商品 ID 后重新查询。')
    if any(observation['truncated'] for observation in observations):
        lines.append('查询结果有截断，仅展示已返回并校验的记录，不代表全部结果。')
    if not {'price', 'stock'}.intersection(fields):
        lines = []
    if historical:
        lines = [line.replace('本轮', '历史').replace('本次', '该次') for line in lines]
    return '\n\n'.join(lines), {'version': 1, 'mode': 'server_catalog',
                                  'entity_scope': 'resolved' if explicit else 'candidates',
                                  'status': 'partial' if facts and missing else 'bound' if facts else 'insufficient',
                                  'facts': facts, 'missing': missing, 'sources': sources,
                                  'observations': observations, 'draft_published': False,
                                  'scope': 'historical' if historical else 'current',
                                  'additional_llm_calls': 0}
