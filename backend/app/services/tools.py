import ast
import json
import math
import operator
import logging
import sqlite3
import httpx
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.commerce.tools import BUSINESS_TOOLS


class Arguments(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True)


class Calculation(Arguments):
    expression: str = Field(min_length=1, max_length=200)


class TimeQuery(Arguments):
    timezone: str = Field(default='Asia/Shanghai', max_length=80)


class MemoryKey(Arguments):
    key: str = Field(min_length=1, max_length=40)


class MemoryValue(MemoryKey):
    value: str = Field(min_length=1, max_length=200)


class Empty(Arguments):
    pass


def calculate(expression):
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 60:
        raise ValueError('表达式过于复杂')
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
           ast.Mod: operator.mod, ast.Pow: operator.pow}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError('指数绝对值不能超过 100')
            result = ops[type(node.op)](left, right)
        else:
            raise ValueError('只支持数字、括号及 + - * / // % ** 运算')
        if type(result) not in (int, float) or not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError('结果超出支持范围')
        return result
    return visit(tree.body)


# The registry is the only executable tool allowlist; schemas and validation share one model.
REGISTRY = {
    **{name: (model, description) for name, (model, description, _) in BUSINESS_TOOLS.items()},
    'calculator': (Calculation, '精确计算算术表达式，支持 + - * / // % ** 和括号。'),
    'current_time': (TimeQuery, '查询指定 IANA 时区的当前日期时间，默认 Asia/Shanghai。'),
    'memory_set': (MemoryValue, '用户明确要求记住时，保存或更新当前会话的稳定事实。'),
    'memory_list': (Empty, '查看当前会话保存的长期记忆。'),
    'memory_delete': (MemoryKey, '用户要求忘记某事实时，按 key 删除当前会话记忆。'),
}
TOOL_SCHEMAS = [{'type': 'function', 'function': {'name': name, 'description': description,
                'parameters': model.model_json_schema()}} for name, (model, description) in REGISTRY.items()]


def execute_tool(name, raw_arguments, memories):
    try:
        if name not in REGISTRY:
            raise ValueError('未知工具')
        if len(raw_arguments) > 2000:
            raise ValueError('工具参数过长')
        args = REGISTRY[name][0].model_validate_json(raw_arguments)
        from app.agents.policy import danger_reason
        if danger_reason(json.dumps(args.model_dump(), ensure_ascii=False)):
            return {'ok': False, 'error': '工具请求包含不允许的操作', 'code': 'policy_blocked'}
        if name in BUSINESS_TOOLS:
            result = BUSINESS_TOOLS[name][2](args.model_dump())
        elif name == 'calculator':
            result = calculate(args.expression)
        elif name == 'current_time':
            result = datetime.now(ZoneInfo(args.timezone)).isoformat()
        elif name == 'memory_list':
            result = dict(memories)
        elif name == 'memory_set':
            updated = {**memories, args.key: args.value}
            if len(updated) > 20 or len(json.dumps(updated, ensure_ascii=False).encode()) > 1800:
                raise ValueError('记忆容量已满，请删除旧记忆后重试')
            memories.update(updated)
            result = {'key': args.key, 'value': args.value}
        else:
            result = {'deleted': memories.pop(args.key, None) is not None}
        # Bound observations before they enter the ReAct context. Keep complete records.
        limited = False
        records = result if isinstance(result, list) else result.get('hits') if isinstance(result, dict) else None
        before = len(records) if isinstance(records, list) else None
        if isinstance(records, list):
            while len(json.dumps(result, ensure_ascii=False).encode()) > 6500 and records:
                records.pop()
                limited = True
        return {'ok': True, 'result': result, **({'truncated': True} if limited else {}),
                **({'result_window': {'before': before, 'retained': len(records)}}
                   if name in ('catalog_search', 'knowledge_search') and isinstance(records, list) else {})}
    except (ValidationError, ValueError, SyntaxError, ArithmeticError, ZoneInfoNotFoundError) as exc:
        return {'ok': False, 'error': str(exc)[:300]}
    except (httpx.HTTPError, sqlite3.Error) as exc:
        logging.getLogger(__name__).warning('Tool %s unavailable: %s', name, type(exc).__name__)
        return {'ok': False, 'error': '工具依赖暂时不可用，请重新尝试。', 'code': 'dependency_unavailable', 'retryable': True}
    except Exception as exc:
        logging.getLogger(__name__).exception('Tool %s failed (%s)', name, type(exc).__name__)
        return {'ok': False, 'error': '工具执行异常，请稍后重试。', 'code': 'tool_error'}
