import logging
import asyncio
import random
import time
import httpx

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, OpenAIError

from pydantic import ValidationError
from app.core.config import get_settings
from app.core.exceptions import ModelServiceError
from app.core.message import Message, model_messages
from app.ops.resilience import CircuitBreaker, CircuitOpen
from app.ops.telemetry import MODEL_REQUESTS, MODEL_LATENCY, TOKENS

logger = logging.getLogger(__name__)

client = None
breaker = None
_clients = {}
_breakers = {}


def initialize_model_client():
    """Keep clients/breakers scoped to their configuration; injected clients stay caller-owned."""
    settings = get_settings()
    key = (settings.model_base_url, settings.model_api_key, settings.model_timeout_seconds)
    active_client = client
    if active_client is None:
        if key not in _clients:
            _clients[key] = AsyncOpenAI(
                api_key=settings.model_api_key.get_secret_value(), base_url=settings.model_base_url,
                timeout=settings.model_timeout_seconds,
                http_client=httpx.AsyncClient(trust_env=False), max_retries=0,
            )
        active_client = _clients[key]
    breaker_key = (*key, settings.model_circuit_threshold, settings.model_circuit_cooldown)
    active_breaker = breaker
    if active_breaker is None:
        if breaker_key not in _breakers:
            _breakers[breaker_key] = CircuitBreaker(settings.model_circuit_threshold, settings.model_circuit_cooldown)
        active_breaker = _breakers[breaker_key]
    return active_client, active_breaker


async def create_completion(**kwargs):
    active_client, active_breaker = initialize_model_client()
    settings = get_settings()
    try:
        active_breaker.before()
    except CircuitOpen as exc:
        MODEL_REQUESTS.labels('circuit_open').inc()
        raise ModelServiceError(503, '模型服务暂时熔断，冷却后将自动探测恢复。') from exc
    try:
        for attempt in range(settings.model_retries + 1):
            started = time.perf_counter()
            try:
                result = await active_client.chat.completions.create(**kwargs)
                MODEL_REQUESTS.labels('success').inc()
                if result.usage:
                    TOKENS.labels('prompt').inc(result.usage.prompt_tokens)
                    TOKENS.labels('completion').inc(result.usage.completion_tokens)
                active_breaker.success()
                return result
            except APITimeoutError:
                # The server may still be generating. Retrying adds duplicate work
                # to the shared GPU queue, even though the caller stopped waiting.
                MODEL_REQUESTS.labels('timeout').inc()
                active_breaker.failure()
                raise
            except (APIConnectionError, APIStatusError) as exc:
                transient = isinstance(exc, APIConnectionError) or exc.status_code in (408,429) or exc.status_code >= 500
                MODEL_REQUESTS.labels('transient_error' if transient else 'permanent_error').inc()
                if not transient:
                    active_breaker.cancel()
                    raise
                if attempt == settings.model_retries:
                    active_breaker.failure()
                    raise
            finally:
                MODEL_LATENCY.observe(time.perf_counter() - started)
            await asyncio.sleep(.2 * (2 ** attempt) + random.random() * .1)
    except BaseException:
        active_breaker.cancel()
        raise


async def complete(messages: list[Message | dict], tools: list[dict], *, final: bool = False,
                   required_tool: str | None = None) -> Message:
    settings = get_settings()
    try:
        messages = model_messages(messages)
    except ValueError as exc:
        raise ModelServiceError(422, '模型上下文的消息或工具配对格式不合法。') from exc
    tool_choice = 'none' if final else (
        {'type': 'function', 'function': {'name': required_tool}} if required_tool else 'auto'
    )
    try:
        completion = await create_completion(
            model=settings.model_name,
            messages=messages,
            **({'tools': tools, 'tool_choice': tool_choice} if tools else {}),
            max_tokens=settings.model_max_tokens,
            temperature=settings.model_temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except APITimeoutError as exc:
        logger.warning("Local Qwen request timed out")
        raise ModelServiceError(504, "共用模型排队或生成超时，请稍后重试；持续超时时检查模型队列及 MODEL_TIMEOUT_SECONDS 配置。") from exc
    except APIConnectionError as exc:
        logger.warning("Could not connect to local model at %s", settings.model_base_url)
        raise ModelServiceError(
            502, "无法连接本地模型服务，请检查 vLLM 是否已启动及服务地址。"
        ) from exc
    except APIStatusError as exc:
        logger.warning("Local model returned HTTP %s", exc.status_code)
        raise ModelServiceError(
            502, "本地模型服务调用失败，请检查模型名称、vLLM 工具解析参数及日志。"
        ) from exc
    except OpenAIError as exc:
        logger.warning("Local model request failed: %s", type(exc).__name__)
        raise ModelServiceError(502, "本地模型调用失败，请检查 vLLM 配置。") from exc

    if not completion.choices:
        raise ModelServiceError(502, "本地模型返回了空回复，请重试。")
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ModelServiceError(502, "模型输出超出长度限制，请缩小问题范围后重试。")
    try:
        return Message(role=choice.message.role, content=choice.message.content,
                       tool_calls=[call.model_dump() for call in choice.message.tool_calls or []],
                       metadata={'usage': completion.usage.model_dump() if completion.usage else None,
                                 'model': completion.model, 'finish_reason': choice.finish_reason})
    except ValidationError as exc:
        raise ModelServiceError(502, '本地模型返回了不合法的消息。') from exc


async def close_model_client() -> None:
    owned = list(_clients.values())
    _clients.clear()
    _breakers.clear()
    await asyncio.gather(*(item.close() for item in owned))
