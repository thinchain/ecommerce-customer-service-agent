import json
import logging
import time
from uuid import uuid4

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter('commerce_http_requests_total', 'API requests', ['route', 'status'])
LATENCY = Histogram('commerce_http_duration_seconds', 'API duration', ['route'], buckets=(.01,.05,.1,.5,1,3,10,30,60,180))
ACTIVE = Gauge('commerce_active_chat_requests', 'Active chat turns in this process')
MODEL_REQUESTS = Counter('commerce_model_requests_total', 'Model attempts', ['outcome'])
MODEL_LATENCY = Histogram('commerce_model_duration_seconds', 'Model attempt duration', buckets=(.1,.5,1,3,10,30,60,120))
TOKENS = Counter('commerce_model_tokens_total', 'Model token usage', ['kind'])
TOOL_CALLS = Counter('commerce_tool_calls_total', 'Tool executions', ['tool', 'outcome'])
TOOL_LATENCY = Histogram('commerce_tool_duration_seconds', 'Tool duration', ['tool'], buckets=(.001,.01,.1,1,5,15,45))
logger = logging.getLogger('commerce.http')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.propagate = False


class ObservabilityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        started, status, request_id = time.perf_counter(), 500, str(uuid4())
        scope.setdefault('state', {})['request_id'] = request_id
        async def tracked_send(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
                message['headers'] = [*message.get('headers', []), (b'x-request-id',request_id.encode())]
            await send(message)
        try:
            await self.app(scope, receive, tracked_send)
        finally:
            # Route template, never user/session IDs: bounded label cardinality.
            route = getattr(scope.get('route'), 'path', 'unmatched')
            elapsed = time.perf_counter()-started
            REQUESTS.labels(route, str(status)).inc()
            LATENCY.labels(route).observe(elapsed)
            logger.info(json.dumps({'request_id':request_id,'route':route,'status':status,'duration_ms':round(elapsed*1000,2)}))
