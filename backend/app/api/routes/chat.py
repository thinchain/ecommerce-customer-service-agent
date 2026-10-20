import json
from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.agent import run_agent
from app.services.memory import MemoryStore
from app.core.exceptions import ModelServiceError
from app.services.tools import MemoryValue, execute_tool
from app.ops.resilience import Admission, CapacityExceeded
from app.ops.telemetry import ACTIVE
from app.core.config import get_settings

router = APIRouter()


@lru_cache
def _admission(capacity):
    return Admission(capacity)


def get_store():
    settings = get_settings()
    return _store(settings.memory_db_path, settings.demo_user_id, settings.agent_timeout_seconds)


@lru_cache
def _store(path, user_id, timeout):
    store = MemoryStore(path, user_id)
    store.lease_seconds = timeout + 30
    return store


@router.post('/sessions')
async def create_session():
    return get_store().get(get_store().create())


@router.get('/sessions/{session_id}')
async def get_session(session_id: UUID):
    try:
        return get_store().get(str(session_id))
    except KeyError as exc:
        raise HTTPException(404, '会话不存在，请新建对话。') from exc


@router.delete('/sessions/{session_id}')
async def delete_session(session_id: UUID):
    try:
        get_store().delete(str(session_id))
    except KeyError as exc:
        raise HTTPException(404, '会话不存在。') from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {'deleted': True}


@router.put('/sessions/{session_id}/memories')
async def save_memory(session_id: UUID, request: MemoryValue):
    return edit_memory(str(session_id), 'memory_set', request.model_dump_json())


@router.delete('/sessions/{session_id}/memories/{key:path}')
async def delete_memory(session_id: UUID, key: str):
    return edit_memory(str(session_id), 'memory_delete', json.dumps({'key': key}))


def edit_memory(session_id, tool, arguments):
    store = get_store()
    try:
        with store.edit(session_id) as session:
            result = execute_tool(tool, arguments, session['memories'])
            if not result['ok']:
                raise HTTPException(422, result['error'])
            store.save(session_id, session['messages'], session['memories'])
            return {'memories': session['memories']}
    except KeyError as exc:
        raise HTTPException(404, '会话不存在。') from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post('/chat', response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        with _admission(get_settings().max_active_chats).enter():
            ACTIVE.inc()
            try:
                return await _chat(request)
            finally:
                ACTIVE.dec()
    except CapacityExceeded as exc:
        raise HTTPException(429, '服务繁忙，请稍后重试。', headers={'Retry-After': '5'}) from exc


async def _chat(request: ChatRequest) -> ChatResponse:
    store = get_store()
    session_id = str(request.session_id) if request.session_id else store.create()
    try:
        return ChatResponse(**await run_agent(request.message, session_id, store))
    except ModelServiceError as exc:
        if request.session_id is None:
            store.delete(session_id)
        raise HTTPException(exc.status_code, exc.detail) from exc
    except KeyError as exc:
        raise HTTPException(404, '会话不存在，请新建对话。') from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
