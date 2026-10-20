import asyncio
import sqlite3
from pathlib import Path

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import get_settings

router = APIRouter()


@router.get('/health/live')
def live():
    return {'status': 'alive'}


def check_database(path, table):
    try:
        with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True,timeout=1) as db:
            return db.execute(f'SELECT count(*) FROM {table}').fetchone()[0] > 0
    except sqlite3.Error:
        return False


@router.get('/health/ready')
async def ready():
    settings = get_settings()
    commerce, knowledge = await asyncio.gather(
        asyncio.to_thread(check_database,settings.commerce_db_path,'products'),
        asyncio.to_thread(check_database,settings.knowledge_db_path,'chunks'))
    model = False
    try:
        async with httpx.AsyncClient(timeout=2,trust_env=False) as client:
            response = await client.get(settings.model_base_url+'/models',headers={'Authorization':'Bearer '+settings.model_api_key.get_secret_value()})
            model = response.is_success
    except httpx.HTTPError:
        pass
    checks = {'commerce':commerce,'knowledge':knowledge,'model':model}
    return JSONResponse({'ready':all(checks.values()),'checks':checks},status_code=200 if all(checks.values()) else 503)


@router.get('/metrics', include_in_schema=False)
def metrics():
    return Response(generate_latest(),headers={'Content-Type':CONTENT_TYPE_LATEST})
