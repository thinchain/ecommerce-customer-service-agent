from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.commerce import router as commerce_router
from app.api.routes.health import router as health_router
from app.ops.telemetry import ObservabilityMiddleware
from app.services.qwen import close_model_client
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings()  # Fail at startup for invalid environment values/combinations.
    try:
        yield
    finally:
        await close_model_client()


app = FastAPI(title="电商客服 Agent API", lifespan=lifespan)
app.include_router(chat_router, prefix="/api")
app.include_router(commerce_router, prefix="/api")
app.include_router(health_router)
app.add_middleware(ObservabilityMiddleware)
