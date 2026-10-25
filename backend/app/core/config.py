"""Validated settings and task-local injection; preserves deployment environment names."""
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AgentConfig(BaseModel):
    """Execution limits independent of environment loading."""
    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)
    max_steps: int = Field(default=4, ge=1, le=8)
    worker_steps: int = Field(default=3, ge=1, le=6)
    max_handoffs: int = Field(default=4, ge=1, le=8)
    max_tool_calls: int = Field(default=16, ge=1, le=40)
    timeout_seconds: float = Field(default=600, gt=0)
    context_bytes: int = Field(default=12000, ge=4500, le=14000)
    max_history_turns: int = Field(default=50, ge=1, le=1000)
    handoff_report_bytes: int = Field(default=6500, ge=1000, le=14000)
    max_calls_per_response: int = Field(default=4, ge=1, le=16)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / '.env', env_file_encoding='utf-8',
        extra='ignore', frozen=True, populate_by_name=True, allow_inf_nan=False,
    )

    # Model client / resilience
    model_base_url: str = Field(default='http://127.0.0.1:8001/v1', validation_alias='OPENAI_BASE_URL')
    model_api_key: SecretStr = Field(default='local-dev-key', validation_alias='OPENAI_API_KEY')
    model_name: str = Field(default='qwen3.5-4b', min_length=1, validation_alias='QWEN_MODEL')
    model_timeout_seconds: float = Field(default=180, gt=0, validation_alias='MODEL_TIMEOUT_SECONDS')
    model_retries: int = Field(default=1, ge=0, le=2, validation_alias='MODEL_RETRIES')
    model_circuit_threshold: int = Field(default=5, ge=1, validation_alias='MODEL_CIRCUIT_THRESHOLD')
    model_circuit_cooldown: float = Field(default=30, gt=0, validation_alias='MODEL_CIRCUIT_COOLDOWN')
    model_temperature: float = Field(default=.3, ge=0, le=2, validation_alias='MODEL_TEMPERATURE')
    model_max_tokens: int = Field(default=1200, ge=1, validation_alias='MODEL_MAX_TOKENS')
    max_active_chats: int = Field(default=2, ge=1, validation_alias='MAX_ACTIVE_CHATS')

    # Agent execution / history
    agent_max_steps: int = Field(default=4, ge=1, le=8, validation_alias='AGENT_MAX_STEPS')
    agent_worker_steps: int = Field(default=3, ge=1, le=6, validation_alias='AGENT_WORKER_STEPS')
    agent_max_handoffs: int = Field(default=4, ge=1, le=8, validation_alias='AGENT_MAX_HANDOFFS')
    agent_max_tool_calls: int = Field(default=16, ge=1, le=40, validation_alias='AGENT_MAX_TOOL_CALLS')
    agent_timeout_seconds: float = Field(default=600, gt=0, validation_alias='AGENT_TIMEOUT_SECONDS')
    agent_context_bytes: int = Field(default=12000, ge=4500, le=14000, validation_alias='AGENT_CONTEXT_BYTES')
    agent_max_history_turns: int = Field(default=50, ge=1, le=1000, validation_alias='AGENT_MAX_HISTORY_TURNS')
    agent_handoff_report_bytes: int = Field(default=6500, ge=1000, le=14000, validation_alias='AGENT_HANDOFF_REPORT_BYTES')
    agent_max_calls_per_response: int = Field(default=4, ge=1, le=16, validation_alias='AGENT_MAX_CALLS_PER_RESPONSE')

    # Storage / business transport
    data_dir: Path = Field(default=PROJECT_ROOT / 'data', validation_alias='DATA_DIR')
    memory_db_path: Path = Field(default_factory=lambda values: values['data_dir'] / 'memory.sqlite3',
                                 validation_alias='MEMORY_DB_PATH')
    commerce_db_path: Path = Field(default_factory=lambda values: values['data_dir'] / 'commerce.sqlite3',
                                   validation_alias='COMMERCE_DB_PATH')
    knowledge_db_path: Path = Field(default_factory=lambda values: values['data_dir'] / 'knowledge.sqlite3',
                                    validation_alias='KNOWLEDGE_DB_PATH')
    demo_user_id: str = Field(default='U00001', min_length=1, validation_alias='DEMO_USER_ID')
    commerce_tool_transport: Literal['local', 'mcp'] = Field(default='local', validation_alias='COMMERCE_TOOL_TRANSPORT')
    catalog_evidence_enabled: bool = Field(default=True, validation_alias='CATALOG_EVIDENCE_ENABLED')

    # Retrieval
    rag_backend: Literal['fts', 'chroma'] = Field(default='fts', validation_alias='RAG_BACKEND')
    chroma_path: Path = Field(default_factory=lambda values: values['data_dir'] / 'chroma', validation_alias='CHROMA_PATH')
    embedding_base_url: str = Field(default='', validation_alias='EMBEDDING_BASE_URL')
    embedding_model: str = Field(default='', validation_alias='EMBEDDING_MODEL')
    embedding_api_key: SecretStr = Field(default='local-dev-key', validation_alias='EMBEDDING_API_KEY')

    # Jev/Kev routing and scope classification
    jev_mode: Literal['off', 'shadow', 'advisory'] = Field(default='off', validation_alias='JEV_MODE')
    jev_scope_enabled: bool = Field(default=False, validation_alias='JEV_SCOPE_ENABLED')
    jev_base_url: str = Field(default='https://api.typesafe.ai/v1', validation_alias='JEV_BASE_URL')
    jev_api_key: SecretStr = Field(default='', validation_alias='JEV_API_KEY')
    jev_model: str = Field(default='jev-latest', min_length=1, validation_alias='JEV_MODEL')
    jev_timeout_seconds: float = Field(default=3, gt=0, validation_alias='JEV_TIMEOUT_SECONDS')

    @field_validator('data_dir', 'memory_db_path', 'commerce_db_path', 'knowledge_db_path', 'chroma_path', mode='before')
    @classmethod
    def normalize_storage_path(cls, value: Path | str) -> Path:
        if not str(value).strip():
            raise ValueError('存储路径不能为空')
        path = Path(value).expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @field_validator('model_base_url', 'embedding_base_url', 'jev_base_url', mode='before')
    @classmethod
    def normalize_url(cls, value: str) -> str:
        value = value.strip().rstrip('/')
        if value:
            parsed = urlsplit(value)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError('服务地址必须包含 http:// 或 https:// 及主机名')
            parsed.port  # Reject malformed/out-of-range port numbers during configuration.
        return value

    @field_validator('model_base_url', 'jev_base_url')
    @classmethod
    def required_url(cls, value: str) -> str:
        if not value:
            raise ValueError('服务地址不能为空')
        return value

    @field_validator('model_name', 'jev_model', 'demo_user_id', 'embedding_model', mode='before')
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode='after')
    def validate_retrieval(self):
        if self.rag_backend == 'chroma' and not (self.embedding_base_url and self.embedding_model):
            raise ValueError('Chroma 模式需要 EMBEDDING_BASE_URL 和 EMBEDDING_MODEL')
        return self

    @property
    def agent(self) -> AgentConfig:
        return AgentConfig(**{name: getattr(self, 'agent_' + name) for name in AgentConfig.model_fields})

    @classmethod
    def from_env(cls, env_file: Path | str | None = PROJECT_ROOT / '.env') -> 'Settings':
        return cls(_env_file=env_file)


_settings_override: ContextVar[Settings | None] = ContextVar('settings_override', default=None)


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    return Settings.from_env()


def get_settings() -> Settings:
    return _settings_override.get() or _load_settings()


@contextmanager
def override_settings(settings: Settings | Mapping[str, Any] | None = None, **updates: Any) -> Iterator[Settings]:
    """Validated task-local overrides; accepts field names or environment aliases."""
    base = settings if isinstance(settings, Settings) else get_settings()
    changes = {**(settings if isinstance(settings, Mapping) else {}), **updates}
    values = base.model_dump()
    for key, value in changes.items():
        name = key if key in Settings.model_fields else next((name for name, field in Settings.model_fields.items()
            if key in (field.validation_alias.choices if isinstance(field.validation_alias, AliasChoices)
                       else [field.validation_alias])), None)
        if name is None:
            raise ValueError(f'未知配置项: {key}')
        values[name] = value
    # Rebase inherited defaults when changing DATA_DIR; keep individually configured paths.
    if 'data_dir' in changes or 'DATA_DIR' in changes:
        for name, filename in {'memory_db_path': 'memory.sqlite3', 'commerce_db_path': 'commerce.sqlite3',
                               'knowledge_db_path': 'knowledge.sqlite3', 'chroma_path': 'chroma'}.items():
            alias = Settings.model_fields[name].validation_alias
            if name not in changes and alias not in changes and getattr(base, name) == base.data_dir / filename:
                values.pop(name)
    # Use validation aliases explicitly so OS aliases cannot shadow init values.
    explicit = {(field.validation_alias.choices[0] if isinstance(field.validation_alias, AliasChoices)
                 else field.validation_alias): values[name] for name, field in Settings.model_fields.items() if name in values}
    configured = Settings(_env_file=None, **explicit) if changes else base
    token = _settings_override.set(configured)
    try:
        yield configured
    finally:
        _settings_override.reset(token)


def __getattr__(name: str):
    """Compatibility for scripts importing old constants; app services use get_settings."""
    aliases = {'MODEL_BASE_URL': 'model_base_url', 'MODEL_API_KEY': 'model_api_key', 'MODEL_NAME': 'model_name'}
    field_name = aliases.get(name, name.lower())
    if name.isupper() and field_name in Settings.model_fields:
        value = getattr(get_settings(), field_name)
        return value.get_secret_value() if isinstance(value, SecretStr) else value
    raise AttributeError(name)
