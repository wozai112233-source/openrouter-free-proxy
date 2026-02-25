"""Configuration helpers for the OpenRouter proxy service."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime configuration loaded from environment variables."""

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    preferred_models: List[str] = field(default_factory=list)
    refresh_interval_seconds: int = 300
    refresh_after_requests: int = 250
    model_quota_cooldown_seconds: int = 300
    model_failure_cooldown_seconds: int = 120
    max_failures_before_cooldown: int = 2
    request_timeout_seconds: float = 60.0
    model_discovery_limit: int = 50
    referer: str | None = None
    app_title: str = "OpenRouter Free Proxy"
    memory_enabled: bool = True
    memory_token_margin: int = 512
    memory_max_messages: int = 200
    memory_max_conversations: int = 1000
    memory_ttl_seconds: int = 60 * 60 * 24
    default_context_window: int = 8000
    log_file_path: str | None = "proxy.log"
    log_level: str = "INFO"
    free_models_url: str | None = "https://openrouter.ai/api/frontend/models/find?max_price=0&order=most-popular"
    refresh_on_demand: bool = False
    model_cache_path: str | None = "model_cache.json"
    reset_model_cache_on_start: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load configuration from environment variables once per process."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for the proxy to work")

    return Settings(
        openrouter_api_key=api_key,
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        preferred_models=_parse_csv(os.getenv("FREE_PREFERRED_MODELS")),
        refresh_interval_seconds=int(os.getenv("MODEL_REFRESH_INTERVAL", "300")),
        refresh_after_requests=int(os.getenv("MODEL_REFRESH_REQUEST_THRESHOLD", "250")),
        model_quota_cooldown_seconds=int(os.getenv("MODEL_QUOTA_COOLDOWN_SECONDS", "300")),
        model_failure_cooldown_seconds=int(os.getenv("MODEL_FAILURE_COOLDOWN_SECONDS", "120")),
        max_failures_before_cooldown=int(os.getenv("MODEL_FAILURE_THRESHOLD", "2")),
        request_timeout_seconds=float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "60")),
        model_discovery_limit=int(os.getenv("MODEL_DISCOVERY_LIMIT", "50")),
        referer=os.getenv("PROXY_REFERER"),
        app_title=os.getenv("PROXY_TITLE", "OpenRouter Free Proxy"),
        memory_enabled=_parse_bool(os.getenv("PROXY_MEMORY_ENABLED"), True),
        memory_token_margin=int(os.getenv("PROXY_MEMORY_TOKEN_MARGIN", "512")),
        memory_max_messages=int(os.getenv("PROXY_MEMORY_MAX_MESSAGES", "200")),
        memory_max_conversations=int(os.getenv("PROXY_MEMORY_MAX_CONVERSATIONS", "1000")),
        memory_ttl_seconds=int(os.getenv("PROXY_MEMORY_TTL_SECONDS", str(60 * 60 * 24))),
        default_context_window=int(os.getenv("PROXY_DEFAULT_CONTEXT_TOKENS", "8000")),
        log_file_path=os.getenv("PROXY_LOG_PATH", "proxy.log"),
        log_level=os.getenv("PROXY_LOG_LEVEL", "INFO"),
        free_models_url=os.getenv(
            "OPENROUTER_FREE_MODELS_URL",
            "https://openrouter.ai/api/frontend/models/find?max_price=0&order=most-popular",
        ),
        refresh_on_demand=_parse_bool(os.getenv("MODEL_REFRESH_ON_DEMAND"), False),
        model_cache_path=os.getenv("MODEL_CACHE_PATH", "model_cache.json"),
        reset_model_cache_on_start=_parse_bool(os.getenv("MODEL_CACHE_RESET_ON_START"), False),
    )
