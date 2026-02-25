"""Model pool with rotation, refresh, and cooldown logic."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import Settings
from openrouter import ModelMetadata, OpenRouterClient

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NoModelAvailableError(RuntimeError):
    """Raised when no backing models are available for a request."""


@dataclass
class ModelState:
    name: str
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_length: Optional[int] = None
    cooldown_until: Optional[datetime] = None
    fail_count: int = 0
    last_error: Optional[str] = None
    last_request_at: Optional[datetime] = None
    refreshed_at: datetime = field(default_factory=_utcnow)

    def is_available(self, now: datetime) -> bool:
        return self.cooldown_until is None or now >= self.cooldown_until


class ModelPool:
    """Manages model discovery, rotation, and cooldowns for the proxy."""

    def __init__(self, client: OpenRouterClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._states: Dict[str, ModelState] = {}
        self._order: List[str] = []
        self._lock = asyncio.Lock()
        self._refresh_lock = asyncio.Lock()
        self._rotation_index = 0
        self._total_requests = 0
        self._running = False
        self._maintainer: Optional[asyncio.Task[None]] = None
        self._on_demand_refresh: Optional[asyncio.Task[None]] = None
        self._cache_path = (
            Path(self._settings.model_cache_path).expanduser() if self._settings.model_cache_path else None
        )
        if self._cache_path and self._settings.reset_model_cache_on_start:
            with contextlib.suppress(FileNotFoundError):
                self._cache_path.unlink()
            logger.info("model_cache.reset", extra={"path": str(self._cache_path)})
        self._blacklist: set[str] = set()
        self._load_from_cache()

    async def start(self) -> None:
        await self.refresh("startup")
        if self._settings.refresh_interval_seconds > 0:
            self._running = True
            self._maintainer = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        self._running = False
        if self._maintainer:
            self._maintainer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._maintainer
            self._maintainer = None
        if self._on_demand_refresh:
            self._on_demand_refresh.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._on_demand_refresh
            self._on_demand_refresh = None

    async def _refresh_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._settings.refresh_interval_seconds)
                try:
                    await self.refresh("interval")
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to refresh model pool in background loop")
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            pass

    async def refresh(self, reason: str = "manual") -> None:
        """Update the underlying model list from OpenRouter."""

        async with self._refresh_lock:
            logger.info("Refreshing model pool: %s", reason)
            models = await self._client.fetch_free_models(limit=self._settings.model_discovery_limit)
            if not models:
                logger.warning("Model discovery returned zero free entries; keeping existing pool")
                return

            timestamp = _utcnow()
            new_states: Dict[str, ModelState] = {}
            ordered = _order_models(models, self._settings.preferred_models)
            for priority, meta in enumerate(ordered):
                if not meta.name:
                    continue
                if meta.name in self._blacklist:
                    logger.info("Skipping blacklisted model %s", meta.name)
                    continue
                previous = self._states.get(meta.name)
                state = ModelState(
                    name=meta.name,
                    priority=priority,
                    metadata=meta.raw,
                    context_length=meta.context_length,
                    refreshed_at=timestamp,
                )
                if previous and previous.cooldown_until and previous.cooldown_until > timestamp:
                    state.cooldown_until = previous.cooldown_until
                    state.last_error = previous.last_error
                new_states[meta.name] = state

        if not new_states:
            logger.warning("No valid models produced after refresh; keeping previous snapshot")
            return

        self._states = new_states
        self._order = [meta.name for meta in ordered if meta.name in new_states]
        self._rotation_index = 0
        logger.info(
            "pool.loaded",
            extra={
                "count": len(self._states),
                "models": self._order,
                "reason": reason,
            },
        )
        self._persist_cache()

    def available(self) -> int:
        now = _utcnow()
        return sum(1 for name in self._order if self._states[name].is_available(now))

    def total(self) -> int:
        return len(self._states)

    async def next_model(self) -> ModelState:
        """Select the next available model according to rotation order."""

        async with self._lock:
            now = _utcnow()
            available: List[ModelState] = [
                self._states[name]
                for name in self._order
                if name in self._states and self._states[name].is_available(now)
            ]
            if not available:
                raise NoModelAvailableError("No models currently available; pool exhausted or cooling down")

            model = available[self._rotation_index % len(available)]
            self._rotation_index = (self._rotation_index + 1) % len(available)
            model.last_request_at = now
            self._total_requests += 1
            if (
                self._settings.refresh_on_demand
                and self._settings.refresh_after_requests
                and self._total_requests % self._settings.refresh_after_requests == 0
            ):
                self.schedule_refresh("request-threshold")
            return model

    def register_success(self, model_name: str) -> None:
        state = self._states.get(model_name)
        if not state:
            return
        state.fail_count = 0
        state.cooldown_until = None
        state.last_error = None
        self._persist_cache()
        logger.info("pool.success", extra={"model": model_name})

    def register_quota_exhausted(self, model_name: str, reason: str | None = None) -> None:
        state = self._states.get(model_name)
        if not state:
            return
        state.cooldown_until = _utcnow() + timedelta(seconds=self._settings.model_quota_cooldown_seconds)
        state.last_error = reason or "quota exhausted"
        state.fail_count = 0
        self._blacklist_model(model_name, state.last_error)
        if self._settings.refresh_on_demand:
            self.schedule_refresh("quota-exhausted")
        self._persist_cache()
        logger.warning(
            "pool.quota_exhausted",
            extra={
                "model": model_name,
                "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
                "reason": reason,
            },
        )

    def register_failure(self, model_name: str, reason: str | None = None) -> None:
        state = self._states.get(model_name)
        if not state:
            return
        state.fail_count += 1
        state.last_error = reason
        if state.fail_count >= self._settings.max_failures_before_cooldown:
            state.cooldown_until = _utcnow() + timedelta(
                seconds=self._settings.model_failure_cooldown_seconds
            )
            state.fail_count = 0
            logger.warning(
                "pool.cooldown.failure",
                extra={
                    "model": model_name,
                    "reason": reason,
                    "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
                },
            )
            if reason and "404" in reason:
                self._blacklist_model(model_name, reason)
        if self._settings.refresh_on_demand:
            self.schedule_refresh("failure")
        self._persist_cache()

    def schedule_refresh(self, reason: str) -> None:
        if not self._settings.refresh_on_demand:
            return
        if self._on_demand_refresh and not self._on_demand_refresh.done():
            return

        async def _runner() -> None:
            try:
                await self.refresh(reason)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("On-demand refresh failed")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - invoked outside of event loop (sync context)
            return
        self._on_demand_refresh = loop.create_task(_runner())

    def snapshot(self) -> List[Dict[str, Any]]:
        now = _utcnow()
        return [
            {
                "name": state.name,
                "priority": state.priority,
                "available": state.is_available(now),
                "context_length": state.context_length,
                "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
                "last_error": state.last_error,
                "last_request_at": state.last_request_at.isoformat() if state.last_request_at else None,
            }
            for state in (self._states[name] for name in self._order if name in self._states)
        ]

    def clear_blacklist(self) -> None:
        if not self._blacklist:
            return
        self._blacklist.clear()
        logger.info("pool.blacklist_cleared")
        self._persist_cache()

    def _load_from_cache(self) -> None:
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            raw = json.loads(self._cache_path.read_text())
        except Exception:
            logger.warning("Failed to load model cache", exc_info=True)
            return
        models = raw.get("models")
        if not isinstance(models, list):
            return
        cached_states: Dict[str, ModelState] = {}
        for entry in models:
            name = entry.get("name")
            if not name:
                continue
            state = ModelState(
                name=name,
                priority=entry.get("priority", 0),
                metadata=entry.get("metadata") or {},
                context_length=entry.get("context_length"),
                refreshed_at=_parse_datetime(entry.get("refreshed_at")) or _utcnow(),
            )
            state.cooldown_until = _parse_datetime(entry.get("cooldown_until"))
            state.last_request_at = _parse_datetime(entry.get("last_request_at"))
            state.fail_count = entry.get("fail_count", 0)
            state.last_error = entry.get("last_error")
            cached_states[name] = state
        order = raw.get("order")
        if not isinstance(order, list):
            order = list(cached_states.keys())
        self._states = cached_states
        self._order = [name for name in order if name in cached_states]
        self._rotation_index = 0
        blacklist = raw.get("blacklist")
        if isinstance(blacklist, list):
            self._blacklist = {item for item in blacklist if isinstance(item, str)}
        logger.info("Loaded %d models from cache", len(self._states))

    def _persist_cache(self) -> None:
        if not self._cache_path:
            return
        try:
            if self._cache_path.parent:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "order": self._order,
                "models": [
                    {
                        "name": state.name,
                        "priority": state.priority,
                        "metadata": state.metadata,
                        "context_length": state.context_length,
                        "cooldown_until": state.cooldown_until.isoformat() if state.cooldown_until else None,
                        "fail_count": state.fail_count,
                        "last_error": state.last_error,
                        "last_request_at": state.last_request_at.isoformat() if state.last_request_at else None,
                        "refreshed_at": state.refreshed_at.isoformat(),
                    }
                    for state in self._states.values()
                ],
                "blacklist": sorted(self._blacklist),
            }
            tmp_path = self._cache_path.with_name(self._cache_path.name + ".tmp")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False))
            tmp_path.replace(self._cache_path)
        except Exception:
            logger.warning("Failed to persist model cache", exc_info=True)

    def _blacklist_model(self, model_name: str, reason: str | None = None) -> None:
        if model_name in self._blacklist:
            return
        self._blacklist.add(model_name)
        logger.warning(
            "pool.blacklist",
            extra={"model": model_name, "reason": reason},
        )


def _order_models(models: List[ModelMetadata], preferred: List[str]) -> List[ModelMetadata]:
    lookup = {model.name: model for model in models if model.name}
    ordered: List[ModelMetadata] = []
    seen: set[str] = set()
    for name in preferred:
        model = lookup.get(name)
        if model:
            ordered.append(model)
            seen.add(name)
    for model in models:
        if not model.name or model.name in seen:
            continue
        ordered.append(model)
        seen.add(model.name)
    return ordered


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
