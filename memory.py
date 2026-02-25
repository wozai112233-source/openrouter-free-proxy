"""Server-side conversation memory with token-aware trimming."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import Settings

try:  # optional dependency for better token counts
    import tiktoken
except Exception:  # pragma: no cover - soft dependency
    tiktoken = None  # type: ignore[assignment]

Message = Dict[str, Any]
logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TokenCounter:
    """Estimates token usage for chat messages."""

    def __init__(self) -> None:
        self._encoding_cache: Dict[str, Any] = {}

    def count(self, messages: List[Message], model: str | None = None) -> int:
        if not messages:
            return 0
        if tiktoken:
            encoding = self._resolve_encoding(model)
            total = 0
            for message in messages:
                total += self._count_with_encoding(encoding, message)
            # OpenAI chat format overhead heuristic
            return total + 4 * len(messages) + 2
        return sum(self._heuristic_tokens(message) for message in messages)

    def _resolve_encoding(self, model: str | None):
        cache_key = model or "cl100k_base"
        if cache_key in self._encoding_cache:
            return self._encoding_cache[cache_key]
        if model:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        self._encoding_cache[cache_key] = encoding
        return encoding

    def _count_with_encoding(self, encoding, message: Message) -> int:
        content = _message_content_as_text(message)
        tokens = encoding.encode(content or "")
        return len(tokens)

    def _heuristic_tokens(self, message: Message) -> int:
        content = _message_content_as_text(message)
        if not content:
            return 4
        # crude: assume 4 characters per token + fixed overhead
        return max(4, int(len(content) / 4) + 4)


def _message_content_as_text(message: Message) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return " ".join(f"{key}: {value}" for key, value in content.items())
    return str(content)


@dataclass
class ConversationState:
    messages: List[Message] = field(default_factory=list)
    updated_at: datetime = field(default_factory=_utcnow)


class ConversationStore:
    """Keeps per-conversation history and trims based on model limits."""

    def __init__(self, settings: Settings, token_counter: TokenCounter | None = None) -> None:
        self._settings = settings
        self._token_counter = token_counter or TokenCounter()
        self._store: Dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    def enabled(self) -> bool:
        return self._settings.memory_enabled

    async def ingest(self, conversation_id: str, new_messages: List[Message]) -> None:
        if not self.enabled() or not conversation_id or not new_messages:
            return
        normalized = [_normalize_message(msg) for msg in new_messages]
        async with self._lock:
            state = self._store.get(conversation_id)
            if state is None:
                state = ConversationState(messages=list(normalized))
                self._store[conversation_id] = state
                logger.info(
                    "memory.new_conversation",
                    extra={
                        "conversation_id": conversation_id,
                        "messages": len(normalized),
                    },
                )
            else:
                delta = _diff_messages(state.messages, normalized)
                if delta:
                    state.messages.extend(delta)
                    logger.debug(
                        "memory.extend",
                        extra={
                            "conversation_id": conversation_id,
                            "delta_messages": len(delta),
                            "total_messages": len(state.messages),
                        },
                    )
            state.updated_at = _utcnow()
            self._prune_conversation(state)
            self._evict_if_needed()

    async def append_assistant(self, conversation_id: str, message: Message) -> None:
        if not self.enabled() or not conversation_id:
            return
        async with self._lock:
            state = self._store.get(conversation_id)
            if not state:
                return
            state.messages.append(_normalize_message(message))
            state.updated_at = _utcnow()
            self._prune_conversation(state)
            logger.debug(
                "memory.append_assistant",
                extra={
                    "conversation_id": conversation_id,
                    "total_messages": len(state.messages),
                },
            )

    async def render(self, conversation_id: str, model_name: str | None, context_limit: int | None) -> Optional[List[Message]]:
        if not self.enabled() or not conversation_id:
            return None
        async with self._lock:
            state = self._store.get(conversation_id)
            if not state:
                return None
            snapshot = [dict(message) for message in state.messages]
        trimmed = self._trim_to_context(snapshot, model_name, context_limit, conversation_id)
        logger.info(
            "memory.render",
            extra={
                "conversation_id": conversation_id,
                "model": model_name,
                "messages": len(snapshot),
                "trimmed_messages": len(trimmed),
            },
        )
        return trimmed

    def _trim_to_context(
        self,
        messages: List[Message],
        model_name: str | None,
        context_limit: int | None,
        conversation_id: str | None,
    ) -> List[Message]:
        limit = context_limit or self._settings.default_context_window
        limit = max(0, limit - self._settings.memory_token_margin)
        if limit <= 0:
            return messages
        total = self._token_counter.count(messages, model_name)
        initial_total = total
        initial_messages = len(messages)
        if total <= limit:
            return messages
        trimmed = list(messages)
        while len(trimmed) > 1 and total > limit:
            index = _next_message_to_drop(trimmed)
            if index is None:
                break
            trimmed.pop(index)
            total = self._token_counter.count(trimmed, model_name)
        final_messages = len(trimmed)
        if initial_total > limit:
            logger.info(
                "memory.trim",
                extra={
                    "conversation_id": conversation_id,
                    "model": model_name,
                    "initial_tokens": initial_total,
                    "final_tokens": total,
                    "limit": limit,
                    "initial_messages": initial_messages,
                    "final_messages": final_messages,
                },
            )
        if total > limit:
            logger.warning(
                "memory.trim_incomplete",
                extra={
                    "conversation_id": conversation_id,
                    "model": model_name,
                    "limit": limit,
                    "tokens": total,
                    "messages": final_messages,
                },
            )
        return trimmed

    def _prune_conversation(self, state: ConversationState) -> None:
        max_messages = self._settings.memory_max_messages
        if max_messages <= 0:
            return
        while len(state.messages) > max_messages:
            drop_index = _oldest_droppable_index(state.messages)
            if drop_index is None:
                state.messages.pop(0)
            else:
                state.messages.pop(drop_index)

    def _evict_if_needed(self) -> None:
        max_conv = self._settings.memory_max_conversations
        if max_conv > 0:
            while len(self._store) > max_conv:
                oldest_id = min(self._store, key=lambda cid: self._store[cid].updated_at)
                self._store.pop(oldest_id, None)
                logger.info(
                    "memory.evict_capacity",
                    extra={
                        "conversation_id": oldest_id,
                        "remaining": len(self._store),
                    },
                )
        ttl = self._settings.memory_ttl_seconds
        if ttl > 0:
            expires_before = _utcnow() - timedelta(seconds=ttl)
            to_delete = [cid for cid, state in self._store.items() if state.updated_at < expires_before]
            for cid in to_delete:
                self._store.pop(cid, None)
                logger.info(
                    "memory.evict_ttl",
                    extra={
                        "conversation_id": cid,
                        "remaining": len(self._store),
                    },
                )


def _normalize_message(message: Message) -> Message:
    role = message.get("role", "user")
    content = message.get("content", "")
    return {"role": role, "content": content}


def _diff_messages(existing: List[Message], incoming: List[Message]) -> List[Message]:
    if not existing:
        return incoming
    prefix_len = 0
    while (
        prefix_len < len(existing)
        and prefix_len < len(incoming)
        and existing[prefix_len] == incoming[prefix_len]
    ):
        prefix_len += 1
    if prefix_len == len(incoming):
        return []
    if prefix_len == len(existing):
        return incoming[prefix_len:]
    # ambiguous situation; append full incoming to avoid losing context
    return incoming


def _next_message_to_drop(messages: List[Message]) -> Optional[int]:
    if len(messages) <= 2:
        return None
    # skip system prompt if it's first
    start = 1 if messages and messages[0].get("role") == "system" else 0
    candidate_range = list(range(start, len(messages) - 1))
    for index in candidate_range:
        if messages[index].get("role") != "system":
            return index
    return candidate_range[0] if candidate_range else None


def _oldest_droppable_index(messages: List[Message]) -> Optional[int]:
    if not messages:
        return None
    for index, message in enumerate(messages):
        if message.get("role") == "system" and index == 0:
            continue
        return index
    return None
