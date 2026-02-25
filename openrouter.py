"""Thin async client for interacting with OpenRouter."""
from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    """Base exception for OpenRouter failures."""

    def __init__(self, message: str, status_code: int | None = None, payload: Any | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class ModelQuotaError(OpenRouterError):
    """Raised when a free model has exhausted its allocation for the period."""


class UpstreamError(OpenRouterError):
    """Raised for other upstream failures."""


@dataclass
class ModelMetadata:
    """Standardised representation of a model entry from OpenRouter."""

    name: str
    display_name: str | None = None
    context_length: int | None = None
    pricing: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


class OpenRouterClient:
    """Simple async wrapper over the OpenRouter REST API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        *,
        timeout: float = 60.0,
        referer: str | None = None,
        app_title: str | None = None,
        free_models_url: str | None = None,
    ) -> None:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if referer:
            headers["HTTP-Referer"] = referer
        if app_title:
            headers["X-Title"] = app_title

        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=headers)
        self._closed = False
        self._lock = asyncio.Lock()
        self._free_models_url = free_models_url

    async def aclose(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    async def fetch_free_models(self, limit: int | None = None) -> List[ModelMetadata]:
        """Return OpenRouter models that are flagged as free."""

        url = self._free_models_url or "/models"
        logger.info("openrouter.fetch_models url=%s limit=%s", url, limit)
        response = await self._client.get(url)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - only triggered by upstream failures
            raise UpstreamError("Unable to fetch models", status_code=exc.response.status_code) from exc

        payload = response.json()
        logger.info(
            "openrouter.models_payload url=%s size=%s",
            url,
            len(_dump(payload)),
        )
        models: List[ModelMetadata] = []
        candidates = _extract_model_payloads(payload)
        for raw in candidates:
            metadata = _build_model_metadata(raw)
            if not metadata:
                continue
            if self._free_models_url is None and not _is_free_model(metadata.raw):
                continue
            models.append(metadata)
            if limit and len(models) >= limit:
                break
        logger.info(
            "openrouter.models_loaded count=%s models=%s",
            len(models),
            [m.name for m in models],
        )
        return models

    async def proxy_chat_completion(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Forward a chat completion request to OpenRouter for the selected model."""

        enriched_payload = {**payload, "model": model}
        logger.info(
            "openrouter.chat_request model=%s message_count=%s payload_length=%s",
            model,
            len(enriched_payload.get("messages") or []),
            _payload_length(enriched_payload),
        )
        response = await self._client.post("/chat/completions", json=enriched_payload)
        return await self._handle_response(model, response)

    async def open_stream_chat_completion(self, model: str, payload: Dict[str, Any]) -> httpx.Response:
        enriched_payload = {**payload, "model": model}
        logger.info(
            "openrouter.chat_request model=%s message_count=%s payload_length=%s",
            model,
            len(enriched_payload.get("messages") or []),
            _payload_length(enriched_payload),
        )
        request = self._client.build_request("POST", "/chat/completions", json=enriched_payload)
        response = await self._client.send(request, stream=True)
        logger.info(
            "openrouter.stream.start model=%s status=%s headers=%s",
            model,
            response.status_code,
            dict(response.headers),
        )
        if response.status_code in {402, 429}:
            body = await response.aread()
            await response.aclose()
            raise ModelQuotaError(
                f"Model {model} reported exhausted quota.",
                status_code=response.status_code,
                payload=body.decode(errors="ignore"),
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = await response.aread()
            await response.aclose()
            raise UpstreamError(
                f"OpenRouter request failed for model {model}", status_code=response.status_code, payload=body.decode(errors="ignore")
            ) from exc
        return response

    async def _handle_response(self, model: str, response: httpx.Response) -> Dict[str, Any]:
        logger.info(
            "openrouter.upstream_response model=%s status=%s headers=%s body_length=%s",
            model,
            response.status_code,
            dict(response.headers),
            len(response.text),
        )
        if response.status_code in {402, 429}:
            raise ModelQuotaError(
                f"Model {model} reported exhausted quota.", status_code=response.status_code, payload=response.text
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise UpstreamError(
                f"OpenRouter request failed for model {model}", status_code=exc.response.status_code, payload=exc.response.text
            ) from exc
        logger.info("openrouter.chat_success model=%s", model)
        return response.json()

    async def _consume_stream_to_full_response(self, model: str, response: httpx.Response) -> Dict[str, Any]:
        text_chunks: List[str] = []
        try:
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("openrouter.stream.invalid_chunk model=%s chunk=%s", model, payload)
                    continue
                text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if text:
                    text_chunks.append(text)
                logger.debug("openrouter.stream.chunk model=%s chunk=%s", model, payload)
        except Exception as exc:
            logger.exception("openrouter.stream.error model=%s", model)
            raise UpstreamError("Streaming response interrupted", payload=str(exc)) from exc
        finally:
            await response.aclose()

        full_text = "".join(text_chunks)
        logger.info("openrouter.stream.completed model=%s text_length=%s", model, len(full_text))
        return {
            "id": "streaming-response",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": full_text},
                }
            ],
        }

def _is_free_model(model_payload: Dict[str, Any]) -> bool:
    pricing = model_payload.get("pricing") or {}
    prompt_cost = _extract_cost(pricing.get("prompt") or pricing.get("input"))
    completion_cost = _extract_cost(pricing.get("completion") or pricing.get("output"))
    if prompt_cost is None and completion_cost is None:
        return False
    return (prompt_cost or 0) == 0 and (completion_cost or 0) == 0


def _extract_cost(value: Any | None) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if "amount" in value:
            return float(value["amount"])
        if "price" in value:
            return float(value["price"])
    return None


def _extract_model_payloads(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "models"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = value.get("models")
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
    return []


def _build_model_metadata(raw: Dict[str, Any]) -> Optional[ModelMetadata]:
    candidate = raw
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        candidate = raw["model"]
    model_id = (
        candidate.get("id")
        or candidate.get("model")
        or candidate.get("slug")
        or candidate.get("name")
    )
    if not model_id:
        return None
    display_name = candidate.get("name") or candidate.get("display_name")
    context_length = _safe_int(
        candidate.get("context_length")
        or candidate.get("contextLength")
        or candidate.get("max_context")
    )
    pricing = candidate.get("pricing")
    if not isinstance(pricing, dict):
        pricing = {}
    return ModelMetadata(
        name=model_id,
        display_name=display_name,
        context_length=context_length,
        pricing=pricing,
        raw=raw,
    )


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dump(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive logging
        return repr(data)


def _payload_length(data: Any) -> int:
    try:
        return len(json.dumps(data, ensure_ascii=False, default=str))
    except Exception:  # pragma: no cover - defensive logging
        return 0


    async def _handle_response(self, model: str, response: httpx.Response) -> Dict[str, Any]:
        logger.info(
            "openrouter.upstream_response model=%s status=%s headers=%s body=%s",
            model,
            response.status_code,
            dict(response.headers),
            response.text,
        )
        if response.status_code in {402, 429}:
            raise ModelQuotaError(
                f"Model {model} reported exhausted quota.", status_code=response.status_code, payload=response.text
            )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise UpstreamError(
                f"OpenRouter request failed for model {model}", status_code=exc.response.status_code, payload=exc.response.text
            ) from exc
        logger.info("openrouter.chat_success model=%s", model)
        return response.json()

    async def _consume_stream(self, model: str, response: httpx.Response) -> Dict[str, Any]:
        logger.info(
            "openrouter.stream.start model=%s status=%s headers=%s",
            model,
            response.status_code,
            dict(response.headers),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise UpstreamError(
                f"OpenRouter request failed for model {model}", status_code=exc.response.status_code, payload=exc.response.text
            ) from exc

        collected: List[str] = []
        try:
            async for line in response.aiter_lines():
                if not line:
                    continue
                collected.append(line)
                logger.debug("openrouter.stream.chunk model=%s line=%s", model, line)
        except Exception as exc:
            logger.exception("openrouter.stream.error model=%s", model)
            raise UpstreamError("Streaming response interrupted", payload=str(exc)) from exc

        text_chunks: List[str] = []
        for raw_line in collected:
            if not raw_line.startswith("data:"):
                continue
            content = raw_line[len("data:") :].strip()
            if content == "[DONE]":
                continue
            try:
                chunk = json.loads(content)
                text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if text:
                    text_chunks.append(text)
            except json.JSONDecodeError:
                logger.warning("openrouter.stream.invalid_chunk model=%s chunk=%s", model, content)
                continue

        full_text = "".join(text_chunks)
        logger.info("openrouter.stream.completed model=%s text=%s", model, full_text)
        return {
            "id": "streaming-response",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": full_text,
                    },
                }
            ],
        }
