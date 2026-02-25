"""FastAPI application exposing a single /free endpoint backed by OpenRouter."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import get_settings
from model_pool import ModelPool, NoModelAvailableError
from memory import ConversationStore, TokenCounter
from openrouter import ModelQuotaError, OpenRouterClient, UpstreamError

settings = get_settings()


def setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.log_file_path:
        handlers.append(logging.FileHandler(settings.log_file_path))
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


setup_logging()
logger = logging.getLogger("openrouterproxy")

client = OpenRouterClient(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    timeout=settings.request_timeout_seconds,
    referer=settings.referer,
    app_title=settings.app_title,
    free_models_url=settings.free_models_url,
)
pool = ModelPool(client=client, settings=settings)
memory = ConversationStore(settings=settings, token_counter=TokenCounter())

app = FastAPI(title=settings.app_title, version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    await pool.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await pool.stop()
    await client.aclose()


@app.post("/free")
async def proxy_free(request: Request) -> JSONResponse:
    return await _proxy_request(request)


@app.post("/free/chat/completions")
@app.post("/v1/chat/completions")
async def proxy_openai_compatible(request: Request) -> JSONResponse:
    return await _proxy_request(request)


async def _proxy_request(request: Request) -> JSONResponse:
    payload: Dict[str, Any] = await request.json()
    conversation_id = (
        payload.get("conversation_id")
        or request.headers.get("X-Conversation-Id")
        or request.headers.get("x-conversation-id")
    )
    is_streaming = bool(payload.get("stream"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        logger.error(
            "proxy.invalid_payload",
            extra={"conversation_id": conversation_id, "payload": payload},
        )
        raise HTTPException(status_code=400, detail="Request body must include a messages list")

    logger.info(
        "proxy.request conversation_id=%s message_count=%s keys=%s",
        conversation_id,
        len(messages),
        list(payload.keys()),
    )
    logger.info(
        "proxy.request_body conversation_id=%s payload_length=%s",
        conversation_id,
        _payload_length(payload),
    )

    if conversation_id and memory.enabled():
        await memory.ingest(conversation_id, messages)

    forward_payload = {k: v for k, v in payload.items() if k != "conversation_id"}
    attempt_errors: list[dict[str, Any]] = []
    max_attempts = max(1, pool.total())

    tried: set[str] = set()
    for attempt in range(max_attempts):
        try:
            model = await pool.next_model()
        except NoModelAvailableError as exc:
            logger.warning("No model available: %s", exc)
            logger.error(
                "proxy.no_model_available",
                extra={
                    "conversation_id": conversation_id,
                    "attempt": attempt + 1,
                    "reason": str(exc),
                },
            )
            break
        if model.name in tried:
            continue
        tried.add(model.name)

        context_limit = model.context_length
        prepared_messages = messages
        if conversation_id and memory.enabled():
            trimmed = await memory.render(conversation_id, model.name, context_limit)
            if trimmed:
                prepared_messages = trimmed
        attempt_payload = dict(forward_payload)
        attempt_payload["messages"] = prepared_messages

        logger.info(
            "proxy.attempt conversation_id=%s model=%s attempt=%s messages=%s context_limit=%s",
            conversation_id,
            model.name,
            attempt + 1,
            len(prepared_messages),
            context_limit,
        )
        logger.info(
        "proxy.attempt_payload conversation_id=%s model=%s payload_length=%s",
        conversation_id,
        model.name,
        _payload_length(attempt_payload),
    )

        try:
            if is_streaming:
                streaming_response = await _proxy_streaming_response(
                    model, attempt_payload, conversation_id, attempt + 1
                )
                return streaming_response
            response = await client.proxy_chat_completion(model.name, attempt_payload)
        except ModelQuotaError as exc:
            logger.info("Model %s quota exhausted", model.name)
            pool.register_quota_exhausted(model.name, str(exc))
            attempt_errors.append({"model": model.name, "reason": "quota", "detail": str(exc)})
            continue
        except UpstreamError as exc:
            logger.warning("Model %s failed: %s", model.name, exc)
            pool.register_failure(model.name, str(exc))
            attempt_errors.append({"model": model.name, "reason": "upstream", "detail": str(exc)})
            continue

        pool.register_success(model.name)
        if conversation_id and memory.enabled():
            assistant_message = _extract_assistant_message(response)
            if assistant_message:
                await memory.append_assistant(conversation_id, assistant_message)
        logger.info(
            "proxy.success conversation_id=%s model=%s attempt=%s",
            conversation_id,
            model.name,
            attempt + 1,
        )
        logger.info(
            "proxy.response_body conversation_id=%s model=%s body_length=%s",
            conversation_id,
            model.name,
            _payload_length(response),
        )
        return JSONResponse(response)

    logger.error(
        "proxy.depleted conversation_id=%s attempts=%s payload_length=%s",
        conversation_id,
        attempt_errors,
        _payload_length(payload),
    )
    raise HTTPException(
        status_code=503,
        detail={
            "message": "No free models are available at the moment.",
            "attempts": attempt_errors,
        },
    )


@app.get("/healthz")
async def health() -> dict[str, Any]:
    return {
        "available_models": pool.available(),
        "total_models": pool.total(),
        "models": pool.snapshot(),
    }


def _extract_assistant_message(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return None
    if message.get("role") != "assistant":
        return None
    return {"role": "assistant", "content": message.get("content")}


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


async def _proxy_streaming_response(
    model: "ModelState",
    payload: Dict[str, Any],
    conversation_id: Optional[str],
    attempt_index: int,
) -> StreamingResponse:
    upstream = await client.open_stream_chat_completion(model.name, payload)

    async def event_stream() -> Any:
        aggregated: List[str] = []
        success = False
        try:
            async for line in upstream.aiter_lines():
                if line.startswith("data:"):
                    text_piece = _extract_stream_text(line)
                    if text_piece:
                        aggregated.append(text_piece)
                yield f"{line}\n"
            success = True
        finally:
            await upstream.aclose()
            if success:
                final_text = "".join(aggregated)
                if conversation_id and memory.enabled() and final_text:
                    await memory.append_assistant(
                        conversation_id,
                        {"role": "assistant", "content": final_text},
                    )
                pool.register_success(model.name)
                logger.info(
                    "proxy.success conversation_id=%s model=%s attempt=%s",
                    conversation_id,
                    model.name,
                    attempt_index,
                )
                logger.info(
                    "proxy.response_body conversation_id=%s model=%s body_length=%s",
                    conversation_id,
                    model.name,
                    _payload_length(
                        {
                            "id": "streaming-response",
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": final_text},
                                }
                            ],
                        }
                    ),
                )
            else:
                logger.warning(
                    "proxy.stream_aborted conversation_id=%s model=%s",
                    conversation_id,
                    model.name,
                )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _extract_stream_text(line: str) -> Optional[str]:
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = chunk.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, list):
        return "".join(str(item) for item in content)
    return content
