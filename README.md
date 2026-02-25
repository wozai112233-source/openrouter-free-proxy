# OpenRouter Free Proxy

This service exposes a single `POST /free` endpoint that forwards requests to OpenRouter. Internally it keeps a pool of the latest free-tier models, rotates through them, and automatically cools down entries that have exhausted their quota or started erroring.

## Features
- Discovers free models from the OpenRouter `/models` endpoint and refreshes the list on a timer.
- Maintains a simple round-robin pool so you only expose one public endpoint.
- Automatically falls back to the next model when the active one hits a quota or 429/402 style failure.
- Background refresh can also be triggered after a configurable number of proxy requests.
- Health endpoint (`GET /healthz`) shows which backing models are available and their cooldown state.

## Configuration
Set these environment variables (an `.env` file works too because `python-dotenv` is loaded automatically):

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENROUTER_API_KEY` | ✅ | – | API key issued by OpenRouter. |
| `OPENROUTER_BASE_URL` | ❌ | `https://openrouter.ai/api/v1` | Override if you run through a mirror. |
| `OPENROUTER_FREE_MODELS_URL` | ❌ | `https://openrouter.ai/api/frontend/models/find?max_price=0&order=most-popular` | Endpoint used to discover zero-cost models; set empty to fall back to `/models`. |
| `FREE_PREFERRED_MODELS` | ❌ | – | Comma separated list of model IDs to prioritise (newest free tier, internal allow list, etc.). |
| `MODEL_REFRESH_INTERVAL` | ❌ | `300` | Seconds between periodic refresh jobs. Set `0` to disable the loop. |
| `MODEL_REFRESH_REQUEST_THRESHOLD` | ❌ | `250` | Trigger an additional refresh each time this many proxied requests have been served. |
| `MODEL_REFRESH_ON_DEMAND` | ❌ | `false` | Whether to allow request/failure-triggered refreshes; `false` means only the interval job runs. |
| `MODEL_CACHE_PATH` | ❌ | `model_cache.json` | Local file used to persist model pool state (cooldowns, ordering) across restarts. |
| `MODEL_QUOTA_COOLDOWN_SECONDS` | ❌ | `300` | Cooldown applied when a model returns 402/429. |
| `MODEL_FAILURE_COOLDOWN_SECONDS` | ❌ | `120` | Cooldown applied after repeated upstream failures. |
| `MODEL_FAILURE_THRESHOLD` | ❌ | `2` | Consecutive failures before putting a model into failure cooldown. |
| `UPSTREAM_TIMEOUT_SECONDS` | ❌ | `60` | HTTP timeout when calling OpenRouter. |
| `MODEL_DISCOVERY_LIMIT` | ❌ | `50` | Maximum number of free models to keep in the pool. |
| `PROXY_REFERER` | ❌ | – | Sent as `HTTP-Referer`, recommended by OpenRouter. |
| `PROXY_TITLE` | ❌ | `OpenRouter Free Proxy` | Sent as `X-Title` header. |
| `PROXY_MEMORY_ENABLED` | ❌ | `true` | Toggle the built-in conversation memory layer. |
| `PROXY_MEMORY_TOKEN_MARGIN` | ❌ | `512` | Tokens reserved for responses when trimming history. |
| `PROXY_MEMORY_MAX_MESSAGES` | ❌ | `200` | Maximum stored turns per conversation. |
| `PROXY_MEMORY_MAX_CONVERSATIONS` | ❌ | `1000` | Number of concurrent conversations cached. |
| `PROXY_MEMORY_TTL_SECONDS` | ❌ | `86400` | Idle conversations expire after this many seconds. |
| `PROXY_DEFAULT_CONTEXT_TOKENS` | ❌ | `8000` | Used if a model lacks a declared context window. |
| `PROXY_LOG_PATH` | ❌ | `proxy.log` | Where to persist combined request/response logs (set empty to disable file logging). |
| `PROXY_LOG_LEVEL` | ❌ | `INFO` | Root log level for both console and file handlers. |

## Running locally
```bash
cd openrouterproxy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-...
uvicorn app:app --host 0.0.0.0 --port 8080
```

### Proxy endpoint + server memory
Send requests identical to OpenAI/OpenRouter chat completions. Provide a `conversation_id` field (or `X-Conversation-Id` header) when you want the proxy to maintain, trim, and replay your chat history automatically. For drop-in compatibility, both `POST /free` and `POST /v1/chat/completions` (以及 `/free/chat/completions`) 指向同一个代理逻辑，`stream=true` 时会原样推送 SSE 事件。
```bash
curl -X POST http://localhost:8080/free \
  -H 'Content-Type: application/json' \
  -H 'X-Conversation-Id: demo-user-123' \
  -d '{
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hi!"}
        ],
        "conversation_id": "demo-user-123"
      }'
```
The proxy injects the currently selected free model, rebuilds history to respect that model's context window, and returns the upstream response unchanged.

### Health endpoint
```bash
curl http://localhost:8080/healthz | jq
```
This shows how many models are available plus their cooldown/error status, which is useful for dashboards or alerts.

## Implementation notes
- `model_pool.py` contains the rotation, cooldown, and refresh logic (periodic job + on-demand triggers after thresholds or failures).
- `memory.py` keeps per-conversation history, trims by model context length, and evicts idle sessions.
- `openrouter.py` is a very small async client wrapper over `/models` and `/chat/completions`.
- Errors like quota exhaustion automatically trigger a cooldown and background refresh so the pool keeps moving to the next free model without user intervention.
- The service logs each inbound request, model attempt, OpenRouter call, and memory trim to make debugging quota exhaustion or history pruning straightforward.
- Upstream responses (status, headers, body) and model discovery payloads are logged to the configured file for full transparency; treat `PROXY_LOG_PATH` as sensitive data.
- Request/response bodies are mirrored to the log file defined by `PROXY_LOG_PATH`, so treat that file as sensitive and rotate it regularly.
