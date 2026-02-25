# Repository Guidelines

## Project Structure & Module Organization
- `openrouterproxy/` hosts the FastAPI service: `app.py` (HTTP entry), `model_pool.py`, `memory.py`, `openrouter.py`, `config.py`, and `README.md`.
- `requirements.txt` pins runtime deps; create a virtualenv at repo root (`python -m venv .venv`).
- No tests directory yet; add new suites under `openrouterproxy/tests/` when introducing automated coverage.
- Assets or docs live alongside the module that owns them (e.g., `README.md` near the service code).

## Build, Test, and Development Commands
- `pip install -r openrouterproxy/requirements.txt` — install FastAPI, httpx, tiktoken, etc.
- `uvicorn app:app --reload --host 0.0.0.0 --port 8080` (run inside `openrouterproxy/`) — launches the proxy with auto-reload.
- `PYTHONPYCACHEPREFIX=./.pycache OPENROUTER_API_KEY=dummy python3 -m compileall .` — quick syntax/bytecode smoke test used in CI until richer tests exist.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; keep functions small and descriptive.
- Prefer dataclasses for structured config/state; use snake_case for functions/variables, PascalCase for classes.
- Keep logs at INFO for request/rotation events and WARN for quota/failure; ensure `logger` names match module paths.
- Environment config keys are uppercase snake case (e.g., `PROXY_MEMORY_ENABLED`).

## Testing Guidelines
- Target async unit tests with `pytest` + `pytest-asyncio` when adding coverage; mirror module paths (e.g., `tests/test_model_pool.py`).
- Name tests `test_<behavior>` and ensure they exercise quota fallback, memory trimming, and OpenRouter client error paths.
- Run `python3 -m compileall .` or future `pytest` suite locally before pushing.

## Commit & Pull Request Guidelines
- Use imperative, concise commit subjects (`feat: add memory trimming logs`, `fix(proxy): handle empty messages`).
- Describe rationale and testing in commit bodies or PR descriptions; call out impacted endpoints (`/free`, `/healthz`).
- PRs should link tracking issues, list env vars/config changes, and include screenshots or curl transcripts when altering HTTP surfaces.

## Security & Configuration Tips
- Never commit real `OPENROUTER_API_KEY`; rely on `.env` (gitignored) or CI secrets.
- Default logging includes conversation IDs—mask or rotate IDs if piping logs to shared sinks.
- Validate new endpoints through `/healthz` to confirm pool state before exposing them publicly.
