# Forward Simulation — Python FastAPI Port

This folder contains a Python FastAPI application that mirrors the Next.js + API functionality from the previous project. It provides:

- A `/` page serving a minimal chat UI.
- A `/api/chat` POST endpoint that forwards messages to an LLM (OpenRouter) and supports tool calls including DB queries and MCP forwarding.

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

2. Set environment variables (example):

```bash
export OPENROUTER_API_KEY="your_key"
export MCP_ENDPOINT="http://localhost:8080/mcp"
export DATABASE_URL="postgres://user:pass@host:5432/dbname"
```

3. Run the server:

```bash
uvicorn server.main:app --reload --host 0.0.0.0 --port 8080
```

Notes
- The DB functionality requires `DATABASE_URL` and expects a `credit_records` table if `fetch_credit_records` is used.
- MCP discovery will attempt several RPC list endpoints. If your MCP wrapper uses a different method name, set `MCP_ENDPOINT` accordingly.

Environment and tests
- Create a local `.env` from the example and populate secrets (do NOT commit credentials):

```bash
cp .env.example .env
# edit .env and fill in OPENROUTER_API_KEY and other vars
```

- The server will load `.env` automatically via `python-dotenv` when `server/main.py` starts.

- Database usage (optional):
	- If you need `fetch_credit_records` to work, set `DATABASE_URL` to a Postgres connection string and install `asyncpg` in your virtualenv.
	- On some macOS/Python combinations building `asyncpg` from source can fail — consider using a Python version with prebuilt wheels (e.g., 3.11 or 3.10) or installing the system build tools (Xcode command-line tools).

Running tests
- A small integration test is provided that checks `fetch_credit_records` when a DB and `asyncpg` are available:

```bash
# (optional) install pytest and pytest-asyncio
pip install pytest pytest-asyncio

# run tests (the DB test will be skipped if DATABASE_URL or asyncpg are not available)
pytest server/tests -q
```

Deployment notes
- When deploying, set `OPENROUTER_API_KEY`, `MCP_ENDPOINT`, and `DATABASE_URL` (if used) in the host environment before starting the process so the app reads them at startup.

MCP endpoint
- This server exposes an internal MCP-compatible JSON-RPC endpoint at `/mcp`.
	- `initialize` returns a minimal capability object.
	- `tools/list` (and several common variants) returns the locally-registered tools.
	- Calling a tool method (for example, `fetch_credit_records`) will invoke the corresponding server function and return its result.
	- If a method is unknown locally and you set the environment variable `MCP_UPSTREAM`, the server will forward the JSON-RPC payload to that upstream MCP and return the upstream response.

- Configuration notes:
	- `MCP_ENDPOINT` is used by the server when it needs to call an MCP wrapper elsewhere (keeps parity with the Next.js behavior). Do not set `MCP_ENDPOINT` to point at this same server unless you intend to call the local `/mcp` endpoint.
	- `MCP_UPSTREAM` (optional) — set this to an external MCP wrapper if you want unknown methods forwarded.

