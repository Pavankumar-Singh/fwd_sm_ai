# server/main.py
import os
import re
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# FastAPI + templates/static
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Pydantic v2
from pydantic import BaseModel, Field, model_validator, ValidationError

# HTTP client
import httpx

# optional libs
try:
    import asyncpg
    HAS_ASYNCPG = True
except Exception:
    asyncpg = None
    HAS_ASYNCPG = False

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except Exception:
    jsonschema = None
    HAS_JSONSCHEMA = False

# -------------------------
# Paths + env
# -------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_URL = os.getenv("OPENROUTER_CHAT_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-r1")
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "120000"))  # ms

# GitHub MCP Server (SSE endpoint)
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT", "http://localhost:8080/mcp")
GITHUB_PAT = os.getenv("GITHUB_PAT")  # Required for GitHub API access
MCP_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "60000"))

DATABASE_URL = os.getenv("DATABASE_URL")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("mcp-agent")

logger.info("ENV: OPENROUTER_API_KEY set=%s MODEL_NAME=%s GITHUB_PAT=%s", 
            bool(OPENROUTER_API_KEY), MODEL_NAME, bool(GITHUB_PAT))

# -------------------------
# FastAPI app + UI mounting
# -------------------------
app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(ROOT, "templates"))
static_dir = os.path.join(ROOT, "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# -------------------------
# Globals
# -------------------------
_db_pool: Optional["asyncpg.pool.Pool"] = None
TOOLS: Dict[str, Dict[str, Any]] = {}
MCP_SESSION_ID: Optional[str] = None
MCP_CLIENT: Optional[httpx.AsyncClient] = None

def now_id() -> str:
    return str(int(asyncio.get_event_loop().time() * 1000))

# -------------------------
# Logging (tool-call logs)
# -------------------------
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
TOOL_CALL_LOG_PATH = LOG_DIR / "tool_calls.jsonl"

async def log_tool_call(tool: str, arguments: dict, result: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tool": tool,
        "arguments": arguments,
        "result": result
    }
    try:
        with open(TOOL_CALL_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.error("Failed to write tool log: %s", e)

# -------------------------
# SQL safety helpers & Pydantic model
# -------------------------
SQL_FORBIDDEN_TOKENS = ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "truncate "]

def is_safe_select_sql(sql: str) -> bool:
    if not sql or not isinstance(sql, str):
        return False
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    if ";" in s:
        return False
    for t in SQL_FORBIDDEN_TOKENS:
        if t in s:
            return False
    return True

class FetchCreditRecordsArgs(BaseModel):
    sql: Optional[str] = Field(None, description="Read-only SELECT SQL")
    where: Optional[str] = Field(None, description="WHERE clause without 'WHERE'")
    fields: Optional[List[str]] = Field(None, description="Columns to select")
    limit: int = Field(100, ge=1, le=2000, description="Max rows to return")

    @model_validator(mode="after")
    def validate_sql(self) -> "FetchCreditRecordsArgs":
        if self.sql and not is_safe_select_sql(self.sql):
            raise ValueError("sql must be a single read-only SELECT without semicolons or DML.")
        return self

# -------------------------
# DB pool
# -------------------------
async def ensure_db_pool():
    global _db_pool
    if not DATABASE_URL or not HAS_ASYNCPG:
        return
    if _db_pool is None:
        _db_pool = await asyncpg.create_pool(DATABASE_URL)

# -------------------------
# Local tool: fetch_credit_records
# -------------------------
async def fetch_credit_records_tool(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    args = args or {}
    try:
        validated = FetchCreditRecordsArgs(**args)
    except ValidationError as e:
        return {"ok": False, "error": "validation_error", "details": json.loads(e.model_dump_json())}

    if not DATABASE_URL or not HAS_ASYNCPG:
        return {"ok": False, "error": "db_not_configured"}

    await ensure_db_pool()
    try:
        async with _db_pool.acquire() as conn:
            if validated.sql:
                sql = validated.sql.strip().rstrip(";")
                limit = min(int(validated.limit or 100), 2000)
                q = f"{sql} LIMIT {limit}"
                rows = await conn.fetch(q)
                return {"ok": True, "rows": [dict(r) for r in rows]}
            fields = "*"
            if validated.fields:
                safe = [f.replace('"', '') for f in validated.fields]
                fields = ",".join(f'"{f}"' for f in safe)
            where = f"WHERE {validated.where}" if validated.where else ""
            limit = min(int(validated.limit or 100), 2000)
            q = f"SELECT {fields} FROM credit_records {where} LIMIT {limit}"
            rows = await conn.fetch(q)
            return {"ok": True, "rows": [dict(r) for r in rows]}
    except Exception as e:
        logger.exception("DB error")
        return {"ok": False, "error": "db_error", "details": str(e)}

# -------------------------
# GitHub MCP Client (SSE/HTTP based)
# -------------------------
async def init_github_mcp_session():
    """Initialize session with GitHub MCP server"""
    global MCP_SESSION_ID, MCP_CLIENT
    
    if not GITHUB_PAT:
        logger.warning("No GITHUB_PAT provided - GitHub MCP tools will not work")
        return None
    
    try:
        if MCP_CLIENT is None:
            MCP_CLIENT = httpx.AsyncClient(timeout=MCP_TIMEOUT/1000.0)
        
        # Initialize MCP session
        init_payload = {
            "jsonrpc": "2.0",
            "id": now_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "prompts": {}
                },
                "clientInfo": {
                    "name": "mcp-agent",
                    "version": "1.0.0"
                }
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GITHUB_PAT}"
        }
        
        resp = await MCP_CLIENT.post(MCP_ENDPOINT, json=init_payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        
        if "result" in result:
            MCP_SESSION_ID = result.get("result", {}).get("sessionId") or now_id()
            logger.info("GitHub MCP session initialized: %s", MCP_SESSION_ID)
            return MCP_SESSION_ID
        
    except Exception as e:
        logger.error("Failed to initialize GitHub MCP: %s", e)
        return None

async def call_github_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call a tool on the GitHub MCP server"""
    if not GITHUB_PAT:
        return {"error": "GitHub token not configured"}
    
    if MCP_SESSION_ID is None:
        await init_github_mcp_session()
    
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": now_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GITHUB_PAT}"
        }
        
        resp = await MCP_CLIENT.post(MCP_ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        
        if "result" in result:
            return result["result"]
        elif "error" in result:
            return {"error": result["error"]}
        
        return result
        
    except Exception as e:
        logger.exception("GitHub MCP tool call failed")
        return {"error": str(e)}

async def list_github_mcp_tools() -> List[Dict[str, Any]]:
    """List available tools from GitHub MCP server"""
    if not GITHUB_PAT:
        return []
    
    if MCP_SESSION_ID is None:
        await init_github_mcp_session()
    
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": now_id(),
            "method": "tools/list",
            "params": {}
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GITHUB_PAT}"
        }
        
        resp = await MCP_CLIENT.post(MCP_ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        
        if "result" in result and "tools" in result["result"]:
            return result["result"]["tools"]
        
        return []
        
    except Exception as e:
        logger.exception("Failed to list GitHub MCP tools")
        return []

# -------------------------
# Build tools registry
# -------------------------
async def build_tools_registry():
    global TOOLS
    
    # Load GitHub MCP tools
    github_tools_list = await list_github_mcp_tools()
    github_tools = {}
    
    for tool in github_tools_list:
        name = tool.get("name")
        if not name:
            continue
        
        async def make_github_fn(tool_name: str):
            async def fn(args: Optional[Dict[str, Any]] = None):
                return await call_github_mcp_tool(tool_name, args or {})
            return fn
        
        github_tools[name] = {
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema", {"type": "object"}),
            "fn": await make_github_fn(name),
            "source": "github_mcp"
        }
    
    # Local tools
    local_tools = {
        "fetch_credit_records": {
            "description": "Query credit_records database. Accepts {'sql':'SELECT...'} or {'where','fields','limit'}.",
            "parameters": FetchCreditRecordsArgs.model_json_schema(),
            "fn": fetch_credit_records_tool,
            "source": "local"
        }
    }
    
    # Merge all tools
    TOOLS = {**github_tools, **local_tools}
    logger.info("Tools loaded: %s (%d from GitHub MCP, %d local)", 
                ", ".join(TOOLS.keys()), len(github_tools), len(local_tools))

# -------------------------
# Model helpers (OpenRouter) with function calling
# -------------------------
async def call_model_with_tools(messages: List[Dict[str, Any]], max_tokens: int = 2000) -> Dict[str, Any]:
    """Call model with tool definitions for native function calling"""
    
    # Build tools array for OpenAI-compatible function calling
    tools_array = []
    for name, tool_def in TOOLS.items():
        tools_array.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool_def.get("description", ""),
                "parameters": tool_def.get("parameters", {"type": "object"})
            }
        })
    
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "tools": tools_array,
        "tool_choice": "auto"  # Let model decide when to use tools
    }
    
    headers = {"Content-Type": "application/json"}
    if OPENROUTER_API_KEY:
        headers["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"
    
    async with httpx.AsyncClient(timeout=MODEL_TIMEOUT/1000.0) as client:
        resp = await client.post(OPENROUTER_CHAT_URL, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()

# -------------------------
# Enhanced system prompt for agentic behavior
# -------------------------
def build_agentic_system_prompt() -> str:
    tool_descriptions = []
    for name, tool_def in TOOLS.items():
        source = tool_def.get("source", "unknown")
        desc = tool_def.get("description", "")
        tool_descriptions.append(f"  • {name} [{source}]: {desc}")
    
    tools_text = "\n".join(tool_descriptions)
    
    return f"""You are an autonomous AI agent with access to tools that you should use proactively to help users.

AGENTIC BEHAVIOR:
- You should take initiative and use tools WITHOUT always asking for permission
- When a user asks something that requires tool use, just do it
- Chain multiple tool calls together when needed to accomplish complex tasks
- Think step-by-step about what information you need and which tools to use
- If a tool call fails, try alternative approaches or different tools

AVAILABLE TOOLS:
{tools_text}

GITHUB OPERATIONS:
When working with GitHub:
- Use get_file_contents to read files from repositories
- Use search_repositories to find relevant repos
- Use create_or_update_file to modify files
- Use create_issue or create_pull_request to make changes
- Always specify owner, repo, and path correctly

DATABASE OPERATIONS:
When querying the database:
- Use fetch_credit_records with SQL queries or structured filters
- Ensure SQL is read-only (SELECT only)
- Be specific about which fields you need

WORKFLOW:
1. Understand the users question, if it is just a greeting, respond appropriately. 
2. Understand what the user wants to accomplish
3. Determine which tools are needed
4. Execute tools in the right sequence
5. Present results in a clear, actionable way
6. Suggest next steps or related actions

Be proactive, helpful, and autonomous. Don't just describe what you could do - actually do it."""

# -------------------------
# Agentic orchestrator with native function calling
# -------------------------
async def agentic_orchestrator(user_input: str, max_rounds: int = 10) -> Dict[str, Any]:
    """Enhanced agent loop with autonomous tool usage"""
    await build_tools_registry()
    
    system_prompt = build_agentic_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    tool_call_history = []
    
    for round_num in range(max_rounds):
        logger.info(f"Agent round {round_num + 1}/{max_rounds}")
        
        # Call model with tools
        response = await call_model_with_tools(messages, max_tokens=2000)
        
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        
        # Add assistant message to history
        messages.append(message)
        
        # Check if model wants to use tools
        tool_calls = message.get("tool_calls", [])
        
        if not tool_calls or finish_reason == "stop":
            # Model has finished - return final response
            content = message.get("content", "")
            return {
                "ok": True,
                "response": content,
                "tool_history": tool_call_history,
                "rounds": round_num + 1
            }
        
        # Execute tool calls
        for tool_call in tool_calls:
            tool_id = tool_call.get("id")
            function = tool_call.get("function", {})
            tool_name = function.get("name")
            
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            
            if tool_name not in TOOLS:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                # Execute the tool
                tool_fn = TOOLS[tool_name]["fn"]
                try:
                    result = await tool_fn(arguments) if asyncio.iscoroutinefunction(tool_fn) else tool_fn(arguments)
                except Exception as e:
                    logger.exception(f"Tool {tool_name} execution failed")
                    result = {"error": str(e)}
            
            # Log tool call
            tool_call_history.append({
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })
            await log_tool_call(tool_name, arguments, result)
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": json.dumps(result)
            })
        
        # Continue loop - model will see tool results and decide next action
    
    # Max rounds reached
    return {
        "ok": False,
        "error": "Max rounds reached",
        "tool_history": tool_call_history,
        "rounds": max_rounds
    }

# -------------------------
# Routes
# -------------------------
@app.get("/")
async def index(request: Request):
    if os.path.isdir(os.path.join(ROOT, "templates")):
        return templates.TemplateResponse("index.html", {"request": request})
    return JSONResponse({"ok": True, "msg": "Agentic MCP-Agent running"})

@app.post("/api/chat")
async def api_chat(req: Request):
    try:
        payload = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    user_input = payload.get("input") or payload.get("message") or ""
    if isinstance(user_input, list):
        user_input = "\n".join([m.get("content","") if isinstance(m, dict) else str(m) for m in user_input])

    try:
        result = await agentic_orchestrator(user_input, max_rounds=10)
    except Exception as e:
        logger.exception("Agent loop failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    return JSONResponse(result)

@app.get("/api/tools")
async def list_tools():
    """List all available tools"""
    await build_tools_registry()
    tools_info = []
    for name, tool_def in TOOLS.items():
        tools_info.append({
            "name": name,
            "description": tool_def.get("description", ""),
            "source": tool_def.get("source", "unknown"),
            "parameters": tool_def.get("parameters", {})
        })
    return JSONResponse({"tools": tools_info})

# -------------------------
# Startup
# -------------------------
@app.on_event("startup")
async def startup():
    try:
        await init_github_mcp_session()
        await build_tools_registry()
    except Exception:
        logger.exception("Startup initialization failed")

@app.on_event("shutdown")
async def shutdown():
    global MCP_CLIENT
    if MCP_CLIENT:
        await MCP_CLIENT.aclose()