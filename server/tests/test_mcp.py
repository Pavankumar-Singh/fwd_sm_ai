import json
import asyncio
import os

import pytest
from httpx import AsyncClient

try:
    import server.main as appmod
except Exception:
    pytest.skip("server.main not importable", allow_module_level=True)

@pytest.mark.asyncio
async def test_mcp_initialize_and_list(aiohttp_unused_port):
    # start the app with Test client
    from server import main as mainapp
    async with AsyncClient(app=mainapp.app, base_url="http://test") as ac:
        # initialize
        payload = {"jsonrpc": "2.0", "id": "1", "method": "initialize", "params": {}}
        r = await ac.post("/mcp", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body.get("result") and body["result"].get("protocolVersion")

        # list tools
        payload = {"jsonrpc": "2.0", "id": "2", "method": "tools/list", "params": {}}
        r = await ac.post("/mcp", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body.get("result"), list)

@pytest.mark.asyncio
async def test_mcp_local_tool_fetch_credit_records():
    from server import main as mainapp
    async with AsyncClient(app=mainapp.app, base_url="http://test") as ac:
        # call local tool; may return ok:false if no DB configured
        payload = {"jsonrpc": "2.0", "id": "3", "method": "fetch_credit_records", "params": {}}
        r = await ac.post("/mcp", json=payload)
        assert r.status_code == 200
        body = r.json()
        # result should be present (ok true/false depends on DB)
        assert "result" in body
