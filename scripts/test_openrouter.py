#!/usr/bin/env python3
import os
import sys
import json

try:
    import httpx
except Exception as e:
    print("Please install httpx in the venv: pip install httpx", file=sys.stderr)
    raise

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_CHAT_URL = os.getenv("OPENROUTER_CHAT_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-r1:free")

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY is not set in the environment", file=sys.stderr)
    sys.exit(2)

body = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "ping"}],
    "temperature": 0.0,
    "max_tokens": 16
}

# Print a redacted summary of the request
print("Calling OpenRouter endpoint:")
print("  URL:", OPENROUTER_CHAT_URL)
print("  model:", MODEL_NAME)
print("  payload keys:", list(body.keys()))

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENROUTER_API_KEY}"}
print(headers)
try:
    resp = httpx.post(OPENROUTER_CHAT_URL, json=body, headers=headers, timeout=30.0)
    print("HTTP status:", resp.status_code)
    # Print full body (helps debugging). Be careful not to show API key.
    print("Response body:", resp.text)
    try:
        data = resp.json()
        print("Top-level keys:", list(data.keys()))
    except Exception:
        print("Response is not JSON or JSON parsing failed")
    resp.raise_for_status()
    print("Request succeeded.")
    sys.exit(0)
except Exception as e:
    print("Request failed:", str(e), file=sys.stderr)
    sys.exit(1)
