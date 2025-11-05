# RCAbench

This small service is the gateway/orchestrator entrypoint that accepts query requests and forwards them to a downstream LLM-like API. It's intentionally minimal for development.

Quick start (local):

1. Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r gateway/requirements.txt
```

2. Run locally:

```bash
uvicorn gateway.main:app --host 0.0.0.0 --port 8080
```

3. Example request:

```bash
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello from test", "downstream_url": "https://ellm.nrp-nautilus.io/v1"}'
```

Environment variables:
- `DOWNSTREAM_URL` - default downstream URL used when `downstream_url` is not provided in the request (default: `https://ellm.nrp-nautilus.io/v1`).
- `PORT` - port to listen on inside the container (default: `8080`).

- `OPENAI_API_KEY` - API key used by the OpenAI python client to authenticate to the downstream endpoint (set this to your Envoy/OpenAI-compatible key).
- `LOG_LEVEL` - logging level (default: `info`).

Note: The gateway will automatically load `gateway/.env` when started locally (if present). You can also export `OPENAI_API_KEY` in your shell prior to running.

Persistent connections (WebSocket)
--------------------------------

The gateway also exposes a WebSocket at `/ws` so clients can open a long-lived
connection and send multiple prompts without re-establishing HTTP connections.

Example (Python):

```python
import asyncio
import websockets
import json

async def run():
    uri = "ws://localhost:8080/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"prompt": "Say hi"}))
        resp = await ws.recv()
        print(resp)

asyncio.run(run())
```

Each message sent should be a JSON object with a `prompt` key. The server will
reply with a JSON object containing `model` and `content` or an `error` key.

"""
