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

Note: The gateway will automatically load `gateway/.env` when started locally (if present). You can also export `OPENAI_API_KEY` in your shell prior to running.

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
