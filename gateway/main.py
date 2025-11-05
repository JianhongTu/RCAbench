from typing import Optional

import logging

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import anyio
from openai import OpenAI

from .config import DOWNSTREAM_URL, OPENAI_API_KEY, DEFAULT_MODEL, PORT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("gateway")

app = FastAPI(title="RCAbench Gateway", version="0.1.0")


class QueryRequest(BaseModel):
    prompt: str
    downstream_url: Optional[str] = None


class QueryResponse(BaseModel):
    downstream: dict


DEFAULT_DOWNSTREAM = DOWNSTREAM_URL

# Create an OpenAI client instance configured to talk to the downstream base_url.
client = OpenAI(api_key=OPENAI_API_KEY, base_url=DEFAULT_DOWNSTREAM)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Accept a prompt and use the OpenAI client to create a chat completion.

    The request will create a single-user message from the `prompt` and call
    `client.chat.completions.create`. The call is executed in a thread to avoid
    blocking the event loop. The endpoint returns a small structured response
    containing the model and the top choice text.
    """
    model = req.downstream_url or DEFAULT_MODEL
    # If the caller provided a downstream_url, treat it as a model override or base_url.
    # Prefer explicit model via `downstream_url` field is unusual; the gateway keeps
    # backward compatibility by allowing callers to set a custom base_url via
    # `downstream_url` but model selection should normally use DEFAULT_MODEL or env var.

    messages = [{"role": "user", "content": req.prompt}]

    logger.info("Creating chat completion with model=%s, base_url=%s", model, DEFAULT_DOWNSTREAM)

    def _call_openai():
        # Use the OpenAI client to create a chat completion synchronously
        return client.chat.completions.create(model=model, messages=messages)

    try:
        completion = await anyio.to_thread.run_sync(_call_openai)
    except Exception as exc:
        logger.exception("OpenAI client call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Downstream request failed: {exc}")

    # Extract text from the first choice if available
    try:
        top_choice = completion.choices[0]
        text = getattr(top_choice.message, "content", None) or top_choice.message["content"]
    except Exception:
        logger.exception("Unexpected completion shape: %s", completion)
        raise HTTPException(status_code=502, detail="Downstream returned unexpected response shape")

    return {"downstream": {"model": getattr(completion, "model", None), "content": text}}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Accepts a persistent websocket connection.

    Clients should send JSON messages with a `prompt` field. The server
    will respond with a JSON object containing `model` and `content`, or an
    `error` field if something goes wrong.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt")
            if not prompt:
                await websocket.send_json({"error": "missing prompt"})
                continue

            messages = [{"role": "user", "content": prompt}]

            def _call_openai():
                return client.chat.completions.create(model=DEFAULT_MODEL, messages=messages)

            try:
                completion = await anyio.to_thread.run_sync(_call_openai)
            except Exception as exc:
                logger.exception("OpenAI client call failed over websocket: %s", exc)
                await websocket.send_json({"error": str(exc)})
                continue

            try:
                top_choice = completion.choices[0]
                text = getattr(top_choice.message, "content", None) or top_choice.message["content"]
            except Exception:
                logger.exception("Unexpected completion shape over websocket: %s", completion)
                await websocket.send_json({"error": "downstream returned unexpected shape"})
                continue

            await websocket.send_json({"model": getattr(completion, "model", None), "content": text})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, log_level="info")
