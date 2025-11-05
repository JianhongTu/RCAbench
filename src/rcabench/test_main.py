import pytest

from fastapi.testclient import TestClient

import gateway.main as mainmod


client = TestClient(mainmod.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_query_forwards(monkeypatch):
    # Monkeypatch the OpenAI client call used inside the endpoint

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeCompletion:
        def __init__(self, text):
            self.choices = [FakeChoice(FakeMessage(text))]
            self.model = "gemma3"

    def fake_create(model, messages):
        # ensure we received expected input shape
        assert isinstance(messages, list)
        return FakeCompletion(f"echo: {messages[0]['content']}")

    # patch the client method used in the module
    monkeypatch.setattr(mainmod.client.chat.completions, "create", fake_create)

    r = client.post("/query", json={"prompt": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["downstream"]["content"] == "echo: hi"
