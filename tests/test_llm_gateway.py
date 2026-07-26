"""
Regression tests for LLM gateway failure handling (backend/generator.py).

Background: _stream_once used to only *log* resp.status_code. A 429 or 502 from
the TAMU gateway therefore produced an empty token list that looked exactly like
a successful empty answer, and the chat endpoint told the user "I don't have
those details — check the sources below". Outages were being reported to users
as missing website content, and showed up in eval runs as retrieval-quality
regressions rather than infrastructure failures.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import generator  # noqa: E402
from generator import LLMGatewayError  # noqa: E402


# ── Error classification ─────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert status in generator._RETRYABLE_STATUS


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retryable(status):
    """A bad API key or malformed payload fails identically on every attempt —
    retrying only multiplies the latency the user waits through."""
    assert status not in generator._RETRYABLE_STATUS


def test_gateway_error_carries_status_and_retryability():
    err = LLMGatewayError("boom", status=503, retryable=True)
    assert err.status == 503
    assert err.retryable is True


# ── Retry loop ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retries_transient_failure_then_succeeds(monkeypatch):
    calls = {"n": 0}

    async def fake_stream_once(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMGatewayError("429", status=429, retryable=True)
        return "the answer", "stop"

    monkeypatch.setattr(generator, "_stream_once", fake_stream_once)
    monkeypatch.setattr(generator, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(generator, "_retry_delay", lambda attempt: 0.0)

    text, finish = await generator._stream_once_with_retry([])
    assert text == "the answer"
    assert finish == "stop"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_does_not_retry_non_retryable(monkeypatch):
    calls = {"n": 0}

    async def fake_stream_once(messages):
        calls["n"] += 1
        raise LLMGatewayError("bad key", status=401, retryable=False)

    monkeypatch.setattr(generator, "_stream_once", fake_stream_once)
    monkeypatch.setattr(generator, "LLM_MAX_RETRIES", 3)
    monkeypatch.setattr(generator, "_retry_delay", lambda attempt: 0.0)

    with pytest.raises(LLMGatewayError):
        await generator._stream_once_with_retry([])
    assert calls["n"] == 1, "a 401 must not be retried"


@pytest.mark.asyncio
async def test_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    async def fake_stream_once(messages):
        calls["n"] += 1
        raise LLMGatewayError("502", status=502, retryable=True)

    monkeypatch.setattr(generator, "_stream_once", fake_stream_once)
    monkeypatch.setattr(generator, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(generator, "_retry_delay", lambda attempt: 0.0)

    with pytest.raises(LLMGatewayError):
        await generator._stream_once_with_retry([])
    assert calls["n"] == 3, "should be 1 initial attempt + 2 retries"


def test_retry_delay_grows_and_is_jittered(monkeypatch):
    monkeypatch.setattr(generator, "LLM_RETRY_BASE_DELAY", 1.0)
    # Jitter multiplier is in [0.5, 1.5), so bounds are deterministic even
    # though the exact value isn't.
    assert 0.5 <= generator._retry_delay(0) < 1.5
    assert 1.0 <= generator._retry_delay(1) < 3.0
    assert 2.0 <= generator._retry_delay(2) < 6.0


# ── HTTP-level behaviour ─────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code, lines=(), text=""):
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text

    async def aread(self):
        return self.text.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, response):
        self._response = response

    def stream(self, *args, **kwargs):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_non_200_raises_instead_of_returning_empty(monkeypatch):
    """The core regression: a 429 must raise, not return ('', '')."""
    resp = _FakeResponse(429, text="rate limited")
    monkeypatch.setattr(generator.httpx, "AsyncClient", lambda **kw: _FakeClient(resp))

    with pytest.raises(LLMGatewayError) as exc:
        await generator._stream_once([{"role": "user", "content": "hi"}])
    assert exc.value.status == 429
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_empty_200_stream_raises(monkeypatch):
    """A 200 that yields no deltas and no finish_reason is a dropped upstream
    stream, not a real empty answer."""
    resp = _FakeResponse(200, lines=["", "data: [DONE]"])
    monkeypatch.setattr(generator.httpx, "AsyncClient", lambda **kw: _FakeClient(resp))

    with pytest.raises(LLMGatewayError):
        await generator._stream_once([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_successful_stream_is_parsed(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    resp = _FakeResponse(200, lines=lines)
    monkeypatch.setattr(generator.httpx, "AsyncClient", lambda **kw: _FakeClient(resp))

    text, finish = await generator._stream_once([{"role": "user", "content": "hi"}])
    assert text == "Hello world"
    assert finish == "stop"
