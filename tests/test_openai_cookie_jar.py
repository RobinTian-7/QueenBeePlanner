"""The OpenAI client never persists response cookies.

An endpoint or load balancer that sets session cookies would otherwise grow
the default httpx cookie jar across the thousands of calls a long run makes
in one process, until the Cookie header alone makes every request fail with
HTTP 431 (request headers too large).  ``OPENAI_KEEP_COOKIES=1`` restores
the default jar."""

from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx")


def _client_or_skip(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_KEEP_COOKIES", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    from exp_graph.llm.openai_client import OpenAIChatClient

    return OpenAIChatClient(platform="openai")


def test_http_client_does_not_store_response_cookies(monkeypatch):
    client = _client_or_skip(monkeypatch)
    http_client = client._client._client  # openai SDK wraps an httpx.Client
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    response = httpx.Response(
        200, headers={"set-cookie": "lb_session=abcdef; Path=/"}, request=request
    )
    # Simulate what httpx does after every response: extract the cookies into
    # the client's jar.  The overridden ``cookies`` property must leave
    # nothing persisted.
    http_client.cookies.extract_cookies(response)
    assert len(http_client.cookies.jar) == 0, "response cookies must never persist"


def test_keep_cookies_env_restores_default(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_KEEP_COOKIES", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    from exp_graph.llm.openai_client import OpenAIChatClient

    client = OpenAIChatClient(platform="openai")
    jar = client._client._client.cookies
    request = httpx.Request("GET", "https://api.openai.com/v1/models")
    response = httpx.Response(
        200, headers={"set-cookie": "lb_session=abcdef; Path=/"}, request=request
    )
    jar.extract_cookies(response)
    assert len(jar.jar) == 1, "opt-out env must restore the default jar"
