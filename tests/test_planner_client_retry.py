"""The planner client retries failed connections and nothing else.

A request whose connection never opened cannot have reached the provider, so
sending it again is safe.  A request that may already be executing
provider-side (read timeout, wall-clock timeout, any other error) must NOT be
re-sent: planner usage is metered only from returned responses, so a second
copy would leave the spend of the first one uncounted.

Also covers the planner client chain: the connect retry wraps a wall-clock
guard over an OpenAI-compatible client with SDK retries off, provider-reported
usage required and the given endpoint and key.
"""

from __future__ import annotations

import pytest

from queenbee.program.clients import _ArmConnectRetryClient, _build_arm_client


# Classification is by exception-type NAME across the MRO (no hard openai/httpx
# import), so these stubs must be named exactly like the real types.
class ConnectTimeout(Exception):
    """Stands in for httpx.ConnectTimeout."""


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    """Mirrors the SDK, where a read timeout subclasses the connect error."""


class _CountingClient:
    def __init__(self, failures: int, error: Exception) -> None:
        self.calls = 0
        self._failures = failures
        self._error = error

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self._failures:
            raise self._error
        return "ok"


def _client(inner):
    return _ArmConnectRetryClient(inner, attempts=3, timeout_attempts=1,
                                  sleep=lambda _seconds: None)


def test_connect_blip_is_retried_and_recovers() -> None:
    inner = _CountingClient(failures=2, error=ConnectTimeout("blip"))
    assert _client(inner).complete("p", "m") == "ok"
    assert inner.calls == 3


def test_read_timeout_is_terminal_despite_connect_error_ancestry() -> None:
    """The SDK's APITimeoutError inherits APIConnectionError: it must not slip
    through on the ancestor's name."""

    inner = _CountingClient(failures=1, error=APITimeoutError("read"))
    with pytest.raises(APITimeoutError):
        _client(inner).complete("p", "m")
    assert inner.calls == 1, "a possibly-billed crossing was re-issued"


def test_plain_connection_error_is_retried() -> None:
    inner = _CountingClient(failures=1, error=APIConnectionError("no route"))
    assert _client(inner).complete("p", "m") == "ok"
    assert inner.calls == 2


def test_unclassified_errors_never_retry() -> None:
    inner = _CountingClient(failures=1, error=ValueError("bad request"))
    with pytest.raises(ValueError):
        _client(inner).complete("p", "m")
    assert inner.calls == 1


def _chain(client) -> list:
    out = []
    while client is not None and len(out) < 10:
        out.append(client)
        client = getattr(client, "_inner", None)
    return out


def test_planner_client_chain_uses_the_given_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("QB_TEST_PLANNER_KEY", "sk-test-not-used")
    monkeypatch.delenv("OPENAI_MAX_COMPLETION_TOKENS", raising=False)
    client = _build_arm_client(
        "openai", timeout_s=123.0, reasoning_effort="high", connect_attempts=8,
        connect_backoff_s=8.0, base_url="http://127.0.0.1:9/v1",
        api_key_env="QB_TEST_PLANNER_KEY", max_completion_tokens=500,
    )
    chain = _chain(client)
    names = [type(node).__name__ for node in chain]
    assert names[0] == "_ArmConnectRetryClient"
    assert client._attempts == 8 and client._timeout_attempts == 1
    assert client._base_delay == 8.0
    assert "TimeoutLLMClient" in names and names[-1] == "OpenAIChatClient"
    timeout = chain[names.index("TimeoutLLMClient")]
    assert timeout._timeout_s == 123.0
    leaf = chain[-1]
    assert leaf._reasoning_effort == "high"
    assert leaf._max_completion_tokens == 500
    assert leaf._require_provider_usage is True
    assert leaf._client.max_retries == 0
    assert str(leaf._client.base_url).rstrip("/") == "http://127.0.0.1:9/v1"
    assert leaf._client.api_key == "sk-test-not-used"


def test_single_connect_attempt_returns_the_guarded_client(monkeypatch) -> None:
    monkeypatch.setenv("QB_TEST_PLANNER_KEY", "sk-test-not-used")
    client = _build_arm_client(
        "openai", timeout_s=5.0, connect_attempts=1,
        base_url="http://127.0.0.1:9/v1", api_key_env="QB_TEST_PLANNER_KEY",
    )
    assert "_ArmConnectRetryClient" not in [type(n).__name__ for n in _chain(client)]


def test_only_openai_compatible_planner_clients_exist() -> None:
    with pytest.raises(ValueError, match="openai"):
        _build_arm_client("fake", timeout_s=1.0)
