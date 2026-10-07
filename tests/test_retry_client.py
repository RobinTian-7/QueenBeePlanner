"""Bounded retry of transient failures in real-LLM clients.

:class:`queenbee.program.retry.RetryLLMClient` retries connection-class
errors (matched by exception-type name, e.g. ``openai.APIConnectionError``
once the SDK's own retries are exhausted) up to ``attempts`` times with
linear backoff. The wall-clock guard's ``LLMTimeoutError`` has its own
bound (``timeout_attempts``: two calls by default, never more than
``attempts``), and real API errors (auth, bad request) propagate
immediately.
"""
import pytest

from queenbee.program.retry import RetryLLMClient


class _Boom(Exception):
    pass


class APIConnectionError(Exception):
    """Name-matched like openai.APIConnectionError (no openai import needed)."""


class LLMTimeoutError(Exception):
    """Name-matched like the wall-clock guard's error."""


class _Flaky:
    def __init__(self, failures, exc):
        self.failures = failures
        self.exc = exc
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return f"ok after {self.calls}"


def test_transient_error_is_retried_with_backoff():
    sleeps: list[float] = []
    inner = _Flaky(2, APIConnectionError("connection error"))
    client = RetryLLMClient(inner, attempts=3, base_delay=2.0, sleep=sleeps.append)
    assert client.complete("p") == "ok after 3"
    assert inner.calls == 3
    assert sleeps == [2.0, 4.0]


def test_non_transient_error_propagates_immediately():
    inner = _Flaky(5, _Boom("bad request"))
    client = RetryLLMClient(inner, attempts=3, sleep=lambda s: None)
    with pytest.raises(_Boom):
        client.complete("p")
    assert inner.calls == 1


def test_wallclock_timeout_defaults_to_exactly_one_bounded_retry():
    """A wall-clock timeout gets one fresh attempt, bounded by the same
    guard: a single slow call recovers, a persistent timeout still fails
    after 2 calls."""
    inner = _Flaky(5, LLMTimeoutError("hard wall-clock timeout"))
    client = RetryLLMClient(inner, attempts=3, sleep=lambda s: None)
    with pytest.raises(LLMTimeoutError):
        client.complete("p")
    assert inner.calls == 2

    recovered = _Flaky(1, LLMTimeoutError("one slow call"))
    client2 = RetryLLMClient(recovered, attempts=3, sleep=lambda s: None)
    assert client2.complete("p") is not None
    assert recovered.calls == 2


def test_wallclock_timeout_attempts_can_be_raised_for_long_runs():
    recovered = _Flaky(3, LLMTimeoutError("several slow calls"))
    client = RetryLLMClient(
        recovered,
        attempts=5,
        timeout_attempts=5,
        sleep=lambda s: None,
    )
    assert client.complete("p") == "ok after 4"
    assert recovered.calls == 4


def test_exhausted_attempts_raise_last_error():
    inner = _Flaky(99, APIConnectionError("still down"))
    client = RetryLLMClient(inner, attempts=3, sleep=lambda s: None)
    with pytest.raises(APIConnectionError):
        client.complete("p")
    assert inner.calls == 3
