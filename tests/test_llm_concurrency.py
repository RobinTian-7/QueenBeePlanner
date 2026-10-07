"""Process-wide LLM concurrency limiting: cap, release, and global wiring."""

from __future__ import annotations

import threading
import time

import pytest

import exp_graph.llm.concurrency as concurrency
from exp_graph.llm.base import LLMResponse, LLMUsage
from exp_graph.llm.concurrency import (
    ConcurrencyLimitedLLMClient,
    LLMConcurrencyLimiter,
    configure_global_llm_concurrency,
    global_llm_limiter,
    maybe_limit_concurrency,
)
from exp_graph.llm.factory import create_llm_client
from exp_graph.llm.fake import FakeLLMClient


@pytest.fixture(autouse=True)
def _reset_global_limiter():
    """Restore the unconfigured global state after every test."""

    yield
    concurrency._GLOBAL_LIMITER = None
    concurrency._GLOBAL_CONFIGURED = False


class _SlowInner:
    """Inner client that sleeps briefly so calls genuinely overlap."""

    def __init__(self, limiter: LLMConcurrencyLimiter) -> None:
        self._limiter = limiter
        self.calls = 0
        self._lock = threading.Lock()
        self.observed_over_cap = False

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        if self._limiter.in_flight > self._limiter.max_concurrent:
            self.observed_over_cap = True
        time.sleep(0.02)
        with self._lock:
            self.calls += 1
        return LLMResponse(text="{}", usage=LLMUsage())


def test_limiter_caps_in_flight_under_thread_storm() -> None:
    limiter = LLMConcurrencyLimiter(4)
    inner = _SlowInner(limiter)
    client = ConcurrencyLimitedLLMClient(inner, limiter)

    threads = [
        threading.Thread(target=client.complete, args=("p", "m"))
        for _ in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert inner.calls == 16
    assert not inner.observed_over_cap
    assert limiter.peak_in_flight <= 4
    # The storm must actually have exercised the limiter, not degenerated to
    # sequential execution.
    assert limiter.peak_in_flight >= 2
    assert limiter.in_flight == 0


def test_slot_released_when_inner_raises() -> None:
    limiter = LLMConcurrencyLimiter(1)

    class _Boom:
        def complete(self, *args, **kwargs):
            raise RuntimeError("provider exploded")

    client = ConcurrencyLimitedLLMClient(_Boom(), limiter)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="provider exploded"):
            client.complete("p", "m")
    assert limiter.in_flight == 0

    class _Ok:
        def complete(self, *args, **kwargs):
            return LLMResponse(text="ok", usage=LLMUsage())

    # The semaphore was not leaked: a fresh call still gets a slot.
    assert (
        ConcurrencyLimitedLLMClient(_Ok(), limiter).complete("p", "m").text
        == "ok"
    )


def test_wrapper_forwards_arguments_unchanged() -> None:
    seen: dict[str, object] = {}

    class _Capture:
        def complete(self, prompt, model_name, temperature=None, json_mode=True):
            seen.update(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                json_mode=json_mode,
            )
            return LLMResponse(text="captured", usage=LLMUsage())

    client = ConcurrencyLimitedLLMClient(_Capture(), LLMConcurrencyLimiter(2))
    response = client.complete("hello", "gpt-4o-mini", 0.0, json_mode=False)
    assert response.text == "captured"
    assert seen == {
        "prompt": "hello",
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "json_mode": False,
    }


def test_invalid_limiter_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        LLMConcurrencyLimiter(0)
    with pytest.raises(ValueError):
        LLMConcurrencyLimiter(True)  # a bool is not a valid concurrency cap
    with pytest.raises(TypeError):
        ConcurrencyLimitedLLMClient(FakeLLMClient(), limiter=None)


def test_global_limiter_reads_env_once(monkeypatch) -> None:
    monkeypatch.setenv(concurrency.LLM_MAX_CONCURRENCY_ENV, "15")
    limiter = global_llm_limiter()
    assert limiter is not None and limiter.max_concurrent == 15
    # Later env changes do not silently retune an already-resolved limiter.
    monkeypatch.setenv(concurrency.LLM_MAX_CONCURRENCY_ENV, "3")
    assert global_llm_limiter() is limiter


def test_global_limiter_disabled_on_unset_zero_or_junk(monkeypatch) -> None:
    for raw in ("", "0", "junk", "-2"):
        concurrency._GLOBAL_LIMITER = None
        concurrency._GLOBAL_CONFIGURED = False
        if raw:
            monkeypatch.setenv(concurrency.LLM_MAX_CONCURRENCY_ENV, raw)
        else:
            monkeypatch.delenv(
                concurrency.LLM_MAX_CONCURRENCY_ENV, raising=False
            )
        assert global_llm_limiter() is None
        inner = FakeLLMClient()
        assert maybe_limit_concurrency(inner) is inner


def test_explicit_configuration_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv(concurrency.LLM_MAX_CONCURRENCY_ENV, "7")
    limiter = configure_global_llm_concurrency(5)
    assert limiter is not None and limiter.max_concurrent == 5
    assert global_llm_limiter() is limiter
    wrapped = maybe_limit_concurrency(FakeLLMClient())
    assert isinstance(wrapped, ConcurrencyLimitedLLMClient)
    assert wrapped.limiter is limiter
    assert configure_global_llm_concurrency(None) is None
    inner = FakeLLMClient()
    assert maybe_limit_concurrency(inner) is inner


def test_factory_fake_path_never_wrapped(monkeypatch) -> None:
    monkeypatch.setenv(concurrency.LLM_MAX_CONCURRENCY_ENV, "15")
    concurrency._GLOBAL_LIMITER = None
    concurrency._GLOBAL_CONFIGURED = False
    client = create_llm_client("fake")
    assert isinstance(client, FakeLLMClient)
