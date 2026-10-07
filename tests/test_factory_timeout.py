"""exp_graph.llm.factory: the wall-clock timeout guard is applied at client
construction, so every real-provider client that create_llm_client builds
is guarded whenever a budget is set (``timeout_s`` or
``EXP_GRAPH_LLM_WALLCLOCK_TIMEOUT``) -- not just a client a caller wraps
explicitly.  The fake client is never wrapped; without a budget the raw
client is returned.
"""
from exp_graph.llm import factory
from exp_graph.llm.fake import FakeLLMClient
from exp_graph.llm.timeout import TimeoutLLMClient


class _RecordingClient:
    """Stand-in for OpenAIChatClient that records its construction kwargs."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def complete(self, prompt, model_name, temperature=None):  # pragma: no cover
        raise AssertionError("not called in these tests")


def test_fake_provider_is_never_wrapped():
    # The fake client is instant/deterministic -> no guard, even if asked.
    client = factory.create_llm_client("fake", timeout_s=5)
    assert isinstance(client, FakeLLMClient)


def test_explicit_timeout_wraps_real_client(monkeypatch):
    monkeypatch.setattr(factory, "OpenAIChatClient", _RecordingClient)
    client = factory.create_llm_client("openai", timeout_s=5)
    assert isinstance(client, TimeoutLLMClient)
    assert client._timeout_s == 5
    assert isinstance(client._inner, _RecordingClient)


def test_env_timeout_wraps_real_client(monkeypatch):
    # Callers that pass no timeout_s inherit the budget set in the
    # EXP_GRAPH_LLM_WALLCLOCK_TIMEOUT environment variable.
    monkeypatch.setattr(factory, "OpenAIChatClient", _RecordingClient)
    monkeypatch.setenv("EXP_GRAPH_LLM_WALLCLOCK_TIMEOUT", "7")
    client = factory.create_llm_client("openai")
    assert isinstance(client, TimeoutLLMClient)
    assert client._timeout_s == 7.0


def test_no_timeout_returns_raw_client(monkeypatch):
    # With no explicit timeout and no environment budget the factory returns
    # the raw client, unguarded.
    monkeypatch.setattr(factory, "OpenAIChatClient", _RecordingClient)
    monkeypatch.delenv("EXP_GRAPH_LLM_WALLCLOCK_TIMEOUT", raising=False)
    client = factory.create_llm_client("openai")
    assert isinstance(client, _RecordingClient)


def test_factory_guard_actually_fires(monkeypatch):
    # A hung provider call made through a factory-built client must raise
    # LLMTimeoutError at the budget instead of blocking the caller.
    import time

    import pytest

    from exp_graph.llm.base import LLMResponse, LLMUsage
    from exp_graph.llm.timeout import LLMTimeoutError

    class _HangingClient:
        def __init__(self, **kwargs) -> None:
            pass

        def complete(self, prompt, model_name, temperature=None):
            time.sleep(5)  # simulate a wedged provider call (releases the GIL)
            return LLMResponse(text="{}", usage=LLMUsage())

    monkeypatch.setattr(factory, "OpenAIChatClient", _HangingClient)
    monkeypatch.setenv("EXP_GRAPH_LLM_WALLCLOCK_TIMEOUT", "0.3")
    client = factory.create_llm_client("openai")

    start = time.perf_counter()
    with pytest.raises(LLMTimeoutError):
        client.complete("p", "m")
    assert time.perf_counter() - start < 2.0  # fired at ~0.3s, not the 5s hang


def test_guard_passes_results_and_errors_through():
    import pytest

    from exp_graph.llm.base import LLMResponse, LLMUsage

    class _Inner:
        def __init__(self) -> None:
            self.calls = []

        def complete(self, prompt, model_name, temperature=None, **kwargs):
            self.calls.append((prompt, model_name, temperature, kwargs))
            if prompt == "boom":
                raise RuntimeError("provider exploded")
            return LLMResponse(text="ok", usage=LLMUsage())

    inner = _Inner()
    client = TimeoutLLMClient(inner, 5)
    assert client.complete("p", "m", 0.0, json_mode=False).text == "ok"
    assert inner.calls[-1] == ("p", "m", 0.0, {"json_mode": False})
    with pytest.raises(RuntimeError, match="provider exploded"):
        client.complete("boom", "m")
    # No budget: a direct pass-through on the caller's thread.
    assert TimeoutLLMClient(inner, None).complete("p", "m").text == "ok"
