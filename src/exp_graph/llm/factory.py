"""LLM client factory.

:func:`create_llm_client` builds a client from a provider name and wraps
every non-fake client in the guards that must hold however the client was
built: a hard per-request wall-clock timeout (when a budget is configured)
and the process-wide concurrency limiter (when one is active).  In the
worker sandbox, a team program's ``create_llm_client`` call goes through
the bootstrap's metered factory, which builds the worker client with this
function and the payload's ``request_timeout`` as the wall-clock budget.
"""

from __future__ import annotations

import os

from exp_graph.llm.base import LLMClient
from exp_graph.llm.concurrency import maybe_limit_concurrency
from exp_graph.llm.fake import FakeLLMClient
from exp_graph.llm.openai_client import OpenAIChatClient
from exp_graph.llm.timeout import TimeoutLLMClient

#: Env var holding the per-request hard wall-clock budget (seconds) applied to
#: every non-fake client built here when no explicit ``timeout_s`` is given,
#: so code that builds its own client without passing a timeout is guarded
#: too.  Unset, empty, unparsable or not positive disables the guard.
WALLCLOCK_TIMEOUT_ENV = "EXP_GRAPH_LLM_WALLCLOCK_TIMEOUT"

_REAL_PROVIDERS = {
    "openai",
    "deepseek",
    "bailian",
    "dashscope",
    "qwen",
    "alibaba",
    "xiaomi",
}


def _wallclock_timeout(timeout_s: float | None) -> float:
    """Resolve the effective wall-clock budget in seconds: ``timeout_s`` when
    given, else :data:`WALLCLOCK_TIMEOUT_ENV` (missing, empty or unparsable:
    0.0, no guard)."""
    if timeout_s is not None:
        return float(timeout_s)
    try:
        return float(os.environ.get(WALLCLOCK_TIMEOUT_ENV, "0") or 0)
    except ValueError:
        return 0.0


def _guarded(client: LLMClient, timeout_s: float | None) -> LLMClient:
    """Wrap a real client in a hard wall-clock timeout when one is configured."""
    budget = _wallclock_timeout(timeout_s)
    if budget > 0:
        return TimeoutLLMClient(client, budget)
    return client


def create_llm_client(
    provider: str,
    *,
    base_url: str | None = None,
    api_key_env: str | None = None,
    thinking_enabled: bool | None = None,
    timeout_s: float | None = None,
) -> LLMClient:
    """Create an LLM client.

    ``fake`` is the deterministic offline client.  Every name in
    ``_REAL_PROVIDERS`` builds an OpenAI-compatible
    :class:`~exp_graph.llm.openai_client.OpenAIChatClient`; ``base_url`` and
    ``api_key_env`` override the provider's default endpoint and key
    variable.  ``auto`` uses OpenAI when ``OPENAI_API_KEY`` is available,
    otherwise the fake client for offline smoke runs.  Any other name raises
    ``ValueError``.

    Every non-fake client is wrapped in a hard wall-clock
    :class:`~exp_graph.llm.timeout.TimeoutLLMClient` when a budget is configured
    (``timeout_s`` arg or the ``EXP_GRAPH_LLM_WALLCLOCK_TIMEOUT`` env var), so a
    hung provider call fails fast instead of freezing the run -- no matter which
    code path constructed the client. When a process-wide concurrency cap is
    active (:func:`exp_graph.llm.concurrency.configure_global_llm_concurrency`
    or the ``EXP_GRAPH_LLM_MAX_CONCURRENCY`` env var) the guarded client is
    also concurrency-limited, outside the timeout guard so an abandoned hung
    call can never hold a slot. The fake client is never wrapped.
    """
    if provider == "fake":
        return FakeLLMClient()
    # Wrapper order (outer -> inner): process limiter -> wall-clock guard ->
    # HTTP client.
    if provider in _REAL_PROVIDERS:
        return maybe_limit_concurrency(
            _guarded(
                OpenAIChatClient(
                    base_url=base_url,
                    api_key_env=api_key_env,
                    platform=provider,
                    thinking_enabled=thinking_enabled,
                ),
                timeout_s,
            )
        )
    if provider == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return maybe_limit_concurrency(
                _guarded(
                    OpenAIChatClient(thinking_enabled=thinking_enabled),
                    timeout_s,
                )
            )
        return FakeLLMClient()
    raise ValueError(
        "provider must be one of: auto, fake, openai, deepseek, bailian, dashscope, qwen, alibaba, xiaomi"
    )
