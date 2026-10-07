"""LLM client chain for planner calls, and the HTTP timeout defaults.

``_build_arm_client`` wraps an OpenAI-compatible chat client (endpoint and
key from the SDK environment, ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``,
unless ``base_url`` / ``api_key_env`` are given) in a wall-clock guard, the
process-wide concurrency limiter and an optional retry of connect-class
failures.
``http_timeout_defaults`` supplies the OpenAI client's HTTP timeouts for
the duration of a run when the environment does not set them.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator, Mapping

from exp_graph.llm.concurrency import maybe_limit_concurrency

from queenbee.program.retry import RetryLLMClient


# Exception type names (matched across the MRO) of connect-class failures,
# which are retried, and of failures after which the request may be running
# at the provider, which are not.  ``APITimeoutError`` subclasses
# ``APIConnectionError`` in the SDK, so the terminal set is consulted first;
# otherwise every SDK timeout would match the connect entry through its MRO.
_ARM_RETRYABLE_ERROR_NAMES = frozenset(
    {"APIConnectionError", "ConnectError", "ConnectTimeout", "PoolTimeout"}
)
_ARM_TERMINAL_ERROR_NAMES = frozenset(
    {
        "APITimeoutError",
        "ReadTimeout",
        "WriteTimeout",
        "ReadError",
        "RemoteProtocolError",
    }
)


class _ArmConnectRetryClient(RetryLLMClient):
    """A :class:`RetryLLMClient` that retries connect-class failures only.

    Retried: ``APIConnectionError`` and httpx's ``ConnectError`` /
    ``ConnectTimeout`` / ``PoolTimeout``, failures that normally occur before
    a request is delivered.  Terminal: timeouts (``APITimeoutError``,
    ``ReadTimeout``, ``WriteTimeout``) and httpx's ``ReadError`` /
    ``RemoteProtocolError``; such a request may already be running, and
    billed, at the provider, while usage is recorded only from returned
    responses.  Classification is by exception-type name across the MRO
    (``__cause__`` is not inspected), so neither ``openai`` nor ``httpx``
    has to be imported.  The ``openai`` SDK wraps httpx errors: every
    timeout reaches this client as ``APITimeoutError`` (terminal, a connect
    timeout included) and every other transport failure as
    ``APIConnectionError`` (retried, a connection dropped mid-request
    included).
    """

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        names = {kind.__name__ for kind in type(exc).__mro__}
        if names & _ARM_TERMINAL_ERROR_NAMES:
            return False
        return bool(names & _ARM_RETRYABLE_ERROR_NAMES)


def _build_arm_client(
    llm: str,
    *,
    timeout_s: float,
    reasoning_effort: str | None = None,
    connect_attempts: int = 1,
    connect_backoff_s: float = 5.0,
    base_url: str | None = None,
    api_key_env: str | None = None,
    max_completion_tokens: int | None = None,
):
    """The planner client chain: ConnectRetry(Limiter(Timeout(OpenAI(max_retries=0)))).

    ``timeout_s`` is the wall-clock guard of each call; the limiter is the
    process-wide LLM concurrency limiter (left out when none is configured).
    The connect retry (present only with ``connect_attempts > 1``) sits
    outside the limiter, so a backoff sleep never holds a limiter slot; the
    SDK runs with ``max_retries=0`` so that nothing retries behind this
    classification, and a response without provider-reported token usage is
    an error.  ``llm`` must be ``"openai"`` (any OpenAI-compatible endpoint).
    """

    if llm != "openai":
        raise ValueError(f"unknown planner client {llm!r}; expected 'openai'")
    from exp_graph.llm.openai_client import OpenAIChatClient
    from exp_graph.llm.timeout import TimeoutLLMClient

    guarded = maybe_limit_concurrency(
        TimeoutLLMClient(
            OpenAIChatClient(
                max_retries=0,
                require_provider_usage=True,
                reasoning_effort=reasoning_effort,
                # Optional endpoint override (e.g. planner and workers served
                # from different endpoints); None falls back to the SDK
                # environment (OPENAI_BASE_URL / OPENAI_API_KEY).
                base_url=base_url,
                api_key_env=api_key_env,
                max_completion_tokens=max_completion_tokens,
            ),
            timeout_s,
        )
    )
    if connect_attempts <= 1:
        return guarded
    return _ArmConnectRetryClient(
        guarded,
        attempts=connect_attempts,
        # A call stopped by the wall-clock guard may still be running and
        # billed at the provider: this layer does not retry it.
        timeout_attempts=1,
        # Connection errors fail fast (refused / unreachable / DNS) instead
        # of waiting out the connect timeout, so the attempts alone span
        # almost no wall clock; the backoff (connect_backoff_s x attempt) is
        # what bridges a short network outage without failing the call.
        base_delay=connect_backoff_s,
    )


#: Connect timeout (seconds) that :func:`http_timeout_defaults` supplies
#: while ``OPENAI_CONNECT_TIMEOUT`` is unset.
DEFAULT_CONNECT_TIMEOUT_S = 30


@contextlib.contextmanager
def env_defaults(values: Mapping[str, str]) -> Iterator[None]:
    """Set the variables of ``values`` that are not set (sandboxed workers
    inherit them) and remove them again on exit."""

    added = [k for k in values if k not in os.environ]
    for key in added:
        os.environ[key] = values[key]
    try:
        yield
    finally:
        for key in added:
            os.environ.pop(key, None)


def http_timeout_defaults(request_timeout: float) -> contextlib.AbstractContextManager[None]:
    """While unset, ``OPENAI_TIMEOUT`` follows ``request_timeout`` and
    ``OPENAI_CONNECT_TIMEOUT`` is :data:`DEFAULT_CONNECT_TIMEOUT_S`: the
    OpenAI client reads its HTTP timeouts from these variables only."""

    return env_defaults({
        "OPENAI_TIMEOUT": f"{float(request_timeout):g}",
        "OPENAI_CONNECT_TIMEOUT": str(DEFAULT_CONNECT_TIMEOUT_S),
    })
