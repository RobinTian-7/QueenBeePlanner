"""Bounded retry for transient connection failures on real-LLM clients.

A single network blip (an ``openai.APIConnectionError`` that survives any
SDK-level retries) would otherwise fail the whole operation that issued the
call, so the client absorbs transient failures itself.
``queenbee.program.clients`` narrows the retryable set further for planner
calls, to connect-class errors that provably never reached the provider.

Scope is deliberately narrow:
* Connection-class errors are retried, matched by exception-type name anywhere
  in the MRO, so there is no hard dependency on openai/httpx imports.
* The wall-clock guard's ``LLMTimeoutError`` (``exp_graph.llm.timeout``) gets a
  separately bounded number of attempts (``timeout_attempts``, two by
  default). Callers may raise that ceiling; every fresh attempt is still
  protected by the same per-request wall-clock guard.
* Every other exception, including API errors such as authentication
  failures, bad requests and rate limits, propagates immediately; retrying
  those is left to the SDK's own retry setting.
"""
from __future__ import annotations

import time
from typing import Any, Callable

# Exception type names treated as transient (matched anywhere in the MRO).
TRANSIENT_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "PoolTimeout",
    }
)


class RetryLLMClient:
    """Wrap an (already timeout-guarded) LLM client with bounded retry.

    ``attempts`` bounds connection-class errors and ``timeout_attempts`` (capped
    by ``attempts``) bounds ``LLMTimeoutError``; any other exception is raised
    on its first occurrence. The wait before retry ``k`` is ``base_delay * k``.
    """

    def __init__(
        self,
        inner: Any,
        *,
        attempts: int = 3,
        timeout_attempts: int = 2,
        base_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._attempts = max(1, int(attempts))
        self._timeout_attempts = max(1, int(timeout_attempts))
        self._base_delay = float(base_delay)
        self._sleep = sleep

    @staticmethod
    def _is_transient(exc: BaseException) -> bool:
        return any(t.__name__ in TRANSIENT_ERROR_NAMES for t in type(exc).__mro__)

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        return type(exc).__name__ == "LLMTimeoutError"

    # Attempt ceiling for this error class; 1 means it is raised without retry.
    def _attempts_for(self, exc: BaseException) -> int:
        if self._is_timeout(exc):
            return min(self._timeout_attempts, self._attempts)
        if self._is_transient(exc):
            return self._attempts
        return 1

    # Arguments pass through untouched, so any inner client signature works;
    # every retry re-sends the same request.
    def complete(self, *args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._inner.complete(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below
                if attempt >= self._attempts_for(exc):
                    raise
                delay = self._base_delay * attempt
                print(
                    f"  [llm retry] {type(exc).__name__}; "
                    f"attempt {attempt}/{self._attempts_for(exc)}, retrying in {delay:.0f}s",
                    flush=True,
                )
                self._sleep(delay)
