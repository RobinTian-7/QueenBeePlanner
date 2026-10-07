"""Process-wide concurrency limiting for real LLM calls.

Parallel callers multiply the in-flight provider requests of a process.
Providers throttle by concurrent connections as much as by tokens, and an
unbounded fan-out turns one slow call site into a rate-limit storm that
poisons every other request in the process.  This module gives the process
one shared gate (a bounded semaphore) so the total number of in-flight LLM
calls stays under an explicit cap no matter how many layers fan out above
it.

The gate is applied where clients are constructed: for the same reason as
the wall-clock guard, :func:`exp_graph.llm.factory.create_llm_client` wraps
every real client it builds (code that builds its own client through the
factory is limited too, not only a client that a caller passes in), and the
planner client chain of :mod:`queenbee.program.clients` applies the same
limiter.

The limiter is per process.  A sandboxed team program runs in a child
process whose environment is rebuilt from an allow-list without
``EXP_GRAPH_LLM_MAX_CONCURRENCY``, so its worker calls are not counted by
the parent's limiter; within one program they are bounded by
``max_parallel_agents``.

Wrapper order: the limiter must sit outside the wall-clock timeout guard.
``TimeoutLLMClient`` abandons a hung call on a daemon thread; if the limiter
were inside the guard, the abandoned thread would hold its slot forever and
leak the semaphore toward deadlock.  Outside the guard, a timed-out call
releases its slot immediately (the abandoned daemon may still drain in the
background: a bounded, rare overshoot of the cap, never a leak).  Retry
wrappers go outside the limiter so backoff sleeps never hold a slot.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from exp_graph.llm.base import LLMResponse

#: Env var holding the process-wide cap on concurrent in-flight real-LLM
#: calls.  Unset, empty, unparsable or not positive disables limiting.
#: Drivers may instead call :func:`configure_global_llm_concurrency`
#: explicitly.
LLM_MAX_CONCURRENCY_ENV = "EXP_GRAPH_LLM_MAX_CONCURRENCY"


class LLMConcurrencyLimiter:
    """Shared gate (a bounded semaphore) bounding concurrent in-flight LLM
    calls.

    ``in_flight`` / ``peak_in_flight`` are observability counters so a caller
    can assert after a run that its configured ceiling was actually respected
    (and actually exercised) instead of trusting the wiring blindly.
    """

    def __init__(self, max_concurrent: int) -> None:
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent < 1
        ):
            raise ValueError("max_concurrent must be a positive integer")
        self._max_concurrent = max_concurrent
        self._semaphore = threading.BoundedSemaphore(max_concurrent)
        self._counter_lock = threading.Lock()
        self._in_flight = 0
        self._peak_in_flight = 0

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def in_flight(self) -> int:
        with self._counter_lock:
            return self._in_flight

    @property
    def peak_in_flight(self) -> int:
        with self._counter_lock:
            return self._peak_in_flight

    def acquire(self) -> None:
        self._semaphore.acquire()
        with self._counter_lock:
            self._in_flight += 1
            if self._in_flight > self._peak_in_flight:
                self._peak_in_flight = self._in_flight

    def release(self) -> None:
        with self._counter_lock:
            self._in_flight -= 1
        self._semaphore.release()

    def __enter__(self) -> "LLMConcurrencyLimiter":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class ConcurrencyLimitedLLMClient:
    """Wrap an ``LLMClient`` so every ``complete`` holds one limiter slot.

    Arguments pass through ``*args`` / ``**kwargs`` untouched, so the wrapper
    works with any inner ``complete`` signature and keyword arguments such as
    ``json_mode`` reach the inner client unchanged.
    """

    def __init__(self, inner: Any, limiter: LLMConcurrencyLimiter) -> None:
        if not isinstance(limiter, LLMConcurrencyLimiter):
            raise TypeError("limiter must be an LLMConcurrencyLimiter")
        self._inner = inner
        self._limiter = limiter

    @property
    def limiter(self) -> LLMConcurrencyLimiter:
        return self._limiter

    def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        with self._limiter:
            return self._inner.complete(*args, **kwargs)


_GLOBAL_LOCK = threading.Lock()
_GLOBAL_LIMITER: LLMConcurrencyLimiter | None = None
_GLOBAL_CONFIGURED = False


def configure_global_llm_concurrency(
    max_concurrent: int | None,
) -> LLMConcurrencyLimiter | None:
    """Set (or disable, with ``None``) the process-wide limiter explicitly.

    An explicit configuration wins over the environment variable and applies
    to clients constructed after the call; already-wrapped clients keep the
    limiter they were built with.  Returns the active limiter.
    """

    global _GLOBAL_LIMITER, _GLOBAL_CONFIGURED
    with _GLOBAL_LOCK:
        _GLOBAL_LIMITER = (
            None if max_concurrent is None else LLMConcurrencyLimiter(max_concurrent)
        )
        _GLOBAL_CONFIGURED = True
        return _GLOBAL_LIMITER


def global_llm_limiter() -> LLMConcurrencyLimiter | None:
    """Return the active process-wide limiter, if any.

    Explicit configuration wins; otherwise the limiter is built lazily from
    ``LLM_MAX_CONCURRENCY_ENV`` (unset, empty, unparsable or not positive:
    no limiting).
    The env read happens once; later env changes need an explicit
    :func:`configure_global_llm_concurrency`.
    """

    global _GLOBAL_LIMITER, _GLOBAL_CONFIGURED
    with _GLOBAL_LOCK:
        if _GLOBAL_CONFIGURED:
            return _GLOBAL_LIMITER
        raw = os.environ.get(LLM_MAX_CONCURRENCY_ENV, "").strip()
        try:
            value = int(raw) if raw else 0
        except ValueError:
            value = 0
        _GLOBAL_LIMITER = LLMConcurrencyLimiter(value) if value > 0 else None
        _GLOBAL_CONFIGURED = True
        return _GLOBAL_LIMITER


def maybe_limit_concurrency(client: Any) -> Any:
    """Wrap ``client`` in the global limiter when one is active.

    Returns ``client`` itself when limiting is disabled, so the default
    (unconfigured) path adds no wrapper at all.
    """

    limiter = global_llm_limiter()
    if limiter is None:
        return client
    return ConcurrencyLimitedLLMClient(client, limiter)


__all__ = [
    "LLM_MAX_CONCURRENCY_ENV",
    "ConcurrencyLimitedLLMClient",
    "LLMConcurrencyLimiter",
    "configure_global_llm_concurrency",
    "global_llm_limiter",
    "maybe_limit_concurrency",
]
