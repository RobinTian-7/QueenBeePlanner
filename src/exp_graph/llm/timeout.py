"""Per-request hard wall-clock timeout for LLM calls.

An HTTP client's read timeout bounds the gap between received bytes, not the
whole request: an endpoint or intermediary that keeps trickling keep-alive
bytes can hold a stalled call open indefinitely, and with it the whole run.
``TimeoutLLMClient`` adds a *hard* wall-clock budget per ``complete`` call so
a hung request fails fast instead.

The guard is applied at client construction by
:func:`exp_graph.llm.factory.create_llm_client` (every non-fake client, when a
budget is configured), so the client a sandboxed team program builds is
guarded as well as the clients a caller wraps explicitly (e.g. the planner
client chain of ``queenbee.program.clients``).

Why a bare daemon thread (not ``concurrent.futures``): a Python thread cannot be
forcibly killed, so the only way to *abandon* a stuck blocking call is to run it
on a ``daemon=True`` thread and stop waiting via ``join(timeout)``. After a
timeout the abandoned thread keeps running in the background but, being a
daemon, does not block interpreter exit. A ``ThreadPoolExecutor`` used as a
context manager (or ``shutdown(wait=True)``) would block on the hung worker at
exit and recreate the very freeze this guard prevents, so it is deliberately
avoided here.
"""

from __future__ import annotations

import threading

from exp_graph.llm.base import LLMClient, LLMResponse


class LLMTimeoutError(RuntimeError):
    """Raised when a wrapped LLM ``complete`` call exceeds its time budget."""


class TimeoutLLMClient:
    """Wrap an inner ``LLMClient`` with a per-request hard wall-clock timeout.

    ``complete`` runs the inner call on a daemon thread and waits at most
    ``timeout_s`` seconds for it. If the thread is still alive after the join the
    call is abandoned (the daemon thread keeps running but cannot block process
    exit) and :class:`LLMTimeoutError` is raised. If the inner call finished, its
    return value is passed through and any exception it raised is re-raised
    unchanged. A non-positive or ``None`` ``timeout_s`` disables the guard:
    calls go straight to the inner client, with no thread.
    """

    def __init__(self, inner: LLMClient, timeout_s: float | None) -> None:
        self._inner = inner
        self._timeout_s = timeout_s

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        # Extra keywords (e.g. json_mode) are forwarded only when the caller
        # passed them, so an inner client whose ``complete`` does not accept
        # them still works on default calls (the same pass-through as the
        # retry and concurrency wrappers).
        # No budget -> behave exactly like the inner client (no thread, no guard).
        if self._timeout_s is None or self._timeout_s <= 0:
            return self._inner.complete(prompt, model_name, temperature, **kwargs)

        # Holder for the worker's outcome. Lists are mutated in place so the
        # calling thread reads whatever the daemon thread stored before joining.
        result: list[LLMResponse] = []
        error: list[BaseException] = []

        def _run() -> None:
            try:
                result.append(
                    self._inner.complete(
                        prompt, model_name, temperature, **kwargs
                    )
                )
            except BaseException as exc:  # noqa: BLE001 - preserved + re-raised below
                error.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(self._timeout_s)

        if thread.is_alive():
            # Hung call: abandon the daemon thread and fail fast. Never join
            # without a timeout here: that would block on the hung call again.
            raise LLMTimeoutError(f"LLM call exceeded {self._timeout_s}s")

        if error:
            raise error[0]
        return result[0]
