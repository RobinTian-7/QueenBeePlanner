"""Infrastructure failures become infra rows (re-run later), never algorithm
failures scored S = 0.

Connection drops and read timeouts, rate limiting (HTTP 429), server-side
failures (HTTP 502 / 503, upstream and deadline errors), a transient "model
not found" reply and the wall-clock guard's ``LLMTimeoutError`` are
infrastructure. The guard's text ("LLM call exceeded <t>s") names no
timeout, so it needs its own markers. Genuine program failures (budget,
data-flow, answer-format and syntax errors) must stay scored failures. The
planner-side classifier ``_is_transport_error`` also matches exception type
names (``APIConnectionError``, ``InternalServerError``).
"""
from __future__ import annotations

import pytest

from queenbee.program.execute import (
    _failure_facts,
    _is_transport_error,
    classify_row_error,
)


def _facts(error_type: str, message: str) -> dict:
    return _failure_facts({"error_type": error_type, "message": message})


@pytest.mark.parametrize(
    "error_type, message",
    [
        ("RuntimeError", "LLM call exceeded 2400.0s"),
        ("LLMTimeoutError", "LLM call exceeded 90.0s"),
        ("APIError", "Error code: 503 - upstream service temporarily unavailable"),
        ("APIError", 'Error code: 504 - {"error": {"type": "deadline_exceeded"}}'),
        ("APIError", 'Error code: 500 - {"error": {"type": "upstream_error"}}'),
        ("APIError", "Error code: 502 - upstream failed after 3 attempts"),
        ("RateLimitError", "Error code: 429 - too many requests"),
        ("BadRequestError", "Error code: 400 - model not found on this replica"),
        ("RemoteProtocolError", "incomplete chunked read"),
        ("RuntimeError", "peer closed connection without sending complete message body"),
        ("RuntimeError", "[Errno 54] Connection reset by peer"),
        ("RuntimeError", "The read operation timed out"),
    ],
)
def test_transport_and_gateway_failures_are_infra(error_type, message):
    facts = _facts(error_type, message)
    assert facts == {"infra": f"{error_type}: {message}"}
    assert classify_row_error(facts) == "infra"


@pytest.mark.parametrize(
    "error_type, message, row_class",
    [
        ("BudgetError", "completion token budget exceeded", "budget"),
        ("DataFlowError", "malformed state or inbox prompt metadata", "exec"),
        ("AnswerFormatError", "final answer is not a JSON object", "format"),
        ("RuntimeError", "generated program exited non-zero", "exec"),
        ("SyntaxError", "invalid syntax (program.py, line 3)", "exec"),
        ("DataFlowError", "12:4: undefined name 'peers'", "undefined_name"),
    ],
)
def test_genuine_program_failures_stay_algorithm_failures(error_type, message, row_class):
    facts = _facts(error_type, message)
    assert facts["infra"] is None
    assert facts["execution_class"] == "algorithm_failure"
    assert facts["S"] == 0.0 and facts["C"] == 0.0 and facts["success"] is False
    assert facts["error"] == f"{error_type}: {message}"
    assert classify_row_error(facts) == row_class


def test_failure_text_is_clipped_and_missing_fields_are_tolerated():
    long = _facts("RuntimeError", "x" * 2000)
    assert len(long["error"]) == 500
    assert _failure_facts(None)["error"] == "None: None"
    assert classify_row_error({"S": 1.0}) is None
    assert classify_row_error("not a row") is None


class APIConnectionError(Exception):
    """Name-matched like openai.APIConnectionError."""


class InternalServerError(Exception):
    """Name-matched like openai.InternalServerError."""


@pytest.mark.parametrize(
    "exc, transport",
    [
        (APIConnectionError("no route to host"), True),
        (InternalServerError("boom"), True),
        (RuntimeError("Error code: 503"), True),
        (TimeoutError("timed out"), True),
        (RuntimeError("LLM call exceeded 30.0s"), True),
        (ValueError("bad request: unknown parameter"), False),
        (KeyError("missing field"), False),
    ],
)
def test_planner_transport_classifier(exc, transport):
    assert _is_transport_error(exc) is transport
