"""Which worker infra rows count as a service outage
(:func:`queenbee.evo.loop.is_connectivity_row`).

Refused connections, HTTP 502 / 503 and name-resolution failures do;
rejected requests (401 / 403), calls that ran over their time limit and rows
without an error do not.
"""

from queenbee.evo.loop import is_connectivity_row


def test_unreachable_service_is_connectivity() -> None:
    assert is_connectivity_row({"infra": "APIConnectionError: Connection refused"})
    assert is_connectivity_row({"infra": "RuntimeError: Error code: 502 - Bad Gateway"})
    assert is_connectivity_row({"infra": "RuntimeError: Error code: 503 - Service Unavailable"})
    assert is_connectivity_row({"error": "ConnectError: [Errno 8] nodename nor servname provided, "
                                         "or not known (name resolution)"})


def test_rejected_or_slow_requests_are_not_connectivity() -> None:
    assert not is_connectivity_row({"infra": "PermissionDeniedError: Error code: 403 - Forbidden"})
    assert not is_connectivity_row({"infra": "AuthenticationError: Error code: 401"})
    assert not is_connectivity_row({"infra": "LLM call exceeded 1800s"})
    assert not is_connectivity_row({"infra": None, "error": None})
