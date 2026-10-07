"""Opt-in tolerance for a count table whose closing brace is missing.

Some worker models, asked to submit a large table of counts as one JSON
object, drop the final ``}`` or write ``)`` / ``]`` in its place; the strict
single-JSON submit parser rejects that answer, and when the one
format-repair call repeats the mistake the run fails.  A task that needs it
asks the worker sandbox for this repair by setting :data:`REPAIR_ENV` to
``"1"`` (``python_worker_bootstrap`` reads it when it starts; unset, nothing
changes).  The host process never installs it; a task's scorer may call
:func:`tolerant_loads` directly.

Only that one shape is accepted: when the strict parse fails and the whole
text is a flat object of ``"<key>": <count>`` pairs (see ``_BROKEN_TABLE``)
whose closing brace is missing or replaced by ``)`` / ``]``, the brace is
restored and the text is parsed again.  The repaired value is exactly the
object that was written.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping

#: Environment variable of the worker sandbox: ``"1"`` installs
#: :func:`tolerant_loads` as ``json.loads`` in that process.
REPAIR_ENV = "QUEENBEE_REPAIR_COUNT_TABLE_JSON"

# One or more ``"<key>": <count>`` pairs (keys of one to three digits,
# non-negative integer counts) after ``{``, ending in ``)``, ``]`` or nothing.
_PAIRS = r'\s*"\d{1,3}"\s*:\s*\d+\s*'
_BROKEN_TABLE = re.compile(r'^\{(?:%s,)*%s[)\]]?\s*$' % (_PAIRS, _PAIRS))
# Captured at import, so tolerant_loads still reaches the strict parser after
# it has replaced json.loads.
_STRICT_LOADS = json.loads


def repair_count_table(text: Any) -> str | None:
    """The repaired JSON text of a brace-broken count table, else None."""

    if not isinstance(text, str):
        return None
    t = text.strip()
    if not t.startswith("{") or t.endswith("}") or not _BROKEN_TABLE.match(t):
        return None
    if t[-1] in ")]":
        t = t[:-1].rstrip()
    return t + "}"


def tolerant_loads(s: Any, *args: Any, **kwargs: Any) -> Any:
    """``json.loads`` that also accepts a brace-broken count table."""

    try:
        return _STRICT_LOADS(s, *args, **kwargs)
    except ValueError:
        fixed = repair_count_table(s.decode() if isinstance(s, (bytes, bytearray)) else s)
        if fixed is None:
            raise
        return _STRICT_LOADS(fixed, *args, **kwargs)


def install_if_requested(environ: Mapping[str, str] | None = None) -> bool:
    """Install :func:`tolerant_loads` as ``json.loads`` of this process when
    :data:`REPAIR_ENV` is ``"1"``; returns whether it is installed."""

    env = os.environ if environ is None else environ
    if env.get(REPAIR_ENV) != "1":
        return False
    if json.loads is not tolerant_loads:
        json.loads = tolerant_loads
    return True


__all__ = ["REPAIR_ENV", "install_if_requested", "repair_count_table", "tolerant_loads"]
