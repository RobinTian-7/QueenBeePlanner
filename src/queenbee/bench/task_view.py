"""Public task view: the ONLY task dict shape allowed into model context.

The global task built by :class:`queenbee.bench.task_bridge.BenchmarkTaskAdapter`
carries a private scoring payload (the answer key and per-agent expected
outputs) that scorers need and models must never see. A denylist of top-level
keys would miss nested leaks such as ``meta.expected_outputs``, so this module
uses an explicit ALLOWLIST:

* :func:`public_task_view` keeps only known-safe top-level fields, each as a
  scalar, and drops everything else, so any field it does not name (nested
  ones included) is private by default.
* :func:`private_scoring_payload` is the single accessor scorers use to reach
  the private payload (kept under ``PRIVATE_SCORING_KEY`` in the global task).
* ``task_ref`` replaces ``case_id`` in anything model-visible: the raw
  Silo-Bench case id (e.g. ``II-12``) indexes a public benchmark file that
  contains the answers, so model context gets only an opaque digest of it.
"""

from __future__ import annotations

import hashlib
from typing import Any

# Reserved key for the private scoring payload inside the global task. The key
# is outside the public_task_view allowlist, and worker prompts are rendered
# from explicitly chosen fields (never from the whole global task), so the
# payload cannot reach a prompt.
PRIVATE_SCORING_KEY = "_private_scoring"

# Allowlisted top-level fields for model context. Deliberately excludes meta,
# shards, case_id, and every scoring field.
_PUBLIC_TOP_LEVEL: tuple[str, ...] = (
    "task_family",
    "benchmark",
    "task_ref",
    "case_name",
    "n_agents",
    "output_type",
    "segmented",
)


def task_ref(case_id: str) -> str:
    """Opaque, stable reference for a case id (safe for model context): the
    first 12 hex digits of its sha256."""
    return hashlib.sha256(str(case_id).encode("utf-8")).hexdigest()[:12]


def public_task_view(global_task: dict[str, Any]) -> dict[str, Any]:
    """Allowlisted, nested-safe view of a global task for model context.

    Keeps the ``_PUBLIC_TOP_LEVEL`` fields that are set (never meta, shards,
    case_id or an answer field) and derives ``task_ref`` from ``case_id`` when
    the task carries none.
    """
    view: dict[str, Any] = {}
    for key in _PUBLIC_TOP_LEVEL:
        if key in global_task and global_task[key] is not None:
            view[key] = _public_scalar(global_task[key])
    if "task_ref" not in view and global_task.get("case_id") is not None:
        view["task_ref"] = task_ref(str(global_task["case_id"]))
    return view


# Allowlisted values pass through only as scalars or strings: a container
# value is never forwarded as a structure.
def _public_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Non-scalars collapse to a truncated string: allowlisted fields are not
    # supposed to be containers in the first place.
    return str(value)[:200]


def private_scoring_payload(global_task: dict[str, Any]) -> dict[str, Any]:
    """Scorer-only accessor for the private payload (empty dict if absent)."""
    payload = global_task.get(PRIVATE_SCORING_KEY)
    return payload if isinstance(payload, dict) else {}
