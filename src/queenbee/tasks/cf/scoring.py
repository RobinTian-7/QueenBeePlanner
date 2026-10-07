"""Count-Frequency scoring: agent 0's count table against the gold table.

RMSE = :func:`queenbee.tasks.cf.cases.compute_rmse` over the value domain
``"0"`` .. ``"255"`` (square root of the SUMMED squared count errors; a
value missing from the answer counts 0), and ``S = exp(-RMSE / 5)``.  An
answer that is not a count table (:func:`is_count_table`: a JSON object
with at least one key of the value domain carrying a number), or a broken
run, scores ``S = 0`` and its row is flagged ``format_fail`` (``cf_rmse``
None).

Answers and gold tables given as JSON text are parsed with
:func:`exp_graph.mas.count_table_json.tolerant_loads`: a flat count table
whose closing brace is missing (or replaced by ``)`` / ``]``) is repaired,
any other malformed text is rejected.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any, Sequence

from exp_graph.mas.count_table_json import tolerant_loads
from queenbee.tasks.cf.cases import VALUE_MAX, VALUE_MIN, canonicalize_counts, compute_rmse

#: The value domain, as the string keys of a count table.
CF_DOMAIN: list[str] = [str(v) for v in range(VALUE_MIN, VALUE_MAX + 1)]
_DOMAIN_SET = frozenset(CF_DOMAIN)
#: S = exp(-RMSE / RMSE_SCALE).
RMSE_SCALE = 5.0


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            num = float(value.strip())
        except ValueError:
            return None
        if not math.isfinite(num):
            return None
        return int(num) if num.is_integer() else num
    return None


def parse_count_table(value: Any) -> dict[str, int | float] | None:
    """An answer as a count table (string keys, numeric values), or None
    when it is not a JSON object (JSON text is parsed first).  Entries whose
    value is not a number are dropped."""

    if isinstance(value, (str, bytes)):
        try:
            value = tolerant_loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, Mapping):
        return None
    out: dict[str, int | float] = {}
    for key, raw in value.items():
        num = _number(raw)
        if num is not None:
            out[str(key)] = num
    return out


def gold_table(gold: Any) -> dict[str, int] | None:
    """The gold count table (a mapping or its JSON text), or None."""

    if isinstance(gold, (str, bytes)):
        try:
            gold = tolerant_loads(gold)
        except (TypeError, ValueError):
            return None
    if not isinstance(gold, Mapping):
        return None
    return {str(k): int(v) for k, v in gold.items()}


def cf_rmse(pred: Any, gold: Any) -> float | None:
    """RMSE of ``pred`` against ``gold`` over :data:`CF_DOMAIN`; None when
    either is not a count object."""

    table = parse_count_table(pred)
    truth = gold_table(gold)
    if table is None or truth is None:
        return None
    return float(compute_rmse(table, truth, CF_DOMAIN))


def score_from_rmse(rmse: float | None) -> float:
    """``S = exp(-RMSE/5)``; None (no parseable answer) or a non-finite or
    negative RMSE -> 0.0."""

    if rmse is None:
        return 0.0
    try:
        value = float(rmse)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value < 0:
        return 0.0
    return math.exp(-value / RMSE_SCALE)


def table_stats(pred: Any, gold: Any) -> dict[str, int] | None:
    """``{sum: total count over the domain, wrong: domain values with a
    wrong count}``; None when either is not a count object."""

    table = parse_count_table(pred)
    truth = gold_table(gold)
    if table is None or truth is None:
        return None
    p, t = canonicalize_counts(table), canonicalize_counts(truth)
    return {"sum": int(sum(p.get(k, 0) for k in CF_DOMAIN)),
            "wrong": int(sum(1 for k in CF_DOMAIN if p.get(k, 0) != t.get(k, 0)))}


def is_count_table(value: Any) -> bool:
    """True when ``value`` parses as a count table with at least one key of
    the value domain carrying a number (``{}`` and wrapped objects such as
    ``{"counts": {...}}`` are not count tables)."""

    table = parse_count_table(value)
    return bool(table) and any(key in _DOMAIN_SET for key in table)


def cf_apply(facts: Mapping[str, Any], *, submissions: Any, sink_id: int,
             ground_truth: Any) -> dict[str, Any]:
    """``facts`` re-scored from the sink's submitted table: ``S``,
    ``stage_score``, ``success`` and ``cf_rmse`` / ``cf_sum`` /
    ``cf_wrong_bins`` / ``cf_format_fail``."""

    answer = None
    for sub in submissions or []:
        d = sub if isinstance(sub, dict) else sub.model_dump()
        try:
            if int(d.get("agent_id", -1)) == int(sink_id):
                answer = d.get("answer")
                break
        except (TypeError, ValueError):
            continue
    out = dict(facts)
    rmse = cf_rmse(answer, ground_truth) if is_count_table(answer) else None
    if rmse is None:
        out.update({"S": 0.0, "stage_score": 0.0, "success": False, "cf_rmse": None,
                    "cf_sum": None, "cf_wrong_bins": None, "cf_format_fail": True})
        return out
    stats = table_stats(answer, ground_truth) or {}
    s = score_from_rmse(rmse)
    out.update({"S": round(s, 6), "stage_score": round(s, 6), "success": rmse <= 0.0,
                "cf_rmse": rmse, "cf_sum": stats.get("sum"), "cf_wrong_bins": stats.get("wrong"),
                "cf_format_fail": False})
    return out


def score_fn(facts: Mapping[str, Any], *, instance: Any, output: Any, payload: Mapping[str, Any],
             score: Any = None) -> dict[str, Any]:
    """The task's ``score_fn``: :func:`cf_apply` on a completed run."""

    return cf_apply(facts, submissions=getattr(output, "submissions", None),
                    sink_id=int(payload.get("selected_primary", 0)),
                    ground_truth=getattr(instance, "ground_truth", None))


def finalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Every row carries ``cf_rmse`` (None unless scored), ``format_fail``
    and ``submitted_total``; a scored row without an RMSE scores 0 and its
    card says ``format`` (or ``budget``)."""

    from queenbee.evo.diagnosis import _error_class

    out = dict(row)
    if out.get("infra"):
        out.setdefault("cf_rmse", None)
        out["format_fail"] = False
        out.setdefault("submitted_total", None)
        return out
    out.setdefault("cf_rmse", None)
    out["format_fail"] = out.get("cf_rmse") is None
    if out["format_fail"]:
        out["S"] = 0.0
        out.setdefault("stage_score", 0.0)
        out["success"] = False
        diag = out.get("diag")
        if not (isinstance(diag, Mapping) and diag.get("failure_class") in ("format", "budget")):
            err = _error_class(out) or "format"
            out["diag"] = dict(diag or {}) | {"cf": True, "S": 0.0, "answer_shape": "collection",
                                              "failure_class": err if err in ("budget",) else "format",
                                              "error_class": err}
    out["submitted_total"] = out.get("cf_sum")
    return out


def summary_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluator summary of scored rows: ``mean_rmse`` over the rows with an
    RMSE (``rmse_rows`` of them) and ``format_fail`` = scored rows whose
    answer was not a count table (or whose run broke)."""

    scored = [r for r in rows if not r.get("infra")]
    rmses = [float(r["cf_rmse"]) for r in scored if r.get("cf_rmse") is not None]
    return {
        "mean_rmse": statistics.fmean(rmses) if rmses else None,
        "rmse_rows": len(rmses),
        "format_fail": len(scored) - len(rmses),
    }


__all__ = [
    "CF_DOMAIN",
    "RMSE_SCALE",
    "cf_apply",
    "cf_rmse",
    "finalize_row",
    "gold_table",
    "is_count_table",
    "parse_count_table",
    "score_fn",
    "score_from_rmse",
    "summary_fields",
    "table_stats",
]
