"""Count-Frequency diagnosis cards (answer-free: only error sizes leave).

A CF card keeps the runtime fields of the built-in card (submit round,
messages, lost bodies, tokens per phase) and replaces its per-agent part:
``cf_rmse`` (agent 0's error), ``cf_sum_dev`` (its counts' sum minus the
array size), ``cf_wrong_bins`` (values with a wrong count), the local error
of the single-shard tables found in message bodies (``cf_local_err`` over
``cf_local_k`` shards, ``cf_local_est`` scaled to all shards,
``cf_merge_excess`` = RMSE - local estimate), the messages' coverage
bookkeeping (``cf_msgs_tables``, ``cf_labels_bad``, ``cf_cover_bad``,
``cf_submit_added``) and a failure class from the shared vocabulary with
the meaning :data:`CF_GLOSSARY` gives it.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from exp_graph.mas.count_table_json import tolerant_loads
from queenbee.tasks.cf.cases import ARRAY_SIZE, BENCHMARK, N_AGENTS
from queenbee.tasks.cf.scoring import CF_DOMAIN, _number, parse_count_table

#: Failure classes of the shared vocabulary, with their meaning on this task.
CF_GLOSSARY: dict[str, str] = {
    "ok": "agent 0 submitted the exact count table",
    "format": "agent 0's answer was not a JSON object of counts, or the run broke "
    "(program error), and scored 0",
    "content-loss": "whole shards are MISSING from agent 0's table (its counts sum "
    "far BELOW the array size), or message bodies were delivered to agents that were "
    "not called that round and were lost",
    "scattered-wrong": "local counting error: the per-shard counts the agents "
    "produced from their own data were already wrong before any combining (most "
    "of the error is in the single-shard counts)",
    "consensus-wrong": "combining error: the per-shard counts were mostly right, but "
    "combining them on the way to agent 0 added error (entries dropped, mis-added, or "
    "a whole shard counted TWICE so the counts sum far ABOVE the array size; see "
    "labels_bad / cover_bad / submit_added on the run line)",
}

_PAIR_RE = re.compile(r'(?<![\w.])"?(\d{1,3})"?\s*[:=]\s*(-?\d+(?:\.\d+)?)(?![\w.])')
_TOTAL_RE = re.compile(r"TOTAL\s*[=:]\s*(\d+)", re.IGNORECASE)


def _dump(output: Any) -> dict[str, Any] | None:
    if output is None:
        return None
    if isinstance(output, dict):
        return output
    try:
        return output.model_dump(mode="json")
    except Exception:  # noqa: BLE001
        return None


def _balanced_objects(text: str, limit: int = 8) -> list[str]:
    out: list[str] = []
    start = text.find("{")
    while start != -1 and len(out) < limit:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[start:i + 1])
                    break
        else:
            break
        start = text.find("{", start + 1)
    return out


def parse_body_table(body: Any) -> dict[str, int] | None:
    """A count table inside a free-text message body (a JSON object, one
    level of nesting, or ``v: c`` / ``v=c`` pairs); keys restricted to the
    value domain, the object with the most such keys winning; None when
    there is none."""

    text = str(body or "")
    candidates: list[dict] = []
    try:
        whole = tolerant_loads(text)
        if isinstance(whole, Mapping):
            candidates.append(dict(whole))
    except ValueError:
        pass
    for blob in _balanced_objects(text):
        try:
            obj = tolerant_loads(blob)
        except ValueError:
            continue
        if isinstance(obj, Mapping):
            candidates.append(dict(obj))
            for value in obj.values():  # one level of nesting ({"counts": {...}})
                if isinstance(value, Mapping):
                    candidates.append(dict(value))
    domain = set(CF_DOMAIN)
    best: dict[str, int] | None = None
    for cand in candidates:
        table = {str(k).strip(): int(v) for k, v in (parse_count_table(cand) or {}).items()
                 if str(k).strip() in domain}
        if table and (best is None or len(table) > len(best)):
            best = table
    if best:
        return best
    pairs: dict[str, int] = {}
    for key, value in _PAIR_RE.findall(text):
        if key in domain and key not in pairs:
            num = _number(value)
            if num is not None:
                pairs[key] = int(num)
    return pairs or None


def single_source_tables(output: Any) -> dict[int, dict[str, int]]:
    """Agent -> the count table of its OWN shard, from the earliest message it
    sent that carries only its own source id and a parseable table."""

    data = _dump(output) or {}
    messages = sorted((m for m in data.get("messages") or [] if isinstance(m, dict)),
                      key=lambda m: (int(m.get("round_sent") or 0), int(m.get("src") or 0)))
    out: dict[int, dict[str, int]] = {}
    for m in messages:
        try:
            src = int(m.get("src"))
            ids = sorted(int(x) for x in (m.get("source_ids") or []))
        except (TypeError, ValueError):
            continue
        if src in out or ids != [src]:
            continue
        table = parse_body_table(m.get("body"))
        if table:
            out[src] = table
    return out


def local_error(output: Any, instance: Any) -> tuple[float | None, int]:
    """``(sqrt of the summed squared errors of the observed single-shard
    tables against their shards' true counts, shards observed)``."""

    shards = list(getattr(instance, "shards", None) or [])
    tables = single_source_tables(output)
    total, k = 0.0, 0
    for src, table in tables.items():
        if not 0 <= src < len(shards):
            continue
        truth = Counter(str(int(v)) for v in shards[src])
        total += sum((float(table.get(key, 0)) - float(truth.get(key, 0))) ** 2 for key in CF_DOMAIN)
        k += 1
    return (math.sqrt(total) if k else None), k


def coverage_bookkeeping(output: Any, instance: Any, submitted_total: Any) -> dict[str, Any]:
    """Answer-free size accounting of the messages.

    ``cf_msgs_tables`` counts the bodies holding a count table; the sum of
    each such table is compared with the body's last TOTAL label, when it has
    one (``cf_labels_bad``: off by more than 2), and with the true number of
    items in the shards its source ids name (``cf_cover_bad``: off by more
    than 16).  ``cf_submit_added`` is agent 0's submitted total minus (the
    sums of the tables it received + the size of its own shard); it is set
    only when every message to agent 0 holds a table and the submitted total
    is a number."""

    shards = list(getattr(instance, "shards", None) or [])
    data = _dump(output) or {}
    msgs = [m for m in data.get("messages") or [] if isinstance(m, dict)]
    with_table = labels_bad = cover_bad = 0
    into0 = 0
    into0_ok = True
    for m in msgs:
        table = parse_body_table(m.get("body"))
        try:
            ids = sorted({int(x) for x in (m.get("source_ids") or [])})
            dst = int(m.get("dst"))
        except (TypeError, ValueError):
            continue
        if not table:
            if dst == 0:
                into0_ok = False
            continue
        with_table += 1
        tsum = sum(int(v) for v in table.values())
        true_items = sum(len(shards[i]) for i in ids if 0 <= i < len(shards)) if shards else None
        totals = _TOTAL_RE.findall(str(m.get("body") or ""))
        if totals and abs(int(totals[-1]) - tsum) > 2:
            labels_bad += 1
        if true_items is not None and abs(tsum - true_items) > 16:
            cover_bad += 1
        if dst == 0:
            into0 += tsum
    out: dict[str, Any] = {"cf_msgs_tables": with_table, "cf_labels_bad": labels_bad,
                           "cf_cover_bad": cover_bad}
    if into0_ok and shards and isinstance(submitted_total, (int, float)):
        out["cf_submit_added"] = int(submitted_total) - (into0 + len(shards[0]))
    return out


def is_cf_instance(instance: Any) -> bool:
    """A Count-Frequency instance: the CF benchmark name or a ``cf_seed`` in
    its meta."""
    meta = getattr(instance, "meta", None) or {}
    return getattr(instance, "benchmark", "") == BENCHMARK or (
        isinstance(meta, Mapping) and meta.get("cf_seed") is not None)


def cf_diag(facts: Mapping[str, Any], output: Any, instance: Any,
            base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The CF card of one run (module docstring) from its row facts, its
    output and the built-in card ``base``."""

    from queenbee.evo.diagnosis import _error_class

    base = dict(base or {})
    d: dict[str, Any] = {k: base.get(k) for k in (
        "phase_tokens", "submit_round", "n_messages", "n_lost_messages")}
    s = facts.get("S")
    d.update({"S": float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else None,
              "agent_correct": None, "n_distinct_answers": None, "majority_correct": None,
              "answer_shape": "collection", "cf": True})
    if facts.get("infra"):
        d["failure_class"] = "infra"
        return d
    rmse = facts.get("cf_rmse")
    if rmse is None:
        err = _error_class(facts) or "format"
        d["failure_class"] = err if err in ("budget", "infra") else "format"
        d["error_class"] = err
        return d
    rmse = float(rmse)
    shards = list(getattr(instance, "shards", None) or [])
    n = len(shards) or N_AGENTS
    size = sum(len(s) for s in shards) or ARRAY_SIZE
    total = facts.get("cf_sum")
    sum_dev = int(total) - int(size) if isinstance(total, (int, float)) else None
    err_obs, k = local_error(output, instance)
    est = err_obs * math.sqrt(n / k) if err_obs is not None and k else None
    d.update({"cf_rmse": rmse, "cf_sum_dev": sum_dev, "cf_wrong_bins": facts.get("cf_wrong_bins"),
              "cf_local_err": err_obs, "cf_local_k": k, "cf_local_est": est,
              "cf_merge_excess": (rmse - est) if est is not None else None})
    d.update(coverage_bookkeeping(output, instance, total))
    lost = int(d.get("n_lost_messages") or 0)
    half_shard = (size // n) // 2
    if rmse <= 0.0:
        d["failure_class"] = "ok"
    elif (sum_dev is not None and sum_dev <= -half_shard) or lost > 0:
        d["failure_class"] = "content-loss"
    elif sum_dev is not None and sum_dev >= half_shard:
        # an over-count is a combining error (a shard counted twice), not a loss
        d["failure_class"] = "consensus-wrong"
    elif est is not None and est * est >= 0.5 * rmse * rmse:
        d["failure_class"] = "scattered-wrong"
    else:
        d["failure_class"] = "consensus-wrong"
    return d


def diag_fn(output: Any, instance: Any, facts: Mapping[str, Any], **kwargs: Any) -> Any:
    """The task's ``diag_fn``: on a CF instance the CF card built on the
    built-in card (built without it when the built-in card fails; the
    built-in card is returned when the CF card cannot be built); any other
    instance gets the built-in card."""

    from queenbee.evo.diagnosis import diagnose_execution

    if not is_cf_instance(instance):
        return diagnose_execution(output, instance, facts, **kwargs)
    try:
        base = diagnose_execution(output, instance, facts, **kwargs)
    except Exception:  # noqa: BLE001 - the CF card is then built without it
        base = None
    try:
        return cf_diag(facts, output, instance, base=base)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        if base is None:
            raise
        return base


def trace_row(row: dict[str, Any], trace: Mapping[str, Any], instance: Any, *,
              source: str | None = None, goal: str | None = None) -> dict[str, Any]:
    """The task's ``trace_row`` hook: on a CF instance, a trace row whose
    card is not a CF card gets the CF card built from the trace's facts and
    output (left unchanged when that fails)."""

    if is_cf_instance(instance):
        diag = row.get("diag")
        if not (isinstance(diag, Mapping) and diag.get("cf")):
            facts = dict(trace.get("facts") or {})
            try:
                row["diag"] = cf_diag(facts, trace.get("output"), instance,
                                      base=diag if isinstance(diag, Mapping) else None)
            except Exception:  # noqa: BLE001 - enrichment is advisory
                pass
    return row


def _fin(value: Any, lo: float, hi: float, nd: int = 3) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not lo <= value <= hi:
        return None
    return round(value, nd)


def _int_in(value: Any, lo: int, hi: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value) if lo <= value <= hi else None


def sanitize_cf_fields(diag: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelisted, type-checked CF card fields (numbers only)."""

    out = {
        "cf_rmse": _fin(diag.get("cf_rmse"), 0.0, 1e5),
        "cf_sum_dev": _int_in(diag.get("cf_sum_dev"), -10**7, 10**7),
        "cf_wrong_bins": _int_in(diag.get("cf_wrong_bins"), 0, len(CF_DOMAIN)),
        "cf_local_err": _fin(diag.get("cf_local_err"), 0.0, 1e5),
        "cf_local_k": _int_in(diag.get("cf_local_k"), 0, 1024),
        "cf_local_est": _fin(diag.get("cf_local_est"), 0.0, 1e5),
        "cf_merge_excess": _fin(diag.get("cf_merge_excess"), -1e5, 1e5),
        "cf_msgs_tables": _int_in(diag.get("cf_msgs_tables"), 0, 10**5),
        "cf_labels_bad": _int_in(diag.get("cf_labels_bad"), 0, 10**5),
        "cf_cover_bad": _int_in(diag.get("cf_cover_bad"), 0, 10**5),
        "cf_submit_added": _int_in(diag.get("cf_submit_added"), -10**7, 10**7),
    }
    return {k: v for k, v in out.items() if v is not None}


def sanitize_extra(diag: Mapping[str, Any]) -> dict[str, Any]:
    """The task's ``sanitize_extra``: the CF fields of a CF card (a card
    marked ``cf`` or holding a ``cf_*`` field), nothing for any other."""

    if diag.get("cf") or any(str(k).startswith("cf_") for k in diag):
        return sanitize_cf_fields(diag)
    return {}


def card_suffix(card: Mapping[str, Any], n_agents: int = N_AGENTS) -> str:
    """The task's ``card_suffix``: the CF fields of a run line."""

    parts = []
    if card.get("cf_rmse") is not None:
        parts.append(f"rmse={float(card['cf_rmse']):.2f}")
    if card.get("cf_sum_dev") is not None:
        parts.append(f"sum_dev={int(card['cf_sum_dev']):+d}")
    if card.get("cf_wrong_bins") is not None:
        parts.append(f"wrong_values={int(card['cf_wrong_bins'])}")
    k = card.get("cf_local_k")
    if card.get("cf_local_err") is not None and k:
        parts.append(f"local={float(card['cf_local_err']):.2f}({int(k)}/{n_agents})")
        if card.get("cf_local_est") is not None:
            parts.append(f"local_est={float(card['cf_local_est']):.2f}")
        if card.get("cf_merge_excess") is not None:
            parts.append(f"merge_excess={float(card['cf_merge_excess']):+.2f}")
    elif card.get("cf_rmse") is not None:
        parts.append("local=n/a")
    m = card.get("cf_msgs_tables")
    if m:
        parts.append(f"labels_bad={int(card.get('cf_labels_bad') or 0)}/{int(m)}")
        parts.append(f"cover_bad={int(card.get('cf_cover_bad') or 0)}/{int(m)}")
    if card.get("cf_submit_added") is not None:
        parts.append(f"submit_added={int(card['cf_submit_added']):+d}")
    return (" " + " ".join(parts)) if parts else ""


__all__ = [
    "CF_GLOSSARY",
    "card_suffix",
    "cf_diag",
    "coverage_bookkeeping",
    "diag_fn",
    "is_cf_instance",
    "local_error",
    "parse_body_table",
    "sanitize_cf_fields",
    "sanitize_extra",
    "single_source_tables",
    "trace_row",
]
