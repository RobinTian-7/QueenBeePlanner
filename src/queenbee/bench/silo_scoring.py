"""Graded PARTIAL-CORRECTNESS scoring of one Silo-Bench answer.

The strict signal is exact match (1.0/0.0). A continuous quality score P in
[0, 1] next to the strict success rate S shows how close the failed answers
came, so the gap ``P - S`` localises where coordination breaks down
(Silo-Bench defines a per-agent partial-correctness score for the same
purpose). ``silo_partial_score(answer, ground_truth, output_type)`` grades
ONE answer against ONE expected value: agent 0's answer under the ``sink``
goal (:mod:`queenbee.bench.engine`) and the answer scored by
:func:`queenbee.bench.silo_protocol.score_protocol_answer`. The per-agent
metrics of the ``all_agents`` goal live in :mod:`queenbee.bench.silo_metrics`.

Why not call Silo-Bench's ``compute_partial_correctness`` directly?
    The benchmark's metric (``src/utils/metrics.py`` of the Silo-Bench
    checkout, next to its ``benchmarks/`` directory) has signature
    ``compute_partial_correctness(submissions, expected_output, level)``: it
    grades a list of *per-agent* submissions against a dict carrying
    ``per_agent_values`` and a paradigm ``level`` ("I"/"II"/"III"). Here there
    is one answer, one canonical ground-truth string and ``output_type``
    (which in the shipped benchmarks is uniformly ``"distributed"`` and so is
    not a usable type discriminator). The per-agent API therefore does not map
    cleanly onto ``(answer, expected)``, so grading dispatches on the value's
    structure instead (below).

    The benchmark's longest-increasing-subsequence helper, which underpins its
    Level-III ordering metric, IS reused for sequence ordering. It is loaded
    lazily by file location (the checkout's ``src/`` is not an installed
    package) and falls back to a local copy when unavailable.

Grading, dispatched on the parsed value's structure (with an ``output_type``
hint that routes "set"-like list answers to Jaccard):
    * exact match (canonical)            -> 1.0
    * numeric scalar                     -> max(0, 1 - |a-t| / max(1, |t|))
    * list / sequence                    -> blend of positional-match fraction and
                                            LIS-based ordering ratio (sort-like);
                                            positional fraction alone when the
                                            elements are unhashable
    * set (``output_type`` hint)         -> Jaccard overlap
    * dict                               -> fraction of keys (union of both) whose
                                            values match
    * string                             -> normalized token-overlap (F1)
    * None / sentinel / type mismatch    -> 0.0

Both ``answer`` and ``ground_truth`` may be raw Python values OR canonical-JSON
strings (e.g. the ``answer_key`` of the private scoring payload built by
:mod:`queenbee.bench.task_bridge`); they are coerced via the same parsing
``canonical_answer`` uses, so ``"9"`` and ``9`` or ``"[3, 1]"`` and ``[3, 1]``
grade identically.
"""

from __future__ import annotations

import importlib.util
import json
import re
from typing import Any, Callable

from queenbee.bench.task_bridge import canonical_answer
from queenbee.paths import default_benchmarks_dir

_UNKNOWN_SENTINELS = {"UNKNOWN", "NONE", "NULL"}


# --------------------------------------------------------------------------- #
# Silo-Bench's LIS helper (lazy, guarded). Reused for sequence ordering quality.
# --------------------------------------------------------------------------- #
def _local_lis_length(seq: list[Any]) -> int:
    """Longest strictly-increasing subsequence length (local fallback)."""
    if not seq:
        return 0
    from bisect import bisect_left

    tails: list[Any] = []
    for x in seq:
        pos = bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)


def _load_official_lis() -> tuple[Callable[[list[Any]], int], bool]:
    """Return (lis_fn, used_official). Import Silo-Bench's helper by file location.

    Any import / attribute / IO / type error, or a helper that fails the
    contract check, falls back to :func:`_local_lis_length`.
    """
    # <silo-bench checkout>/src/utils/metrics.py sits next to its benchmarks/ dir.
    metrics_path = default_benchmarks_dir().parent / "src" / "utils" / "metrics.py"
    try:
        spec = importlib.util.spec_from_file_location(
            "queenbee.bench._silo_official_metrics", metrics_path
        )
        if spec is None or spec.loader is None:
            raise ImportError("no loader for silo metrics")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = module._longest_increasing_subsequence_length  # type: ignore[attr-defined]
        # Smoke-check the contract so an upstream refactor cannot silently
        # change the scores.
        if fn([0, 2, 1, 3]) != 3:
            raise ImportError("silo LIS contract changed")
        return fn, True
    except (ImportError, AttributeError, OSError, TypeError):
        return _local_lis_length, False


_LIS: tuple[Callable[[list[Any]], int], bool] | None = None


def _lis_length(seq: list[Any]) -> int:
    """LIS length via Silo-Bench's helper when the checkout provides it
    (loaded on first use), else the local copy."""
    global _LIS
    if _LIS is None:
        _LIS = _load_official_lis()
    return _LIS[0](seq)


# --------------------------------------------------------------------------- #
# Coercion helpers (mirror canonical_answer parsing, but return live values)
# --------------------------------------------------------------------------- #
def _coerce(value: Any) -> Any:
    """Best-effort parse to a live Python value; None for unknown/empty sentinels.

    Strings that look like numbers or JSON are parsed; bare booleans are folded;
    everything else is returned as-is. Mirrors ``canonical_answer`` so that a raw
    value and its canonical-string form coerce to the same thing.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() in _UNKNOWN_SENTINELS:
            return None
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
    return value


def _as_number(value: Any) -> float | None:
    """The value as a float when it is an int or float, else None; booleans
    are rejected, so ``True`` never grades as the number 1."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Per-structure graders (each returns a score in [0, 1])
# --------------------------------------------------------------------------- #
def _score_numeric(a: float, t: float) -> float:
    """1.0 when equal, else the normalized distance max(0, 1 - |a-t| / max(1, |t|))."""
    if a == t:
        return 1.0
    return max(0.0, 1.0 - abs(a - t) / max(1.0, abs(t)))


def _score_set(a: Any, t: Any) -> float:
    """Jaccard overlap |a & t| / |a | t| (two empty sets -> 1.0; unhashable
    elements -> 0.0)."""
    try:
        a_set, t_set = set(a), set(t)
    except TypeError:
        return 0.0
    if not a_set and not t_set:
        return 1.0
    union = a_set | t_set
    if not union:
        return 1.0
    return len(a_set & t_set) / len(union)


def _score_sequence(a: list[Any], t: list[Any]) -> float:
    """Blend positional-match fraction with an LIS-based ordering ratio.

    The positional fraction rewards getting the right value in the right slot; the
    ordering ratio (longest correctly-ordered subsequence / target length, via
    :func:`_lis_length`) rewards sort-like answers that are globally ordered
    even when individual positions are shifted. Their mean keeps near-misses such
    as ``[1,2,4,3]`` vs ``[1,2,3,4]`` strictly inside (0, 1). With unhashable
    elements the positional fraction alone is returned.
    """
    if not a and not t:
        return 1.0
    if not t:
        return 0.0

    positional = sum(1 for x, y in zip(a, t) if x == y) / len(t)

    # Map each answer element to the position of its first occurrence in target,
    # then the LIS over those positions is the longest correctly-ordered
    # subsequence.
    target_pos: dict[Any, int] = {}
    for i, v in enumerate(t):
        try:
            target_pos.setdefault(v, i)
        except TypeError:  # unhashable element -> ordering ratio not applicable
            return positional
    try:
        positions = [target_pos[x] for x in a if x in target_pos]
    except TypeError:
        return positional
    ordering = _lis_length(positions) / len(t)

    return (positional + ordering) / 2.0


def _score_dict(a: dict[Any, Any], t: dict[Any, Any]) -> float:
    """Fraction of the keys (union of both key sets) present in both dicts with
    equal values."""
    if not a and not t:
        return 1.0
    keys = set(a) | set(t)
    if not keys:
        return 1.0
    matches = sum(1 for k in keys if k in a and k in t and a[k] == t[k])
    return matches / len(keys)


def _score_string(a: str, t: str) -> float:
    """Token-overlap F1 (harmonic mean of precision and recall) of the
    lowercase word tokens, each target token matched at most once."""
    a_tokens, t_tokens = _tokenize(a), _tokenize(t)
    if not a_tokens and not t_tokens:
        return 1.0
    if not a_tokens or not t_tokens:
        return 0.0
    overlap = 0
    remaining = list(t_tokens)
    for tok in a_tokens:
        if tok in remaining:
            remaining.remove(tok)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_tokens)
    recall = overlap / len(t_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def silo_partial_score(answer: Any, ground_truth: Any, output_type: str) -> float:
    """Graded partial-correctness in [0, 1] for one Silo-Bench answer.

    ``answer`` / ``ground_truth`` may be raw values or canonical-JSON strings.
    ``output_type`` is a coarse hint; the value's structure drives dispatch, with
    a "set"-like hint routing list comparisons to Jaccard. A null/sentinel answer,
    or one whose structure cannot be sensibly compared to the truth (e.g. an
    unparseable string against a numeric truth), scores 0.0. Exact (canonical)
    matches always score 1.0.
    """
    # Exact-match shortcut keeps partial >= the strict signal and avoids
    # float drift on clean hits.
    if canonical_answer(answer) == canonical_answer(ground_truth) != "UNKNOWN":
        return 1.0

    a = _coerce(answer)
    t = _coerce(ground_truth)
    if a is None or t is None:
        return 0.0

    hint = (output_type or "").strip().lower()

    # Numeric scalar (reject bools so they fall through to exact-only handling).
    a_num, t_num = _as_number(a), _as_number(t)
    if a_num is not None and t_num is not None:
        return _score_numeric(a_num, t_num)

    # Set hint: grade list/sequence answers as unordered sets.
    if hint in {"set", "set_of_values", "unordered_set"} and isinstance(
        a, (list, tuple)
    ) and isinstance(t, (list, tuple)):
        return _score_set(a, t)

    if isinstance(t, dict):
        return _score_dict(a, t) if isinstance(a, dict) else 0.0

    if isinstance(t, (list, tuple)):
        if not isinstance(a, (list, tuple)):
            return 0.0
        return _score_sequence(list(a), list(t))

    if isinstance(t, str):
        return _score_string(a, t) if isinstance(a, str) else 0.0

    if isinstance(t, bool):
        return 1.0 if a == t else 0.0

    # Unknown / incomparable structure.
    return 0.0
