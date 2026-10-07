"""Silo-Bench metrics over the agents' final answers.

A direct implementation of the benchmark's metric definitions: success rate
``S`` (the fraction of agents whose answer is exactly right), partial score
``P`` (the mean per-agent quality ``q_i``), token consumption and
communication density.  Per-agent quality depends on the template level:
Level I is exact match (numbers within a relative tolerance), Level II uses
position-wise element accuracy and Level III the longest correctly ordered
subsequence.  Exact-match correctness compares the
:func:`queenbee.bench.task_bridge.canonical_answer` forms; the module does
not import the upstream benchmark package.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from typing import Any

from queenbee.bench.task_bridge import canonical_answer


def _normalize(value: Any) -> Any:
    """Decode common string encodings of an answer: numeric strings become
    ``int`` / ``float`` and JSON list / object strings are parsed; lists and
    dicts are normalized element-wise."""
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text)
        except (TypeError, ValueError):
            pass
        try:
            return float(text)
        except (TypeError, ValueError):
            pass
        if text.startswith(("[", "{")):
            try:
                return _normalize(json.loads(text))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return text
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _lis_length(values: list[int]) -> int:
    """Length of the longest strictly increasing subsequence (patience
    sorting, O(n log n))."""
    tails: list[int] = []
    for value in values:
        index = bisect_left(tails, value)
        if index == len(tails):
            tails.append(value)
        else:
            tails[index] = value
    return len(tails)


def silo_agent_quality(
    answer: Any,
    expected: Any,
    *,
    level: str,
    tolerance: float = 0.01,
) -> float:
    """Silo-Bench per-agent quality ``q_i`` in ``[0, 1]``, by template level.

    * ``"I"``: 1.0 when a numeric answer lies within ``tolerance`` (relative)
      of a numeric target (a zero target needs exactly 0); non-numeric
      values must be equal.
    * ``"II"``: the fraction of target positions whose element the answer
      list matches at the same position.
    * ``"III"``: the length of the longest subsequence of answer elements
      that appear in target order, over the target length.

    Comparisons the level rule cannot make (non-list answers, unhashable
    elements) fall back to exact equality."""
    actual = _normalize(answer)
    target = _normalize(expected)

    if level == "I":
        if (
            isinstance(target, (int, float))
            and not isinstance(target, bool)
            and isinstance(actual, (int, float))
            and not isinstance(actual, bool)
        ):
            if target == 0:
                return 1.0 if actual == 0 else 0.0
            return 1.0 if abs(actual - target) <= tolerance * abs(target) else 0.0
        return 1.0 if actual == target else 0.0

    if level == "II":
        if isinstance(target, list) and isinstance(actual, list):
            if not target:
                return 1.0 if not actual else 0.0
            matches = sum(1 for exp, got in zip(target, actual) if exp == got)
            return matches / len(target)
        return 1.0 if actual == target else 0.0

    if level == "III":
        if isinstance(target, list) and isinstance(actual, list):
            if not target:
                return 1.0 if not actual else 0.0
            try:
                expected_positions = {value: idx for idx, value in enumerate(target)}
                positions = [
                    expected_positions[value]
                    for value in actual
                    if value in expected_positions
                ]
            except TypeError:
                return 1.0 if actual == target else 0.0
            return _lis_length(positions) / len(target)
        return 1.0 if actual == target else 0.0

    # Case ids outside the I / II / III levels (synthetic test fixtures) get
    # exact-match quality.
    return 1.0 if actual == target else 0.0


def evaluate_silo_submissions(
    *,
    case_id: str,
    answers: list[Any],
    expected_outputs: list[Any],
    submitted_rounds: list[int | None] | None = None,
) -> dict[str, Any]:
    """Silo-Bench ``S`` and ``P`` of one run plus an auditable per-agent record.

    An agent is correct when its canonical answer equals its canonical
    expected output (a missing answer is wrong); ``S`` is the fraction of
    correct agents and ``P`` the mean :func:`silo_agent_quality`.  The level
    is the prefix of ``case_id`` (``II-11`` -> ``II``)."""
    n_agents = len(expected_outputs)
    level = str(case_id).split("-", 1)[0]
    rounds = list(submitted_rounds or [])
    records: list[dict[str, Any]] = []
    correct_count = 0
    quality_sum = 0.0

    for agent_id in range(n_agents):
        answer = answers[agent_id] if agent_id < len(answers) else None
        expected = expected_outputs[agent_id]
        correct = (
            answer is not None
            and canonical_answer(answer) == canonical_answer(expected)
        )
        quality = silo_agent_quality(answer, expected, level=level)
        correct_count += int(correct)
        quality_sum += quality
        records.append(
            {
                "agent_id": agent_id,
                "answer": answer,
                "correct": bool(correct),
                "partial": float(quality),
                "submitted_round": rounds[agent_id] if agent_id < len(rounds) else None,
            }
        )

    success_rate = correct_count / n_agents if n_agents else 0.0
    partial = quality_sum / n_agents if n_agents else 0.0
    return {
        "paper_S": success_rate,
        "paper_P": partial,
        "per_agent_submissions": records,
        "per_agent_answers": [record["answer"] for record in records],
        "per_agent_correct": [record["correct"] for record in records],
        "per_agent_partial": [record["partial"] for record in records],
    }


def silo_token_consumption(output_tokens: int, rounds_executed: int) -> float:
    """Silo-Bench token consumption: generated (completion) tokens per
    executed round; 0.0 when no round ran."""
    if rounds_executed <= 0:
        return 0.0
    return float(output_tokens) / rounds_executed


def silo_communication_density(communication_events: int, n_agents: int) -> float:
    """Silo-Bench communication density: communication events (messages)
    per ordered agent pair, ``events / (n * (n - 1))``; 0.0 for fewer than
    two agents."""
    denominator = n_agents * (n_agents - 1)
    if denominator <= 0:
        return 0.0
    return float(communication_events) / denominator
