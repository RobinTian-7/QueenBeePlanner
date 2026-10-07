"""Answer-free diagnosis cards of executed team-program runs.

Diagnosis cards are one of the three experience components of QueenBee-Evo
(with the hypothesis ledger and the program archive).  Every executed row
gets a small card: which agents were right (as a bitstring), how many
distinct answers the agents converged on, whether the plurality was right,
which failure class the run belongs to, where the completion tokens went per
phase, the round at which the team submitted, and how many message bodies
were delivered to agents that were not called that round.  Under the
message_only_v2 worker contract an agent reads only the bodies delivered in
the round it is called in; any other body is lost, although its source ids
still count as known.

Leak rule: a card never carries an answer value, a ground-truth value or any
fragment of either.  Correctness enters only as booleans / bit positions,
answers only as a count of distinct canonical forms, and the answer shape is
parsed from the task's public ``**Output:**`` sentence (the same text every
worker sees), never from the ground truth.  :func:`sanitize_diag` is the
single whitelist every renderer reads a card through.

Main entry points:

* :func:`diagnose_execution` - one card per executed row (called from
  ``queenbee.program.execute.execute_python_source_on_case``);
* :func:`build_failure_map` / :func:`render_failure_map` - (answer shape x
  failure class) -> units, over the T rows of each unit;
* :func:`render_diag_card` - the per-unit card block of the planner prompt.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Iterable, Mapping

from queenbee.tasks import get_task

# --------------------------------------------------------------------------- #
# Vocabularies
# --------------------------------------------------------------------------- #

#: Closed failure-class vocabulary.  The order is the tie-break rank of
#: :func:`dominant_failure_class` and of the failure-map cell order.
FAILURE_CLASSES: tuple[str, ...] = (
    "ok",
    # Most specific / most actionable first: the team already computed the
    # right result; only the exact answer form differs (wins count ties).
    "precision-near-miss",
    # Opt-in (:func:`local_only_enabled`): at least half of the wrong agents
    # answered from their own shard alone -- a work-instruction failure that
    # would otherwise read as a generic wrong-answer class.
    "local-only",
    "divergent",
    "consensus-wrong",
    "scattered-wrong",
    "shape-mismatch",
    "format",
    "content-loss",
    "budget",
    "infra",
)

#: Closed answer-shape vocabulary.
ANSWER_SHAPES: tuple[str, ...] = ("scalar", "collection", "segment")

#: Row-level error classes (same names as
#: ``queenbee.program.execute.classify_row_error``).
ERROR_CLASSES: tuple[str, ...] = (
    "format", "budget", "infra", "undefined_name", "exec",
)

#: One-line meaning of every failure class (rendered into planner prompts).
FAILURE_CLASS_GLOSSARY: dict[str, str] = {
    "ok": "every agent submitted the right answer",
    "divergent": "some agents right, some wrong (consensus/integration failure)",
    "consensus-wrong": "all agents agree on the same wrong answer (a shared "
    "computation or merge error propagated to everyone)",
    "scattered-wrong": "every agent wrong, with several different answers",
    "precision-near-miss": "the wrong agents computed essentially the right "
    "result but not in the exact form the grader requires (numeric rounding/"
    "precision, number-vs-string types); the grader demands an exact match",
    "local-only": "each wrong agent submitted exactly the answer its OWN "
    "data shard alone gives: it computed from its own shard as if the data it "
    "received did not exist (whether or not that data was delivered to it)",
    "shape-mismatch": "answers do not have the type the task's Output sentence "
    "asks for (e.g. a list where one number is expected, or null)",
    "format": "the run broke the answer/contract format (unparseable answer or "
    "a program error) and scored 0",
    "content-loss": "message bodies were delivered to agents that were NOT "
    "called that round, so the text was lost (source ids still counted)",
    "budget": "the run exhausted a budget cap and scored 0",
    "infra": "infrastructure failure (gateway/timeout); not the program's fault",
}


def failure_class_glossary() -> Mapping[str, str]:
    """:data:`FAILURE_CLASS_GLOSSARY` with the active task's ``glossary``
    entries over it (merged at use time; the module dict never changes)."""

    extra = get_task().glossary
    return FAILURE_CLASS_GLOSSARY if extra is None else {**FAILURE_CLASS_GLOSSARY, **extra}


_BITS = re.compile(r"^[01]{1,64}$")
#: Characters :func:`clean_unit_id` keeps: letters, digits, ``_``, ``-``,
#: ``@``, ``#`` and ``.``.  Template ids and the ``@`` rung suffix
#: (``II-11@x5``) pass through intact, so the planner's ``target_units``
#: match the host's unit keys.
_UNIT_ID_DROP = re.compile(r"[^A-Za-z0-9_\-@#.]")


def clean_unit_id(value: Any, limit: int = 32) -> str:
    """Unit-id sanitizer of the diagnosis renderers and of the planner's
    ``target_units`` (``queenbee.program.mint``): keeps only the characters
    listed above and truncates to ``limit``."""

    return _UNIT_ID_DROP.sub("", str(value if value is not None else ""))[:limit]
_OUTPUT_SECTION = re.compile(
    r"\*\*Output:\*\*[ \t]*\n(.*?)(?=\n[ \t]*\n|\n\*\*|\Z)", re.S
)
_SEGMENT_HINT = re.compile(
    r"\beach agent submits\b[^.]*\b(their|its|own)\b|\bportion\b|\bsegment\b",
    re.I,
)
_COLLECTION_HINT = re.compile(
    r"\b(array|arrays|list|lists|dictionary|dictionaries|dict|mapping|set of|"
    r"tuples?|pairs|sequence|matrix|table)\b",
    re.I,
)
_SCALAR_HINT = re.compile(
    r"\b(a single|an? integer|a string|a floating|a number|a boolean|"
    r"integer|number|count|value)\b",
    re.I,
)

# --------------------------------------------------------------------------- #
# Public-text helpers
# --------------------------------------------------------------------------- #


def output_sentence(task_text: Any) -> str | None:
    """The public ``**Output:**`` sentence of a Silo-Bench task text."""

    if not isinstance(task_text, str) or not task_text:
        return None
    match = _OUTPUT_SECTION.search(task_text)
    if not match:
        return None
    text = " ".join(match.group(1).split())
    return text or None


def answer_shape_from_text(
    task_text: Any, *, segmented: bool | None = None
) -> str | None:
    """scalar | collection | segment, parsed from PUBLIC task text only.

    ``segmented`` (the instance's public ``is_segmented`` flag) wins when
    True; otherwise the Output sentence decides.  None when there is no
    Output sentence or it matches no shape.
    """

    if segmented:
        return "segment"
    sentence = output_sentence(task_text)
    if sentence is None:
        return None
    if _SEGMENT_HINT.search(sentence):
        return "segment"
    if _COLLECTION_HINT.search(sentence):
        return "collection"
    if _SCALAR_HINT.search(sentence):
        return "scalar"
    return None


def instance_answer_shape(instance: Any) -> str | None:
    """Answer shape of a benchmark instance from its public task prompt."""

    meta = getattr(instance, "meta", None) or {}
    segmented = bool(meta.get("is_segmented")) if isinstance(meta, dict) else False
    text = getattr(instance, "task_prompt", "") or ""
    return answer_shape_from_text(text, segmented=segmented)


# --------------------------------------------------------------------------- #
# Diagnosis of one execution
# --------------------------------------------------------------------------- #


def _dump(model: Any) -> Any:
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        try:
            return model.model_dump(mode="json")
        except Exception:  # noqa: BLE001 - diagnostics never raise
            return None
    return model


def _canonical(value: Any) -> str:
    from queenbee.bench.task_bridge import canonical_answer

    try:
        return canonical_answer(value)
    except Exception:  # noqa: BLE001 - diagnostics never raise
        return json.dumps(value, sort_keys=True, default=str)


def _parsed(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                return value
    return value


def _shape_ok(answer: Any, shape: str | None) -> bool:
    if shape is None:
        return True
    value = _parsed(answer)
    if value is None:
        return False
    if shape == "scalar":
        return not isinstance(value, (list, dict, tuple))
    if shape == "collection":
        return isinstance(value, (list, dict, tuple))
    if shape == "segment":
        return isinstance(value, (list, tuple, dict))
    return True


def _error_class(facts: Mapping[str, Any]) -> str | None:
    if facts.get("infra"):
        return "infra"
    err = str(facts.get("error") or "")
    if not err:
        return None
    low = err.lower()
    if "answerformaterror" in low:
        return "format"
    if "budgeterror" in low:
        return "budget"
    if "undefined name" in low:
        return "undefined_name"
    return "exec"


def _phase_round_spans(source: str | None, n_agents: int) -> list[int] | None:
    """Rounds per PHASES entry as the seed program's phase interpreter runs
    them.

    Reads the PHASES literal with ``ast.literal_eval`` and never executes
    program text; names of literal module constants are resolved through the
    AST.  At most the first 8 entries count.  None when PHASES cannot be read
    as a list of dicts.
    """

    if not source or "PHASES = [" not in source:
        return None
    import ast

    match = re.search(r"^PHASES = (\[.*?\n\])", source, re.S | re.M)
    phases: Any = None
    if match:
        try:
            phases = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            phases = None
    if not isinstance(phases, list):
        # A PHASES entry may name a module-level constant of the genome
        # (e.g. ``"work_instruction": MERGE_WI`` defined above PHASES):
        # resolve literal constants through the AST, still never executing
        # program text.
        phases = _phases_via_constants(source)
    if not isinstance(phases, list):
        return None
    spans: list[int] = []
    for phase in phases[:8]:
        if not isinstance(phase, dict):
            return None
        try:
            rounds = int(phase.get("rounds", 1))
        except (TypeError, ValueError):
            rounds = 1
        if bool(phase.get("scale_rounds_to_agents")):
            rounds = n_agents - 1
        spans.append(max(1, rounds))
    return spans or None


def _phases_via_constants(source: str) -> list[dict[str, Any]] | None:
    """The PHASES list with Name references to literal module constants
    resolved (only the ``rounds`` / ``scale_rounds_to_agents`` keys matter
    for round spans; other values that are not literals become None).
    None when PHASES is not a list of dict displays / constant names."""

    import ast

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    constants: dict[str, Any] = {}
    phases_node: ast.AST | None = None
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        name = node.targets[0].id
        if name == "PHASES":
            phases_node = node.value
            continue
        try:
            constants[name] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            constants.pop(name, None)

    def _value(expr: ast.AST) -> Any:
        if isinstance(expr, ast.Name):
            if expr.id in constants:
                return constants[expr.id]
            raise ValueError(expr.id)
        return ast.literal_eval(expr)

    if not isinstance(phases_node, ast.List):
        return None
    out: list[dict[str, Any]] = []
    for element in phases_node.elts:
        if isinstance(element, ast.Name):
            value = constants.get(element.id)
            if not isinstance(value, dict):
                return None
            out.append(dict(value))
            continue
        if not isinstance(element, ast.Dict):
            return None
        entry: dict[str, Any] = {}
        for key_node, value_node in zip(element.keys, element.values):
            if key_node is None:
                return None
            try:
                key = _value(key_node)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                return None
            try:
                entry[key] = _value(value_node)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                if key in ("rounds", "scale_rounds_to_agents"):
                    return None
                entry[key] = None
        out.append(entry)
    return out


def _phase_tokens(
    calls: list[dict[str, Any]], spans: list[int] | None
) -> list[int] | None:
    """Completion tokens per PHASES entry; the last element is the submit
    barrier.  Rounds past the last phase count toward the last phase."""

    if not calls or not spans:
        return None
    buckets = [0] * (len(spans) + 1)
    bounds: list[int] = []
    total = 0
    for span in spans:
        total += span
        bounds.append(total)
    for call in calls:
        tokens = call.get("completion_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int):
            continue
        if str(call.get("mode")) == "submit":
            buckets[-1] += tokens
            continue
        rnd = call.get("round")
        if isinstance(rnd, bool) or not isinstance(rnd, int):
            continue
        index = len(spans) - 1
        for i, bound in enumerate(bounds):
            if rnd < bound:
                index = i
                break
        buckets[index] += tokens
    return buckets


def _lost_messages(
    messages: list[dict[str, Any]],
    calls: list[dict[str, Any]] | None,
    submit_round: int | None,
) -> int | None:
    """Message bodies delivered to an agent that was not called that round.

    Bodies delivered at the submit round are read at the barrier, so they
    never count.  Without a call ledger the called set is inferred from the
    senders: an agent called in send mode with no recipients sends nothing,
    so bodies delivered to it in that round are counted as lost (an
    over-count, only in that rare case)."""

    if messages is None:
        return None
    called: set[tuple[int, int]] = set()
    if calls:
        for call in calls:
            if str(call.get("mode")) == "submit":
                continue
            try:
                called.add((int(call["round"]), int(call["agent_id"])))
            except (KeyError, TypeError, ValueError):
                continue
    else:
        for message in messages:
            try:
                called.add((int(message["round_sent"]), int(message["src"])))
            except (KeyError, TypeError, ValueError):
                continue
    lost = 0
    for message in messages:
        try:
            delivered = int(message["round_delivered"])
            dst = int(message["dst"])
        except (KeyError, TypeError, ValueError):
            continue
        if submit_round is not None and delivered >= submit_round:
            continue
        if (delivered, dst) not in called:
            lost += 1
    return lost


def _empty_diag(facts: Mapping[str, Any], shape: str | None) -> dict[str, Any]:
    s_value = facts.get("S")
    return {
        "S": float(s_value) if isinstance(s_value, (int, float))
        and not isinstance(s_value, bool) else None,
        "agent_correct": None,
        "n_distinct_answers": None,
        "majority_correct": None,
        "failure_class": None,
        "answer_shape": shape,
        "phase_tokens": None,
        "submit_round": None,
        "n_messages": None,
        "n_lost_messages": None,
    }


def diagnose_execution(
    output: Any,
    instance: Any,
    facts: Mapping[str, Any],
    *,
    per_agent_correct: list[bool] | None = None,
    ledger: Mapping[str, Any] | None = None,
    source: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    """Answer-free diagnosis card of one executed row (never raises).

    ``output``: the program's ``PythonProgramOutput`` (or its dict dump), or
    None when the run produced no output.  ``facts``: the row facts (``S``,
    ``error``, ``infra``, ...).  Optional extras make the card more precise:
    ``per_agent_correct`` (the scorer's per-agent booleans - used verbatim
    when given for every agent, so the card agrees with ``S``), ``ledger``
    (the runtime's call ledger, for exact content-loss counts and per-phase
    tokens) and ``source`` (the executed program, for the PHASES round
    spans).  ``goal`` is the information goal (default ``all_agents``);
    under any other goal a single answer is scored and the card has no
    per-agent picture.  The opt-in refinements of :func:`refine_card` are
    applied last.
    """

    shape = None
    try:
        shape = instance_answer_shape(instance)
    except Exception:  # noqa: BLE001
        shape = None
    diag = _empty_diag(facts, shape)
    try:
        return refine_card(_diagnose(
            output, instance, facts, diag,
            per_agent_correct=per_agent_correct, ledger=ledger,
            source=source, goal=goal,
        ), output, instance)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        err = _error_class(facts)
        if err is not None:
            diag["failure_class"] = _class_for_error(err)
            diag["error_class"] = err
        return diag


def _class_for_error(err: str) -> str:
    if err in ("infra", "budget", "format"):
        return err
    # Any other error (``undefined_name``, or ``exec`` such as a DataFlowError
    # of a validated program) means the run broke its contract: reported as
    # ``format`` (its answers never reached the scorer), with the precise
    # ``error_class`` alongside.
    return "format"


def _diagnose(
    output: Any,
    instance: Any,
    facts: Mapping[str, Any],
    diag: dict[str, Any],
    *,
    per_agent_correct: list[bool] | None,
    ledger: Mapping[str, Any] | None,
    source: str | None,
    goal: str | None,
) -> dict[str, Any]:
    n_agents = int(getattr(instance, "n_agents", 0) or 0)
    ledger_dict = dict(ledger or {})
    calls = [c for c in (ledger_dict.get("calls") or []) if isinstance(c, dict)]
    spans = _phase_round_spans(source, n_agents) if n_agents else None
    diag["phase_tokens"] = _phase_tokens(calls, spans)

    err = _error_class(facts)
    data = _dump(output)
    if err is not None or not isinstance(data, dict):
        if err is not None:
            diag["failure_class"] = _class_for_error(err)
            diag["error_class"] = err
        return diag

    submissions = [s for s in (data.get("submissions") or []) if isinstance(s, dict)]
    messages = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
    rounds = [
        s.get("submitted_round") for s in submissions
        if isinstance(s.get("submitted_round"), int)
        and not isinstance(s.get("submitted_round"), bool)
    ]
    submit_round = max(rounds) if rounds else None
    if submit_round is None:
        executed = data.get("rounds_executed")
        if isinstance(executed, int) and not isinstance(executed, bool) and executed > 0:
            submit_round = executed - 1
    diag["submit_round"] = submit_round
    diag["n_messages"] = len(messages)
    diag["n_lost_messages"] = _lost_messages(messages, calls or None, submit_round)

    goal = goal or "all_agents"
    answers_by_id: dict[int, Any] = {}
    for item in submissions:
        try:
            answers_by_id[int(item.get("agent_id"))] = item.get("answer")
        except (TypeError, ValueError):
            continue
    if not n_agents:
        n_agents = len(answers_by_id)
    answers = [answers_by_id.get(agent_id) for agent_id in range(n_agents)]
    meta = getattr(instance, "meta", None) or {}
    segmented = bool(meta.get("is_segmented")) if isinstance(meta, dict) else False

    if goal != "all_agents" or not n_agents:
        # Single scored answer (sink goal): no per-agent picture.
        if facts.get("success"):
            diag["failure_class"] = "ok"
        else:
            first = answers[0] if answers else None
            mismatch = 0 if _shape_ok(first, diag["answer_shape"]) else 1
            diag["failure_class"] = _wrong_class(
                diag, n_wrong=1, n_mismatch=mismatch, distinct=1,
                segmented=False,
            )
        return diag

    correct = list(per_agent_correct) if per_agent_correct is not None else None
    fallback = False
    if correct is None or len(correct) != n_agents:
        correct = _score_agents(instance, answers)
        fallback = True
    canon = [_canonical(a) for a in answers]
    s_value = diag.get("S")
    if correct is not None:
        correct = [bool(c) for c in correct]
        if (
            fallback
            and isinstance(s_value, float)
            and abs(sum(correct) / n_agents - s_value) > 1e-6
        ):
            # The fallback comparator (used only when the scorer's per-agent
            # booleans are missing) disagrees with the row's S - e.g. a
            # string-typed answer the benchmark parser accepts.  S is
            # authoritative: drop the per-agent picture and classify from S.
            correct = None
    if correct is None:
        return _classify_from_s(
            diag, answers=answers, canon=canon, n_agents=n_agents,
            segmented=segmented,
        )
    diag["agent_correct"] = "".join("1" if c else "0" for c in correct)
    n_correct = sum(correct)

    if segmented:
        diag["n_distinct_answers"] = None
        diag["majority_correct"] = 2 * n_correct > n_agents
    else:
        counts = Counter(canon)
        diag["n_distinct_answers"] = len(counts)
        ranked = counts.most_common()
        top_n = ranked[0][1]
        tied = [form for form, count in ranked if count == top_n]
        verdicts = set()
        for form in tied:
            for index, candidate in enumerate(canon):
                if candidate == form:
                    verdicts.add(bool(correct[index]))
                    break
        if len(verdicts) == 1:
            # A unique plurality, or a tie whose tied answers share one
            # verdict (the plurality verdict then holds however the tie
            # breaks).
            diag["majority_correct"] = verdicts.pop()
        else:
            # A tie between the right answer and a wrong one: no plurality.
            diag["majority_correct"] = None
            diag["plurality_tie"] = True

    if n_correct == n_agents:
        diag["failure_class"] = "ok"
        return diag
    wrong_idx = [i for i, c in enumerate(correct) if not c]
    mismatch = sum(1 for i in wrong_idx if not _shape_ok(answers[i], diag["answer_shape"]))
    if mismatch and 2 * mismatch >= len(wrong_idx):
        diag["failure_class"] = "shape-mismatch"
        return diag
    near = _near_miss_flags(instance, answers, wrong_idx)
    if near is not None:
        diag["n_near_miss"] = int(sum(near))
        if 2 * sum(near) >= len(wrong_idx):
            diag["failure_class"] = "precision-near-miss"
            return diag
    if diag.get("n_lost_messages"):
        diag["failure_class"] = "content-loss"
        return diag
    if n_correct > 0:
        diag["failure_class"] = "divergent"
        return diag
    if segmented:
        diag["failure_class"] = "scattered-wrong"
        return diag
    wrong_forms = {canon[i] for i in wrong_idx}
    diag["failure_class"] = (
        "consensus-wrong" if len(wrong_forms) == 1 else "scattered-wrong"
    )
    return diag


def _classify_from_s(
    diag: dict[str, Any],
    *,
    answers: list[Any],
    canon: list[str],
    n_agents: int,
    segmented: bool,
) -> dict[str, Any]:
    """Failure class when no trustworthy per-agent booleans exist: the
    number of right agents comes from S (= fraction of agents right), the
    rest from answer shapes / distinct counts (still answer-free)."""

    s_value = diag.get("S")
    if not isinstance(s_value, float) or not n_agents:
        return diag
    if not segmented:
        diag["n_distinct_answers"] = len(set(canon))
    if s_value >= 1.0 - 1e-9:
        diag["failure_class"] = "ok"
        return diag
    n_correct = int(round(s_value * n_agents))
    n_wrong = max(1, n_agents - n_correct)
    mismatch = sum(1 for a in answers if not _shape_ok(a, diag["answer_shape"]))
    mismatch = min(mismatch, n_wrong)
    if mismatch and 2 * mismatch >= n_wrong:
        diag["failure_class"] = "shape-mismatch"
    elif diag.get("n_lost_messages"):
        diag["failure_class"] = "content-loss"
    elif n_correct > 0:
        diag["failure_class"] = "divergent"
    elif segmented:
        diag["failure_class"] = "scattered-wrong"
    else:
        diag["failure_class"] = (
            "consensus-wrong" if len(set(canon)) == 1 else "scattered-wrong"
        )
    return diag


def _wrong_class(
    diag: dict[str, Any], *, n_wrong: int, n_mismatch: int, distinct: int,
    segmented: bool,
) -> str:
    if n_mismatch and 2 * n_mismatch >= n_wrong:
        return "shape-mismatch"
    if diag.get("n_lost_messages"):
        return "content-loss"
    return "consensus-wrong" if distinct == 1 and not segmented else "scattered-wrong"


def _to_num(value: Any) -> Any:
    """Parse numeric strings and JSON list / object text, recursing into
    lists and dicts (a local comparison helper; nothing it returns leaves
    this module)."""

    if isinstance(value, str):
        text = value.strip()
        for cast in (int, float):
            try:
                return cast(text)
            except (TypeError, ValueError):
                pass
        if text[:1] in ("[", "{"):
            try:
                return _to_num(json.loads(text))
            except (TypeError, ValueError):
                return text
        return text
    if isinstance(value, (list, tuple)):
        return [_to_num(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_num(v) for k, v in value.items()}
    return value


def _near(a: Any, b: Any, rel: float = 1e-3, abs_tol: float = 1e-3) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= max(abs_tol, rel * abs(float(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_near(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_near(a[k], b[k]) for k in a)
    return a == b


def _near_miss_flags(instance: Any, answers: list[Any], wrong_idx: list[int]) -> list[bool] | None:
    """For each wrong agent: is its answer equal to the expected one up to
    numeric tolerance / number-vs-string typing?  Only booleans leave this
    function (answer-free)."""

    meta = getattr(instance, "meta", None) or {}
    expected = list(meta.get("expected_outputs") or []) if isinstance(meta, dict) else []
    if len(expected) != len(answers):
        truth = getattr(instance, "ground_truth", None)
        if truth is None:
            return None
        if isinstance(truth, dict) and isinstance(truth.get("per_agent_values"), list) \
                and len(truth["per_agent_values"]) == len(answers):
            expected = list(truth["per_agent_values"])
        else:
            expected = [truth for _ in answers]
    flags = []
    for index in wrong_idx:
        answer = answers[index]
        if answer is None:
            flags.append(False)
            continue
        try:
            got, want = _to_num(_parsed(answer)), _to_num(expected[index])
            # Near-miss = NOT exactly equal but equal up to numeric tolerance /
            # number-vs-string typing (an exact match marked wrong by the
            # scorer is a different failure, not a precision one).
            exact = _canonical(answer) == _canonical(expected[index]) or got == want
            flags.append(bool(not exact and _near(got, want)))
        except Exception:  # noqa: BLE001 - diagnostics never raise
            flags.append(False)
    return flags


#: Environment switch of the opt-in local-only refinement (``--local-only-diag``
#: sets it).  While it is off, no card carries ``n_local_only`` or the
#: ``local-only`` class.
LOCAL_ONLY_ENV = "QB_DIAG_LOCAL_ONLY"
#: Only the generic wrong-answer classes are refined; error / shape /
#: near-miss / content-loss cards keep their more specific class.
_LOCAL_ONLY_REFINES = ("divergent", "consensus-wrong", "scattered-wrong")


def local_only_enabled() -> bool:
    return os.environ.get(LOCAL_ONLY_ENV, "") == "1"


def _local_only_flags(instance: Any, answers: list[Any], wrong_idx: list[int]) -> list[bool] | None:
    """For each wrong agent: is its answer exactly the task's answer computed
    from that agent's OWN shard alone?  The host's independent gold checker
    (``queenbee.evo.ladder.recompute_gold``: shards + public statement only)
    runs on a one-shard copy of the instance.  Only booleans leave
    (answer-free).  None when the instance has no case id, when its shard
    count differs from the answer count, or when
    ``queenbee.evo.ladder.assert_dev_template`` refuses the template (TEST or
    unknown); an agent whose checker call raises gets False."""

    from queenbee.evo.ladder import assert_dev_template, recompute_gold

    shards = list(getattr(instance, "shards", None) or [])
    case_id = str(getattr(instance, "case_id", "") or "")
    if not case_id or len(shards) != len(answers):
        return None
    try:
        assert_dev_template(case_id)
    except Exception:  # noqa: BLE001 - TEST / unknown template: no flag
        return None
    text = str(getattr(instance, "task_prompt", "") or "")
    flags = []
    for index in wrong_idx:
        answer = answers[index]
        if answer is None:
            flags.append(False)
            continue
        try:
            local = recompute_gold({
                "case_id": case_id, "task_description": text,
                "agent_configs": [{"input_shard": shards[index]}],
            })[0]
            flags.append(bool(
                _canonical(answer) == _canonical(local)
                or _to_num(_parsed(answer)) == _to_num(local)
            ))
        except Exception:  # noqa: BLE001 - diagnostics never raise
            flags.append(False)
    return flags


def apply_local_only(diag: Any, output: Any, instance: Any) -> Any:
    """Local-only refinement of a card, in place (no-op unless enabled):
    count the wrong agents whose answer is their own-shard-only answer
    (``n_local_only``) and, when that is at least half of the wrong agents of
    a generic wrong-answer card, re-class it ``local-only``.  Works on fresh
    cards and on cards stored in traces (of the card it reads only the
    per-agent bits)."""

    if not local_only_enabled() or not isinstance(diag, dict):
        return diag
    try:
        bits = diag.get("agent_correct")
        data = _dump(output)
        if not (isinstance(bits, str) and _BITS.match(bits)) or not isinstance(data, dict):
            return diag
        by_id: dict[int, Any] = {}
        for item in data.get("submissions") or []:
            if isinstance(item, dict):
                try:
                    by_id[int(item.get("agent_id"))] = item.get("answer")
                except (TypeError, ValueError):
                    continue
        answers = [by_id.get(i) for i in range(len(bits))]
        wrong = [i for i, bit in enumerate(bits) if bit == "0"]
        flags = _local_only_flags(instance, answers, wrong) if wrong else None
        if flags is None:
            return diag
        diag["n_local_only"] = int(sum(flags))
        if diag.get("failure_class") in _LOCAL_ONLY_REFINES and 2 * sum(flags) >= len(wrong):
            diag["failure_class"] = "local-only"
    except Exception:  # noqa: BLE001 - diagnostics never raise
        pass
    return diag


#: Environment switch of the opt-in answer-seen refinement (``--attempt-diag``
#: sets it): cards count the wrong agents that already HAD the right answer in
#: their final snapshot.
ATTEMPT_ENV = "QB_DIAG_ATTEMPT"


def attempt_diag_enabled() -> bool:
    return os.environ.get(ATTEMPT_ENV, "") == "1"


def _value_token(value: Any) -> str | None:
    """The regex of a scalar gold value as a standalone token (None for
    collections / booleans: the check is scalar-only)."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int):
        return rf"(?<![\d.]){re.escape(str(value))}(?![\d]|\.\d)"
    if isinstance(value, float):
        return rf"(?<![\d.]){re.escape(repr(value))}(?!\d)"
    if isinstance(value, str) and value.strip():
        return rf"(?<![A-Za-z0-9_]){re.escape(value.strip())}(?![A-Za-z0-9_])"
    return None


def apply_answer_seen(diag: Any, output: Any, instance: Any) -> Any:
    """Answer-seen refinement of a card, in place (no-op unless enabled): for a
    scalar-answer task, ``n_answer_seen`` = wrong agents whose final snapshot
    -- the bodies delivered to them at the submit round plus the bodies they
    sent in their last sending round -- already contains the expected answer
    as a standalone token, i.e. the right result reached them and they
    submitted something else.  Only the count leaves (answer-free)."""

    if not attempt_diag_enabled() or not isinstance(diag, dict):
        return diag
    try:
        bits = diag.get("agent_correct")
        data = _dump(output)
        submit_round = diag.get("submit_round")
        if diag.get("answer_shape") != "scalar" or not (isinstance(bits, str) and _BITS.match(bits)) \
                or not isinstance(data, dict) or not isinstance(submit_round, int):
            return diag
        meta = getattr(instance, "meta", None) or {}
        expected = list(meta.get("expected_outputs") or []) if isinstance(meta, dict) else []
        if len(expected) != len(bits):
            expected = [getattr(instance, "ground_truth", None)] * len(bits)
        messages = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        wrong = [i for i, bit in enumerate(bits) if bit == "0"]
        if not wrong:
            return diag
        # A value that already occurs in the raw data or the task text (e.g. a
        # small component count among node ids) proves nothing: no count then.
        public = json.dumps(list(getattr(instance, "shards", None) or []), default=str) \
            + "\n" + str(getattr(instance, "task_prompt", "") or "")
        seen = 0
        for i in wrong:
            pattern = _value_token(_to_num(expected[i]))
            if pattern is None or re.search(pattern, public):
                return diag
            bodies = [str(m.get("body") or "") for m in messages
                      if m.get("dst") == i and m.get("round_delivered") == submit_round]
            own = [m for m in messages if m.get("src") == i]
            if own:
                last = max(int(m.get("round_sent") or 0) for m in own)
                bodies += [str(m.get("body") or "") for m in own if int(m.get("round_sent") or 0) == last]
            if any(re.search(pattern, body) for body in bodies):
                seen += 1
        diag["n_answer_seen"] = seen
    except Exception:  # noqa: BLE001 - diagnostics never raise
        pass
    return diag


def refine_card(diag: Any, output: Any, instance: Any) -> Any:
    """Apply every opt-in card refinement (local-only, then answer-seen)."""

    return apply_answer_seen(apply_local_only(diag, output, instance), output, instance)


def _score_agents(instance: Any, answers: list[Any]) -> list[bool] | None:
    """Per-agent exact correctness by canonical-answer comparison (the
    fallback when the scorer supplies no per-agent booleans); None without
    expected outputs or a ground truth."""

    meta = getattr(instance, "meta", None) or {}
    expected = list(meta.get("expected_outputs") or []) if isinstance(meta, dict) else []
    if len(expected) != len(answers):
        truth = getattr(instance, "ground_truth", None)
        if truth is None:
            return None
        expected = [truth for _ in answers]
    out = []
    for answer, target in zip(answers, expected):
        out.append(answer is not None and _canonical(answer) == _canonical(target))
    return out


# --------------------------------------------------------------------------- #
# Whitelist (the ONE place renderers read a card through)
# --------------------------------------------------------------------------- #


def _nonneg_int(value: Any, cap: int = 10**9) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > cap:
        return None
    return int(value)


def sanitize_diag(diag: Any) -> dict[str, Any] | None:
    """Closed-vocabulary, type-checked copy of a card (None if not a mapping).

    Unknown keys are dropped; a malformed value becomes None, and the
    optional counts (``n_near_miss``, ``n_local_only``, ``n_answer_seen``)
    are left out when malformed or zero.  Nothing free-text survives, so no
    answer or ground-truth fragment can ride through a card into a prompt.
    The active task's ``sanitize_extra`` (when set) adds the fields it
    whitelists from the raw card.
    """

    if not isinstance(diag, Mapping):
        return None
    out: dict[str, Any] = {}
    s_value = diag.get("S")
    out["S"] = (
        round(float(s_value), 4)
        if isinstance(s_value, (int, float)) and not isinstance(s_value, bool)
        and 0.0 <= float(s_value) <= 1.0
        else None
    )
    bits = diag.get("agent_correct")
    out["agent_correct"] = bits if isinstance(bits, str) and _BITS.match(bits) else None
    out["n_distinct_answers"] = _nonneg_int(diag.get("n_distinct_answers"), 10_000)
    majority = diag.get("majority_correct")
    out["majority_correct"] = majority if isinstance(majority, bool) else None
    if out["majority_correct"] is None and diag.get("plurality_tie") is True:
        out["plurality_tie"] = True
    fclass = diag.get("failure_class")
    out["failure_class"] = fclass if fclass in FAILURE_CLASSES else None
    shape = diag.get("answer_shape")
    out["answer_shape"] = shape if shape in ANSWER_SHAPES else None
    tokens = diag.get("phase_tokens")
    if isinstance(tokens, list) and 0 < len(tokens) <= 16:
        clean = [_nonneg_int(t) for t in tokens]
        out["phase_tokens"] = clean if all(t is not None for t in clean) else None
    else:
        out["phase_tokens"] = None
    out["submit_round"] = _nonneg_int(diag.get("submit_round"), 10_000)
    out["n_messages"] = _nonneg_int(diag.get("n_messages"), 10**6)
    out["n_lost_messages"] = _nonneg_int(diag.get("n_lost_messages"), 10**6)
    near = _nonneg_int(diag.get("n_near_miss"), 10_000)
    if near:
        out["n_near_miss"] = near
    local = _nonneg_int(diag.get("n_local_only"), 10_000)
    if local:
        out["n_local_only"] = local
    seen = _nonneg_int(diag.get("n_answer_seen"), 10_000)
    if seen:
        out["n_answer_seen"] = seen
    err = diag.get("error_class")
    if err in ERROR_CLASSES:
        out["error_class"] = err
    extra = get_task().sanitize_extra
    if extra is not None:
        out.update(extra(diag) or {})
    return out


def ensure_diag(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """The row's sanitized card; a row without one (e.g. its diagnosis
    raised) gets a minimal card derived from its row facts (``S``,
    ``error``, ``infra``)."""

    if not isinstance(row, Mapping):
        return None
    card = sanitize_diag(row.get("diag"))
    if card is not None:
        return card
    derived = _empty_diag(row, None)
    err = _error_class(row)
    if err is not None:
        derived["failure_class"] = _class_for_error(err)
        derived["error_class"] = err
    elif isinstance(derived["S"], float) and derived["S"] >= 1.0:
        derived["failure_class"] = "ok"
    return sanitize_diag(derived)


# --------------------------------------------------------------------------- #
# Failure map + rendering
# --------------------------------------------------------------------------- #

_CLASS_RANK = {name: index for index, name in enumerate(FAILURE_CLASSES)}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def dominant_failure_class(classes: Any) -> str | None:
    """The dominant class of one unit's runs (the ONE tie-break rule).

    ``classes``: a ``{class: count}`` mapping or an iterable of class names
    (None / unknown names ignored).  Failing classes win over ``ok``; among
    them the most frequent wins and a count tie goes to the class listed
    first in :data:`FAILURE_CLASSES` (precision-near-miss, local-only,
    divergent, consensus-wrong, ...).
    ``"ok"`` when every counted run is ok; None when nothing is counted.
    The failure map applies this rule and the scheduler's target briefs read
    the map's dominant class, so a unit never carries two different dominant
    classes in one prompt."""

    if isinstance(classes, Mapping):
        counts = Counter(
            {k: int(v) for k, v in classes.items() if k in FAILURE_CLASSES}
        )
    else:
        counts = Counter(c for c in (classes or []) if c in FAILURE_CLASSES)
    failing = {k: v for k, v in counts.items() if k != "ok" and v > 0}
    if failing:
        return sorted(failing, key=lambda k: (-failing[k], _CLASS_RANK[k]))[0]
    return "ok" if counts.get("ok") else None


#: Planner-facing label of a failing unit whose runs carry no diagnosis
#: (e.g. a swallowed diagnosis error).  Not a card class
#: (:func:`sanitize_diag` drops it from cards): only the failure map
#: assigns it.
UNCLASSIFIED = "unknown"
_UNCLASSIFIED_GLOSS = (
    f"  {UNCLASSIFIED}: the incumbent fails this unit but its runs carry no "
    "diagnosis (e.g. cached before diagnosis existed)"
)


def build_failure_map(
    rows_by_unit: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    attempts: Mapping[str, int] | None = None,
    refuted: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """(answer shape x failure class) -> units, over each unit's rows.

    ``rows_by_unit``: unit id -> that unit's executed rows, normally all
    from one program (facts dicts carrying ``diag``; each is read through
    :func:`ensure_diag`, so a row without a card gets a minimal one).
    ``attempts`` / ``refuted`` (optional ledger annotations): how many
    proposals have targeted a unit so far, and the mechanism labels refuted
    on it.

    Infra rows are reported but never enter a unit's mean S or its dominant
    class (not the program's fault)."""

    units: dict[str, dict[str, Any]] = {}
    for unit_id in sorted(rows_by_unit):
        cards = [ensure_diag(row) for row in rows_by_unit[unit_id] or []]
        cards = [c for c in cards if c is not None]
        scored = [c for c in cards if c.get("failure_class") != "infra"]
        classes = Counter(
            c.get("failure_class") or UNCLASSIFIED for c in scored
        )
        s_values = [c["S"] for c in scored if isinstance(c.get("S"), float)]
        shapes = Counter(c.get("answer_shape") for c in cards if c.get("answer_shape"))
        dominant = dominant_failure_class(classes)
        if dominant == "ok" and UNCLASSIFIED in classes:
            dominant = None
        if dominant is None:
            if (
                scored
                and len(s_values) == len(scored)
                and min(s_values) >= 1.0 - 1e-9
            ):
                # Undiagnosed runs that all scored S=1 are solved runs.
                dominant = "ok"
            elif not scored and cards:
                dominant = "infra"
            else:
                dominant = UNCLASSIFIED
        entry: dict[str, Any] = {
            "n": len(scored),
            "n_infra": len(cards) - len(scored),
            "S_mean": round(_mean(s_values), 4) if s_values else None,
            "shape": shapes.most_common(1)[0][0] if shapes else None,
            "classes": dict(sorted(classes.items())),
            "dominant": dominant,
            "solved": bool(scored) and dominant == "ok",
            "agent_correct": [c.get("agent_correct") for c in scored],
        }
        if attempts and unit_id in attempts:
            count = _nonneg_int(attempts[unit_id], 10_000)
            if count is not None:
                entry["attempts"] = count
        if refuted and unit_id in refuted:
            labels = [
                re.sub(r"[^A-Za-z0-9_\-+ ]", "", str(x))[:40]
                for x in list(refuted[unit_id])[:6]
            ]
            entry["refuted"] = [x for x in labels if x]
        units[unit_id] = entry
    cells: dict[tuple[str, str], list[str]] = {}
    infra_only: list[str] = []
    for unit_id, entry in units.items():
        dominant = entry["dominant"]
        if dominant == "ok":
            continue
        if dominant == "infra":
            # Never measured (infrastructure only): not a program failure.
            infra_only.append(unit_id)
            continue
        if dominant == UNCLASSIFIED and not (
            isinstance(entry["S_mean"], float) and entry["S_mean"] < 1.0 - 1e-9
        ):
            continue
        key = (entry["shape"] or "unknown-shape", dominant)
        cells.setdefault(key, []).append(unit_id)
    ordered = sorted(
        cells.items(),
        key=lambda kv: (-len(kv[1]), _CLASS_RANK.get(kv[0][1], 99), kv[0][0]),
    )
    return {
        "units": units,
        "cells": [
            {"shape": shape, "failure_class": fclass, "units": sorted(members)}
            for (shape, fclass), members in ordered
        ],
        "solved": sorted(u for u, e in units.items() if e["solved"]),
        "infra_only": sorted(infra_only),
        "n_rows": sum(e["n"] + e["n_infra"] for e in units.values()),
    }


def _unit_brief(unit_id: str, entry: Mapping[str, Any]) -> str:
    s_mean = entry.get("S_mean")
    bits = [f"S={s_mean:.2f}" if isinstance(s_mean, float) else "S=n/a"]
    bits.append(f"runs={entry.get('n', 0)}")
    if entry.get("attempts") is not None:
        bits.append(f"attempted {entry['attempts']}x")
    if entry.get("refuted"):
        bits.append("refuted: " + ", ".join(entry["refuted"]))
    return f"{clean_unit_id(unit_id) or '?'} ({'; '.join(bits)})"


def failure_glossary_lines(classes: Iterable[str] | None = None) -> list[str]:
    wanted = list(classes) if classes is not None else list(FAILURE_CLASSES)
    glossary = failure_class_glossary()
    return [
        f"  {name}: {glossary[name]}"
        for name in FAILURE_CLASSES
        if name in wanted and name != "ok"
    ]


def render_failure_map(fmap: Mapping[str, Any], *, max_chars: int = 3000) -> str:
    """Planner-facing text of a failure map (answer-free, bounded).

    Whole lines only: when the budget runs out the remaining cells are
    summarized as ``... +k more cells``."""

    if not isinstance(fmap, Mapping):
        return ""
    units = fmap.get("units") or {}
    cells = list(fmap.get("cells") or [])
    head = (
        "=== FAILURE MAP (current incumbent on TRAINING units; answer-free) ===\n"
        "Each line: answer shape x failure class -> units (incumbent mean S "
        "over its runs). Answer shape comes from each task's public Output "
        "sentence.\n"
    )
    present = sorted({c.get("failure_class") for c in cells if c.get("failure_class")})
    glossary = ""
    if present:
        gloss = failure_glossary_lines(present)
        if UNCLASSIFIED in present:
            gloss.append(_UNCLASSIFIED_GLOSS)
        glossary = "Failure classes seen:\n" + "\n".join(gloss) + "\n"
    line_cap = max(80, min(600, max_chars // 3))
    lines: list[str] = []
    for cell in cells:
        prefix = f"- {cell.get('shape')} x {cell.get('failure_class')}: "
        members = list(cell.get("units") or [])
        line = prefix
        for index, unit_id in enumerate(members):
            brief = _unit_brief(unit_id, units.get(unit_id) or {})
            more = f" +{len(members) - index} more units"
            candidate = line + (", " if index else "") + brief
            if len(candidate) + len(more) > line_cap and index:
                line += more
                break
            line = candidate
        lines.append(line[:line_cap])
    solved = [clean_unit_id(u) or "?" for u in fmap.get("solved") or []]
    tail = (
        "Solved by the incumbent on every run: " + ", ".join(solved)
        if solved else "Solved by the incumbent on every run: none"
    )[:line_cap] + "\n"
    infra_only = [clean_unit_id(u) or "?" for u in fmap.get("infra_only") or []]
    if infra_only:
        tail = (
            "Not measured (infrastructure failures only): "
            + ", ".join(infra_only)
        )[:line_cap] + "\n" + tail
    failing_units = [
        u for u, e in units.items()
        if isinstance(e, Mapping)
        and isinstance(e.get("S_mean"), float)
        and e["S_mean"] < 1.0 - 1e-9
    ]
    if not cells and not failing_units:
        lines.append("- no failing training unit in the incumbent's rows")
    if len(head) > max_chars:
        return head[:max_chars]
    text = head
    reserve = len(f"... +{len(lines)} more cells\n")
    if len(text) + len(glossary) + reserve <= max_chars // 2 + reserve:
        text += glossary
    kept = 0
    for line in lines:
        if len(text) + len(line) + 1 + reserve + len(tail) > max_chars:
            break
        text += line + "\n"
        kept += 1
    if kept < len(lines) and len(text) + reserve <= max_chars:
        text += f"... +{len(lines) - kept} more cells\n"
    if len(text) + len(tail) <= max_chars:
        text += tail
    return text


def _card_line(index: int, card: Mapping[str, Any]) -> str:
    parts = [f"run{index}:"]
    s_value = card.get("S")
    parts.append(f"S={s_value:.2f}" if isinstance(s_value, float) else "S=n/a")
    parts.append(f"class={card.get('failure_class') or 'unknown'}")
    if card.get("agent_correct"):
        parts.append(f"agents_right={card['agent_correct']}")
    if card.get("n_distinct_answers") is not None:
        parts.append(f"distinct_answers={card['n_distinct_answers']}")
    if card.get("majority_correct") is not None:
        parts.append(
            "plurality=" + ("right" if card["majority_correct"] else "wrong")
        )
    elif card.get("plurality_tie"):
        parts.append("plurality=tie")
    if card.get("submit_round") is not None:
        parts.append(f"submit@r{card['submit_round']}")
    if card.get("n_messages") is not None:
        parts.append(f"msgs={card['n_messages']}")
    if card.get("n_lost_messages"):
        parts.append(f"lost_bodies={card['n_lost_messages']}")
    if card.get("n_near_miss"):
        parts.append(f"near_miss_agents={card['n_near_miss']}")
    if card.get("n_local_only"):
        parts.append(f"own_shard_only_agents={card['n_local_only']}")
    if card.get("n_answer_seen"):
        parts.append(f"answer_seen_agents={card['n_answer_seen']}")
    tokens = card.get("phase_tokens")
    if tokens:
        parts.append(
            "tokens/phase=[" + ",".join(str(t) for t in tokens[:-1])
            + f"|submit {tokens[-1]}]"
        )
    if card.get("error_class"):
        parts.append(f"err={card['error_class']}")
    suffix = get_task().card_suffix
    return " ".join(parts) + (suffix(card) if suffix is not None else "")


def render_diag_card(unit_id: str, diag_rows: Iterable[Any]) -> str:
    """One unit's diagnosis block: a head line plus one line per run.

    ``diag_rows`` may be rows (facts dicts carrying ``diag``) or bare
    cards; everything is read through :func:`sanitize_diag`."""

    cards: list[dict[str, Any]] = []
    for item in diag_rows or []:
        if isinstance(item, Mapping) and "diag" in item:
            card = ensure_diag(item)
        elif isinstance(item, Mapping) and "failure_class" in item:
            card = sanitize_diag(item)
        else:
            card = ensure_diag(item) if isinstance(item, Mapping) else None
        if card is not None:
            cards.append(card)
    uid = clean_unit_id(unit_id) or "?"
    if not cards:
        return f"[{uid}] no diagnosed runs"
    scored = [c for c in cards if c.get("failure_class") != "infra"]
    s_values = [c["S"] for c in scored if isinstance(c.get("S"), float)]
    shape = next((c.get("answer_shape") for c in cards if c.get("answer_shape")), None)
    classes = Counter(c.get("failure_class") or "unknown" for c in cards)
    head = (
        f"[{uid}] shape={shape or 'n/a'} runs={len(cards)} mean S="
        + (f"{sum(s_values) / len(s_values):.2f}" if s_values else "n/a")
        + " classes: "
        + ", ".join(f"{k} x{v}" for k, v in sorted(classes.items()))
    )
    lines = [head]
    for index, card in enumerate(cards, start=1):
        lines.append("  " + _card_line(index, card))
    return "\n".join(lines)


__all__ = [
    "ANSWER_SHAPES",
    "ERROR_CLASSES",
    "FAILURE_CLASSES",
    "FAILURE_CLASS_GLOSSARY",
    "ATTEMPT_ENV",
    "LOCAL_ONLY_ENV",
    "apply_answer_seen",
    "apply_local_only",
    "attempt_diag_enabled",
    "refine_card",
    "local_only_enabled",
    "UNCLASSIFIED",
    "answer_shape_from_text",
    "build_failure_map",
    "clean_unit_id",
    "diagnose_execution",
    "dominant_failure_class",
    "ensure_diag",
    "failure_class_glossary",
    "failure_glossary_lines",
    "instance_answer_shape",
    "output_sentence",
    "render_diag_card",
    "render_failure_map",
    "sanitize_diag",
]
