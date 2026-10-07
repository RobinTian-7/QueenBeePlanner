"""The task interface of QueenBee-Evo.

A :class:`TaskSpec` names everything a task family supplies to the loop, the
racer, the screen, the planner prompt and the evaluator.  Every field left
``None`` keeps the built-in Silo-Bench behaviour of its dispatch point, so
the Silo-Bench task (:mod:`queenbee.tasks.silo`) is a :class:`TaskSpec` with
no hooks.

Dispatch points read the active task (:func:`queenbee.tasks.get_task`) when
they run, never at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

#: Information goals: every agent submits and is scored, or agent 0 alone.
GOALS: tuple[str, ...] = ("all_agents", "sink")

_NAME = re.compile(r"[a-z][a-z0-9_\-]{0,39}")
_RUNG = re.compile(r"[ox](\d+)")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _ints(value: Sequence[int], label: str, *, low: int = 2) -> tuple[int, ...]:
    out = tuple(int(v) for v in value)
    if not out or any(v < low for v in out) or len(set(out)) != len(out):
        raise ValueError(f"{label}: distinct integers >= {low} expected, got {value!r}")
    return out


@dataclass(frozen=True, eq=False)
class TaskSpec:
    """One task family (see the module docstring; ``None`` = built-in).

    Units and ids
        ``goal``: ``"all_agents"`` or ``"sink"``.  ``rungs``: rung -> team
        size (rung names are ``o<n>`` / ``x<n>``); ``rung_cost``: rung ->
        execution-equivalents of one run (built-in: 2 at n >= 10, else 1);
        ``rung_order`` / ``rung_next``: curriculum order and moves (an empty
        ``rung_next`` disables the curriculum).  ``test_ids`` (sealed),
        ``dev_ids`` (a sequence or a zero-argument callable) and
        ``template_pattern`` (a regex with at most one group that finds
        template ids in any text) are given together or not at all.

    Programs
        ``fp_ns`` / ``fp_scheme``: team sizes and scheme id of the behaviour
        fingerprint (built-in ``(2, 5, 10)``).  ``mint_n_agents``: team size of
        the mint dry run, the planner request, the prompt's budget line and
        the template statements (built-in 5).  ``seed_source()``: the seed
        program (also the census program and the call-cap reference of the
        screen).

    Instances
        ``instance_path(unit_or_case_id)`` -> instance file, or None when
        there is none (a real loop run of a task with this hook needs no
        ladder manifest);
        ``load_instance(path)`` -> instance object; ``t_templates(T)`` -> the
        planner's task statements of the T templates.

    Execution
        ``execute(req) -> row`` replaces the default executor;
        ``score_fn`` / ``diag_fn`` are passed to
        :func:`queenbee.program.execute.evaluate_python_source_on_cases`;
        ``worker_env``: extra environment variables of the worker sandbox;
        ``trace_row(row, trace, instance, *, source, goal) -> row`` runs after
        :func:`queenbee.evo.credit.trace_row`; ``finalize_row(row) -> row``
        runs on every row of :func:`queenbee.evo.common.real_execute` and of
        the evaluator.

    Diagnosis cards
        ``sanitize_extra(card) -> {field: value}`` adds whitelisted card
        fields to :func:`queenbee.evo.diagnosis.sanitize_diag`;
        ``card_suffix(card) -> text`` is appended to each run line;
        ``glossary`` overrides failure-class meanings where they are read;
        ``per_agent_cards=False``: the ledger's post-change summary leaves
        out the per-agent counts when no card carries per-agent correctness.
        ``trace_unit_evidence=False``: unit classes take no near-miss /
        structural evidence from trace directories (those heuristics read
        Silo-Bench per-agent answers); traces still enrich census rows.

    Screen (S0)
        ``screen_ns``: team sizes simulated (built-in ``(5, 10)``);
        ``wi_vocabulary()``: the work-instruction lint vocabulary;
        ``screen_program`` / ``program_fp_key``: full replacements of
        :func:`queenbee.evo.screen.screen_for_task` /
        :func:`queenbee.evo.screen.fp_key_for_task` (same call).

    Planner texts
        ``header``, ``unit_legend``, ``render_api_card(*, budgets,
        preserve_dups)`` and ``diag_cards_block(unit_rows, targets)``.

    Run defaults
        ``thresholds`` (a mapping or a JSON path) when no thresholds file is
        given; ``cli_defaults``: option dest -> default for the command-line
        tools (an option given on the command line wins);
        ``default_split(split_seed) -> {"name", "T", "V"}`` when no split file
        is given (a real loop run of a task with this hook needs no split
        file).

    Evaluator
        ``summary_fields(rows) -> {field: value}``: extra fields of the
        evaluator's repeat aggregates and arm summaries, computed from the
        rows they cover.
    """

    name: str
    goal: str | None = None
    rungs: Mapping[str, int] | None = None
    rung_cost: Mapping[str, int] | None = None
    rung_order: Mapping[str, int] | None = None
    rung_next: Mapping[str, str] | None = None
    test_ids: Sequence[str] | None = None
    dev_ids: Sequence[str] | Callable[[], Sequence[str]] | None = None
    template_pattern: str | None = None
    fp_ns: Sequence[int] | None = None
    fp_scheme: str | None = None
    mint_n_agents: int | None = None
    seed_source: Callable[[], str] | None = None
    instance_path: Callable[[str], str | Path | None] | None = None
    load_instance: Callable[[Path], Any] | None = None
    t_templates: Callable[[Sequence[str]], list[dict[str, str]]] | None = None
    execute: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None
    score_fn: Callable[..., dict[str, Any]] | None = None
    diag_fn: Callable[..., Any] | None = None
    worker_env: Mapping[str, str] | None = None
    trace_row: Callable[..., dict[str, Any]] | None = None
    finalize_row: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None
    sanitize_extra: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    card_suffix: Callable[[Mapping[str, Any]], str] | None = None
    glossary: Mapping[str, str] | None = None
    per_agent_cards: bool = True
    trace_unit_evidence: bool = True
    screen_ns: Sequence[int] | None = None
    wi_vocabulary: Callable[[], Any] | None = None
    screen_program: Callable[..., Any] | None = None
    program_fp_key: Callable[..., str | None] | None = None
    header: str | None = None
    unit_legend: str | None = None
    render_api_card: Callable[..., str] | None = None
    diag_cards_block: Callable[[Mapping[str, list], list[str]], str] | None = None
    thresholds: Mapping[str, Any] | str | Path | None = None
    cli_defaults: Mapping[str, Any] = field(default_factory=dict)
    default_split: Callable[[int], Mapping[str, Any]] | None = None
    summary_fields: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        def put(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        if not isinstance(self.name, str) or not _NAME.fullmatch(self.name):
            raise ValueError(f"task name {self.name!r}: use [a-z][a-z0-9_-]{{0,39}}")
        if self.goal is not None and self.goal not in GOALS:
            raise ValueError(f"task {self.name}: goal {self.goal!r} not in {GOALS}")
        if self.rungs is not None:
            rungs = {str(k): int(v) for k, v in dict(self.rungs).items()}
            for rung, n in rungs.items():
                match = _RUNG.fullmatch(rung)
                if not match or int(match.group(1)) != n or n < 2:
                    raise ValueError(f"task {self.name}: rung {rung!r} -> {n} (want o<n> / x<n> -> n >= 2)")
            put("rungs", rungs)
        for name in ("rung_cost", "rung_order"):
            value = getattr(self, name)
            if value is not None:
                put(name, {str(k): int(v) for k, v in dict(value).items()})
        if self.rung_next is not None:
            put("rung_next", {str(k): str(v) for k, v in dict(self.rung_next).items()})
        ids = (self.test_ids, self.dev_ids, self.template_pattern)
        if any(v is not None for v in ids) and any(v is None for v in ids):
            raise ValueError(f"task {self.name}: test_ids, dev_ids and template_pattern go together")
        if self.template_pattern is not None:
            rx = re.compile(self.template_pattern)
            if rx.groups > 1:
                raise ValueError(f"task {self.name}: template_pattern needs at most one group")
            put("test_ids", tuple(str(t) for t in self.test_ids or ()))
            if not callable(self.dev_ids):
                put("dev_ids", tuple(str(t) for t in self.dev_ids or ()))
                if set(self.test_ids) & set(self.dev_ids):
                    raise ValueError(f"task {self.name}: test_ids and dev_ids overlap")
            listed = list(self.test_ids) + ([] if callable(self.dev_ids) else list(self.dev_ids))
            for template in listed:
                if rx.findall(template) != [template]:
                    raise ValueError(f"task {self.name}: template_pattern does not find {template!r}")
        if self.fp_ns is not None:
            put("fp_ns", _ints(self.fp_ns, f"task {self.name}: fp_ns"))
        if self.screen_ns is not None:
            put("screen_ns", _ints(self.screen_ns, f"task {self.name}: screen_ns"))
        if self.mint_n_agents is not None:
            put("mint_n_agents", _ints([self.mint_n_agents], f"task {self.name}: mint_n_agents")[0])
        if self.worker_env is not None:
            env = {str(k): str(v) for k, v in dict(self.worker_env).items()}
            bad = [k for k in env if not _ENV_NAME.fullmatch(k)]
            if bad:
                raise ValueError(f"task {self.name}: bad worker_env names {bad}")
            put("worker_env", env)
        if self.glossary is not None:
            put("glossary", {str(k): str(v) for k, v in dict(self.glossary).items()})
        put("cli_defaults", dict(self.cli_defaults or {}))
        put("per_agent_cards", bool(self.per_agent_cards))
        put("trace_unit_evidence", bool(self.trace_unit_evidence))


__all__ = ["GOALS", "TaskSpec"]
