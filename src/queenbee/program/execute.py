"""Runtime side: run team programs and simulate their communication policy.

A program source is executed unchanged (no planner or repair call) on one
benchmark case in the sandboxed ``CodeProcessRunner`` and scored by the
benchmark scorer; the run is reduced to the row facts the evolution loop
reads (quality ``S``, cost ``C``, an answer-free diagnosis card).  Also: the
seed program, the behaviour fingerprint simulator and the
infrastructure-error classifier.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from exp_graph.mas.python_code import DEFAULT_MESSAGE_ONLY_V2_PROGRAM
from exp_graph.mas.python_code_generation import PythonCodePlanningResult
from exp_graph.mas.python_code_runner import (
    CodeProcessRunner,
    PythonExecutionLimits,
)

from queenbee.bench.config import RunConfig
from queenbee.bench.engine import (
    _build_python_execution_payload,
    _protocol_adapter,
    _resolved_python_execution_timeout,
    _score_python_execution,
)
from queenbee.program.budgets import PythonRunBudgets
from queenbee.program.genome import genome_region
from queenbee.program.phase_seed import build_phase_seed_source
from queenbee.tasks import get_task

#: ``score_fn(facts, *, instance, output, payload, score) -> facts``: called
#: once per completed run with the benchmark's row facts; its return value
#: replaces them (a failed run is never re-scored).
ScoreFn = Callable[..., dict[str, Any]]
#: ``diag_fn(output, instance, facts, **context) -> card``: builds the
#: diagnosis card in place of ``queenbee.evo.diagnosis.diagnose_execution``
#: (same call; ``output`` is None for a failed run).
DiagFn = Callable[..., Any]

# Markers (matched against the lower-cased error type and message) of
# failures that are infrastructure (transport, rate limit, server, timeout)
# rather than the program's fault: such a run becomes an infra row, not S = 0.
_INFRA_ERROR_MARKERS = (
    "connection",
    "timeout",
    "timed out",
    "rate limit",
    "ratelimit",
    "service unavailable",
    "bad gateway",
    "model not found",   # transient HTTP 400 from a server replica lacking the model
    "429",
    "502",
    "503",
    # The wall-clock guard's LLMTimeoutError ("LLM call exceeded <t>s") is
    # infrastructure too, although its text names no timeout.
    "llm call exceeded",
    "llmtimeouterror",
    "deadline_exceeded",
    "upstream_error",
    "upstream failed",
    "remoteprotocolerror",
    "incomplete chunked read",
    "peer closed connection",
    "connection reset",
)


def seed_python_source(
    worker_contract: str, *, base: str = "sfs_phase"
) -> tuple[str, str, str]:
    """(origin, title, source) of the deterministic phase-structured seed.

    The seed is the phase-structured program of the ``message_only_v2``
    worker contract (``base="sfs_phase"``): a ``PHASES`` list (relay, then
    broadcast_last) interpreted by fixed policy functions, spliced into the
    contract's default program.  It needs no planner call.  Under the
    default Silo-Bench task this is the base text of the seed program (v1),
    the text the census runs; :mod:`queenbee.evo.seed` derives the text
    evolution starts from by two behaviour-preserving extensions (phase
    work-instruction forwarding and the ``digest`` phase kind).
    """

    if worker_contract != "message_only_v2":
        raise ValueError(
            f"unknown python worker contract {worker_contract!r}; "
            "the seed program needs message_only_v2"
        )
    wanted = str(base or "sfs_phase").strip().lower()
    if wanted not in ("sfs_phase", "phase_sfs", "sfs-phase"):
        raise ValueError(f"unknown seed base {base!r}; the seed is 'sfs_phase'")
    return (
        "seed:python_sfs_phase",
        "Python phase-structured sfs (relay + broadcast_last)",
        build_phase_seed_source(DEFAULT_MESSAGE_ONLY_V2_PROGRAM.strip() + "\n"),
    )


def classify_row_error(row: dict[str, Any]) -> str | None:
    """Answer-free failure class of a row.

    ``"infra"`` (an infrastructure row), ``"format"`` (AnswerFormatError),
    ``"budget"`` (BudgetError), ``"undefined_name"`` (an undefined-name
    error) or ``"exec"`` (any other error); None for a non-dict or a row
    without an error."""

    if not isinstance(row, dict):
        return None
    if row.get("infra"):
        return "infra"
    err = str(row.get("error") or "")
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


def task_worker_env() -> dict[str, str]:
    """The active task's extra environment of the worker sandbox."""

    return dict(get_task().worker_env or {})


def _is_transport_error(exc: Exception) -> bool:
    """True when an exception (e.g. from a planner call) is a transport,
    rate-limit, server or timeout failure worth retrying, judged by its type
    name and message."""
    haystack = f"{type(exc).__name__} {exc}".lower()
    if "apiconnection" in haystack or "internalservererror" in haystack:
        return True
    return any(marker in haystack for marker in _INFRA_ERROR_MARKERS)


def _run_config(
    *,
    llm_provider: str,
    worker_model: str,
    goal: str,
    worker_contract: str,
    max_parallel_agents: int,
    request_timeout: float,
    n_agents: int,
    budgets: PythonRunBudgets | None = None,
) -> RunConfig:
    budgets = budgets or PythonRunBudgets()
    return RunConfig(
        benchmark="silo_bench",
        silo_eval_mode=goal,
        llm_provider=llm_provider,
        model_name=worker_model,
        temperature=0.0,
        n_agents=n_agents,
        max_parallel_agents=max(1, int(max_parallel_agents)),
        python_worker_contract=worker_contract,
        request_timeout=float(request_timeout),
        max_rounds=budgets.max_rounds,
        python_max_model_calls=budgets.max_model_calls,
        python_max_completion_tokens=budgets.max_completion_tokens,
        python_max_messages=budgets.max_messages,
    )


def _facts_from_score(score: Any, *, goal: str) -> dict[str, Any]:
    """Row facts of a scored run.

    Goal ``all_agents``: ``S`` is the Silo-Bench success rate and
    ``stage_score`` the partial score ``P``.  Goal ``sink``: ``S`` is the
    sink agent's partial score and ``stage_score`` 1.0 for an exact sink
    answer, else 0.0.  ``C`` is the run's token count and ``D`` its
    communication density."""
    extra = dict(score.extra or {})
    if goal == "all_agents":
        quality = float(extra.get("paper_S", 0.0))
        stage = float(extra.get("paper_P", score.partial or 0.0))
    else:
        quality = float(extra.get("sink_partial", score.partial or 0.0))
        stage = 1.0 if extra.get("sink_exact") else 0.0
    return {
        "infra": None,
        "execution_class": "completed",
        "success": bool(score.success),
        "S": round(quality, 6),
        "stage_score": round(stage, 6),
        "C": float(score.tokens or 0),
        "D": float(extra.get("communication_density", 0.0)),
        "rounds_executed": extra.get("rounds_executed"),
        "worker_model_calls": extra.get("worker_model_calls"),
        "prompt_tokens": extra.get("prompt_tokens"),
        "completion_tokens": extra.get("completion_tokens"),
        "model_calls": int(score.n_model_calls or 0),
        "n_messages": int(score.n_messages or 0),
    }


def _failure_facts(failure: dict[str, Any] | None) -> dict[str, Any]:
    """Row facts of a failed run: ``{"infra": text}`` when the error matches
    an infrastructure marker (never scored; the racer and the evaluator
    re-run such rows), else an algorithm failure scored ``S = 0`` with the
    error text.  Texts are clipped to 500 characters."""
    failure = failure or {}
    message = f"{failure.get('error_type')}: {failure.get('message')}"
    if any(marker in message.lower() for marker in _INFRA_ERROR_MARKERS):
        return {"infra": message[:500]}
    return {
        "infra": None,
        "execution_class": "algorithm_failure",
        "success": False,
        "S": 0.0,
        "stage_score": 0.0,
        "C": 0.0,
        "error": message[:500],
    }


def execute_python_source_on_case(
    *,
    source: str,
    instance: Any,
    llm_provider: str,
    worker_model: str,
    goal: str,
    worker_contract: str,
    max_parallel_agents: int,
    request_timeout: float,
    artifacts_dir: Path,
    budgets: PythonRunBudgets | None = None,
    score_fn: ScoreFn | None = None,
    diag_fn: DiagFn | None = None,
) -> dict[str, Any]:
    """Execute one program source, unchanged, on one case; reduce the run to
    row facts.

    ``score_fn`` (see :data:`ScoreFn`) re-scores a completed run and
    ``diag_fn`` (see :data:`DiagFn`) builds the diagnosis card; both None =
    the benchmark's own scoring and card.  With ``QB_TRACE_DIR`` set, a
    completed run also writes a trace (:func:`_maybe_dump_trace`)."""

    n_agents = int(instance.n_agents)
    cfg = _run_config(
        llm_provider=llm_provider,
        worker_model=worker_model,
        goal=goal,
        worker_contract=worker_contract,
        max_parallel_agents=max_parallel_agents,
        request_timeout=request_timeout,
        n_agents=n_agents,
        budgets=budgets,
    )
    task_adapter = _protocol_adapter(instance, information_goal=goal)
    global_task = task_adapter.build_global_task()
    payload = _build_python_execution_payload(
        instance=instance,
        cfg=cfg,
        task_adapter=task_adapter,
        global_task=global_task,
        n_agents=n_agents,
    )
    runner = CodeProcessRunner(
        PythonExecutionLimits(
            timeout_seconds=_resolved_python_execution_timeout(
                cfg, n_agents=n_agents
            ),
            cpu_seconds=cfg.python_cpu_seconds,
            memory_mb=cfg.python_memory_mb,
            max_output_bytes=cfg.python_max_output_bytes,
        ),
        extra_env=task_worker_env(),
    )
    execution = runner.run(source, payload)
    if not execution.runtime_success or execution.output is None:
        facts = _failure_facts(execution.failure)
        _attach_diag(
            facts, None, instance, diag_fn=diag_fn, source=source,
            ledger=getattr(execution, "ledger", None), goal=goal,
        )
        return facts
    planning = PythonCodePlanningResult(
        source=source,
        execution=execution,
        artifacts_dir=artifacts_dir,
        provenance="lineage_frozen_source",
        architect_prompt=None,
        attempts=[],
        planner_model_calls=0,
        repair_model_calls=0,
    )
    score = _score_python_execution(
        planning,
        instance=instance,
        cfg=cfg,
        task_adapter=task_adapter,
        global_task=global_task,
        extra={
            "case_id": instance.case_id,
            "planner": False,
            "topology": "python:lineage",
            "objective": "balanced",
            "silo_eval_mode": goal,
        },
    )
    facts = _facts_from_score(score, goal=goal)
    if score_fn is not None:
        facts = dict(
            score_fn(
                facts,
                instance=instance,
                output=execution.output,
                payload=payload,
                score=score,
            )
        )
    _attach_diag(
        facts,
        execution.output,
        instance,
        diag_fn=diag_fn,
        per_agent_correct=(dict(score.extra or {})).get("per_agent_correct"),
        ledger=getattr(execution, "ledger", None),
        source=source,
        goal=goal,
    )
    _maybe_dump_trace(source, instance, execution.output, facts)
    return facts


def _attach_diag(
    facts: dict[str, Any],
    output: Any,
    instance: Any,
    *,
    diag_fn: DiagFn | None = None,
    **kwargs: Any,
) -> None:
    """facts['diag'] = the answer-free diagnosis card.

    ``diag_fn`` replaces ``queenbee.evo.diagnosis.diagnose_execution`` (same
    call).  Never raises: a diagnosis bug must not cost a paid execution."""

    try:
        if diag_fn is None:
            from queenbee.evo.diagnosis import diagnose_execution

            diag_fn = diagnose_execution
        facts["diag"] = diag_fn(output, instance, facts, **kwargs)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass


def _maybe_dump_trace(source: str, instance: Any, output: Any, facts: dict) -> None:
    """Opt-in execution trace (env ``QB_TRACE_DIR``; unset = no effect).

    Writes ``<case_id>_<source sha12>_<ms>.json`` with the case id, agent
    count, source sha12, row facts, the full program output and the case's
    ground truth.  The file holds the ground truth, so it never enters a
    prompt: trace readers derive only answer-free facts from it (see
    :func:`queenbee.evo.credit.trace_row`).  Write errors are ignored."""

    import os

    trace_dir = os.environ.get("QB_TRACE_DIR")
    if not trace_dir:
        return
    try:
        import hashlib
        import time as _time

        sha = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        out_dir = Path(trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "case_id": getattr(instance, "case_id", None),
            "n_agents": getattr(instance, "n_agents", None),
            "source_sha": sha,
            "facts": facts,
            "output": output.model_dump(mode="json") if output is not None else None,
            "ground_truth": getattr(instance, "ground_truth", None),
        }
        name = f"{record['case_id']}_{sha}_{int(_time.time() * 1000)}.json"
        (out_dir / name).write_text(json.dumps(record, default=str))
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass


def evaluate_python_source_on_cases(
    *,
    source: str,
    instances: tuple[Any, ...],
    seeds: tuple[int, ...],
    arm_label: str,
    llm_provider: str,
    worker_model: str,
    goal: str,
    worker_contract: str,
    max_parallel_cases: int,
    max_parallel_agents: int,
    request_timeout: float,
    artifacts_dir: Path,
    budgets: PythonRunBudgets | None = None,
    progress: Any = None,
    score_fn: ScoreFn | None = None,
    diag_fn: DiagFn | None = None,
) -> list[dict[str, Any]]:
    """One program source over a tuple of cases, up to ``max_parallel_cases``
    cases at a time.

    Rows come back in case order.  ``seeds`` pair one-to-one with
    ``instances`` but never enter the payload (the workers run at
    temperature 0): each is only recorded as its row's ``seed`` field.  A
    case whose execution raises becomes a failure row (infra or S = 0).
    ``score_fn`` / ``diag_fn``: see :func:`execute_python_source_on_case`.
    """

    if len(instances) != len(seeds):
        raise ValueError("instances and seeds must pair one-to-one")

    def _run_one(index: int) -> dict[str, Any]:
        instance = instances[index]
        try:
            row = execute_python_source_on_case(
                source=source,
                instance=instance,
                llm_provider=llm_provider,
                worker_model=worker_model,
                goal=goal,
                worker_contract=worker_contract,
                max_parallel_agents=max_parallel_agents,
                request_timeout=request_timeout,
                artifacts_dir=artifacts_dir,
                budgets=budgets,
                score_fn=score_fn,
                diag_fn=diag_fn,
            )
        except Exception as exc:  # noqa: BLE001 - a failed run is a result
            row = _failure_facts(
                {"error_type": type(exc).__name__, "message": str(exc)}
            )
            _attach_diag(
                row, None, instance, diag_fn=diag_fn, source=source, goal=goal
            )
        row["case_id"] = instance.case_id
        row["seed"] = int(seeds[index])
        if progress is not None:
            progress(f"{arm_label} {instance.case_id} done")
        return row

    workers = max(1, min(int(max_parallel_cases), len(instances)))
    if workers == 1:
        return [_run_one(index) for index in range(len(instances))]
    rows: list[dict[str, Any] | None] = [None] * len(instances)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one, index): index
            for index in range(len(instances))
        }
        for future, index in futures.items():
            rows[index] = future.result()
    return [row for row in rows if row is not None]


def _policy_sections(source: str) -> list[tuple[str, dict[str, Any]]]:
    """Candidate pure-policy sections to simulate, in order.

    First, when present, the policy section proper (from the line
    ``PHASES = [`` up to ``def main``, executed in an empty namespace).
    Then the whole genome region, with ``json`` available (the host prefix
    imports it), whenever it differs from that section: a genome with
    constants / helpers defined above PHASES (the host splices and validates
    them like any other genome code) or one without a PHASES literal."""

    out: list[tuple[str, dict[str, Any]]] = []
    lines = source.splitlines()
    legacy: str | None = None
    try:
        start = next(
            i for i, l in enumerate(lines) if l.startswith("PHASES = [")
        )
        end = next(i for i, l in enumerate(lines) if l.startswith("def main"))
    except StopIteration:
        pass
    else:
        legacy = "\n".join(lines[start:end])
        out.append((legacy, {}))
    region = genome_region(source)
    if region is not None and region.strip() != (legacy or "").strip():
        out.append((region, {"json": json}))
    return out


def simulate_program_behavior(
    source: str, *, n_agents: int, max_rounds: int, goal: str = "all_agents"
) -> tuple[Any, ...] | None:
    """Deterministic behavioural fingerprint of the communication policy.

    One entry per simulated round before the submit round in which some
    agent is not idle: ``(frozenset of send edges (src, dst), every agent's
    mode, every agent's work instruction)``.  The submit round itself is not
    in the tuple (:func:`queenbee.evo.screen.behavior_fp` adds it).
    Executes ONLY the validator-vetted pure policy section (from the PHASES
    literal to main; when that alone fails, the whole genome region - see
    :func:`_policy_sections`), never the program entrypoint.  Returns None
    when extraction/execution fails - callers must then use another
    fingerprint or treat the program as distinct, never assume
    equivalence."""

    for section, namespace in _policy_sections(source):
        fingerprint = _behavior_of(
            section, namespace, n_agents=n_agents, max_rounds=max_rounds,
            goal=goal,
        )
        if fingerprint is not None:
            return fingerprint
    return None


def _behavior_of(
    section: str,
    namespace: dict[str, Any],
    *,
    n_agents: int,
    max_rounds: int,
    goal: str,
) -> tuple[Any, ...] | None:
    """Fingerprint of one candidate section (None on any error).

    Feeds the policy the counters ``main()`` passes: ``known_source_count``
    (source ids credited on delivery, whether the body is read or not),
    ``inbox_count`` (messages delivered this round) and
    ``selected_primary = 0``."""
    try:
        exec(section, namespace)  # noqa: S102
        submit = int(namespace["plan_submit_round"](n_agents, max_rounds, goal))
        knowledge = [{i} for i in range(n_agents)]
        inbox = [0] * n_agents
        rounds: list[frozenset] = []
        for r in range(min(submit, max_rounds)):
            decisions = [
                namespace["plan_communication_turn"](
                    r, a, n_agents, goal, 0, len(knowledge[a]), inbox[a]
                )
                for a in range(n_agents)
            ]
            edges = []
            modes = []
            wis = []
            for a, d in enumerate(decisions):
                mode = str(d.get("mode") or "idle")
                modes.append(mode)
                # The work instruction is observable behaviour (it changes
                # what the worker produces); without it, programs that differ
                # only in their instructions would share a fingerprint.
                wis.append(str(d.get("work_instruction") or ""))
                if mode == "send":
                    for t in d.get("recipients") or []:
                        edges.append((a, int(t)))
            snapshot = [set(k) for k in knowledge]
            fresh = [0] * n_agents
            for s, t in edges:
                knowledge[t] |= snapshot[s]
                fresh[t] += 1
            inbox = fresh
            # Rounds in which every agent idles are left out.  reflect and
            # idle are both free no-ops (only send calls a worker), yet the
            # modes stay in the fingerprint: programs that differ only in
            # reflect vs idle count as distinct, which at worst costs a race
            # between two equivalent programs that ends in a tie.
            if any(mode != "idle" for mode in modes):
                rounds.append((frozenset(edges), tuple(modes), tuple(wis)))
        return tuple(rounds)
    except Exception:  # noqa: BLE001 - fingerprinting is advisory
        return None
