"""Execution payload, wall-clock budget and scoring for one run of a team
program on one benchmark instance."""

from __future__ import annotations

from math import ceil
from typing import Any

from exp_graph.mas.information_flow import coverage_by_agent
from exp_graph.mas.python_code_generation import PythonCodePlanningResult

from queenbee.bench.config import RunConfig
from queenbee.bench.instance import BenchmarkInstance
from queenbee.bench.scoring import ScoreResult
from queenbee.bench.silo_metrics import (
    evaluate_silo_submissions,
    silo_communication_density,
    silo_token_consumption,
)
from queenbee.bench.silo_protocol import SiloProtocolAdapter
from queenbee.bench.silo_scoring import silo_partial_score
from queenbee.bench.task_bridge import (
    canonical_answer,
    private_expected_outputs,
)


def _protocol_adapter(instance: BenchmarkInstance, *, information_goal: str = "sink"):
    """Task adapter that renders worker prompts and scores answers for ``instance``."""
    return SiloProtocolAdapter(instance, information_goal=information_goal)


def _build_python_execution_payload(
    *,
    instance: BenchmarkInstance,
    cfg: RunConfig,
    task_adapter: Any,
    global_task: dict[str, Any],
    n_agents: int,
) -> dict[str, Any]:
    """The JSON payload the program's ``main()`` reads on stdin.

    One communication prompt and one submit prompt per agent (rendered by the
    task adapter from that agent's local observation), the task description,
    the information goal, the agent that answers under ``sink``
    (``selected_primary`` = 0), the round, call, token and message budgets,
    the worker parallelism and the worker LLM settings (in the worker
    sandbox, ``request_timeout`` is the wall-clock budget of each worker
    call; ``None`` when the configured timeout is not positive).  Only the
    ``message_only_v2`` worker contract is supported.
    """
    observations = task_adapter.split_into_local_observations(global_task, n_agents)
    worker_contract = getattr(cfg, "python_worker_contract", "message_only_v2")
    if worker_contract != "message_only_v2":
        raise ValueError(f"unsupported python worker contract {worker_contract!r}")
    agents = [
        {
            "agent_id": agent_id,
            "communication_prompt": (
                task_adapter.format_python_communication_prompt(
                    global_task=global_task,
                    local_observation=observation,
                )
            ),
            "submit_prompt": task_adapter.format_python_submit_prompt(
                global_task=global_task,
                local_observation=observation,
            ),
        }
        for agent_id, observation in enumerate(observations)
    ]
    describe = getattr(task_adapter, "describe_task", None)
    task_description = describe() if callable(describe) else "Multi-agent task."
    return {
        "execution_contract_version": cfg.python_execution_contract_version,
        "worker_contract": worker_contract,
        "task_description": task_description,
        "information_goal": getattr(cfg, "silo_eval_mode", "sink") or "sink",
        "selected_primary": 0,
        "n_agents": n_agents,
        "max_rounds": cfg.max_rounds,
        "max_parallel_agents": max(1, int(cfg.max_parallel_agents)),
        "budgets": {
            "max_model_calls": cfg.python_max_model_calls,
            "max_completion_tokens": cfg.python_max_completion_tokens,
            "max_messages": cfg.python_max_messages,
        },
        "worker_llm": {
            "provider": cfg.llm_provider,
            "model_name": cfg.model_name,
            "base_url": cfg.base_url,
            "api_key_env": cfg.api_key_env,
            "temperature": cfg.temperature,
            "request_timeout": (
                float(cfg.request_timeout) if cfg.request_timeout > 0 else None
            ),
        },
        "agents": agents,
    }


def _resolved_python_execution_timeout(cfg: RunConfig, *, n_agents: int) -> float:
    """Return a whole-program wall budget that cannot undercut normal LLM waves.

    A configured ``python_execution_timeout`` is used as is (it must be
    positive).  Otherwise: a team program is synchronous between logical
    rounds, but the worker calls of one round run in parallel waves of at
    most ``max_parallel_agents``.  The automatic budget therefore gives every
    wave of every round (``max_rounds`` x waves per round) one complete
    per-request allowance (``request_timeout``, or 120 s when that is not
    positive), adds a 30 s process/setup margin and never goes below 120 s.
    """
    configured = cfg.python_execution_timeout
    if configured is not None:
        if float(configured) <= 0:
            raise ValueError("python_execution_timeout must be positive")
        return float(configured)
    # Every worker call site is a complete_batch, so one round runs in
    # ceil(n_agents / max_parallel_agents) waves.
    parallel_agents = min(
        max(1, int(cfg.max_parallel_agents)),
        max(1, int(n_agents)),
    )
    waves_per_round = ceil(max(1, int(n_agents)) / parallel_agents)
    per_request = (
        float(cfg.request_timeout) if float(cfg.request_timeout) > 0 else 120.0
    )
    return max(
        120.0,
        float(max(1, int(cfg.max_rounds)) * waves_per_round) * per_request + 30.0,
    )


def _score_python_execution(
    planning: PythonCodePlanningResult,
    *,
    instance: BenchmarkInstance,
    cfg: RunConfig,
    task_adapter: Any,
    global_task: dict[str, Any],
    extra: dict[str, Any],
) -> ScoreResult:
    """Grade a completed execution and reduce it to a :class:`ScoreResult`.

    Under the ``all_agents`` goal every agent's answer is graded against its
    own expected output (every agent against the instance's ground truth
    when there is no per-agent list of that length): success means every
    agent is exact, and ``partial`` is Silo-Bench's P (mean per-agent
    quality).  Under ``sink`` only agent 0's answer counts (against agent
    0's expected output on segmented instances, otherwise through the task
    adapter's scorer).  ``extra`` is updated in place with the goal-specific
    fields, information coverage, Silo-Bench's token consumption C and
    communication density D, usage counters and the message and submission
    logs.
    """
    execution = planning.execution
    output = execution.output
    if output is None:
        raise RuntimeError("successful Python planning result lacks output")
    n_agents = cfg.n_agents or instance.n_agents
    answers_by_id = {item.agent_id: item.answer for item in output.submissions}
    answers = [answers_by_id.get(agent_id) for agent_id in range(n_agents)]
    knowledge = [set(items) for items in execution.final_knowledge]
    coverage = coverage_by_agent(knowledge)
    usage = execution.authoritative_usage
    density = silo_communication_density(len(output.messages), n_agents)
    consumption = silo_token_consumption(
        int(usage.completion_tokens),
        int(output.rounds_executed),
    )
    goal = getattr(cfg, "silo_eval_mode", "sink") or "sink"
    if goal == "all_agents":
        expected_outputs = private_expected_outputs(global_task) or (
            instance.meta.get("expected_outputs") or []
        )
        if len(expected_outputs) != n_agents:
            expected_outputs = [instance.ground_truth for _ in range(n_agents)]
        graded = evaluate_silo_submissions(
            case_id=instance.case_id,
            answers=answers,
            expected_outputs=list(expected_outputs),
            submitted_rounds=[
                item.submitted_round
                for item in sorted(output.submissions, key=lambda item: item.agent_id)
            ],
        )
        correct = list(graded["per_agent_correct"])
        success = n_agents > 0 and all(correct)
        partial = float(graded["paper_P"])
        final_answer = answers[0] if answers else None
        extra.update(
            {
                "information_goal": "all_agents",
                "per_agent_answers": graded["per_agent_answers"],
                "per_agent_correct": correct,
                "per_agent_partial": graded["per_agent_partial"],
                "per_agent_submissions": graded["per_agent_submissions"],
                "paper_S": float(graded["paper_S"]),
                "paper_P": partial,
                "all_agents_exact": success,
                "information_coverage_by_agent": coverage,
                "mean_information_coverage": (
                    sum(coverage) / len(coverage) if coverage else 0.0
                ),
                "min_information_coverage": min(coverage) if coverage else 0.0,
                "all_agents_full_information": bool(
                    coverage and min(coverage) >= 1.0
                ),
            }
        )
    else:
        sink_id = 0
        answer = answers[sink_id] if answers else None
        if instance.segmented:
            expected_outputs = private_expected_outputs(global_task) or (
                instance.meta.get("expected_outputs") or []
            )
            expected = expected_outputs[sink_id] if expected_outputs else None
            success = answer is not None and canonical_answer(answer) == canonical_answer(
                expected
            )
            partial = float(
                silo_partial_score(
                    answer,
                    expected,
                    global_task.get("output_type", "scalar"),
                )
            )
        else:
            scored = task_adapter.score_protocol_answer(answer, global_task)
            success = bool(scored.get("exact_match", False))
            partial = float(scored.get("partial", 0.0))
        final_answer = answer
        extra.update(
            {
                "information_goal": "sink",
                "sink_id": sink_id,
                "sink_exact": success,
                "sink_partial": partial,
                "sink_information_coverage": (
                    coverage[sink_id] if coverage else 0.0
                ),
            }
        )
    extra.update(
        {
            "paper_C": consumption,
            "paper_D": density,
            "communication_density": density,
            "rounds_executed": output.rounds_executed,
            "worker_model_calls": usage.model_calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "python_messages": [
                message.model_dump(mode="json") for message in output.messages
            ],
            "python_submissions": [
                submission.model_dump(mode="json")
                for submission in output.submissions
            ],
            "python_submit_barrier": execution.ledger.get("submit_barrier"),
        }
    )
    return ScoreResult(
        success=bool(success),
        partial=float(partial),
        n_messages=len(output.messages),
        n_model_calls=int(usage.model_calls),
        tokens=int(usage.prompt_tokens) + int(usage.completion_tokens),
        final_answer=final_answer,
        extra=extra,
    )
