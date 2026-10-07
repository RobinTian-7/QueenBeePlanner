"""Planning-result holder and offline dry-run payload for generated programs.

:class:`PythonCodePlanningResult` carries an executed program to the scorer.
:func:`_dry_run_payload` and :func:`_dry_run_canaries` build the synthetic
host payload of the offline dry run that checks a freshly generated program
(:mod:`queenbee.program.mint`) before any paid execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from exp_graph.mas.python_code_runner import PythonExecutionResult
from exp_graph.mas.schemas import MASRuntimeConfig, PlannerRequest


class PythonCodePlanningResult:
    """A program's source, its authoritative execution result and audit
    metadata (provenance label, generation prompt, attempts, planner and
    repair call counts)."""

    def __init__(
        self,
        *,
        source: str,
        execution: PythonExecutionResult,
        artifacts_dir: Path,
        provenance: str,
        architect_prompt: str | None,
        attempts: list[dict[str, Any]],
        planner_model_calls: int,
        repair_model_calls: int,
    ) -> None:
        self.source = source
        self.execution = execution
        self.artifacts_dir = artifacts_dir
        self.provenance = provenance
        self.architect_prompt = architect_prompt
        self.attempts = attempts
        self.planner_model_calls = planner_model_calls
        self.repair_model_calls = repair_model_calls


def _dry_run_payload(
    request: PlannerRequest,
    runtime: MASRuntimeConfig,
) -> dict[str, Any]:
    """Host payload of the offline dry run: the ``fake`` worker, synthetic
    private prompts that each carry their agent's canary, at most two rounds
    and call / token budgets sized for them."""
    canaries = _dry_run_canaries(request.n_agents)
    worker_contract = str(
        getattr(runtime, "python_worker_contract", "message_only_v2")
    )
    agents: list[dict[str, Any]] = []
    for agent_id in range(request.n_agents):
        private_text = (
            f"{canaries[str(agent_id)]}\n"
            f"Synthetic private prompt for agent {agent_id}."
        )
        agents.append(
            {
                "agent_id": agent_id,
                "communication_prompt": private_text,
                "submit_prompt": (
                    private_text
                    + "\nPUBLIC_ANSWER_REQUIREMENT:\n"
                    "Return one JSON string for this synthetic wiring check."
                ),
            }
        )
    return {
        "execution_contract_version": runtime.python_execution_contract_version,
        "worker_contract": worker_contract,
        "task_description": "Synthetic offline wiring check.",
        "information_goal": request.information_goal,
        "selected_primary": 0,
        "n_agents": request.n_agents,
        "max_rounds": min(2, runtime.python_max_rounds),
        "budgets": {
            "max_model_calls": max(1, request.n_agents * 2),
            "max_completion_tokens": max(1000, request.n_agents * 500),
            "max_messages": max(1, runtime.python_max_messages),
        },
        "worker_llm": {
            "provider": "fake",
            "model_name": "fake",
            "base_url": None,
            "api_key_env": None,
            "temperature": 0.0,
        },
        "agents": agents,
    }


def _dry_run_canaries(n_agents: int) -> dict[str, str]:
    """Per-agent canary strings: each worker prompt must carry its own
    agent's canary and no other agent's."""
    return {
        str(agent_id): f"PYTHON_LOCAL_CANARY_{agent_id}_A91F"
        for agent_id in range(n_agents)
    }
