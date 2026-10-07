"""Request and runtime configuration models for generated team programs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ObjectiveName = Literal["accuracy_first", "budget_first", "balanced"]
# The evaluation's information goal.  "sink": all information converges on
# the selected_primary agent, which alone submits and is graded.
# "all_agents": every agent must end with full information and submit a
# correct answer of its own (a majority cannot mask an individual failure).
# The team program receives the goal, and the runtime's submit barrier, the
# S0 screen and the scoring all branch on it.
InformationGoal = Literal["sink", "all_agents"]
# Worker output contract, kept in lockstep with
# exp_graph.mas.python_code.PYTHON_WORKER_CONTRACTS by a test.
PythonWorkerContract = Literal["message_only_v2"]


class ObjectiveSpec(BaseModel):
    """Named objective weights (accuracy / cost / stability).

    A :class:`PlannerRequest` carries one (``"balanced"`` by default); the
    mint and execution paths do not read the weights.
    """

    name: ObjectiveName = "balanced"
    accuracy_weight: float = 0.5
    cost_weight: float = 0.35
    stability_weight: float = 0.15

    @classmethod
    def from_name(cls, name: ObjectiveName) -> "ObjectiveSpec":
        """Preset weights: ``accuracy_first`` favours accuracy,
        ``budget_first`` favours cost, any other name is the balanced
        default."""
        if name == "accuracy_first":
            return cls(
                name=name,
                accuracy_weight=0.70,
                cost_weight=0.15,
                stability_weight=0.15,
            )
        if name == "budget_first":
            return cls(
                name=name,
                accuracy_weight=0.25,
                cost_weight=0.65,
                stability_weight=0.10,
            )
        return cls(name=name)


class MASRuntimeConfig(BaseModel):
    """Runtime options for running a team program: the worker LLM
    (``"fake"`` is the offline deterministic client), worker concurrency,
    sandbox limits, contract versions, per-run budgets and the information
    goal."""

    llm_provider: str = "fake"
    model_name: str = "fake"
    temperature: float = 0.0
    max_parallel_agents: int = Field(default=1, ge=1)
    python_execution_timeout: float = Field(default=300.0, gt=0)
    python_cpu_seconds: int = 10
    python_memory_mb: int = 512
    python_max_output_bytes: int = 1_000_000
    python_execution_contract_version: str = "python_mas_v1"
    python_worker_contract: PythonWorkerContract = "message_only_v2"
    python_max_rounds: int = 4
    python_max_model_calls: int = 32
    python_max_completion_tokens: int = 20_000
    python_max_messages: int = 64
    information_goal: InformationGoal = "sink"


class PlannerRequest(BaseModel):
    """Request for minting a team program: team size, information goal and
    worker contract (the offline dry run builds its synthetic payload from
    these)."""

    n_agents: int
    objective: ObjectiveSpec = Field(
        default_factory=lambda: ObjectiveSpec.from_name("balanced")
    )
    information_goal: InformationGoal = "sink"
    python_worker_contract: PythonWorkerContract = "message_only_v2"

    @model_validator(mode="after")
    def validate_request(self) -> "PlannerRequest":
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        return self
