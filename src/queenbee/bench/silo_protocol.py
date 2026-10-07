"""Silo-Bench task adapter: worker prompts and answer scoring for one instance."""

from __future__ import annotations

from typing import Any

from exp_graph.mas.leakage_audit import assert_prompt_clean

from queenbee.bench.silo_scoring import silo_partial_score
from queenbee.bench.task_bridge import (
    BenchmarkTaskAdapter,
    canonical_answer,
    private_answer_key,
)


def score_protocol_answer(answer: Any, global_task: dict[str, Any]) -> dict[str, Any]:
    """Score one Silo global answer: strict exact match plus a graded ``partial``.

    ``primary_metric``/``exact_match`` are the strict 1.0/0.0 success signal;
    ``partial`` adds a graded partial-correctness value in [0, 1] (see
    ``silo_scoring.silo_partial_score``). The function is module-level so callers
    can recompute ``partial`` from a final answer without a task-adapter
    instance; the adapter method delegates here.
    """
    if answer is None:
        return {"primary_metric": 0.0, "exact_match": False, "partial": 0.0}
    # Ground truth comes only from the private scoring payload, which is kept
    # out of every model-visible rendering of the task.
    ground_truth = private_answer_key(global_task)
    success = canonical_answer(answer) == ground_truth
    partial = silo_partial_score(
        answer, ground_truth, global_task.get("output_type", "scalar")
    )
    return {
        "primary_metric": 1.0 if success else 0.0,
        "exact_match": success,
        "partial": float(partial),
    }


class SiloProtocolAdapter(BenchmarkTaskAdapter):
    """One Silo-Bench instance: per-agent worker prompts plus answer scoring.

    Both worker prompts pass through ``assert_prompt_clean``, which raises
    instead of returning a prompt that carries forbidden tokens such as
    answer-key or optimal-topology fields (see ``exp_graph.mas.leakage_audit``).
    """

    def format_python_communication_prompt(
        self,
        *,
        global_task: dict[str, Any],
        local_observation: dict[str, Any],
    ) -> str:
        prompt = super().format_python_communication_prompt(
            global_task=global_task,
            local_observation=local_observation,
        )
        return assert_prompt_clean(
            prompt,
            context="silo python communication prompt",
        )

    def format_python_submit_prompt(
        self,
        *,
        global_task: dict[str, Any],
        local_observation: dict[str, Any],
    ) -> str:
        prompt = super().format_python_submit_prompt(
            global_task=global_task,
            local_observation=local_observation,
        )
        return assert_prompt_clean(
            prompt,
            context="silo python submit prompt",
        )

    def score_protocol_answer(
        self, answer: Any, global_task: dict[str, Any]
    ) -> dict[str, Any]:
        # Delegates to the module-level function: strict exact match drives
        # primary_metric/exact_match, the graded value rides along in ``partial``.
        return score_protocol_answer(answer, global_task)
