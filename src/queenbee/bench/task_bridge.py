"""BenchmarkInstance -> global task, per-agent observations and worker prompts.

:class:`BenchmarkTaskAdapter` flattens one instance into the run-side global
task dict, splits it into per-agent observations (one pre-built shard each)
and renders the two worker prompts of the message_only_v2 contract.  Answer
data lives only in the global task's private scoring payload.  Worker
prompts are rendered from explicitly chosen fields (task statement, agent
count, the agent's id and shard, public output requirement), never from the
whole global task; the structured task context safe to show a model is the
allowlisted ``public_task_view``, which names the case only by its opaque
``task_ref``.
"""

from __future__ import annotations

import json
from typing import Any

from queenbee.bench.instance import BenchmarkInstance
from queenbee.bench.task_view import (
    PRIVATE_SCORING_KEY,
    private_scoring_payload,
    public_task_view,
    task_ref,
)

# Key of the canonical ground truth INSIDE the private scoring payload
# (global_task[PRIVATE_SCORING_KEY]).  Scorers reach the payload through
# private_answer_key (this key) and private_expected_outputs; neither the
# worker prompts nor public_task_view (an explicit allowlist) include it.
GROUND_TRUTH_KEY = "answer_key"


def private_answer_key(global_task: dict[str, Any]) -> Any:
    """Scorer accessor: the canonical global answer (None when the private
    payload has none)."""
    return private_scoring_payload(global_task).get(GROUND_TRUTH_KEY)


def private_expected_outputs(global_task: dict[str, Any]) -> list[Any]:
    """Scorer accessor: the per-agent expected outputs (empty when absent)."""
    value = private_scoring_payload(global_task).get("expected_outputs")
    return list(value) if isinstance(value, list) else []


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _public_output_requirement(global_task: dict[str, Any]) -> str:
    """The public Output section of the task prompt, joined into one line.

    Collects the text after an ``**Output:**`` (or ``Output:``) line up to
    the next bold heading line; without such a section, a generic sentence
    naming the public ``output_type``.  Never consults scoring data.
    """
    task_prompt = str(global_task.get("task_prompt") or "")
    collected: list[str] = []
    in_output = False
    for line in task_prompt.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not in_output:
            marker = "**output:**"
            if lowered.startswith(marker):
                in_output = True
                remainder = stripped[len(marker) :].strip()
                if remainder:
                    collected.append(remainder)
            elif lowered.startswith("output:"):
                in_output = True
                remainder = stripped[len("output:") :].strip()
                if remainder:
                    collected.append(remainder)
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            break
        if stripped:
            collected.append(stripped)
    if collected:
        return " ".join(collected)
    output_type = public_task_view(global_task).get("output_type", "unspecified")
    return (
        "Follow the answer shape requested by the public task statement. "
        f"The benchmark's public output_type is {_dumps(output_type)}."
    )


def canonical_answer(value: Any) -> str:
    """Canonical string form of an answer for grouping and exact-match.

    Numbers, lists, and dicts are normalized via canonical JSON. Numeric or
    JSON-looking strings are parsed first so that "9" and 9, or "[3, 1]" and
    [3, 1], compare equal. Python-style booleans ("True"/"False") are folded to
    JSON booleans ("true"/"false"). Empty/unknown sentinels collapse to ``UNKNOWN``.
    """
    if value is None:
        return "UNKNOWN"
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() in {"UNKNOWN", "NONE", "NULL"}:
            return "UNKNOWN"
        if text.lower() in {"true", "false"}:
            return _dumps(text.lower() == "true")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
        return _dumps(parsed)
    return _dumps(value)


class BenchmarkTaskAdapter:
    """Adapter for one BenchmarkInstance: global task, local observations
    and worker prompts."""

    def __init__(
        self,
        instance: BenchmarkInstance,
        *,
        information_goal: str = "sink",
    ) -> None:
        self.instance = instance
        self.task_name = f"benchmark::{instance.benchmark}::{instance.case_id}"
        # Information goal ("sink" / "all_agents") of the run this adapter
        # serves.  Nothing rendered here depends on it: the submit barrier
        # and the scoring take the goal from the run configuration.
        self.information_goal = information_goal

    def build_global_task(self, **kwargs: Any) -> dict[str, Any]:
        """The run-side global task dict of the instance (keyword arguments
        are ignored).

        Answer fields (the canonical answer ``answer_key`` and the per-agent
        ``expected_outputs``) go only into the private payload under
        ``PRIVATE_SCORING_KEY``, which neither the worker prompts nor
        ``public_task_view`` include.
        """
        inst = self.instance
        # Answer fields (expected_output[s]) and reference-topology fields
        # (optimal_*, theoretical_complexity) never enter the run-side meta;
        # the per-agent expected outputs go to the private payload instead.
        meta = {
            key: value
            for key, value in inst.meta.items()
            if key
            not in {
                "expected_outputs",
                "expected_output",
                "optimal_topology",
                "optimal_message_count",
                "theoretical_complexity",
            }
        }
        return {
            "task_name": self.task_name,
            "task_family": inst.benchmark,
            "benchmark": inst.benchmark,
            "case_id": inst.case_id,
            "task_ref": task_ref(inst.case_id),
            "case_name": inst.case_name,
            "n_agents": inst.n_agents,
            "shards": list(inst.shards),
            "task_prompt": inst.task_prompt,
            "meta": meta,
            "output_type": inst.meta.get("output_type", "scalar"),
            # True for a segmented task (each agent has its own expected
            # output), False for a task with one shared answer.
            "segmented": inst.segmented,
            PRIVATE_SCORING_KEY: {
                GROUND_TRUTH_KEY: canonical_answer(inst.ground_truth),
                "expected_outputs": list(inst.meta.get("expected_outputs") or []),
                "output_type": inst.meta.get("output_type", "scalar"),
            },
        }

    def split_into_local_observations(
        self, global_task: dict[str, Any], n_agents: int
    ) -> list[dict[str, Any]]:
        """One local observation per agent, holding that agent's shard.

        Instances are pre-sharded, so ``n_agents`` must equal the number of
        shards.
        """
        shards = global_task["shards"]
        if n_agents != len(shards):
            raise ValueError(
                f"benchmark instance has {len(shards)} shards but n_agents={n_agents}; "
                "benchmark instances are pre-sharded and cannot be re-split"
            )
        # Observations are treated as model-visible, so they carry the opaque
        # task_ref instead of the case id (which indexes public benchmark
        # files that hold the answers) and a task_name without it.
        return [
            {
                "task_name": f"benchmark::{global_task['benchmark']}",
                "benchmark": global_task["benchmark"],
                "task_ref": global_task.get("task_ref")
                or task_ref(str(global_task.get("case_id", ""))),
                "agent_id": agent_id,
                "n_agents": n_agents,
                "input_shard": shards[agent_id],
            }
            for agent_id in range(n_agents)
        ]

    def format_task_prompt_context(
        self, global_task: dict[str, Any], local_observation: dict[str, Any]
    ) -> str:
        """The task statement for one agent: ``{agent_id}`` and
        ``{input_shard}`` filled in, then the agent count and the agent's
        identity.  The private shard is appended there only when the
        statement has no ``{input_shard}`` placeholder, so it never appears
        twice."""
        agent_id = local_observation["agent_id"]
        shard_json = json.dumps(local_observation["input_shard"], ensure_ascii=True)
        template = global_task["task_prompt"] or f"Task: {global_task['case_name']}"
        had_shard_token = "{input_shard}" in template
        rendered = template.replace("{agent_id}", str(agent_id)).replace(
            "{input_shard}", shard_json
        )
        lines = [rendered, f"Total agents: {local_observation['n_agents']}"]
        if had_shard_token:
            lines.append(f"You are agent {agent_id}.")
        else:
            lines.append(f"You are agent {agent_id}. Your private shard: {shard_json}")
        return "\n".join(lines) + "\n"

    # The message_only_v2 worker contract gives each agent two model-visible
    # contexts.  Neither asks for a structured state or action object: the
    # communication prompt asks for message content only, and the host part
    # of the program appends the fixed FINAL OUTPUT CONTRACT after the submit
    # prompt.
    def format_python_communication_prompt(
        self,
        *,
        global_task: dict[str, Any],
        local_observation: dict[str, Any],
    ) -> str:
        task_context = self.format_task_prompt_context(
            global_task, local_observation
        ).rstrip()
        return (
            "TASK_AND_PRIVATE_DATA:\n"
            f"{task_context}\n\n"
            "COMMUNICATION_PURPOSE:\n"
            "Preserve the task facts needed by another Agent or by your later "
            "final-answer call."
        )

    def format_python_submit_prompt(
        self,
        *,
        global_task: dict[str, Any],
        local_observation: dict[str, Any],
    ) -> str:
        task_context = self.format_task_prompt_context(
            global_task, local_observation
        ).rstrip()
        requirement = _public_output_requirement(global_task)
        return (
            "TASK_AND_PRIVATE_DATA:\n"
            f"{task_context}\n\n"
            "PUBLIC_ANSWER_REQUIREMENT:\n"
            f"{requirement}"
        )

    def describe_task(self) -> str:
        """Short task brief: the case name and the task statement (e.g.
        ``"Global Max: Find the global maximum..."``).

        Placeholders become ``<id>`` / ``<shard>``, whitespace is collapsed
        and the brief is cut to 400 characters.  The execution engine picks
        it up duck-typed as the payload's ``task_description``.
        """
        inst = self.instance
        prompt = (inst.task_prompt or "").replace("{agent_id}", "<id>").replace(
            "{input_shard}", "<shard>"
        )
        prompt = " ".join(prompt.split())
        brief = f"{inst.case_name}: {prompt}" if prompt else inst.case_name
        return brief[:400]

    def format_adjudication_context(self, global_task: dict[str, Any]) -> dict[str, Any]:
        """Task context safe to show a model: the allowlisted public view
        (:func:`public_task_view`).  The allowlist keeps nested fields such
        as ``meta.expected_outputs`` out, treats every field it does not
        name as private and names the case only by its ``task_ref``."""
        return public_task_view(global_task)
