"""``queenbee.bench.task_bridge``: canonical answers, the private scoring
payload, per-agent observations and the worker prompts."""

import json

import pytest

from queenbee.bench.instance import BenchmarkInstance
from queenbee.bench.task_bridge import (
    BenchmarkTaskAdapter,
    canonical_answer,
    private_answer_key,
    private_expected_outputs,
)
from queenbee.bench.task_view import PRIVATE_SCORING_KEY, public_task_view, task_ref


def _instance():
    return BenchmarkInstance(
        benchmark="silo_bench", case_id="I-01", case_name="Global Max",
        n_agents=2, shards=[[1, 5, 3], [9, 2]], ground_truth=9,
        task_prompt="Find the global maximum. You are agent {agent_id}.",
        meta={"output_type": "distributed"},
    )


def test_canonical_answer_scalar_and_json():
    assert canonical_answer(9) == "9"
    assert canonical_answer("9") == "9"           # numeric string normalizes
    assert canonical_answer("  9 ") == "9"
    assert canonical_answer([3, 1, 2]) == "[3,1,2]"
    assert canonical_answer("[3, 1, 2]") == "[3,1,2]"
    assert canonical_answer({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_answer(None) == "UNKNOWN"
    assert canonical_answer("unknown") == "UNKNOWN"
    assert canonical_answer("hello world") == "hello world"  # non-JSON string kept


def test_build_global_task_and_adjudication_hides_truth():
    adapter = BenchmarkTaskAdapter(_instance())
    gt = adapter.build_global_task()
    # The ground truth lives ONLY in the private scoring payload.
    assert private_answer_key(gt) == "9"
    assert "answer_key" not in gt
    assert gt["n_agents"] == 2
    # The model-visible context is an explicit allowlist: no answers, no
    # shards, no meta, no lookupable case_id -- only the opaque task_ref.
    context = adapter.format_adjudication_context(gt)
    assert "answer_key" not in json.dumps(context)
    assert "expected_output" not in json.dumps(context)
    assert "shards" not in context
    assert "meta" not in context
    assert "case_id" not in context
    assert context["task_ref"]


def test_global_task_keeps_answers_out_of_meta():
    inst = BenchmarkInstance(
        benchmark="silo_bench", case_id="II-99", case_name="Segments",
        n_agents=2, shards=[[1], [2]], ground_truth=[1],
        meta={
            "output_type": "distributed",
            "is_segmented": True,
            "expected_outputs": [[1], [3]],
            "optimal_topology": "chain",
            "difficulty": "easy",
        },
    )
    gt = BenchmarkTaskAdapter(inst, information_goal="all_agents").build_global_task()
    assert gt["meta"] == {
        "output_type": "distributed",
        "is_segmented": True,
        "difficulty": "easy",
    }
    assert gt["segmented"] is True
    assert gt["task_ref"] == task_ref("II-99")
    assert private_expected_outputs(gt) == [[1], [3]]
    assert private_answer_key(gt) == "[1]"
    assert private_expected_outputs({}) == [] and private_answer_key({}) is None
    view = public_task_view(gt)
    assert PRIVATE_SCORING_KEY not in view
    assert set(view) <= {
        "task_family", "benchmark", "task_ref", "case_name",
        "n_agents", "output_type", "segmented",
    }


def test_split_gives_each_agent_only_its_shard():
    adapter = BenchmarkTaskAdapter(_instance())
    gt = adapter.build_global_task()
    obs = adapter.split_into_local_observations(gt, 2)
    assert len(obs) == 2
    assert obs[0]["input_shard"] == [1, 5, 3]
    assert obs[1]["input_shard"] == [9, 2]
    assert obs[0]["agent_id"] == 0
    # Observations are model-visible: the opaque task_ref, never the case id.
    assert obs[0]["task_ref"] == task_ref("I-01")
    assert "I-01" not in json.dumps(obs)


def test_grading_compares_canonical_forms():
    gt = BenchmarkTaskAdapter(_instance()).build_global_task()
    assert canonical_answer("9") == private_answer_key(gt)
    assert canonical_answer(9.0) != private_answer_key(gt)
    assert canonical_answer("7") != private_answer_key(gt)
    assert canonical_answer(None) != private_answer_key(gt)


def test_canonical_answer_booleans():
    assert canonical_answer(True) == "true"
    assert canonical_answer("True") == "true"
    assert canonical_answer("false") == "false"
    inst = BenchmarkInstance(
        benchmark="silo_bench", case_id="I-03", case_name="Distributed Vote",
        n_agents=2, shards=[[1], [0]], ground_truth=True,
    )
    gt = BenchmarkTaskAdapter(inst).build_global_task()
    assert canonical_answer("True") == private_answer_key(gt)
    assert canonical_answer("False") != private_answer_key(gt)


def test_split_rejects_wrong_agent_count():
    adapter = BenchmarkTaskAdapter(_instance())
    gt = adapter.build_global_task()
    with pytest.raises(ValueError, match="cannot be re-split"):
        adapter.split_into_local_observations(gt, 3)


def test_prompt_does_not_double_embed_shard():
    inst = BenchmarkInstance(
        benchmark="silo_bench", case_id="I-01", case_name="Global Max",
        n_agents=2, shards=[[1, 5, 3], [9, 2]], ground_truth=9,
        task_prompt="Max. You are agent {agent_id} holding {input_shard}.",
    )
    adapter = BenchmarkTaskAdapter(inst)
    gt = adapter.build_global_task()
    obs = adapter.split_into_local_observations(gt, 2)
    prompt = adapter.format_task_prompt_context(gt, obs[0])
    assert prompt.count("[1, 5, 3]") == 1


def test_prompt_without_shard_placeholder_appends_the_shard():
    adapter = BenchmarkTaskAdapter(_instance())
    gt = adapter.build_global_task()
    obs = adapter.split_into_local_observations(gt, 2)
    prompt = adapter.format_task_prompt_context(gt, obs[1])
    assert prompt == (
        "Find the global maximum. You are agent 1.\n"
        "Total agents: 2\n"
        "You are agent 1. Your private shard: [9, 2]\n"
    )


def test_submit_prompt_quotes_the_public_output_section():
    inst = BenchmarkInstance(
        benchmark="silo_bench", case_id="I-01", case_name="Global Max",
        n_agents=2, shards=[[1, 5, 3], [9, 2]], ground_truth=9,
        task_prompt=(
            "**Task:** Find the maximum of {input_shard}.\n"
            "**Output:** A single\ninteger.\n"
            "**Notes:**\nIgnore me.\n"
        ),
    )
    adapter = BenchmarkTaskAdapter(inst)
    gt = adapter.build_global_task()
    obs = adapter.split_into_local_observations(gt, 2)
    submit = adapter.format_python_submit_prompt(global_task=gt, local_observation=obs[0])
    assert submit.endswith("PUBLIC_ANSWER_REQUIREMENT:\nA single integer.")
    communicate = adapter.format_python_communication_prompt(
        global_task=gt, local_observation=obs[0]
    )
    assert communicate.endswith(
        "COMMUNICATION_PURPOSE:\nPreserve the task facts needed by another Agent "
        "or by your later final-answer call."
    )
    # Without an Output section the requirement falls back to the output_type.
    plain = BenchmarkTaskAdapter(_instance())
    gt = plain.build_global_task()
    obs = plain.split_into_local_observations(gt, 2)
    fallback = plain.format_python_submit_prompt(global_task=gt, local_observation=obs[0])
    assert fallback.endswith(
        "Follow the answer shape requested by the public task statement. "
        'The benchmark\'s public output_type is "distributed".'
    )
