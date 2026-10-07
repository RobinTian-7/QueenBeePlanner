"""SiloProtocolAdapter: worker prompts, the execution payload and its wall budget."""

from __future__ import annotations

import json

import pytest

from exp_graph.mas.leakage_audit import PromptLeakageError, find_leakage_tokens
from exp_graph.mas.python_code import PythonExecutionPayload

from queenbee.bench.config import RunConfig
from queenbee.bench.engine import (
    _build_python_execution_payload,
    _protocol_adapter,
    _resolved_python_execution_timeout,
)
from queenbee.bench.instance import BenchmarkInstance
from queenbee.bench.silo_protocol import SiloProtocolAdapter
from queenbee.bench.task_bridge import BenchmarkTaskAdapter, private_answer_key
from queenbee.bench.task_view import task_ref

GOALS = ("all_agents", "sink")


def _global_max_instance() -> BenchmarkInstance:
    return BenchmarkInstance(
        benchmark="silo_bench",
        case_id="I-01",
        case_name="Global Max",
        n_agents=2,
        shards=[[3, 1, 9, 2], [5, 8, 4]],
        ground_truth=9,
        task_prompt=(
            "Find the GLOBAL MAXIMUM across all agents' data. "
            "You are Agent {agent_id} and hold: {input_shard}"
        ),
        meta={"output_type": "distributed"},
    )


def _distributed_sort_instance() -> BenchmarkInstance:
    return BenchmarkInstance(
        benchmark="silo_bench",
        case_id="III-21",
        case_name="Distributed Sort",
        n_agents=2,
        shards=[[3, 1], [4, 2]],
        ground_truth=[1, 2, 3, 4],
        task_prompt=(
            "**Task: Distributed Sort**\n"
            "Return the globally sorted ascending list of all agents' values.\n"
            "You are Agent {agent_id} and hold: {input_shard}\n"
            "**Output:** One ascending list of integers.\n"
        ),
        meta={
            "output_type": "distributed",
            "is_segmented": False,
            "expected_outputs": [[1, 2, 3, 4], [1, 2, 3, 4]],
        },
    )


def test_silo_adapter_extends_the_benchmark_task_adapter():
    inst = _global_max_instance()
    adapter = SiloProtocolAdapter(inst)
    assert isinstance(adapter, BenchmarkTaskAdapter)
    assert adapter.information_goal == "sink"

    global_task = adapter.build_global_task()
    assert private_answer_key(global_task) == "9"
    assert "answer_key" not in global_task
    assert global_task["case_id"] == "I-01"
    assert global_task["n_agents"] == 2
    assert global_task["task_ref"] == task_ref("I-01")


@pytest.mark.parametrize("goal", GOALS)
def test_worker_prompts_carry_the_shard_but_no_answer_or_case_id(goal):
    adapter = SiloProtocolAdapter(_distributed_sort_instance(), information_goal=goal)
    gt = adapter.build_global_task()
    observations = adapter.split_into_local_observations(gt, 2)
    for obs in observations:
        prompts = {
            "communication": adapter.format_python_communication_prompt(
                global_task=gt, local_observation=obs
            ),
            "submit": adapter.format_python_submit_prompt(
                global_task=gt, local_observation=obs
            ),
        }
        shard = json.dumps(obs["input_shard"])
        for name, prompt in prompts.items():
            assert find_leakage_tokens(prompt) == [], name
            assert shard in prompt, name
            assert "III-21" not in prompt, name
            assert "[1, 2, 3, 4]" not in prompt and "[1,2,3,4]" not in prompt, name
        assert prompts["communication"].startswith("TASK_AND_PRIVATE_DATA:\n")
        assert "COMMUNICATION_PURPOSE:" in prompts["communication"]
        assert prompts["submit"].endswith(
            "PUBLIC_ANSWER_REQUIREMENT:\nOne ascending list of integers."
        )


def test_prompts_that_would_leak_are_refused():
    inst = _global_max_instance()
    inst.task_prompt += " The expected_output is 9."
    adapter = SiloProtocolAdapter(inst, information_goal="all_agents")
    gt = adapter.build_global_task()
    obs = adapter.split_into_local_observations(gt, 2)[0]
    with pytest.raises(PromptLeakageError, match="silo python communication prompt"):
        adapter.format_python_communication_prompt(global_task=gt, local_observation=obs)
    with pytest.raises(PromptLeakageError, match="silo python submit prompt"):
        adapter.format_python_submit_prompt(global_task=gt, local_observation=obs)


def _cfg(goal: str, contract: str = "message_only_v2", **overrides) -> RunConfig:
    values = dict(
        silo_eval_mode=goal,
        llm_provider="openai",
        model_name="worker-model",
        base_url="http://localhost:9/v1",
        api_key_env="WORKER_KEY",
        n_agents=2,
        max_parallel_agents=2,
        python_worker_contract=contract,
        request_timeout=60.0,
        max_rounds=8,
        python_max_model_calls=40,
        python_max_completion_tokens=5000,
        python_max_messages=30,
    )
    values.update(overrides)
    return RunConfig(**values)


@pytest.mark.parametrize("goal", GOALS)
def test_execution_payload_carries_split_worker_prompts(goal):
    inst = _distributed_sort_instance()
    adapter = _protocol_adapter(inst, information_goal=goal)
    assert isinstance(adapter, SiloProtocolAdapter)
    gt = adapter.build_global_task()
    payload = _build_python_execution_payload(
        instance=inst,
        cfg=_cfg(goal),
        task_adapter=adapter,
        global_task=gt,
        n_agents=2,
    )
    PythonExecutionPayload.model_validate(payload)
    assert payload["worker_contract"] == "message_only_v2"
    assert payload["information_goal"] == goal
    assert payload["selected_primary"] == 0
    assert payload["n_agents"] == 2
    assert payload["max_rounds"] == 8
    assert payload["budgets"] == {
        "max_model_calls": 40,
        "max_completion_tokens": 5000,
        "max_messages": 30,
    }
    assert payload["worker_llm"] == {
        "provider": "openai",
        "model_name": "worker-model",
        "base_url": "http://localhost:9/v1",
        "api_key_env": "WORKER_KEY",
        "temperature": 0.0,
        "request_timeout": 60.0,
    }
    assert payload["task_description"] == adapter.describe_task()
    observations = adapter.split_into_local_observations(gt, 2)
    for agent, obs in zip(payload["agents"], observations):
        assert set(agent) == {"agent_id", "communication_prompt", "submit_prompt"}
        assert agent["communication_prompt"] == adapter.format_python_communication_prompt(
            global_task=gt, local_observation=obs
        )
        assert agent["submit_prompt"] == adapter.format_python_submit_prompt(
            global_task=gt, local_observation=obs
        )
    # No scoring data crosses into the program payload.
    blob = json.dumps(payload)
    assert "_private_scoring" not in blob and "III-21" not in blob


def test_payload_refuses_an_unsupported_worker_contract():
    inst = _distributed_sort_instance()
    adapter = _protocol_adapter(inst, information_goal="all_agents")
    with pytest.raises(ValueError, match="unsupported python worker contract"):
        _build_python_execution_payload(
            instance=inst,
            cfg=_cfg("all_agents", "bogus_v9"),
            task_adapter=adapter,
            global_task=adapter.build_global_task(),
            n_agents=2,
        )


def test_payload_without_a_request_timeout_leaves_it_to_the_worker():
    inst = _global_max_instance()
    adapter = _protocol_adapter(inst, information_goal="all_agents")
    payload = _build_python_execution_payload(
        instance=inst,
        cfg=_cfg("all_agents", "message_only_v2", request_timeout=0.0),
        task_adapter=adapter,
        global_task=adapter.build_global_task(),
        n_agents=2,
    )
    assert payload["worker_llm"]["request_timeout"] is None


def test_wall_budget_counts_parallel_waves_per_round():
    # message_only_v2 batches a round, so 5 agents with width 5 make one wave.
    cfg = _cfg("all_agents", "message_only_v2", max_parallel_agents=5,
               request_timeout=100.0, max_rounds=4)
    assert _resolved_python_execution_timeout(cfg, n_agents=5) == 4 * 100.0 + 30.0
    cfg = _cfg("all_agents", "message_only_v2", max_parallel_agents=2,
               request_timeout=100.0, max_rounds=4)
    assert _resolved_python_execution_timeout(cfg, n_agents=5) == 4 * 3 * 100.0 + 30.0
    cfg = _cfg("all_agents", "message_only_v2", max_parallel_agents=1,
               request_timeout=100.0, max_rounds=4)
    assert _resolved_python_execution_timeout(cfg, n_agents=5) == 4 * 5 * 100.0 + 30.0
    # No request timeout -> 120 s per request; never below 120 s overall.
    cfg = _cfg("all_agents", "message_only_v2", request_timeout=0.0, max_rounds=1)
    assert _resolved_python_execution_timeout(cfg, n_agents=2) == 150.0
    cfg = _cfg("all_agents", "message_only_v2", request_timeout=1.0, max_rounds=1)
    assert _resolved_python_execution_timeout(cfg, n_agents=2) == 120.0


def test_explicit_wall_budget_wins_and_must_be_positive():
    cfg = _cfg("sink", "message_only_v2", python_execution_timeout=42.5)
    assert _resolved_python_execution_timeout(cfg, n_agents=2) == 42.5
    cfg = _cfg("sink", "message_only_v2", python_execution_timeout=0)
    with pytest.raises(ValueError, match="must be positive"):
        _resolved_python_execution_timeout(cfg, n_agents=2)
