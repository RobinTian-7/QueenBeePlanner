"""Offline tests of the message_only_v2 worker contract.

Covers the default team program and its execution payload, the synchronized
submit barrier (every expected submitter answers in the same round, from the
final delivery), the strict one-JSON-value answer parser and its single
repair re-ask, the metered worker client's checks on every call (planner
controls, runtime provenance, budgets, canaries), and the subprocess
runner's fail-closed reconciliation of program output with the runtime
ledger.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import get_args

import pytest
from exp_graph.llm.base import LLMResponse, LLMUsage
from exp_graph.mas.python_code import (
    DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
    MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION,
    MESSAGE_ONLY_V2_SUBMIT_INSTRUCTION,
    PYTHON_WORKER_CONTRACTS,
    PythonCodeError,
    PythonExecutionPayload,
    validate_python_source,
)
from exp_graph.mas.python_code_generation import _dry_run_canaries, _dry_run_payload
from exp_graph.mas.python_code_runner import CodeProcessRunner
from exp_graph.mas.python_worker_bootstrap import _MeteredClient, _value_sha256
from exp_graph.mas.schemas import MASRuntimeConfig, PlannerRequest, PythonWorkerContract


def _payload(
    *,
    goal: str = "all_agents",
    n_agents: int = 2,
    max_rounds: int = 2,
    max_model_calls: int = 20,
    max_completion_tokens: int = 4000,
    max_messages: int = 30,
    max_parallel_agents: int = 1,
) -> dict:
    return {
        "execution_contract_version": "python_mas_v1",
        "worker_contract": "message_only_v2",
        "task_description": "Synthetic offline barrier check.",
        "information_goal": goal,
        "selected_primary": 0,
        "n_agents": n_agents,
        "max_rounds": max_rounds,
        "max_parallel_agents": max_parallel_agents,
        "budgets": {
            "max_model_calls": max_model_calls,
            "max_completion_tokens": max_completion_tokens,
            "max_messages": max_messages,
        },
        "worker_llm": {
            "provider": "fake",
            "model_name": "fake",
            "base_url": None,
            "api_key_env": None,
            "temperature": 0.0,
        },
        "agents": [
            {
                "agent_id": agent_id,
                "communication_prompt": f"COMMUNICATION_PRIVATE_{agent_id}",
                "submit_prompt": (
                    f"SUBMIT_PRIVATE_{agent_id}\n"
                    "PUBLIC_ANSWER_REQUIREMENT:\nReturn one JSON string."
                ),
            }
            for agent_id in range(n_agents)
        ],
    }


def _request(
    worker_contract: str = "message_only_v2", goal: str = "all_agents"
) -> PlannerRequest:
    return PlannerRequest(
        n_agents=2,
        information_goal=goal,
        python_worker_contract=worker_contract,
    )


def _runtime(
    worker_contract: str = "message_only_v2", goal: str = "all_agents"
) -> MASRuntimeConfig:
    return MASRuntimeConfig(
        llm_provider="fake",
        model_name="fake",
        python_max_rounds=2,
        information_goal=goal,
        python_worker_contract=worker_contract,
    )


class _ScriptedWorker:
    def __init__(self, texts: list[str], *, completion_tokens: int = 1) -> None:
        self.texts = list(texts)
        self.completion_tokens = completion_tokens
        self.prompts: list[str] = []
        self.json_modes: list[bool] = []

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        del model_name, temperature
        self.prompts.append(prompt)
        self.json_modes.append(json_mode)
        text = self.texts.pop(0)
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                prompt_tokens=2,
                completion_tokens=self.completion_tokens,
            ),
        )


def _auth(
    *,
    n_agents: int = 1,
    max_rounds: int = 1,
    max_model_calls: int = 8,
    max_completion_tokens: int = 100,
    goal: str = "all_agents",
    max_parallel_agents: int = 1,
    max_messages: int = 20,
    canaries: dict[str, str] | None = None,
    communication_prompts: dict[str, str] | None = None,
) -> dict:
    return {
        "worker_llm": {
            "provider": "fake",
            "model_name": "fake",
            "base_url": None,
            "api_key_env": None,
            "temperature": 0.0,
        },
        "worker_contract": "message_only_v2",
        "budgets": {
            "max_model_calls": max_model_calls,
            "max_completion_tokens": max_completion_tokens,
            "max_messages": max_messages,
        },
        "agent_canaries": canaries or {},
        "max_output_bytes": 100_000,
        "n_agents": n_agents,
        "information_goal": goal,
        "selected_primary": 0,
        "max_rounds": max_rounds,
        "max_parallel_agents": max_parallel_agents,
        "agent_communication_prompts": communication_prompts or {
            str(agent_id): f"communication prompt {agent_id}"
            for agent_id in range(n_agents)
        },
        "agent_submit_prompts": {
            str(agent_id): (
                f"submit prompt {agent_id}\n"
                "PUBLIC_ANSWER_REQUIREMENT:\nReturn one JSON value."
            )
            for agent_id in range(n_agents)
        },
    }


def _metered_client(
    tmp_path: Path,
    inner: _ScriptedWorker,
    auth: dict,
) -> tuple[_MeteredClient, dict]:
    ledger = {
        "factory_calls": 1,
        "worker_contract": "message_only_v2",
        "calls": [],
        "usage": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "errors": [],
    }
    return _MeteredClient(inner, auth, ledger, tmp_path / "ledger.json"), ledger


def _submit_prompt(
    *,
    agent_id: int = 0,
    submit_round: int = 0,
    known_ids: list[int] | None = None,
    previous_output: str = "",
    inbox: list[dict] | None = None,
    submit_prompt: str | None = None,
) -> str:
    control = {"mode": "submit", "recipients": []}
    known_ids = [agent_id] if known_ids is None else known_ids
    inbox = [] if inbox is None else inbox
    submit_prompt = submit_prompt or (
        f"submit prompt {agent_id}\n"
        "PUBLIC_ANSWER_REQUIREMENT:\nReturn one JSON value."
    )
    return (
        "PYTHON_WORKER_CONTRACT:message_only_v2\n"
        f"PYTHON_AGENT_ID:{agent_id}\n"
        f"PYTHON_ROUND:{submit_round}\n"
        f"PYTHON_SUBMIT_ROUND:{submit_round}\n"
        "PYTHON_CONTROL_JSON:"
        + json.dumps(control, sort_keys=True)
        + "\nKNOWN_SOURCE_IDS_JSON:"
        + json.dumps(known_ids)
        + "\nPREVIOUS_OUTPUT_JSON:"
        + json.dumps(previous_output)
        + "\nDELIVERED_INBOX_JSON:"
        + json.dumps(inbox, sort_keys=True)
        + "\n"
        + "SUBMIT_PROMPT:\n"
        + submit_prompt
        + "\n"
        + MESSAGE_ONLY_V2_SUBMIT_INSTRUCTION
    )


def test_v2_program_is_valid_and_unknown_contracts_fail_closed() -> None:
    assert tuple(get_args(PythonWorkerContract)) == PYTHON_WORKER_CONTRACTS
    assert PYTHON_WORKER_CONTRACTS == ("message_only_v2",)
    assert validate_python_source(DEFAULT_MESSAGE_ONLY_V2_PROGRAM).valid
    assert validate_python_source(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        worker_contract="message_only_v2",
    ).valid
    unknown = validate_python_source(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        worker_contract="bogus_v9",
    )
    assert not unknown.valid
    assert "unknown python worker contract" in unknown.errors[0]["message"]


@pytest.mark.parametrize("goal", ["all_agents", "sink"])
def test_dry_run_payload_runs_the_v2_program_offline(goal: str) -> None:
    request = _request(goal=goal)
    runtime = _runtime(goal=goal)
    payload = _dry_run_payload(request, runtime)
    assert PythonExecutionPayload.model_validate(payload).worker_contract == (
        "message_only_v2"
    )
    canaries = _dry_run_canaries(request.n_agents)
    for agent in payload["agents"]:
        assert canaries[str(agent["agent_id"])] in agent["communication_prompt"]
        assert canaries[str(agent["agent_id"])] in agent["submit_prompt"]
    result = CodeProcessRunner().run(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        payload,
        agent_canaries=canaries,
    )
    assert result.runtime_success, result.failure
    assert result.output is not None
    assert result.output.rounds_executed == 2
    expected = [0, 1] if goal == "all_agents" else [0]
    assert result.ledger["submit_barrier"]["expected_agent_ids"] == expected
    assert result.ledger["submit_barrier"]["synchronized"] is True


def test_v2_rejects_legacy_scalar_worker_calls() -> None:
    scalar = DEFAULT_MESSAGE_ONLY_V2_PROGRAM.replace(
        "client.complete_batch(",
        "client.complete(",
    )
    report = validate_python_source(scalar, worker_contract="message_only_v2")
    assert not report.valid
    assert any(
        "requires complete_batch" in error["message"] for error in report.errors
    )


def test_v2_payload_requires_split_prompts_and_rejects_unknown_prompt_fields() -> None:
    assert PythonExecutionPayload.model_validate(_payload()).worker_contract == (
        "message_only_v2"
    )
    extra = _payload()
    extra["agents"][0]["local_prompt"] = "an extra private prompt"
    with pytest.raises(ValueError, match="local_prompt"):
        PythonExecutionPayload.model_validate(extra)
    missing_submit = _payload()
    del missing_submit["agents"][0]["submit_prompt"]
    with pytest.raises(ValueError, match="submit_prompt"):
        PythonExecutionPayload.model_validate(missing_submit)
    other = _payload()
    other["worker_contract"] = "bogus_v9"
    with pytest.raises(ValueError, match="worker_contract"):
        PythonExecutionPayload.model_validate(other)


def test_submit_context_precedes_immutable_final_output_contract() -> None:
    prompt = _submit_prompt(
        submit_prompt=(
            "TASK_AND_PRIVATE_DATA:\nFind the maximum of [3, 827].\n\n"
            "PUBLIC_ANSWER_REQUIREMENT:\n"
            "A single integer representing the global maximum value."
        )
    )
    context = prompt.split("SUBMIT_PROMPT:\n", 1)[1].split(
        "\nFINAL OUTPUT CONTRACT:\n", 1
    )[0]
    assert "belief_state" not in context
    assert "structured_state" not in context
    assert "consensus_key" not in context
    assert prompt.index("PUBLIC_ANSWER_REQUIREMENT:") < prompt.index(
        "FINAL OUTPUT CONTRACT:"
    )
    assert prompt.endswith(
        "Your entire response must be the single benchmark answer value.\n"
    )


def test_two_agents_communicate_then_submit_together_from_final_delivery() -> None:
    result = CodeProcessRunner().run(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        _payload(n_agents=2, max_rounds=2, max_model_calls=4),
    )
    assert result.runtime_success, result.failure
    assert result.output is not None
    assert result.output.rounds_executed == 2
    assert {(item.round_sent, item.round_delivered) for item in result.output.messages} == {
        (0, 1)
    }
    assert [item.submitted_round for item in result.output.submissions] == [1, 1]
    assert [item.answer for item in result.output.submissions] == [
        "UNKNOWN",
        "UNKNOWN",
    ]
    calls = result.ledger["calls"]
    assert {(item["round"], item["mode"]) for item in calls} == {
        (0, "send"),
        (1, "submit"),
    }
    submit_calls = [item for item in calls if item["mode"] == "submit"]
    assert len(submit_calls) == 2
    assert all(item["delivered_inbox"] for item in submit_calls)
    assert all(item["known_source_ids"] == [0, 1] for item in submit_calls)
    assert result.final_knowledge == [[0, 1], [0, 1]]
    assert result.ledger["submit_barrier"] == {
        "answer_format": "single_json_value",
        "enabled": True,
        "expected_agent_ids": [0, 1],
        "final_snapshot_sha256": result.ledger["submit_barrier"][
            "final_snapshot_sha256"
        ],
        "information_goal": "all_agents",
        "observed_agent_ids": [0, 1],
        "parser": "json.loads_whole_response",
        "submit_round": 1,
        "synchronized": True,
        "worker_contract": "message_only_v2",
    }
    assert len(result.ledger["submit_barrier"]["final_snapshot_sha256"]) == 64


class _BarrierWorker:
    """Fail if two provider calls from one batch do not overlap."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(2)

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        del model_name, temperature
        assert json_mode is False
        agent_id = int(prompt.split("PYTHON_AGENT_ID:", 1)[1].splitlines()[0])
        self.barrier.wait(timeout=2.0)
        return LLMResponse(
            text=json.dumps(f"answer-{agent_id}"),
            usage=LLMUsage(prompt_tokens=2, completion_tokens=1),
        )


def test_complete_batch_overlaps_same_round_calls_and_commits_in_agent_order(
    tmp_path: Path,
) -> None:
    client, ledger = _metered_client(
        tmp_path,
        _BarrierWorker(),  # type: ignore[arg-type]
        _auth(n_agents=2, max_parallel_agents=2),
    )

    responses = client.complete_batch(
        [_submit_prompt(agent_id=0), _submit_prompt(agent_id=1)],
        model_name="fake",
        temperature=0.0,
        json_mode=False,
    )

    assert [json.loads(response.text) for response in responses] == [
        "answer-0",
        "answer-1",
    ]
    assert [call["agent_id"] for call in ledger["calls"]] == [0, 1]
    assert ledger["submit_barrier"]["synchronized"] is True
    assert ledger["parallelism"] == {
        "max_parallel_agents": 2,
        "batch_calls": 1,
        "max_batch_size": 2,
        "max_workers_used": 2,
    }


def test_default_v2_program_uses_host_bounded_parallel_batches() -> None:
    result = CodeProcessRunner().run(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        _payload(
            n_agents=2,
            max_rounds=2,
            max_model_calls=4,
            max_parallel_agents=2,
        ),
    )

    assert result.runtime_success, result.failure
    assert result.ledger["parallelism"]["batch_calls"] == 2
    assert result.ledger["parallelism"]["max_batch_size"] == 2
    assert result.ledger["parallelism"]["max_workers_used"] == 2


def test_planner_may_choose_one_global_earlier_barrier() -> None:
    source = DEFAULT_MESSAGE_ONLY_V2_PROGRAM.replace(
        "    return max_rounds - 1\n",
        "    return 1\n",
        1,
    )
    result = CodeProcessRunner().run(
        source,
        _payload(n_agents=2, max_rounds=3, max_model_calls=4),
    )
    assert result.runtime_success, result.failure
    assert result.output is not None
    assert result.output.rounds_executed == 2
    assert [item.submitted_round for item in result.output.submissions] == [1, 1]


def test_communication_policy_cannot_return_submit() -> None:
    source = DEFAULT_MESSAGE_ONLY_V2_PROGRAM.replace(
        '    return {"mode": "send", "recipients": recipients}\n',
        '    return {"mode": "submit", "recipients": []}\n',
        1,
    )
    result = CodeProcessRunner().run(source, _payload())
    assert not result.runtime_success
    assert result.failure is not None
    assert result.failure["error_type"] == "DataFlowError"
    assert result.authoritative_usage.model_calls == 0


def test_invalid_global_submit_round_fails_before_worker_calls() -> None:
    source = DEFAULT_MESSAGE_ONLY_V2_PROGRAM.replace(
        "    return max_rounds - 1\n",
        "    return max_rounds\n",
        1,
    )
    result = CodeProcessRunner().run(source, _payload())
    assert not result.runtime_success
    assert result.failure is not None
    assert result.failure["error_type"] == "DataFlowError"
    assert result.authoritative_usage.model_calls == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", 42),
        ("[1, 2, 3]", [1, 2, 3]),
        ('"accepted"', "accepted"),
        ('{"answer": 42}', {"answer": 42}),
    ],
)
def test_submit_parser_preserves_one_native_json_value(
    tmp_path: Path,
    raw: str,
    expected,
) -> None:
    inner = _ScriptedWorker([raw])
    client, ledger = _metered_client(tmp_path, inner, _auth())
    response = client.complete(
        _submit_prompt(),
        model_name="fake",
        temperature=0.0,
        json_mode=False,
    )
    assert response.text == raw
    assert inner.json_modes == [False]
    assert ledger["calls"][0]["answer_sha256"] == _value_sha256(expected)
    if isinstance(expected, dict):
        assert ledger["calls"][0]["answer_sha256"] != _value_sha256(42)


@pytest.mark.parametrize(
    "raw",
    ["The answer is 42", "Answer: [1, 2, 3]", "```json\n42\n```", "NaN"],
)
def test_submit_parser_rejects_any_non_json_wrapper(
    tmp_path: Path,
    raw: str,
) -> None:
    # The one-shot repair re-asks once (the same prompt plus a repair
    # notice); the original and the repaired output go through the SAME
    # strict parser, so a wrapped answer still fails, with the rejected
    # output's hash on record.
    inner = _ScriptedWorker([raw, raw])
    client, ledger = _metered_client(tmp_path, inner, _auth())
    with pytest.raises(RuntimeError, match="AnswerFormatError"):
        client.complete(
            _submit_prompt(),
            model_name="fake",
            temperature=0.0,
            json_mode=False,
        )
    assert len(ledger["calls"]) == 1
    assert len(inner.prompts) == 2
    assert "ANSWER_FORMAT_REPAIR_NOTICE" in inner.prompts[1]
    assert ledger["calls"][0]["answer_format_valid"] is False
    assert ledger["calls"][0]["answer_repaired"] is False
    assert ledger["calls"][0]["rejected_output_sha256"] == _value_sha256(raw)
    assert "answer_sha256" not in ledger["calls"][0]


def test_submit_repair_recovers_empty_truncated_output(
    tmp_path: Path,
) -> None:
    # A reply cut off at the completion-token limit leaves the submit
    # content empty.  The repaired re-emission is accepted, marked as
    # repaired and metered as one call record that folds in the usage of
    # both provider calls.
    inner = _ScriptedWorker(["", "42"], completion_tokens=7)
    client, ledger = _metered_client(tmp_path, inner, _auth())
    response = client.complete(
        _submit_prompt(),
        model_name="fake",
        temperature=0.0,
        json_mode=False,
    )
    assert response.text == "42"
    assert len(inner.prompts) == 2
    record = ledger["calls"][0]
    assert record["answer_format_valid"] is True
    assert record["answer_repaired"] is True
    assert record["rejected_output_sha256"] == _value_sha256("")
    assert record["worker_output_sha256"] == _value_sha256("42")
    assert record["answer_sha256"] == _value_sha256(42)
    assert record["model_calls"] == 2
    assert record["completion_tokens"] == 14
    assert ledger["usage"]["model_calls"] == 2
    assert ledger["usage"]["completion_tokens"] == 14


def test_submit_repair_second_failure_raises(
    tmp_path: Path,
) -> None:
    inner = _ScriptedWorker(["", "still not json"])
    client, ledger = _metered_client(tmp_path, inner, _auth())
    with pytest.raises(RuntimeError, match="AnswerFormatError"):
        client.complete(
            _submit_prompt(),
            model_name="fake",
            temperature=0.0,
            json_mode=False,
        )
    assert len(ledger["calls"]) == 1
    assert ledger["calls"][0]["answer_repaired"] is False
    assert "answer_sha256" not in ledger["calls"][0]


def test_early_submit_and_partial_budget_fail_before_calling_worker(
    tmp_path: Path,
) -> None:
    early_inner = _ScriptedWorker(["42"])
    early_client, _ = _metered_client(
        tmp_path,
        early_inner,
        _auth(n_agents=1, max_rounds=2),
    )
    early_prompt = _submit_prompt(submit_round=1).replace(
        "PYTHON_ROUND:1", "PYTHON_ROUND:0"
    )
    with pytest.raises(RuntimeError, match="only at the synchronized barrier"):
        early_client.complete(
            early_prompt,
            model_name="fake",
            temperature=0.0,
            json_mode=False,
        )
    assert early_inner.prompts == []

    budget_inner = _ScriptedWorker(['"a"', '"b"'])
    budget_client, budget_ledger = _metered_client(
        tmp_path,
        budget_inner,
        _auth(n_agents=2, max_rounds=1, max_model_calls=1),
    )
    with pytest.raises(RuntimeError, match="before barrier"):
        budget_client.complete(
            _submit_prompt(),
            model_name="fake",
            temperature=0.0,
            json_mode=False,
        )
    assert budget_inner.prompts == []
    assert budget_ledger["calls"] == []
    assert "submit_barrier" not in budget_ledger

    token_inner = _ScriptedWorker(
        ['"long first"', '"second"'],
        completion_tokens=2,
    )
    token_client, token_ledger = _metered_client(
        tmp_path,
        token_inner,
        _auth(
            n_agents=2,
            max_rounds=1,
            max_model_calls=2,
            max_completion_tokens=2,
        ),
    )
    with pytest.raises(RuntimeError, match="complete barrier"):
        token_client.complete(
            _submit_prompt(),
            model_name="fake",
            temperature=0.0,
            json_mode=False,
        )
    assert len(token_ledger["calls"]) == 1
    assert token_ledger["submit_barrier"]["observed_agent_ids"] == []
    assert token_ledger["submit_barrier"]["synchronized"] is False


def test_sequential_physical_calls_do_not_share_submit_answers(
    tmp_path: Path,
) -> None:
    inner = _ScriptedWorker(['"ANSWER_ZERO_SECRET"', '"answer one"'])
    client, ledger = _metered_client(
        tmp_path,
        inner,
        _auth(n_agents=2, max_rounds=1),
    )
    for agent_id in range(2):
        client.complete(
            _submit_prompt(agent_id=agent_id),
            model_name="fake",
            temperature=0.0,
            json_mode=False,
        )
    assert "ANSWER_ZERO_SECRET" not in inner.prompts[1]
    assert ledger["submit_barrier"]["synchronized"] is True
    assert ledger["submit_barrier"]["observed_agent_ids"] == [0, 1]


def test_sink_barrier_submits_only_selected_primary() -> None:
    result = CodeProcessRunner().run(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        _payload(goal="sink", n_agents=3, max_rounds=2, max_model_calls=4),
    )
    assert result.runtime_success, result.failure
    assert result.output is not None
    submissions = {item.agent_id: item for item in result.output.submissions}
    assert submissions[0].submitted_round == 1
    assert submissions[1].submitted_round is None
    assert submissions[2].submitted_round is None
    assert result.ledger["submit_barrier"]["expected_agent_ids"] == [0]


def test_host_rejects_tampered_native_submission() -> None:
    runner = CodeProcessRunner()
    payload = _payload(n_agents=2, max_rounds=2, max_model_calls=4)
    result = runner.run(DEFAULT_MESSAGE_ONLY_V2_PROGRAM, payload)
    assert result.runtime_success and result.output is not None
    tampered = result.output.model_copy(deep=True)
    tampered.submissions[0].answer = 42
    with pytest.raises(PythonCodeError, match="parsed Worker answer"):
        runner._validate_output(
            tampered,
            payload,
            result.authoritative_usage,
            result.ledger,
        )


# --------------------------------------------------------------------------- #
# Answer-format repair inside complete_batch.  The batch first fetches every
# reply (concurrently, up to max_parallel_agents), then replays the replies in
# input order through the per-call checks; a repair re-ask must reach the real
# provider client (``_repair_inner``), never the replay queue, or it would
# consume the next agent's fetched answer (a silent answer swap) or
# under/overflow the queue.
# --------------------------------------------------------------------------- #


def test_batch_submit_repair_uses_real_client_and_keeps_answers_in_place(
    tmp_path: Path,
) -> None:
    inner = _ScriptedWorker(["not-json prose", '"B"', "42"])
    client, ledger = _metered_client(
        tmp_path, inner, _auth(n_agents=2, max_parallel_agents=1)
    )

    committed = client.complete_batch(
        [_submit_prompt(agent_id=0), _submit_prompt(agent_id=1)],
        model_name="fake",
        temperature=0.0,
        json_mode=False,
    )

    # Each answer stays in place: agent 0 gets its own repaired answer and
    # agent 1's answer is not consumed by the repair.
    assert json.loads(committed[0].text) == 42
    assert json.loads(committed[1].text) == "B"
    # The repair call reached the real client: agent 0's original prompt
    # plus the repair notice.
    assert len(inner.prompts) == 3
    assert inner.prompts[2].startswith(_submit_prompt(agent_id=0))
    assert "ANSWER_FORMAT_REPAIR_NOTICE" in inner.prompts[2]
    # Ledger audit trail.
    by_agent = {call["agent_id"]: call for call in ledger["calls"]}
    assert by_agent[0]["answer_repaired"] is True
    assert by_agent[0]["rejected_output_sha256"] == _value_sha256("not-json prose")
    assert by_agent[1]["answer_repaired"] is False
    assert by_agent[1]["rejected_output_sha256"] is None
    # The repair left the replay queue alone, so the overflow / underflow
    # checks passed (complete_batch raised nothing).


def test_batch_submit_repair_on_last_prompt_does_not_underflow(
    tmp_path: Path,
) -> None:
    inner = _ScriptedWorker(['"A"', "prose garbage", '"7"'])
    client, ledger = _metered_client(
        tmp_path, inner, _auth(n_agents=2, max_parallel_agents=1)
    )

    committed = client.complete_batch(
        [_submit_prompt(agent_id=0), _submit_prompt(agent_id=1)],
        model_name="fake",
        temperature=0.0,
        json_mode=False,
    )

    assert json.loads(committed[0].text) == "A"
    assert json.loads(committed[1].text) == "7"
    by_agent = {call["agent_id"]: call for call in ledger["calls"]}
    assert by_agent[1]["answer_repaired"] is True


def test_submit_repair_second_failure_still_raises_answer_format_error(
    tmp_path: Path,
) -> None:
    inner = _ScriptedWorker(["first bad", "still bad"])
    client, _ledger = _metered_client(
        tmp_path, inner, _auth(n_agents=1, max_parallel_agents=1)
    )

    with pytest.raises(RuntimeError, match="AnswerFormatError"):
        client.complete_batch(
            [_submit_prompt(agent_id=0)],
            model_name="fake",
            temperature=0.0,
            json_mode=False,
        )
    assert len(inner.prompts) == 2  # the original call + one repair, never more


# --------------------------------------------------------------------------- #
# Metered client: communication-call semantics (authoritative runtime state)
# --------------------------------------------------------------------------- #


class _TextWorker:
    """Plain-text worker: scripted outputs first, then a fixed text."""

    def __init__(self, texts: list[str] | None = None) -> None:
        self.texts = list(texts or [])
        self.json_modes: list[bool] = []

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        del prompt, model_name, temperature
        self.json_modes.append(json_mode)
        text = self.texts.pop(0) if self.texts else "worker text output"
        return LLMResponse(
            text=text,
            usage=LLMUsage(prompt_tokens=7, completion_tokens=3),
        )


def _comm_prompt(
    auth: dict,
    *,
    agent_id: int,
    round_idx: int,
    control: dict,
    submit_round: int,
    known: list[int] | None = None,
    prev: str = "",
    inbox: list | None = None,
    instruction: str | None = None,
) -> str:
    """The canonical communication prompt the default program would render."""
    instruction = (
        MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION if instruction is None else instruction
    )
    work = control.get("work_instruction")
    phase = (
        f"PHASE_WORK_INSTRUCTION:\n{work}\n" if isinstance(work, str) and work else ""
    )
    return (
        "PYTHON_WORKER_CONTRACT:message_only_v2\n"
        f"PYTHON_AGENT_ID:{agent_id}\n"
        f"PYTHON_ROUND:{round_idx}\n"
        f"PYTHON_SUBMIT_ROUND:{submit_round}\n"
        "PYTHON_CONTROL_JSON:"
        + json.dumps(control, sort_keys=True)
        + "\nKNOWN_SOURCE_IDS_JSON:"
        + json.dumps([agent_id] if known is None else known)
        + "\nPREVIOUS_OUTPUT_JSON:"
        + json.dumps(prev)
        + "\nDELIVERED_INBOX_JSON:"
        + json.dumps(inbox or [], sort_keys=True)
        + "\nCOMMUNICATION_PROMPT:\n"
        + auth["agent_communication_prompts"][str(agent_id)]
        + "\n"
        + phase
        + instruction
    )


def _call(client: _MeteredClient, prompt: str) -> LLMResponse:
    return client.complete(
        prompt,
        model_name="fake",
        temperature=0.0,
        json_mode=False,
    )


REFLECT = {"mode": "reflect", "recipients": []}


def test_send_updates_state_creates_envelopes_and_ledger(tmp_path: Path) -> None:
    auth = _auth(n_agents=2, max_rounds=2)
    client, ledger = _metered_client(
        tmp_path, _TextWorker(["  summary text  "]), auth  # type: ignore[arg-type]
    )
    control = {"mode": "send", "recipients": [1]}
    _call(
        client,
        _comm_prompt(auth, agent_id=0, round_idx=0, control=control, submit_round=1),
    )
    assert client.previous_outputs[0] == "summary text"
    assert len(client.pending_envelopes) == 1
    envelope = client.pending_envelopes[0]
    assert (envelope["round_sent"], envelope["round_delivered"]) == (0, 1)
    assert (envelope["src"], envelope["dst"]) == (0, 1)
    assert envelope["source_ids"] == [0]
    assert envelope["body"] == "summary text"
    entry = ledger["calls"][0]
    assert set(entry) == {
        "agent_id", "round", "submit_round", "mode", "recipients",
        "known_source_ids", "delivered_inbox", "prompt_sha256",
        "worker_output_sha256", "inbox_bytes", "answer_format_valid",
        "answer_repaired", "rejected_output_sha256", "model_calls",
        "prompt_tokens", "completion_tokens",
    }
    assert entry["mode"] == "send" and entry["recipients"] == [1]
    assert entry["known_source_ids"] == [0] and entry["submit_round"] == 1
    assert entry["worker_output_sha256"] == _value_sha256("summary text")
    # No raw text may land in the persisted ledger, only hashes and counters.
    assert "summary text" not in (tmp_path / "ledger.json").read_text()


def test_reflect_updates_state_without_sending(tmp_path: Path) -> None:
    auth = _auth(n_agents=2, max_rounds=2)
    client, ledger = _metered_client(tmp_path, _TextWorker(), auth)  # type: ignore[arg-type]
    _call(
        client,
        _comm_prompt(auth, agent_id=0, round_idx=0, control=REFLECT, submit_round=1),
    )
    assert client.previous_outputs[0] == "worker text output"
    assert client.pending_envelopes == []
    assert ledger["calls"][0]["mode"] == "reflect"


def test_submitted_agent_cannot_be_called_again(tmp_path: Path) -> None:
    auth = _auth(n_agents=1, max_rounds=2)
    client, _ = _metered_client(tmp_path, _ScriptedWorker(['"done"']), auth)
    _call(client, _submit_prompt(agent_id=0, submit_round=0))
    assert client.submitted_rounds == {0: 0}
    with pytest.raises(RuntimeError, match="submitted agent was called again"):
        _call(
            client,
            _comm_prompt(
                auth, agent_id=0, round_idx=1, control=REFLECT, submit_round=0
            ),
        )


def test_worker_json_mode_must_be_text(tmp_path: Path) -> None:
    auth = _auth(n_agents=2, max_rounds=2)
    client, _ = _metered_client(tmp_path, _TextWorker(), auth)  # type: ignore[arg-type]
    prompt = _comm_prompt(
        auth, agent_id=0, round_idx=0, control=REFLECT, submit_round=1
    )
    with pytest.raises(RuntimeError, match="json_mode=False"):
        client.complete(prompt, model_name="fake", temperature=0.0)
    with pytest.raises(RuntimeError, match="json_mode=False"):
        client.complete(prompt, model_name="fake", temperature=0.0, json_mode=True)


@pytest.mark.parametrize(
    ("control", "match"),
    [
        ({"mode": "idle", "recipients": []}, "not send/reflect/submit"),
        ({"mode": "reflect", "recipients": [1]}, "recipients require send"),
        ({"mode": "send", "recipients": [2]}, "recipient outside"),
        ({"mode": "send", "recipients": [0]}, "recipient outside"),
        ({"mode": "send", "recipients": [1, 1]}, "duplicate recipients"),
        ({"mode": "send", "recipients": [-1]}, "recipient outside"),
        ({"mode": "send"}, "malformed planner control"),
        (
            {"mode": "send", "recipients": [1], "extra": True},
            "malformed planner control",
        ),
        (
            {"mode": "send", "recipients": [1], "work_instruction": 7},
            "must be a string",
        ),
        (
            {"mode": "send", "recipients": [1], "work_instruction": ""},
            "1..800 characters",
        ),
        (
            {"mode": "send", "recipients": [1], "work_instruction": "x" * 801},
            "1..800 characters",
        ),
        (
            {"mode": "submit", "recipients": [], "work_instruction": "w"},
            "forbidden at the submit barrier",
        ),
    ],
)
def test_planner_control_violations_fail_closed(
    tmp_path: Path, control: dict, match: str
) -> None:
    auth = _auth(n_agents=2, max_rounds=2)
    inner = _TextWorker()
    client, ledger = _metered_client(tmp_path, inner, auth)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match=match):
        _call(
            client,
            _comm_prompt(
                auth, agent_id=0, round_idx=0, control=control, submit_round=1
            ),
        )
    assert ledger["calls"] == []
    assert inner.json_modes == []


def test_forged_known_source_ids_fail_closed(tmp_path: Path) -> None:
    auth = _auth(n_agents=2, max_rounds=2)
    client, _ = _metered_client(tmp_path, _TextWorker(), auth)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="runtime provenance"):
        _call(
            client,
            _comm_prompt(
                auth,
                agent_id=0,
                round_idx=0,
                control=REFLECT,
                submit_round=1,
                known=[0, 1],
            ),
        )


def test_forged_previous_output_fails_closed(tmp_path: Path) -> None:
    auth = _auth(n_agents=2, max_rounds=2)
    client, _ = _metered_client(tmp_path, _TextWorker(), auth)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="previous output differs"):
        _call(
            client,
            _comm_prompt(
                auth,
                agent_id=0,
                round_idx=0,
                control=REFLECT,
                submit_round=1,
                prev="fabricated memory",
            ),
        )


def test_non_canonical_instruction_fails_closed(tmp_path: Path) -> None:
    auth = _auth(n_agents=2, max_rounds=2)
    inner = _TextWorker()
    client, _ = _metered_client(tmp_path, inner, auth)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="canonical message_only_v2 form"):
        _call(
            client,
            _comm_prompt(
                auth,
                agent_id=0,
                round_idx=0,
                control=REFLECT,
                submit_round=1,
                instruction="INSTRUCTION:\nDo something else entirely.\n",
            ),
        )
    assert inner.json_modes == []


def test_same_round_output_is_not_visible_and_delivery_is_next_round(
    tmp_path: Path,
) -> None:
    auth = _auth(n_agents=2, max_rounds=3)
    client, _ = _metered_client(
        tmp_path,
        _TextWorker(["from zero", "from one", "one round later"]),  # type: ignore[arg-type]
        auth,
    )
    _call(
        client,
        _comm_prompt(
            auth,
            agent_id=0,
            round_idx=0,
            control={"mode": "send", "recipients": [1]},
            submit_round=2,
        ),
    )
    fresh = dict(client.pending_envelopes[0])
    fresh.pop("merged")
    # Same-round visibility of agent 0's fresh envelope must be rejected.
    with pytest.raises(RuntimeError, match="does not match runtime deliveries"):
        _call(
            client,
            _comm_prompt(
                auth,
                agent_id=1,
                round_idx=0,
                control=REFLECT,
                submit_round=2,
                inbox=[fresh],
            ),
        )
    # The same envelope is the mandatory inbox one round later.
    _call(
        client,
        _comm_prompt(auth, agent_id=1, round_idx=0, control=REFLECT, submit_round=2),
    )
    _call(
        client,
        _comm_prompt(
            auth,
            agent_id=1,
            round_idx=1,
            control=REFLECT,
            submit_round=2,
            known=[0, 1],
            prev="from one",
            inbox=[fresh],
        ),
    )
    assert sorted(client.known_sources[1]) == [0, 1]


def test_idle_round_deliveries_merge_without_a_call(tmp_path: Path) -> None:
    auth = _auth(n_agents=3, max_rounds=4)
    client, _ = _metered_client(tmp_path, _TextWorker(), auth)  # type: ignore[arg-type]
    _call(
        client,
        _comm_prompt(
            auth,
            agent_id=0,
            round_idx=0,
            control={"mode": "send", "recipients": [1]},
            submit_round=3,
        ),
    )
    # Agent 1 idles through round 1 (no call); at round 2 the missed delivery
    # is merged into provenance but is not part of this round's inbox.
    _call(
        client,
        _comm_prompt(
            auth,
            agent_id=1,
            round_idx=2,
            control=REFLECT,
            submit_round=3,
            known=[0, 1],
        ),
    )
    assert sorted(client.known_sources[1]) == [0, 1]


def test_message_budget_fails_closed(tmp_path: Path) -> None:
    auth = _auth(n_agents=3, max_rounds=2, max_messages=1)
    client, _ = _metered_client(tmp_path, _TextWorker(), auth)  # type: ignore[arg-type]
    _call(
        client,
        _comm_prompt(
            auth,
            agent_id=0,
            round_idx=0,
            control={"mode": "send", "recipients": [1]},
            submit_round=1,
        ),
    )
    with pytest.raises(RuntimeError, match="message budget exhausted"):
        _call(
            client,
            _comm_prompt(
                auth,
                agent_id=1,
                round_idx=0,
                control={"mode": "send", "recipients": [2]},
                submit_round=1,
            ),
        )


def test_round_ordering_and_duplicate_calls_fail_closed(tmp_path: Path) -> None:
    auth = _auth(n_agents=2, max_rounds=3)
    client, _ = _metered_client(tmp_path, _TextWorker(), auth)  # type: ignore[arg-type]
    _call(
        client,
        _comm_prompt(auth, agent_id=0, round_idx=1, control=REFLECT, submit_round=2),
    )
    with pytest.raises(RuntimeError, match="monotonic round ordering"):
        _call(
            client,
            _comm_prompt(
                auth, agent_id=1, round_idx=0, control=REFLECT, submit_round=2
            ),
        )
    with pytest.raises(RuntimeError, match="more than once in the same round"):
        _call(
            client,
            _comm_prompt(
                auth,
                agent_id=0,
                round_idx=1,
                control=REFLECT,
                submit_round=2,
                prev="worker text output",
            ),
        )
    with pytest.raises(RuntimeError, match="inconsistent PYTHON_SUBMIT_ROUND"):
        _call(
            client,
            _comm_prompt(
                auth, agent_id=1, round_idx=1, control=REFLECT, submit_round=1
            ),
        )


def test_cross_agent_canary_leak_fails_closed(tmp_path: Path) -> None:
    canaries = {"0": "CANARY_ZERO_A91F", "1": "CANARY_ONE_B82E"}
    auth = _auth(
        n_agents=2,
        max_rounds=2,
        canaries=canaries,
        communication_prompts={
            "0": "CANARY_ZERO_A91F plus CANARY_ONE_B82E leaked",
            "1": "CANARY_ONE_B82E private",
        },
    )
    inner = _TextWorker()
    client, _ = _metered_client(tmp_path, inner, auth)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="local prompt leaked"):
        _call(
            client,
            _comm_prompt(
                auth, agent_id=0, round_idx=0, control=REFLECT, submit_round=1
            ),
        )
    assert inner.json_modes == []


def test_unsupported_wrapper_contract_fails_closed(tmp_path: Path) -> None:
    auth = {**_auth(), "worker_contract": "bogus_v9"}
    inner = _TextWorker()
    client, _ = _metered_client(tmp_path, inner, auth)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="unsupported worker contract"):
        _call(client, _submit_prompt())
    assert inner.json_modes == []


# --------------------------------------------------------------------------- #
# Subprocess runner: provenance and fail-closed reconciliation
# --------------------------------------------------------------------------- #


def test_all_agents_run_has_runtime_owned_provenance() -> None:
    result = CodeProcessRunner().run(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        _payload(n_agents=3, max_rounds=3, max_model_calls=20),
    )
    assert result.runtime_success, result.failure
    output = result.output
    assert output is not None
    calls = {
        (int(c["round"]), int(c["agent_id"])): c for c in result.ledger["calls"]
    }
    # Messages deliver exactly one round later and carry runtime provenance.
    for message in output.messages:
        assert message.round_delivered == message.round_sent + 1
        call = calls[(message.round_sent, message.src)]
        assert call["mode"] == "send"
        assert message.dst in call["recipients"]
        assert sorted(message.source_ids) == call["known_source_ids"]
        assert _value_sha256(message.body) == call["worker_output_sha256"]
    # Planner send controls and emitted messages correspond one-to-one.
    expected = {
        (r, a, d)
        for (r, a), c in calls.items()
        if c["mode"] == "send"
        for d in c["recipients"]
    }
    assert {(m.round_sent, m.src, m.dst) for m in output.messages} == expected
    assert result.final_knowledge == [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    for item in output.submissions:
        call = calls[(item.submitted_round, item.agent_id)]
        assert call["mode"] == "submit"
        assert call["answer_sha256"] == _value_sha256(item.answer)
    # Authoritative usage reconciles with the program-reported usage.
    assert output.usage == result.authoritative_usage


def test_sink_primary_is_not_called_before_the_barrier() -> None:
    result = CodeProcessRunner().run(
        DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
        _payload(goal="sink", n_agents=3, max_rounds=2, max_model_calls=4),
    )
    assert result.runtime_success, result.failure
    pairs = {(int(c["round"]), int(c["agent_id"])) for c in result.ledger["calls"]}
    # The sink idles in round 0 (idle = no worker call, no message).
    assert pairs == {(0, 1), (0, 2), (1, 0)}
    assert result.final_knowledge[0] == [0, 1, 2]


def test_program_forging_source_ids_fails_closed() -> None:
    forged = DEFAULT_MESSAGE_ONLY_V2_PROGRAM.replace(
        '"source_ids": list(snapshot_states[agent_id]["source_ids"]),\n'
        '                        "body": worker_output,',
        '"source_ids": [0, 1],\n'
        '                        "body": worker_output,',
    )
    assert forged != DEFAULT_MESSAGE_ONLY_V2_PROGRAM
    assert validate_python_source(forged).valid
    result = CodeProcessRunner().run(
        forged,
        _payload(n_agents=2, max_rounds=3, max_model_calls=8),
    )
    assert not result.runtime_success
    assert result.failure is not None
    assert result.failure["error_type"] == "DataFlowError"


def test_unsupported_payload_contract_fails_closed() -> None:
    runner = CodeProcessRunner()
    payload = _payload(n_agents=2, max_rounds=2, max_model_calls=4)
    other = {**payload, "worker_contract": "bogus_v9"}
    rejected = runner.run(DEFAULT_MESSAGE_ONLY_V2_PROGRAM, other)
    assert not rejected.runtime_success
    assert rejected.failure is not None
    assert rejected.failure["error_type"] == "APIError"
    assert rejected.authoritative_usage.model_calls == 0
    result = runner.run(DEFAULT_MESSAGE_ONLY_V2_PROGRAM, payload)
    assert result.runtime_success and result.output is not None
    with pytest.raises(PythonCodeError, match="unsupported python worker contract"):
        runner._validate_output(
            result.output,
            other,
            result.authoritative_usage,
            result.ledger,
        )
