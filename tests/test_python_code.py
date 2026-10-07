"""Offline security, semantics, and isolation tests for generated Python programs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import exp_graph
from exp_graph.llm.openai_client import _DEFAULT_API_KEY_ENVS
from exp_graph.mas.information_flow import coverage_by_agent
from exp_graph.mas.leakage_audit import (
    PromptLeakageError,
    assert_prompt_clean,
    find_leakage_tokens,
)
from exp_graph.mas.python_code import (
    DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
    PythonExecutionPayload,
    PythonProgramOutput,
    validate_python_source,
)
from exp_graph.mas.python_code_runner import CodeProcessRunner, PythonExecutionLimits

PROGRAM = DEFAULT_MESSAGE_ONLY_V2_PROGRAM


def _edit(source: str, old: str, new: str) -> str:
    """Replace the first occurrence of ``old`` (which must exist)."""
    assert old in source, old
    return source.replace(old, new, 1)


def _payload(
    *,
    goal: str = "all_agents",
    n_agents: int = 3,
    max_rounds: int = 2,
    max_model_calls: int = 20,
) -> dict:
    return {
        "execution_contract_version": "python_mas_v1",
        "task_description": "Synthetic offline wiring check.",
        "information_goal": goal,
        "selected_primary": 0,
        "n_agents": n_agents,
        "max_rounds": max_rounds,
        "budgets": {
            "max_model_calls": max_model_calls,
            "max_completion_tokens": 4000,
            "max_messages": 30,
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
                "agent_id": i,
                "communication_prompt": f"PRIVATE_COMMUNICATION_{i}",
                "submit_prompt": f"PRIVATE_SUBMIT_{i}",
            }
            for i in range(n_agents)
        ],
    }


def test_default_program_validates_and_executes_round_semantics() -> None:
    assert PythonExecutionPayload.model_validate(_payload()).worker_contract == (
        "message_only_v2"
    )
    report = validate_python_source(PROGRAM)
    assert report.valid, report.errors

    result = CodeProcessRunner().run(PROGRAM, _payload())
    assert result.runtime_success, result.failure
    assert result.output is not None
    assert result.output.rounds_executed == 2
    assert result.authoritative_usage.model_calls == 6
    # Round 0 broadcasts (3 agents x 2 peers); round 1 is the submit barrier.
    assert len(result.output.messages) == 6
    assert all(
        (message.round_sent, message.round_delivered) == (0, 1)
        for message in result.output.messages
    )
    round_one = [call for call in result.ledger["calls"] if call["round"] == 1]
    assert len(round_one) == 3
    assert all(call["mode"] == "submit" for call in round_one)
    assert all(len(call["delivered_inbox"]) == 2 for call in round_one)
    assert all(call["known_source_ids"] == [0, 1, 2] for call in round_one)
    assert len(result.output.submissions) == 3
    assert all(item.submitted_round == 1 for item in result.output.submissions)


@pytest.mark.parametrize(
    "snippet",
    [
        "import os\n",
        "import subprocess\n",
        "import socket\n",
        "open('x')\n",
        "eval('1')\n",
        "exec('x=1')\n",
        "while True:\n    pass\n",
    ],
)
def test_forbidden_python_constructs_are_rejected(snippet: str) -> None:
    report = validate_python_source(snippet + PROGRAM)
    assert not report.valid
    assert report.errors[0]["error_type"] == "PolicyError"


def test_recursion_and_unknown_bound_are_rejected() -> None:
    recursive = _edit(
        PROGRAM,
        "def reject_nonfinite(value):",
        "def reject_nonfinite(value):\n    return reject_nonfinite(value)",
    )
    assert not validate_python_source(recursive).valid
    unknown = _edit(
        PROGRAM,
        "for round_idx in range(max_rounds):",
        "for round_idx in iter(payload['agents']):",
    )
    report = validate_python_source(unknown)
    assert not report.valid
    assert any("bounded" in error["message"] for error in report.errors)


def test_authorized_model_configuration_must_be_used_directly() -> None:
    source = _edit(
        PROGRAM,
        'model_name=worker_cfg["model_name"],',
        'model_name="other-model",',
    )
    report = validate_python_source(source)
    assert not report.valid
    assert any(error["error_type"] == "APIError" for error in report.errors)


def test_cross_agent_and_combined_private_prompts_are_rejected() -> None:
    cross = _edit(
        PROGRAM,
        'agents[agent_id]["communication_prompt"]',
        'agents[(agent_id + 1) % n_agents]["communication_prompt"]',
    )
    assert any(
        error["error_type"] == "DataFlowError"
        for error in validate_python_source(cross).errors
    )
    combined = _edit(
        PROGRAM,
        'communication_prompt = agents[agent_id]["communication_prompt"]',
        'communication_prompt = (agents[agent_id]["communication_prompt"] + '
        'agents[(agent_id + 1) % n_agents]["communication_prompt"])',
    )
    report = validate_python_source(combined)
    assert any(
        "multiple private prompts" in error["message"] for error in report.errors
    )


def test_prompt_fields_outside_the_contract_are_rejected() -> None:
    source = _edit(
        PROGRAM,
        'agents[agent_id]["communication_prompt"]',
        'agents[agent_id]["local_prompt"]',
    )
    report = validate_python_source(source)
    assert not report.valid
    assert any(
        "local_prompt is not available under the message_only_v2 worker contract"
        in error["message"]
        for error in report.errors
    )


def test_header_based_cross_agent_leak_fails_closed() -> None:
    # Agent a's prompt declares agent a + 1: static validation cannot see it,
    # the runtime rejects the call before any worker sees the prompt.
    source = _edit(
        PROGRAM,
        '"PYTHON_AGENT_ID:" + str(agent_id)',
        '"PYTHON_AGENT_ID:" + str((agent_id + 1) % n_agents)',
    )
    assert validate_python_source(source).valid
    canaries = {str(i): f"CANARY_{i}_SECRET" for i in range(3)}
    payload = _payload()
    for item in payload["agents"]:
        canary = canaries[str(item["agent_id"])]
        item["communication_prompt"] = canary
        item["submit_prompt"] = canary
    result = CodeProcessRunner().run(
        source,
        payload,
        agent_canaries=canaries,
    )
    assert not result.runtime_success
    assert result.failure["error_type"] == "DataFlowError"  # type: ignore[index]
    assert result.authoritative_usage.model_calls == 0


@pytest.mark.parametrize(
    "source",
    [
        _edit(
            PROGRAM,
            'worker_prompt = control_header + submit_prompt + "\\n" + SUBMIT_INSTRUCTION',
            'worker_prompt = control_header + submit_prompt + "\\n" + SUBMIT_INSTRUCTION'
            ' + "HIDDEN_CHANNEL"',
        ),
        _edit(
            PROGRAM,
            '"previous_output": worker_output,',
            '"previous_output": worker_output + " [unreported state]",',
        ),
        _edit(
            PROGRAM,
            "sys.stdout.write(json.dumps(output))",
            'messages[0]["body"] = "body-not-delivered"\n'
            "    sys.stdout.write(json.dumps(output))",
        ),
        _edit(
            PROGRAM,
            "sys.stdout.write(json.dumps(output))",
            'submissions[0]["answer"] = 12345\n'
            "    sys.stdout.write(json.dumps(output))",
        ),
    ],
)
def test_runtime_ledger_rejects_hidden_prompt_state_or_message_channels(
    source: str,
) -> None:
    assert validate_python_source(source).valid
    result = CodeProcessRunner().run(source, _payload())
    assert not result.runtime_success
    assert result.failure["error_type"] == "DataFlowError"  # type: ignore[index]


def test_budget_and_program_usage_lies_fail_closed() -> None:
    exhausted = CodeProcessRunner().run(
        PROGRAM,
        _payload(max_model_calls=2),
    )
    assert not exhausted.runtime_success
    assert exhausted.failure["error_type"] == "BudgetError"  # type: ignore[index]

    lying = _edit(
        PROGRAM,
        '"usage": usage,',
        '"usage": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},',
    )
    result = CodeProcessRunner().run(lying, _payload())
    assert not result.runtime_success
    assert result.failure["error_type"] == "BudgetError"  # type: ignore[index]


def test_stdin_payload_rejects_private_or_unknown_fields() -> None:
    payload = _payload()
    payload["ground_truth"] = 9
    result = CodeProcessRunner().run(PROGRAM, payload)
    assert not result.runtime_success
    assert result.failure["error_type"] == "APIError"  # type: ignore[index]
    assert "9" not in result.failure["message"]  # type: ignore[index]


def test_stdout_limit_is_enforced_during_child_execution() -> None:
    runner = CodeProcessRunner(PythonExecutionLimits(max_output_bytes=1024))
    honest = runner.run(PROGRAM, _payload(n_agents=2))
    assert honest.runtime_success, honest.failure
    source = _edit(
        PROGRAM,
        "sys.stdout.write(json.dumps(output))",
        'output["submissions"][0]["answer"] = "x" * 5000\n'
        "    sys.stdout.write(json.dumps(output))",
    )
    assert validate_python_source(source).valid
    result = runner.run(source, _payload(n_agents=2))
    assert not result.runtime_success
    assert result.failure["error_type"] == "BudgetError"  # type: ignore[index]


def test_child_env_forwards_worker_api_key(monkeypatch) -> None:
    """The sandbox must forward the worker's real credential into the subprocess.

    When ``api_key_env`` is null, the provider's default key variable is
    forwarded (for plain ``openai``: the OpenAI SDK's implicit
    OPENAI_API_KEY). The ``fake`` worker provider used for dry runs needs no
    credential and must not receive one.
    """
    runner = CodeProcessRunner(
        PythonExecutionLimits(
            timeout_seconds=10,
            cpu_seconds=10,
            memory_mb=512,
            max_output_bytes=1_000_000,
        )
    )

    def _env(provider: str, api_key_env: str | None = None) -> dict:
        return runner._child_env(
            {"worker_llm": {"provider": provider, "api_key_env": api_key_env}}
        )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("CUSTOM_WORKER_KEY", "sk-test-custom")

    # openai + null api_key_env -> forward the SDK-default OPENAI_API_KEY.
    assert _env("openai").get("OPENAI_API_KEY") == "sk-test-openai"
    # provider-default mappings are honored.
    for provider, key_env in _DEFAULT_API_KEY_ENVS.items():
        monkeypatch.setenv(key_env, f"sk-test-{provider}")
        assert _env(provider).get(key_env) == f"sk-test-{provider}"
    # an explicit api_key_env wins over the provider default.
    assert (
        _env("openai", "CUSTOM_WORKER_KEY").get("CUSTOM_WORKER_KEY")
        == "sk-test-custom"
    )
    # the fake dry-run worker needs no credential and must not receive one.
    assert "OPENAI_API_KEY" not in _env("fake")


def test_child_env_is_an_allow_list(monkeypatch) -> None:
    """Only allow-listed variables reach the sandbox (interpreter settings,
    the worker credential, endpoint and transport settings, the completion
    cap); any other variable is dropped."""
    runner = CodeProcessRunner()
    payload = {"worker_llm": {"provider": "openai", "api_key_env": None}}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("UNRELATED_SETTING", "not-for-the-sandbox")
    monkeypatch.delenv("QUEENBEE_WORKER_MAX_COMPLETION_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_MAX_COMPLETION_TOKENS", raising=False)

    env = runner._child_env(payload)
    assert env["OPENAI_BASE_URL"] == "http://localhost:1/v1"
    assert "UNRELATED_SETTING" not in env
    assert "OPENAI_MAX_COMPLETION_TOKENS" not in env
    # the child imports the runtime packages through the forwarded path
    src_dir = Path(exp_graph.__file__).resolve().parents[1]
    assert src_dir in {
        Path(entry).resolve() for entry in env["PYTHONPATH"].split(os.pathsep)
    }

    # the worker-tier cap wins over the global cap inside the sandbox
    monkeypatch.setenv("OPENAI_MAX_COMPLETION_TOKENS", "4096")
    assert runner._child_env(payload)["OPENAI_MAX_COMPLETION_TOKENS"] == "4096"
    monkeypatch.setenv("QUEENBEE_WORKER_MAX_COMPLETION_TOKENS", "2048")
    assert runner._child_env(payload)["OPENAI_MAX_COMPLETION_TOKENS"] == "2048"


def test_output_schema_rejects_extra_fields() -> None:
    with pytest.raises(Exception):
        PythonProgramOutput.model_validate(
            {
                "submissions": [],
                "rounds_executed": 0,
                "messages": [],
                "usage": {
                    "model_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
                "errors": [],
                "private_state": {},
            }
        )


def test_leakage_audit_is_case_insensitive_and_fails_loud() -> None:
    clean = "Find the global maximum of your private values."
    assert find_leakage_tokens(clean) == []
    assert assert_prompt_clean(clean) is clean
    assert find_leakage_tokens("see the Ground_Truth below") == ["ground_truth"]
    assert find_leakage_tokens("a POW2 schedule", allowed_tokens=["pow2"]) == []
    with pytest.raises(PromptLeakageError, match="optimal_topology"):
        assert_prompt_clean("the optimal_topology is a star", context="unit")


def test_coverage_by_agent_counts_known_ids_in_range() -> None:
    assert coverage_by_agent([]) == []
    assert coverage_by_agent([{0, 1}, {1}, {0, 1, 2, 7}]) == pytest.approx(
        [2 / 3, 1 / 3, 1.0]
    )
