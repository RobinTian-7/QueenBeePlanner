"""Isolated subprocess execution for validated model-generated Python code.

:class:`CodeProcessRunner` validates a team program, runs it in a fresh
interpreter (``exp_graph.mas.python_worker_bootstrap``) under wall-clock and
POSIX resource limits, with the worker LLM client metered inside the
sandbox, and then reconciles the program's stdout against the authoritative
call ledger the bootstrap writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, Field

from exp_graph.llm.openai_client import _DEFAULT_API_KEY_ENVS
from exp_graph.mas.python_code import (
    PythonCodeError,
    PythonExecutionPayload,
    PythonProgramOutput,
    PythonUsage,
    validate_python_source,
)


class PythonExecutionLimits(BaseModel):
    timeout_seconds: float = Field(default=300.0, gt=0)
    cpu_seconds: int = Field(default=10, ge=1)
    memory_mb: int = Field(default=512, ge=64)
    max_output_bytes: int = Field(default=1_000_000, ge=1024)


class PythonExecutionResult(BaseModel):
    runtime_success: bool
    output: PythonProgramOutput | None = None
    authoritative_usage: PythonUsage = Field(
        default_factory=lambda: PythonUsage(
            model_calls=0,
            prompt_tokens=0,
            completion_tokens=0,
        )
    )
    ledger: dict[str, Any] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    failure: dict[str, Any] | None = None
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    final_knowledge: list[list[int]] = Field(default_factory=list)


class CodeProcessRunner:
    """Run one validated source file with process-local client metering.

    ``extra_env``: environment variables added to the sandbox process."""

    def __init__(
        self,
        limits: PythonExecutionLimits | None = None,
        *,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self.limits = limits or PythonExecutionLimits()
        self.extra_env = {str(k): str(v) for k, v in (extra_env or {}).items()}

    def run(
        self,
        source: str,
        payload: dict[str, Any],
        *,
        agent_canaries: dict[str, str] | None = None,
    ) -> PythonExecutionResult:
        try:
            payload = PythonExecutionPayload.model_validate(payload).model_dump(
                mode="json"
            )
        except Exception:
            return self._failure(
                PythonCodeError(
                    "APIError",
                    "stdin payload violates the sanitized execution contract",
                )
            )
        if int(payload["budgets"]["max_model_calls"]) < int(payload["n_agents"]):
            return self._failure(
                PythonCodeError(
                    "BudgetError",
                    "model-call budget cannot execute one simultaneous agent round",
                )
            )
        worker_contract = str(payload["worker_contract"])
        report = validate_python_source(source, worker_contract=worker_contract)
        if not report.valid:
            return self._failure(
                PythonCodeError("PolicyError", "source failed static validation"),
                stderr=json.dumps(report.model_dump(mode="json")),
            )
        with tempfile.TemporaryDirectory(prefix="queenbee-python-") as raw_dir:
            work_dir = Path(raw_dir)
            program_path = work_dir / "program.py"
            auth_path = work_dir / "auth.json"
            ledger_path = work_dir / "ledger.json"
            stdout_path = work_dir / "stdout.txt"
            stderr_path = work_dir / "stderr.txt"
            program_path.write_text(source, encoding="utf-8")
            auth = {
                "worker_llm": payload["worker_llm"],
                "worker_contract": worker_contract,
                "budgets": payload["budgets"],
                "agent_canaries": agent_canaries or {},
                "max_output_bytes": self.limits.max_output_bytes,
                "n_agents": payload["n_agents"],
                "information_goal": payload["information_goal"],
                "selected_primary": payload["selected_primary"],
                "max_rounds": payload["max_rounds"],
                "max_parallel_agents": payload["max_parallel_agents"],
                "agent_communication_prompts": {
                    str(item["agent_id"]): item["communication_prompt"]
                    for item in payload["agents"]
                },
                "agent_submit_prompts": {
                    str(item["agent_id"]): item["submit_prompt"]
                    for item in payload["agents"]
                },
            }
            auth_path.write_text(json.dumps(auth), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "exp_graph.mas.python_worker_bootstrap",
                "--program",
                str(program_path),
                "--auth",
                str(auth_path),
                "--ledger",
                str(ledger_path),
            ]
            try:
                with stdout_path.open("wb") as stdout_handle, stderr_path.open(
                    "wb"
                ) as stderr_handle:
                    completed = subprocess.run(
                        command,
                        input=json.dumps(payload).encode("utf-8"),
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        timeout=self.limits.timeout_seconds,
                        cwd=work_dir,
                        env=self._child_env(payload),
                        preexec_fn=self._resource_limiter(),
                        check=False,
                    )
            except subprocess.TimeoutExpired:
                ledger = self._read_ledger(ledger_path)
                try:
                    usage = PythonUsage.model_validate(
                        ledger.get(
                            "usage",
                            {
                                "model_calls": 0,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            },
                        )
                    )
                except Exception:
                    usage = PythonUsage(
                        model_calls=0,
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
                return self._failure(
                    PythonCodeError(
                        "BudgetError",
                        f"python execution exceeded {self.limits.timeout_seconds}s",
                    ),
                    stdout=_read_limited_text(
                        stdout_path, self.limits.max_output_bytes
                    ),
                    stderr=_read_limited_text(stderr_path, 20_000),
                    ledger=ledger,
                    usage=usage,
                )
            stdout = _read_limited_text(
                stdout_path,
                self.limits.max_output_bytes + 1,
            )
            stderr = self._redact(
                _read_limited_text(stderr_path, 20_001),
                payload,
                agent_canaries or {},
            )
            ledger = self._read_ledger(ledger_path)
            usage = PythonUsage.model_validate(
                ledger.get(
                    "usage",
                    {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                )
            )
            for canary in (agent_canaries or {}).values():
                if canary and canary in stdout:
                    return self._failure(
                        PythonCodeError(
                            "DataFlowError",
                            "a private prompt canary reached program stdout",
                        ),
                        stdout="[REDACTED_CANARY_OUTPUT]",
                        stderr=stderr,
                        return_code=completed.returncode,
                        ledger=ledger,
                        usage=usage,
                    )
            if len(stdout.encode("utf-8")) > self.limits.max_output_bytes:
                return self._failure(
                    PythonCodeError("BudgetError", "stdout exceeds output-size limit"),
                    stdout=stdout[:1000],
                    stderr=stderr,
                    return_code=completed.returncode,
                    ledger=ledger,
                    usage=usage,
                )
            if completed.returncode != 0:
                message = _last_error_message(ledger) or "generated program exited non-zero"
                error_type = _classify_runtime_error(message)
                return self._failure(
                    PythonCodeError(error_type, message),
                    stdout=stdout,
                    stderr=stderr,
                    return_code=completed.returncode,
                    ledger=ledger,
                    usage=usage,
                )
            try:
                parsed = json.loads(stdout.strip())
                output = PythonProgramOutput.model_validate(parsed)
                knowledge = self._validate_output(output, payload, usage, ledger)
            except PythonCodeError as exc:
                return self._failure(
                    exc,
                    stdout=stdout,
                    stderr=stderr,
                    return_code=completed.returncode,
                    ledger=ledger,
                    usage=usage,
                )
            except Exception as exc:
                return self._failure(
                    PythonCodeError("OutputSchemaError", str(exc)),
                    stdout=stdout,
                    stderr=stderr,
                    return_code=completed.returncode,
                    ledger=ledger,
                    usage=usage,
                )
            return PythonExecutionResult(
                runtime_success=True,
                output=output,
                authoritative_usage=usage,
                ledger=ledger,
                stdout=stdout,
                stderr=stderr,
                return_code=completed.returncode,
                resource_limits=self._resource_report(),
                final_knowledge=[sorted(items) for items in knowledge],
            )

    def _validate_output(
        self,
        output: PythonProgramOutput,
        payload: dict[str, Any],
        authoritative_usage: PythonUsage,
        ledger: dict[str, Any],
    ) -> list[set[int]]:
        worker_contract = str(
            payload.get("worker_contract") or "message_only_v2"
        )
        if worker_contract != "message_only_v2":
            raise PythonCodeError(
                "APIError",
                f"unsupported python worker contract {worker_contract!r}",
            )
        return self._validate_message_only_output(
            output,
            payload,
            authoritative_usage,
            ledger,
            worker_contract=worker_contract,
        )

    def _validate_message_only_output(
        self,
        output: PythonProgramOutput,
        payload: dict[str, Any],
        authoritative_usage: PythonUsage,
        ledger: dict[str, Any],
        *,
        worker_contract: str,
    ) -> list[set[int]]:
        """Reconcile message-only stdout against the authoritative ledger.

        Routing comes from the program's own planning code (its send /
        reflect / submit controls) and provenance from the runtime wrapper,
        so this check demands EXACT correspondence: every send-mode call
        produces exactly its recipients' messages with the wrapper's source
        ids and worker-output hash, every submit-mode call is exactly one
        submission, the required agents (all of them, or the selected primary
        under the ``sink`` goal) submit together at one barrier round, idle
        rounds are legal, and host-reconstructed knowledge must equal the
        wrapper's per-call source ledger. Returns that knowledge (the source
        ids each agent holds).
        """
        n_agents = int(payload["n_agents"])
        max_rounds = int(payload["max_rounds"])
        budgets = payload["budgets"]
        if str(ledger.get("worker_contract") or "") != worker_contract:
            raise PythonCodeError(
                "APIError",
                "authoritative ledger worker contract differs from the payload",
            )
        if int(ledger.get("factory_calls", 0) or 0) != 1:
            raise PythonCodeError(
                "APIError",
                "authoritative ledger requires exactly one executed factory call",
            )
        calls = list(ledger.get("calls") or [])
        if not calls:
            raise PythonCodeError(
                "APIError",
                "authoritative ledger contains no executed worker calls",
            )
        if sum(int(item.get("model_calls", 0) or 0) for item in calls) != int(
            authoritative_usage.model_calls
        ):
            raise PythonCodeError(
                "BudgetError",
                "authoritative call entries do not sum to authoritative usage",
            )
        ids = [submission.agent_id for submission in output.submissions]
        if sorted(ids) != list(range(n_agents)) or len(set(ids)) != n_agents:
            raise PythonCodeError(
                "OutputSchemaError",
                "submissions must contain exactly one slot for every agent",
            )
        if output.rounds_executed > max_rounds:
            raise PythonCodeError("BudgetError", "round budget exceeded")
        if len(output.messages) > int(budgets["max_messages"]):
            raise PythonCodeError("BudgetError", "message budget exceeded")
        if output.usage != authoritative_usage:
            raise PythonCodeError(
                "BudgetError",
                "program-reported usage does not match authoritative client ledger",
            )
        if authoritative_usage.model_calls > int(budgets["max_model_calls"]):
            raise PythonCodeError("BudgetError", "model call budget exceeded")
        if authoritative_usage.completion_tokens > int(
            budgets["max_completion_tokens"]
        ):
            raise PythonCodeError("BudgetError", "completion token budget exceeded")
        if output.errors:
            raise PythonCodeError(
                "RuntimeError",
                "generated program reported runtime/action errors",
            )
        call_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        call_rounds: list[int] = []
        v2_submit_round: int | None = None
        for call in calls:
            try:
                agent_id = int(call["agent_id"])
                round_idx = int(call["round"])
                mode = str(call["mode"])
                recipients = [int(value) for value in call.get("recipients") or []]
                known_ids = [
                    int(value) for value in call.get("known_source_ids") or []
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise PythonCodeError(
                    "OutputSchemaError",
                    "authoritative call ledger lacks control metadata",
                ) from exc
            if not 0 <= agent_id < n_agents or not 0 <= round_idx < max_rounds:
                raise PythonCodeError(
                    "OutputSchemaError",
                    "authoritative worker call has invalid agent or round",
                )
            if mode not in {"send", "reflect", "submit"}:
                raise PythonCodeError(
                    "DataFlowError",
                    "worker call outside the send/reflect/submit control modes",
                )
            if mode != "send" and recipients:
                raise PythonCodeError(
                    "DataFlowError",
                    "recipients recorded outside a send-mode call",
                )
            if len(set(recipients)) != len(recipients) or any(
                recipient < 0 or recipient >= n_agents or recipient == agent_id
                for recipient in recipients
            ):
                raise PythonCodeError(
                    "DataFlowError",
                    "planner recipients are invalid for the agent range",
                )
            pair = (round_idx, agent_id)
            if pair in call_by_pair:
                raise PythonCodeError(
                    "DataFlowError",
                    "an agent was called more than once in the same round",
                )
            call_by_pair[pair] = {
                **call,
                "mode": mode,
                "recipients": recipients,
                "known_source_ids": sorted(known_ids),
            }
            call_rounds.append(round_idx)
            try:
                declared_submit_round = int(call["submit_round"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PythonCodeError(
                    "OutputSchemaError",
                    "message_only_v2 call lacks submit-round metadata",
                ) from exc
            if not 0 <= declared_submit_round < max_rounds:
                raise PythonCodeError(
                    "DataFlowError",
                    "message_only_v2 submit barrier is outside the round budget",
                )
            if v2_submit_round is None:
                v2_submit_round = declared_submit_round
            elif v2_submit_round != declared_submit_round:
                raise PythonCodeError(
                    "DataFlowError",
                    "message_only_v2 calls disagree on the global submit round",
                )
            if mode == "submit" and round_idx != declared_submit_round:
                raise PythonCodeError(
                    "DataFlowError",
                    "an agent submitted before the synchronized barrier",
                )
            if mode != "submit" and round_idx >= declared_submit_round:
                raise PythonCodeError(
                    "DataFlowError",
                    "communication occurred at or after the submit barrier",
                )
        if output.rounds_executed != max(call_rounds) + 1:
            raise PythonCodeError(
                "OutputSchemaError",
                "rounds_executed disagrees with the authoritative call ledger",
            )
        submitted_round_by_agent = {
            item.agent_id: item.submitted_round for item in output.submissions
        }
        for submission in output.submissions:
            if submission.submitted_round is None:
                if submission.answer is not None:
                    raise PythonCodeError(
                        "DataFlowError",
                        f"agent {submission.agent_id} reports an answer "
                        "without a submit-mode worker call",
                    )
                continue
            if not 0 <= submission.submitted_round < output.rounds_executed:
                raise PythonCodeError(
                    "OutputSchemaError",
                    f"invalid submitted_round for agent {submission.agent_id}",
                )
            action_call = call_by_pair.get(
                (submission.submitted_round, submission.agent_id)
            )
            if action_call is None or action_call["mode"] != "submit":
                raise PythonCodeError(
                    "DataFlowError",
                    f"agent {submission.agent_id} submitted without a "
                    "Planner submit control",
                )
            if action_call.get("answer_sha256") != _value_sha256(
                submission.answer
            ):
                raise PythonCodeError(
                    "DataFlowError",
                    f"agent {submission.agent_id} submission was not its "
                    "parsed Worker answer",
                )
            if action_call.get("answer_format_valid") is not True:
                raise PythonCodeError(
                    "AnswerFormatError",
                    f"agent {submission.agent_id} submit output was not valid JSON",
                )
            if any(
                call_agent == submission.agent_id
                and call_round > submission.submitted_round
                for call_round, call_agent in call_by_pair
            ):
                raise PythonCodeError(
                    "DataFlowError",
                    f"agent {submission.agent_id} was called after submission",
                )
        expected_submit_pairs = {
            pair
            for pair, call in call_by_pair.items()
            if call["mode"] == "submit"
        }
        actual_submit_pairs = {
            (item.submitted_round, item.agent_id)
            for item in output.submissions
            if item.submitted_round is not None
        }
        if actual_submit_pairs != expected_submit_pairs:
            raise PythonCodeError(
                "DataFlowError",
                "program submissions do not match Planner submit controls",
            )
        if v2_submit_round is None:
            raise PythonCodeError(
                "OutputSchemaError",
                "message_only_v2 ledger has no global submit round",
            )
        submit_round = v2_submit_round
        expected_submit_ids = (
            list(range(n_agents))
            if payload["information_goal"] == "all_agents"
            else [int(payload["selected_primary"])]
        )
        expected_barrier_pairs = {
            (submit_round, agent_id) for agent_id in expected_submit_ids
        }
        if actual_submit_pairs != expected_barrier_pairs:
            raise PythonCodeError(
                "DataFlowError",
                "required agents did not submit together at the final barrier",
            )
        if output.rounds_executed != submit_round + 1:
            raise PythonCodeError(
                "OutputSchemaError",
                "message_only_v2 execution did not end at the submit barrier",
            )
        barrier = ledger.get("submit_barrier")
        if not isinstance(barrier, dict):
            raise PythonCodeError(
                "OutputSchemaError",
                "authoritative ledger lacks submit-barrier audit metadata",
            )
        frozen_hash = str(barrier.get("final_snapshot_sha256") or "")
        if (
            barrier.get("enabled") is not True
            or barrier.get("worker_contract") != "message_only_v2"
            or barrier.get("information_goal") != payload["information_goal"]
            or int(barrier.get("submit_round", -1)) != submit_round
            or barrier.get("expected_agent_ids") != expected_submit_ids
            or barrier.get("observed_agent_ids") != expected_submit_ids
            or barrier.get("synchronized") is not True
            or barrier.get("answer_format") != "single_json_value"
            or barrier.get("parser") != "json.loads_whole_response"
            or len(frozen_hash) != 64
        ):
            raise PythonCodeError(
                "DataFlowError",
                "submit-barrier audit metadata is incomplete or inconsistent",
            )
        by_delivery: dict[int, list[Any]] = {}
        by_send: dict[int, list[Any]] = {}
        message_keys: set[tuple[int, int, int]] = set()
        for message in output.messages:
            if not (0 <= message.src < n_agents and 0 <= message.dst < n_agents):
                raise PythonCodeError(
                    "OutputSchemaError", "message agent id out of range"
                )
            if message.round_sent >= output.rounds_executed:
                raise PythonCodeError(
                    "OutputSchemaError", "message sent after execution"
                )
            if message.round_delivered >= output.rounds_executed:
                raise PythonCodeError(
                    "OutputSchemaError",
                    "messages must be actually delivered before execution ends",
                )
            submitted_round = submitted_round_by_agent.get(message.src)
            if submitted_round is not None and message.round_sent >= submitted_round:
                raise PythonCodeError(
                    "DataFlowError",
                    f"agent {message.src} sent after or while submitting",
                )
            action_call = call_by_pair.get((message.round_sent, message.src))
            if action_call is None or action_call["mode"] != "send":
                raise PythonCodeError(
                    "DataFlowError",
                    "a message sender had no send-mode Planner control",
                )
            if message.dst not in action_call["recipients"]:
                raise PythonCodeError(
                    "DataFlowError",
                    "program sent to a recipient absent from the Planner control",
                )
            message_key = (message.round_sent, message.src, message.dst)
            if message_key in message_keys:
                raise PythonCodeError(
                    "DataFlowError",
                    "duplicate delivery to one recipient in a round",
                )
            message_keys.add(message_key)
            if sorted(message.source_ids) != list(
                action_call["known_source_ids"]
            ):
                raise PythonCodeError(
                    "DataFlowError",
                    "message provenance differs from the runtime source ledger",
                )
            if _value_sha256(message.body) != action_call.get(
                "worker_output_sha256"
            ):
                raise PythonCodeError(
                    "DataFlowError",
                    "message body differs from the Worker output",
                )
            by_send.setdefault(message.round_sent, []).append(message)
            by_delivery.setdefault(message.round_delivered, []).append(message)
        expected_message_keys = {
            (round_idx, agent_id, recipient)
            for (round_idx, agent_id), call in call_by_pair.items()
            if call["mode"] == "send"
            for recipient in call["recipients"]
        }
        if message_keys != expected_message_keys:
            raise PythonCodeError(
                "DataFlowError",
                "emitted messages do not match Planner send controls",
            )
        for (round_idx, agent_id), call in call_by_pair.items():
            expected_inbox = sorted(
                (
                    {
                        "round_sent": item.round_sent,
                        "round_delivered": item.round_delivered,
                        "src": item.src,
                        "dst": item.dst,
                        "source_ids": sorted(item.source_ids),
                        "body_sha256": _value_sha256(item.body),
                    }
                    for item in by_delivery.get(round_idx, [])
                    if item.dst == agent_id
                ),
                key=lambda item: json.dumps(item, sort_keys=True),
            )
            actual_inbox = sorted(
                list(call.get("delivered_inbox") or []),
                key=lambda item: json.dumps(item, sort_keys=True),
            )
            if actual_inbox != expected_inbox:
                raise PythonCodeError(
                    "DataFlowError",
                    "worker inbox does not match messages delivered from prior rounds",
                )
        knowledge = [{agent_id} for agent_id in range(n_agents)]
        for round_idx in range(output.rounds_executed):
            for delivered in by_delivery.get(round_idx, []):
                knowledge[delivered.dst].update(delivered.source_ids)
            for (call_round, agent_id), call in call_by_pair.items():
                if call_round != round_idx:
                    continue
                if call["known_source_ids"] != sorted(knowledge[agent_id]):
                    raise PythonCodeError(
                        "DataFlowError",
                        "runtime source ledger diverges from host-reconstructed "
                        "knowledge",
                    )
            for sent in by_send.get(round_idx, []):
                if sorted(sent.source_ids) != sorted(knowledge[sent.src]):
                    raise PythonCodeError(
                        "DataFlowError",
                        f"agent {sent.src} message provenance must equal its "
                        "reconstructed knowledge",
                    )
        return knowledge

    def _child_env(self, payload: dict[str, Any]) -> dict[str, str]:
        entries = [str(Path(item)) for item in sys.path if item]
        env = {
            "PYTHONPATH": os.pathsep.join(entries),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        # Forward the worker's credential into the sandbox: api_key_env when
        # set, else the provider's default key variable, resolved exactly as
        # OpenAIChatClient.__init__ does (a mapped provider -> its variable;
        # any other, e.g. plain openai -> the SDK's implicit OPENAI_API_KEY).
        # The offline ``fake`` provider needs no credential and gets none.
        # Validated programs cannot read env/files/network, so forwarding this
        # one key is not an exfiltration surface.
        provider = str(payload["worker_llm"].get("provider", "")).lower()
        api_key_env = payload["worker_llm"].get("api_key_env")
        if provider and provider != "fake":
            key_env = (
                api_key_env
                or _DEFAULT_API_KEY_ENVS.get(provider)
                or "OPENAI_API_KEY"
            )
            if os.environ.get(key_env):
                env[str(key_env)] = os.environ[str(key_env)]
        for name in (
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
            # Worker reasoning effort (OpenAI reasoning models): the payload
            # schema has no effort field, so the setting rides the environment.
            "OPENAI_REASONING_EFFORT",
            # The HTTP deadline rides the environment too: the payload's
            # request_timeout only sets the OUTER wall-clock guard
            # (timeout_s), while the inner httpx deadline comes from
            # OPENAI_TIMEOUT (default 120 s). Without forwarding these, a long
            # worker call fails after about 3 x 120 s (three attempts with the
            # default OPENAI_MAX_RETRIES=2) whatever request_timeout says.
            "OPENAI_TIMEOUT",
            "OPENAI_CONNECT_TIMEOUT",
            "OPENAI_MAX_RETRIES",
            # The worker endpoint must ride into the sandbox with the credential:
            # any OpenAI-compatible endpoint other than api.openai.com is
            # addressed through it. Without it the SDK falls back to
            # api.openai.com, which rejects a key issued for another endpoint
            # (HTTP 401) on every worker call: every program scores S=0 and
            # races see only ties.
            "OPENAI_BASE_URL",
        ):
            if os.environ.get(name):
                env[name] = os.environ[name]
        # Only workers run in the sandbox: the worker completion cap
        # (QUEENBEE_WORKER_MAX_COMPLETION_TOKENS) wins over the global one.
        _wcap = os.environ.get("QUEENBEE_WORKER_MAX_COMPLETION_TOKENS")
        if _wcap:
            env["OPENAI_MAX_COMPLETION_TOKENS"] = _wcap
        elif os.environ.get("OPENAI_MAX_COMPLETION_TOKENS"):
            env["OPENAI_MAX_COMPLETION_TOKENS"] = os.environ[
                "OPENAI_MAX_COMPLETION_TOKENS"
            ]
        env.update(self.extra_env)
        return env

    def _resource_limiter(self):
        if os.name != "posix":
            return None

        limits = self.limits

        def apply_limits() -> None:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
            memory = limits.memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            except (ValueError, OSError):
                pass
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            try:
                resource.setrlimit(
                    resource.RLIMIT_FSIZE,
                    (
                        max(10_000_000, limits.max_output_bytes + 1),
                        max(10_000_000, limits.max_output_bytes + 1),
                    ),
                )
            except (ValueError, OSError):
                pass

        return apply_limits

    def _resource_report(self) -> dict[str, Any]:
        return {
            "wallclock_timeout_seconds": self.limits.timeout_seconds,
            "cpu_seconds": self.limits.cpu_seconds,
            "memory_mb": self.limits.memory_mb,
            "max_output_bytes": self.limits.max_output_bytes,
            "posix_rlimits_requested": os.name == "posix",
            "platform_caveat": (
                "RLIMIT_AS/FSIZE availability is platform-dependent; wallclock, "
                "schema, AST, API and ledger checks remain enforced."
            ),
        }

    @staticmethod
    def _read_ledger(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _redact(
        text: str,
        payload: dict[str, Any],
        canaries: dict[str, str],
    ) -> str:
        redacted = text
        for agent in payload.get("agents", []):
            for field in ("communication_prompt", "submit_prompt"):
                prompt = str(agent.get(field) or "")
                if len(prompt) >= 8:
                    redacted = redacted.replace(
                        prompt,
                        f"[REDACTED_{field.upper()}]",
                    )
        for canary in canaries.values():
            redacted = redacted.replace(canary, "[REDACTED_CANARY]")
        return redacted[:20_000]

    def _failure(
        self,
        error: PythonCodeError,
        *,
        stdout: str = "",
        stderr: str = "",
        return_code: int | None = None,
        ledger: dict[str, Any] | None = None,
        usage: PythonUsage | None = None,
    ) -> PythonExecutionResult:
        return PythonExecutionResult(
            runtime_success=False,
            authoritative_usage=usage
            or PythonUsage(model_calls=0, prompt_tokens=0, completion_tokens=0),
            ledger=ledger or {},
            stdout=stdout[:20_000],
            stderr=stderr[:20_000],
            return_code=return_code,
            failure=error.as_dict(),
            resource_limits=self._resource_report(),
        )


def _read_limited_text(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        return handle.read(max(0, limit)).decode("utf-8", errors="replace")


def _value_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _last_error_message(ledger: dict[str, Any]) -> str | None:
    errors = ledger.get("errors") or []
    if errors and isinstance(errors[-1], dict):
        return str(errors[-1].get("message") or errors[-1].get("type"))
    return None


def _classify_runtime_error(message: str):
    if "file too large" in message.lower():
        return "BudgetError"
    for prefix in (
        "SyntaxError",
        "PolicyError",
        "APIError",
        "DataFlowError",
        "BudgetError",
        "OutputSchemaError",
        "AnswerFormatError",
    ):
        if prefix in message:
            return prefix
    return "RuntimeError"
