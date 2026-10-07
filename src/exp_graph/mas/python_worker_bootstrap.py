"""Process-local security and metering wrapper for generated team programs.

This module runs only inside the isolated child process that
:class:`exp_graph.mas.python_code_runner.CodeProcessRunner` starts
(``--program``, ``--auth``, ``--ledger``).  Before the team program runs, it
replaces ``exp_graph.llm.factory.create_llm_client`` in that process with a
factory whose client keeps the public ``LLMClient.complete`` API but makes
the provider / model configuration, the ``message_only_v2`` worker contract,
the budgets and the usage accounting authoritative to the host: every worker
call is checked against the host's authorization and the runtime state, and
recorded in a JSON ledger file.  The program's stdout is capped at the
output-size budget.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import exp_graph.llm.factory as factory_module
from exp_graph.llm.base import (
    LLMResponse,
    LLMUsage,
    combine_usage,
    estimate_tokens,
)
from exp_graph.llm.factory import create_llm_client as _real_factory
from exp_graph.mas.count_table_json import install_if_requested
from exp_graph.mas.python_code import (
    MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION,
    MESSAGE_ONLY_V2_SUBMIT_INSTRUCTION,
)

_MESSAGE_ONLY_V2_MODES = ("send", "reflect", "submit")

# Appended to the submit prompt for the one-shot format repair call.
# A reasoning model's submit call can exhaust the per-call token cap while
# still reasoning (empty or truncated content) or wrap the answer in prose;
# the strict single-JSON parser would then fail the whole run.  Asking for a
# fresh answer, instead of extracting a JSON fragment from the rejected text,
# never guesses which value is the answer, so the ledger's answer rule (one
# whole-response JSON value) still holds.
_ANSWER_FORMAT_REPAIR_NOTICE = (
    "\nANSWER_FORMAT_REPAIR_NOTICE:\n"
    "The previous response to this exact prompt was rejected because the "
    "submit output was not exactly one valid JSON value (it was empty, "
    "truncated, or wrapped in prose).\n"
    "Emit the final submit output again NOW: exactly one valid JSON value, "
    "nothing else. No reasoning, no prose, no Markdown, no code fences, "
    "no NaN/Infinity."
)


class _LimitedStdout:
    """Byte counter on the child's stdout: the output-size budget is enforced
    in the child, before the host buffers any output."""

    def __init__(self, inner: Any, max_bytes: int) -> None:
        self.inner = inner
        self.max_bytes = max_bytes
        self.written = 0

    def write(self, value: str) -> int:
        size = len(value.encode("utf-8"))
        if self.written + size > self.max_bytes:
            raise RuntimeError("BudgetError: stdout output-size budget exceeded")
        written = self.inner.write(value)
        self.written += size
        return written

    def flush(self) -> None:
        self.inner.flush()


class _PythonSmokeClient:
    """Offline worker of the ``fake`` provider: it never solves a task (a
    submit returns the JSON string ``"UNKNOWN"``, other calls fixed text)."""

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        del model_name, temperature, json_mode
        control = _prompt_json_line(prompt, "PYTHON_CONTROL_JSON")
        if control.get("mode") == "submit":
            # A JSON string exercises the native-value answer parsing
            # without solving the task.
            text = json.dumps("UNKNOWN")
        else:
            text = (
                "Offline smoke summary: deterministic fake worker text. "
                "No offline task solver is used."
            )
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(text),
            ),
        )


class _BatchPreflightClient:
    """Return contract-valid placeholders while the wrapper validates a batch."""

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        del model_name, temperature, json_mode
        if _prompt_json_line(prompt, "PYTHON_CONTROL_JSON").get("mode") == "submit":
            text = "null"
        else:
            text = "parallel batch preflight"
        return LLMResponse(
            text=text,
            usage=LLMUsage(model_calls=1, prompt_tokens=0, completion_tokens=1),
        )


class _BatchReplayClient:
    """Hand back provider responses already fetched in parallel, in order, so
    they are committed through the normal authoritative path."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.index = 0

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        del prompt, model_name, temperature, json_mode
        if self.index >= len(self.responses):
            raise RuntimeError("APIError: parallel response replay underflow")
        response = self.responses[self.index]
        self.index += 1
        return response


class _MeteredClient:
    """The client a team program gets from ``create_llm_client``: it checks
    every worker call against the host authorization and the runtime state,
    enforces the budgets and records each call in the ledger."""

    # Runtime state saved before, and restored after, a batch's pre-flight.
    _BATCH_STATE_FIELDS = (
        "previous_outputs",
        "known_sources",
        "pending_envelopes",
        "submitted_rounds",
        "pending_barrier_submissions",
        "declared_submit_round",
        "frozen_barrier_states",
        "frozen_barrier_inboxes",
        "called_pairs",
        "max_round_seen",
        "total_messages",
    )

    def __init__(self, inner: Any, auth: dict[str, Any], ledger: dict[str, Any], path: Path):
        self.inner = inner
        self.auth = auth
        self.ledger = ledger
        self.path = path
        self.worker_contract = str(auth.get("worker_contract") or "")
        n_agents = int(auth["n_agents"])
        # Authoritative message_only_v2 runtime state; neither the worker nor
        # the team program can override it.  previous_outputs: each agent's
        # last send / reflect output; known_sources: the host-merged
        # provenance (the agent ids whose data reached each agent);
        # pending_envelopes: every sent envelope, delivered or not (bodies
        # stay in memory only, never in the ledger file); submitted_rounds:
        # agent -> submit round, set once every required submitter has
        # submitted.  The other fields hold the submits accepted so far, the
        # submit round every prompt must declare, each agent's view frozen at
        # the submit barrier, the (round, agent) pairs already called, the
        # highest round called (rounds never go back) and the number of
        # messages sent.
        self.previous_outputs: dict[int, str] = {
            agent_id: "" for agent_id in range(n_agents)
        }
        self.known_sources: dict[int, set[int]] = {
            agent_id: {agent_id} for agent_id in range(n_agents)
        }
        self.pending_envelopes: list[dict[str, Any]] = []
        self.submitted_rounds: dict[int, int] = {}
        self.pending_barrier_submissions: dict[int, int] = {}
        self.declared_submit_round: int | None = None
        self.frozen_barrier_states: dict[int, dict[str, Any]] = {}
        self.frozen_barrier_inboxes: dict[int, list[dict[str, Any]]] = {}
        self.called_pairs: set[tuple[int, int]] = set()
        self.max_round_seen = -1
        self.total_messages = 0

    def _snapshot_batch_state(self) -> dict[str, Any]:
        return {
            "ledger": copy.deepcopy(self.ledger),
            "state": {
                name: copy.deepcopy(getattr(self, name))
                for name in self._BATCH_STATE_FIELDS
            },
        }

    def _restore_batch_state(self, snapshot: dict[str, Any]) -> None:
        self.ledger.clear()
        self.ledger.update(copy.deepcopy(snapshot["ledger"]))
        for name, value in snapshot["state"].items():
            setattr(self, name, copy.deepcopy(value))
        _write_ledger(self.path, self.ledger)

    def complete_batch(
        self,
        prompts: list[str],
        model_name: str,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> list[LLMResponse]:
        """Execute one logical round's independent worker calls concurrently.

        The team program chooses the communication topology, but the trusted
        wrapper owns concurrency: it first runs the fail-closed validation and
        budget logic against placeholders, restores the state, performs only
        the provider calls in parallel, then commits the responses in input
        order through the same authoritative path as :meth:`complete`.
        """
        if not isinstance(prompts, list) or any(
            not isinstance(prompt, str) for prompt in prompts
        ):
            raise RuntimeError("APIError: complete_batch requires a list of prompts")
        if not prompts:
            return []
        n_agents = int(self.auth["n_agents"])
        if len(prompts) > n_agents:
            raise RuntimeError(
                "BudgetError: a parallel batch cannot exceed the agent count"
            )
        rounds = {_prompt_header_int(prompt, "PYTHON_ROUND") for prompt in prompts}
        if len(rounds) != 1:
            raise RuntimeError(
                "DataFlowError: complete_batch may contain only one logical round"
            )

        snapshot = self._snapshot_batch_state()
        real_inner = self.inner
        # During the batch self.inner is the placeholder or the replay client,
        # but a submit-format repair call must reach the real provider:
        # otherwise a repair during the replay would consume the next agent's
        # fetched response (silently swapping answers) or cause a replay
        # underflow / overflow.  Repair calls therefore use _repair_inner,
        # the real client while the batch runs.
        self._repair_inner = real_inner
        try:
            self.inner = _BatchPreflightClient()
            for prompt in prompts:
                self.complete(
                    prompt,
                    model_name=model_name,
                    temperature=temperature,
                    json_mode=json_mode,
                )
        finally:
            self.inner = real_inner
            self._restore_batch_state(snapshot)

        max_workers = min(
            len(prompts),
            max(1, int(self.auth.get("max_parallel_agents", 1))),
        )
        parallelism = self.ledger.setdefault(
            "parallelism",
            {
                "max_parallel_agents": int(
                    self.auth.get("max_parallel_agents", 1)
                ),
                "batch_calls": 0,
                "max_batch_size": 0,
                "max_workers_used": 0,
            },
        )
        parallelism["inflight"] = {
            "round": next(iter(rounds)),
            "batch_size": len(prompts),
            "workers": max_workers,
        }
        _write_ledger(self.path, self.ledger)

        def invoke(prompt: str) -> LLMResponse:
            if json_mode is None:
                return real_inner.complete(
                    prompt,
                    model_name=model_name,
                    temperature=temperature,
                )
            return real_inner.complete(
                prompt,
                model_name=model_name,
                temperature=temperature,
                json_mode=json_mode,
            )

        if max_workers == 1:
            responses = [invoke(prompt) for prompt in prompts]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                responses = list(executor.map(invoke, prompts))

        replay = _BatchReplayClient(responses)
        try:
            self.inner = replay
            committed = [
                self.complete(
                    prompt,
                    model_name=model_name,
                    temperature=temperature,
                    json_mode=json_mode,
                )
                for prompt in prompts
            ]
        finally:
            self.inner = real_inner
            self._repair_inner = None
        if replay.index != len(responses):
            raise RuntimeError("APIError: parallel response replay overflow")
        parallelism.pop("inflight", None)
        parallelism["batch_calls"] = int(parallelism.get("batch_calls", 0)) + 1
        parallelism["max_batch_size"] = max(
            int(parallelism.get("max_batch_size", 0)),
            len(prompts),
        )
        parallelism["max_workers_used"] = max(
            int(parallelism.get("max_workers_used", 0)),
            max_workers,
        )
        _write_ledger(self.path, self.ledger)
        return committed

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        """One worker call: the model name and temperature must match the
        host authorization and the call budget must have room; the call then
        runs under the ``message_only_v2`` contract, the only one supported."""
        expected = self.auth["worker_llm"]
        if model_name != expected["model_name"]:
            raise RuntimeError("APIError: worker model_name differs from host authorization")
        if temperature != expected["temperature"]:
            raise RuntimeError("APIError: worker temperature differs from host authorization")
        calls = int(self.ledger["usage"]["model_calls"])
        if calls >= int(self.auth["budgets"]["max_model_calls"]):
            raise RuntimeError("BudgetError: model call budget exhausted")
        if self.worker_contract != "message_only_v2":
            raise RuntimeError(
                f"APIError: unsupported worker contract {self.worker_contract!r}"
            )
        return self._complete_message_only_v2(
            prompt,
            model_name=model_name,
            temperature=temperature,
            json_mode=json_mode,
        )

    def _prepare_submit_barrier(self, submit_round: int) -> None:
        """Freeze every agent's post-communication view before any submit call
        and record the barrier (expected submitters, a digest of the frozen
        views, the answer format) in the ledger.  Raises ``BudgetError`` when
        the remaining call or completion-token budget cannot cover one submit
        per expected submitter."""
        n_agents = int(self.auth["n_agents"])
        if self.auth["information_goal"] == "all_agents":
            expected_submit_ids = list(range(n_agents))
        else:
            expected_submit_ids = [int(self.auth["selected_primary"])]
        remaining_calls = int(self.auth["budgets"]["max_model_calls"]) - int(
            self.ledger["usage"]["model_calls"]
        )
        if remaining_calls < len(expected_submit_ids):
            raise RuntimeError(
                "BudgetError: synchronized submit call budget exhausted before barrier"
            )
        remaining_tokens = int(
            self.auth["budgets"]["max_completion_tokens"]
        ) - int(self.ledger["usage"]["completion_tokens"])
        if remaining_tokens < len(expected_submit_ids):
            raise RuntimeError(
                "BudgetError: synchronized submit token budget exhausted before barrier"
            )

        snapshot_fingerprint: dict[str, Any] = {}
        for agent_id in range(n_agents):
            due_at_barrier: list[dict[str, Any]] = []
            for envelope in self.pending_envelopes:
                if envelope["dst"] != agent_id or envelope["merged"]:
                    continue
                if int(envelope["round_delivered"]) > submit_round:
                    continue
                envelope["merged"] = True
                self.known_sources[agent_id].update(
                    int(value) for value in envelope["source_ids"]
                )
                if int(envelope["round_delivered"]) == submit_round:
                    due_at_barrier.append(envelope)
            public_inbox = [_public_envelope(item) for item in due_at_barrier]
            self.frozen_barrier_states[agent_id] = {
                "previous_output": self.previous_outputs[agent_id],
                "source_ids": sorted(self.known_sources[agent_id]),
            }
            self.frozen_barrier_inboxes[agent_id] = public_inbox
            snapshot_fingerprint[str(agent_id)] = {
                "previous_output_sha256": _value_sha256(
                    self.previous_outputs[agent_id]
                ),
                "known_source_ids": sorted(self.known_sources[agent_id]),
                "delivered_inbox": [
                    _sanitize_message_envelope(item) for item in public_inbox
                ],
            }
        self.ledger["submit_barrier"] = {
            "enabled": True,
            "worker_contract": "message_only_v2",
            "information_goal": str(self.auth["information_goal"]),
            "submit_round": submit_round,
            "expected_agent_ids": expected_submit_ids,
            "observed_agent_ids": [],
            "synchronized": False,
            "final_snapshot_sha256": _value_sha256(snapshot_fingerprint),
            "answer_format": "single_json_value",
            "parser": "json.loads_whole_response",
        }
        _write_ledger(self.path, self.ledger)

    def _complete_message_only_v2(
        self,
        prompt: str,
        *,
        model_name: str,
        temperature: float | None,
        json_mode: bool | None,
    ) -> LLMResponse:
        """One worker call under the message_only_v2 contract.

        The prompt must equal the canonical prompt rebuilt from the runtime
        state (control object, known source ids, previous output, delivered
        inbox, the agent's private prompt); communication happens only before
        the submit round, and every required submitter submits once, at that
        round.  A submit output must parse as exactly one JSON value (one
        repair call is allowed).
        """
        if json_mode is not False:
            raise RuntimeError(
                "APIError: message_only_v2 workers must be called with json_mode=False"
            )
        contract = _prompt_header_str(prompt, "PYTHON_WORKER_CONTRACT")
        if contract != "message_only_v2":
            raise RuntimeError(
                "DataFlowError: worker prompt declares a different worker contract"
            )
        n_agents = int(self.auth["n_agents"])
        max_rounds = int(self.auth["max_rounds"])
        agent_id = _prompt_header_int(prompt, "PYTHON_AGENT_ID")
        round_idx = _prompt_header_int(prompt, "PYTHON_ROUND")
        submit_round = _prompt_header_int(prompt, "PYTHON_SUBMIT_ROUND")
        if submit_round < 0 or submit_round >= max_rounds:
            raise RuntimeError(
                "DataFlowError: submit barrier is outside the round budget"
            )
        if self.declared_submit_round is None:
            self.declared_submit_round = submit_round
        elif self.declared_submit_round != submit_round:
            raise RuntimeError(
                "DataFlowError: inconsistent PYTHON_SUBMIT_ROUND declaration"
            )
        if not 0 <= agent_id < n_agents:
            raise RuntimeError("DataFlowError: worker agent id outside the agent range")
        if not 0 <= round_idx < max_rounds:
            raise RuntimeError("DataFlowError: worker round outside the round budget")
        if round_idx < self.max_round_seen:
            raise RuntimeError(
                "DataFlowError: worker call violates monotonic round ordering"
            )
        if (round_idx, agent_id) in self.called_pairs:
            raise RuntimeError(
                "DataFlowError: an agent was called more than once in the same round"
            )
        if agent_id in self.submitted_rounds:
            raise RuntimeError("DataFlowError: a submitted agent was called again")

        control = _prompt_json_line(prompt, "PYTHON_CONTROL_JSON")
        known_ids = _prompt_json_line(prompt, "KNOWN_SOURCE_IDS_JSON")
        previous_output = _prompt_json_line(prompt, "PREVIOUS_OUTPUT_JSON")
        delivered_inbox = _prompt_json_line(prompt, "DELIVERED_INBOX_JSON")
        if (
            not isinstance(control, dict)
            or not isinstance(known_ids, list)
            or not isinstance(previous_output, str)
            or not isinstance(delivered_inbox, list)
        ):
            raise RuntimeError(
                "DataFlowError: malformed control/state prompt metadata"
            )
        mode = str(control.get("mode") or "")
        raw_recipients = control.get("recipients")
        # A communication round's control object may carry a work_instruction
        # (a per-round instruction placed in the prompt's
        # PHASE_WORK_INSTRUCTION section): 1..800 characters, forbidden at
        # the submit round.  The team program renders the same section
        # (exp_graph.mas.python_code); the canonical-prompt check below
        # requires an exact match.
        allowed_keys = {"mode", "recipients"}
        work_instruction = str(control.get("work_instruction") or "")
        if "work_instruction" in control:
            allowed_keys = {"mode", "recipients", "work_instruction"}
            if not isinstance(control["work_instruction"], str):
                raise RuntimeError(
                    "DataFlowError: work_instruction must be a string"
                )
            if not work_instruction or len(work_instruction) > 800:
                raise RuntimeError(
                    "DataFlowError: work_instruction must be 1..800 characters"
                )
        if set(control) != allowed_keys or not isinstance(
            raw_recipients, list
        ):
            raise RuntimeError("DataFlowError: malformed planner control object")
        if mode == "submit" and work_instruction:
            raise RuntimeError(
                "DataFlowError: work_instruction is forbidden at the submit barrier"
            )
        if mode not in _MESSAGE_ONLY_V2_MODES:
            raise RuntimeError(
                "DataFlowError: planner control mode is not send/reflect/submit"
            )
        recipients = [int(value) for value in raw_recipients]
        if mode != "send" and recipients:
            raise RuntimeError("DataFlowError: recipients require send mode")
        if len(set(recipients)) != len(recipients):
            raise RuntimeError("DataFlowError: duplicate recipients")
        for recipient in recipients:
            if recipient < 0 or recipient >= n_agents or recipient == agent_id:
                raise RuntimeError(
                    "DataFlowError: recipient outside the agent range"
                )
        if mode == "submit":
            if round_idx != submit_round:
                raise RuntimeError(
                    "DataFlowError: submit is allowed only at the synchronized barrier"
                )
            expected_submit_ids = (
                list(range(n_agents))
                if self.auth["information_goal"] == "all_agents"
                else [int(self.auth["selected_primary"])]
            )
            if agent_id not in expected_submit_ids:
                raise RuntimeError(
                    "DataFlowError: agent is not a required barrier submitter"
                )
            if not self.frozen_barrier_states:
                self._prepare_submit_barrier(submit_round)
            expected_state = self.frozen_barrier_states[agent_id]
            expected_inbox_items = self.frozen_barrier_inboxes[agent_id]
        else:
            if round_idx >= submit_round:
                raise RuntimeError(
                    "DataFlowError: communication is forbidden at the submit barrier"
                )
            if mode == "send" and self.total_messages + len(recipients) > int(
                self.auth["budgets"]["max_messages"]
            ):
                raise RuntimeError("BudgetError: message budget exhausted")
            due_now: list[dict[str, Any]] = []
            for envelope in self.pending_envelopes:
                if envelope["dst"] != agent_id or envelope["merged"]:
                    continue
                if int(envelope["round_delivered"]) > round_idx:
                    continue
                envelope["merged"] = True
                self.known_sources[agent_id].update(
                    int(value) for value in envelope["source_ids"]
                )
                if int(envelope["round_delivered"]) == round_idx:
                    due_now.append(envelope)
            expected_state = {
                "previous_output": self.previous_outputs[agent_id],
                "source_ids": sorted(self.known_sources[agent_id]),
            }
            expected_inbox_items = [_public_envelope(item) for item in due_now]

        expected_inbox = sorted(
            json.dumps(_public_envelope(item), sort_keys=True)
            for item in expected_inbox_items
        )
        actual_inbox = sorted(
            json.dumps(_public_envelope(item), sort_keys=True)
            for item in delivered_inbox
        )
        if actual_inbox != expected_inbox:
            raise RuntimeError(
                "DataFlowError: worker inbox does not match runtime deliveries"
            )
        if list(known_ids) != list(expected_state["source_ids"]):
            raise RuntimeError(
                "DataFlowError: prompt source ids differ from runtime provenance"
            )
        if previous_output != expected_state["previous_output"]:
            raise RuntimeError(
                "DataFlowError: previous output differs from the frozen runtime state"
            )
        instruction = (
            MESSAGE_ONLY_V2_SUBMIT_INSTRUCTION
            if mode == "submit"
            else MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION
        )
        if mode == "submit":
            private_prompt = str(
                self.auth["agent_submit_prompts"][str(agent_id)]
            )
            prompt_header = "SUBMIT_PROMPT:\n"
        else:
            private_prompt = str(
                self.auth["agent_communication_prompts"][str(agent_id)]
            )
            prompt_header = "COMMUNICATION_PROMPT:\n"
        expected_prompt = (
            "PYTHON_WORKER_CONTRACT:message_only_v2\n"
            f"PYTHON_AGENT_ID:{agent_id}\n"
            f"PYTHON_ROUND:{round_idx}\n"
            f"PYTHON_SUBMIT_ROUND:{submit_round}\n"
            "PYTHON_CONTROL_JSON:"
            + json.dumps(control, sort_keys=True)
            + "\nKNOWN_SOURCE_IDS_JSON:"
            + json.dumps(known_ids)
            + "\nPREVIOUS_OUTPUT_JSON:"
            + json.dumps(previous_output)
            + "\nDELIVERED_INBOX_JSON:"
            + json.dumps(delivered_inbox, sort_keys=True)
            + "\n"
            + prompt_header
            + private_prompt
            + "\n"
            + (
                "PHASE_WORK_INSTRUCTION:\n" + work_instruction + "\n"
                if work_instruction and mode != "submit"
                else ""
            )
            + instruction
        )
        if prompt != expected_prompt:
            raise RuntimeError(
                "DataFlowError: worker prompt differs from canonical message_only_v2 form"
            )
        _validate_canaries(prompt, agent_id, self.auth.get("agent_canaries", {}))
        response = self.inner.complete(
            prompt,
            model_name=model_name,
            temperature=temperature,
            json_mode=False,
        )
        usage = response.usage
        worker_output = str(response.text).strip()
        answer_value: Any = None
        answer_format_valid = True
        answer_repaired = False
        rejected_output_sha256: str | None = None
        if mode == "submit":
            try:
                answer_value = json.loads(
                    worker_output,
                    parse_constant=_reject_nonfinite_json,
                )
            except (json.JSONDecodeError, ValueError):
                answer_format_valid = False
            if not answer_format_valid:
                # One-shot repair: send the identical prompt plus the repair
                # notice and parse the repair output with the same strict
                # parser.  A second failure raises the AnswerFormatError
                # below, so a submit that stays malformed still fails the run.
                # The repair call's usage is added to this call's usage, so
                # the budgets count it.
                rejected_output_sha256 = _value_sha256(worker_output)
                repair_client = getattr(self, "_repair_inner", None) or self.inner
                repair_response = repair_client.complete(
                    prompt + _ANSWER_FORMAT_REPAIR_NOTICE,
                    model_name=model_name,
                    temperature=temperature,
                    json_mode=False,
                )
                usage = combine_usage([usage, repair_response.usage])
                repair_output = str(repair_response.text).strip()
                try:
                    answer_value = json.loads(
                        repair_output,
                        parse_constant=_reject_nonfinite_json,
                    )
                except (json.JSONDecodeError, ValueError):
                    pass
                else:
                    answer_format_valid = True
                    answer_repaired = True
                    worker_output = repair_output
                    response = LLMResponse(text=repair_response.text, usage=usage)

        self.called_pairs.add((round_idx, agent_id))
        self.max_round_seen = round_idx
        if mode == "send":
            for recipient in recipients:
                self.pending_envelopes.append(
                    {
                        "round_sent": round_idx,
                        "round_delivered": round_idx + 1,
                        "src": agent_id,
                        "dst": recipient,
                        "source_ids": sorted(self.known_sources[agent_id]),
                        "body": worker_output,
                        "merged": False,
                    }
                )
            self.total_messages += len(recipients)
        if mode in {"send", "reflect"}:
            self.previous_outputs[agent_id] = worker_output

        self.ledger["usage"]["model_calls"] += int(usage.model_calls)
        self.ledger["usage"]["prompt_tokens"] += int(usage.prompt_tokens)
        self.ledger["usage"]["completion_tokens"] += int(usage.completion_tokens)
        call_record = {
            "agent_id": agent_id,
            "round": round_idx,
            "submit_round": submit_round,
            "mode": mode,
            "recipients": recipients,
            "known_source_ids": list(expected_state["source_ids"]),
            "delivered_inbox": [
                _sanitize_message_envelope(item) for item in delivered_inbox
            ],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "worker_output_sha256": _value_sha256(worker_output),
            "inbox_bytes": len(json.dumps(delivered_inbox)),
            "answer_format_valid": answer_format_valid,
            "answer_repaired": answer_repaired,
            "rejected_output_sha256": rejected_output_sha256,
            "model_calls": int(usage.model_calls),
            "prompt_tokens": int(usage.prompt_tokens),
            "completion_tokens": int(usage.completion_tokens),
        }
        submit_budget_error = False
        if mode == "submit" and answer_format_valid:
            call_record["answer_sha256"] = _value_sha256(answer_value)
            barrier = self.ledger["submit_barrier"]
            candidate_count = len(self.pending_barrier_submissions) + 1
            remaining_submitters = len(barrier["expected_agent_ids"]) - candidate_count
            completion_remaining = int(
                self.auth["budgets"]["max_completion_tokens"]
            ) - int(self.ledger["usage"]["completion_tokens"])
            if completion_remaining < remaining_submitters:
                submit_budget_error = True
            else:
                self.pending_barrier_submissions[agent_id] = round_idx
                if sorted(self.pending_barrier_submissions) == barrier[
                    "expected_agent_ids"
                ]:
                    self.submitted_rounds = dict(
                        self.pending_barrier_submissions
                    )
                    barrier["observed_agent_ids"] = sorted(
                        self.submitted_rounds
                    )
                    barrier["synchronized"] = (
                        len(set(self.submitted_rounds.values())) == 1
                    )
        self.ledger["calls"].append(call_record)
        _write_ledger(self.path, self.ledger)
        if self.ledger["usage"]["completion_tokens"] > int(
            self.auth["budgets"]["max_completion_tokens"]
        ):
            raise RuntimeError("BudgetError: completion token budget exceeded")
        if submit_budget_error:
            raise RuntimeError(
                "BudgetError: submit response left insufficient token budget "
                "for the complete barrier"
            )
        if mode == "submit" and not answer_format_valid:
            raise RuntimeError(
                "AnswerFormatError: submit output must be exactly one valid JSON value"
            )
        return response


def _prompt_header_str(prompt: str, header: str) -> str:
    marker = f"{header}:"
    if marker not in prompt:
        raise RuntimeError(f"DataFlowError: worker prompt lacks {header}")
    return prompt.split(marker, 1)[1].splitlines()[0].strip()


def _public_envelope(item: Any) -> dict[str, Any]:
    """Project one envelope onto the exact fields a recipient may see."""
    if not isinstance(item, dict):
        raise RuntimeError("DataFlowError: inbox entry is not an object")
    try:
        return {
            "round_sent": int(item["round_sent"]),
            "round_delivered": int(item["round_delivered"]),
            "src": int(item["src"]),
            "dst": int(item["dst"]),
            "source_ids": sorted(int(value) for value in item["source_ids"]),
            "body": str(item["body"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DataFlowError: malformed inbox envelope") from exc


def _prompt_header_int(prompt: str, header: str) -> int:
    marker = f"{header}:"
    if marker not in prompt:
        raise RuntimeError(f"DataFlowError: worker prompt lacks {header}")
    raw = prompt.split(marker, 1)[1].splitlines()[0].strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"DataFlowError: invalid {header}") from exc


def _prompt_json_line(prompt: str, header: str) -> Any:
    marker = f"{header}:"
    if marker not in prompt:
        raise RuntimeError(f"DataFlowError: worker prompt lacks {header}")
    raw = prompt.split(marker, 1)[1].splitlines()[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DataFlowError: invalid {header}") from exc


def _sanitize_message_envelope(item: Any) -> dict[str, Any]:
    """Ledger form of an envelope: the body is replaced by its sha256."""
    if not isinstance(item, dict):
        raise RuntimeError("DataFlowError: inbox entry is not an object")
    try:
        return {
            "round_sent": int(item["round_sent"]),
            "round_delivered": int(item["round_delivered"]),
            "src": int(item["src"]),
            "dst": int(item["dst"]),
            "source_ids": sorted(int(value) for value in item.get("source_ids", [])),
            "body_sha256": _value_sha256(item.get("body")),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DataFlowError: malformed inbox envelope") from exc


def _value_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _validate_canaries(prompt: str, agent_id: int, canaries: dict[str, str]) -> None:
    """Leak check: the prompt carries this agent's canary and no other's."""
    if not canaries:
        return
    expected = canaries.get(str(agent_id))
    if expected is None or expected not in prompt:
        raise RuntimeError("DataFlowError: current agent canary is missing")
    for other_id, canary in canaries.items():
        if int(other_id) != agent_id and canary in prompt:
            raise RuntimeError("DataFlowError: another agent's local prompt leaked")


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.write_text(json.dumps(ledger, sort_keys=True), encoding="utf-8")


def main() -> int:
    """Child entry point: start the ledger, install the metered factory and
    run ``--program`` under the ``--auth`` authorization."""
    # Opt-in count-table JSON repair (count_table_json.REPAIR_ENV, set by the
    # Count-Frequency task); off by default.
    install_if_requested()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--program", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    auth = json.loads(Path(args.auth).read_text(encoding="utf-8"))
    ledger_path = Path(args.ledger)
    ledger: dict[str, Any] = {
        "factory_calls": 0,
        "worker_contract": str(auth.get("worker_contract") or ""),
        "calls": [],
        "usage": {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        "parallelism": {
            "max_parallel_agents": int(auth.get("max_parallel_agents", 1)),
            "batch_calls": 0,
            "max_batch_size": 0,
            "max_workers_used": 0,
        },
        "errors": [],
    }
    _write_ledger(ledger_path, ledger)

    def secure_factory(
        provider: str,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        thinking_enabled: bool | None = None,
        timeout_s: float | None = None,
    ) -> _MeteredClient:
        """The team program's ``create_llm_client`` (the static validator
        allows no other client import): it may be called once, with exactly
        the host-authorized provider, base URL and API-key variable."""
        del thinking_enabled, timeout_s
        expected = auth["worker_llm"]
        ledger["factory_calls"] += 1
        if ledger["factory_calls"] != 1:
            raise RuntimeError("APIError: create_llm_client called more than once")
        if provider != expected["provider"]:
            raise RuntimeError("APIError: provider differs from host authorization")
        if base_url != expected.get("base_url"):
            raise RuntimeError("APIError: base_url differs from host authorization")
        if api_key_env != expected.get("api_key_env"):
            raise RuntimeError("APIError: api_key_env differs from host authorization")
        inner = (
            _PythonSmokeClient()
            if provider == "fake"
            else _real_factory(
                provider,
                base_url=base_url,
                api_key_env=api_key_env,
                timeout_s=expected.get("request_timeout"),
            )
        )
        _write_ledger(ledger_path, ledger)
        return _MeteredClient(inner, auth, ledger, ledger_path)

    # Install the metered factory, cap stdout, then run the team program as
    # __main__; an exception is recorded in the ledger and re-raised.
    factory_module.create_llm_client = secure_factory
    sys.argv = [args.program]
    sys.stdout = _LimitedStdout(sys.stdout, int(auth["max_output_bytes"]))
    try:
        runpy.run_path(args.program, run_name="__main__")
    except BaseException as exc:
        ledger["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)[:1000]}
        )
        _write_ledger(ledger_path, ledger)
        raise
    _write_ledger(ledger_path, ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
