"""Genome-mode minting: splicing, reply parsing, re-mint, planner-call caps.

Offline (fake provider / stub planners only).  Pins:

* ``genome_bounds`` / ``genome_region`` / ``splice_genome`` round-trip host
  programs byte for byte;
* a full-file reply with scaffold drift keeps the host bytes, splices into a
  valid program and simulates like the reply's own program;
* ``check_genome`` / ``parse_hypothesis`` / ``parse_genome_reply`` edge cases;
* ``mint_python_challenger``: success, ONE re-mint with the exact diagnostic,
  ``MintFailed`` after two failures, the per-call deadline and output cap
  (explicit, environment, defaults), ``hypothesis_out``;
* ``send`` with empty recipients passes the validator AND the real bootstrap
  runtime (dry run) - a paid compute-only call, nothing delivered;
* the validator's empty-source wording.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pytest

import queenbee.program.execute as ex
import queenbee.program.genome as gn
import queenbee.program.mint as mt
from exp_graph.mas.python_code import (
    DEFAULT_MESSAGE_ONLY_V2_PROGRAM,
    validate_python_source,
)
from exp_graph.mas.python_code_generation import _dry_run_canaries, _dry_run_payload
from exp_graph.mas.python_code_runner import (
    CodeProcessRunner,
    PythonExecutionLimits,
)

_MESH = '{"kind": "broadcast_last", "rounds": 1},\n    {"kind": "mesh", "rounds": 1},'
_BROADCAST = '{"kind": "broadcast_last", "rounds": 1},'


def _seed() -> str:
    return ex.seed_python_source("message_only_v2", base="sfs_phase")[2]


def _env(n_agents: int = 5):
    request = mt.build_planner_request(
        n_agents=n_agents, goal="all_agents", worker_contract="message_only_v2"
    )
    runtime = mt.build_mint_runtime(
        llm_provider="fake", worker_model="fake", goal="all_agents",
        worker_contract="message_only_v2", max_parallel_agents=1,
        request_timeout=30.0, n_agents=n_agents,
    )
    return request, runtime


def _runner(runtime) -> CodeProcessRunner:
    return CodeProcessRunner(
        PythonExecutionLimits(timeout_seconds=runtime.python_execution_timeout)
    )


def _variant_genome(seed: str) -> str:
    return gn.genome_region(seed).replace(_BROADCAST, _MESH, 1)


# --------------------------------------------------------------------------- #
# splicing
# --------------------------------------------------------------------------- #


def _hosts() -> dict[str, str]:
    seed = _seed()
    return {
        "seed": seed,
        "contract_default": DEFAULT_MESSAGE_ONLY_V2_PROGRAM.strip() + "\n",
        "constant_above_phases": gn.splice_genome(
            seed, "HUB = 0\n\n\n" + gn.genome_region(seed)
        ),
    }


@pytest.mark.parametrize("name", sorted(_hosts()))
def test_splice_of_own_genome_is_byte_identical(name) -> None:
    host = _hosts()[name]
    region = gn.genome_region(host)
    assert region is not None
    assert "def main" not in region
    assert "MESSAGE_INSTRUCTION" not in region
    assert gn.splice_genome(host, region) == host
    # Whitespace noise around the genome is normalized away.
    assert gn.splice_genome(host, "\n\n" + region.strip("\n") + "\n\n\n\n") == host


def test_phase_seed_genome_starts_at_phases_and_ends_before_main() -> None:
    seed = _seed()
    region = gn.genome_region(seed)
    assert region.startswith("PHASES = [")
    assert region.rstrip().endswith('return {"mode": "reflect", "recipients": []}')
    _start, end = gn.genome_bounds(seed)
    assert seed[end:].startswith("def main(")


def test_splice_rejects_empty_genome_and_regionless_incumbent() -> None:
    with pytest.raises(ValueError):
        gn.splice_genome(_seed(), "   \n")
    with pytest.raises(ValueError):
        gn.splice_genome("x = 1\n", "PHASES = []\n")
    assert gn.genome_bounds(None) is None
    assert gn.genome_region("x = 1\n") is None


def _drifted_full_file(seed: str) -> str:
    """The mesh variant typed as a full file whose SUBMIT_INSTRUCTION drifted."""

    full = gn.splice_genome(seed, _variant_genome(seed))
    drifted = full.replace(
        "Return exactly one valid JSON value and nothing else.",
        "Return exactly one JSON value and nothing else.",
        1,
    )
    assert drifted != full
    return drifted


def test_full_file_reply_with_scaffold_drift_keeps_host_bytes() -> None:
    seed = _seed()
    full = _drifted_full_file(seed)
    reply = 'HYPOTHESIS: {"mechanism": "add a mesh round"}\n```python\n' + full + "```\n"
    _hyp, genome, notes = mt.parse_genome_reply(reply, incumbent_source=seed)
    assert genome is not None, notes
    assert "full-file reply: took its genome region" in notes
    assert any("DISCARDED" in n and "SUBMIT_INSTRUCTION" in n for n in notes)
    assert gn.check_genome(genome, incumbent_source=seed) == []
    spliced = gn.splice_genome(seed, genome)
    request, runtime = _env()
    runner = _runner(runtime)
    assert mt._validate_spliced(
        spliced, incumbent_source=seed, worker_contract="message_only_v2",
        runner=runner, request=request, runtime=runtime,
    ) is None
    # Same schedule as the reply's own program; the host bytes are the seed's.
    fp_spliced = ex.simulate_program_behavior(spliced, n_agents=5, max_rounds=64)
    assert fp_spliced is not None
    assert fp_spliced == ex.simulate_program_behavior(full, n_agents=5, max_rounds=64)
    bounds, seed_bounds = gn.genome_bounds(spliced), gn.genome_bounds(seed)
    assert spliced[: bounds[0]] == seed[: seed_bounds[0]]
    assert spliced[bounds[1]:] == seed[seed_bounds[1]:]
    # The drifted full file itself breaks the canonical worker prompt.
    dry_full = runner.run(
        full, _dry_run_payload(request, runtime),
        agent_canaries=_dry_run_canaries(request.n_agents),
    )
    assert not dry_full.runtime_success
    assert "canonical" in json.dumps(dry_full.failure)


def test_truncated_full_file_reply_takes_everything_from_the_first_genome_line() -> None:
    seed = _seed()
    full = gn.splice_genome(seed, _variant_genome(seed))
    truncated = full[: full.find("def main(")]
    reply = "HYPOTHESIS: x\n```python\n" + truncated + "```\n"
    _hyp, genome, notes = mt.parse_genome_reply(reply, incumbent_source=seed)
    assert any("without a usable main()" in n for n in notes)
    assert genome == _variant_genome(seed).strip("\n")


# --------------------------------------------------------------------------- #
# genome checks / reply parsing
# --------------------------------------------------------------------------- #


def test_check_genome_reports_exact_problems() -> None:
    seed = _seed()
    good = gn.genome_region(seed)
    assert gn.check_genome(good, incumbent_source=seed) == []
    bad_syntax = good.replace("PHASES = [", "PHASES = [(", 1)
    problems = gn.check_genome(bad_syntax, incumbent_source=seed)
    assert problems and problems[0].startswith("SyntaxError at genome line")
    rebind = good + '\n\nSUBMIT_INSTRUCTION = "x"\n'
    assert any("SUBMIT_INSTRUCTION" in p for p in gn.check_genome(rebind, incumbent_source=seed))
    imported = "import os\n" + good
    assert any("imports are host-owned" in p for p in gn.check_genome(imported))
    main_def = good + "\n\ndef main():\n    return 1\n"
    assert any("'main'" in p for p in gn.check_genome(main_def))
    call = good + "\n\nmain()\n"
    assert any("top-level expression" in p for p in gn.check_genome(call))
    missing = good.replace("def plan_submit_round", "def plan_submit_round_x")
    assert any("plan_submit_round()" in p for p in gn.check_genome(missing))
    assert gn.check_genome("") == ["the genome is empty"]


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            'HYPOTHESIS: {"target_units": ["II-13"], "failure_class": '
            '"divergent", "mechanism": "vote", "predicted_effect": "agree"}\n',
            {"target_units": ["II-13"], "failure_class": "divergent",
             "mechanism": "vote", "predicted_effect": "agree"},
        ),
        (
            "**HYPOTHESIS:** {'target_units': [], 'failure_class': 'none',\n"
            " 'mechanism': 'cheaper relay', 'predicted_effect': 'cost down'}\n",
            {"target_units": [], "failure_class": "none",
             "mechanism": "cheaper relay", "predicted_effect": "cost down"},
        ),
        ("HYPOTHESIS: make agents vote\n", {"raw": "make agents vote"}),
        (
            "my hypothesis: prose first\n"
            'HYPOTHESIS: {"mechanism": "the header line wins"}\n',
            {"mechanism": "the header line wins"},
        ),
    ],
)
def test_parse_hypothesis_variants(text, expected) -> None:
    assert mt.parse_hypothesis(text) == expected


def test_parse_hypothesis_absent_and_garbage() -> None:
    assert mt.parse_hypothesis("no header here") is None
    assert "raw" in mt.parse_hypothesis('HYPOTHESIS: {"a": [1,}')
    clipped = mt.sanitize_hypothesis({"mechanism": "x" * 1000, "evil": "y"})
    assert clipped == {"mechanism": "x" * 300}
    assert mt.sanitize_hypothesis("not a dict") is None


def test_parse_genome_reply_genome_only_block() -> None:
    seed = _seed()
    genome = gn.genome_region(seed)
    reply = (
        'HYPOTHESIS: {"target_units": ["I-01"], "failure_class": "divergent", '
        '"mechanism": "m", "predicted_effect": "p"}\n\n'
        "Some prose.\n```python\n" + genome + "```\n"
    )
    hyp, got, notes = mt.parse_genome_reply(reply, incumbent_source=seed)
    assert hyp["target_units"] == ["I-01"]
    assert got == genome.strip("\n")
    assert notes == []


def test_parse_genome_reply_unclosed_fence_and_empty() -> None:
    seed = _seed()
    genome = gn.genome_region(seed)
    _hyp, got, notes = mt.parse_genome_reply(
        "HYPOTHESIS: x\n```python\n" + genome, incumbent_source=seed
    )
    assert got is not None and got.startswith("PHASES = [")
    assert any("unclosed" in n for n in notes)
    assert mt.parse_genome_reply("", incumbent_source=seed)[1] is None
    assert mt.parse_genome_reply("```python\nx = 1\n```")[1] is None


def test_parse_genome_reply_drops_a_hypothesis_line_inside_the_block() -> None:
    seed = _seed()
    genome = gn.genome_region(seed)
    reply = '```python\nHYPOTHESIS: {"mechanism": "inside"}\n' + genome + "```\n"
    hyp, got, notes = mt.parse_genome_reply(reply, incumbent_source=seed)
    assert hyp == {"mechanism": "inside"}
    assert got == genome.strip("\n")
    assert "dropped a HYPOTHESIS line from inside the code block" in notes


# --------------------------------------------------------------------------- #
# genome-mode mint
# --------------------------------------------------------------------------- #


class _Usage:
    def __init__(self, prompt_tokens: int = 10, completion_tokens: int = 20) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = _Usage()


class _StubPlanner:
    def __init__(self, replies: list[str], delay_s: float = 0.0) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.delay_s = delay_s

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        self.calls.append({"prompt": prompt, "json_mode": json_mode,
                           "temperature": temperature})
        if self.delay_s:
            time.sleep(self.delay_s)
        return _Resp(self.replies.pop(0))


def _reply(genome: str, mechanism: str = "add a mesh round") -> str:
    return (
        'HYPOTHESIS: {"target_units": ["I-01"], "failure_class": "divergent", '
        f'"mechanism": "{mechanism}", "predicted_effect": "agree"}}\n'
        "```python\n" + genome + "\n```\n"
    )


def _mint(tmp_path: Path, planner: Any, **kw: Any) -> str:
    request, runtime = _env()
    kw.setdefault("incumbent_source", _seed())
    return mt.mint_python_challenger(
        planner_client=planner, planner_model="planner-x", prompt="PROMPT",
        request=request, runtime=runtime, workdir=tmp_path / "mint",
        genome_only=True, **kw,
    )


def test_genome_mint_success_single_call(tmp_path) -> None:
    seed = _seed()
    genome = _variant_genome(seed)
    planner = _StubPlanner([_reply(genome)])
    log: list[dict[str, Any]] = []
    hyp: dict[str, Any] = {}
    source = _mint(tmp_path, planner, usage_log=log, hypothesis_out=hyp,
                   prompt_dump_dir=tmp_path / "mint")
    assert source == gn.splice_genome(seed, genome)
    assert [e["call"] for e in log] == ["mint"]
    assert log[0]["prompt_tokens"] == 10 and log[0]["completion_tokens"] == 20
    assert hyp["mechanism"] == "add a mesh round"
    assert all(c["json_mode"] is False and c["temperature"] == 0.0 for c in planner.calls)
    diag = json.loads((tmp_path / "mint" / "mint_diag.json").read_text())
    assert diag["status"] == "ok" and diag["interface"] == "genome"
    assert (tmp_path / "mint" / "mint_prompt.txt").read_text() == "PROMPT"
    assert (tmp_path / "mint" / "mint_reply_00.txt").read_text() == _reply(genome)
    assert (tmp_path / "mint" / "mint_genome_00.py").read_text() == genome.strip("\n") + "\n"
    assert (tmp_path / "mint" / "mint_attempt_00.py").read_text() == source


def test_genome_mint_remints_once_with_exact_diagnostic(tmp_path) -> None:
    seed = _seed()
    broken = _variant_genome(seed) + "\n\nSUBMIT_INSTRUCTION = 'x'\n"
    fixed = _variant_genome(seed)
    planner = _StubPlanner([_reply(broken, "first"), _reply(fixed, "second")])
    log: list[dict[str, Any]] = []
    hyp: dict[str, Any] = {}
    source = _mint(tmp_path, planner, usage_log=log, hypothesis_out=hyp,
                   prompt_dump_dir=tmp_path / "mint")
    assert source == gn.splice_genome(seed, fixed)
    # Two full mint calls, never a repair call.
    assert [e["call"] for e in log] == ["mint", "remint"]
    remint_prompt = planner.calls[1]["prompt"]
    assert remint_prompt.startswith("PROMPT")
    assert "REJECTED BEFORE ANY EXECUTION" in remint_prompt
    assert "SUBMIT_INSTRUCTION" in remint_prompt and "host-owned" in remint_prompt
    assert remint_prompt.endswith(mt._GENOME_OUTPUT_CONTRACT)
    assert (tmp_path / "mint" / "remint_prompt.txt").read_text() == remint_prompt
    assert hyp["mechanism"] == "second"


def test_genome_mint_validator_failure_is_mapped_to_genome_lines(tmp_path) -> None:
    seed = _seed()
    # while-loop: the fail-closed validator rejects it inside the genome.
    bad = _variant_genome(seed).replace(
        "def phase_relay(local_round, agent_id, n_agents, selected_primary,\n"
        "                known_source_count, inbox_count):\n",
        "def phase_relay(local_round, agent_id, n_agents, selected_primary,\n"
        "                known_source_count, inbox_count):\n"
        "    while False:\n        return {}\n",
        1,
    )
    planner = _StubPlanner([_reply(bad), _reply(bad)])
    with pytest.raises(mt.MintFailed) as info:
        _mint(tmp_path, planner)
    exc = info.value
    assert exc.kind == "invalid"
    assert "validator" in exc.reason and "while loops are forbidden" in exc.reason
    assert "genome line" in exc.reason
    assert exc.hypothesis and exc.hypothesis["target_units"] == ["I-01"]
    assert len(exc.attempts) == 2
    assert isinstance(exc, RuntimeError)
    diag = json.loads((tmp_path / "mint" / "mint_diag.json").read_text())
    assert diag["status"] == "mint_failed"


def test_genome_mint_empty_replies_fail_after_one_remint(tmp_path) -> None:
    planner = _StubPlanner(["", "no code at all"])
    with pytest.raises(mt.MintFailed) as info:
        _mint(tmp_path, planner)
    assert info.value.reason.startswith("extract:")
    assert info.value.hypothesis is None
    assert len(planner.calls) == 2


def test_genome_mint_dry_run_failure_is_reported(tmp_path) -> None:
    seed = _seed()
    # A recipient outside the agent range: validator-clean, dies at dry run.
    bad = _variant_genome(seed).replace(
        'return {"mode": "send", "recipients": [agent_id + 1]}',
        'return {"mode": "send", "recipients": [agent_id + 99]}',
        1,
    )
    planner = _StubPlanner([_reply(bad), _reply(bad)])
    with pytest.raises(mt.MintFailed) as info:
        _mint(tmp_path, planner)
    assert info.value.reason.startswith("dry_run:")
    assert "recipient" in info.value.reason


def test_genome_mint_deadline_raises_mint_failed_without_retry(tmp_path) -> None:
    planner = _StubPlanner([_reply(_variant_genome(_seed()))] * 3, delay_s=2.0)
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    with pytest.raises(mt.MintFailed) as info:
        _mint(tmp_path, planner, usage_log=log, deadline_s=0.3)
    assert time.monotonic() - started < 1.5
    assert info.value.kind == "planner_deadline"
    assert len(log) == 1 and "PlannerDeadlineExceeded" in log[0]["error"]
    assert log[0]["prompt_tokens"] is None


def test_genome_mint_requires_incumbent_with_region(tmp_path) -> None:
    with pytest.raises(ValueError):
        _mint(tmp_path, _StubPlanner([]), incumbent_source=None)
    with pytest.raises(ValueError):
        _mint(tmp_path, _StubPlanner([]), incumbent_source="x = 1\n")


def test_only_genome_mode_minting_is_available(tmp_path) -> None:
    request, runtime = _env()
    with pytest.raises(ValueError, match="genome"):
        mt.mint_python_challenger(
            planner_client=_StubPlanner([]), planner_model="m", prompt="P",
            request=request, runtime=runtime, workdir=tmp_path,
            genome_only=False, incumbent_source=_seed(),
        )


class _CapRecorder:
    """Innermost client stand-in exposing ``_max_completion_tokens``."""

    def __init__(self, reply: str, initial: int = 7777) -> None:
        self._max_completion_tokens = initial
        self.seen: list[Any] = []
        self.reply = reply

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        self.seen.append(self._max_completion_tokens)
        return _Resp(self.reply)


class _Wrapper:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        return self._inner.complete(prompt, model_name, temperature, json_mode=json_mode)


def test_per_call_output_cap_is_applied_and_restored(tmp_path) -> None:
    inner = _CapRecorder(_reply(_variant_genome(_seed())))
    client = _Wrapper(_Wrapper(inner))
    _mint(tmp_path, client, max_completion_tokens=64000, deadline_s=30)
    assert inner.seen == [64000]
    assert inner._max_completion_tokens == 7777
    assert mt._CAP_STATE == {}


def test_planner_caps_default_to_the_reported_settings(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("QB_PLANNER_MAX_COMPLETION_TOKENS", raising=False)
    monkeypatch.delenv("QB_PLANNER_DEADLINE_S", raising=False)
    assert mt.DEFAULT_PLANNER_MAX_COMPLETION_TOKENS == 96_000
    assert mt.DEFAULT_PLANNER_DEADLINE_S == 3600.0
    seen: list[dict[str, Any]] = []
    real = mt._planner_call

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append({k: kwargs[k] for k in ("max_completion_tokens", "deadline_s")})
        return real(*args, **kwargs)

    monkeypatch.setattr(mt, "_planner_call", spy)
    inner = _CapRecorder(_reply(_variant_genome(_seed())))
    _mint(tmp_path / "default", _Wrapper(inner))
    assert inner.seen == [96_000] and inner._max_completion_tokens == 7777
    assert seen == [{"max_completion_tokens": 96_000, "deadline_s": 3600.0}]
    # An explicit 0 disables either knob.
    inner0 = _CapRecorder(_reply(_variant_genome(_seed())))
    _mint(tmp_path / "off", _Wrapper(inner0), max_completion_tokens=0, deadline_s=0)
    assert inner0.seen == [7777]
    assert seen[-1] == {"max_completion_tokens": 0, "deadline_s": 0}


def test_planner_env_knobs_apply_only_when_kwargs_absent(tmp_path, monkeypatch) -> None:
    genome = _variant_genome(_seed())
    monkeypatch.setenv("QB_PLANNER_MAX_COMPLETION_TOKENS", "64000")
    monkeypatch.setenv("QB_PLANNER_DEADLINE_S", "0.3")
    inner = _CapRecorder(_reply(genome))
    _mint(tmp_path / "env", _Wrapper(inner))
    assert inner.seen == [64000]
    # The env deadline is live: a slow planner is cut off.
    with pytest.raises(mt.MintFailed) as info:
        _mint(tmp_path / "slow", _StubPlanner([_reply(genome)], delay_s=2.0))
    assert info.value.kind == "planner_deadline"
    # Explicit kwargs win over the env.
    inner2 = _CapRecorder(_reply(genome))
    _mint(tmp_path / "kw", _Wrapper(inner2), max_completion_tokens=1234, deadline_s=30)
    assert inner2.seen == [1234]
    # Invalid / non-positive env values fall back to the defaults.
    monkeypatch.setenv("QB_PLANNER_MAX_COMPLETION_TOKENS", "zero")
    monkeypatch.setenv("QB_PLANNER_DEADLINE_S", "-5")
    inner3 = _CapRecorder(_reply(genome))
    _mint(tmp_path / "bad", _Wrapper(inner3))
    assert inner3.seen == [mt.DEFAULT_PLANNER_MAX_COMPLETION_TOKENS]


class _TogglePlanner:
    """Offline planner: reads the incumbent genome from the prompt and toggles
    one trailing mesh phase, so consecutive challengers differ in behaviour."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt, model_name, temperature=None, json_mode=True):
        self.calls += 1
        match = re.search(r"INCUMBENT GENOME:\n```python\n(.*?)\n```", prompt, re.S)
        genome = match.group(1) if match else ""
        if _MESH in genome:
            genome = genome.replace(_MESH, _BROADCAST, 1)
        else:
            genome = genome.replace(_BROADCAST, _MESH, 1)
        return _Resp(_reply(genome, mechanism=f"toggle {self.calls}"))


def test_two_offline_rounds_toggle_the_schedule_and_back(tmp_path) -> None:
    request, runtime = _env()
    incumbent = _seed()
    planner = _TogglePlanner()
    fingerprints = [ex.simulate_program_behavior(incumbent, n_agents=5, max_rounds=64)]
    for round_index in range(2):
        genome = gn.genome_region(incumbent).rstrip()
        prompt = f"INCUMBENT GENOME:\n```python\n{genome}\n```\n" + mt._GENOME_OUTPUT_CONTRACT
        hyp: dict[str, Any] = {}
        incumbent = mt.mint_python_challenger(
            planner_client=planner, planner_model="fake", prompt=prompt,
            request=request, runtime=runtime,
            workdir=tmp_path / f"round-{round_index}", genome_only=True,
            incumbent_source=incumbent, hypothesis_out=hyp,
        )
        assert hyp["mechanism"] == f"toggle {round_index + 1}"
        fingerprints.append(
            ex.simulate_program_behavior(incumbent, n_agents=5, max_rounds=64)
        )
    assert planner.calls == 2
    assert fingerprints[0] != fingerprints[1]
    assert fingerprints[2] == fingerprints[0]  # toggled back
    assert incumbent == _seed()


def test_planner_request_carries_only_the_dry_run_fields() -> None:
    request = mt.build_planner_request(
        n_agents=3, goal="sink", worker_contract="message_only_v2"
    )
    assert request.n_agents == 3
    assert request.information_goal == "sink"
    assert request.python_worker_contract == "message_only_v2"
    assert request.objective.name == "balanced"


def test_planner_usage_entry_shapes() -> None:
    class _U:
        prompt_tokens = 5
        completion_tokens = True  # not an int count
        reasoning_tokens = 3

    class _R:
        usage = _U()

    entry = mt.planner_usage_entry("mint", "m", _R(), 1.23456)
    assert entry == {"call": "mint", "model": "m", "prompt_tokens": 5,
                     "completion_tokens": None, "wall_s": 1.235,
                     "reasoning_tokens": 3}
    failed = mt.planner_usage_entry("remint", "m", None, -1, error="E" * 400)
    assert failed["wall_s"] == 0.0 and failed["prompt_tokens"] is None
    assert failed["error"] == "E" * 300


# --------------------------------------------------------------------------- #
# compute-only sends + validator wording
# --------------------------------------------------------------------------- #


def test_send_with_empty_recipients_is_a_compute_only_call() -> None:
    seed = _seed()
    genome = gn.genome_region(seed)
    genome = genome.replace(
        "PHASES = [\n", 'PHASES = [\n    {"kind": "compute_only", "rounds": 1},\n', 1
    )
    genome = genome.replace(
        "def phase_turn(kind, local_round, agent_id, n_agents, selected_primary,\n"
        "               known_source_count, inbox_count):\n",
        "def phase_compute_only(local_round, agent_id, n_agents, selected_primary,\n"
        "                       known_source_count, inbox_count):\n"
        '    return {"mode": "send", "recipients": []}\n\n\n'
        "def phase_turn(kind, local_round, agent_id, n_agents, selected_primary,\n"
        "               known_source_count, inbox_count):\n"
        '    if kind == "compute_only":\n'
        "        return phase_compute_only(local_round, agent_id, n_agents,\n"
        "                                  selected_primary, known_source_count,\n"
        "                                  inbox_count)\n",
        1,
    )
    source = gn.splice_genome(seed, genome)
    assert validate_python_source(source, worker_contract="message_only_v2").valid
    request, runtime = _env()
    result = _runner(runtime).run(
        source, _dry_run_payload(request, runtime),
        agent_canaries=_dry_run_canaries(request.n_agents),
    )
    assert result.runtime_success, result.failure
    round0 = [c for c in result.ledger["calls"] if c["round"] == 0]
    assert [(c["mode"], c["recipients"]) for c in round0] == [("send", [])] * 5
    assert [m for m in result.output.messages if m.round_sent == 0] == []
    # Paid: five compute calls in round 0 (plus the submit barrier).
    assert result.authoritative_usage.model_calls >= 5


def test_validator_wording_names_an_empty_source() -> None:
    report = validate_python_source("", worker_contract="message_only_v2")
    message = report.errors[0]["message"]
    assert message.startswith("program must call create_llm_client exactly once")
    assert "source is empty" in message
    other = validate_python_source("x = 1\n", worker_contract="message_only_v2")
    assert other.errors[0]["message"].endswith("(found 0)")
