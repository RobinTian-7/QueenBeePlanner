"""The seed program (v1) the evolution loop starts from (``queenbee.evo.seed``).

``evo_seed_source()`` is the phase-structured seed ``v1_seed_source()`` (the
program ``queenbee.evaluate --seed-program`` runs for the census) plus two
capability extensions applied by a deterministic text transform: phase-level
``wi`` forwarding and the ``digest`` phase kind.  The loop registers the evo
seed under the id ``v1`` together with the census rows, which is sound
because the two sources behave identically.

Offline (fake provider; the fake runs need the Silo-Bench files and are
skipped without them).  Pins:

* the evo seed passes the fail-closed validator and a fake run;
* identity with ``v1_seed_source()``: the code outside the genome region is
  identical, the behaviour fingerprint (``simulate_program_behavior``) is
  identical at n in {2, 5, 10}, every control JSON is identical (policy
  level, on and off the trajectory), and every worker prompt is identical in
  a fake run (runtime ledger ``prompt_sha256``);
* the interpreter forwards a phase's ``wi`` on send actions only, attaching
  it to a copy of the action, and a phase function's own
  ``work_instruction`` wins;
* ``digest``: agents with inbox this round send to ``[]``, others reflect;
  the runtime accepts it (fake run ledger shows the empty-recipient call);
* the transform is deterministic and fails loudly on a changed seed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

import queenbee.program.execute as pl
import queenbee.program.genome as pg
from exp_graph.mas.python_code import validate_python_source
from queenbee.evo import screen as S
from queenbee.evo import seed as es
from queenbee.paths import default_benchmarks_dir

#: sha256 of ``v1_seed_source()`` and of the evo seed.  Result rows and traces
#: are keyed on a program's sha256, so any edit of either source shows here.
V1_SHA256 = "a302510d734091a01544e467e41389b28aed8ed45325ef80dcbc0176dba0d2e9"
EVO_SHA256 = "e86d13746233668ab774ce1bf4786c21206f17cea950998569d998a6427642aa"
_V1_PHASES = '''PHASES = [
    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},
    {"kind": "broadcast_last", "rounds": 1},
]'''


def _with_phases(source: str, literal: str) -> str:
    assert source.count(_V1_PHASES) == 1
    return source.replace(_V1_PHASES, literal, 1)


def _policy(source: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(pg.genome_region(source), namespace)  # noqa: S102 - test fixture
    return namespace


def _control(action: dict[str, Any]) -> str:
    """The PYTHON_CONTROL_JSON main() renders for one action."""

    control = {
        "mode": str(action["mode"]),
        "recipients": [int(r) for r in action["recipients"]],
    }
    wi = str(action.get("work_instruction") or "")
    if wi:
        control["work_instruction"] = wi
    return json.dumps(control, sort_keys=True)


def _trajectory_controls(source: str, n: int, max_rounds: int) -> list[str]:
    ns = _policy(source)
    submit = int(ns["plan_submit_round"](n, max_rounds, "all_agents"))
    knowledge = [{a} for a in range(n)]
    inbox = [0] * n
    out = [f"submit={submit}"]
    for rnd in range(submit):
        edges = []
        for agent in range(n):
            action = ns["plan_communication_turn"](
                rnd, agent, n, "all_agents", 0, len(knowledge[agent]), inbox[agent]
            )
            out.append(f"{rnd}:{agent}:{_control(action)}")
            if action["mode"] == "send":
                edges.extend((agent, int(d)) for d in action["recipients"])
        snapshot = [set(k) for k in knowledge]
        fresh = [0] * n
        for src, dst in edges:
            knowledge[dst] |= snapshot[src]
            fresh[dst] += 1
        inbox = fresh
    return out


def _instance(case_id: str = "I-01", n: int = 5):
    bench = default_benchmarks_dir()
    if not (bench / f"{case_id}_n{n}.json").is_file():
        pytest.skip("Silo-Bench benchmarks not found (set SILO_BENCH_DIR)")
    from queenbee.bench.silo_bench import SiloBenchAdapter

    return next(SiloBenchAdapter(bench).iter_instances(cases=[case_id], agent_counts=[n]))


def _fake_run(source: str, instance: Any) -> Any:
    """runner.run as execute_python_source_on_case does under the default
    task (fake worker), returning the execution (output + authoritative
    ledger)."""

    from exp_graph.mas.python_code_runner import CodeProcessRunner, PythonExecutionLimits
    from queenbee.bench.engine import (
        _build_python_execution_payload,
        _protocol_adapter,
        _resolved_python_execution_timeout,
    )
    from queenbee.program.budgets import PythonRunBudgets

    n = int(instance.n_agents)
    cfg = pl._run_config(
        llm_provider="fake", worker_model="fake", goal="all_agents",
        worker_contract="message_only_v2", max_parallel_agents=n,
        request_timeout=30.0, n_agents=n,
        budgets=PythonRunBudgets.for_rounds(64, n_agents=n),
    )
    adapter = _protocol_adapter(instance, information_goal="all_agents")
    payload = _build_python_execution_payload(
        instance=instance, cfg=cfg, task_adapter=adapter,
        global_task=adapter.build_global_task(), n_agents=n,
    )
    runner = CodeProcessRunner(PythonExecutionLimits(
        timeout_seconds=_resolved_python_execution_timeout(cfg, n_agents=n),
        cpu_seconds=cfg.python_cpu_seconds, memory_mb=cfg.python_memory_mb,
        max_output_bytes=cfg.python_max_output_bytes,
    ))
    return runner.run(source, payload)


def _calls(execution: Any) -> list[tuple]:
    calls = [c for c in (execution.ledger or {}).get("calls") or []]
    return sorted(
        (int(c["round"]), int(c["agent_id"]), str(c.get("mode")),
         tuple(c.get("recipients") or []), c.get("prompt_sha256"))
        for c in calls
    )


# --------------------------------------------------------------------------- #
# validity + v1 identity
# --------------------------------------------------------------------------- #


def test_evo_seed_passes_the_validator() -> None:
    report = validate_python_source(es.evo_seed_source(), worker_contract="message_only_v2")
    assert report.valid, report.errors


def test_seed_shape_and_origin() -> None:
    origin, title, source = es.evo_seed()
    assert origin == es.EVO_SEED_ORIGIN and "+wi +digest" in title
    assert source == es.evo_seed_source()
    assert len(es.evo_seed_sha256()) == 64
    assert pg.genome_region(source).startswith("PHASES = [")
    assert "def phase_digest(" in source and 'kind == "digest"' in source
    assert 'phase.get("wi", "")' in source
    assert "work_instruction" not in pg.genome_region(es.v1_seed_source())  # no default WI


def test_seed_programs_are_byte_stable() -> None:
    assert hashlib.sha256(es.v1_seed_source().encode("utf-8")).hexdigest() == V1_SHA256
    assert es.evo_seed_sha256() == EVO_SHA256
    assert hashlib.sha256(es.evo_seed_source().encode("utf-8")).hexdigest() == EVO_SHA256


def test_prefix_and_main_are_byte_identical_to_v1() -> None:
    v1 = es.v1_seed_source()
    evo = es.evo_seed_source()
    v_start, v_end = pg.genome_bounds(v1)
    e_start, e_end = pg.genome_bounds(evo)
    assert evo[:e_start] == v1[:v_start]
    assert evo[e_end:] == v1[v_end:]
    assert v1 == pl.seed_python_source("message_only_v2", base="sfs_phase")[2]


@pytest.mark.parametrize("n", [2, 5, 10])
@pytest.mark.parametrize("max_rounds", [3, 8, 16, 64])
def test_fingerprint_and_coverage_identical_to_v1(n: int, max_rounds: int) -> None:
    v1 = es.v1_seed_source()
    evo = es.evo_seed_source()
    fp_v1 = pl.simulate_program_behavior(v1, n_agents=n, max_rounds=max_rounds)
    fp_evo = pl.simulate_program_behavior(evo, n_agents=n, max_rounds=max_rounds)
    assert fp_v1 is not None and fp_v1 == fp_evo
    sim_v1 = S.simulate_readable_coverage(v1, n, max_rounds)
    sim_evo = S.simulate_readable_coverage(evo, n, max_rounds)
    assert sim_v1.ok and sim_evo.ok
    assert sim_v1.summary() == sim_evo.summary()
    assert sim_v1.readable_sets == sim_evo.readable_sets
    assert S.behavior_fp(v1, n, max_rounds=max_rounds) == S.behavior_fp(evo, n, max_rounds=max_rounds)


@pytest.mark.parametrize("n", [2, 5, 10])
def test_control_json_identical_to_v1(n: int) -> None:
    v1 = es.v1_seed_source()
    evo = es.evo_seed_source()
    # On the executed trajectory ...
    for max_rounds in (8, 64):
        assert _trajectory_controls(v1, n, max_rounds) == _trajectory_controls(
            evo, n, max_rounds
        )
    # ... and on every (known_source_count, inbox_count) state off it.
    a, b = _policy(v1), _policy(evo)
    for goal in ("all_agents", "sink"):
        for rnd in range(n + 2):
            for agent in range(n):
                for known in range(1, n + 1):
                    for inbox in range(0, n):
                        args = (rnd, agent, n, goal, 0, known, inbox)
                        assert _control(a["plan_communication_turn"](*args)) == _control(
                            b["plan_communication_turn"](*args)
                        )


def test_worker_prompts_byte_identical_in_a_fake_run() -> None:
    inst = _instance("I-01", 5)
    run_v1 = _fake_run(es.v1_seed_source(), inst)
    run_evo = _fake_run(es.evo_seed_source(), inst)
    assert run_v1.runtime_success and run_evo.runtime_success
    calls_v1, calls_evo = _calls(run_v1), _calls(run_evo)
    assert len(calls_v1) == 10  # 5 relay/broadcast sends + 5 submits
    assert calls_v1 == calls_evo  # every prompt (control JSON included) identical
    assert run_v1.output.model_dump(mode="json") == run_evo.output.model_dump(mode="json")


def test_fake_run_through_execute_python_source_on_case(tmp_path) -> None:
    from queenbee.program.budgets import PythonRunBudgets

    facts = pl.execute_python_source_on_case(
        source=es.evo_seed_source(), instance=_instance("II-11", 5),
        llm_provider="fake", worker_model="fake", goal="all_agents",
        worker_contract="message_only_v2", max_parallel_agents=5,
        request_timeout=30.0, artifacts_dir=tmp_path,
        budgets=PythonRunBudgets.for_rounds(64, n_agents=5),
    )
    assert facts["n_messages"] == 8 and facts["model_calls"] == 10
    assert facts["diag"]["n_lost_messages"] == 0


# --------------------------------------------------------------------------- #
# capability extensions
# --------------------------------------------------------------------------- #

_WI = "Put every raw item you know, one per token, in your message body."


def test_interpreter_forwards_phase_wi_on_send_only() -> None:
    literal = (
        "PHASES = [\n"
        '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True, '
        f'"wi": "{_WI}"}},\n'
        '    {"kind": "broadcast_last", "rounds": 1},\n'
        "]"
    )
    source = _with_phases(es.evo_seed_source(), literal)
    assert validate_python_source(source, worker_contract="message_only_v2").valid
    controls = [json.loads(c.split(":", 2)[2]) for c in _trajectory_controls(source, 5, 64)[1:]]
    relay = controls[: 4 * 5]
    broadcast = controls[4 * 5:]
    sends = [c for c in relay if c["mode"] == "send"]
    assert len(sends) == 4 and all(c["work_instruction"] == _WI for c in sends)
    assert all("work_instruction" not in c for c in relay if c["mode"] != "send")
    assert all("work_instruction" not in c for c in broadcast)
    # The behaviour fingerprint sees the instruction (not a duplicate of v1).
    assert pl.simulate_program_behavior(source, n_agents=5, max_rounds=64) != (
        pl.simulate_program_behavior(es.v1_seed_source(), n_agents=5, max_rounds=64)
    )


def test_phase_function_work_instruction_wins_over_phase_wi() -> None:
    literal = (
        "PHASES = [\n"
        '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True, '
        '"wi": "phase text"},\n'
        '    {"kind": "broadcast_last", "rounds": 1},\n'
        "]"
    )
    source = _with_phases(es.evo_seed_source(), literal)
    old = '        return {"mode": "send", "recipients": [agent_id + 1]}\n'
    new = ('        return {"mode": "send", "recipients": [agent_id + 1], '
           '"work_instruction": "own text"}\n')
    assert source.count(old) == 1
    source = source.replace(old, new, 1)
    ns = _policy(source)
    action = ns["plan_communication_turn"](0, 0, 5, "all_agents", 0, 1, 0)
    assert action == {"mode": "send", "recipients": [1], "work_instruction": "own text"}


def test_phase_wi_attaches_to_a_copy_of_a_shared_action() -> None:
    """A genome whose phase function returns ONE module-level dict from two
    phases with different ``wi`` gets each phase's own WI (the shared dict is
    never mutated)."""

    literal = (
        "SEND_NEXT = {\"mode\": \"send\", \"recipients\": [1]}\n\n"
        "PHASES = [\n"
        '    {"kind": "shared", "rounds": 1, "wi": "first text"},\n'
        '    {"kind": "shared", "rounds": 1, "wi": "second text"},\n'
        "]"
    )
    source = _with_phases(es.evo_seed_source(), literal)
    old = '    if kind == "digest":\n'
    new = ('    if kind == "shared":\n'
           '        if agent_id == 0:\n'
           '            return SEND_NEXT\n'
           '        return {"mode": "reflect", "recipients": []}\n') + old
    assert source.count(old) == 1
    source = source.replace(old, new, 1)
    assert validate_python_source(source, worker_contract="message_only_v2").valid
    ns = _policy(source)
    first = ns["plan_communication_turn"](0, 0, 5, "all_agents", 0, 1, 0)
    second = ns["plan_communication_turn"](1, 0, 5, "all_agents", 0, 1, 0)
    assert first["work_instruction"] == "first text"
    assert second["work_instruction"] == "second text"
    assert ns["SEND_NEXT"] == {"mode": "send", "recipients": [1]}


def test_digest_kind_semantics() -> None:
    ns = _policy(es.evo_seed_source())
    assert ns["phase_digest"](0, 3, 5, 0, 2, 1) == {"mode": "send", "recipients": []}
    assert ns["phase_digest"](0, 3, 5, 0, 2, 0) == {"mode": "reflect", "recipients": []}
    assert ns["phase_turn"]("digest", 0, 1, 5, 0, 1, 2) == {"mode": "send", "recipients": []}


def test_digest_and_wi_run_in_the_fake_runtime() -> None:
    literal = (
        "PHASES = [\n"
        '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True, '
        f'"wi": "{_WI}"}},\n'
        '    {"kind": "digest", "rounds": 1, "wi": "Merge your inbox into one body."},\n'
        '    {"kind": "broadcast_last", "rounds": 1},\n'
        "]"
    )
    source = _with_phases(es.evo_seed_source(), literal)
    assert validate_python_source(source, worker_contract="message_only_v2").valid
    run = _fake_run(source, _instance("I-01", 5))
    assert run.runtime_success, run.failure
    calls = _calls(run)
    # relay r0..r3 (4 sends), digest r4 (agent 4 read agent 3's relay),
    # broadcast r5 (agent 4), submit r6 (5 agents).
    digest = [c for c in calls if c[0] == 4]
    assert [(c[1], c[2], c[3]) for c in digest] == [(4, "send", ())]
    assert sum(1 for c in calls if c[2] == "submit") == 5 and len(calls) == 11
    # The WI rides in the relay prompts: they differ from v1's relay prompts.
    v1_calls = _calls(_fake_run(es.v1_seed_source(), _instance("I-01", 5)))
    assert calls[0][4] != v1_calls[0][4]


# --------------------------------------------------------------------------- #
# transform hygiene
# --------------------------------------------------------------------------- #


def test_transform_is_deterministic_and_anchored() -> None:
    v1 = es.v1_seed_source()
    assert es.apply_evo_extensions(v1) == es.evo_seed_source()
    assert es.apply_evo_extensions(v1) == es.apply_evo_extensions(v1)
    with pytest.raises(RuntimeError, match="anchor"):
        es.apply_evo_extensions(v1.replace("def phase_turn(kind, ", "def turn(kind, "))
    with pytest.raises(RuntimeError, match="anchor"):
        es.apply_evo_extensions(es.evo_seed_source())  # not idempotent by design
