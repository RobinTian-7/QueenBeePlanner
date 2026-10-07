"""Tests for ``queenbee.evo.screen``, the zero-cost S0 screen.

The readable-coverage simulator is checked against the real runtime: each
program's own ``main()`` runs in-process with a content-tracking stub worker
whose every output is the set of shard ids it has read (own shard + its
previous output + the bodies in its delivered inbox), so the submitted
answers are exactly the runtime's readable sets.  Further tests cover
cross-checks against executed runs of the seed program (v1), agreement with
``diagnosis._lost_messages``, no false rejection of known-good programs, and
each S0 rule."""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from exp_graph.mas.python_code import DEFAULT_MESSAGE_ONLY_V2_PROGRAM
from queenbee.evo import ladder as L
from queenbee.evo import screen as S
from queenbee.evo.seed import evo_seed_source, v1_seed_source
from queenbee.paths import default_benchmarks_dir

#: Without the Silo-Bench files the work-instruction (WI) lint has no
#: vocabulary: the lint tests skip and every other screen runs on an empty
#: vocabulary.
_BENCH_OK = all((default_benchmarks_dir() / f"{t}_n5.json").is_file() for t in L.DEV_TEMPLATE_IDS)
needs_bench = pytest.mark.skipif(not _BENCH_OK, reason="Silo-Bench benchmarks not found")

V1 = v1_seed_source()
#: The message_only_v2 contract's default program: all-to-all every round
#: under the all_agents goal (the screen's default).
DEFAULT = DEFAULT_MESSAGE_ONLY_V2_PROGRAM.strip() + "\n"
_PHASES_RE = re.compile(r"^PHASES = \[.*?\n\]", re.S | re.M)


@pytest.fixture(autouse=True)
def _vocabulary_without_benchmarks(monkeypatch):
    if not _BENCH_OK:
        empty = S.WIVocabulary(unigrams=frozenset(), bigrams=frozenset(), templates=(),
                               source_texts=())
        monkeypatch.setattr(S, "default_wi_vocabulary", lambda benchmarks_dir=None: empty)


def _evo_seed() -> str:
    return evo_seed_source()


def with_phases(source: str, phases: str) -> str:
    assert len(_PHASES_RE.findall(source)) == 1
    return _PHASES_RE.sub(lambda _m: "PHASES = [\n" + phases + "\n]", source, count=1)


def _monolithic(default_source: str) -> str:
    """The schedule of the seed program (v1), a serial relay followed by the
    last agent's broadcast, written straight into the default program's two
    policy functions (no PHASES list)."""

    old_schedule = (
        "def plan_submit_round(n_agents, max_rounds, information_goal):\n"
        "    return max_rounds - 1\n"
    )
    new_schedule = (
        "def plan_submit_round(n_agents, max_rounds, information_goal):\n"
        "    if max_rounds <= 1 or n_agents <= 1:\n"
        "        return 0\n"
        "    return min(max_rounds - 1, n_agents)\n"
    )
    old_turn = (
        '    if n_agents == 1:\n'
        '        return {"mode": "reflect", "recipients": []}\n'
        '    recipients = []\n'
        '    for recipient in range(n_agents):\n'
        '        if recipient != agent_id:\n'
        '            recipients.append(recipient)\n'
        '    return {"mode": "send", "recipients": recipients}\n'
    )
    new_turn = (
        '    if n_agents == 1:\n'
        '        return {"mode": "reflect", "recipients": []}\n'
        '    if round_idx < n_agents - 1:\n'
        '        if agent_id == round_idx:\n'
        '            return {"mode": "send", "recipients": [agent_id + 1]}\n'
        '        return {"mode": "reflect", "recipients": []}\n'
        '    if round_idx == n_agents - 1 and agent_id == n_agents - 1:\n'
        '        recipients = []\n'
        '        for r in range(n_agents):\n'
        '            if r != agent_id:\n'
        '                recipients.append(r)\n'
        '        return {"mode": "send", "recipients": recipients}\n'
        '    return {"mode": "reflect", "recipients": []}\n'
    )
    assert default_source.count(old_schedule) == 1 and default_source.count(old_turn) == 1
    return default_source.replace(old_schedule, new_schedule, 1).replace(old_turn, new_turn, 1)


_RELAY = '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},'
_BROADCAST = '    {"kind": "broadcast_last", "rounds": 1},'
_CLOSING_WI = ("The next round is the synchronized submit: state your complete final "
               "answer now and the key figures it rests on.")


def _with_closing_round(source: str) -> str:
    """``source`` plus one closing all-to-all round whose phase function
    returns its own work instruction (bodies delivered at the barrier)."""

    func = (
        "\n\n\ndef phase_closing(local_round, agent_id, n_agents, selected_primary,\n"
        "                  known_source_count, inbox_count):\n"
        "    recipients = []\n"
        "    for r in range(n_agents):\n"
        "        if r != agent_id:\n"
        "            recipients.append(r)\n"
        "    return {\"mode\": \"send\", \"recipients\": recipients,\n"
        f"            \"work_instruction\": {json.dumps(_CLOSING_WI)}}}"
    )
    anchor = "\n\n\ndef phase_turn(kind, "
    assert source.count(anchor) == 1
    source = source.replace(anchor, func + anchor, 1)
    old = '    if kind == "relay":\n'
    assert source.count(old) == 1
    source = source.replace(
        old,
        '    if kind == "closing":\n'
        '        return phase_closing(local_round, agent_id, n_agents, selected_primary,\n'
        '                             known_source_count, inbox_count)\n' + old, 1)
    return with_phases(source, _RELAY + "\n" + _BROADCAST + '\n    {"kind": "closing", "rounds": 1},')


def _without_agent0_in_the_broadcast(source: str) -> str:
    """Agent 0 is left out of the final broadcast: it is blind by design
    (readable = known = [1, n, ..., n])."""

    old = ("    if agent_id == n_agents - 1:\n        recipients = []\n"
           "        for r in range(n_agents):\n            if r != agent_id:\n")
    assert source.count(old) == 1
    return source.replace(old, old.replace("if r != agent_id:", "if r != agent_id and r != 0:"), 1)


MONOLITHIC = _monolithic(DEFAULT)
CLOSING = _with_closing_round(V1)
DEGRADED = _without_agent0_in_the_broadcast(V1)


# a lost-message trap: after one all-to-all round everybody knows every id,
# phase_mesh then goes quiet ("reflect") and the round-0 bodies are lost
MESH_TRAP = with_phases(V1, '    {"kind": "mesh", "rounds": 2},')
# an early broadcast that most agents miss (lost edges) but whose content
# still reaches everyone through the relay + final broadcast
LOSSY_OK = with_phases(
    V1,
    '    {"kind": "broadcast_last", "rounds": 1},\n'
    '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},\n'
    '    {"kind": "broadcast_last", "rounds": 1},',
)
HUB = with_phases(
    V1,
    '    {"kind": "gather_to_hub", "rounds": 1},\n'
    '    {"kind": "broadcast_from_hub", "rounds": 1},',
)
PEER = with_phases(V1, '    {"kind": "one_peer_step", "rounds": 3},')


# --------------------------------------------------------------------------- #
# the real main() with a content-tracking stub worker
# --------------------------------------------------------------------------- #


class _ContentClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def _ids(text: str) -> set[int]:
        return set(json.loads(text[5:])) if text.startswith("READ=") else set()

    def complete_batch(self, prompts, *, model_name, temperature, json_mode):
        del model_name, temperature, json_mode
        out = []
        for prompt in prompts:
            head = dict(re.findall(
                r"^(PYTHON_AGENT_ID|PYTHON_ROUND|PREVIOUS_OUTPUT_JSON|DELIVERED_INBOX_JSON):(.*)$",
                prompt, flags=re.M))
            agent, rnd = int(head["PYTHON_AGENT_ID"]), int(head["PYTHON_ROUND"])
            submit = "\nSUBMIT_PROMPT:\n" in prompt
            ids = {agent} | self._ids(json.loads(head["PREVIOUS_OUTPUT_JSON"]))
            bodies = json.loads(head["DELIVERED_INBOX_JSON"])
            for message in bodies:
                ids |= self._ids(message["body"])
            self.calls.append({"round": rnd, "agent_id": agent, "mode": "submit" if submit else "send"})
            text = json.dumps(sorted(ids)) if submit else "READ=" + json.dumps(sorted(ids))
            out.append(SimpleNamespace(
                text=text,
                usage=SimpleNamespace(model_calls=1, prompt_tokens=1, completion_tokens=1),
            ))
        return out


def run_real_main(source: str, n_agents: int) -> tuple[dict, list[dict]]:
    import exp_graph.llm.factory as factory
    from queenbee.program.budgets import PythonRunBudgets

    budgets = PythonRunBudgets.for_rounds(64, n_agents=n_agents)
    payload = {
        "n_agents": n_agents, "max_rounds": 64, "information_goal": "all_agents",
        "selected_primary": 0,
        "budgets": {"max_model_calls": budgets.max_model_calls,
                    "max_completion_tokens": budgets.max_completion_tokens,
                    "max_messages": budgets.max_messages},
        "worker_llm": {"provider": "stub", "model_name": "stub", "base_url": None,
                       "api_key_env": None, "temperature": 0.0},
        "agents": [{"agent_id": i, "communication_prompt": f"COMM {i}",
                    "submit_prompt": f"SUBMIT {i}"} for i in range(n_agents)],
    }
    client = _ContentClient()
    stdout = io.StringIO()
    with mock.patch.object(factory, "create_llm_client", lambda *_a, **_k: client), \
            mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), \
            contextlib.redirect_stdout(stdout):
        exec(compile(source, "<program>", "exec"), {"__name__": "__main__"})  # noqa: S102
    return json.loads(stdout.getvalue()), client.calls


def _programs() -> dict[str, str]:
    progs = {
        "v1": V1, "monolithic": MONOLITHIC, "mesh_trap": MESH_TRAP, "lossy_ok": LOSSY_OK,
        "hub": HUB, "peer": PEER, "closing": CLOSING, "degraded": DEGRADED,
    }
    evo = _evo_seed()
    if evo is not None:
        progs["evo_seed"] = evo
        progs["evo_digest"] = with_phases(
            evo, '    {"kind": "mesh", "rounds": 1},\n    {"kind": "digest", "rounds": 1},')
        progs["evo_digest_trap"] = with_phases(
            evo, '    {"kind": "mesh", "rounds": 1},\n    {"kind": "digest", "rounds": 1},\n'
                 '    {"kind": "mesh", "rounds": 1},\n    {"kind": "relay", "rounds": 1},')
    return progs


@pytest.mark.parametrize("n", [2, 5, 10])
@pytest.mark.parametrize("name", sorted(_programs()))
def test_simulator_matches_the_real_main(name, n):
    from queenbee.evo.diagnosis import _lost_messages

    source = _programs()[name]
    out, calls = run_real_main(source, n)
    sim = S.simulate_readable_coverage(source, n, 64)
    assert sim.ok, sim.error
    answers = {s["agent_id"]: s["answer"] for s in out["submissions"]}
    assert sim.readable_sets == [answers[a] for a in range(n)]  # readable == what was read
    assert sim.submit_round == out["rounds_executed"] - 1
    assert sim.calls == len(calls) == out["usage"]["model_calls"]
    assert sim.messages == len(out["messages"])
    sim_edges = sorted((m["round_sent"], m["src"], m["dst"], tuple(m["source_ids"])) for m in sim.message_log)
    run_edges = sorted((m["round_sent"], m["src"], m["dst"], tuple(m["source_ids"])) for m in out["messages"])
    assert sim_edges == run_edges
    assert sim.lost_edges == _lost_messages(out["messages"], calls, sim.submit_round)
    # the trace-side recomputation agrees too (with the exact call ledger)
    trace = S.readable_from_messages(out["messages"], n, sim.submit_round, calls=calls)
    assert trace["readable"] == sim.readable and trace["lost_edges"] == sim.lost_edges
    # id coverage: main() merges a message's source ids on delivery
    known = [{a} for a in range(n)]
    for m in out["messages"]:
        if m["round_delivered"] <= sim.submit_round:
            known[m["dst"]] |= set(m["source_ids"])
    assert sim.known == [len(k) for k in known]


@pytest.mark.parametrize("name", sorted(_programs()))
def test_credit_readable_agrees_with_the_simulator(name):
    """``queenbee.evo.credit``'s trace-side readable coverage equals the
    simulator's, both with the runtime call ledger and with the ledger implied
    by the program source."""

    from queenbee.evo.credit import credit_fields

    source = _programs()[name]
    for n in (5, 10):
        out, calls = run_real_main(source, n)
        inst = SimpleNamespace(n_agents=n, shards=[[i] for i in range(n)], case_id="II-11@x5")
        sim = S.simulate_readable_coverage(source, n, 64)
        assert credit_fields(out, inst, source, calls=calls)["readable"] == sim.readable
        assert credit_fields(out, inst, source)["readable"] == sim.readable, (name, n)


def test_the_trap_programs_really_lose_content():
    sim = S.simulate_readable_coverage(MESH_TRAP, 5)
    assert sim.known == [5] * 5 and sim.readable == [1] * 5 and sim.lost_edges == 20
    lossy = S.simulate_readable_coverage(LOSSY_OK, 5)
    assert lossy.readable == [5] * 5 and lossy.lost_edges == 3
    evo = _evo_seed()
    if evo is not None:
        dig = S.simulate_readable_coverage(_programs()["evo_digest"], 5)
        assert dig.readable == [5] * 5 and dig.lost_edges == 0 and dig.calls == 5 + 5 + 5


# --------------------------------------------------------------------------- #
# executed runs of the seed program (v1); trace records carry no call ledger
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 5, 10])
def test_crosscheck_against_executed_v1_runs(n):
    out, _calls = run_real_main(V1, n)
    record = {"case_id": "II-11", "n_agents": n, "output": out}
    check = S.crosscheck_trace(V1, record)
    assert check["agree"], check
    assert check["ledger_basis"] == "sim"
    assert check["lost_diag"] == check["lost_sim"] == 0
    assert check["readable_sim"] == [n] * n
    # a run of another program does not agree with v1's simulation
    other, _ = run_real_main(HUB, n)
    assert not S.crosscheck_trace(V1, {"n_agents": n, "output": other})["agree"]


# --------------------------------------------------------------------------- #
# S0: no false rejections, and each rule
# --------------------------------------------------------------------------- #


def test_known_good_programs_are_not_rejected():
    for name, source in [("v1", V1), ("monolithic", MONOLITHIC), ("closing", CLOSING),
                         ("hub", HUB)]:
        res = S.screen_program(source)
        assert res.ok, (name, res.reasons)
        assert res.dup_of is None and res.lint_hits == []
    closing = S.screen_program(CLOSING)
    assert closing.wis == [_CLOSING_WI] and closing.penalties == []
    assert (closing.pred_calls5, closing.pred_calls10, closing.pred_messages5) == (15, 30, 28)
    evo = _evo_seed()
    if evo is not None:
        assert S.screen_program(evo).ok
    # a program blind at agent 0 by design is rejected for coverage ONLY
    res = S.screen_program(DEGRADED)
    assert not res.ok and res.reasons
    assert all("coverage" in r.lower() or "readable" in r.lower() for r in res.reasons)


def test_v1_screen_numbers():
    res = S.screen_program(V1)
    assert res.readable5 == [5] * 5 and res.readable10 == [10] * 10
    assert (res.pred_calls5, res.pred_calls10) == (10, 20)
    assert (res.pred_messages5, res.pred_messages10) == (8, 18)
    assert res.lost_edges5 == 0 and res.penalties == [] and res.rank_penalty == 0.0
    assert S.v1_reference_calls() == {5: 10, 10: 20}
    json.dumps(res.to_dict())  # serialisable


def test_coverage_is_hard_by_default_and_one_short_agent_is_a_penalty_when_lenient():
    degraded = DEGRADED
    res = S.screen_program(degraded)
    assert not res.ok and res.readable5 == [1, 5, 5, 5, 5] and res.known5 == [1, 5, 5, 5, 5]
    assert [r.split(":")[0] for r in res.reasons] == ["coverage@n5", "coverage@n10"]
    lenient = S.screen_program(degraded, strict_coverage=False)  # lenient coverage mode
    assert lenient.ok and any(p.startswith("coverage@n5") for p in lenient.penalties)
    assert lenient.rank_penalty > 0


@pytest.mark.parametrize("phases,readable5", [
    ('    {"kind": "relay", "rounds": 1},', [1, 2, 1, 1, 1]),
    ('    {"kind": "one_peer_step", "rounds": 1},', [2, 2, 2, 2, 2]),
])
def test_near_silent_programs_never_pass_even_when_lenient(phases, readable5):
    silent = with_phases(V1, phases)
    for strict in (True, False):
        res = S.screen_program(silent, strict_coverage=strict)
        assert not res.ok and res.readable5 == readable5
        assert any(r.startswith("coverage@n5") for r in res.reasons)


def test_coverage_is_checked_at_every_n():
    res = S.screen_program(PEER)  # 3 unscaled peer steps: enough at n=5, not at n=10
    assert res.readable5 == [5] * 5 and res.readable10 == [8] * 10
    assert not res.ok and [r.split(":")[0] for r in res.reasons] == ["coverage@n10"]


def test_unread_content_is_a_hard_reject():
    res = S.screen_program(MESH_TRAP)
    assert not res.ok
    assert any(r.startswith("unread-content@n5") for r in res.reasons)
    assert any(r.startswith("unread-content@n10") for r in res.reasons)


def test_lost_edges_alone_are_only_a_penalty():
    res = S.screen_program(LOSSY_OK)
    assert res.ok and res.lost_edges5 == 3
    assert any(p.startswith("lost-edges@n5") for p in res.penalties)


def test_call_caps():
    res = S.screen_program(DEFAULT)  # all-to-all every round
    assert not res.ok and any(r.startswith("calls@n5") for r in res.reasons)
    # 2 < ratio <= 3: penalty only
    twice = ('    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},\n'
             '    {"kind": "broadcast_last", "rounds": 1},\n')
    mid = with_phases(V1, twice + twice.rstrip("\n"))
    res = S.screen_program(mid, v1_calls={5: 10, 10: 20})
    assert (res.pred_calls5, res.pred_calls10) == (15, 30) and res.ok, res.reasons
    res = S.screen_program(mid, v1_calls={5: 6, 10: 12})  # 15/6 = 2.5x
    assert res.ok and any(p.startswith("calls@n5") for p in res.penalties)
    res = S.screen_program(mid, v1_calls={5: 4, 10: 8})  # 15/4 = 3.75x
    assert not res.ok and any(r.startswith("calls@n5") for r in res.reasons)
    # calls_reject_ratio=None: no hard cap, the excess is only a penalty
    res = S.screen_program(mid, v1_calls={5: 4, 10: 8}, calls_reject_ratio=None)
    assert res.ok and any(p.startswith("calls@n5") for p in res.penalties)


def _with_wi(source: str, wi: str) -> str:
    return with_phases(source, '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True,'
                               f' "wi": {json.dumps(wi)}}},\n    {{"kind": "broadcast_last", "rounds": 1}},')


@needs_bench
def test_wi_lint_vocabulary_is_dev_only_and_specific():
    vocab = S.default_wi_vocabulary()
    assert vocab.templates == L.DEV_TEMPLATE_IDS
    for term in ("palindrome", "apple", "xor", "median", "pagerank"):
        assert term in vocab.unigrams
    for term in ("answer", "majority", "sum", "sort", "portion", "word", "vote", "distinct",
                 "submit", "global", "boundary", "agent", "final"):
        assert S._stem(term) not in vocab.unigrams, term
    assert ("prefix", "sum") in vocab.bigrams and ("hash", "based") not in vocab.bigrams
    with pytest.raises(L.LadderLeakError):
        S.build_wi_vocabulary(templates=[L.TEST_TEMPLATE_IDS[0]])
    assert S.lint_work_instructions(["Compute the Prefix Sums of your segment"]) == ["prefix sum"]
    assert S.lint_work_instructions(["Find palindromes that cross boundaries"]) == ["palindrome"]
    generic = ("Reply with FINAL=<answer>. If every agent must give the same answer, adopt the "
               "majority; if each agent submits its own portion, keep it. At most 40 words; "
               "sort and sum raw values, count distinct answers, use hash-based partitioning.")
    assert S.lint_work_instructions([generic]) == []


@needs_bench
def test_wi_lint_rejects_task_specific_instructions():
    evo = _evo_seed()
    bad = _with_wi(evo, "Forward your running prefix sum to the next agent.")
    res = S.screen_program(bad)
    assert not res.ok and res.lint_hits == ["prefix sum"]
    assert res.wis == ["Forward your running prefix sum to the next agent."]
    good = _with_wi(evo, "Forward every raw value you hold or received, verbatim.")
    assert S.screen_program(good).ok
    # a phase wi that is never used is linted too (static scan of PHASES)
    unused = with_phases(evo, '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},\n'
                              '    {"kind": "broadcast_last", "rounds": 1},\n'
                              '    {"kind": "reflect_only", "rounds": 0, "wi": "xor checksum"},')
    assert "xor" in S.screen_program(unused).lint_hits


def test_overlong_work_instruction_is_the_runtime_error():
    evo = _evo_seed()
    res = S.screen_program(_with_wi(evo, "x" * 801))
    assert not res.ok
    assert any("work_instruction exceeds 800 characters" in r for r in res.reasons)


def test_policy_crash_and_bad_actions_are_reasons_not_exceptions():
    crash = V1.replace('    if kind == "relay":\n', '    if kind == "relay":\n        _boom = 1 // 0\n', 1)
    res = S.screen_program(crash, validate=False)
    assert not res.ok and any("ZeroDivisionError" in r for r in res.reasons)
    self_send = V1.replace("return {\"mode\": \"send\", \"recipients\": [agent_id + 1]}",
                           "return {\"mode\": \"send\", \"recipients\": [agent_id]}", 1)
    res = S.screen_program(self_send, validate=False)
    assert any("recipient outside the agent range" in r for r in res.reasons)
    res = S.screen_program("print('hi')\n")
    assert not res.ok and any(r.startswith("validation") for r in res.reasons)


def test_fingerprint_duplicates_and_submit_round_sensitivity():
    fps = {n: S.behavior_fp(V1, n) for n in (5, 10)}
    key = S.fp_key(fps)
    assert key and all(v.startswith("lb:") for v in fps.values())
    res = S.screen_program(V1, known_fps={key: "v1"})
    assert not res.ok and res.dup_of == "v1" and res.fp_key == key
    evo = _evo_seed()
    if evo is not None:  # evo_seed_source() (v1 + wi forwarding + digest) behaves like v1
        assert S.screen_program(evo, known_fps={key: "v1"}).dup_of == "v1"
    assert S.screen_program(CLOSING, known_fps={key: "v1"}).dup_of is None
    # a trailing all-idle round: same simulate_program_behavior tuple,
    # different barrier
    idle = V1.replace(
        "def phase_turn(kind,",
        "def phase_rest(local_round, agent_id, n_agents, selected_primary,\n"
        "               known_source_count, inbox_count):\n"
        "    return {\"mode\": \"idle\", \"recipients\": []}\n\n\n"
        "def phase_turn(kind,", 1,
    ).replace(
        '    if kind == "relay":\n',
        '    if kind == "rest":\n'
        '        return phase_rest(local_round, agent_id, n_agents, selected_primary,\n'
        '                          known_source_count, inbox_count)\n'
        '    if kind == "relay":\n', 1,
    )
    idle = with_phases(idle, '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},\n'
                             '    {"kind": "broadcast_last", "rounds": 1},\n'
                             '    {"kind": "rest", "rounds": 1},')
    from queenbee.program.execute import simulate_program_behavior

    assert simulate_program_behavior(idle, n_agents=5, max_rounds=64) == \
        simulate_program_behavior(V1, n_agents=5, max_rounds=64)
    assert S.behavior_fp(idle, 5) != S.behavior_fp(V1, 5)
    trailing = S.screen_program(idle)  # ...and the broadcast dies unread
    assert not trailing.ok and any(r.startswith("unread-content") for r in trailing.reasons)
    # the program simulator may decline a program: the screen then keys on
    # its own control log ("rs:").  The monolithic program runs v1's schedule
    # with v1's host globals, so its key equals v1's
    assert S.behavior_fp(MONOLITHIC, 5)[:3] in ("lb:", "rs:")
    with mock.patch("queenbee.program.execute.simulate_program_behavior",
                    lambda *_a, **_k: None):
        own = S.behavior_fp(MONOLITHIC, 5)
        assert own.startswith("rs:") and own == S.behavior_fp(V1, 5)
        assert S.behavior_fp(HUB, 5) != own


def test_tabu_hook():
    hyp = {"failure_class": "divergent", "mechanism": "Final  RECONCILE round!"}
    res = S.screen_program(V1, tabu=[("divergent", "final reconcile round")], hypothesis=hyp)
    assert not res.ok and any(r.startswith("tabu") for r in res.reasons)
    assert S.screen_program(V1, tabu=[("divergent", "other")], hypothesis=hyp).ok
    res = S.screen_program(V1, tabu=lambda src, h: "tabu: callable says no")
    assert res.reasons == ["tabu: callable says no"]


def test_main_variants_are_recognised():
    for source in (V1, MONOLITHIC, CLOSING, HUB, _evo_seed()):
        assert S.main_semantics(source) == S.MAIN_V2
    tampered = V1.replace("        worker_output = response.text.strip()\n",
                          "        worker_output = response.text.strip()[:10]\n", 1)
    assert tampered != V1 and S.main_semantics(tampered) == S.MAIN_UNKNOWN
    res = S.screen_program(tampered)
    assert not res.ok and any(r.startswith("main: unrecognised") for r in res.reasons)
    assert S.main_semantics("print(1)\n") == S.MAIN_UNKNOWN
    # an unrecognised main() is still simulated with the message_only_v2
    # main() semantics
    sim = S.simulate_readable_coverage(tampered, 5)
    assert sim.ok and sim.semantics == S.MAIN_UNKNOWN and sim.readable == [5] * 5


# --------------------------------------------------------------------------- #
# top-level rebinding, host globals, digest traces, policy deadline
# --------------------------------------------------------------------------- #


def _rebinding_programs() -> dict[str, str]:
    one_relay = '[{"kind": "relay", "rounds": 1}]'
    return {
        "if_rebinds_phases": V1.replace("\n\ndef phase_relay(",
                                        f"\n\nif True:\n    PHASES = {one_relay}\n\n\ndef phase_relay(", 1),
        "append_to_phases": V1.replace("\n\ndef phase_relay(",
                                       '\n\nPHASES.append({"kind": "mesh", "rounds": 1})\n\n\ndef phase_relay(', 1),
        "main_guard": V1.replace("\n\nmain()", "\n\nif __name__ == \"__main__\":\n    main()", 1),
    }


@pytest.mark.parametrize("name", sorted(_rebinding_programs()))
@pytest.mark.parametrize("n", [2, 5, 10])
def test_top_level_rebinding_is_simulated_like_the_runtime(name, n):
    source = _rebinding_programs()[name]
    assert source != V1
    out, calls = run_real_main(source, n)
    sim = S.simulate_readable_coverage(source, n, 64)
    assert sim.ok, sim.error
    answers = {s["agent_id"]: s["answer"] for s in out["submissions"]}
    assert sim.readable_sets == [answers[a] for a in range(n)]
    assert sim.calls == len(calls) and sim.messages == len(out["messages"])
    assert sim.submit_round == out["rounds_executed"] - 1


@needs_bench
def test_a_top_level_if_cannot_hide_the_real_schedule():
    source = _rebinding_programs()["if_rebinds_phases"]
    res = S.screen_program(source, validate=False)
    assert res.readable5 == [1, 2, 1, 1, 1] and res.pred_calls5 == 6  # one relay round, not v1
    assert not res.ok and res.fp_key != S.program_fp_key(V1)
    guard = S.screen_program(_rebinding_programs()["main_guard"])
    assert guard.readable5 == [5] * 5 and guard.pred_calls5 == 10
    hidden_wi = V1.replace("\n\ndef phase_relay(",
                           '\n\nif False:\n    PHASES = [{"kind": "relay", "rounds": 1, "wi": "xor it"}]'
                           "\n\n\ndef phase_relay(", 1)
    assert "xor" in S.screen_program(hidden_wi, validate=False).lint_hits


def test_fingerprint_covers_the_globals_main_reads():
    v1_key = S.program_fp_key(V1)
    assert S.host_digest(V1) is not None
    other_msg = V1.replace("Return only the message body as compact plain text.",
                           "Return only the message body.", 1)
    assert other_msg != V1 and S.host_digest(other_msg) != S.host_digest(V1)
    assert S.program_fp_key(other_msg) != v1_key
    assert S.screen_program(other_msg, known_fps={v1_key: "v1"}).dup_of is None
    rebound = V1.replace("\n\ndef phase_relay(",
                         '\n\nMESSAGE_INSTRUCTION = "Say only OK."\n\n\ndef phase_relay(', 1)
    assert S.program_fp_key(rebound) != v1_key
    # layout / comments / the policy entry points' text are not host content
    cosmetic = V1.replace("def reject_nonfinite(value):\n",
                          "def reject_nonfinite(value):  # comment\n", 1)
    assert cosmetic != V1 and S.host_digest(cosmetic) == S.host_digest(V1)
    evo = _evo_seed()
    if evo is not None:
        assert S.host_digest(evo) == S.host_digest(V1)
        assert S.program_fp_key(evo) == v1_key


def test_known_fps_for_seeds_the_duplicate_check():
    known = S.known_fps_for({"v1": V1, "monolithic": MONOLITHIC, "hub": HUB, "broken": "print(1)\n"})
    assert sorted(known.values()) == ["hub", "v1"]  # monolithic behaves like v1: first id wins
    no_op = V1.replace("def phase_relay(local_round,", "def phase_relay(local_round,  ", 1)
    assert no_op != V1
    res = S.screen_program(no_op, known_fps=known)
    assert not res.ok and res.dup_of == "v1" and res.reasons == ["duplicate-of:v1"]
    evo = _evo_seed()
    if evo is not None:  # evo_seed_source() (v1 + wi forwarding + digest) behaves like v1
        assert S.known_fps_for([("evo_seed", evo), ("v1", V1)]) == {S.program_fp_key(V1): "evo_seed"}


@pytest.mark.parametrize("n", [5, 10])
def test_crosscheck_sees_digest_sends_through_the_implied_ledger(n):
    evo = _evo_seed()
    source = with_phases(evo, '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},\n'
                              '    {"kind": "broadcast_last", "rounds": 1},\n'
                              '    {"kind": "digest", "rounds": 1},')
    out, calls = run_real_main(source, n)
    record = {"n_agents": n, "output": out}
    check = S.crosscheck_trace(source, record)
    assert check["agree"], check
    assert check["ledger_basis"] == "sim"
    assert check["readable_trace"] == check["readable_sim"] == [n] * n
    assert check["lost_diag"] == check["lost_sim"] == 0
    sim = S.simulate_readable_coverage(source, n)
    implied = S.sim_call_ledger(sim)
    key = lambda c: (c["round"], c["agent_id"], c["mode"])  # noqa: E731
    assert sorted(map(key, implied)) == sorted(map(key, calls))  # == the runtime's ledger
    assert S.crosscheck_trace(source, record, calls=calls)["ledger_basis"] == "given"
    # the sender-inferred called set misses the digest sends to []
    blind = S.readable_from_messages(out["messages"], n, sim.submit_round)
    assert blind["readable"] != [n] * n and blind["lost_edges"] > 0


def test_policy_code_that_never_returns_is_a_rejection_not_a_hang():
    import time

    spin = V1.replace('    if kind == "relay":\n',
                      '    if kind == "relay":\n        while True:\n            pass\n', 1)
    started = time.monotonic()
    res = S.screen_program(spin, validate=False, timeout_s=0.3)
    assert time.monotonic() - started < 5
    assert not res.ok and any("screen deadline" in r for r in res.reasons)
    at_load = V1.replace("\n\ndef phase_relay(", "\n\nwhile True:\n    pass\n\n\ndef phase_relay(", 1)
    sim = S.simulate_readable_coverage(at_load, 5, timeout_s=0.3)
    assert not sim.ok and "screen deadline" in sim.error
    time.sleep(0.5)  # no stray interrupt reaches the caller afterwards
    assert S.simulate_readable_coverage(V1, 5, timeout_s=0.3).ok
