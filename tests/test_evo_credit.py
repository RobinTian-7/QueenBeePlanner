"""Credit fields of the diagnosis cards (:mod:`queenbee.evo.credit`).

Offline.  Pins:

* the TEST guard shared by the evolution modules (it refuses the 12 TEST
  templates in every form);
* readable coverage counts content actually read (as a called recipient or
  at the submit barrier), not source ids, with the called set from the call
  ledger, the simulated policy (a send to ``[]`` leaves no message) or the
  trace's senders;
* raw-input retention: distinct shard items of the other agents present in
  the bodies an agent read (numbers at shard / rounded / truncated
  precision, strings as tokens, long letter strings as 4-grams);
* the field whitelist, the never-raise contract, the card renderer, and a
  zero-hit leakage audit on executed traces of the seed program (v1) with
  the fake worker.
"""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace
from typing import Any

import pytest

import queenbee.program.execute as pl
import queenbee.program.mint as pm
from queenbee.evo import common as C
from queenbee.evo import diagnosis as dg
from queenbee.evo import credit as cr
from queenbee.evo import seed as es
from queenbee.paths import default_benchmarks_dir

_BENCH = default_benchmarks_dir()


def _inst(shards, *, case_id="I-99", expected=None):
    n = len(shards)
    expected = expected if expected is not None else [7] * n
    return SimpleNamespace(
        case_id=case_id, n_agents=n, shards=list(shards),
        task_prompt="**Output:**\nA single integer.\n", ground_truth=expected[0],
        meta={"expected_outputs": list(expected), "is_segmented": False},
    )


def _msg(sent, src, dst, body="", ids=None):
    return {"round_sent": sent, "round_delivered": sent + 1, "src": src,
            "dst": dst, "source_ids": ids or [src], "body": body}


def _out(n, messages, submit_round):
    return {
        "submissions": [{"agent_id": a, "answer": 7, "submitted_round": submit_round}
                        for a in range(n)],
        "rounds_executed": submit_round + 1,
        "messages": list(messages),
        "usage": {"model_calls": 1, "prompt_tokens": 1, "completion_tokens": 1},
        "errors": [],
    }


# --------------------------------------------------------------------------- #
# TEST guard
# --------------------------------------------------------------------------- #


def test_test_ids_are_the_shared_holdout() -> None:
    assert tuple(cr.TEST_CASE_IDS) == tuple(C.TEST_IDS)
    assert len(set(cr.TEST_CASE_IDS)) == 12


@pytest.mark.parametrize("bad", [
    "I-04", "II-15", "III-30", "I-10@x5", "II-19@o10", "III-24_n5.json",
    "/x/III-29_abc_123.json",
])
def test_guard_refuses_every_test_form(bad: str) -> None:
    with pytest.raises(RuntimeError, match="LEAKAGE_GUARD"):
        cr.assert_not_test([bad])
    with pytest.raises(RuntimeError, match="LEAKAGE_GUARD"):
        cr.credit_fields(_out(1, [], 0), _inst([[1]], case_id=bad))
    with pytest.raises(RuntimeError, match="LEAKAGE_GUARD"):
        cr.load_dev_instance(bad.split("@")[0].split("_")[0].split("/")[-1])


def test_guard_accepts_dev_ids() -> None:
    cr.assert_not_test(["I-01", "II-11@x5", "III-21@o10", "II-13_n5.json", "I-99"])
    assert cr.templates_in("x III-21@o5 y I-01") == ["III-21", "I-01"]


# --------------------------------------------------------------------------- #
# readable coverage
# --------------------------------------------------------------------------- #


def test_readable_counts_read_content_not_ids() -> None:
    # 0 -> 1 delivered r1, but agent 1 is only called at r2: body LOST.
    # 1 -> 2 at r2, delivered r3 = barrier: read by agent 2 at submit.
    messages = [_msg(0, 0, 1, "a"), _msg(2, 1, 2, "b", ids=[0, 1])]
    inst = _inst([[1], [2], [3]])
    credit = cr.credit_fields(_out(3, messages, 3), inst)
    assert credit["basis"] == "trace"
    assert credit["readable"] == [1, 1, 2]
    # Counting ids instead would credit agent 1 with 2 and agent 2 with 3.
    ledger = [
        {"agent_id": 0, "round": 0, "mode": "send"},
        {"agent_id": 1, "round": 1, "mode": "send"},  # send to [] reads it
        {"agent_id": 1, "round": 2, "mode": "send"},
        {"agent_id": 0, "round": 3, "mode": "submit"},
    ]
    credit = cr.credit_fields(_out(3, messages, 3), inst, calls=ledger)
    assert credit["basis"] == "ledger" and credit["readable"] == [1, 2, 3]
    # The whole runtime ledger (``execution.ledger``) is accepted as well.
    assert cr.credit_fields(_out(3, messages, 3), inst, calls={"calls": ledger}) == credit


def test_readable_agrees_with_diagnosis_lost_messages() -> None:
    # Every lost body is exactly one un-read delivery.
    messages = [_msg(0, 0, 1), _msg(0, 0, 2), _msg(1, 2, 0), _msg(3, 1, 0)]
    out = _out(3, messages, 4)
    lost = dg._lost_messages(messages, None, 4)
    assert lost == 2  # 0->1 (1 idle at r1), 2->0 (0 idle at r2)
    credit = cr.credit_fields(out, _inst([[1], [2], [3]]))
    # agent 2 read 0->2 at r1; agent 0 read only 1->0 at the barrier (the
    # relay 2->0 carrying {0, 2} was lost); agent 1 read nothing.
    assert credit["readable"] == [2, 1, 2]


@pytest.mark.parametrize("n", [2, 5, 10])
def test_sim_basis_recovers_digest_sends(n: int) -> None:
    """Relay chain, then digest: agent n-1 reads the last relay only through
    a send to [] (no message), visible to the simulator but not to the
    trace."""

    source = es.evo_seed_source().replace(
        '    {"kind": "broadcast_last", "rounds": 1},\n',
        '    {"kind": "digest", "rounds": 1},\n', 1,
    )
    messages = [_msg(r, r, r + 1, f"k{r}") for r in range(n - 1)]
    out = _out(n, messages, n)  # relay r0..r(n-2), digest r(n-1), submit rn
    inst = _inst([[a] for a in range(n)])
    sim = cr.credit_fields(out, inst, source)
    assert sim["basis"] == "sim"
    assert sim["readable"] == list(range(1, n + 1))
    trace = cr.credit_fields(out, inst)
    assert trace["basis"] == "trace"
    assert trace["readable"][-1] == 1  # without the digest call it looked lost
    # v1 does not match this trace, so the simulated basis is rejected and
    # the trace basis is used.
    assert cr.credit_fields(out, inst, es.v1_seed_source())["basis"] == "trace"


# --------------------------------------------------------------------------- #
# raw-input retention
# --------------------------------------------------------------------------- #


def test_shard_items_flatten_answer_free_types() -> None:
    items = cr.shard_items({
        "users": {"11": [0.5, 2.0], "x": [True, None]},
        "edges": [[1, 2], [2, 3]],
        "words": ["Apple", "INFO: System started", "mkjxwupszhkmizm"],
    })
    assert ("i", 11) in items and ("f", 0.5) in items and ("i", 2) in items
    assert ("i", 1) in items and ("i", 3) in items
    assert ("s", "apple") in items and ("s", "info: system started") in items
    assert ("g", "mkjx") in items and ("g", "kmiz") in items
    assert not any(kind == "s" and value == "x" for kind, value in items)
    assert not any(value is True for _k, value in items)


def test_number_matching_precision() -> None:
    body = cr._Bodies(["v=39.30 w=39.2 z=5.0 q=15 r=-81 id_7 7a"])
    assert body.has(("f", 39.29545878664831))  # rounded to 2 decimals
    assert cr._Bodies(["39.295"]).has(("f", 39.29545878664831))
    assert cr._Bodies(["39.2"]).has(("f", 39.29545878664831))  # truncated
    assert not cr._Bodies(["39.4 3929"]).has(("f", 39.29545878664831))
    assert body.has(("i", 5)) and body.has(("i", -81)) and body.has(("i", 15))
    assert not body.has(("i", 7))  # only inside identifiers
    assert not cr._Bodies(["x=81"]).has(("i", -81))
    strings = cr._Bodies(["Words: APPLE, pineapple; fig"])
    assert strings.has(("s", "apple")) and strings.has(("s", "fig"))
    assert not strings.has(("s", "date"))


def test_retention_is_per_agent_over_other_agents_items() -> None:
    shards = [[101, 202, 303, 404], [1.25, 2.5], ["apple", "fig"]]
    messages = [
        _msg(0, 0, 1, "data=101,202 partial"),
        _msg(1, 1, 2, "agent0 101 202 303 404 agent1 1.25 2.50"),
    ]
    ledger = [
        {"agent_id": 0, "round": 0, "mode": "send"},
        {"agent_id": 1, "round": 1, "mode": "send"},
    ]
    credit = cr.credit_fields(_out(3, messages, 2), _inst(shards), calls=ledger)
    # agent 0 read nothing; agent 1 read 2 of agent 0's 4 items and none of
    # agent 2's 2 items (2 of 6); agent 2 read all 6 items of agents 0 and 1.
    assert credit["retention"] == [0.0, round(2 / 6, 3), 1.0]
    assert credit["readable"] == [1, 2, 3]


def test_retention_none_without_foreign_items() -> None:
    credit = cr.credit_fields(_out(2, [], 1), _inst([[], []]))
    assert credit["retention"] == [None, None]
    assert credit["readable"] == [1, 1]


# --------------------------------------------------------------------------- #
# whitelist + never-raise
# --------------------------------------------------------------------------- #


def test_sanitize_credit_is_a_closed_whitelist() -> None:
    clean = cr.sanitize_credit(
        {"readable": [5, 6, -1, True, "5", 2.0], "retention": [0.5, 1.2, "0.1",
         None, float("nan"), 0.12345], "basis": "sim", "answer": 42,
         "note": "leak"},
        n_agents=5,
    )
    assert clean == {"readable": [5, None, None, None, None, None],
                     "retention": [0.5, None, None, None, None, 0.123],
                     "basis": "sim"}
    assert cr.sanitize_credit("x") is None
    assert cr.sanitize_credit({"basis": "bogus"}) == {"readable": [], "retention": []}
    assert cr.sanitize_credit({"readable": list(range(200))})["readable"] == []


@pytest.mark.parametrize("output,reason", [
    (None, "no_output"), ("junk", "no_output"), ({}, "no_submit_round"),
    ({"messages": "x"}, "no_submit_round"),
    ({"submissions": [{"agent_id": 0}]}, "no_submit_round"),
])
def test_credit_never_raises_and_never_fakes_success(output: Any, reason: str) -> None:
    """A run without credit is None (never an empty dict that a coverage
    count ``credit is not None`` would mistake for success)."""

    assert cr.credit_fields(output, _inst([[1], [2]]), "not python (") is None
    row: dict[str, Any] = {"credit_error": "stale"}
    cr.attach_credit(row, output, _inst([[1], [2]]))
    assert row["credit"] is None and row["credit_error"] == reason
    assert cr.render_credit_suffix(row["credit"]) == ""


def test_internal_failure_is_reported_not_swallowed_as_empty(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise ValueError("simulated bug")

    monkeypatch.setattr(cr, "_readable_sets", boom)
    out = _out(2, [_msg(0, 0, 1, "a")], 1)
    assert cr.credit_fields(out, _inst([[1], [2]])) is None
    row: dict[str, Any] = {}
    cr.attach_credit(row, out, _inst([[1], [2]]))
    assert row["credit"] is None and row["credit_error"] == "exception"
    monkeypatch.undo()
    cr.attach_credit(row, out, _inst([[1], [2]]))
    assert row["credit"]["readable"] == [1, 2] and "credit_error" not in row
    with pytest.raises(TypeError):
        cr.attach_credit(row, out, _inst([[1], [2]]), bogus=1)


# --------------------------------------------------------------------------- #
# rendering (diagnosis-card extension)
# --------------------------------------------------------------------------- #


def _row(S, credit, fclass="divergent", bits="10000", phase_tokens=None, msgs=8):
    return {"S": S, "infra": None, "success": S >= 1.0,
            "diag": {"S": S, "failure_class": fclass, "agent_correct": bits,
                     "n_distinct_answers": 2, "majority_correct": False,
                     "answer_shape": "scalar", "submit_round": 5,
                     "phase_tokens": phase_tokens,
                     "n_messages": msgs, "n_lost_messages": 0},
            "credit": credit}


def test_card_extension_appends_credit_to_run_lines() -> None:
    credit = {"readable": [5, 5, 3, 5, 5], "retention": [0.27, 1.0, None, 0.5, 1.0],
              "basis": "sim"}
    rows = [
        _row(0.2, credit),
        "not a row",
        _row(1.0, None, "ok", "11111"),
        _row(1.0, credit, "ok", "11111"),
    ]
    text = cr.render_credit_card("III-22@x5", rows)
    lines = text.split("\n")
    assert lines[0].startswith("[III-22@x5] shape=scalar runs=3")
    assert lines[1].endswith("readable=[5 5 3 5 5]/5 raw_kept%=[27 100 - 50 100]")
    assert "readable" not in lines[2]
    # Retention is shown on failed runs only (S<1): on solved runs a low raw
    # count only means the bodies carried computed results.
    assert lines[3].endswith("readable=[5 5 3 5 5]/5")
    assert "raw_kept" not in lines[3]
    # Without credit the card is exactly the diagnosis card.
    plain = dg.render_diag_card("I-01", [_row(1.0, None, "ok", "11111")])
    assert cr.render_credit_card("I-01", [_row(1.0, None, "ok", "11111")]) == plain


def test_counts_are_rendered_leak_safe() -> None:
    """Token / message counts >= 1000 appear in thousands, credit numbers
    have <= 3 characters and no credit list is a JSON list: none of them can
    coincide with an audited ground-truth form (>= 4 chars, JSON lists)."""

    row = _row(0.4, {"readable": [5, 5, 5, 5, 5], "retention": [0.1234, 0, 1, 0.5, None]},
               phase_tokens=[1331, 402, 12500, 2210], msgs=1024)
    line = cr.render_credit_card("I-01@o5", [row]).split("\n")[1]
    assert "tokens/phase=[1.3k,402,12.5k|submit 2.2k]" in line
    assert "msgs=1.0k" in line and "1331" not in line and "1024" not in line
    assert "readable=[5 5 5 5 5]/5 raw_kept%=[12 0 100 50 -]" in line
    assert "agents_right=10000" in line  # correctness bits untouched
    audit = cr.audit_credit_text(
        line, [1331, 402, 2210, 1024, 12500, [5, 5, 5, 5, 5], 0.1234, 0.12, "12.5"]
    )
    assert audit == []


def test_real_ground_truth_vs_realistic_phase_tokens() -> None:
    """II-11@o5's ground truth holds 41 four-digit numbers in 1001..1953, so
    raw per-phase token counts in that range trip the prompt's value audit
    (LEAKAGE_GUARD) in the diagnosis rendering; the credit card renders
    counts that never collide with them."""

    if not _BENCH.is_dir():
        pytest.skip("silo-bench missing")
    values = cr.forbidden_values_of(cr.load_dev_instance("II-11", 5))
    forms: set[str] = set()
    for value in values:
        forms |= set(pm._value_forms(value, min_chars=4))
    four_digit = sorted(int(f) for f in forms if f.isdigit() and len(f) == 4)
    assert len(four_digit) >= 40 and 1000 < four_digit[0] < four_digit[-1] < 2000
    tokens = four_digit[:3] + [four_digit[-1]]
    raw = dg.render_diag_card("II-11@o5", [_row(0.6, None, phase_tokens=tokens)])
    assert cr.audit_credit_text(raw, values) != []  # the diagnosis rendering collides
    for picks in (tokens, four_digit[3:7], four_digit[-4:], [1331, 402, 2210]):
        row = _row(0.6, {"readable": [5] * 5, "retention": [0.5] * 5},
                   phase_tokens=picks, msgs=picks[0])
        text = cr.render_credit_card("II-11@o5", [row])
        assert cr.audit_credit_text(text, values) == [], picks


def test_cards_render_targets_first_within_budget() -> None:
    rows = {f"I-0{k}@o5": [_row(0.2, {"readable": [5] * 5, "retention": [0.1] * 5})]
            for k in range(1, 9)}
    text = cr.render_diag_cards_with_credit(rows, max_chars=700, order=["I-08@o5"])
    assert text.startswith("[I-08@o5]")
    assert "more units" in text and len(text) <= 760


# --------------------------------------------------------------------------- #
# executed traces (fake worker)
# --------------------------------------------------------------------------- #


def _v1_trace(tmp_path, monkeypatch, case_id: str, n: int) -> tuple[dict, Any]:
    """One fake-worker run of v1 on ``case_id`` at ``n`` agents; returns its
    ``QB_TRACE_DIR`` record and the instance."""

    from queenbee.program.budgets import PythonRunBudgets

    if not (_BENCH / f"{case_id}_n{n}.json").is_file():
        pytest.skip("silo-bench missing")
    inst = cr.load_dev_instance(case_id, n)
    traces = tmp_path / f"traces_{case_id}_{n}"
    monkeypatch.setenv("QB_TRACE_DIR", str(traces))
    pl.execute_python_source_on_case(
        source=es.v1_seed_source(), instance=inst, llm_provider="fake", worker_model="fake",
        goal="all_agents", worker_contract="message_only_v2",
        max_parallel_agents=n, request_timeout=30.0, artifacts_dir=tmp_path / "art",
        budgets=PythonRunBudgets.for_rounds(64, n_agents=n),
    )
    [path] = list(traces.glob("*.json"))
    sha12 = hashlib.sha256(es.v1_seed_source().encode("utf-8")).hexdigest()[:12]
    assert path.name.startswith(f"{case_id}_{sha12}_")
    return cr.load_trace(path), inst


@pytest.mark.parametrize("case_id,n", [("I-01", 5), ("II-11", 5), ("III-21", 5), ("II-13", 10)])
def test_v1_traces_credit_cards_and_zero_leak_hits(tmp_path, monkeypatch, case_id, n) -> None:
    trace, inst = _v1_trace(tmp_path, monkeypatch, case_id, n)
    row = cr.trace_row(trace, inst, source=es.v1_seed_source())
    assert row["credit"]["basis"] == "sim"  # simulator == trace, edge by edge
    assert row["credit"]["readable"] == [n] * n  # v1: relay chain + broadcast
    assert row["diag"]["n_lost_messages"] == 0
    assert all(v is None or 0.0 <= v <= 1.0 for v in row["credit"]["retention"])
    assert "ground_truth" not in row
    text = cr.render_credit_card(f"{case_id}@o{n}", [row])
    hits = cr.audit_credit_text(
        text, cr.forbidden_values_of(inst) + [trace.get("ground_truth")]
    )
    assert hits == [], hits
    # The shard data itself never reaches the card either.
    for shard in inst.shards:
        for item in (shard if isinstance(shard, list) else [shard])[:5]:
            form = json.dumps(item)
            if len(form) >= 5:
                assert form not in text


def test_fake_run_trace_roundtrip(tmp_path, monkeypatch) -> None:
    if not _BENCH.is_dir():
        pytest.skip("silo-bench missing")
    from queenbee.program.budgets import PythonRunBudgets

    source = es.evo_seed_source().replace(
        '    {"kind": "broadcast_last", "rounds": 1},\n',
        '    {"kind": "digest", "rounds": 1},\n    {"kind": "broadcast_last", "rounds": 1},\n',
        1,
    )
    inst = cr.load_dev_instance("III-21", 5)
    monkeypatch.setenv("QB_TRACE_DIR", str(tmp_path / "traces"))
    facts = pl.execute_python_source_on_case(
        source=source, instance=inst, llm_provider="fake", worker_model="fake",
        goal="all_agents", worker_contract="message_only_v2",
        max_parallel_agents=5, request_timeout=30.0, artifacts_dir=tmp_path,
        budgets=PythonRunBudgets.for_rounds(64, n_agents=5),
    )
    assert facts["model_calls"] == 11  # 4 relay + 1 digest + 1 broadcast + 5 submit
    [trace_path] = list((tmp_path / "traces").glob("*.json"))
    row = cr.trace_row(cr.load_trace(trace_path), inst, source=source)
    assert row["credit"]["basis"] == "sim"
    assert row["credit"]["readable"] == [5, 5, 5, 5, 5]
    # The fake worker text carries no shard item.
    assert row["credit"]["retention"] == [0.0] * 5
    assert os.environ["QB_TRACE_DIR"].endswith("traces")
