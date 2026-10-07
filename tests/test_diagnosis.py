"""Answer-free diagnosis cards and the failure map (``queenbee.evo.diagnosis``).

Offline.  Pins the card keys, the closed failure-class vocabulary and its
rules, the answer shape parsed from public Output sentences (the shape counts
on the 18 T / V templates), the leak rule (no answer / ground-truth value ever
reaches a card or a rendering), the failure map, the opt-in local-only and
answer-seen refinements, and the hook in
``queenbee.program.execute.execute_python_source_on_case`` (fake provider).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import queenbee.program.execute as pl
import queenbee.program.mint as pm
from queenbee.evo import diagnosis as dg
from queenbee.paths import default_benchmarks_dir

_BENCH = default_benchmarks_dir()
# The 18 development templates (T and V sides).  TEST templates are never
# read here.
_DEV = (
    "I-01 I-02 I-03 I-06 I-07 I-08 II-11 II-12 II-13 II-16 II-17 II-18 "
    "III-21 III-22 III-23 III-26 III-27 III-28"
).split()
_CONTRACT_KEYS = {
    "S", "agent_correct", "n_distinct_answers", "majority_correct",
    "failure_class", "answer_shape", "phase_tokens", "submit_round",
    "n_messages",
}


def _instance(expected, *, task="**Output:**\nA single integer.\n", segmented=False):
    n = len(expected)
    return SimpleNamespace(
        case_id="I-99",
        n_agents=n,
        task_prompt=task,
        ground_truth=expected[0],
        meta={"expected_outputs": list(expected), "is_segmented": segmented},
    )


def _output(answers, *, messages=(), submit_round=3):
    return {
        "submissions": [
            {"agent_id": i, "answer": a, "submitted_round": submit_round}
            for i, a in enumerate(answers)
        ],
        "rounds_executed": submit_round + 1,
        "messages": list(messages),
        "usage": {"model_calls": 1, "prompt_tokens": 1, "completion_tokens": 1},
        "errors": [],
    }


def _facts(S=0.0, **extra):
    return {"S": S, "infra": None, "success": S >= 1.0, **extra}


# --------------------------------------------------------------------------- #
# answer shape (public text only)
# --------------------------------------------------------------------------- #


def test_answer_shape_counts_match_f7_on_dev_templates() -> None:
    if not _BENCH.is_dir():
        pytest.skip("silo-bench submodule missing")
    from queenbee.bench.silo_bench import SiloBenchAdapter

    adapter = SiloBenchAdapter(_BENCH)
    shapes = {
        inst.case_id: dg.instance_answer_shape(inst)
        for inst in adapter.iter_instances(cases=_DEV, agent_counts=[5])
    }
    assert set(shapes) == set(_DEV)
    collection = sorted(c for c, s in shapes.items() if s == "collection")
    segment = sorted(c for c, s in shapes.items() if s == "segment")
    scalar = [c for c, s in shapes.items() if s == "scalar"]
    assert collection == ["II-12", "II-17", "II-18", "III-27", "III-28"]
    assert segment == ["II-11", "III-21"]
    assert len(scalar) == 11


def test_answer_shape_parser_rules() -> None:
    assert dg.answer_shape_from_text("x") is None
    assert dg.answer_shape_from_text(
        "**Output:**\nThe complete difference array.\n"
    ) == "collection"
    assert dg.answer_shape_from_text(
        "**Output:**\nA single integer.\n", segmented=True
    ) == "segment"
    assert dg.output_sentence(
        "**Output:**\nA single\ninteger.\n\n**Communication Protocol:**\nX"
    ) == "A single integer."


# --------------------------------------------------------------------------- #
# failure classes
# --------------------------------------------------------------------------- #


def test_ok_card_has_every_contract_key() -> None:
    card = dg.diagnose_execution(
        _output([7, 7, 7]), _instance([7, 7, 7]), _facts(1.0)
    )
    assert _CONTRACT_KEYS <= set(card)
    assert card["failure_class"] == "ok"
    assert card["agent_correct"] == "111"
    assert card["n_distinct_answers"] == 1 and card["majority_correct"] is True
    assert card["answer_shape"] == "scalar"
    assert card["submit_round"] == 3 and card["n_messages"] == 0


@pytest.mark.parametrize(
    "answers,expected_class,bits,distinct,majority",
    [
        ([7, 7, 8], "divergent", "110", 2, True),
        ([7, 8, 8], "divergent", "100", 2, False),
        ([8, 8, 8], "consensus-wrong", "000", 1, False),
        ([8, 9, 10], "scattered-wrong", "000", 3, False),
        ([[1], [2], 9], "shape-mismatch", "000", 3, False),
        ([None, None, 7], "shape-mismatch", "001", 2, False),
    ],
)
def test_failure_classes(answers, expected_class, bits, distinct, majority) -> None:
    # S is consistent with the comparator (S = fraction of agents right): a
    # fallback verdict that disagrees with S would be dropped and the card
    # classified from S alone.
    card = dg.diagnose_execution(
        _output(answers), _instance([7, 7, 7]), _facts(bits.count("1") / 3)
    )
    assert card["failure_class"] == expected_class
    assert card["agent_correct"] == bits
    assert card["n_distinct_answers"] == distinct
    assert card["majority_correct"] is majority


def test_production_per_agent_correct_is_used_verbatim() -> None:
    card = dg.diagnose_execution(
        _output([7, 7, 7]), _instance([7, 7, 7]), _facts(2 / 3),
        per_agent_correct=[True, False, True],
    )
    assert card["agent_correct"] == "101"
    assert card["failure_class"] == "divergent"


def test_content_loss_uses_the_call_ledger() -> None:
    # agent 0 -> agent 1 at round 0 (delivered r1), but agent 1 only
    # speaks at round 2: the body was lost.  The barrier is round 3.
    messages = [
        {"round_sent": 0, "round_delivered": 1, "src": 0, "dst": 1,
         "source_ids": [0], "body": "SECRET-BODY"},
        {"round_sent": 2, "round_delivered": 3, "src": 1, "dst": 2,
         "source_ids": [0, 1], "body": "late"},
    ]
    ledger = {"calls": [
        {"agent_id": 0, "round": 0, "mode": "send", "completion_tokens": 5},
        {"agent_id": 1, "round": 2, "mode": "send", "completion_tokens": 6},
        {"agent_id": 0, "round": 3, "mode": "submit", "completion_tokens": 1},
        {"agent_id": 1, "round": 3, "mode": "submit", "completion_tokens": 1},
        {"agent_id": 2, "round": 3, "mode": "submit", "completion_tokens": 1},
    ]}
    card = dg.diagnose_execution(
        _output([7, 8, 8], messages=messages), _instance([7, 7, 7]),
        _facts(1 / 3), ledger=ledger,
    )
    assert card["n_lost_messages"] == 1  # the barrier delivery is read
    assert card["failure_class"] == "content-loss"
    # The same run is not content-loss once the recipient is called.
    ledger["calls"].append({"agent_id": 1, "round": 1, "mode": "send",
                            "completion_tokens": 2})
    card = dg.diagnose_execution(
        _output([7, 8, 8], messages=messages), _instance([7, 7, 7]),
        _facts(1 / 3), ledger=ledger,
    )
    assert card["n_lost_messages"] == 0 and card["failure_class"] == "divergent"


def test_phase_tokens_follow_the_phases_literal() -> None:
    seed = pl.seed_python_source("message_only_v2", base="sfs_phase")[2]
    ledger = {"calls": [
        {"agent_id": 0, "round": 0, "mode": "send", "completion_tokens": 10},
        {"agent_id": 3, "round": 3, "mode": "send", "completion_tokens": 20},
        {"agent_id": 4, "round": 4, "mode": "send", "completion_tokens": 40},
        {"agent_id": 0, "round": 5, "mode": "submit", "completion_tokens": 3},
    ]}
    card = dg.diagnose_execution(
        _output([1] * 5, submit_round=5), _instance([1] * 5), _facts(1.0),
        ledger=ledger, source=seed,
    )
    # relay scales to n-1 = 4 rounds, broadcast_last 1 round, then submit.
    assert card["phase_tokens"] == [30, 40, 3]


@pytest.mark.parametrize(
    "facts,expected_class,err",
    [
        ({"infra": "ConnectionError: x"}, "infra", "infra"),
        (_facts(0.0, error="RuntimeError: BudgetError: exhausted"), "budget", "budget"),
        (_facts(0.0, error="RuntimeError: AnswerFormatError: bad"), "format", "format"),
        (_facts(0.0, error="RuntimeError: DataFlowError: x"), "format", "exec"),
    ],
)
def test_error_rows(facts, expected_class, err) -> None:
    card = dg.diagnose_execution(None, _instance([7, 7]), facts)
    assert card["failure_class"] == expected_class
    assert card["error_class"] == err
    assert card["agent_correct"] is None


def test_segment_tasks_have_no_consensus_notion() -> None:
    inst = _instance([[1, 2], [3, 4]], task="**Output:**\nEach agent submits "
                     "THEIR portion of the array.\n", segmented=True)
    card = dg.diagnose_execution(_output([[1, 2], [4, 3]]), inst, _facts(0.5))
    assert card["answer_shape"] == "segment"
    assert card["n_distinct_answers"] is None
    assert card["failure_class"] == "divergent"
    card = dg.diagnose_execution(_output([[9], [9]]), inst, _facts(0.0))
    assert card["failure_class"] == "scattered-wrong"


def test_diagnose_never_raises() -> None:
    card = dg.diagnose_execution(object(), SimpleNamespace(), {"S": "bad"})
    assert card["failure_class"] is None and card["S"] is None


# --------------------------------------------------------------------------- #
# leak rule
# --------------------------------------------------------------------------- #


def test_cards_and_renderings_never_carry_answers_or_truth() -> None:
    truth = 918273645
    wrong = 564738291
    inst = _instance([truth] * 3)
    card = dg.diagnose_execution(_output([truth, wrong, wrong]), inst, _facts(1 / 3))
    rows = {"I-99": [{"S": 1 / 3, "diag": card}]}
    blobs = [
        json.dumps(card),
        dg.render_diag_card("I-99", rows["I-99"]),
        dg.render_failure_map(dg.build_failure_map(rows)),
    ]
    for blob in blobs:
        assert str(truth) not in blob and str(wrong) not in blob
    assert pm.audit_text_leaks(
        "\n".join(blobs), forbidden_values=[truth, wrong]
    ) == []


def test_sanitize_diag_is_a_closed_whitelist() -> None:
    dirty = {
        "S": 0.5, "agent_correct": "10;DROP", "n_distinct_answers": -1,
        "majority_correct": "yes", "failure_class": "the answer is 42",
        "answer_shape": "scalar", "phase_tokens": [1, "x"], "submit_round": 3,
        "n_messages": 2, "answer": 42, "ground_truth": 42,
        "error_class": "exec",
    }
    clean = dg.sanitize_diag(dirty)
    assert clean["agent_correct"] is None
    assert clean["n_distinct_answers"] is None
    assert clean["majority_correct"] is None
    assert clean["failure_class"] is None
    assert clean["phase_tokens"] is None
    assert "answer" not in clean and "ground_truth" not in clean
    assert clean["error_class"] == "exec"
    assert dg.sanitize_diag("nope") is None


# --------------------------------------------------------------------------- #
# failure map + cards
# --------------------------------------------------------------------------- #


def _card(fc, S, shape="scalar", bits="10"):
    return {"S": S, "diag": {"S": S, "failure_class": fc, "answer_shape": shape,
                             "agent_correct": bits}}


def test_failure_map_groups_by_shape_and_dominant_class() -> None:
    rows = {
        "I-01": [_card("ok", 1.0, bits="11"), _card("ok", 1.0, bits="11")],
        "II-13": [_card("divergent", 0.5), _card("divergent", 0.5)],
        "III-22": [_card("divergent", 0.5), _card("consensus-wrong", 0.0, bits="00")],
        "II-17": [_card("consensus-wrong", 0.0, "collection", "00"),
                  {"infra": "ConnectionError: x"}],
        "III-21": [{"S": 0.0, "error": "RuntimeError: BudgetError: x"}],
    }
    fmap = dg.build_failure_map(rows, attempts={"II-13": 2}, refuted={"II-13": ["vote-merge"]})
    assert fmap["solved"] == ["I-01"]
    units = fmap["units"]
    assert units["III-22"]["dominant"] == "divergent"  # tie -> class priority
    assert units["II-17"]["n_infra"] == 1 and units["II-17"]["S_mean"] == 0.0
    assert units["III-21"]["dominant"] == "budget"
    assert units["II-13"]["attempts"] == 2 and units["II-13"]["refuted"] == ["vote-merge"]
    cells = {(c["shape"], c["failure_class"]): c["units"] for c in fmap["cells"]}
    assert cells[("scalar", "divergent")] == ["II-13", "III-22"]
    assert cells[("collection", "consensus-wrong")] == ["II-17"]
    text = dg.render_failure_map(fmap)
    assert text.startswith("=== FAILURE MAP")
    assert "scalar x divergent: II-13 (S=0.50; runs=2; attempted 2x; refuted: vote-merge)" in text
    assert "Solved by the incumbent on every run: I-01" in text
    assert "divergent: some agents right" in text  # glossary of seen classes


def test_failure_map_rendering_is_bounded_by_whole_lines() -> None:
    rows = {f"I-{i:02d}": [_card("divergent", 0.5)] for i in range(60)}
    rows.update({f"II-{i:02d}": [_card("consensus-wrong", 0.0, "collection")] for i in range(60)})
    text = dg.render_failure_map(dg.build_failure_map(rows), max_chars=400)
    assert len(text) <= 400
    assert all(line.startswith(("===", "Each", "Failure", "  ", "- ", "...", "Solved"))
               for line in text.splitlines())


def test_render_diag_card_reads_rows_or_bare_cards() -> None:
    card = {"S": 0.4, "failure_class": "divergent", "answer_shape": "segment",
            "agent_correct": "11000", "phase_tokens": [100, 20, 30],
            "submit_round": 5, "n_messages": 8, "n_lost_messages": 2,
            "majority_correct": False}
    from_row = dg.render_diag_card("III-21", [{"S": 0.4, "diag": card}])
    from_card = dg.render_diag_card("III-21", [card])
    assert from_row == from_card
    assert from_row.splitlines()[0].startswith("[III-21] shape=segment runs=1 mean S=0.40")
    assert "agents_right=11000" in from_row and "lost_bodies=2" in from_row
    assert "tokens/phase=[100,20|submit 30]" in from_row
    assert dg.render_diag_card("X", []) == "[X] no diagnosed runs"


# --------------------------------------------------------------------------- #
# hook in execute_python_source_on_case (fake provider, dev template I-01)
# --------------------------------------------------------------------------- #


def test_execute_attaches_diag_with_fake_provider() -> None:
    if not _BENCH.is_dir():
        pytest.skip("silo-bench submodule missing")
    from queenbee.bench.silo_bench import SiloBenchAdapter
    from queenbee.program.budgets import PythonRunBudgets

    inst = next(SiloBenchAdapter(_BENCH).iter_instances(cases=["I-01"], agent_counts=[5]))
    seed = pl.seed_python_source("message_only_v2", base="sfs_phase")[2]
    facts = pl.execute_python_source_on_case(
        source=seed, instance=inst, llm_provider="fake", worker_model="fake",
        goal="all_agents", worker_contract="message_only_v2",
        max_parallel_agents=5, request_timeout=30.0,
        artifacts_dir=Path(tempfile.mkdtemp()),
        budgets=PythonRunBudgets.for_rounds(64, n_agents=5),
    )
    card = facts["diag"]
    assert _CONTRACT_KEYS <= set(card)
    assert card["answer_shape"] == "scalar"
    assert card["n_messages"] == facts["n_messages"] == 8
    assert card["submit_round"] == 5 and card["n_lost_messages"] == 0
    assert card["failure_class"] in dg.FAILURE_CLASSES
    assert len(card["phase_tokens"]) == 3
    assert card["S"] == facts["S"]
    assert str(inst.ground_truth) not in json.dumps(card)


#: An II-16 (trapped rain water) instance: each agent's own-shard answer
#: and the gold answer of the whole array.
_II16_SHARDS = [[2, 0, 2], [3, 0, 0, 3], [1, 0, 1, 0, 1], [4, 1, 4], [2, 1, 3]]
_II16_OWN = [2, 6, 2, 3, 1]
_II16_GOLD = 26


def _ii16_instance():
    return SimpleNamespace(
        case_id="II-16", n_agents=5, shards=[list(s) for s in _II16_SHARDS],
        task_prompt="**Output:**\nA single integer: the total trapped water.\n",
        ground_truth=_II16_GOLD,
        meta={"expected_outputs": [_II16_GOLD] * 5, "is_segmented": False},
    )


def test_local_only_is_off_by_default_and_cards_stay_unchanged(monkeypatch) -> None:
    monkeypatch.delenv(dg.LOCAL_ONLY_ENV, raising=False)
    inst = _ii16_instance()
    # every agent answered from its own shard alone
    card = dg.diagnose_execution(_output(_II16_OWN), inst, {"S": 0.0},
                                 per_agent_correct=[False] * 5)
    assert card["failure_class"] == "scattered-wrong"
    assert "n_local_only" not in card


def test_local_only_flags_own_shard_answers_without_leaking(monkeypatch) -> None:
    monkeypatch.setenv(dg.LOCAL_ONLY_ENV, "1")
    inst = _ii16_instance()
    card = dg.diagnose_execution(_output(_II16_OWN), inst, {"S": 0.0},
                                 per_agent_correct=[False] * 5)
    assert card["failure_class"] == "local-only" and card["n_local_only"] == 5
    clean = dg.sanitize_diag(card)
    assert clean["n_local_only"] == 5 and clean["failure_class"] == "local-only"
    text = dg.render_diag_card("II-16@o5", [clean])
    assert "own_shard_only_agents=5" in text and str(_II16_GOLD) not in text
    # a wrong answer that is NOT the own-shard answer stays generic
    other = dg.diagnose_execution(_output([7, 8, 9, 10, 11]), inst, {"S": 0.0},
                                  per_agent_correct=[False] * 5)
    assert other["failure_class"] == "scattered-wrong" and other["n_local_only"] == 0
    # a stored trace card is refined the same way (reads only the bits)
    stored = {"failure_class": "scattered-wrong", "agent_correct": "00000", "S": 0.0}
    assert dg.apply_local_only(stored, _output(_II16_OWN), inst)["failure_class"] == "local-only"


def test_local_only_never_touches_test_templates(monkeypatch) -> None:
    monkeypatch.setenv(dg.LOCAL_ONLY_ENV, "1")
    inst = _instance([5] * 5)  # I-99 is not a dev template: refused, no flag
    card = dg.diagnose_execution(_output([1, 2, 3, 4, 6]), inst, {"S": 0.0})
    assert card["failure_class"] == "scattered-wrong" and "n_local_only" not in card


def _hub_broadcast_output(answers, body):
    msgs = [{"round_sent": 1, "round_delivered": 2, "src": 0, "dst": d,
             "source_ids": [0, 1, 2, 3, 4], "body": body} for d in range(1, 5)]
    return _output(answers, messages=msgs, submit_round=2)


_TOTAL = '{"total_trapped_water":%d}' % _II16_GOLD


def test_answer_seen_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv(dg.ATTEMPT_ENV, raising=False)
    card = dg.diagnose_execution(_hub_broadcast_output(_II16_OWN, _TOTAL),
                                 _ii16_instance(), {"S": 0.0}, per_agent_correct=[False] * 5)
    assert "n_answer_seen" not in card


def test_answer_seen_counts_agents_that_held_the_right_answer(monkeypatch) -> None:
    monkeypatch.setenv(dg.ATTEMPT_ENV, "1")
    inst = _ii16_instance()
    # the hub broadcast the right total, every agent (the hub included)
    # still submitted its own-shard answer
    card = dg.diagnose_execution(_hub_broadcast_output(_II16_OWN, _TOTAL),
                                 inst, {"S": 0.0}, per_agent_correct=[False] * 5)
    assert card["n_answer_seen"] == 5
    text = dg.render_diag_card("II-16@o5", [dg.sanitize_diag(card)])
    assert "answer_seen_agents=5" in text and str(_II16_GOLD) not in text
    # a near number is not the answer (token boundary)
    near = f"total {_II16_GOLD}0 or 1{_II16_GOLD}.5"
    card = dg.diagnose_execution(_hub_broadcast_output(_II16_OWN, near),
                                 inst, {"S": 0.0}, per_agent_correct=[False] * 5)
    assert card.get("n_answer_seen", 0) == 0


def test_answer_seen_skips_values_already_in_the_data(monkeypatch) -> None:
    monkeypatch.setenv(dg.ATTEMPT_ENV, "1")
    gold = 4  # connected components; the node ids hold a 4 too
    inst = SimpleNamespace(
        case_id="III-23", n_agents=5,
        shards=[[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[8, 9], [1, 2]]],
        task_prompt="**Output:**\nA single integer: the number of components.\n",
        ground_truth=gold, meta={"expected_outputs": [gold] * 5, "is_segmented": False},
    )
    card = dg.diagnose_execution(_hub_broadcast_output([8, 6, 7, 6, 6], f"components: {gold}"),
                                 inst, {"S": 0.0}, per_agent_correct=[False] * 5)
    assert card["answer_shape"] == "scalar"
    assert "n_answer_seen" not in card
