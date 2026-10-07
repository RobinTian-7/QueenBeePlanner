"""QueenBee-Evo memory (``queenbee.evo.memory``).

No network.  Covers the host-side mechanism tags (and their parity with the
prompt's ledger key), the per-unit Pareto archive (1/n margin, leaders,
confirmation, parent eligibility, shrunk T score, merge pairs, exemplars),
the ledger (verdicts, repeats, tabu + the S0 screen hook, citations), the
scheduler (identical slot schedule across arms, UCB + refutation penalty,
specialist parents, each experience component turned off), JSON
checkpoints, the prompt state feeding ``prompt.build_evo_prompt`` for every
arm, a two-generation fake loop through the real screen and the Racer with
a resume in between, and the T-harm veto of the final selection.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from queenbee.evo import memory as M
from queenbee.evo import prompt as ep
from queenbee.evo import race as R
from queenbee.evo.common import TEST_IDS
from queenbee.evo.seed import evo_seed_source

# An example T / V split of the 18 development templates.
EXAMPLE_T = ("I-01", "I-02", "I-06", "I-08", "II-11", "II-12", "II-13", "II-17",
             "III-21", "III-22", "III-26", "III-27")
EXAMPLE_V = ("I-03", "I-07", "II-16", "II-18", "III-23", "III-28")
#: Diagnosis on, ledger and archive off (``--no-ledger --no-archive``).
C1_ONLY = {"diag": True, "ledger": False, "archive": False}
SEED = evo_seed_source()


def _vocabulary() -> Any:
    """An empty work-instruction vocabulary: the screen never reads the
    benchmark files (no program screened here carries a work instruction)."""

    from queenbee.evo.screen import WIVocabulary

    return WIVocabulary(unigrams=frozenset(), bigrams=frozenset(), templates=(),
                        source_texts=())


def _t_templates() -> list[dict[str, str]]:
    """Public statements of the example T templates (synthetic text)."""

    return [{"unit_id": t, "title": f"Example task {t}",
             "output_sentence": "Each agent submits one integer.",
             "protocol_sentence": "Every agent submits with submit_result(answer)."}
            for t in EXAMPLE_T]


def _prepend(entry: str, source: str = SEED) -> str:
    return source.replace("PHASES = [\n", "PHASES = [\n    " + entry + "\n", 1)


def _row(pid: str, unit: str, s: float, rep: int = 0, *, cls: str | None = None,
         shape: str = "scalar", c: float = 1000.0) -> dict[str, Any]:
    fclass = cls or ("ok" if s >= 1 else "divergent")
    return {"program_id": pid, "unit_id": unit, "S": s, "C": c, "rep": rep,
            "model_calls": 10, "prompt_tokens": 400, "completion_tokens": 600,
            "diag": {"S": s, "failure_class": fclass, "answer_shape": shape,
                     "agent_correct": "11111" if s >= 1 else "11000"}}


FRONTIER = ["I-02@o5", "II-11@x5", "II-12@x5", "III-21@x5", "III-22@x5"]
GUARDS = ["I-01@o5", "I-06@o5"]
V1_S = {"I-02@o5": 0.6, "II-11@x5": 0.4, "II-12@x5": 0.2, "III-21@x5": 0.8, "III-22@x5": 0.0}
CLASS = {"I-02@o5": "divergent", "II-11@x5": "divergent", "II-12@x5": "consensus-wrong",
         "III-21@x5": "divergent", "III-22@x5": "scattered-wrong"}
SHAPE = {"I-02@o5": "scalar", "II-11@x5": "scalar", "II-12@x5": "collection",
         "III-21@x5": "segment", "III-22@x5": "scalar"}


def _v1_archive() -> M.Archive:
    arch = M.Archive(t_templates=EXAMPLE_T)
    arch.add_program("v1", gen=0)
    for u, s in V1_S.items():
        arch.add_rows([_row("v1", u, s, 0, cls=CLASS[u], shape=SHAPE[u]),
                       _row("v1", u, s, 1, cls=CLASS[u], shape=SHAPE[u])])
    for g in GUARDS:
        arch.add_rows([_row("v1", g, 1.0, 0), _row("v1", g, 1.0, 1)])
    return arch


def _ab_archive() -> M.Archive:
    arch = _v1_archive()
    arch.add_program("a", gen=0, parent_ids=["v1"], reached_r2=True, fp="fp-a")
    arch.add_program("b", gen=0, parent_ids=["v1"], reached_r2=True, fp="fp-b")
    arch.add_rows([_row("a", "II-11@x5", 1.0), _row("a", "II-12@x5", 0.2, cls="consensus-wrong"),
                   _row("a", "III-21@x5", 0.8), _row("a", "I-01@o5", 1.0),
                   _row("a", "I-03@o5", 0.0)])  # V unit: stored, never a T fact
    arch.add_rows([_row("b", "II-12@x5", 0.8, shape="collection"), _row("b", "II-11@x5", 0.4),
                   _row("b", "III-21@x5", 0.8)])
    return arch


# --------------------------------------------------------------------------- #
# mechanism tags
# --------------------------------------------------------------------------- #


def test_mechanism_tags_vocabulary() -> None:
    T = M.mechanism_tags
    assert T(SEED, SEED) == ["none"]
    assert T(SEED, _prepend('{"kind": "mesh", "rounds": 1},')) == ["add_mesh"]
    assert T(SEED, _prepend('{"kind": "mesh", "rounds": 1, "wi": "Restate what you read."},')) \
        == ["add_mesh", "wi_add"]
    no_bcast = SEED.replace('    {"kind": "broadcast_last", "rounds": 1},\n', "")
    assert T(SEED, no_bcast) == ["drop_broadcast_last"]
    assert T(SEED, SEED.replace('"relay", "rounds": 4, "scale_rounds_to_agents": True',
                                '"relay", "rounds": 2')) == ["rounds_relay"]
    swapped = SEED.replace(
        '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},\n'
        '    {"kind": "broadcast_last", "rounds": 1},\n',
        '    {"kind": "broadcast_last", "rounds": 1},\n'
        '    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},\n')
    assert T(SEED, swapped) == ["reorder"]
    with_wi = SEED.replace('{"kind": "broadcast_last", "rounds": 1}',
                           '{"kind": "broadcast_last", "rounds": 1, "wi": "A."}')
    assert T(SEED, with_wi) == ["wi_add"]
    assert T(with_wi, with_wi.replace('"wi": "A."', '"wi": "B."')) == ["wi_edit"]
    assert T(with_wi, SEED) == ["wi_drop"]
    edited = SEED.replace("if agent_id == local_round and agent_id + 1 < n_agents:",
                          "if agent_id <= local_round and agent_id + 1 < n_agents:")
    assert T(SEED, edited) == ["edit_relay"]
    interp = SEED.replace("    if max_rounds <= 1 or n_agents <= 1:\n        return 0\n",
                          "    if max_rounds <= 1 or n_agents <= 2:\n        return 0\n")
    assert T(SEED, interp) == ["interp"]
    new_kind = _prepend('{"kind": "echo", "rounds": 1},').replace(
        "\n\ndef phase_turn(",
        "\n\ndef phase_echo(local_round, agent_id, n_agents, selected_primary,\n"
        "               known_source_count, inbox_count):\n"
        '    return {"mode": "send", "recipients": [], "work_instruction": "Sum."}\n'
        "\n\ndef phase_turn(").replace(
        '    if kind == "digest":',
        '    if kind == "echo":\n        return phase_echo(local_round, agent_id, n_agents,\n'
        "                          selected_primary, known_source_count, inbox_count)\n"
        '    if kind == "digest":')
    assert T(SEED, new_kind) == ["add_echo", "newkind", "wi_fn"]
    assert T(SEED, SEED.replace("PHASES = [", "# comment\nPHASES = [")) == ["cosmetic"]
    assert T(SEED, SEED + "\nEXTRA = 1\n") == ["other_code"]
    broken = SEED.replace("    if max_rounds <= 1 or n_agents <= 1:", "    if max_rounds <= 1 or n_agents <= 1")
    assert T(SEED, broken) == ["unparsed_change"]
    assert "add_mesh" in T(SEED, _prepend('{"kind": "mesh", "rounds": 1},', broken))
    assert all(len(t) <= 30 for t in T(SEED, _prepend('{"kind": "' + "x" * 60 + '", "rounds": 1},')))


def test_mech_key_matches_prompt_ledger_key() -> None:
    cases = [
        (["add_mesh", "wi_add"], None),
        (["wi_add", "add_mesh", "add_mesh"], "ignored"),
        ([], "Add a Mesh round before submit!"),
        ([], None),
        (["x" * 50], None),
    ]
    for tags, claim in cases:
        entry = {"target_class": "divergent", "mech_tags_host": tags, "mech_claim_norm": claim}
        assert ep._ledger_key(entry) == ("divergent", M.mech_key(tags, claim))


def test_normalize_mechanism_is_the_prompt_rule() -> None:
    assert M.normalize_mechanism is ep.normalize_mechanism
    assert M.normalize_mechanism("Add a Mesh-round!!  before submit") == \
        "add a mesh round before submit"
    assert M.normalize_mechanism("wi_add+add_mesh") == "wi_add add_mesh"
    assert M.normalize_mechanism(None) == "" and M.normalize_mechanism(42) == "42"


def test_diff_summary_and_edit_size() -> None:
    child = _prepend('{"kind": "mesh", "rounds": 1},')
    assert "mesh" in (M.diff_summary(SEED, child) or "")
    assert M.edit_size(SEED, child) == {"added": 1, "removed": 0}


# --------------------------------------------------------------------------- #
# archive
# --------------------------------------------------------------------------- #


def test_archive_pareto_front_margin_and_leaders() -> None:
    arch = _ab_archive()
    assert not arch.is_t("I-03@o5") and "I-03@o5" not in arch.units()
    par = arch.pareto()
    units = par["units"]
    assert units["II-11@x5"]["front"] == ["a"] and units["II-11@x5"]["leader"] == "a"
    assert units["II-12@x5"]["front"] == ["b"] and units["II-12@x5"]["leader"] == "b"
    assert set(units["III-21@x5"]["front"]) == {"a", "b", "v1"}
    assert units["III-21@x5"]["leader"] is None
    assert units["I-02@o5"]["leader"] is None  # a lone program wins nothing
    assert units["II-11@x5"]["margin"] == pytest.approx(0.2)
    assert arch.wins("a") == ["II-11@x5"] and arch.wins("b") == ["II-12@x5"]
    # within one quantum is still on the front (off-grid means)
    arch.add_rows([_row("b", "II-11@x5", 1.0, r) for r in (1, 2, 3)])  # b on II-11: 0.85
    assert arch.pareto()["units"]["II-11@x5"]["front"] == ["a", "b"]
    assert arch.pareto()["units"]["II-11@x5"]["leader"] is None
    # o10: the margin is 0.1
    arch.add_rows([_row("v1", "II-11@o10", 0.6), _row("c", "II-11@o10", 0.7)])
    e = arch.pareto()["units"]["II-11@o10"]
    assert e["margin"] == pytest.approx(0.1) and e["leader"] == "c"


def test_newly_front_winning_and_confirmation() -> None:
    arch = _ab_archive()
    winners = arch.newly_front_winning()
    assert [(w["program_id"], w["unit_id"]) for w in winners] == \
        [("a", "II-11@x5"), ("b", "II-12@x5")]
    assert arch.newly_front_winning(limit=1, mark=True)[0]["program_id"] == "a"
    assert [w["program_id"] for w in arch.newly_front_winning()] == ["b"]  # a is already marked
    # parent eligibility: an unconfirmed program is a parent at most once
    assert arch.eligible_parent("a")
    arch.note_parent_use("a")
    assert not arch.eligible_parent("a") and arch.eligible_parent("v1")
    assert not arch.eligible_parent("b", {"b": 1})
    # confirmation holds (1.0 again) -> eligible again
    arch.add_rows([dict(_row("a", "II-11@x5", 1.0, 1), purpose="confirm")])
    assert arch.resolve_confirmations([{"program_id": "a", "unit_id": "II-11@x5"}]) == \
        {"a|II-11@x5": True}
    assert arch.is_confirmed("a") and arch.eligible_parent("a")
    # a failed confirmation: the replication rep (0.0) does not beat v1 (0.2)
    # by a quantum, although the merged mean (0.4) still leads the unit
    arch.add_rows([dict(_row("b", "II-12@x5", 0.0, 1), purpose="confirm")])
    assert arch.pareto()["units"]["II-12@x5"]["leader"] == "b"
    assert arch.resolve_confirmations([("b", "II-12@x5")]) == {"b|II-12@x5": False}
    assert arch.confirm_failed["b"] == ["II-12@x5"] and not arch.is_confirmed("b")


def test_shrunk_score_selection_parents_and_merges() -> None:
    arch = _ab_archive()
    # a: (1-0.4) + (0.2-0.2) + (0.8-0.8) + (1-1) over 4 units -> 0.6 / 6
    assert arch.shrunk_score("a") == pytest.approx(0.6 / 6)
    assert arch.shrunk_score("b") == pytest.approx(0.6 / 5)
    assert arch.shrunk_score("v1") == 0.0
    assert arch.top_by_shrunk(3) == ["b", "a"]
    assert arch.top_by_shrunk(3, min_units=4) == ["a"]
    assert arch.final_candidates(1) == ["b", "v1"]
    assert arch.best_incumbent() == "b"
    assert arch.best_incumbent(extra_uses={"b": 1}) == "a"
    assert arch.specialist("II-11@x5") == "a"
    assert arch.specialist("II-12@x5") == "b"
    assert arch.specialist("II-11@x5", extra_uses={"a": 1}) in ("b", "v1")
    pairs = arch.merge_pairs()
    assert pairs == [{"parent_ids": ["b", "a"],
                      "wins": {"b": ["II-12@x5"], "a": ["II-11@x5"]}}]
    assert arch.merge_pairs(exclude=[["a", "b"]]) == []
    rows = arch.best_per_unit_rows()
    # a's single 1.0 rep on II-11 is not established: the failure map keeps
    # v1's (census) rows there until a is confirmed
    assert rows["II-11@x5"][0]["S"] == 0.4 and "I-03@o5" not in rows
    arch.confirmed["a"] = ["II-11@x5"]
    assert arch.best_per_unit_rows()["II-11@x5"][0]["S"] == 1.0
    fmap = arch.failure_map(FRONTIER)
    assert fmap["units"]["III-22@x5"]["dominant"] == "scattered-wrong"
    # one program's view: its own rows, v1's census where it has none
    view = arch.best_per_unit_rows(FRONTIER, program_id="b")
    assert view["II-12@x5"][0]["S"] == 0.8 and view["III-22@x5"][0]["S"] == 0.0
    assert arch.view_rows("b", "I-02@o5")[0] == "v1"


def test_exemplars_fixed_and_failed() -> None:
    arch = _ab_archive()
    arch.add_rows([_row("c", "II-11@x5", 0.2)])
    entries = [
        {"program_id": "a", "target_class": "divergent", "verdict": "confirmed",
         "target_dS": 0.6, "observed": {"II-11@x5": 0.6, "I-03@o5": -1.0},
         "target_units": ["II-11@x5"]},
        {"program_id": "c", "target_class": "divergent", "verdict": "refuted",
         "target_dS": -0.2, "observed": {"II-11@x5": -0.2}, "target_units": ["II-11@x5"]},
        {"program_id": "b", "target_class": "consensus-wrong", "verdict": "confirmed",
         "target_dS": 0.6, "observed": {"II-12@x5": 0.6}, "target_units": ["II-12@x5"]},
    ]
    ex = arch.exemplars(entries, target_class="divergent")
    assert [(e["program_id"], e["role"]) for e in ex] == [("a", "fixed"), ("c", "failed")]
    assert ex[0]["observed"] == {"II-11@x5": 0.6}  # V units dropped
    assert [e["program_id"] for e in arch.exemplars(entries, target_class="divergent",
                                                    exclude=["a"])] == ["c"]
    by_unit = arch.exemplars(entries, unit_id="II-12@x5")
    assert [(e["program_id"], e["role"]) for e in by_unit] == [("b", "fixed")]
    assert arch.exemplars(entries) == []


def test_archive_json_roundtrip_and_guards() -> None:
    arch = _ab_archive()
    arch.note_parent_use("a")
    arch.newly_front_winning(mark=True)
    data = json.loads(json.dumps(arch.to_dict()))
    again = M.Archive.from_dict(data)
    assert again.to_dict() == arch.to_dict()
    assert again.pareto() == arch.pareto() and again.shrunk_score("a") == arch.shrunk_score("a")
    for test_id in TEST_IDS:
        with pytest.raises(R.LeakageGuardError):
            arch.add_rows([_row("a", f"{test_id}@o5", 1.0)])
    assert arch.add_rows([{"program_id": "a", "unit_id": "I-02@o5", "S": None, "infra": "x"}]) == 0


def test_confirmation_without_a_scored_rep_is_undecided_and_retried() -> None:
    # A confirmation without a scored rep (e.g. every attempt failed on
    # infra) is undecided, never "confirmed" on the lucky rep it is meant
    # to check, and is offered again until MAX_CONFIRM_ATTEMPTS is used up.
    arch = _ab_archive()
    winners = arch.newly_front_winning(mark=True, limit=1)
    assert winners[0]["program_id"] == "a"
    for attempt in range(1, M.MAX_CONFIRM_ATTEMPTS + 1):
        assert arch.resolve_confirmations(winners) == {"a|II-11@x5": None}
        assert not arch.is_confirmed("a") and "a" not in arch.confirm_failed
        again = [w["program_id"] for w in arch.newly_front_winning()]
        assert ("a" in again) == (attempt < M.MAX_CONFIRM_ATTEMPTS)
    # a scored confirmation rep later decides it (an infra row is never one)
    arch.add_rows([{"program_id": "a", "unit_id": "II-11@x5", "rep": 5, "S": None,
                    "infra": "502", "purpose": "confirm"}])
    arch.add_rows([dict(_row("a", "II-11@x5", 1.0, 6), purpose="confirm")])
    assert arch.resolve_confirmations(winners) == {"a|II-11@x5": True}
    assert arch.resolve_confirmations(winners) == {"a|II-11@x5": True}  # idempotent
    back = M.Archive.from_dict(json.loads(json.dumps(arch.to_dict())))
    assert back.confirm_undecided == arch.confirm_undecided


def test_unit_rows_are_in_numeric_rep_order() -> None:
    arch = _v1_archive()
    arch.add_rows([_row("a", "II-11@x5", 0.0, r) for r in (10, 2)])
    arch.add_rows([dict(_row("a", "II-11@x5", 1.0, 11), purpose="confirm")])
    assert [r["rep"] for r in arch.unit_rows("a", "II-11@x5")] == [2, 10, 11]


def test_archive_fails_closed_without_t_templates() -> None:
    with pytest.raises(ValueError):
        M.Archive()
    with pytest.raises(ValueError):
        M.Archive.from_dict({"v1_id": "v1", "t_templates": None})


def test_a_single_lucky_rep_does_not_close_a_unit() -> None:
    # Two unconfirmed programs each hit 1.0 once: the units stay open, so the
    # target slots keep three distinct R1 units in every arm.
    units = ["II-11@x5", "II-12@x5", "III-22@x5"]
    arch = M.Archive(t_templates=EXAMPLE_T)
    for u in units:
        arch.add_rows([_row("v1", u, s, i) for i, s in enumerate([1.0, 0.0, 1.0, 1.0, 0.0])])
    arch.add_program("x", parent_ids=["v1"])
    arch.add_rows([_row("x", "II-11@x5", 1.0)])
    arch.add_program("y", parent_ids=["v1"])
    arch.add_rows([_row("y", "II-12@x5", 1.0)])
    assert arch.robust_best("II-11@x5") == pytest.approx(0.6)
    for sched in (M.Scheduler(arm="full"), M.Scheduler(C1_ONLY), M.Scheduler(arm="mf_elite")):
        briefs = sched.briefs(3, archive=arch, ledger=M.Ledger(), t_units=units)
        assert sorted(b["target_units"][0] for b in briefs[:3]) == sorted(units), sched.flags
    # established evidence (a second rep, or a confirmation) does close it
    arch.add_rows([_row("x", "II-11@x5", 1.0, 1)])
    assert arch.robust_best("II-11@x5") == 1.0
    full = M.Scheduler(arm="full").briefs(3, archive=arch, ledger=M.Ledger(), t_units=units)
    assert "II-11@x5" not in {b["target_units"][0] for b in full[:3]}


def test_merge_pairs_from_leaders_of_disjoint_units() -> None:
    # Two specialists measured on disjoint units still form a merge pair.
    units = ["II-11@x5", "II-12@x5", "III-22@x5", "III-27@x5"]
    arch = M.Archive(t_templates=EXAMPLE_T)
    for u in units:
        arch.add_rows([_row("v1", u, s, i) for i, s in enumerate([1.0, 0.0, 1.0, 1.0, 0.0])])
    arch.add_rows([_row("x", "II-11@x5", 1.0), _row("x", "III-27@x5", 1.0)])
    arch.add_rows([_row("y", "II-12@x5", 1.0), _row("y", "III-22@x5", 1.0)])
    assert arch.merge_pairs() == [{"parent_ids": ["x", "y"],
                                   "wins": {"x": ["II-11@x5", "III-27@x5"],
                                            "y": ["II-12@x5", "III-22@x5"]}}]
    assert arch.merge_pairs(extra_uses={"y": 1}) == []  # y already used once (unconfirmed)


def test_arms_without_the_archive_see_only_the_parent_view() -> None:
    # z (not the incumbent) solved III-22 twice: the full arm sees it, the
    # arms without the archive (view = the incumbent, here v1) never do.
    arch = _v1_archive()
    arch.add_program("z", parent_ids=["v1"])
    arch.add_rows([_row("z", "III-22@x5", 1.0, r) for r in (0, 1)])
    arch.add_rows([_row("z", "II-12@x5", 0.2, 0, cls="scattered-wrong")])
    assert arch.best_incumbent() == "v1"
    full = M.Scheduler(arm="full")
    fb = M.Scheduler(C1_ONLY)
    assert full.view_program(arch) is None and fb.view_program(arch) == "v1"
    info_full = full._unit_info(arch, FRONTIER, None, None)
    info_fb = fb._unit_info(arch, FRONTIER, None, None)
    assert info_full["III-22@x5"]["closed"] and not info_fb["III-22@x5"]["closed"]
    assert info_fb["III-22@x5"]["best"] == 0.0
    fb_briefs = fb.briefs(0, archive=arch, ledger=M.Ledger(), t_units=FRONTIER)
    assert "III-22@x5" in {b["target_units"][0] for b in fb_briefs[:3]}
    assert "III-22@x5" not in {b["target_units"][0] for b in
                               full.briefs(0, archive=arch, ledger=M.Ledger(), t_units=FRONTIER)[:3]}
    # the prompt state: no other program's rows, the parent's failure map
    t_templates = _t_templates()
    st = M.prompt_state(archive=arch, ledger=M.Ledger(), flags=C1_ONLY, sources={"v1": SEED},
                        t_templates=t_templates, units=FRONTIER, parent_id="v1")
    assert set(st["rows"]) == {"v1"}
    assert st["fmap"]["units"]["II-12@x5"]["dominant"] == "consensus-wrong"  # v1's, not z's
    st2 = M.prompt_state(archive=arch, ledger=M.Ledger(), flags=C1_ONLY, sources={"v1": SEED},
                         t_templates=t_templates, units=FRONTIER)
    assert st2["fmap"] is None  # build_evo_prompt derives it from the parent's rows
    text = ep.build_evo_prompt(st2, fb_briefs[0], C1_ONLY, forbidden_case_ids=EXAMPLE_V)
    assert "=== FAILURE MAP" in text
    full_state = M.prompt_state(archive=arch, ledger=M.Ledger(), flags="full",
                                sources={"v1": SEED}, t_templates=t_templates, units=FRONTIER)
    assert "z" in full_state["rows"]


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #


MESH = _prepend('{"kind": "mesh", "rounds": 1},')
MESH2 = _prepend('{"kind": "mesh", "rounds": 2},')
HUB = _prepend('{"kind": "gather_to_hub", "rounds": 1},')


def _brief(cls: str | None = "divergent", unit: str = "II-11@x5", **kw: Any) -> dict[str, Any]:
    return {"brief_id": kw.pop("brief_id", "g0-s0"), "kind": "target", "target_units": [unit],
            "target_class": cls, "parent_ids": ["v1"], "target_shape": "scalar",
            "cell": ["scalar", cls] if cls else None, **kw}


def test_ledger_verdicts_repeats_tabu_and_screen_hook() -> None:
    from queenbee.evo.screen import screen_program

    th = R.Thresholds()
    led = M.Ledger()
    e0 = led.record("s1", gen=0, brief=_brief(), hypothesis={"mechanism": "x"},
                    screen_reasons=["coverage@n5: agents [0]"], status="screened")
    assert e0["verdict"] == "screened" and e0["mech_tags_host"] == []
    refuted = {"executed": True, "target_dS": -0.2, "observed": {"II-11@x5": -0.2}}
    e1 = led.record("p1", gen=0, brief=_brief(brief_id="g0-s1"), parent_source=SEED,
                    child_source=MESH, outcome=refuted, thresholds=th,
                    hypothesis={"failure_class": "divergent", "mechanism": "Add a mesh round"})
    assert e1["verdict"] == "refuted" and e1["mech_key"] == "add_mesh"
    assert e1["mech_claim_norm"] == "add a mesh round" and not e1["repeat_of_refuted"]
    assert e1["diff_summary"] and e1["edit_size"]["added"] == 1
    assert led.tabu() == []
    tie = led.record("p2t", gen=1, brief=_brief(brief_id="g1-s0"), parent_source=SEED,
                     child_source=MESH2, outcome={"executed": True, "target_dS": 0.0,
                                                  "observed": {"II-11@x5": 0.0}},
                     thresholds=th)
    assert tie["verdict"] == "inconclusive" and tie["repeat_of_refuted"]  # a tie refutes nothing
    led.entries.pop()
    e2 = led.record("p2", gen=1, brief=_brief(brief_id="g1-s0"), parent_source=SEED,
                    child_source=MESH2, outcome={"executed": True, "target_dS": -0.2,
                                                 "observed": {"II-11@x5": -0.2}},
                    thresholds=th)
    assert e2["verdict"] == "refuted" and e2["repeat_of_refuted"] and not e2["repeat_of_tabu"]
    assert led.tabu() == [["divergent", "add_mesh"]]
    assert led.is_tabu("divergent", "add_mesh") and not led.is_tabu("consensus-wrong", "add_mesh")
    # the screen hook: host tags of parent -> child, brief class
    hook = led.tabu_fn(SEED, target_class="divergent")
    third = _prepend('{"kind": "mesh", "rounds": 3},')
    assert hook(third, {"mechanism": "totally different words"}).startswith("tabu:")
    assert hook(HUB, None) is None
    res = screen_program(third, tabu=hook, hypothesis={"mechanism": "whatever"},
                         vocabulary=_vocabulary())
    assert not res.ok and any(r.startswith("tabu:") for r in res.reasons)
    assert screen_program(HUB, tabu=hook, vocabulary=_vocabulary()).ok
    # another class: not tabu
    assert led.tabu_fn(SEED, target_class="consensus-wrong")(third, None) is None
    # a confirmed hypothesis citing ledger entry L2 and its class
    good = {"executed": True, "target_dS": 0.4, "guard_ok": True, "reached_r2": True,
            "observed": {"II-11@x5": 0.4}}
    e3 = led.record("p3", gen=1, brief=_brief(brief_id="g1-s1"), parent_source=SEED,
                    child_source=HUB, outcome=good, thresholds=th,
                    hypothesis={"failure_class": "divergent", "predicted_dS": {"II-11@x5": 0.4},
                                "mechanism": "unlike L2 (refuted), gather to a hub first"})
    assert e3["verdict"] == "confirmed" and e3["cites_ledger"] == ["L2"] and e3["cites_class"]
    assert e3["predicted"] == {"II-11@x5": 0.4}
    # re-recording the same (program, gen) replaces the entry (resume)
    led.record("p3", gen=1, brief=_brief(brief_id="g1-s1"), parent_source=SEED,
               child_source=HUB, outcome=good, thresholds=th)
    assert [e["program_id"] for e in led.entries] == ["s1", "p1", "p2", "p3"]
    summ = led.summary()
    assert summ["verdicts"] == {"screened": 1, "refuted": 2, "confirmed": 1}
    assert summ["repeat_of_refuted"] == 1
    cells = led.cell_stats()
    assert cells["scalar|divergent"]["refuted"] == 2 and cells["scalar|divergent"]["n"] == 4
    assert M.Ledger.from_dict(json.loads(json.dumps(led.to_dict()))).entries == led.entries
    assert led.prompt_view(C1_ONLY) == {"ledger": [], "tabu": []}
    assert led.prompt_view("full")["tabu"] == [["divergent", "add_mesh"]]


def test_tabu_needs_independent_refutations_and_skips_economize() -> None:
    # Two proposals paired against ONE fresh parent rep (same parent, R1
    # unit and generation) are one piece of evidence.
    th = R.Thresholds()
    bad = {"executed": True, "target_dS": -0.2, "observed": {"II-11@x5": -0.2}}
    led = M.Ledger()
    for i in range(2):
        led.record(f"s{i}", gen=0, brief=_brief(brief_id=f"g0-s{i}"), parent_source=SEED,
                   child_source=MESH if i == 0 else MESH2, outcome=bad, thresholds=th)
    assert led.refuted_counts()[("divergent", "add_mesh")] == 1 and led.tabu() == []
    # economize refutations are not evidence against a (class, mechanism) pair
    econ = dict(_brief(None, brief_id="g1-s3"), kind="economize")
    for g in (1, 2):
        led.record(f"e{g}", gen=g, brief=dict(econ, brief_id=f"g{g}-s3"), parent_source=SEED,
                   child_source=MESH, outcome=bad, thresholds=th)
    assert led.entries[-1]["verdict"] == "refuted" and led.tabu() == []
    # an independent draw (next generation) makes it tabu
    led.record("s9", gen=1, brief=_brief(brief_id="g1-s0"), parent_source=SEED,
               child_source=MESH, outcome=bad, thresholds=th)
    assert led.tabu() == [["divergent", "add_mesh"]]
    assert led.entries[-1]["repeat_of_refuted"] and not led.entries[-1]["repeat_of_tabu"]


def test_cell_stats_count_only_real_pulls_and_summary_by_kind() -> None:
    cell = ["scalar", "divergent"]
    entries = [
        {"program_id": "m", "verdict": "screened", "status": "mint_failed", "cell": cell},
        {"program_id": "b", "verdict": "screened", "status": "not_raced_budget", "cell": cell,
         "target_dS": None},
        {"program_id": "i", "verdict": "inconclusive", "status": "infra_final", "cell": cell},
        {"program_id": "f", "verdict": "inconclusive", "status": "no_fresh_parent", "cell": cell},
        {"program_id": "h", "verdict": "confirmed", "status": "executed", "cell": cell,
         "target_dS": 0.4, "brief_kind": "target"},
        {"program_id": "g", "verdict": "inconclusive", "status": "executed", "cell": cell,
         "target_dS": 0.0, "brief_kind": "merge"},
    ]
    stats = M.Ledger(entries).cell_stats()["scalar|divergent"]
    assert stats["n"] == 3 and sorted(stats["rewards"]) == [0.0, 0.0, 0.4]
    summ = M.Ledger(entries).summary()
    assert set(summ["by_kind"]) == {"target", "merge", "unknown"}
    assert summ["by_kind"]["target"]["confirmed_rate"] == 1.0
    assert summ["by_kind"]["merge"]["hit_rate_dS_ge"] == {0.2: 0.0}


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #


ARMS = ("full", "mf_elite")
ABLATIONS = ({"diag": False}, {"ledger": False}, {"archive": False}, C1_ONLY)


def _schedulers() -> list[M.Scheduler]:
    return [M.Scheduler(arm=a) for a in ARMS] + [M.Scheduler(f) for f in ABLATIONS]


def test_slot_schedule_is_identical_in_every_arm() -> None:
    arch, led = _ab_archive(), M.Ledger()
    for gen in range(6):
        kinds = {tuple(s.slot_kinds(gen)) for s in _schedulers()}
        assert len(kinds) == 1
        assert list(kinds)[0] == ("target",) * 3 + (M.ROTATION[gen % 3],)
        for s in _schedulers():
            briefs = s.briefs(gen, archive=arch, ledger=led, t_units=FRONTIER)
            assert [b["slot_kind"] for b in briefs] == list(list(kinds)[0])
            assert [b["brief_id"] for b in briefs] == [f"g{gen}-s{i}" for i in range(4)]
            assert all(b["target_units"] and b["parent_ids"] for b in briefs)


def test_full_briefs_use_cells_specialists_and_merges() -> None:
    arch, led = _ab_archive(), M.Ledger()
    s = M.Scheduler(arm="full")
    briefs = s.briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    targets = briefs[:3]
    assert all(b["kind"] == "target" and b["target_class"] and b["cell"] for b in targets)
    r1 = [b["target_units"][0] for b in targets]
    assert len(set(r1)) == 3  # different R1 units in one generation
    for b in targets:
        unit = b["target_units"][0]
        assert b["parent_ids"] == [arch.specialist(unit)] or not arch.eligible_parent(
            arch.specialist(unit))
        assert b["cell"] == [b["target_shape"], b["target_class"]]
    # a and b are unconfirmed: the target slots spend their one parent use
    # (a's single lucky 1.0 does not close II-11), so no merge pair is left
    assert briefs[3]["kind"] == "explore" and briefs[3]["fallback"] == "no_merge_pair"
    arch.confirmed.update({"a": ["II-11@x5"], "b": ["II-12@x5"]})
    briefs = s.briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    merge = briefs[3]
    assert merge["kind"] == "merge" and merge["parent_ids"] == ["b", "a"]
    assert merge["wins"] == {"b": ["II-12@x5"], "a": ["II-11@x5"]}
    assert merge["target_units"][0] == "II-11@x5"  # a unit the second parent wins
    # gen 0 with v1 only: no merge pair -> explore, recorded
    only_v1 = s.briefs(0, archive=_v1_archive(), ledger=led, t_units=FRONTIER)
    assert only_v1[3]["kind"] == "explore" and only_v1[3]["fallback"] == "no_merge_pair"
    assert all(b["parent_ids"] == ["v1"] for b in only_v1)
    assert s.briefs(1, archive=arch, ledger=led, t_units=FRONTIER)[3]["kind"] == "economize"


def test_unconfirmed_parent_is_used_at_most_once() -> None:
    arch, led = _v1_archive(), M.Ledger()
    arch.add_program("a", parent_ids=["v1"], reached_r2=True)
    for u in FRONTIER:
        arch.add_rows([_row("a", u, 1.0)])  # a is the specialist of every unit
    s = M.Scheduler(arm="full")
    briefs = s.briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    parents = [p for b in briefs for p in b["parent_ids"]]
    assert parents.count("a") == 1 and parents.count("v1") == 3
    assert s.commit(briefs, arch) == 4 and s.commit(briefs, arch) == 0  # idempotent
    assert arch.parent_uses["a"] == 1
    later = s.briefs(1, archive=arch, ledger=led, t_units=FRONTIER)
    assert all("a" not in b["parent_ids"] for b in later)
    arch.confirmed["a"] = ["II-11@x5"]
    again = s.briefs(1, archive=arch, ledger=led, t_units=FRONTIER)
    assert sum("a" in b["parent_ids"] for b in again) >= 3


def test_ablations_degrade_each_slot_correctly() -> None:
    arch = _ab_archive()
    led = M.Ledger([{"program_id": "a", "target_class": "divergent", "verdict": "confirmed",
                     "target_dS": 0.6, "observed": {"II-11@x5": 0.6}, "gen": 0,
                     "target_units": ["II-11@x5"], "cell": ["scalar", "divergent"]}])
    # --no-diag: lowest-S units (best ESTABLISHED mean S: a's and b's single
    # reps do not count), no class / cell
    nd = M.Scheduler({"diag": False}).briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    assert [b["target_units"][0] for b in nd[:3]] == ["III-22@x5", "II-12@x5", "II-11@x5"]
    assert all(b["target_class"] is None and b["cell"] is None for b in nd)
    # the archive is still on: the specialists b (II-12) and a (II-11) are the
    # parents, and (not confirmed) cannot also be merge parents this generation
    assert nd[1]["parent_ids"] == ["b"] and nd[2]["parent_ids"] == ["a"]
    assert nd[3]["kind"] == "explore" and nd[3]["fallback"] == "no_merge_pair"
    arch.confirmed.update({"a": ["II-11@x5"], "b": ["II-12@x5"]})
    nd = M.Scheduler({"diag": False}).briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    # II-12@x5 (best 0.8: headroom 0.2 < MIN_HEADROOM_FRAC x 1.0) takes no slot of
    # its own; the 1.0-headroom unit gets the third slot again
    assert [b["target_units"][0] for b in nd[:3]] == ["III-22@x5", "I-02@o5", "III-22@x5"]
    assert nd[3]["kind"] == "merge" and nd[3]["parent_ids"] == ["b", "a"]
    arch.confirmed.clear()
    # --no-ledger --no-archive: best incumbent parent, no exemplars, merge -> explore
    fb = M.Scheduler(C1_ONLY).briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    assert fb[0]["parent_ids"] == ["b"] and all(not b["exemplars"] for b in fb)
    assert fb[3]["kind"] == "explore" and fb[3]["fallback"] == "no_archive"
    assert all(b["target_class"] for b in fb[:3])  # diagnosis still on
    # full: exemplars on the divergent class slot (a fixed it), never the parent itself
    full = M.Scheduler(arm="full").briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    div = [b for b in full[:3] if b["target_class"] == "divergent"]
    assert div and all(e["program_id"] not in b["parent_ids"]
                       for b in div for e in b["exemplars"])
    # mf_elite: no class, parent = best incumbent, targets = its lowest-S units
    el = M.Scheduler(arm="mf_elite").briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    assert all(b["target_class"] is None and not b["exemplars"] for b in el)
    assert el[0]["parent_ids"] == ["b"] and el[3]["kind"] == "explore"


def test_refutation_penalty_only_with_the_ledger() -> None:
    arch = M.Archive(t_templates=EXAMPLE_T)
    arch.add_rows([_row("v1", "II-11@x5", 0.4, cls="scattered-wrong"),
                   _row("v1", "II-12@x5", 0.4, cls="divergent")])
    fmap = {"units": {}, "cells": [
        {"shape": "scalar", "failure_class": "scattered-wrong", "units": ["II-11@x5"]},
        {"shape": "scalar", "failure_class": "divergent", "units": ["II-12@x5"]}]}
    entries = [{"program_id": f"x{i}", "verdict": "refuted", "target_dS": 0.0,
                "cell": ["scalar", "scattered-wrong"], "target_units": ["II-11@x5"]}
               for i in range(2)] + \
              [{"program_id": f"y{i}", "verdict": "screened", "target_dS": None,
                "cell": ["scalar", "divergent"], "target_units": ["II-12@x5"]} for i in range(2)]
    led = M.Ledger(entries)
    full = M.Scheduler(arm="full").briefs(2, archive=arch, ledger=led, t_units=["II-11@x5", "II-12@x5"],
                                          fmap=fmap)
    noled = M.Scheduler({"ledger": False}).briefs(2, archive=arch, ledger=led,
                                                  t_units=["II-11@x5", "II-12@x5"], fmap=fmap)
    assert full[0]["target_class"] == "divergent"
    assert noled[0]["target_class"] == "scattered-wrong"


def test_scheduler_checkpoint_roundtrip() -> None:
    arch, led = _ab_archive(), M.Ledger()
    arch.confirmed.update({"a": ["II-11@x5"], "b": ["II-12@x5"]})
    s = M.Scheduler(arm="full")
    b0 = s.briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
    s.commit(b0, arch)
    s2 = M.Scheduler.from_dict(json.loads(json.dumps(s.to_dict())))
    arch2 = M.Archive.from_dict(json.loads(json.dumps(arch.to_dict())))
    assert s2.to_dict() == s.to_dict()
    assert s2.briefs(1, archive=arch2, ledger=led, t_units=FRONTIER) == \
        s.briefs(1, archive=arch, ledger=led, t_units=FRONTIER)
    assert s.merged_pairs == [["a", "b"]]
    json.dumps(b0)  # briefs are JSON


# --------------------------------------------------------------------------- #
# prompt state -> build_evo_prompt, every arm
# --------------------------------------------------------------------------- #


def test_prompt_state_feeds_build_evo_prompt_for_every_arm() -> None:
    arch = _ab_archive()
    arch.confirmed.update({"a": ["II-11@x5"], "b": ["II-12@x5"]})  # merge parents
    sources = {"v1": SEED, "a": MESH, "b": HUB}
    led = M.Ledger()
    led.record("a", gen=0, brief=_brief(), parent_source=SEED, child_source=MESH,
               outcome={"executed": True, "target_dS": 0.6, "guard_ok": True,
                        "observed": {"II-11@x5": 0.6}}, thresholds=R.Thresholds(),
               hypothesis={"failure_class": "divergent", "mechanism": "mesh first"})
    t_templates = _t_templates()
    seen: dict[str, str] = {}
    for name, flags in (("full", "full"), ("c1_only", C1_ONLY), ("mf_elite", "mf_elite"),
                        *((f"abl{i}", a) for i, a in enumerate(ABLATIONS))):
        sched = M.Scheduler(arm=flags) if isinstance(flags, str) else M.Scheduler(flags)
        on = ep.coerce_flags(flags)
        state = M.prompt_state(archive=arch, ledger=led, flags=flags, sources=sources,
                               t_templates=t_templates, forbidden_case_ids=EXAMPLE_V,
                               units=FRONTIER)
        assert all(not u.startswith("I-03") for rows in state["rows"].values() for u in rows)
        briefs = sched.briefs(0, archive=arch, ledger=led, t_units=FRONTIER)
        for b in briefs:
            text = ep.build_evo_prompt(state, b, flags, forbidden_case_ids=EXAMPLE_V)
            seen[f"{name}:{b['slot']}"] = text
            assert ("=== FAILURE MAP" in text) == on.diag
            assert ("=== LEDGER" in text) == on.ledger
            assert ("=== CONTRASTIVE EXEMPLARS" in text) <= on.archive
    assert "=== MERGE PARENTS" in seen["full:3"]
    assert "=== MERGE PARENTS" not in seen["c1_only:3"]


# --------------------------------------------------------------------------- #
# two generations end to end (real screen, Racer, scripted worker) + resume
# --------------------------------------------------------------------------- #


class _Resolver:
    def get(self, unit_id: str):
        R.assert_dev_unit(unit_id, where="resolver")
        return SimpleNamespace(case_id=R.template_of(unit_id), n_agents=R.unit_n_agents(unit_id),
                               ground_truth=None), "sha-" + unit_id


def _execute(req: dict[str, Any]) -> dict[str, Any]:
    unit, src = req["unit_id"], req["source"]
    base = V1_S.get(unit, 1.0)
    s = min(1.0, base + 0.6) if '{"kind": "gather_to_hub"' in src else base
    return {"infra": None, "S": s, "C": 1000.0, "model_calls": 10,
            "diag": {"S": s, "failure_class": "ok" if s >= 1 else CLASS.get(unit, "divergent"),
                     "answer_shape": SHAPE.get(unit, "scalar")}}


_VARIANTS = ['{"kind": "gather_to_hub", "rounds": 1},', '{"kind": "mesh", "rounds": 1},',
             '{"kind": "one_peer_step", "rounds": 2},', '{"kind": "digest", "rounds": 1},']


def _generation(gen: int, root: Path, ckpt: dict[str, Any] | None) -> dict[str, Any]:
    from queenbee.evo.screen import known_fps_for, screen_program

    arch = M.Archive.from_dict(ckpt["archive"]) if ckpt else M.Archive(t_templates=EXAMPLE_T)
    led = M.Ledger.from_dict(ckpt["ledger"]) if ckpt else M.Ledger()
    sched = M.Scheduler.from_dict(ckpt["scheduler"]) if ckpt else M.Scheduler(arm="full")
    sources = dict(ckpt["sources"]) if ckpt else {"v1": SEED}
    racer = R.Racer(root, config=R.RaceConfig(fake=True, traces=False, parallel_cases=2,
                                              infra_backoff_s=0.0),
                    resolver=_Resolver(), execute=_execute,
                    split={"T": list(EXAMPLE_T), "V": list(EXAMPLE_V)})
    if not ckpt:
        racer.register_program("v1", SEED)
        for u in FRONTIER + GUARDS:
            racer.register_external_rows("v1", u, [_row("v1", u, V1_S.get(u, 1.0), r,
                                                        cls=CLASS.get(u), shape=SHAPE.get(u, "scalar"))
                                                   for r in range(2)])
            arch.add_rows(racer.exec_row(r) for r in racer.scored_rows("v1", u))
    briefs = sched.briefs(gen, archive=arch, ledger=led, t_units=FRONTIER)
    sched.commit(briefs, arch)
    known = known_fps_for({pid: sources[pid] for pid in sources})
    cands, screened = [], []
    for b in briefs:
        parent = b["parent_ids"][0]
        pid = f"g{gen}s{b['slot']}"
        child = _prepend(_VARIANTS[(gen + b["slot"]) % len(_VARIANTS)], sources[parent])
        hyp = {"failure_class": b["target_class"] or "divergent", "mechanism": f"variant {pid}"}
        res = screen_program(child, known_fps=known, hypothesis=hyp, vocabulary=_vocabulary(),
                             tabu=led.tabu_fn(sources[parent], target_class=b["target_class"]))
        if not res.ok:
            led.record(pid, gen=gen, brief=b, hypothesis=hyp, parent_source=sources[parent],
                       child_source=child, screen_reasons=res.reasons)
            screened.append(pid)
            continue
        sources[pid] = child
        racer.register_program(pid, child)
        arch.add_program(pid, gen=gen, parent_ids=b["parent_ids"], brief_id=b["brief_id"],
                         hypothesis=hyp, target_units=b["target_units"],
                         target_class=b["target_class"])
        cands.append({**b, "program_id": pid, "rank_penalty": res.rank_penalty,
                      "hypothesis": hyp, "child": child})

    def winners(rows: list[dict]) -> list[dict]:
        arch.add_rows(rows)
        return arch.newly_front_winning(mark=True, limit=1)

    result = racer.race_generation(gen, cands, t_frontier=FRONTIER, guards=GUARDS,
                                   shapes=SHAPE, winners_fn=winners)
    arch.add_rows(result["rows"])
    arch.resolve_confirmations((c["program_id"], c["unit_id"]) for c in
                               result["confirm"].get("confirm", {}).values())
    for c in cands:
        out = result["outcomes"][c["program_id"]]
        arch.update_outcome(out)
        led.record(c["program_id"], gen=gen, brief=c, hypothesis=c["hypothesis"],
                   parent_source=sources[c["parent_ids"][0]], child_source=c["child"],
                   outcome=out, thresholds=racer.thresholds)
    return {"archive": arch.to_dict(), "ledger": led.to_dict(),
            "scheduler": sched.to_dict(), "sources": sources,
            "racer": racer.summary(), "screened": screened}


def test_two_generation_fake_loop_with_resume(tmp_path: Path) -> None:
    root = tmp_path / "run"
    ck0 = json.loads(json.dumps(_generation(0, root, None)))
    led0 = M.Ledger.from_dict(ck0["ledger"])
    assert len(led0.entries) == 4  # 100% of proposals get a host verdict
    assert {e["verdict"] for e in led0.entries} <= set(R.VERDICTS)
    hub = [e for e in led0.entries if "add_gather_to_hub" in e["mech_tags_host"]]
    assert hub and hub[0]["verdict"] == "confirmed"
    arch0 = M.Archive.from_dict(ck0["archive"])
    assert arch0.top_by_shrunk(3)  # at least one program reached R2
    spent0 = ck0["racer"]["exec_eq_spent"]
    assert 0 < spent0 <= 13  # one generation: R1 + fresh parent reps, R2, <= 1 confirmation
    # generation 1 from the checkpoint (as after a restart)
    ck1 = _generation(1, root, ck0)
    led1 = M.Ledger.from_dict(ck1["ledger"])
    assert len(led1.entries) == 8
    assert ck1["racer"]["meter_agrees_with_log"]
    arch1 = M.Archive.from_dict(ck1["archive"])
    assert arch1.final_candidates()[-1] == "v1"
    # the restored scheduler committed gen 1's four briefs exactly once
    sched = M.Scheduler.from_dict(ck1["scheduler"])
    assert [i["gen"] for i in sched.issued].count(1) == 4


def test_final_t_harm_veto_measures_the_worst_t_drop_against_v1() -> None:
    from types import SimpleNamespace

    from queenbee.evo import loop as L

    arch = M.Archive(t_templates=EXAMPLE_T)
    arch.add_rows([_row("v1", "II-11@x5", 1.0), _row("v1", "II-11@x5", 0.8, rep=1),
                   _row("v1", "III-22@x5", 0.0),
                   _row("c", "II-11@x5", 0.2, shape="segment"), _row("c", "III-22@x5", 1.0)])
    harm = L.EvoRun._t_harm(SimpleNamespace(archive=arch), "c")
    assert harm["unit"] == "II-11@x5" and abs(harm["loss"] - 0.7) < 1e-9
    assert harm["loss"] > L.T_HARM_MAX  # vetoed at final selection
    assert L.EvoRun._t_harm(SimpleNamespace(archive=arch), "v1")["loss"] == 0.0
