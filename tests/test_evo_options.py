"""QueenBee-Evo optional behaviours of ``queenbee.evo.loop``, offline.

Each behaviour is behind its own default-off flag; every block pins (i) the
flag-off behaviour and (ii) the flag-on behaviour on synthetic data:

* ``--econ-neutral``: an economize proposal that kept S but saved no tokens
  is ``inconclusive`` + ``econ_result: not_economized`` -- never a refuted
  mechanism in the ledger digest, the exemplars or the tabu;
* ``--final-coverage-topup K``: a narrow high scorer (< 3 T units) gets T
  reps before the final plan and becomes a candidate; the reserve keeps V
  selection for ``TOPUP_KEEP_CANDIDATES``; T only, before the plan;
  idempotent on resume;
* ``--explore-on-lockin``: mechanism families, ``Ledger.lockin``, one
  reserved explore slot with the directive (only in arms with the
  hypothesis ledger);
* ``--preserve-dups``: the API-card rule, multiset ``raw_kept%`` and
  ``dup_lost`` (no false loss on a mesh, no double count of a re-sent relay);
* the CLI flags and the run config (``config.json`` records the options
  that are on; a resume cannot add or drop one), and an end-to-end fake run
  with all four on that stays budget-matched and leak-clean.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import test_evo_loop as T
from queenbee.evo import api_card as ac
from queenbee.evo import common as P
from queenbee.evo import credit as cr
from queenbee.evo import loop as L
from queenbee.evo import memory as M
from queenbee.evo import prompt as ep
from queenbee.evo.seed import evo_seed_source

OPTIONS = {"econ_neutral": True, "final_coverage_topup": 2, "explore_on_lockin": True,
           "preserve_dups": True}

SEED = evo_seed_source()
FRONTIER = ["I-02@o5", "II-11@x5", "II-12@x5", "III-21@x5", "III-22@x5"]
GUARDS = ["I-01@o5", "I-06@o5"]
V1_S = {"I-02@o5": 0.6, "II-11@x5": 0.4, "II-12@x5": 0.2, "III-21@x5": 0.8, "III-22@x5": 0.0}
CLASS = {"I-02@o5": "divergent", "II-11@x5": "divergent", "II-12@x5": "consensus-wrong",
         "III-21@x5": "divergent", "III-22@x5": "scattered-wrong"}
SHAPE = {"I-02@o5": "scalar", "II-11@x5": "scalar", "II-12@x5": "collection",
         "III-21@x5": "segment", "III-22@x5": "scalar"}


def _prepend(entry: str, source: str = SEED) -> str:
    return source.replace("PHASES = [\n", "PHASES = [\n    " + entry + "\n", 1)


def _row(pid: str, unit: str, s: float, rep: int = 0, *, cls: str | None = None,
         shape: str = "scalar", c: float = 1000.0) -> dict[str, Any]:
    fclass = cls or ("ok" if s >= 1 else "divergent")
    return {"program_id": pid, "unit_id": unit, "S": s, "C": c, "rep": rep,
            "model_calls": 10, "prompt_tokens": 400, "completion_tokens": 600,
            "diag": {"S": s, "failure_class": fclass, "answer_shape": shape,
                     "agent_correct": "11111" if s >= 1 else "11000"}}


def _v1_archive() -> M.Archive:
    arch = M.Archive(t_templates=T.SPLIT["T"])
    arch.add_program("v1", gen=0)
    for u, s in V1_S.items():
        arch.add_rows([_row("v1", u, s, 0, cls=CLASS[u], shape=SHAPE[u]),
                       _row("v1", u, s, 1, cls=CLASS[u], shape=SHAPE[u])])
    for g in GUARDS:
        arch.add_rows([_row("v1", g, 1.0, 0), _row("v1", g, 1.0, 1)])
    return arch


@pytest.fixture(autouse=True)
def _restore_multiset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--preserve-dups`` sets a process-wide env var: restore it after
    every test (setenv records the original, then the var is removed)."""

    monkeypatch.setenv(cr.MULTISET_ENV, "0")
    monkeypatch.delenv(cr.MULTISET_ENV)


# --------------------------------------------------------------------------- #
# CLI flags and the run config (config.json)
# --------------------------------------------------------------------------- #


def _args(*extra: str) -> Any:
    return L.build_parser().parse_args(["--arm", "full", "--root", "/nonexistent/r",
                                        "--split-seed", "1", *extra])


def test_flags_default_off() -> None:
    off = L.config_from_args(_args())
    assert (off.econ_neutral, off.final_coverage_topup, off.explore_on_lockin,
            off.preserve_dups) == (False, 0, False, False)
    assert set(L.OPTION_KEYS) == set(OPTIONS)
    assert not set(L.OPTION_KEYS) & set(off.frozen())
    on = L.config_from_args(_args("--econ-neutral", "--final-coverage-topup",
                                  "--explore-on-lockin", "--preserve-dups"))
    assert (on.econ_neutral, on.final_coverage_topup, on.explore_on_lockin,
            on.preserve_dups) == (True, L.DEFAULT_TOPUP_K, True, True)
    assert {k: on.frozen()[k] for k in OPTIONS} == OPTIONS
    assert on.variant == off.variant == "full"  # options, not arms
    one = L.config_from_args(_args("--explore-on-lockin"))
    assert (one.econ_neutral, one.final_coverage_topup, one.explore_on_lockin,
            one.preserve_dups) == (False, 0, True, False)
    assert L.config_from_args(_args("--final-coverage-topup", "3")).final_coverage_topup == 3
    assert L.config_from_args(_args("--final-coverage-topup", "0")).final_coverage_topup == 0
    assert L.optional_behaviours(SimpleNamespace()) == {
        "econ_neutral": False, "final_coverage_topup": 0, "explore_on_lockin": False,
        "preserve_dups": False}
    with pytest.raises(ValueError):
        L.EvoConfig(root=Path("/x"), final_coverage_topup=-1)


def test_flags_are_frozen_and_a_resume_cannot_change_them(tmp_path: Path) -> None:
    cfg = T._cfg(tmp_path, econ_neutral=True)
    L.EvoRun(cfg, L.EvoDeps(execute=T.Scripted(), planner_client=T.CountingPlanner())).close()
    assert json.loads((cfg.root / "config.json").read_text())["econ_neutral"] is True
    with pytest.raises(L.EvoStateError, match="different config"):  # dropping an option
        L.EvoRun(T._resume(cfg, econ_neutral=False),
                 L.EvoDeps(execute=T.Scripted(), planner_client=T.CountingPlanner()))
    (tmp_path / "plain").mkdir()
    plain = T._cfg(tmp_path / "plain")
    L.EvoRun(plain, L.EvoDeps(execute=T.Scripted(), planner_client=T.CountingPlanner())).close()
    assert not set(L.OPTION_KEYS) & set(json.loads((plain.root / "config.json").read_text()))
    with pytest.raises(L.EvoStateError, match="different config"):  # adding one
        L.EvoRun(T._resume(plain, preserve_dups=True),
                 L.EvoDeps(execute=T.Scripted(), planner_client=T.CountingPlanner()))
    L.EvoRun(T._resume(plain), L.EvoDeps(execute=T.Scripted(),
                                         planner_client=T.CountingPlanner())).close()


# --------------------------------------------------------------------------- #
# --econ-neutral
# --------------------------------------------------------------------------- #


def test_economize_verdict_off_is_unchanged_on_is_neutral() -> None:
    ev = L.economize_verdict
    base = {"executed": True, "target_dS": 0.0, "guard_ok": True}
    for neutral in (False, True):
        assert ev({"executed": False}, token_ratio=None, neutral_no_saving=neutral) == "screened"
        assert ev(base, token_ratio=0.7, neutral_no_saving=neutral) == "confirmed"
        assert ev(base, token_ratio=0.9, neutral_no_saving=neutral) == "inconclusive"
        assert ev({**base, "target_dS": -0.2}, token_ratio=1.1,
                  neutral_no_saving=neutral) == "refuted"  # S lower stays refuted
        assert ev({**base, "target_dS": None}, token_ratio=1.1,
                  neutral_no_saving=neutral) == "inconclusive"
    for ratio in (1.0, 1.047):
        assert ev(base, token_ratio=ratio) == "refuted"  # the default rule
        assert ev(base, token_ratio=ratio, neutral_no_saving=True) == "inconclusive"
        assert ev({**base, "target_dS": 0.2}, token_ratio=ratio, neutral_no_saving=True) \
            == "inconclusive"  # S improved: still a cost result, not a quality claim
    assert L.econ_no_saving(base, 1.047) and not L.econ_no_saving(base, 0.9)
    assert not L.econ_no_saving({**base, "target_dS": -0.2}, 1.2)
    assert not L.econ_no_saving({**base, "executed": False}, 1.2)


def test_econ_neutral_loop_outcome_and_ledger_entry(tmp_path: Path) -> None:
    unit = "III-21@o5"
    fake = {"program_id": "e", "executed": True, "target_dS": 0.0, "guard_ok": True,
            "target_units_executed": [unit], "verdict": "refuted", "status": "executed",
            "observed": {unit: 0.0},
            "per_unit": {unit: {"C": 10470.0, "baseline_C": 10000.0, "baseline_src": "fresh"}}}
    brief = {"kind": "economize", "parent_ids": ["v1"]}
    results = {}
    for name, neutral in (("off", False), ("on", True)):
        (tmp_path / name).mkdir()
        run = T._open(tmp_path / name, econ_neutral=neutral)
        try:
            run.racer.outcome = lambda cand, gen, r2_plan=None: dict(fake)  # type: ignore[method-assign]
            out = run._outcome(4, brief, {"program_id": "e"}, None)
            entry = {"verdict": "refuted"}
            run._annotate_entry(entry, out)
            results[name] = (out, entry)
            worse = dict(fake, target_dS=-0.2, observed={unit: -0.2})
            run.racer.outcome = lambda cand, gen, r2_plan=None: dict(worse)  # type: ignore[method-assign]
            assert run._outcome(4, brief, {"program_id": "e"}, None)["verdict"] == "refuted"
        finally:
            run.close()
    off_out, off_entry = results["off"]
    assert off_out["verdict"] == "refuted" and "econ_result" not in off_out
    assert off_entry["verdict"] == "refuted" and "econ_result" not in off_entry
    on_out, on_entry = results["on"]
    assert on_out["token_ratio"] == pytest.approx(1.047)
    assert on_out["verdict"] == "inconclusive" and on_out["econ_result"] == L.ECON_NOT_ECONOMIZED
    assert on_entry == {"verdict": "inconclusive", "verdict_rule": "economize",
                        "token_ratio": pytest.approx(1.047),
                        "econ_result": L.ECON_NOT_ECONOMIZED}


def _econ_entry(pid: str, verdict: str, *, neutral: bool) -> dict[str, Any]:
    entry = {"program_id": pid, "gen": 4, "brief_kind": "economize", "verdict": verdict,
             "target_class": "divergent", "target_units": ["III-21@o5"], "parent_ids": ["p"],
             "mech_tags_host": ["add_mesh", "drop_relay", "wi_add"],
             "mech_key": "add_mesh+drop_relay+wi_add", "observed": {"III-21@o5": 0.0},
             "target_dS": 0.0}
    if neutral:
        entry["econ_result"] = L.ECON_NOT_ECONOMIZED
    return entry


def test_not_economized_is_no_refutation_in_digest_exemplars_or_tabu() -> None:
    old = [_econ_entry("g004_s3", "refuted", neutral=False)]
    new = [_econ_entry("g004_s3", "inconclusive", neutral=True)]
    old_text = ep._block_ledger({"ledger": old, "tabu": []}, set())
    new_text = ep._block_ledger({"ledger": new, "tabu": []}, set())
    # flag off: the default digest line
    assert "- divergent x add_mesh+drop_relay+wi_add: 1 refuted; mean dS=+0.00" in old_text
    assert "not economized" not in old_text
    # flag on: a neutral count, the S evidence (mean dS) kept, legend line added
    assert "- divergent x add_mesh+drop_relay+wi_add: 1 not economized; mean dS=+0.00 over " \
           "1 executed" in new_text
    assert "refuted;" not in new_text and "not economized = an economize proposal" in new_text
    # never a tabu / refutation count, never the "failed" contrastive exemplar
    led = M.Ledger(new + [_econ_entry("g005_s3", "inconclusive", neutral=True)])
    assert led.tabu() == [] and not led.refuted_counts()
    arch = _v1_archive()
    arch.add_program("g004_s3", gen=4, parent_ids=["v1"])
    arch.add_rows([_row("g004_s3", "II-11@x5", 1.0)])
    for e in old + new:
        e["target_units"] = ["II-11@x5"]
        e["observed"] = {"II-11@x5": 0.0}
    assert [x["role"] for x in arch.exemplars(old, target_class="divergent")] == ["failed"]
    assert arch.exemplars(new, target_class="divergent") == []


def test_not_economized_is_no_unsuccessful_attempt_in_the_attempt_block() -> None:
    after = {"class": "divergent", "n_agents": 5, "own_shard_only": 0, "answer_seen": 5}
    old = [_econ_entry("g004_s3", "refuted", neutral=False) | {"after_diag": after}]
    new = [_econ_entry("g004_s3", "inconclusive", neutral=True) | {"after_diag": after}]
    head = "Unsuccessful attempts and the failure LEFT AFTER the change"
    old_text = ep._block_ledger({"ledger": old, "tabu": []}, set())
    assert head in old_text and "- L1 (gen 4; targets III-21@o5; refuted)" in old_text
    new_text = ep._block_ledger({"ledger": new, "tabu": []}, set())
    assert head not in new_text and "- L1 (gen 4" not in new_text  # S evidence: digest only
    assert "1 not economized; mean dS=+0.00 over 1 executed" in new_text
    # a real failure next to it keeps its ledger entry number and its line
    both = new + [dict(old[0], program_id="g005_s1", gen=5, brief_kind="target")]
    text = ep._block_ledger({"ledger": both, "tabu": []}, set())
    assert "- L2 (gen 5; targets III-21@o5; refuted)" in text and "- L1 (gen 4" not in text


# --------------------------------------------------------------------------- #
# --final-coverage-topup K
# --------------------------------------------------------------------------- #

#: S per program on ANY unit (V and T alike) for the top-up tests.
TOPUP_S = {"wide": 0.6, "narrow": 1.0, "mid": 0.8, "flat": 0.6, "v1": 0.6}


def _topup_execute(req: dict[str, Any]) -> dict[str, Any]:
    s = TOPUP_S.get(req["job"].program_id, 0.6)
    return {"infra": None, "execution_class": "completed", "S": s, "C": 1000.0,
            "model_calls": 10, "success": s >= 1.0}


def _topup_run(tmp_path: Path, **kw: Any) -> L.EvoRun:
    """wide: 3 T units (the current candidate); narrow: II-12@o5 at 1.0 x 2
    (shrunk score 0.167); mid: III-22@o5 at 0.8 (shrunk 0.1); flat: I-02@o5
    at the census mean of the seed program (v1), so shrunk 0.0 and never
    topped up."""

    from queenbee.evo.seed import evo_seed_source
    from queenbee.program.genome import genome_region, splice_genome

    kw.setdefault("budget", 60.0)
    kw.setdefault("final_reserve", 30.0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = T._open(tmp_path, execute=_topup_execute, **kw)
    v1 = evo_seed_source()
    genome = genome_region(v1)
    rows = {"wide": [("I-01@o5", 0.8, 0), ("II-11@o5", 0.6, 0), ("III-21@o5", 0.6, 0)],
            "narrow": [("II-12@o5", 1.0, 0), ("II-12@o5", 1.0, 1)],
            "mid": [("III-22@o5", 0.8, 0)],
            "flat": [("I-02@o5", 0.6, 0)]}
    for index, (pid, items) in enumerate(rows.items()):
        prog = run.racer.register_program(pid, splice_genome(v1, L.fake_mutation(genome, index)[0]),
                                          arm="full", meta={"gen": 0, "parent_ids": ["v1"]})
        run.archive.add_program(pid, gen=0, parent_ids=["v1"], fp=prog.fp)
        run.add_rows([{"program_id": pid, "unit_id": u, "S": s, "C": 1000.0, "rep": rep}
                      for u, s, rep in items])
    return run


def test_topup_lets_a_narrow_high_scorer_become_a_candidate(tmp_path: Path) -> None:
    run = _topup_run(tmp_path, final_coverage_topup=2)
    try:
        assert run._final_candidates()[0] == ["wide"]  # narrow (1 unit) cannot compete
        rec = run.coverage_topup()
        assert rec["done"] and rec["programs"] == ["narrow", "mid"]  # flat: shrunk 0
        assert set(rec["ranked"]) == {"narrow", "mid"}
        # 2 more units each: templates the program has not run, the current
        # candidate's units first
        assert rec["units"] == {"narrow": ["I-01@o5", "II-11@o5"],
                                "mid": ["I-01@o5", "II-11@o5"]}
        records = [r for r in run.racer.log.records if r.get("purpose") == L.TOPUP_PURPOSE]
        assert len(records) == 4 and {r["batch_id"] for r in records} == {L.TOPUP_BATCH}
        assert all(run.pool.side(r["unit_id"]) == "T" for r in records)
        assert rec["after"]["narrow"]["n_T_units"] == 3 == run.archive.n_t_units("mid")
        # paid from the final reserve, never from evolution
        assert run.spent("evolution") == 0.0 and run.spent("final") == rec["cost"] == 4.0
        cands, _vetoed = run._final_candidates()
        assert cands[0] == "narrow" and set(cands) == {"wide", "narrow", "mid"}
        # the budget meter books them where spent() does: the final reserve
        phases = json.loads((run.root / "budget_meter.json").read_text())["phases"]
        assert phases["train_val"]["executions"] == 4 and phases["train_stage1"]["executions"] == 0
        T._meter_agrees(run.root)
        # idempotent: a second call (resume) runs nothing
        n = len(run.racer.log.records)
        assert run.coverage_topup()["done"] and len(run.racer.log.records) == n
    finally:
        run.close()


def test_topup_respects_k_the_bar_and_the_reserve(tmp_path: Path, monkeypatch) -> None:
    run = _topup_run(tmp_path / "k1", final_coverage_topup=1)
    try:
        assert run.coverage_topup()["programs"] == ["narrow"]
    finally:
        run.close()
    # V for TOPUP_KEEP_CANDIDATES (2 x 3 units x 2 reps) + 2 guards = 14 must stay
    for reserve, want in ((14.0, []), (17.0, ["narrow"]), (18.0, ["narrow", "mid"])):
        run = _topup_run(tmp_path / f"r{int(reserve)}", final_coverage_topup=2)
        try:
            spent = run.cfg.budget - reserve
            monkeypatch.setattr(run, "spent", lambda scope="evolution", _s=spent: _s)
            plan = run._plan_topup(2)
            assert plan["programs"] == want, reserve
            assert plan["cost"] <= plan["cap"] + 1e-9
            assert plan["cap"] == pytest.approx(reserve - 14.0)
            if reserve == 17.0:
                assert plan["skipped"] == {"mid": "reserve"}
        finally:
            run.close()


def test_final_selection_tops_up_before_the_plan_and_flag_off_never(tmp_path: Path) -> None:
    for name, k in (("on", 2), ("off", 0)):
        run = _topup_run(tmp_path / name, final_coverage_topup=k)
        try:
            final = run.final_selection()
        finally:
            run.close()
        root = tmp_path / name / "root"
        records = T._jsonl(root / "executions.jsonl")
        topups = [r for r in records if r["purpose"] == L.TOPUP_PURPOSE]
        vsel = [r for r in records if r["purpose"] == L.V_PURPOSE]
        assert vsel and all(L.template_of(r["unit_id"]) in T.SPLIT["V"] for r in vsel)
        if k:
            assert "narrow" in final["candidates"] and final["coverage_topup"]["programs"]
            assert topups and max(r["at"] for r in topups) <= final["planned_at"]
            assert all(L.template_of(r["unit_id"]) in T.SPLIT["T"] for r in topups)
            assert json.loads((root / "state" / "topup.json").read_text())["done"]
        else:
            assert final["candidates"] == ["v1", "wide"] and "coverage_topup" not in final
            assert not topups and not (root / "state" / "topup.json").exists()
        assert sum(float(r.get("exec_eq") or 1) for r in records
                   if not r.get("external") and not r.get("cached")) <= 60.0


# --------------------------------------------------------------------------- #
# --explore-on-lockin
# --------------------------------------------------------------------------- #


def test_mechanism_families_and_ledger_lockin() -> None:
    fam = M.mechanism_family
    assert fam(["wi_add"]) == fam(["wi_add", "wi_edit"]) == fam(["wi_fn"]) == "instructions"
    for tags in (["add_mesh", "wi_add"], ["drop_relay"], ["rounds_relay"], ["edit_relay"],
                 ["reorder"], ["newkind", "wi_add"], ["dispatch"], ["interp"]):
        assert fam(tags) == "topology", tags
    assert fam(["helper"]) == fam(["other_code", "cosmetic"]) == "code"
    assert fam(["none"]) is None and fam([]) is None and fam(None) is None

    def ledger(*families: str, extra: int = 0) -> M.Ledger:
        tags = {"instructions": ["wi_add"], "topology": ["add_digest", "wi_add"],
                "code": ["helper"]}
        entries = [{"verdict": "confirmed", "mech_tags_host": tags[f]} for f in families]
        entries += [{"verdict": "refuted", "mech_tags_host": ["wi_add"]}] * extra
        return M.Ledger(entries)

    assert ledger("instructions", "instructions", extra=3).lockin() is None  # < 3 confirmed
    locked = ledger(*["instructions"] * 5, "topology", "topology")
    assert locked.lockin() == {"family": "instructions", "n": 5, "of": 7,
                               "counts": {"topology": 2, "instructions": 5}}
    assert ledger("instructions", "instructions", "topology", "topology").lockin() is None
    assert ledger("topology", "topology", "topology", "code").lockin()["family"] == "topology"


def _locked_ledger() -> M.Ledger:
    return M.Ledger([{"verdict": "confirmed", "mech_tags_host": ["wi_add"], "program_id": f"p{i}"}
                     for i in range(4)])


def test_scheduler_reserves_one_explore_slot_while_locked_in() -> None:
    arch = _v1_archive()
    led = _locked_ledger()
    for gen in (0, 1, 2):
        plain = M.Scheduler(arm="full").briefs(gen, archive=arch, ledger=led,
                                               t_units=FRONTIER)
        on = M.Scheduler(arm="full")
        on.explore_on_lockin = True
        briefs = on.briefs(gen, archive=arch, ledger=led, t_units=FRONTIER)
        assert not any("lockin" in b for b in plain)
        locked = [b for b in briefs if "lockin" in b]
        assert len(locked) == 1, gen
        b = locked[0]
        assert b["kind"] == "explore" and b["lockin"] == {"family": "instructions", "n": 4, "of": 4}
        kinds = on.slot_kinds(gen)
        if "explore" in kinds:  # the rotating explore slot carries the directive
            assert b["slot"] == kinds.index("explore") and b["fallback"] is None
            assert [x["kind"] for x in briefs] == [x["kind"] for x in plain]
        else:  # else the last target slot turns explore; the rotating slot is unchanged
            assert b["slot"] == on.n_target - 1 and b["fallback"] == "lockin"
            assert briefs[-1]["kind"] == plain[-1]["kind"]
            assert sum(x["kind"] == "target" for x in briefs) == on.n_target - 1
        # the slots before the reserved one are the plain schedule's (later ones
        # keep their kind; their units may move: briefs are filled in order)
        assert briefs[: b["slot"]] == plain[: b["slot"]]
    # not locked in, or without the ledger: inert
    calm = M.Scheduler(arm="full")
    calm.explore_on_lockin = True
    assert not any("lockin" in x for x in calm.briefs(1, archive=arch, ledger=M.Ledger(),
                                                       t_units=FRONTIER))
    for flags in (L.arm_flags("mf_elite"), L.arm_flags("full", no_ledger=True),
                  L.arm_flags("full", no_ledger=True, no_archive=True)):
        s = M.Scheduler(flags)
        s.explore_on_lockin = True
        assert not any("lockin" in x for x in s.briefs(1, archive=arch, ledger=led,
                                                       t_units=FRONTIER)), flags


def test_lockin_directive_renders_with_the_ledger_and_is_logged() -> None:
    brief = {"kind": "explore", "target_units": ["II-11@x5"], "parent_ids": ["v1"],
             "lockin": {"family": "instructions", "n": 5, "of": 7}}
    full_flags = L.arm_flags("full")
    full = ep.arm_brief(brief, full_flags)
    text = ep._block_brief(ep._coerce_brief(full, full_flags), {})
    assert ("Exploration directive (host): 5 of the 7 confirmed improvements so far only "
            "rewrote work instructions on an unchanged communication pattern. This proposal "
            "must use a DIFFERENT mechanism: change the communication topology itself") in text
    plain = ep._block_brief(ep._coerce_brief({k: v for k, v in brief.items() if k != "lockin"},
                                             full_flags), {})
    assert "Exploration directive" not in plain  # the directive is the one extra line
    assert [x for x in text.split("\n") if not x.startswith("Exploration directive")] \
        == plain.split("\n")
    assert "lockin" not in ep.arm_brief(brief, L.arm_flags("mf_elite"))
    topo = dict(brief, lockin={"family": "topology", "n": 3, "of": 4})
    assert "keep the parent's communication topology" in ep._block_brief(
        ep._coerce_brief(topo, full_flags), {})
    # the ledger keeps the directive and whether the child left the dominant family
    led = M.Ledger()
    entry = led.record("c", gen=3, brief=dict(brief, brief_id="g3-s3"), parent_source=SEED,
                       child_source=_prepend('{"kind": "mesh", "rounds": 1},'))
    assert entry["lockin"] == {"family": "instructions", "n": 5, "of": 7,
                               "child_family": "topology"}
    assert "lockin" not in M.Ledger().record("d", gen=3, brief={"brief_id": "g3-s0"})


# --------------------------------------------------------------------------- #
# --preserve-dups
# --------------------------------------------------------------------------- #


def test_api_card_rule_only_with_the_flag() -> None:
    assert ac.render_api_card() == ac.API_CARD
    assert ac.render_api_card(preserve_dups=False) == ac.API_CARD
    card = ac.render_api_card(preserve_dups=True)
    assert card.count(ac.PRESERVE_DUPS_RULE) == 1 and ac.PRESERVE_DUPS_RULE not in ac.API_CARD
    anchor = "- The program must work for any n_agents and any task type.\n"
    assert card.replace(ac.PRESERVE_DUPS_RULE, "") == ac.API_CARD
    assert card.index(ac.PRESERVE_DUPS_RULE) == card.index(anchor) + len(anchor)
    caps = SimpleNamespace(max_rounds=64, max_model_calls=None, max_messages=None,
                           max_completion_tokens=None)
    assert ac.render_api_card(budgets=caps, preserve_dups=True) == card + \
        "Budget caps of this run: max_rounds=64.\n"
    assert "repeated values" in ac.PRESERVE_DUPS_RULE and "append any new items" in \
        ac.PRESERVE_DUPS_RULE


def _relay_credit() -> dict[str, Any]:
    """3 agents; agent 0 holds 49 twice; a relay 0 -> 1 -> 2 that wrote
    each value once ("append any new items" read as set semantics)."""

    shards = [[49, 49, 3], [7], [8]]
    messages = [T_msg(0, 0, 1, "49 3"), T_msg(1, 1, 2, "49 3 7")]
    calls = [{"agent_id": 0, "round": 0, "mode": "send"},
             {"agent_id": 1, "round": 1, "mode": "send"}]
    return cr.credit_fields(_out(3, messages, 2), _inst(shards), calls=calls)


def T_msg(sent: int, src: int, dst: int, body: str) -> dict[str, Any]:
    return {"round_sent": sent, "round_delivered": sent + 1, "src": src, "dst": dst,
            "source_ids": [src], "body": body}


def _out(n: int, messages: list, submit_round: int) -> dict[str, Any]:
    return {"submissions": [{"agent_id": a, "answer": 7, "submitted_round": submit_round}
                            for a in range(n)],
            "rounds_executed": submit_round + 1, "messages": list(messages),
            "usage": {"model_calls": 1, "prompt_tokens": 1, "completion_tokens": 1},
            "errors": []}


def _inst(shards: list) -> Any:
    n = len(shards)
    return SimpleNamespace(case_id="I-99", n_agents=n, shards=list(shards),
                           task_prompt="**Output:**\nA single integer.\n", ground_truth=7,
                           meta={"expected_outputs": [7] * n, "is_segmented": False})


def test_multiset_retention_shows_dropped_duplicates(monkeypatch) -> None:
    off = _relay_credit()  # distinct items: the dropped 49 is invisible
    assert off["retention"] == [0.0, round(2 / 3, 3), 1.0] and "dup_lost" not in off
    monkeypatch.setenv(cr.MULTISET_ENV, "1")
    on = _relay_credit()
    # agent 2 needs 49 x2, 3, 7 and read 49 once: 3 of 4 kept, one repeated
    # value lost; agent 1 needs 49 x2, 3, 8: 2 of 4 kept, one lost; agent 0
    # read nothing
    assert on["retention"] == [0.0, 0.5, 0.75] and on["dup_lost"] == [0, 1, 1]
    assert on["readable"] == off["readable"]
    clean = cr.sanitize_credit(on, n_agents=3)
    assert clean["dup_lost"] == [0, 1, 1]
    suffix = cr.render_credit_suffix(clean, n_agents=3)
    assert suffix.endswith("raw_kept%=[0 50 75] dup_lost=[0 1 1]")
    assert "dup_lost" not in cr.render_credit_suffix(clean, n_agents=3, show_retention=False)
    assert cr.credit_legend() == cr.CREDIT_LEGEND_MULTISET != cr.CREDIT_LEGEND
    assert "WITH multiplicity" in cr.credit_legend() and "dup_lost=[..]" in cr.credit_legend()
    monkeypatch.setenv(cr.MULTISET_ENV, "0")
    assert cr.credit_legend() == cr.CREDIT_LEGEND and _relay_credit() == off


def test_multiset_no_false_loss_on_a_mesh_and_no_double_count_of_a_resent_relay(
        monkeypatch) -> None:
    monkeypatch.setenv(cr.MULTISET_ENV, "1")
    # mesh: agents 0 and 1 both hold 5, each sends its raw shard to agent 2
    mesh = cr.credit_fields(_out(3, [T_msg(0, 0, 2, "5"), T_msg(0, 1, 2, "5")], 1),
                            _inst([[5], [5], []]),
                            calls=[{"agent_id": a, "round": 0, "mode": "send"} for a in (0, 1)])
    assert mesh["retention"][2] == 1.0 and mesh["dup_lost"][2] == 0
    # one sender re-sends its (deduplicated) list twice: still one copy of 5
    resent = cr.credit_fields(
        _out(2, [T_msg(0, 0, 1, "5 6"), T_msg(1, 0, 1, "5 6")], 2), _inst([[5, 5, 6], []]),
        calls=[{"agent_id": 0, "round": r, "mode": "send"} for r in (0, 1)]
        + [{"agent_id": 1, "round": r, "mode": "send"} for r in (1,)])
    assert resent["retention"][1] == round(2 / 3, 3) and resent["dup_lost"][1] == 1
    # the full multiset arrives: nothing lost
    full = cr.credit_fields(_out(2, [T_msg(0, 0, 1, "5, 5, 6")], 1), _inst([[5, 5, 6], []]),
                            calls=[{"agent_id": 0, "round": 0, "mode": "send"}])
    assert full["retention"][1] == 1.0 and full["dup_lost"][1] == 0
    assert cr.shard_item_counts({"names": ["ann", "Ann", "bob"], "k": [2.5, 2.5]}) == {
        ("s", "ann"): 2, ("s", "bob"): 1, ("f", 2.5): 2}


def test_multiset_pipelined_pieces_add_up_and_shared_4grams_are_no_duplicates(
        monkeypatch) -> None:
    # agent 1 forwards its own shard, then (next round) agent 2's: both pieces
    # reach agent 0 verbatim, 5 twice -- no loss (distinct pieces add up; a
    # per-body maximum would count a single 5)
    shards = [[], [5, 6], [5, 9]]
    msgs = [T_msg(0, 2, 1, "5 9"), T_msg(0, 1, 0, "5 6"), T_msg(1, 1, 0, "5 9")]
    calls = [{"agent_id": 2, "round": 0, "mode": "send"}, {"agent_id": 1, "round": 0, "mode": "send"},
             {"agent_id": 1, "round": 1, "mode": "send"}, {"agent_id": 0, "round": 1, "mode": "send"}]
    off = cr.credit_fields(_out(3, msgs, 2), _inst(shards), calls=calls)
    monkeypatch.setenv(cr.MULTISET_ENV, "1")
    on = cr.credit_fields(_out(3, msgs, 2), _inst(shards), calls=calls)
    assert off["retention"][0] == on["retention"][0] == 1.0 and on["dup_lost"][0] == 0
    # a cumulative relay (growing list) still counts once; a deduplicated union loses one 5
    need = [cr.shard_item_counts(x) for x in shards]
    assert cr._retention_multiset(0, need, {1: ["5 6", "5 6 5 9"]}) == (1.0, 0)
    assert cr._retention_multiset(0, need, {1: ["5 6", "5 6 9"]}) == (0.75, 1)
    assert cr._retention_multiset(0, need, {1: ["5 6", "5 6"]}) == (0.5, 1)  # a re-send: one 5
    # long letter strings that share a 4-gram, all relayed verbatim in one body
    dna = ["ACGTACGTTTGACCA", "GGGACGTCCCATTAG", "TTTACGTAAAGGGCC"]
    grams = [cr.shard_item_counts([x]) for x in dna] + [cr.shard_item_counts([])]
    assert cr._retention_multiset(3, grams, {0: [" ".join(dna)]}) == (1.0, 0)
    assert cr._retention_multiset(3, grams, {0: [" ".join(dna[:2])]})[1] == 0
    monkeypatch.setenv(cr.MULTISET_ENV, "0")
    assert cr.credit_fields(_out(3, msgs, 2), _inst(shards), calls=calls) == off


def test_sanitize_dup_lost_is_a_closed_whitelist() -> None:
    clean = cr.sanitize_credit({"readable": [1], "retention": [0.5],
                                "dup_lost": [3, 5000, -1, True, "2", 1.5]})
    assert clean["dup_lost"] == [3, cr.DUP_LOST_MAX, None, None, None, None]
    assert "dup_lost" not in cr.sanitize_credit({"readable": [1], "retention": [0.5]})
    assert "dup_lost" not in cr.sanitize_credit({"dup_lost": list(range(200))})


def test_preserve_dups_reaches_the_planner_prompt(tmp_path: Path) -> None:
    texts = {}
    for name, on in (("off", False), ("on", True)):
        (tmp_path / name).mkdir()
        run = T._open(tmp_path / name, preserve_dups=on)
        try:
            state = run.plan_generation(0, after="test")
            texts[name] = [(run.root / b["prompt_path"]).read_text() for b in state["briefs"]]
            assert (os.environ.get(cr.MULTISET_ENV) == "1") is on
        finally:
            run.close()
    assert all(ac.PRESERVE_DUPS_RULE in t and ac.API_CARD not in t for t in texts["on"])
    assert all(ac.PRESERVE_DUPS_RULE not in t and ac.API_CARD in t for t in texts["off"])
    for off, on in zip(texts["off"], texts["on"]):  # the card rule + the card legend only
        assert on.replace(ac.PRESERVE_DUPS_RULE, "").replace(
            cr.CREDIT_LEGEND_MULTISET, cr.CREDIT_LEGEND) == off


# --------------------------------------------------------------------------- #
# end to end: all four on, run-root leak audit
# --------------------------------------------------------------------------- #


def _assert_leak_clean(root: Path) -> None:
    """No V / TEST id in a prompt or a mint file, V units only in final
    selection and after every other paid execution, the loop's own audit clean."""

    held_out = list(T.SPLIT["V"]) + list(P.TEST_IDS)
    files = list((root / "prompts").glob("*.txt")) + list((root / "mints").glob("**/*.txt"))
    assert files
    for path in files:
        text = path.read_text()
        for tid in held_out:
            assert not re.search(r"(?<![\w-])" + re.escape(tid) + r"(?!\d)", text), (path, tid)
    records = [r for r in T._jsonl(root / "executions.jsonl") if not r.get("external")]
    v_rows = [r for r in records if L.template_of(r["unit_id"]) in T.SPLIT["V"]]
    assert all(r["purpose"] == L.V_PURPOSE for r in v_rows)
    assert not any(r["purpose"] == L.V_PURPOSE for r in records if r not in v_rows)
    if v_rows:
        assert max(r["at"] for r in records if r not in v_rows) <= min(r["at"] for r in v_rows)
    assert json.loads((root / "summary.json").read_text())["leak_hits"] == []


def test_fake_run_with_all_options_is_budget_matched_and_leak_clean(
        tmp_path: Path, monkeypatch) -> None:
    """The top-up and the lock-in really fire here: 5 T units for a final
    candidate (so the fake run's programs are under-covered and the top-up
    bar is 0) and a lock-in from the first confirmed entry on."""

    monkeypatch.setattr(L, "MIN_FINAL_UNITS", 5)
    lockin = M.Ledger.lockin
    monkeypatch.setattr(M.Ledger, "lockin",
                        lambda self, **kw: lockin(self, min_confirmed=1, share=0.5))
    results = {}
    for name, kw in (("off", {}), ("on", OPTIONS)):
        (tmp_path / name).mkdir()
        cfg, summary, _ex, _p = T._run(tmp_path / name, budget=70.0, final_reserve=24.0, **kw)
        assert summary["run_status"] == "ok" and summary["spent"]["total"] <= cfg.budget + 1e-9
        T._meter_agrees(cfg.root)
        _assert_leak_clean(cfg.root)
        results[name] = summary
    summary = results["on"]
    assert "coverage_topup" in summary and "coverage_topup" not in results["off"]
    config = json.loads((tmp_path / "on" / "root" / "config.json").read_text())
    assert {k: config[k] for k in OPTIONS} == OPTIONS
    root = tmp_path / "on" / "root"
    records = T._jsonl(root / "executions.jsonl")
    topups = [r for r in records if r["purpose"] == L.TOPUP_PURPOSE]
    assert summary["coverage_topup"]["programs"] and topups  # the top-up ran T reps ...
    assert all(L.template_of(r["unit_id"]) in T.SPLIT["T"] for r in topups)
    final = json.loads((root / "state" / "final.json").read_text())
    assert max(r["at"] for r in topups) <= final["planned_at"]  # ... before the final plan
    assert summary["spent"]["final"] == pytest.approx(sum(
        float(r.get("exec_eq") or 1) for r in records if r["purpose"] in L.FINAL_PURPOSES
        and not r.get("cached") and not r["row"].get("infra")))
    assert not [r for r in T._jsonl(tmp_path / "off" / "root" / "executions.jsonl")
                if r["purpose"] == L.TOPUP_PURPOSE]
    briefs = [b for g in T._gens(root) for b in g["briefs"]]
    assert any(b.get("lockin") for b in briefs)  # the lock-in reserved a slot ...
    assert not any(b.get("lockin") for g in T._gens(tmp_path / "off" / "root")
                   for b in g["briefs"])
    led = json.loads((root / "state" / "memory.json").read_text())
    assert any(e.get("lockin") for e in led["ledger"]["entries"])  # ... and logged it
    for e in led["ledger"]["entries"]:  # no economize refutation at an S that held
        if e.get("verdict_rule") == "economize" and (e.get("token_ratio") or 0) >= 1.0 \
                and e.get("target_dS") is not None and e["target_dS"] >= 0:
            assert e["verdict"] == "inconclusive" and e["econ_result"] == L.ECON_NOT_ECONOMIZED
