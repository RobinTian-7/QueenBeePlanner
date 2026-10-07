"""QueenBee-Evo racing (``queenbee.evo.race``).

No network: a scripted executor, a fake instance resolver and explicit
fingerprints (one test uses the real behaviour fingerprints of the two
source texts of the seed program (v1): the base text the census runs and
the extended text evolution starts from).
Covers the TEST / split guards, rep allocation (fresh reps never cached),
the exec cache (across and inside batches, only under the same worker
configuration), resume after a kill in the middle of a batch (no execution
repeated, meter == log), infra re-runs, the budget cap, o10
execution-equivalents, R1 / R2 / confirmation staging and paired dS vs the
fresh parent, the guard (and the vacuous guard), host verdicts, diag +
credit fields, the per-batch trace directory and thresholds.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from queenbee.evo import race as R
from queenbee.evo.common import TEST_IDS
from queenbee.evo.seed import evo_seed_source, v1_seed_source

# An example T / V split of the 18 development templates.
EXAMPLE_T = ("I-01", "I-02", "I-06", "I-08", "II-11", "II-12", "II-13", "II-17",
             "III-21", "III-22", "III-26", "III-27")
EXAMPLE_V = ("I-03", "I-07", "II-16", "II-18", "III-23", "III-28")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class FakeResolver:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, unit_id: str):
        R.assert_dev_unit(unit_id, where="fake resolver")
        self.calls += 1
        n = R.unit_n_agents(unit_id)
        inst = SimpleNamespace(case_id=R.template_of(unit_id), n_agents=n,
                               ground_truth={"answer": unit_id})
        return inst, "sha-" + unit_id


class Script:
    """execute(req) -> row from ``table[(program, unit)]`` (a float, or a
    list indexed by the job's rep, its last entry for higher reps; else
    ``default``); ``infra[key]`` = number of infra rows first; ``fmt`` keys
    give format failures; ``kill_after`` raises SystemExit once that many
    rows were scored."""

    def __init__(self, table: dict | None = None, *, default: float = 1.0,
                 infra: dict[str, int] | None = None, kill_after: int | None = None,
                 fmt: set[str] | None = None, cls: dict[str, str] | None = None,
                 shape: dict[str, str] | None = None) -> None:
        self.table = dict(table or {})
        self.default = default
        self.infra = dict(infra or {})
        self.kill_after = kill_after
        self.fmt = set(fmt or ())
        self.cls = dict(cls or {})
        self.shape = dict(shape or {})
        self.calls: list[str] = []
        self.ok_calls: list[str] = []

    def __call__(self, req: dict[str, Any]) -> dict[str, Any]:
        job = req["job"]
        if self.kill_after is not None and len(self.ok_calls) >= self.kill_after:
            raise SystemExit("killed")
        self.calls.append(job.key)
        if self.infra.get(job.key, 0) > 0:
            self.infra[job.key] -= 1
            return {"infra": "ConnectionError: 502 bad gateway"}
        self.ok_calls.append(job.key)
        value = self.table.get((job.program_id, job.unit_id), self.default)
        if isinstance(value, list):
            value = value[min(int(job.rep), len(value) - 1)]
        s = float(value)
        fclass = "format" if job.key in self.fmt else self.cls.get(
            job.unit_id, "ok" if s >= 1 else "divergent")
        row = {"infra": None, "execution_class": "completed", "success": s >= 1,
               "S": s, "C": 1000.0 + 10 * len(self.ok_calls), "model_calls": 10,
               "prompt_tokens": 400, "completion_tokens": 600,
               "diag": {"S": s, "failure_class": fclass,
                        "answer_shape": self.shape.get(job.unit_id, "scalar"),
                        "agent_correct": "11111" if s >= 1 else "11000",
                        "secret_answer": 42}}
        if fclass == "format":
            row["error"] = "AnswerFormatError: bad"
            row["S"] = 0.0
            row["diag"]["S"] = 0.0
        return row


def _racer(tmp_path: Path, script: Script, *, split: Any = None, **cfg: Any) -> R.Racer:
    cfg.setdefault("parallel_cases", 1)
    cfg.setdefault("infra_backoff_s", 0.0)
    cfg.setdefault("traces", False)
    return R.Racer(tmp_path / "race", config=R.RaceConfig(fake=True, **cfg),
                   resolver=FakeResolver(), execute=script, split=split)


def _programs(racer: R.Racer, *names: str, fp_of: dict[str, str] | None = None) -> None:
    fp_of = fp_of or {}
    for name in ("v1",) + names:
        racer.register_program(name, f"# program {name}\n", fp=fp_of.get(name, "fp-" + name))


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("test_id", TEST_IDS)
def test_test_units_are_refused_before_anything_runs(test_id: str, tmp_path: Path) -> None:
    script = Script()
    racer = _racer(tmp_path, script)
    _programs(racer, "c1")
    for unit in (f"{test_id}@o5", f"{test_id}@x5", f"{test_id}@o10"):
        with pytest.raises(R.LeakageGuardError):
            R.assert_dev_unit(unit, where="t")
        with pytest.raises(R.LeakageGuardError):
            racer.plan_batch("g0:r1", [("c1", unit, "r1", None)])
        with pytest.raises(R.LeakageGuardError):
            racer.run([R.Job("c1", unit, 0, "r1", "g0:r1")])
        with pytest.raises(R.LeakageGuardError):
            racer.register_external_rows("v1", unit, [{"S": 1.0}])
    assert script.calls == []
    assert racer.plan("g0:r1") is None


def test_split_sides_are_enforced(tmp_path: Path) -> None:
    split = {"T": list(EXAMPLE_T), "V": list(EXAMPLE_V)}
    racer = _racer(tmp_path, Script(), split=split)
    _programs(racer, "c1")
    with pytest.raises(R.LeakageGuardError):
        racer.plan_batch("g0:r1", [("c1", "I-03@o5", "r1", None)])  # V unit in R1
    with pytest.raises(R.LeakageGuardError):
        racer.plan_batch("g0:vsel", [("c1", "I-01@o5", "vselect", None)])  # T unit in V-select
    jobs = racer.plan_batch("g9:vsel", [("c1", "I-03@o5", "vselect", None)])
    assert jobs[0].purpose == "vselect"
    assert racer.plan_batch("census", [("v1", "I-03@x5", "census", None),
                                       ("v1", "I-01@x5", "census", None)])
    with pytest.raises(R.RaceError):
        racer.plan_batch("g0:x", [("c1", "I-01@o5", "test", None)])
    with pytest.raises(R.RaceError):
        R.assert_dev_unit("I-01", where="bare template")


def test_job_key_and_coercion() -> None:
    job = R.Job("c1", "II-11@x5", 2, "r1", "g3:r1")
    assert job.key == "c1|II-11@x5|r2"
    assert R.coerce_job(job.to_dict()) == job
    duck = SimpleNamespace(program_id="c1", unit_id="II-11@x5", rep=2, purpose="r1")
    assert R.coerce_job(duck).key == job.key
    assert R.gen_of_batch("g3:r1") == 3 and R.gen_of_batch("census") is None
    assert R.exec_eq("II-11@o10") == 2 and R.exec_eq("II-11@x5") == 1


def test_racer_needs_a_resolver(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        R.Racer(tmp_path / "race", config=R.RaceConfig(fake=True))  # type: ignore[call-arg]
    with pytest.raises(R.RaceError, match="resolver"):
        R.Racer(tmp_path / "race", config=R.RaceConfig(fake=True), resolver=None)
    assert not (tmp_path / "race").exists()


# --------------------------------------------------------------------------- #
# reps, cache, resume
# --------------------------------------------------------------------------- #


def test_fresh_reps_are_new_indices_and_never_cached(tmp_path: Path) -> None:
    script = Script({("v1", "II-11@x5"): 0.4})
    racer = _racer(tmp_path, script)
    _programs(racer, "c1")
    added = racer.register_external_rows(
        "v1", "II-11@x5", [{"S": 0.4, "C": 9.0}, {"S": 0.6, "C": 9.0}, {"S": 0.4, "C": 9.0}])
    assert added == 3
    assert racer.register_external_rows("v1", "II-11@x5", [{"S": 0.4}] * 3) == 0  # idempotent
    jobs = racer.plan_batch("g0:r1", [("c1", "II-11@x5", "r1", None),
                                      ("v1", "II-11@x5", "parent_fresh", None)])
    assert [(j.program_id, j.rep) for j in jobs] == [("c1", 0), ("v1", 3)]
    rows = racer.run(jobs)
    assert sorted(script.calls) == ["c1|II-11@x5|r0", "v1|II-11@x5|r3"]
    assert all(not r["cached"] for r in rows)
    # next generation: another fresh rep, index 4, executed (not the cache)
    jobs2 = racer.plan_batch("g1:r1", [("v1", "II-11@x5", "parent_fresh", None)])
    assert jobs2[0].rep == 4
    racer.run(jobs2)
    assert script.calls[-1] == "v1|II-11@x5|r4"
    # external rows cost nothing
    assert racer.exec_eq_spent() == 3
    assert racer.log.summary()["external_rows"] == 3


def test_exec_cache_across_and_inside_batches(tmp_path: Path) -> None:
    script = Script({("v1", "I-02@o5"): 0.6})
    racer = _racer(tmp_path, script)
    _programs(racer, "dup_a", "dup_b", fp_of={"dup_a": "fp-v1", "dup_b": "fp-v1"})
    racer.register_external_rows("v1", "I-02@o5", [{"S": 0.6}, {"S": 0.8}])
    # a behaviourally identical program asking for rep 0 reuses v1's census rep 0
    rows = racer.run_batch("g0:r1", [("dup_a", "I-02@o5", "r1", None)])
    assert rows[0]["cached"] and rows[0]["S"] == 0.6 and script.calls == []
    # inside one batch: two same-fp programs on a new unit run ONCE
    rows = racer.run_batch("g0:r2", [("dup_a", "I-06@o5", "r2", None),
                                     ("dup_b", "I-06@o5", "r2", None)])
    assert len(script.calls) == 1
    assert sorted(r["cached"] for r in rows) == [False, True]
    snap = racer.meter.snapshot()
    assert snap["totals"]["train_executions"] == 1
    assert snap["totals"]["train_cached_rows"] == 2


def test_real_behaviour_fingerprint_shares_rows(tmp_path: Path) -> None:
    script = Script()
    racer = _racer(tmp_path, script)
    v1 = racer.register_program("v1", v1_seed_source())
    same = racer.register_program("noop", evo_seed_source())  # +wi forwarding +digest: same behaviour
    assert v1.source_sha256 != same.source_sha256
    assert v1.fp and v1.fp == same.fp
    racer.run_batch("g0:r1", [("v1", "I-01@o5", "r1", 0), ("noop", "I-01@o5", "r1", 0)])
    assert len(script.calls) == 1


def test_resume_after_kill_repeats_nothing(tmp_path: Path) -> None:
    requests = [(f"c{i}", "II-12@x5", "r1", None) for i in range(4)] + \
               [("v1", "II-12@x5", "parent_fresh", None)]
    killer = Script({("v1", "II-12@x5"): 0.4}, kill_after=2)
    racer = _racer(tmp_path, killer)
    _programs(racer, "c0", "c1", "c2", "c3")
    with pytest.raises(SystemExit):
        racer.run_batch("g0:r1", requests, gen=0)
    done = list(killer.ok_calls)
    assert len(done) == 2
    # a new process on the same root
    script = Script({("v1", "II-12@x5"): 0.4})
    racer2 = _racer(tmp_path, script)
    assert set(racer2.programs) == {"v1", "c0", "c1", "c2", "c3"}
    rows = racer2.run_batch("g0:r1", requests, gen=0)
    assert len(rows) == 5 and all(r["S"] is not None for r in rows)
    assert not set(script.ok_calls) & set(done)
    assert len(script.ok_calls) + len(done) == 5
    summary = racer2.summary()
    assert summary["log"]["executions"] == 5
    assert summary["meter_train_executions"] == 5 and summary["meter_agrees_with_log"]
    # a third call executes nothing: every row comes from the log
    racer2.run_batch("g0:r1", requests, gen=0)
    assert len(script.ok_calls) + len(done) == 5


def test_infra_reruns_and_infra_final_never_scored_zero(tmp_path: Path) -> None:
    script = Script({("c1", "II-11@x5"): 0.8, ("c2", "II-11@x5"): 0.8},
                    infra={"c1|II-11@x5|r0": 2, "c2|II-11@x5|r0": 5})
    racer = _racer(tmp_path, script)
    _programs(racer, "c1", "c2")
    rows = racer.run_batch("g0:r1", [("c1", "II-11@x5", "r1", None),
                                     ("c2", "II-11@x5", "r1", None)])
    by = {r["program_id"]: r for r in rows}
    assert by["c1"]["S"] == 0.8 and not by["c1"]["infra_final"]
    assert by["c2"]["S"] is None and by["c2"]["infra_final"]
    assert script.calls.count("c2|II-11@x5|r0") == 3  # 1 + 2 re-runs
    snap = racer.meter.snapshot()["phases"]["train_stage1"]
    assert snap["executions"] == 6 and snap["infra"] == 5 and snap["scored"] == 1
    # a later call does not execute an exhausted job again
    racer.run_batch("g0:r1", [])
    racer.run(racer.plan("g0:r1"))
    assert script.calls.count("c2|II-11@x5|r0") == 3
    out = racer.outcome({"program_id": "c2", "parent_id": "v1",
                         "target_units": ["II-11@x5"]}, 0)
    assert out["status"] == "infra_final" and out["verdict"] == "inconclusive"


def test_budget_cap_preflight_and_o10_exec_eq(tmp_path: Path) -> None:
    script = Script()
    racer = _racer(tmp_path, script)
    _programs(racer, "c1")
    with pytest.raises(R.BudgetExceeded):
        racer.run_batch("g0:r1", [("c1", "I-01@o10", "r1", None),
                                  ("c1", "I-02@o5", "r1", None)], cap_exec_eq=2)
    assert script.calls == []
    racer.run_batch("g0:r1", [], cap_exec_eq=3)
    racer.run(racer.plan("g0:r1"), cap_exec_eq=3)
    assert racer.exec_eq_spent() == 3
    assert racer.meter.snapshot()["counters"]["exec_eq_surcharge"] == 1


# --------------------------------------------------------------------------- #
# rows: diag + credit
# --------------------------------------------------------------------------- #


def test_exec_rows_carry_sanitized_diag_and_trace_credit(tmp_path: Path) -> None:
    def execute(req: dict[str, Any]) -> dict[str, Any]:
        inst, prog = req["instance"], req["source"]
        facts = {"S": 0.6, "C": 1234.0, "diag": {"S": 0.6, "failure_class": "divergent",
                                                  "answer_shape": "scalar", "leak": "x"}}
        trace_dir = Path(os.environ["QB_TRACE_DIR"])
        sha12 = R._sha256(prog)[:12]
        (trace_dir / f"{inst.case_id}_{sha12}_{int(time.time() * 1000)}.json").write_text(
            json.dumps({"case_id": inst.case_id, "facts": facts, "output": {"x": 1},
                        "ground_truth": inst.ground_truth}))
        return {"infra": None, "S": 0.6, "C": 1234.0, "diag": facts["diag"]}

    seen = []

    def credit_from_trace(trace: dict, instance: Any, source: str) -> dict:
        seen.append(trace["case_id"])
        return {"credit": {"readable": [5, 5, 3, 5, 99], "retention": [1.0, 0.5, 2.0, None, 0.3],
                           "basis": "trace"}}

    racer = R.Racer(tmp_path / "race", config=R.RaceConfig(fake=True, traces=True,
                                                           parallel_cases=1),
                    resolver=FakeResolver(), execute=execute,
                    credit_from_trace=credit_from_trace)
    racer.register_program("v1", "# v1\n", fp="fp-v1")
    row = racer.run_batch("g0:r1", [("v1", "II-13@o5", "r1", None)])[0]
    assert seen == ["II-13"]
    assert row["diag"]["failure_class"] == "divergent" and "leak" not in row["diag"]
    assert row["credit"] == {"readable": [5, 5, 3, 5, None],
                             "retention": [1.0, 0.5, None, None, 0.3], "basis": "trace"}
    assert row["trace"].startswith("traces/")
    for key in ("unit_id", "program_id", "fp", "rep", "batch_id", "purpose", "gen"):
        assert key in row
    assert "QB_TRACE_DIR" not in os.environ or os.environ["QB_TRACE_DIR"] != str(tmp_path)


def test_trace_dir_is_scoped_to_each_batch(tmp_path: Path, monkeypatch) -> None:
    seen: list[str | None] = []

    def execute(req: dict[str, Any]) -> dict[str, Any]:
        seen.append(os.environ.get("QB_TRACE_DIR"))
        return {"infra": None, "S": 1.0, "C": 1.0}

    def racer_for(name: str, traces: bool) -> R.Racer:
        r = R.Racer(tmp_path / name, config=R.RaceConfig(fake=True, traces=traces),
                    resolver=FakeResolver(), execute=execute)
        r.register_program("c1", "# c1\n", fp=None)
        return r

    monkeypatch.setenv("QB_TRACE_DIR", "/outer/traces")
    racer_for("a", True).run_batch("g0:r1", [("c1", "II-11@x5", "r1", None)])
    racer_for("b", False).run_batch("g0:r1", [("c1", "II-11@x5", "r1", None)])
    assert seen == [str(tmp_path / "a" / "traces" / "g0_r1"), None]
    assert (tmp_path / "a" / "traces" / "g0_r1").is_dir()
    assert os.environ["QB_TRACE_DIR"] == "/outer/traces"  # restored
    monkeypatch.delenv("QB_TRACE_DIR")
    racer_for("c", True).run_batch("g0:r1", [("c1", "II-11@x5", "r1", None)])
    assert seen[-1] == str(tmp_path / "c" / "traces" / "g0_r1")
    assert "QB_TRACE_DIR" not in os.environ


# --------------------------------------------------------------------------- #
# race stages
# --------------------------------------------------------------------------- #


FRONTIER = ["I-02@o5", "II-11@x5", "II-12@x5", "III-21@x5", "III-22@x5"]
GUARDS = ["I-01@o5", "I-06@o5", "II-13@o5"]
SHAPES = {"I-02@o5": "scalar", "II-11@x5": "scalar", "II-12@x5": "collection",
          "III-21@x5": "segment", "III-22@x5": "scalar"}


def _stage_racer(tmp_path: Path, table: dict, **kw: Any) -> R.Racer:
    script = Script(table, **kw)
    racer = _racer(tmp_path, script)
    _programs(racer, "a", "b", "c", "d")
    for unit in FRONTIER:
        racer.register_external_rows("v1", unit, [{"S": 0.4}, {"S": 0.4}])
    for unit in GUARDS:
        racer.register_external_rows("v1", unit, [{"S": 1.0}, {"S": 1.0}])
    racer._script = script  # type: ignore[attr-defined]
    return racer


def _cands() -> list[dict[str, Any]]:
    return [
        {"program_id": "a", "parent_ids": ["v1"], "target_units": ["II-11@x5", "III-22@x5"],
         "brief_id": "g0-s0", "target_class": "divergent"},
        {"program_id": "b", "parent_ids": ["v1"], "target_units": ["II-11@x5"],
         "brief_id": "g0-s1"},
        {"program_id": "c", "parent_ids": ["v1"], "target_units": ["II-12@x5"],
         "brief_id": "g0-s2", "screen": {"rank_penalty": 0.5}},
        {"program_id": "d", "parent_ids": ["v1"], "target_units": ["III-21@x5"],
         "brief_id": "g0-s3"},
    ]


def test_r1_pairs_against_one_fresh_parent_rep_per_unit(tmp_path: Path) -> None:
    table = {("v1", u): 0.4 for u in FRONTIER}
    table.update({("a", "II-11@x5"): 1.0, ("b", "II-11@x5"): 0.2, ("c", "II-12@x5"): 0.8,
                  ("d", "III-21@x5"): 1.0})
    racer = _stage_racer(tmp_path, table, fmt={"d|III-21@x5|r0"})
    r1 = racer.race_r1(0, _cands())
    fresh = [j for j in racer.plan("g0:r1") if j.purpose == "parent_fresh"]
    assert sorted(j.unit_id for j in fresh) == ["II-11@x5", "II-12@x5", "III-21@x5"]
    assert all(j.rep == 2 for j in fresh)  # after census reps 0, 1
    by = {c["program_id"]: c for c in r1["candidates"]}
    assert by["a"]["d1"] == pytest.approx(0.6) and by["a"]["survive"]
    assert by["b"]["status"] == "below_tau_r1" and not by["b"]["survive"]
    assert by["c"]["d1"] == pytest.approx(0.4) and by["c"]["survive"]
    assert by["d"]["status"] == "format_fail" and not by["d"]["survive"]
    # a and c both >= tau_hi 0.4 -> top-2
    assert r1["r2_selected"] == ["a", "c"]
    # a second R1 batch in the same generation does not re-run the fresh reps
    racer.race_r1(0, [{"program_id": "b", "parent_ids": ["v1"],
                       "target_units": ["II-11@x5"]}], tag="r1b")
    assert not [j for j in racer.plan("g0:r1b") if j.purpose == "parent_fresh"]


def test_r1_without_fresh_parent_never_uses_archived_mean(tmp_path: Path) -> None:
    table = {("a", "II-11@x5"): 1.0}
    racer = _stage_racer(tmp_path, table, infra={"v1|II-11@x5|r2": 9})
    r1 = racer.race_r1(0, _cands()[:1])
    cand = r1["candidates"][0]
    assert cand["status"] == "no_fresh_parent" and cand["d1"] is None
    assert r1["r2_selected"] == []


def test_select_for_r2_rules() -> None:
    th = R.Thresholds()
    cands = [{"program_id": "a", "d1": 0.2, "survive": True, "C": 5.0},
             {"program_id": "b", "d1": 0.2, "survive": True, "C": 3.0},
             {"program_id": "c", "d1": 0.0, "survive": True, "C": 1.0}]
    assert R.select_for_r2(cands, th) == ["b"]  # tie on d1 -> fewer tokens
    cands[0]["d1"] = cands[1]["d1"] = 0.4
    assert R.select_for_r2(cands, th) == ["b", "a"]
    assert R.select_for_r2([{"program_id": "x", "d1": -0.2, "survive": False}], th) == []


def test_pick_r2_units_is_stable_and_shape_aware() -> None:
    units = R.pick_r2_units("a", "II-11@x5", FRONTIER + ["II-11@o5"], gen=0, shapes=SHAPES)
    assert len(units) == 2
    assert all(R.template_of(u) != "II-11" for u in units)
    assert all(SHAPES[u] != "scalar" for u in units)  # different shape first
    assert units == R.pick_r2_units("a", "II-11@x5", list(reversed(FRONTIER)), gen=0,
                                    shapes=SHAPES)
    other = R.pick_r2_units("a", "II-11@x5", FRONTIER, gen=0, shapes=SHAPES, other_tier=True)
    assert all(R.tier_of(u) != "II" for u in other)
    assert R.pick_guard(GUARDS, gen=0) == "I-01@o5"
    assert R.pick_guard(GUARDS, gen=1) == "I-06@o5"
    assert R.pick_guard(GUARDS, gen=0, exclude_templates={"I-01"}) == "I-06@o5"
    assert R.pick_guard([], gen=0) is None


def test_r2_outcome_tier_guard_and_verdicts(tmp_path: Path) -> None:
    table = {("v1", u): 0.4 for u in FRONTIER}
    table.update({("a", u): 1.0 for u in FRONTIER})   # a: +0.6 everywhere, guard kept
    table.update({("c", u): 0.8 for u in FRONTIER})   # c: +0.4, but breaks the guard
    table.update({("a", g): 1.0 for g in GUARDS})
    table.update({("c", g): 0.6 for g in GUARDS})
    racer = _stage_racer(tmp_path, table)
    cands = _cands()[:1] + _cands()[2:3]
    r1 = racer.race_r1(0, cands)
    assert r1["r2_selected"] == ["a", "c"]
    r2 = racer.race_r2(0, r1, t_frontier=FRONTIER, guards=GUARDS, shapes=SHAPES)
    plan_a, plan_c = r2["plan"]["a"], r2["plan"]["c"]
    assert len(plan_a["units"]) == 2 and plan_a["guard"] in GUARDS
    assert plan_a["guard"] != plan_c["guard"]  # rotating guard per slot
    oa, oc = r2["outcomes"]["a"], r2["outcomes"]["c"]
    assert oa["reached_r2"] and oa["D"] == pytest.approx(0.6)
    assert oa["guard_ok"] is True and oa["high_tier"] and oa["verdict"] == "confirmed"
    assert oa["target_dS"] == pytest.approx(0.6)  # II-11 (fresh) + III-22 (archived)
    assert oa["per_unit"]["II-11@x5"]["baseline_src"] == "fresh"
    assert oc["guard_ok"] is False and oc["guard"]["drop"] == pytest.approx(0.4)
    assert not oc["high_tier"] and oc["verdict"] == "inconclusive"
    # resume: re-planning R2 keeps the stored units / guard, runs nothing new
    before = len(racer._script.calls)  # type: ignore[attr-defined]
    again = racer.race_r2(0, r1, t_frontier=list(reversed(FRONTIER)), guards=GUARDS[::-1],
                          shapes={})
    assert again["plan"] == r2["plan"]
    assert len(racer._script.calls) == before  # type: ignore[attr-defined]


def test_race_generation_with_confirmation(tmp_path: Path) -> None:
    from queenbee.evo.memory import Archive

    table = {("v1", u): 0.4 for u in FRONTIER}
    table.update({("a", u): [1.0, 1.0] for u in FRONTIER})
    table.update({("a", g): 1.0 for g in GUARDS})
    racer = _stage_racer(tmp_path, table)
    archive = Archive(t_templates=EXAMPLE_T)
    for unit in FRONTIER + GUARDS:
        archive.add_rows(racer.exec_row(r) for r in racer.scored_rows("v1", unit))

    def winners(rows: list[dict]) -> list[dict]:
        archive.add_rows(rows)
        return archive.newly_front_winning(mark=True, limit=1)

    res = racer.race_generation(0, _cands()[:1], t_frontier=FRONTIER, guards=GUARDS,
                                shapes=SHAPES, winners_fn=winners)
    conf = [j for j in racer.plan("g0:confirm")]
    assert len(conf) == 1 and conf[0].purpose == "confirm" and conf[0].program_id == "a"
    assert conf[0].rep == 1  # a fresh draw after the R1 / R2 rep 0
    out = res["outcomes"]["a"]
    assert out["confirm_units"] == [conf[0].unit_id]
    assert out["verdict"] == "confirmed"
    archive.add_rows(res["confirm"]["rows"])
    assert all(archive.resolve_confirmations([(conf[0].program_id, conf[0].unit_id)]).values())


def test_host_verdict_rules() -> None:
    th = R.Thresholds()
    assert R.host_verdict(None, th) == "screened"
    assert R.host_verdict({"executed": False}, th) == "screened"
    assert R.host_verdict({"executed": False, "status": "infra_final"}, th) == "inconclusive"
    # a tie is no evidence against the hypothesis: refuted only one quantum below
    assert R.host_verdict({"executed": True, "target_dS": 0.0}, th) == "inconclusive"
    assert R.host_verdict({"executed": True, "target_dS": -0.1,
                           "r1_unit": "II-11@x5"}, th) == "inconclusive"
    assert R.host_verdict({"executed": True, "target_dS": -0.2}, th) == "refuted"
    assert R.host_verdict({"executed": True, "target_dS": -0.1,
                           "r1_unit": "II-11@o10"}, th) == "refuted"  # 1/n = 0.1
    # refute_quanta = 0 gives the strict rule "refuted iff dS <= 0"
    literal = R.Thresholds(refute_quanta=0.0)
    assert R.host_verdict({"executed": True, "target_dS": 0.0}, literal) == "refuted"
    assert R.Thresholds.from_dict(literal.to_dict()) == literal
    assert R.host_verdict({"executed": True, "target_dS": 0.2, "guard_ok": True}, th) == "confirmed"
    assert R.host_verdict({"executed": True, "target_dS": 0.2, "guard_ok": None}, th) == "inconclusive"
    assert R.host_verdict({"executed": True, "target_dS": 0.1, "guard_ok": True}, th) == "inconclusive"
    assert R.host_verdict({"executed": True, "target_dS": None}, th) == "inconclusive"


def test_outcome_r1_unit_requires_the_fresh_parent_rep(tmp_path: Path) -> None:
    # the fresh parent rep on the R1 unit is infra_final -> no dS there (never
    # the archived mean), status no_fresh_parent, inconclusive
    table = {("v1", "II-11@x5"): 0.0, ("a", "II-11@x5"): 1.0}
    racer = _stage_racer(tmp_path, table, infra={"v1|II-11@x5|r2": 9})
    r1 = racer.race_r1(0, _cands()[:1])
    cand = r1["candidates"][0]
    assert cand["status"] == "no_fresh_parent"
    out = racer.outcome(cand, 0)
    assert out["target_dS"] is None and out["d1"] is None and out["observed"] == {}
    assert out["per_unit"]["II-11@x5"]["baseline_src"] == "no_fresh_parent"
    assert out["status"] == "no_fresh_parent" and out["verdict"] == "inconclusive"


def test_r2_baseline_falls_back_to_v1_and_D_needs_every_unit(tmp_path: Path) -> None:
    # an evolved parent P has rows on its own unit only; the R2 units are
    # paired with v1's census mean, so D covers all 3 units
    frontier = ["II-11@x5", "II-12@x5", "III-27@x5"]
    script = Script({("c2", "II-11@x5"): 1.0}, default=0.6)
    racer = _racer(tmp_path, script)
    _programs(racer, "P", "c2")
    for unit in frontier + GUARDS:
        racer.register_external_rows("v1", unit, [{"S": 0.6}, {"S": 0.6}])
    racer.run_batch("g0:r1", [("P", "II-11@x5", "r1", None)])
    r1 = racer.race_r1(1, [{"program_id": "c2", "parent_ids": ["P"],
                            "target_units": ["II-11@x5"]}])
    r2 = racer.race_r2(1, r1, t_frontier=frontier, guards=GUARDS)
    out = r2["outcomes"]["c2"]
    assert set(out["r2_units"]) == set(frontier)
    assert out["per_unit"]["II-11@x5"]["baseline_src"] == "fresh"
    assert {out["per_unit"][u]["baseline_src"] for u in frontier[1:]} == {"v1_archived"}
    assert out["D"] == pytest.approx((0.4 + 0.0 + 0.0) / 3, abs=1e-6)
    assert not out["high_tier"]  # D 0.13 < tau_2
    # without any baseline on an R2 unit, D is None (never over fewer units)
    racer2 = _racer(tmp_path / "b", Script({("c3", "II-11@x5"): 1.0}, default=0.6))
    _programs(racer2, "c3")
    racer2.register_external_rows("v1", "II-11@x5", [{"S": 0.2}])
    r1b = racer2.race_r1(0, [{"program_id": "c3", "parent_ids": ["v1"],
                              "target_units": ["II-11@x5"]}])
    ob = racer2.race_r2(0, r1b, t_frontier=frontier, guards=GUARDS)["outcomes"]["c3"]
    assert ob["D"] is None and "no dS" in ob["D_note"] and not ob["high_tier"]


def test_target_dS_is_the_r1_unit_and_guard_fails_closed(tmp_path: Path) -> None:
    # R2 landing on another target unit of the brief is reported in
    # target_dS_all; the verdict uses the R1 unit
    table = {("v1", u): 0.4 for u in FRONTIER}
    table.update({("a", "II-11@x5"): 1.0, ("a", "III-22@x5"): 0.4, ("a", "III-21@x5"): 1.0})
    racer = _stage_racer(tmp_path, table)
    r1 = racer.race_r1(0, _cands()[:1])  # a: targets II-11 (R1) + III-22
    # an empty guard pool: the guard condition cannot be checked -> fail closed
    r2 = racer.race_r2(0, r1, t_frontier=["II-11@x5", "III-22@x5", "III-21@x5"], guards=[])
    out = r2["outcomes"]["a"]
    assert set(out["r2_units"]) == {"II-11@x5", "III-22@x5", "III-21@x5"}
    assert out["target_dS"] == pytest.approx(0.6)
    assert out["target_dS_all"] == pytest.approx(0.3)
    assert out["guard_ok"] is None and "not checked" in out["guard"]["note"]
    assert not out["high_tier"] and out["verdict"] == "inconclusive"
    assert out["target_dC"] is not None  # economize reads the cost delta


def test_exec_cache_and_external_rows_need_the_same_worker_config(tmp_path: Path) -> None:
    script = Script({("v1", "I-02@o5"): 0.6})
    racer = _racer(tmp_path, script, worker_model="Worker-Model")
    _programs(racer, "dup", fp_of={"dup": "fp-v1"})
    want = racer.config.worker_fingerprint()
    assert want["worker_model"] == "Worker-Model" and want["goal"] == R.GOAL
    ok = {"fingerprint": {"model": "Worker-Model", "worker_contract": want["worker_contract"],
                          "goal": R.GOAL, "python_budgets": {"max_rounds": R.PYTHON_MAX_ROUNDS}}}
    assert racer.register_external_rows("v1", "I-02@o5", [{"S": 0.6, "max_rounds": 64}],
                                        config=ok) == 1
    with pytest.raises(R.RaceError):
        racer.register_external_rows("v1", "I-06@o5", [{"S": 1.0}],
                                     config={"model": "Other-Model"})
    with pytest.raises(R.RaceError):
        racer.register_external_rows("v1", "I-06@o5", [{"S": 1.0, "max_rounds": 32}])
    assert racer.run_batch("g0:r1", [("dup", "I-02@o5", "r1", None)])[0]["cached"]
    # the same root under another worker model: nothing is served from the cache
    other = _racer(tmp_path, script, worker_model="Other-Model")
    assert any("other worker configs" in w for w in other.warnings)
    rows = other.run_batch("g1:r1", [("dup", "I-02@o5", "r1", 1)])
    assert not rows[0]["cached"] and script.calls == ["dup|I-02@o5|r1"]


def test_host_exception_is_infra_and_leak_guards_propagate(tmp_path: Path) -> None:
    calls: list[str] = []

    def boom(req: dict[str, Any]) -> dict[str, Any]:
        calls.append(req["job"].key)
        raise RuntimeError("instance loader exploded")

    racer = R.Racer(tmp_path / "race", config=R.RaceConfig(fake=True, traces=False,
                                                           infra_backoff_s=0.0),
                    resolver=FakeResolver(), execute=boom)
    racer.register_program("c1", "# c1\n", fp="fp-c1")
    row = racer.run_batch("g0:r1", [("c1", "II-11@x5", "r1", None)])[0]
    assert row["S"] is None and row["infra_final"] and len(calls) == 3
    assert str(row["infra"]).startswith("host:RuntimeError")
    assert racer.exec_eq_spent() == 0

    def leak(req: dict[str, Any]) -> dict[str, Any]:
        raise R.LeakageGuardError("LEAKAGE_GUARD: test")

    racer2 = R.Racer(tmp_path / "race2", config=R.RaceConfig(fake=True, traces=False),
                     resolver=FakeResolver(), execute=leak)
    racer2.register_program("c1", "# c1\n", fp="fp-c1")
    with pytest.raises(R.LeakageGuardError):
        racer2.run_batch("g0:r1", [("c1", "II-11@x5", "r1", None)])
    with pytest.raises(R.LeakageGuardError):
        R._failure_row(R.LeakageGuardError("x"))  # also when a wrapper calls it from its except


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #


def test_thresholds_roundtrip_and_loading(tmp_path: Path) -> None:
    th = R.Thresholds()
    assert (th.tau_r1, th.tau_2, th.tau_h, th.tau_hi, th.loss_max) == (0.0, 0.2, 0.2, 0.4, 1.0)
    assert th.guard_allowance("II-11@x5") == pytest.approx(0.2)
    assert th.guard_allowance("II-11@o10") == pytest.approx(0.1)
    assert th.guard_ok(0.2, "II-11@x5") and not th.guard_ok(0.4, "II-11@x5")
    custom = R.Thresholds(tau_2=0.3, tau_h=0.25, fpr=0.08, tpr=0.7, calib_sha256="abc")
    assert R.Thresholds.from_dict(json.loads(json.dumps(custom.to_dict()))) == custom
    path = tmp_path / "th.json"
    path.write_text(json.dumps({"thresholds": custom.to_dict()}))
    assert R.load_thresholds(path) == custom
    assert R.load_thresholds(str(path)) == custom
    assert R.load_thresholds(None) == R.Thresholds()
    assert R.load_thresholds(custom) is custom
    # a flat JSON / mapping works too; unknown keys are ignored
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"tau_2": 0.35, "tau_h": 0.3, "refute_quanta": 0,
                                "source": "file", "unrelated": [1, 2]}))
    loaded = R.load_thresholds(flat)
    assert (loaded.tau_2, loaded.tau_h, loaded.refute_quanta, loaded.source) == \
        (0.35, 0.3, 0.0, "file")
    assert (loaded.tau_r1, loaded.tau_hi, loaded.loss_max) == (0.0, 0.4, 1.0)
    assert R.load_thresholds({"thresholds": {"tau_hi": 0.5}}).tau_hi == 0.5
    assert R.load_thresholds({"tau_r1": 0.1}).tau_r1 == 0.1


def test_vacuous_guard_confirms_when_the_pool_has_no_guard_unit(tmp_path: Path) -> None:
    """A weak worker may solve no T unit stably, which leaves the guard pool
    empty.  With vacuous_guard the "no regression" condition then holds
    vacuously and a clear R1 gain that reached R2 is confirmed; a real guard
    unit, when there is one, still binds."""
    table = {("v1", u): 0.4 for u in FRONTIER}
    table.update({("a", "II-11@x5"): 1.0, ("a", "III-22@x5"): 0.4, ("a", "III-21@x5"): 1.0})
    script = Script(table)
    racer = _racer(tmp_path, script, vacuous_guard=True)
    _programs(racer, "a", "b", "c", "d")
    for unit in FRONTIER:
        racer.register_external_rows("v1", unit, [{"S": 0.4}, {"S": 0.4}])
    r1 = racer.race_r1(0, _cands()[:1])
    out = racer.race_r2(0, r1, t_frontier=["II-11@x5", "III-22@x5", "III-21@x5"],
                        guards=[])["outcomes"]["a"]
    assert out["guard_ok"] is True and "vacuous" in out["guard"]["note"]
    assert out["target_dS"] == pytest.approx(0.6) and out["verdict"] == "confirmed"
