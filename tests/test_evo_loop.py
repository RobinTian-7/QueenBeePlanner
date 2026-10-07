"""QueenBee-Evo evolution loop (``queenbee.evo.loop``), offline.

No network: a scripted executor (S depends on the program's structure) or
the fake worker provider, and the loop's fake planner.  The loop runs on
``queenbee.evo.race.Racer`` and ``queenbee.evo.memory`` (Archive / Ledger /
Scheduler).  Covers: arm flags, model / endpoint / planner-cap resolution,
the fail-fast check of real runs, TEST / V / T guards, an end-to-end run of
the full arm over >= 3 generations with async minting and final selection,
a crash in the middle of R2 (simulated, and a real ``kill -9`` of a child
process) + resume (no execution repeated, meter == rows), both arms and
every ablation switch, tabu at S0, the curriculum, the V guard, the
final-selection tie rule, worker and planner outages, inherited programs,
pinned irreducible templates and CLI smoke runs with the fake provider
(with a census file and with the synthetic census).
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from queenbee.evo import common as P
from queenbee.evo import loop as L
from queenbee.evo import memory as M
from queenbee.paths import default_benchmarks_dir

#: The loop resolves real Silo-Bench instances (and builds its x5 ladder from
#: them): every run below needs the benchmark data.
HAVE_BENCH = (default_benchmarks_dir() / "I-01_n5.json").is_file()
NO_BENCH = "Silo-Bench data missing (third_party/acl26-silo-bench or $SILO_BENCH_DIR)"
needs_bench = pytest.mark.skipif(not HAVE_BENCH, reason=NO_BENCH)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

#: Census S of the seed program (v1) per o5 template (2 reps); templates at
#: 1.0 are guard units, the others frontier units.
CENSUS_S: dict[str, list[float]] = {
    "I-01": [0.4, 0.6], "I-02": [0.6, 0.6], "I-03": [0.4, 0.4], "I-06": [0.8, 0.6],
    "I-07": [1.0, 1.0], "I-08": [1.0, 1.0],
    "II-11": [0.2, 0.4], "II-12": [0.4, 0.6], "II-13": [0.6, 0.8], "II-16": [0.4, 0.6],
    "II-17": [1.0, 1.0], "II-18": [1.0, 1.0],
    "III-21": [0.4, 0.2], "III-22": [0.6, 0.4], "III-23": [0.2, 0.4], "III-26": [0.6, 0.6],
    "III-27": [1.0, 1.0], "III-28": [1.0, 1.0],
}
SPLIT = {"name": "t1",
         "T": ["I-01", "I-02", "I-06", "I-08", "II-11", "II-12", "II-13", "II-17",
               "III-21", "III-22", "III-26", "III-27"],
         "V": ["I-03", "I-07", "II-16", "II-18", "III-23", "III-28"]}


def _cls(t: str, s: float) -> str:
    if s >= 1.0:
        return "ok"
    return "divergent" if t.startswith("II-") else "consensus-wrong"


def _shape(t: str) -> str:
    return "collection" if t.startswith("III-") else "scalar"


def _census(tmp_path: Path, table: dict[str, list[float]] | None = None) -> Path:
    """A census file of the seed program (v1), written by ``loop.fake_census``;
    its synthetic diagnosis fields follow the same rules as ``_cls`` /
    ``_shape``."""

    path = tmp_path / "census.json"
    path.write_text(json.dumps(L.fake_census(table or CENSUS_S)))
    return path


def _split_file(tmp_path: Path, split: dict | None = None) -> Path:
    path = tmp_path / ("split.json" if split is None else "split_custom.json")
    path.write_text(json.dumps(split or SPLIT))
    return path


def _phases(source: str) -> list[dict[str, Any]]:
    import ast

    match = re.search(r"^PHASES = (\[\n.*?^\])\n", source, re.S | re.M)
    return [dict(p) for p in ast.literal_eval(match.group(1))] if match else []


def _features(source: str) -> dict[str, bool]:
    phases = _phases(source)
    return {"wi": any(p.get("wi") for p in phases),
            "mesh": bool(phases) and phases[0].get("kind") == "mesh",
            "digest": any(p.get("kind") == "digest" for p in phases)}


class Kill(BaseException):
    """Simulates kill -9: not an Exception, so no handler in the racer or
    the loop turns it into a row; it propagates out of the run."""


class Scripted:
    """execute(req) -> row.  A program starts from its template's census mean
    (x5 / o10: 0.4), so v1 scores its census mean; a phase work instruction
    lifts tier II / III by 0.4, a digest round lifts tier I by 0.2, a leading
    mesh round costs tier I 0.2 (S is then rounded to a multiple of 0.2 in
    [0, 1]).  ``kill_at = (purpose, k)`` raises Kill at the k-th job of that
    purpose; ``infra[key]`` = number of infra rows the job returns first."""

    def __init__(self, *, kill_at: tuple[str, int] | None = None,
                 infra: dict[str, int] | None = None, census: dict | None = None) -> None:
        self.census = census or CENSUS_S
        self.calls: list[str] = []
        self.by_purpose: Counter = Counter()
        self.kill_at = kill_at
        self.infra = dict(infra or {})
        self._lock = threading.Lock()

    def __call__(self, req: dict[str, Any]) -> dict[str, Any]:
        job = req["job"]
        with self._lock:
            self.calls.append(job.key)
            self.by_purpose[job.purpose] += 1
            if self.kill_at and job.purpose == self.kill_at[0] \
                    and self.by_purpose[job.purpose] == self.kill_at[1]:
                raise Kill(job.key)
            if self.infra.get(job.key, 0) > 0:
                self.infra[job.key] -= 1
                return {"infra": "ConnectionError: 502 bad gateway"}
        t, rung = L.template_of(job.unit_id), L.rung_of(job.unit_id)
        base = (sum(self.census[t]) / 2) if rung == "o5" else 0.4
        f = _features(req["source"])
        tier = t.split("-")[0]
        s = base
        if f["wi"] and tier in ("II", "III"):
            s += 0.4
        if f["digest"] and tier == "I":
            s += 0.2
        if f["mesh"] and tier == "I":
            s -= 0.2
        s = round(min(1.0, max(0.0, round(s * 5) / 5)), 4)
        calls = 10 + 5 * f["mesh"] + 2 * f["digest"]
        return {"infra": None, "execution_class": "completed", "success": s >= 1, "S": s,
                "C": 1000.0 * calls, "model_calls": calls, "prompt_tokens": 400 * calls,
                "completion_tokens": 600 * calls,
                "diag": {"S": s, "failure_class": _cls(t, s), "answer_shape": _shape(t)}}


class CountingPlanner(L.FakeEvoPlanner):
    """Fake planner that records the peak number of calls in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.inflight = 0
        self.peak = 0
        self._g = threading.Lock()

    def complete(self, prompt: str, *a: Any, **kw: Any) -> Any:
        import time

        with self._g:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        try:
            time.sleep(0.02)
            return super().complete(prompt, *a, **kw)
        finally:
            with self._g:
                self.inflight -= 1


def _cfg(tmp_path: Path, census_S: dict[str, list[float]] | None = None, **kw: Any) -> L.EvoConfig:
    if not HAVE_BENCH:
        pytest.skip(NO_BENCH)
    census = _census(tmp_path, census_S)
    kw.setdefault("arm", "full")
    kw.setdefault("planner_model", "test-planner")
    kw.setdefault("worker_model", "test-worker")
    kw.setdefault("budget", 40.0)
    kw.setdefault("final_reserve", 12.0)
    if "split_file" not in kw:
        kw["split_file"] = _split_file(tmp_path)
    kw.setdefault("planner_concurrency", 3)
    kw.setdefault("outage_wait_s", 0.0)
    return L.EvoConfig(
        root=tmp_path / "root", split_seed=1, unit_sources=(census,),
        trace_dirs=(tmp_path / "no_traces",), traces=False, infra_backoff_s=0.0,
        planner_backoff_s=0.0, parallel_cases=1, **kw,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _gens(root: Path) -> list[dict[str, Any]]:
    out = []
    g = 0
    while (root / "state" / f"generation_{g:03d}.json").is_file():
        out.append(json.loads((root / "state" / f"generation_{g:03d}.json").read_text()))
        g += 1
    return out


def _run(tmp_path: Path, execute: Any = None, planner: Any = None, **kw: Any):
    cfg = _cfg(tmp_path, **kw)
    execute = execute or Scripted()
    planner = planner or CountingPlanner()
    summary = L.run_evo(cfg, L.EvoDeps(execute=execute, planner_client=planner))
    return cfg, summary, execute, planner


def _resume(cfg: L.EvoConfig, **kw: Any) -> L.EvoConfig:
    return L.EvoConfig(**{**cfg.__dict__, "resume": True, **kw})


def _meter_agrees(root: Path) -> None:
    records = [r for r in _jsonl(root / "executions.jsonl") if not r.get("external")]
    meter = json.loads((root / "budget_meter.json").read_text())
    phases = meter["phases"]
    paid = [r for r in records if not r["cached"]]
    assert sum(p["executions"] for p in phases.values()) == len(paid)
    assert sum(p["scored"] for p in phases.values()) == sum(1 for r in paid if not r["row"].get("infra"))
    assert sum(p["cached_rows"] for p in phases.values()) == sum(1 for r in records if r["cached"])
    # train_val = the final reserve: V selection (+ the coverage top-up T reps)
    assert phases["train_val"]["executions"] == sum(1 for r in paid if r["purpose"] in L.FINAL_PURPOSES)


def _open(tmp_path: Path, **kw: Any) -> L.EvoRun:
    ex = kw.pop("execute", None) or Scripted()
    return L.EvoRun(_cfg(tmp_path, **kw), L.EvoDeps(execute=ex, planner_client=CountingPlanner()))


# --------------------------------------------------------------------------- #
# pure pieces
# --------------------------------------------------------------------------- #


def test_arm_flags_and_config() -> None:
    assert L.ARMS == ("full", "mf_elite")
    assert L.arm_flags("full") == L.EvoFlags(True, True, True)
    assert L.arm_flags("mf_elite") == L.EvoFlags(False, False, False)
    assert L.arm_flags("mf_elite") == L.arm_flags("full", no_diag=True, no_ledger=True,
                                                  no_archive=True)
    assert L.arm_flags("full", no_diag=True) == L.EvoFlags(False, True, True)
    assert L.arm_flags("full", no_ledger=True) == L.EvoFlags(True, False, True)
    assert L.arm_flags("full", no_archive=True) == L.EvoFlags(True, True, False)
    assert L.parent_policy(L.arm_flags("full")) == "specialist"
    assert L.parent_policy(L.arm_flags("full", no_archive=True)) == "incumbent"
    assert L.parent_policy(L.arm_flags("mf_elite")) == "incumbent"
    for arm in ("oracle", "full-noledger", "Full"):
        with pytest.raises(ValueError):
            L.arm_flags(arm)
        with pytest.raises(ValueError):
            L.EvoConfig(root=Path("/x"), arm=arm)
    cfg = L.EvoConfig(root=Path("/nonexistent/x"), arm="full", no_ledger=True, split_seed=2, budget=80)
    assert cfg.variant == "full-noledger"
    assert cfg.evolution_budget == 60.0
    with pytest.raises(ValueError):
        L.EvoConfig(root=Path("/x"), planner_concurrency=4)  # planner calls in flight <= 3


def test_models_benchmarks_and_planner_caps_resolve(tmp_path: Path, monkeypatch) -> None:
    for var in (L.PLANNER_MODEL_ENV, L.WORKER_MODEL_ENV, "SILO_BENCH_DIR", "QB_PLANNER_DEADLINE_S"):
        monkeypatch.delenv(var, raising=False)
    real = L.EvoConfig(root=tmp_path / "r")
    assert real.planner_model is None and real.worker_model is None  # no built-in model
    fake = L.EvoConfig(root=tmp_path / "f", fake=True)
    assert fake.planner_model == fake.worker_model == L.FAKE_MODEL
    monkeypatch.setenv(L.PLANNER_MODEL_ENV, "planner-x")
    monkeypatch.setenv(L.WORKER_MODEL_ENV, "worker-y")
    env = L.EvoConfig(root=tmp_path / "e")
    assert (env.planner_model, env.worker_model) == ("planner-x", "worker-y")
    flag = L.EvoConfig(root=tmp_path / "g", planner_model="p", worker_model="w")
    assert (flag.planner_model, flag.worker_model) == ("p", "w")
    args = L.build_parser().parse_args(["--arm", "full", "--root", str(tmp_path / "c"),
                                        "--split-seed", "1", "--worker-model", "w2"])
    cli = L.config_from_args(args)
    assert (cli.planner_model, cli.worker_model) == ("planner-x", "w2")
    # benchmarks: queenbee.paths.default_benchmarks_dir(), $SILO_BENCH_DIR overrides it
    assert real.benchmarks_dir == default_benchmarks_dir()
    monkeypatch.setenv("SILO_BENCH_DIR", str(tmp_path / "bench"))
    assert L.EvoConfig(root=tmp_path / "h").benchmarks_dir == tmp_path / "bench"
    # planner caps: explicit defaults (also on the CLI); 0 switches one off
    assert (real.planner_max_completion_tokens, real.planner_deadline_s) == (96000, 3600.0)
    assert (cli.planner_max_completion_tokens, cli.planner_deadline_s) == (96000, 3600.0)
    assert real.effective_planner_deadline() == 3600.0
    assert L.EvoConfig(root=tmp_path / "i", planner_deadline_s=0).effective_planner_deadline() is None
    unset = L.EvoConfig(root=tmp_path / "j", planner_deadline_s=None)
    assert unset.effective_planner_deadline() == 3600.0
    monkeypatch.setenv("QB_PLANNER_DEADLINE_S", "120")
    assert unset.effective_planner_deadline() == 120.0
    with pytest.raises(ValueError):
        L.EvoConfig(root=tmp_path / "k", planner_deadline_s=-1)
    # planner endpoint: none by default (SDK environment), else as given
    assert real.planner_base_url is None and real.planner_api_key_env is None
    args = L.build_parser().parse_args(["--arm", "full", "--root", str(tmp_path / "d"),
                                        "--split-seed", "1", "--planner-base-url", "http://h/v1",
                                        "--planner-api-key-env", "PLANNER_KEY"])
    cli = L.config_from_args(args)
    assert (cli.planner_base_url, cli.planner_api_key_env) == ("http://h/v1", "PLANNER_KEY")


def test_real_runs_fail_fast_before_writing_the_root(tmp_path: Path, monkeypatch, capsys) -> None:
    for var in (L.PLANNER_MODEL_ENV, L.WORKER_MODEL_ENV, L.DEFAULT_API_KEY_ENV, "PLANNER_KEY"):
        monkeypatch.delenv(var, raising=False)
    census, split = _census(tmp_path), _split_file(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    cfg = L.EvoConfig(root=tmp_path / "real", unit_sources=(census,), ladder_manifests=(manifest,),
                      split_file=split, planner_api_key_env="PLANNER_KEY")
    assert L.missing_settings(cfg) == [
        f"--worker-model (or ${L.WORKER_MODEL_ENV})", f"--planner-model (or ${L.PLANNER_MODEL_ENV})",
        "$PLANNER_KEY", f"${L.DEFAULT_API_KEY_ENV}"]
    with pytest.raises(L.EvoStateError, match="--planner-model"):
        L.EvoRun(cfg)
    assert not (tmp_path / "real").exists()
    # the CLI stops with the same message (exit 2) before anything is written
    argv = ["--arm", "full", "--split-seed", "1", "--root", str(tmp_path / "cli"),
            "--units-from", str(census), "--ladder-manifest", str(manifest), "--split-file", str(split)]
    with pytest.raises(SystemExit) as info:
        L.main(argv)
    assert info.value.code == 2 and "--worker-model" in capsys.readouterr().err
    assert not (tmp_path / "cli").exists()
    # models from the environment, one key for planner and workers
    monkeypatch.setenv(L.PLANNER_MODEL_ENV, "p")
    monkeypatch.setenv(L.WORKER_MODEL_ENV, "w")
    plain = L.EvoConfig(**{**cfg.__dict__, "planner_model": None, "worker_model": None,
                           "planner_api_key_env": None})
    assert L.missing_settings(plain) == [f"${L.DEFAULT_API_KEY_ENV}"]
    monkeypatch.setenv(L.DEFAULT_API_KEY_ENV, "dummy")
    assert L.missing_settings(plain) == []
    # injected parts need nothing of their own; --fake needs nothing at all
    monkeypatch.delenv(L.DEFAULT_API_KEY_ENV)
    injected = L.EvoDeps(execute=lambda req: {}, planner_client=L.FakeEvoPlanner())
    assert L.missing_settings(L.EvoConfig(root=tmp_path / "x", worker_model="w"), injected) == []
    assert L.missing_settings(L.EvoConfig(root=tmp_path / "f", fake=True)) == []


def test_fake_census_is_a_loadable_dev_census() -> None:
    from queenbee.evo.units import load_units

    doc = L.fake_census()
    reps = doc["arms"]["v1"]["repeats"]
    assert doc["config"]["agents"] == 5 and sorted(reps) == ["0", "1"]
    templates = [r["case_id"] for r in reps["0"]["rows"]]
    assert sorted(templates) == sorted(P.DEV_IDS) and not set(templates) & set(P.TEST_IDS)
    classes = Counter(u.klass for u in load_units([doc]).values())
    assert classes == {"frontier": 12, "guard": 6}
    assert L.fake_census() == doc  # deterministic, so a run can pin it by its sha256
    one = L.fake_census({"II-11": [0.5]})["arms"]["v1"]["repeats"]
    assert list(one) == ["0"] and one["0"]["rows"][0]["diag"]["failure_class"] == "divergent"
    with pytest.raises(L.LeakageGuardError):
        L.fake_census({P.TEST_IDS[0]: [0.5]})


def test_batch_ids() -> None:
    assert L.batch_id(3, "r1") == "g3:r1" and L.batch_key("g3:r1") == (3, 0)
    assert L.batch_key("g12:curr") == (12, 3)
    assert L.batch_key("final:vselect") is None and L.batch_key("census") is None
    with pytest.raises(ValueError):
        L.batch_id(0, "test")


@needs_bench
def test_fake_mutations_are_screen_clean_and_host_tagged() -> None:
    from queenbee.evo.screen import known_fps_for, screen_program
    from queenbee.evo.seed import evo_seed_source
    from queenbee.program.genome import genome_region, splice_genome

    v1 = evo_seed_source()
    genome = genome_region(v1)
    known = known_fps_for({"v1": v1})
    tags = set()
    for index in range(4):
        new_genome, mechanism = L.fake_mutation(genome, index)
        child = splice_genome(v1, new_genome)
        result = screen_program(child, known_fps=known)
        assert result.ok, (index, result.reasons)
        tags.add(tuple(M.mechanism_tags(v1, child)))
        assert mechanism
    assert len(tags) >= 3 and ("none",) not in tags


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("test_id", P.TEST_IDS)
def test_test_ids_are_refused_by_the_racer_and_pool(tmp_path: Path, test_id: str) -> None:
    run = _open(tmp_path)
    try:
        with pytest.raises(L.LeakageGuardError):
            run.racer.run([L.Job("v1", f"{test_id}@o5", 9, "r1")], cap_exec_eq=99)
        with pytest.raises(L.LeakageGuardError):
            run.racer.run([L.Job("v1", f"{test_id}@o5", 9, "vselect")], cap_exec_eq=99)
        with pytest.raises(L.LeakageGuardError):
            run.pool.side(f"{test_id}@x5")
        with pytest.raises(L.LeakageGuardError):
            run.pool.path(f"{test_id}@o5")
    finally:
        run.close()


def test_split_sides_and_budget_preflight(tmp_path: Path) -> None:
    ex = Scripted()
    run = _open(tmp_path, execute=ex)
    try:
        with pytest.raises(L.LeakageGuardError):  # V unit in an evolution purpose
            run.racer.run([L.Job("v1", "II-16@o5", 9, "r1")], cap_exec_eq=99)
        with pytest.raises(L.LeakageGuardError):  # T unit in final selection
            run.racer.run([L.Job("v1", "II-11@o5", 9, "vselect")], cap_exec_eq=99)
        with pytest.raises((L.LeakageGuardError, L.RaceError)):  # purpose "test" never runs
            run.racer.run([L.Job("v1", "II-11@o5", 9, "test")], cap_exec_eq=99)
        assert ex.calls == []
        with pytest.raises(L.BudgetExceeded):  # nothing runs when a batch exceeds the cap
            run.racer.run([L.Job("v1", "II-11@o5", 7, "parent_fresh"),
                           L.Job("v1", "II-12@o5", 7, "parent_fresh")], cap_exec_eq=1)
        assert ex.calls == []
        # census rows are external dS baselines that cost no budget; V rows
        # never enter the archive
        assert run.spent("total") == 0.0
        assert run.archive.unit_stats("v1")["II-11@o5"]["n"] == 2
        assert not any(L.template_of(u) in SPLIT["V"] for u in run.archive.units(t_only=False))
    finally:
        run.close()


def test_split_file_with_a_test_template_raises(tmp_path: Path) -> None:
    bad = dict(SPLIT, T=SPLIT["T"][:-1] + ["II-14"])
    cfg = _cfg(tmp_path, split_file=_split_file(tmp_path, bad))
    with pytest.raises((L.LeakageGuardError, RuntimeError)):
        L.EvoRun(cfg, L.EvoDeps(execute=Scripted(), planner_client=CountingPlanner()))


def test_loop_source_names_no_test_template() -> None:
    source = Path(L.__file__).read_text()
    for test_id in P.TEST_IDS:
        assert not re.search(r"(?<![\w-])" + re.escape(test_id) + r"(?!\d)", source), test_id


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


def test_full_run_async_generations_and_final_selection(tmp_path: Path) -> None:
    cfg, summary, ex, planner = _run(tmp_path, budget=50.0, final_reserve=12.0)
    root = cfg.root
    gens = _gens(root)
    assert len(gens) >= 3 and all(g["done"] for g in gens)
    # K = 4 proposals in every generation: 3 target slots + 1 rotating slot
    for g in gens:
        kinds = [b["slot_kind"] for b in g["briefs"]]
        assert len(kinds) == 4 and kinds[:3] == ["target"] * 3 and kinds[3] in M.ROTATION
    assert gens[0]["briefs"][3]["kind"] == "explore"  # gen 0: no pair to merge, so explore
    assert all(b["target_class"] for b in gens[0]["briefs"][:3])  # from the diagnosis cards
    assert all(b["parent_ids"] == ["v1"] for b in gens[0]["briefs"])
    # asynchronous overlap: gen g+1 planned from the state after R1 of gen g
    assert gens[1]["planned_after"] == "g0 r1"
    events = [e["event"] for e in _jsonl(root / "events.jsonl")]
    assert events.index("async_mint_launched") < events.index("r2_done")
    # at most 3 planner calls in flight, every call metered
    assert 1 <= planner.peak <= 3
    meter = json.loads((root / "budget_meter.json").read_text())
    assert meter["planner"]["calls"] == len(_jsonl(root / "mints.jsonl")) >= 12
    # every proposal got a host verdict (ledger)
    ledger = json.loads((root / "state" / "memory.json").read_text())["ledger"]["entries"]
    for g in gens:
        assert len(g["verdicts"]) == 4
        assert set(g["verdicts"].values()) <= {"confirmed", "refuted", "inconclusive", "screened"}
    assert len(ledger) == 4 * len(gens)
    assert summary["verdicts"].get("confirmed", 0) + summary["verdicts"].get("refuted", 0) >= 1
    # the hypothesis ledger reaches the prompts from gen 2 on (gen 1 is planned
    # after R1 of gen 0)
    gen2 = [p.read_text() for p in (root / "prompts").glob("g002_*.txt")]
    assert gen2 and any("=== LEDGER" in t for t in gen2)
    # the R1 fresh parent rep is in the candidates' batch; R2 has a guard
    records = _jsonl(root / "executions.jsonl")
    r1 = [r for r in records if r["batch_id"] == "g0:r1"]
    assert {r["purpose"] for r in r1} == {"r1", "parent_fresh"}
    assert any(r["purpose"] == "r2" for r in records) and any(r["purpose"] == "guard" for r in records)
    # budget: evolution <= B - reserve, total <= B; the run is budget-matched
    assert summary["spent"]["evolution"] <= cfg.evolution_budget + 1e-9
    assert summary["spent"]["total"] <= cfg.budget + 1e-9
    evo = json.loads((root / "state" / "evolution.json").read_text())
    assert summary["run_status"] == "ok" and evo["budget_matched"] and evo["reason"] == "budget"
    assert cfg.evolution_budget - summary["spent"]["evolution"] < L.MIN_GEN_COST
    assert all(g.get("fresh_parent_reps") is not None for g in summary["generations"]
               if g["status"] == "done")
    _meter_agrees(root)
    # final selection: V only there; V_frontier[:3] x 2 reps; v1 from the census
    final = json.loads((root / "state" / "final.json").read_text())
    assert final["done"] and final["candidates"][0] == "v1"
    assert final["V_units"] and all(L.template_of(u) in SPLIT["V"] for u in final["V_units"])
    vrec = [r for r in records if r["purpose"] == "vselect"]
    assert vrec and all(str(r["batch_id"]).startswith("final:") for r in vrec)
    assert not any(r["program_id"] == "v1" for r in vrec)  # the census has 2 V reps
    assert all(L.template_of(r["unit_id"]) in SPLIT["T"] for r in records if r["purpose"] != "vselect")
    champ = json.loads((root / "champion.json").read_text())
    assert champ["program_id"] == summary["champion"] == final["champion"]
    assert L._sha256(champ["source"]) == champ["sha256"]
    assert champ["lineage"][-1]["program_id"] == "v1" and champ["V_scores"]["mean_S"] is not None
    assert (root / "champion.py").read_text() == champ["source"]
    # prompts: T units only, never a V / TEST id
    for path in (root / "prompts").glob("*.txt"):
        text = path.read_text()
        for held_out in SPLIT["V"] + list(P.TEST_IDS):
            assert not re.search(r"(?<![\w-])" + re.escape(held_out) + r"(?!\d)", text), (path, held_out)
    assert summary["leak_hits"] == []
    config = json.loads((root / "config.json").read_text())
    assert config["variant"] == "full"
    # V is used exactly once: another call is a no-op (no new executions)
    n = len(ex.calls)
    again = L.run_evo(_resume(cfg), L.EvoDeps(execute=ex, planner_client=planner))
    assert len(ex.calls) == n and again["champion"] == summary["champion"]


def test_resume_after_kill_mid_r2_repeats_no_execution(tmp_path: Path) -> None:
    killer = Scripted(kill_at=("r2", 2))
    cfg = _cfg(tmp_path, budget=40.0, final_reserve=12.0)
    with pytest.raises(Kill) as info:
        L.run_evo(cfg, L.EvoDeps(execute=killer, planner_client=CountingPlanner()))
    killed = info.value.args[0]
    for thread in threading.enumerate():  # a real kill -9 leaves no planner thread behind
        if thread.name.startswith("evo-planner"):
            thread.join(timeout=30)
    root = cfg.root
    before = [r for r in _jsonl(root / "executions.jsonl") if not r.get("external")]
    assert any(r["purpose"] == "r2" for r in before)
    finished_before = {r["slot_id"] for r in _jsonl(root / "mints.jsonl")
                       if r["status"] in L.FINAL_MINT_STATUSES}
    resumed = Scripted()
    summary = L.run_evo(_resume(cfg), L.EvoDeps(execute=resumed, planner_client=CountingPlanner()))
    scored_before = {r["key"] for r in before if not r["row"].get("infra")}
    assert not scored_before & set(resumed.calls)
    assert killed not in scored_before and killed in resumed.calls
    all_calls = [k for k in killer.calls if k != killed] + resumed.calls
    assert max(Counter(all_calls).values()) == 1
    records = [r for r in _jsonl(root / "executions.jsonl") if not r.get("external")]
    assert max(Counter(r["key"] for r in records if not r["cached"]).values()) == 1
    _meter_agrees(root)
    finals = Counter(r["slot_id"] for r in _jsonl(root / "mints.jsonl")
                     if r["status"] in L.FINAL_MINT_STATUSES)
    assert finished_before and max(finals.values()) == 1  # no finished mint is re-minted
    assert summary["champion"] and (root / "champion.json").is_file()
    assert summary["spent"]["total"] <= cfg.budget + 1e-9
    deps = lambda: L.EvoDeps(execute=Scripted(), planner_client=CountingPlanner())  # noqa: E731
    with pytest.raises(L.EvoStateError, match="different config"):  # another config is refused
        L.EvoRun(_resume(cfg, arm="mf_elite"), deps())
    with pytest.raises(L.EvoStateError, match="already holds a run"):  # a new run on a used root too
        L.EvoRun(L.EvoConfig(**{**cfg.__dict__, "resume": False}), deps())


def test_infra_rows_are_rerun_never_scored_zero(tmp_path: Path) -> None:
    ex = Scripted(infra={"v1|II-11@o5|r2": 1})  # census holds reps 0, 1: the fresh rep is 2
    run = _open(tmp_path, execute=ex)
    try:
        rows = run.racer.run_batch("g0:r1", [("v1", "II-11@o5", "parent_fresh", None)], gen=0,
                                   cap_exec_eq=5)
    finally:
        run.close()
    assert rows[0]["key"] == "v1|II-11@o5|r2" and rows[0]["S"] is not None
    assert ex.calls.count("v1|II-11@o5|r2") == 2
    assert run.spent("evolution") == 1.0  # the infra attempt costs nothing


@pytest.mark.parametrize("arm,ablation", [
    ("full", {}), ("mf_elite", {}),
    ("full", {"no_diag": True}), ("full", {"no_ledger": True}), ("full", {"no_archive": True}),
    ("full", {"no_ledger": True, "no_archive": True}),
])
def test_every_arm_and_ablation_runs(tmp_path: Path, arm: str, ablation: dict) -> None:
    cfg, summary, _ex, _p = _run(tmp_path, arm=arm, budget=34.0, final_reserve=8.0,
                                 max_generations=3, **ablation)
    root = cfg.root
    flags = cfg.flags
    gens = [g for g in _gens(root) if g.get("briefs")]
    assert len(gens) >= 2
    config = json.loads((root / "config.json").read_text())
    assert config["flags"] == {"diag": flags.diag, "ledger": flags.ledger, "archive": flags.archive}
    prompts = [p.read_text() for p in sorted((root / "prompts").glob("*.txt"))]
    has = lambda marker: any(marker in t for t in prompts)  # noqa: E731
    assert has("=== FAILURE MAP") == flags.diag
    assert has("=== DIAGNOSIS CARDS") == flags.diag
    if not flags.ledger:
        assert not has("=== LEDGER")
    if not flags.archive:
        assert not has("=== CONTRASTIVE EXEMPLARS") and not has("=== MERGE PARENTS")
        assert all(b["kind"] != "merge" for g in gens for b in g["briefs"])
    for g in gens:
        assert [b["slot_kind"] for b in g["briefs"]][:3] == ["target"] * 3  # same in every arm
        for b in g["briefs"]:
            if not flags.diag:
                assert b["target_class"] is None and b["cell"] is None
        if g.get("done"):
            assert len(g["verdicts"]) == len(g["briefs"])
    assert (root / "champion.json").is_file()
    assert summary["spent"]["total"] <= cfg.budget + 1e-9
    _meter_agrees(root)


def test_mf_elite_is_full_with_every_component_off(tmp_path: Path) -> None:
    seen = {}
    for name, kw in (("elite", {"arm": "mf_elite"}),
                     ("off", {"no_diag": True, "no_ledger": True, "no_archive": True})):
        (tmp_path / name).mkdir()
        run = _open(tmp_path / name, **kw)
        try:
            assert run.flags == L.EvoFlags(False, False, False) and run.policy == "incumbent"
            state = run.plan_generation(0, after="test")
            seen[name] = ([{k: b.get(k) for k in ("kind", "target_units", "target_class",
                                                  "parent_ids")} for b in state["briefs"]],
                          [(run.root / b["prompt_path"]).read_text() for b in state["briefs"]])
        finally:
            run.close()
    assert seen["elite"] == seen["off"]


# --------------------------------------------------------------------------- #
# memory / curriculum / final selection pieces
# --------------------------------------------------------------------------- #


def test_tabu_pair_is_screened_before_execution(tmp_path: Path) -> None:
    from queenbee.evo.seed import evo_seed_source
    from queenbee.program.genome import genome_region, splice_genome

    run = _open(tmp_path)
    try:
        v1 = evo_seed_source()
        child_genome, _m = L.fake_mutation(genome_region(v1), 0)  # a wi on phase 0
        child = splice_genome(v1, child_genome)
        key = M.mech_key(M.mechanism_tags(v1, child))
        refuted = {"verdict": "refuted", "target_class": "divergent", "mech_key": key,
                   "target_units": ["II-11@o5"], "observed": {"II-11@o5": -0.2}}
        run.ledger = M.Ledger([dict(refuted, program_id="a"), dict(refuted, program_id="b")])
        assert run.ledger.tabu() == [["divergent", key]]
        run.racer.register_program("g001_s0", child, arm="full", meta={"gen": 1, "parent_ids": ["v1"]})
        brief = {"slot": 0, "slot_id": "g001_s0", "brief_id": "g1-s0", "kind": "target",
                 "target_units": ["II-11@o5"], "target_class": "divergent", "parent_ids": ["v1"]}
        state = {"gen": 1, "briefs": [brief], "done": False}
        mint = {"status": "ok", "program_id": "g001_s0",
                "hypothesis": {"failure_class": "divergent", "mechanism": "anything"}}
        run.screen_generation(1, state, [mint])
        slot = state["slots"][0]
        assert not slot["s0_pass"] and any(r.startswith("tabu") for r in slot["screen"]["reasons"])
    finally:
        run.close()


def test_curriculum_promotes_a_saturated_template_one_rung(tmp_path: Path) -> None:
    table = {t: [1.0, 1.0] for t in CENSUS_S}
    for t in ("I-01", "II-11", "III-21", "I-03", "II-16", "III-23"):
        table[t] = [0.4, 0.6]  # 3 frontier T units (< 4) + 3 frontier V units
    cfg, summary, _ex, _p = _run(tmp_path, execute=Scripted(census=table), census_S=table,
                                 budget=30.0, final_reserve=8.0, max_generations=2)
    root = cfg.root
    cur = _gens(root)[0]["curriculum"]
    assert len(cur["frontier_left"]) == 3 and len(cur["promotions"]) == 1
    promo = cur["promotions"][0]
    assert promo["from"] == "I-02@o5" and promo["to"] == "I-02@x5"
    records = _jsonl(root / "executions.jsonl")
    curr = [r for r in records if r["purpose"] == "curriculum"]
    assert curr and all(r["batch_id"] == "g0:curr" and r["unit_id"] == "I-02@x5" for r in curr)
    assert len(curr) == len(cur["promotions"])  # 1 execution per promotion
    assert "I-02@x5" in json.loads((root / "state" / "pool.json").read_text())["extra"]
    run = L.EvoRun(_resume(cfg), L.EvoDeps(execute=Scripted(census=table),
                                           planner_client=CountingPlanner()))
    try:  # the promoted unit is in play for the next plans (x5: S 0.4 < 1)
        assert "I-02@x5" in run.t_frontier() and "I-02@x5" in run.pool.pool_T()
    finally:
        run.close()
    assert summary["spent"]["total"] <= cfg.budget + 1e-9
    _meter_agrees(root)


def test_v_guard_loss_disqualifies_the_leader(tmp_path: Path) -> None:
    from queenbee.evo.seed import evo_seed_source
    from queenbee.program.genome import genome_region, splice_genome

    def execute(req: dict[str, Any]) -> dict[str, Any]:
        s = 0.4 if req["job"].program_id == "cand" else 1.0
        return {"infra": None, "S": s, "C": 1000.0, "model_calls": 10}

    run = _open(tmp_path, execute=execute)
    try:
        v1 = evo_seed_source()
        run.racer.register_program("cand", splice_genome(v1, L.fake_mutation(genome_region(v1), 1)[0]),
                                   arm="full", meta={"gen": 0, "parent_ids": ["v1"]})
        guards = run.pool.V_guard()[:2]
        assert guards and all(L.template_of(u) in SPLIT["V"] for u in guards)
        passed, detail = run._v_guard("cand", {"V_guards": guards})
        assert not passed and all(v["loss"] == 0.6 and not v["ok"] for v in detail["units"].values())
        vg = [r for r in run.racer.log.records if r.get("purpose") == "vselect"]
        assert len(vg) == len(guards) and all(r["batch_id"] == "final:vguard:cand" for r in vg)
        assert run.spent("final") == len(guards) and run.spent("evolution") == 0.0
    finally:
        run.close()


def test_final_order_ties_and_fallback() -> None:
    scores = {"v1": {"mean_S": 0.60, "tokens": 15000.0},
              "a": {"mean_S": 0.80, "tokens": 30000.0},
              "b": {"mean_S": 0.77, "tokens": 12000.0},
              "c": {"mean_S": 0.70, "tokens": 5000.0}}
    assert L.EvoRun._final_order(["v1", "a", "b", "c"], scores) == ["b", "a", "c", "v1"]
    assert L.EvoRun._final_order(["v1"], {"v1": {"mean_S": None}}) == ["v1"]


@needs_bench
def test_cli_fake_provider_smoke(tmp_path: Path, monkeypatch) -> None:
    for var in (L.PLANNER_MODEL_ENV, L.WORKER_MODEL_ENV):
        monkeypatch.delenv(var, raising=False)
    census = _census(tmp_path)
    split = _split_file(tmp_path)
    root = tmp_path / "cli"
    summary = L.main([
        "--arm", "full", "--no-ledger", "--no-archive", "--split-seed", "2", "--budget", "24",
        "--final-reserve", "6", "--root", str(root), "--fake", "--units-from", str(census),
        "--split-file", str(split), "--trace-dir", str(tmp_path / "no_traces"),
        "--max-generations", "1", "--no-traces", "--parallel-cases", "2", "--infra-backoff", "0",
    ])
    assert (root / "champion.json").is_file() and summary["champion"]
    assert summary["spent"]["total"] <= 24
    records = [r for r in _jsonl(root / "executions.jsonl") if not r.get("external")]
    assert records and all(not r["row"].get("infra") for r in records)
    config = json.loads((root / "config.json").read_text())
    assert config["fake"] is True and config["variant"] == "full-noledger-noarchive"
    assert config["flags"] == {"diag": True, "ledger": False, "archive": False}
    assert config["planner_model"] == config["worker_model"] == L.FAKE_MODEL


@needs_bench
def test_cli_fake_run_needs_no_inputs(tmp_path: Path, monkeypatch) -> None:
    """``--fake`` without ``--units-from`` / ``--split-file`` / ``--ladder-manifest``:
    a synthetic census, the split drawn from ``--split-seed`` and the run's own
    x5 ladder, end to end offline."""

    for var in (L.PLANNER_MODEL_ENV, L.WORKER_MODEL_ENV):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "fake"
    summary = L.main(["--arm", "full", "--split-seed", "1", "--budget", "30", "--root", str(root),
                      "--fake", "--max-generations", "1", "--no-traces", "--parallel-cases", "4",
                      "--infra-backoff", "0"])
    assert (root / "champion.json").is_file() and summary["champion"]
    assert summary["spent"]["total"] <= 30 and summary["leak_hits"] == []
    config = json.loads((root / "config.json").read_text())
    assert config["inputs"]["unit_sources"] == [
        ["<synthetic census>", L._sha256(json.dumps(L.fake_census(), sort_keys=True))]]
    split = json.loads((root / "state" / "split.json").read_text())
    assert len(split["T"]) == 12 and len(split["V"]) == 6
    assert not set(split["T"] + split["V"]) & set(P.TEST_IDS)
    assert split["unit_sources"] == ["<synthetic census>"]


def test_cli_http_timeouts_follow_the_request_timeout(tmp_path: Path, monkeypatch) -> None:
    """While the run lasts, unset OPENAI_TIMEOUT / OPENAI_CONNECT_TIMEOUT get
    --request-timeout / 30 s (the sandboxed workers inherit them); values the
    user set are kept, and the defaults are removed afterwards."""

    seen: list[tuple[str | None, str | None]] = []

    def run(cfg: Any, deps: Any = None) -> dict[str, Any]:
        seen.append((os.environ.get("OPENAI_TIMEOUT"), os.environ.get("OPENAI_CONNECT_TIMEOUT")))
        return {"run_status": "ok"}

    monkeypatch.setattr(L, "run_evo", run)
    for var in ("OPENAI_TIMEOUT", "OPENAI_CONNECT_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    argv = ["--arm", "full", "--split-seed", "1", "--root", str(tmp_path / "r"), "--fake",
            "--request-timeout", "123"]
    L.main(argv)
    assert seen[-1] == ("123", "30")
    assert "OPENAI_TIMEOUT" not in os.environ and "OPENAI_CONNECT_TIMEOUT" not in os.environ
    monkeypatch.setenv("OPENAI_TIMEOUT", "77")
    L.main(argv)
    assert seen[-1] == ("77", "30")
    assert os.environ["OPENAI_TIMEOUT"] == "77" and "OPENAI_CONNECT_TIMEOUT" not in os.environ


# --------------------------------------------------------------------------- #
# stop rules, outages, final selection, measurement rules
# --------------------------------------------------------------------------- #


class Outage(Scripted):
    """Every execution after the first ``after`` returns an HTTP 502 infra
    row (the service is unreachable) until ``healed`` is set."""

    def __init__(self, after: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.after = after
        self.n = 0
        self.healed = False

    def __call__(self, req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.n += 1
            down = self.n > self.after and not self.healed
        if down:
            with self._lock:
                self.calls.append(req["job"].key)
            return {"infra": "ConnectionError: 502 bad gateway", "execution_class": "infra"}
        return super().__call__(req)


class DupPlanner(L.FakeEvoPlanner):
    """Answers the parent genome unchanged (a duplicate / no-op proposal)
    for the first ``n_dup`` calls (``None`` = always)."""

    def __init__(self, n_dup: int | None) -> None:
        super().__init__()
        self.n_dup = n_dup
        self.k = 0
        self._g = threading.Lock()

    def complete(self, prompt: str, *a: Any, **kw: Any) -> Any:
        from types import SimpleNamespace

        with self._g:
            self.k += 1
            k = self.k
        if self.n_dup is None or k <= self.n_dup:
            genome = L._PARENT_GENOME_RE.search(prompt).group(1)
            return SimpleNamespace(
                text='HYPOTHESIS: {"failure_class": "divergent", "mechanism": "x"}\n```python\n'
                + genome + "\n```\n", usage={"prompt_tokens": 1, "completion_tokens": 1})
        return super().complete(prompt, *a, **kw)


def _join_planner_threads() -> None:
    """A stopped run's planner threads die with its process (the CLI exits
    hard); in-process tests wait for them before resuming."""

    for thread in threading.enumerate():
        if thread.name.startswith("evo-planner"):
            thread.join(timeout=60)


def test_worker_outage_stops_resumable_and_burns_no_attempts(tmp_path: Path) -> None:
    ex = Outage(after=6)
    cfg = _cfg(tmp_path, budget=40.0, final_reserve=12.0)
    with pytest.raises(L.InfraOutage):
        L.run_evo(cfg, L.EvoDeps(execute=ex, planner_client=CountingPlanner()))
    root = cfg.root
    assert not (root / "state" / "evolution.json").exists()  # nothing marked done
    assert not (root / "state" / "final.json").exists()
    assert json.loads((root / "state" / "outage.json").read_text())["reason"]
    records = [r for r in _jsonl(root / "executions.jsonl") if not r.get("external")]
    infra = Counter(r["key"] for r in records if r["row"].get("infra"))
    # sequential batch: the first failing job may use its own <= 3 attempts
    # before the streak spans a second unit (one job alone cannot be told
    # apart from a job-specific failure); every later job is held back
    exhausted = [k for k, n in infra.items() if n >= 1 + cfg.max_infra_reruns]
    assert len(exhausted) <= 1 and sum(infra.values()) <= 1 + cfg.max_infra_reruns
    scored_before = {r["key"] for r in records if not r["row"].get("infra")}
    _join_planner_threads()
    # the service is back: the same command resumes and finishes budget-matched
    ex.healed = True
    ex.calls.clear()
    summary = L.run_evo(_resume(cfg), L.EvoDeps(execute=ex, planner_client=CountingPlanner()))
    assert not scored_before & set(ex.calls)
    assert summary["run_status"] == "ok" and summary["evolution"]["budget_matched"]
    assert cfg.evolution_budget - summary["spent"]["evolution"] < L.MIN_GEN_COST
    _meter_agrees(root)


def test_unit_specific_infra_is_not_an_outage() -> None:
    rows = iter([{"infra": "ConnectionError: 502"}] * 5)
    guard = L.OutageGuard(lambda req: next(rows), wait_s=0.0)
    job = lambda pid, unit: {"job": L.Job(pid, unit, 1, "vselect")}  # noqa: E731
    for pid in ("a", "b", "c"):  # one unit for every program: never an outage
        assert guard(job(pid, "III-23@o5"))["infra"]
    guard2 = L.OutageGuard(lambda req: {"infra": "LLM call exceeded 1800s"}, wait_s=0.0)
    for unit in ("I-01@o5", "II-11@o5", "III-21@o5"):  # not a connectivity error
        assert guard2(job("a", unit))["infra"]
    guard3 = L.OutageGuard(lambda req: {"infra": "APIConnectionError: Connection refused"},
                           wait_s=0.0)
    guard3(job("a", "I-01@o5"))
    guard3(job("b", "II-11@o5"))
    with pytest.raises(L.InfraOutage):
        guard3(job("c", "III-21@o5"))
    with pytest.raises(L.InfraOutage):  # tripped: the next job does not even run
        guard3(job("d", "I-02@o5"))
    # a rejected request (e.g. a bad key) is never an outage to wait out
    guard4 = L.OutageGuard(lambda req: {"infra": "PermissionDeniedError: Error code: 403"},
                           wait_s=0.0)
    for unit in ("I-01@o5", "II-11@o5", "III-21@o5", "I-02@o5"):
        assert guard4(job("a", unit))["infra"]
    assert L.is_outage_error(ConnectionError("HTTP 502 Bad Gateway"))
    assert L.is_outage_error(OSError("[Errno 51] Network is unreachable"))
    assert not L.is_outage_error(RuntimeError("Error code: 403 - Forbidden"))
    assert not L.is_outage_error(ValueError("invalid request: unknown field"))


@pytest.mark.parametrize("arm", ["mf_elite", "full"])
def test_duplicate_proposals_do_not_end_evolution(tmp_path: Path, arm: str) -> None:
    cfg, summary, _ex, _p = _run(tmp_path, arm=arm, planner=DupPlanner(12),
                                 budget=40.0, final_reserve=12.0)
    evo = json.loads((cfg.root / "state" / "evolution.json").read_text())
    assert evo["reason"] == "budget" and evo["budget_matched"] and evo["generations"] > 3
    assert summary["run_status"] == "ok"
    assert cfg.evolution_budget - summary["spent"]["evolution"] < L.MIN_GEN_COST


def test_idle_cap_is_a_failed_run(tmp_path: Path) -> None:
    cfg, summary, ex, _p = _run(tmp_path, planner=DupPlanner(None), budget=30.0,
                                final_reserve=8.0, max_idle_generations=2)
    evo = json.loads((cfg.root / "state" / "evolution.json").read_text())
    assert evo["reason"] == "idle" and evo["status"] == "failed" and not evo["budget_matched"]
    assert summary["run_status"] == "failed:idle" and summary["spent"]["evolution"] == 0.0
    assert evo["generations"] == 2 and "not budget-matched" in evo["note"]
    assert (cfg.root / "champion.json").is_file()
    assert json.loads((cfg.root / "champion.json").read_text())["run_status"] == "failed:idle"


class FlakyPlanner(L.FakeEvoPlanner):
    """The first ``n_fail`` calls raise ``exc``; ``kill_at`` raises Kill."""

    def __init__(self, n_fail: int, exc: Exception, kill_at: int | None = None) -> None:
        super().__init__()
        self.n_fail, self.exc, self.kill_at, self.n = n_fail, exc, kill_at, 0
        self._g = threading.Lock()

    def complete(self, prompt: str, *a: Any, **kw: Any) -> Any:
        with self._g:
            self.n += 1
            n = self.n
        if self.kill_at is not None and n == self.kill_at:
            raise Kill(f"kill -9 during planner call {n}")
        if n <= self.n_fail:
            raise self.exc
        return super().complete(prompt, *a, **kw)


def test_planner_outage_stops_resumable_without_losing_proposals(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, budget=30.0, final_reserve=8.0, planner_concurrency=1)
    down = FlakyPlanner(10**6, OSError("[Errno 51] Network is unreachable"))
    with pytest.raises(L.InfraOutage):
        L.run_evo(cfg, L.EvoDeps(execute=Scripted(), planner_client=down))
    _join_planner_threads()
    mints = _jsonl(cfg.root / "mints.jsonl")
    assert mints and not any(m["status"] in L.FINAL_MINT_STATUSES for m in mints)
    assert {m["status"] for m in mints} <= {"planner_error", "planner_outage"}
    assert not (cfg.root / "state" / "evolution.json").exists()
    summary = L.run_evo(_resume(cfg), L.EvoDeps(execute=Scripted(), planner_client=CountingPlanner()))
    assert summary["run_status"] == "ok"
    mints = _jsonl(cfg.root / "mints.jsonl")
    assert not any(m["status"] == "planner_error_final" for m in mints)
    finals = Counter(m["slot_id"] for m in mints if m["status"] in L.FINAL_MINT_STATUSES)
    assert max(finals.values()) == 1


def test_deterministic_planner_error_is_final_after_retries(tmp_path: Path, monkeypatch) -> None:
    run = _open(tmp_path)
    try:
        run.planner_client = lambda: FlakyPlanner(10**6, ValueError("400: invalid field"))  # type: ignore[method-assign]
        state = run.plan_generation(0, after="test")
        rec = run.mint_slot(0, state["briefs"][0])
        assert rec["status"] == "planner_error_final" and rec["try"] == L.MAX_PLANNER_ERROR_RETRIES + 1
    finally:
        run.close()


def test_crash_during_a_planner_retry_keeps_counters_and_meters_every_call(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, budget=40.0, final_reserve=12.0)
    flaky = FlakyPlanner(1, RuntimeError("HTTP 403 Forbidden"), kill_at=2)
    run = L.EvoRun(cfg, L.EvoDeps(execute=Scripted(), planner_client=flaky))
    state = run.plan_generation(0, after="test")
    brief = state["briefs"][0]
    with pytest.raises(Kill):
        run.mint_slot(0, brief)
    run.close()
    run2 = L.EvoRun(_resume(cfg), L.EvoDeps(execute=Scripted(), planner_client=L.FakeEvoPlanner()))
    rec = run2.mint_slot(0, brief)
    run2.close()
    lines = _jsonl(cfg.root / "mints.jsonl")
    assert [(m["try"], m["status"]) for m in lines] == [(1, "planner_error"), (2, "ok")]
    assert rec["planner_error_total"] == 1 and len({m["call_id"] for m in lines}) == 2
    run3 = L.EvoRun(_resume(cfg), L.EvoDeps(execute=Scripted(), planner_client=L.FakeEvoPlanner()))
    try:
        snap = run3.meter.snapshot()
        with_usage = sum(len(m["usage"]) for m in lines)
        # every recorded call once + the call lost to the kill (unknown usage)
        assert snap["planner"]["calls"] == with_usage + 1
        assert snap["counters"]["lost_inflight_planner_calls"] == 1
    finally:
        run3.close()


class VInfra(Scripted):
    """vselect jobs of evolved programs on one V unit return infra rows: on
    every rep, or only on the reps in ``reps`` (a replacement rep has
    another rep index and runs normally)."""

    def __init__(self, unit_template: str, reps: set[int] | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.unit_template, self.reps = unit_template, reps

    def __call__(self, req: dict[str, Any]) -> dict[str, Any]:
        job = req["job"]
        if job.purpose == "vselect" and job.program_id != "v1" \
                and L.template_of(job.unit_id) == self.unit_template \
                and (self.reps is None or job.rep in self.reps):
            with self._lock:
                self.calls.append(job.key)
            return {"infra": "ConnectionError: 502 bad gateway", "execution_class": "infra"}
        return super().__call__(req)


def test_final_selection_never_ranks_incomplete_candidates(tmp_path: Path) -> None:
    cfg, summary, _ex, _p = _run(tmp_path, execute=VInfra("III-23"), budget=50.0, final_reserve=12.0)
    final = json.loads((cfg.root / "state" / "final.json").read_text())
    evolved = [p for p in final["candidates"] if p != "v1"]
    assert evolved and final["done"]
    assert set(final["incomplete"]) == set(evolved)  # III-23 never scored for them
    assert final["order"] == ["v1"] and final["champion"] == "v1" == summary["champion"]
    assert final["replacements"] and all(r["batch_id"].startswith("final:vselect:fix")
                                         for r in final["replacements"])
    assert final["V_scores"]["v1"]["complete"]
    assert summary["spent"]["total"] <= cfg.budget + 1e-9
    _meter_agrees(cfg.root)


def test_final_selection_replaces_infra_exhausted_v_jobs(tmp_path: Path) -> None:
    cfg, summary, _ex, _p = _run(tmp_path, execute=VInfra("III-23", reps={1}), budget=50.0,
                                 final_reserve=12.0)
    final = json.loads((cfg.root / "state" / "final.json").read_text())
    evolved = [p for p in final["candidates"] if p != "v1"]
    assert evolved and not final["incomplete"] and final["replacements"]
    for pid in evolved:  # every V unit x 2 scored reps (rep 1 replaced by another rep)
        per_unit = final["V_scores"][pid]["per_unit"]
        assert all(len(v["S"]) == 2 for v in per_unit.values())
        means = [v["mean"] for v in per_unit.values()]
        assert final["V_scores"][pid]["mean_S"] == pytest.approx(sum(means) / len(means), abs=1e-4)
    assert summary["spent"]["total"] <= cfg.budget + 1e-9


def _three_programs(run: L.EvoRun) -> dict[str, str]:
    from queenbee.evo.seed import evo_seed_source
    from queenbee.program.genome import genome_region, splice_genome

    v1 = evo_seed_source()
    genome = genome_region(v1)
    out = {}
    for name, index, parents in (("pa", 0, ["v1"]), ("pb", 1, ["v1"]), ("m", 2, ["pa", "pb"])):
        out[name] = splice_genome(v1, L.fake_mutation(genome, index)[0])
        run.racer.register_program(name, out[name], arm="full",
                                   meta={"gen": 0, "parent_ids": parents})
    return out


def test_merge_child_is_measured_against_its_best_parent(tmp_path: Path) -> None:
    S = {"pa": 0.4, "pb": 0.8, "m": 0.8}

    def execute(req: dict[str, Any]) -> dict[str, Any]:
        return {"infra": None, "S": S[req["job"].program_id], "C": 1000.0, "model_calls": 10}

    run = _open(tmp_path, execute=execute)
    try:
        _three_programs(run)
        unit = "II-11@o5"
        run.racer.run_batch("g0:r2", [("pb", unit, "r2", None)], gen=0, cap_exec_eq=10)
        cand = {"program_id": "m", "parent_ids": ["pa", "pb"], "target_units": [unit],
                "brief_id": "g1-s3", "kind": "merge", "target_class": None, "rank_penalty": 0.0}
        state = {"r1": {"cands": [cand]}}
        res = run._r1_result(1, state)
        c = res["candidates"][0]
        assert c["fresh_S"] == 0.4 and c["d1_vs_first_parent"] == pytest.approx(0.4)
        assert c["d1"] == pytest.approx(0.0) and c["merge_bar"]["src"].startswith("merge:pb")
        outcome = run.racer.outcome(c, 1)
        assert outcome["observed"][unit] == pytest.approx(0.0)  # not +0.4 vs the weaker parent
        assert outcome["baseline_rule"].startswith("merge")
        # outside a merge child's outcome the parent keeps its own (fresh) baseline
        assert run.racer.baseline("pa", unit, 1) == (pytest.approx(0.4), "fresh")
    finally:
        run.close()


def test_economize_verdict_and_loop_outcome(tmp_path: Path) -> None:
    ev = L.economize_verdict
    base = {"executed": True, "target_dS": 0.0, "guard_ok": True}
    assert ev({"executed": False}, token_ratio=None) == "screened"
    assert ev(base, token_ratio=0.7) == "confirmed"
    assert ev(base, token_ratio=0.9) == "inconclusive"
    assert ev(base, token_ratio=1.0) == "refuted"
    assert ev({**base, "target_dS": -0.2}, token_ratio=0.5) == "refuted"
    assert ev({**base, "guard_ok": None}, token_ratio=0.5) == "inconclusive"
    run = _open(tmp_path)
    try:
        unit = "II-17@o5"
        fake = {"program_id": "e", "executed": True, "target_dS": 0.0, "guard_ok": True,
                "target_units_executed": [unit], "verdict": "refuted", "status": "executed",
                "per_unit": {unit: {"C": 7000.0, "baseline_C": 10000.0, "baseline_src": "fresh"}}}
        run.racer.outcome = lambda cand, gen, r2_plan=None: dict(fake)  # type: ignore[method-assign]
        out = run._outcome(1, {"kind": "economize", "parent_ids": ["v1"]}, {"program_id": "e"}, None)
        assert out["verdict"] == "confirmed" and out["token_ratio"] == pytest.approx(0.7)
        entry = {"verdict": "refuted"}
        run._annotate_entry(entry, out)
        assert entry["verdict"] == "confirmed" and entry["verdict_rule"] == "economize"
        other = run._outcome(1, {"kind": "target", "parent_ids": ["v1"]}, {"program_id": "e"}, None)
        assert other["verdict"] == "refuted" and "verdict_rule" not in other
    finally:
        run.close()


def test_unraced_proposals_are_screened_not_executed(tmp_path: Path) -> None:
    run = _open(tmp_path)
    try:
        progs = _three_programs(run)
        brief = {"slot": 0, "slot_id": "g001_s0", "brief_id": "g1-s0", "kind": "target",
                 "target_units": ["II-11@o5"], "target_class": "divergent", "parent_ids": ["v1"]}
        slot = {"slot_id": "g001_s0", "brief_id": "g1-s0", "slot": 0, "kind": "target",
                "status": "ok", "program_id": "pa", "parent_id": "v1", "s0_pass": True,
                "hypothesis": None, "screen": {"ok": True, "reasons": []}}
        state = {"gen": 1, "briefs": [brief], "slots": [slot],
                 "r1": {"cands": [], "status": "budget_stop", "dropped_budget": ["pa"]}}
        run.verdicts(1, state)
        entry = run.ledger.entries[-1]
        assert entry["verdict"] == "screened" and entry["status"] == "not_raced_budget"
        assert run.ledger.summary()["confirmed_rate"] is None  # not an executed proposal
        assert progs
    finally:
        run.close()


def test_screening_resume_re_adds_programs_to_the_archive(tmp_path: Path) -> None:
    run = _open(tmp_path)
    try:
        _three_programs(run)
        brief = {"slot": 0, "slot_id": "g000_s0", "brief_id": "g0-s0", "kind": "target",
                 "target_units": ["II-11@o5"], "target_class": "divergent", "parent_ids": ["v1"]}
        slot = {"slot_id": "g000_s0", "slot": 0, "status": "ok", "program_id": "pa",
                "s0_pass": True, "hypothesis": {"mechanism": "x"}}
        # the generation file holds the screened slot, memory.json does not (a kill in between)
        state = {"gen": 0, "briefs": [brief], "slots": [slot]}
        assert "pa" not in run.archive.programs
        run.screen_generation(0, state, [])
        rec = run.archive.programs["pa"]
        assert rec["parent_ids"] == ["v1"] and rec["fp"] and rec["brief_id"] == "g0-s0"
    finally:
        run.close()


def test_prompt_value_hits_are_redacted_and_id_hits_stop(tmp_path: Path) -> None:
    from types import SimpleNamespace

    text = ("=== PARENT GENOME ===\nX = 12345\n=== PARENT RECORD (x) ===\n- u: tokens=12345 S=0.50\n"
            "=== OUTPUT ===\n12345\n")
    red, n = L.redact_evidence_values(text, [12345])
    assert n == 1 and "tokens=[redacted]" in red
    assert red.count("12345") == 2  # program text and non-evidence sections untouched
    run = _open(tmp_path)
    try:
        state = run.plan_generation(0, after="test")
        brief = state["briefs"][0]
        clean, n0 = run.build_prompt(brief, state["t_units"])
        assert n0 == 0
        match = re.search(r"S=(\d\.\d\d) \(", clean)  # a number rendered in PARENT RECORD
        run.forbidden_values_T = lambda: [match.group(1)]  # type: ignore[method-assign]
        redacted, n1 = run.build_prompt(brief, state["t_units"])
        assert n1 >= 1 and "[redacted]" in redacted
        run.pool.split = SimpleNamespace(T=run.pool.split.T, V=tuple(run.pool.split.V)
                                         + (L.template_of(brief["target_units"][0]),))
        with pytest.raises(L.LeakageGuardError):  # a held-out id reaching a prompt stops the run
            run.build_prompt(brief, state["t_units"])
    finally:
        run.close()


def test_inputs_are_pinned_and_real_runs_must_name_them(tmp_path: Path) -> None:
    with pytest.raises(L.EvoStateError, match="--units-from"):
        L.EvoRun(L.EvoConfig(root=tmp_path / "real", arm="full", split_seed=1))
    assert not (tmp_path / "real").exists()
    cfg = _cfg(tmp_path)
    run = _open(tmp_path)
    run.close()
    config = json.loads((cfg.root / "config.json").read_text())
    census = cfg.unit_sources[0]
    assert config["inputs"]["unit_sources"] == [[str(census), L._file_sha(census)]]
    assert config["inputs"]["split_file"][1] == L._file_sha(cfg.split_file)
    census.write_text(census.read_text().replace('"S": 0.4', '"S": 0.6', 1))
    with pytest.raises(L.EvoStateError, match="different config"):
        L.EvoRun(_resume(cfg), L.EvoDeps(execute=Scripted(), planner_client=CountingPlanner()))


def test_write_split_is_deterministic_and_never_overwritten(tmp_path: Path) -> None:
    run = _open(tmp_path)
    run.close()
    census = run.cfg.unit_sources[0]
    manifest = run.root / "ladder" / "manifest_x5.json"
    out = tmp_path / "splits" / "split_s1.json"
    doc = L.main(["--write-split", str(out), "--split-seed", "1", "--units-from", str(census),
                  "--ladder-manifest", str(manifest)])
    assert len(doc["T"]) == 12 and len(doc["V"]) == 6 and not set(doc["T"]) & set(doc["V"])
    assert not set(doc["T"] + doc["V"]) & set(P.TEST_IDS)
    assert doc["unit_sources"] == [[str(census), L._file_sha(census)]]
    again = L.write_split_file(out, split_seed=1, unit_sources=[census], ladder_manifests=[manifest])
    assert (again["T"], again["V"]) == (doc["T"], doc["V"])
    other = dict(doc, V=doc["V"][::-1][:5] + doc["T"][:1])
    out.write_text(json.dumps(other))
    with pytest.raises(L.EvoStateError, match="another split"):
        L.write_split_file(out, split_seed=1, unit_sources=[census], ladder_manifests=[manifest])


_KILL_CHILD = r'''
import os, signal, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[3]]
import test_evo_loop as T
from queenbee.evo import loop as L

class KillR2(T.Scripted):
    r2 = 0

    def __call__(self, req):
        if req["job"].purpose == "r2":
            with self._lock:
                self.r2 += 1
                n = self.r2
            if n == 2:
                os.kill(os.getpid(), signal.SIGKILL)   # a real kill -9: nothing more is logged
        return T.Scripted.__call__(self, req)

cfg = T._cfg(Path(sys.argv[2]), budget=40.0, final_reserve=12.0)
L.run_evo(cfg, L.EvoDeps(execute=KillR2(), planner_client=T.CountingPlanner()))
'''


@needs_bench
def test_real_kill_minus_9_mid_r2_then_resume(tmp_path: Path) -> None:
    import signal
    import subprocess
    import sys

    script = tmp_path / "child.py"
    script.write_text(_KILL_CHILD)
    src = Path(L.__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, str(script), str(Path(__file__).parent), str(tmp_path),
                           str(src)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == -signal.SIGKILL, proc.stderr[-2000:]
    cfg = _cfg(tmp_path, budget=40.0, final_reserve=12.0)  # same bytes: same pinned inputs
    root = cfg.root
    before = [r for r in _jsonl(root / "executions.jsonl") if not r.get("external")]
    assert any(r["purpose"] == "r2" for r in before)
    lost = L.lost_inflight(root / "exec_inflight.jsonl")
    assert [r["key"] for r in lost] and all("|" in r["key"] for r in lost)  # the killed r2 job
    scored_before = {r["key"] for r in before if not r["row"].get("infra")}
    resumed = Scripted()
    summary = L.run_evo(_resume(cfg), L.EvoDeps(execute=resumed, planner_client=CountingPlanner()))
    assert not scored_before & set(resumed.calls)  # no scored execution repeated
    assert {r["key"] for r in lost} <= set(resumed.calls)  # the lost job is re-run
    records = [r for r in _jsonl(root / "executions.jsonl") if not r.get("external")]
    assert max(Counter(r["key"] for r in records if not r["cached"]
                       and not r["row"].get("infra")).values()) == 1
    _meter_agrees(root)
    meter = json.loads((root / "budget_meter.json").read_text())
    assert meter["counters"]["lost_inflight_executions"] == len(lost)
    finals = Counter(r["slot_id"] for r in _jsonl(root / "mints.jsonl")
                     if r["status"] in L.FINAL_MINT_STATUSES)
    assert max(finals.values()) == 1
    assert summary["run_status"] == "ok" and (root / "champion.json").is_file()


@pytest.mark.parametrize("arm,no_archive", [("mf_elite", False), ("full", True)])
def test_arms_without_c3_get_the_parents_view(tmp_path: Path, monkeypatch, arm: str,
                                              no_archive: bool) -> None:
    captured: dict[str, Any] = {}
    original = M.prompt_state

    def spy(**kw: Any) -> Any:
        captured.update(kw)
        return original(**kw)

    monkeypatch.setattr(M, "prompt_state", spy)
    run = _open(tmp_path, arm=arm, no_archive=no_archive)
    try:
        state = run.plan_generation(0, after="test")
        assert captured["parent_id"] == state["briefs"][-1]["parent_ids"][0]
        assert run.scheduler.view_program(run.archive) is not None  # never the whole archive
    finally:
        run.close()


def test_planner_client_uses_its_own_endpoint_and_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PLANNER_KEY", "planner-secret")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    run = _open(tmp_path, planner_base_url="http://127.0.0.1:9/v1", planner_api_key_env="PLANNER_KEY")
    try:
        client = run._real_planner_client(100.0)
    finally:
        run.close()
    chain = []
    while client is not None and len(chain) < 8:
        chain.append(client)
        client = getattr(client, "_inner", None)
    names = [type(c).__name__ for c in chain]
    # ConnectRetry(Limiter(Timeout(OpenAI(retries=0)))); the limiter only when configured
    assert names[0] == "_ArmConnectRetryClient" and names[-1] == "OpenAIChatClient"
    assert names.index("TimeoutLLMClient") < names.index("OpenAIChatClient")
    sdk = chain[-1]._client
    assert str(sdk.base_url).rstrip("/") == "http://127.0.0.1:9/v1"
    assert sdk.api_key == "planner-secret" and sdk.max_retries == 0


def _inherited_source(v1: str, wi: str = "Report every value you computed and every raw item you hold.") -> str:
    """The seed program (v1) with a phase work instruction: the Scripted
    executor lifts tier II / III by 0.4."""

    old = '{"kind": "broadcast_last", "rounds": 1}'
    assert v1.count(old) == 1
    return v1.replace(old, '{"kind": "broadcast_last", "rounds": 1, "wi": ' + json.dumps(wi) + '}')


def test_inherited_program_seeds_the_archive_and_is_charged(tmp_path: Path) -> None:
    from queenbee.evo.seed import evo_seed_source

    src = tmp_path / "lineage.py"
    src.write_text(_inherited_source(evo_seed_source()))
    cfg, summary, ex, _planner = _run(tmp_path, budget=50.0, final_reserve=12.0,
                                      inherit_programs=(("lineage", src),))
    root = cfg.root
    assert cfg.variant == "full-inherit"
    marker = json.loads((root / "state" / "inherit.json").read_text())
    assert marker["done"] and list(marker["programs"]) == ["inh_lineage"]
    records = _jsonl(root / "executions.jsonl")
    inh = [r for r in records if r["purpose"] == "inherit"]
    assert inh and {r["unit_id"] for r in inh} == set(marker["units"])
    assert all(L.template_of(r["unit_id"]) in SPLIT["T"] for r in inh)
    assert all(r["program_id"] == "inh_lineage" and not r["cached"] for r in inh)
    # charged to the evolution budget (arms stay budget-matched)
    assert summary["spent"]["evolution"] <= cfg.evolution_budget + 1e-9
    _meter_agrees(root)
    # the inherited program is a parent from generation 0 on (its tier II / III units lead)
    gen0 = _gens(root)[0]
    assert any("inh_lineage" in b["parent_ids"] for b in gen0["briefs"])
    frozen = json.loads((root / "config.json").read_text())
    assert frozen.get("inherit_programs") or frozen.get("frozen", {}).get("inherit_programs")
    # a resume never re-seeds
    n = len(ex.calls)
    L.run_evo(_resume(cfg), L.EvoDeps(execute=ex, planner_client=CountingPlanner()))
    assert len(ex.calls) == n


def test_inherited_program_naming_a_v_template_is_refused(tmp_path: Path) -> None:
    from queenbee.evo.seed import evo_seed_source

    src = tmp_path / "leaky.py"
    src.write_text(_inherited_source(evo_seed_source(), wi="Handle II-16 like the others."))
    with pytest.raises(L.EvoStateError, match="TEST / V"):
        _run(tmp_path, budget=40.0, final_reserve=12.0, inherit_programs=(("leaky", src),))


def test_irreducible_templates_are_pinned_regardless_of_traces(tmp_path: Path) -> None:
    """Benchmark rounding floors are a property of the benchmark, not of the
    worker: --irreducible-templates marks them irreducible even without
    near-miss traces (e.g. a weak worker that fails for other reasons)."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base = _open(tmp_path / "a")
    try:
        assert base.pool.klass("II-12@o5") == "frontier"  # 0.4 / 0.6, no traces
    finally:
        base.close()
    run = _open(tmp_path / "b", irreducible_templates=("II-12",))
    try:
        assert run.pool.klass("II-12@o5") == "irreducible"
        assert "II-12@o5" not in run.pool.census_frontier_T()
        assert run.pool.klass("II-11@o5") == "frontier"  # others untouched
        cfg = json.loads((run.root / "config.json").read_text())
        assert cfg.get("irreducible_templates") == ["II-12"]
    finally:
        run.close()
