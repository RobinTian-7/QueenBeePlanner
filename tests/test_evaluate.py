"""queenbee.evaluate: resume, manifest, split and integrity plumbing on
synthetic Silo-Bench files (no network, no API key)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from queenbee import evaluate as ev

ALL_TEMPLATES = tuple(
    [f"I-{i:02d}" for i in range(1, 11)]
    + [f"II-{i}" for i in range(11, 21)]
    + [f"III-{i}" for i in range(21, 31)]
)
TEST12 = ("I-04", "I-05", "I-09", "I-10", "II-14", "II-15", "II-19", "II-20",
          "III-24", "III-25", "III-29", "III-30")
DEV18 = tuple(t for t in ALL_TEMPLATES if t not in TEST12)


def _instance_doc(template: str, n_agents: int, salt: int) -> dict:
    """A small global-maximum instance in the Silo-Bench file layout."""

    shards = [[salt + 3 * a + 1, salt + a + 7] for a in range(n_agents)]
    best = max(v for shard in shards for v in shard)
    return {
        "case_id": template,
        "case_name": f"Global Max {template}",
        "paradigm": "Paradigm I",
        "metadata": {"num_agents": n_agents, "output_type": "distributed", "is_segmented": False},
        "task_description": (
            "Find the GLOBAL MAXIMUM across all agents' data. "
            "You are Agent {agent_id} and hold: {input_shard}\n"
            "**Output:** A single integer."
        ),
        "agent_configs": [
            {"agent_id": a, "input_shard": shard, "expected_output": best}
            for a, shard in enumerate(shards)
        ],
    }


def write_bench(directory: Path, n_agents: int = 2) -> Path:
    """``<template>_n<n>.json`` for all 30 templates (distinct content)."""

    directory.mkdir(parents=True, exist_ok=True)
    for salt, template in enumerate(ALL_TEMPLATES):
        doc = _instance_doc(template, n_agents, salt)
        (directory / f"{template}_n{n_agents}.json").write_text(json.dumps(doc))
    return directory


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_REASONING_EFFORT",
                 "OPENAI_TIMEOUT", "OPENAI_CONNECT_TIMEOUT", "QUEENBEE_WORKER_MODEL",
                 "QB_TRACE_DIR", "SILO_BENCH_DIR"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ==========================================================================
# resume planning / merging
# ==========================================================================


def _ok(case: str, seed: int = 0, s: float = 1.0) -> dict:
    return {"case_id": case, "seed": seed, "success": s >= 1.0, "S": s, "C": 10.0, "infra": None}


def _bad(case: str, seed: int = 0, history: int = 0) -> dict:
    row = {"case_id": case, "seed": seed, "infra": "RuntimeError: 503"}
    if history:
        row["infra_history"] = ["RuntimeError: 503"] * history
    return row


def _runner(outcomes: dict[str, list[str]], calls: list):
    """A ``run_cases`` stub that records each call and emits one row per
    case: ``outcomes[case]`` is consumed in order ('ok' / 'bad'; 'ok' once
    it is empty)."""

    def run_cases(rep, todo, seeds, attempt, emit):
        calls.append((rep, list(todo), list(seeds), attempt))
        for c, s in zip(todo, seeds):
            kind = outcomes.get(c, []).pop(0) if outcomes.get(c) else "ok"
            emit(_ok(c, s) if kind == "ok" else _bad(c, s))

    return run_cases


def test_plan_repeat_lists_infra_and_missing_cases_and_honours_rerun_cap():
    done = {"rows": [_ok("A"), _bad("B"), _ok("C")]}
    assert ev.plan_repeat(done, ["A", "B", "C", "D"]) == ["B", "D"]
    assert ev.plan_repeat(None, ["A", "B"]) == ["A", "B"]
    assert ev.plan_repeat({"rows": [_ok("A")]}, ["A"]) == []
    # 1 first failure + 3 re-runs = 4 failed executions -> no more re-runs
    assert ev.plan_repeat({"rows": [_bad("B", history=2)]}, ["B"]) == ["B"]
    assert ev.plan_repeat({"rows": [_bad("B", history=3)]}, ["B"]) == []
    assert ev.plan_repeat({"rows": [_bad("B") | {"infra_final": True}]}, ["B"]) == []
    assert ev.plan_repeat({"rows": [_bad("B", history=3)]}, ["B"], max_infra_reruns=5) == ["B"]


def test_merge_repeat_union_semantics_history_and_final_flag():
    done = {"rows": [_ok("A", 1000), _bad("B", 1001), _ok("X", 1002)], "wall_s": 5.0}
    merged = ev.merge_repeat(done, ["A", "B"], [_ok("B", 1001, 0.5)], wall_s=2.0, at="t")
    # X is outside the requested layout but stays a scored row (never hidden)
    assert [r["case_id"] for r in merged["rows"]] == ["A", "B", "X"]
    assert merged["rows"][0] == _ok("A", 1000)
    assert merged["rows"][1]["S"] == 0.5
    assert merged["rows"][1]["infra_history"] == ["RuntimeError: 503"]
    assert merged["aggregate"]["infra"] == 0 and merged["aggregate"]["cases_scored"] == 3
    assert merged["wall_s"] == 7.0
    assert merged["reruns"] == [{"at": "t", "cases": ["B"], "missing": [], "infra_retry": ["B"],
                                 "still_infra": [], "wall_s": 2.0}]
    with pytest.raises(ValueError, match="scored row"):
        ev.merge_repeat(done, ["A"], [_ok("A")], wall_s=0.0)
    # the 4th failed execution is final
    last = ev.merge_repeat({"rows": [_bad("B", history=2)]}, ["B"], [_bad("B")], wall_s=0.0)
    assert last["rows"][0]["infra_final"] is True and len(last["rows"][0]["infra_history"]) == 3
    assert last["aggregate"]["infra_final"] == 1
    # a first pass has no reruns log
    first = ev.merge_repeat(None, ["A", "B"], [_ok("B"), _ok("A")], wall_s=1.0)
    assert [r["case_id"] for r in first["rows"]] == ["A", "B"] and "reruns" not in first


def test_run_arm_reruns_only_infra_rows_and_resumes_to_completion():
    calls: list = []
    run_cases = _runner({"B": ["bad", "bad"]}, calls)  # B fails twice, then succeeds
    entry: dict = {}
    logs: list[str] = []
    saves: list[int] = []
    cases = ["A", "B", "C"]
    for _ in range(3):  # three launches with the same --out
        ev.run_arm(entry, "arm", cases, 2, run_cases, log=logs.append,
                   save=lambda: saves.append(1))
    assert calls[0] == (0, ["A", "B", "C"], [1000, 1001, 1002], 0)
    assert calls[1] == (1, ["A", "B", "C"], [1100, 1101, 1102], 0)
    assert calls[2] == (0, ["B"], [1001], 1)
    assert calls[3] == (1, ["B"], [1101], 1)
    assert len(calls) == 4  # third launch: everything cached
    assert len(saves) == 8  # one checkpoint per executed case
    for key in ("0", "1"):
        rep = entry["repeats"][key]
        assert rep["aggregate"]["infra"] == 0
        assert [r["case_id"] for r in rep["rows"]] == cases
        assert len(rep["reruns"]) == 1 and rep["reruns"][0]["infra_retry"] == ["B"]
    assert entry["summary"]["repeats"] == 2 and entry["summary"]["success_rate"] == 1.0
    assert sum("cached" in line for line in logs) == 2


def test_run_arm_caps_infra_reruns_and_marks_final():
    calls: list = []
    run_cases = _runner({"B": ["bad"] * 10}, calls)
    entry: dict = {}
    for _ in range(6):
        ev.run_arm(entry, "arm", ["A", "B"], 1, run_cases, log=lambda t: None)
    # first pass + 3 re-runs of B, then never again
    assert [c[1] for c in calls] == [["A", "B"], ["B"], ["B"], ["B"]]
    row = entry["repeats"]["0"]["rows"][1]
    assert row["infra_final"] is True and len(row["infra_history"]) == 3


def test_run_arm_refuses_a_missing_or_duplicate_row():
    entry: dict = {}

    def short(rep, todo, seeds, attempt, emit):
        emit(_ok(todo[0]))

    with pytest.raises(RuntimeError, match="no row"):
        ev.run_arm(entry, "arm", ["A", "B"], 1, short, log=lambda t: None)

    def twice(rep, todo, seeds, attempt, emit):
        emit(_ok(todo[0]))
        emit(_ok(todo[0]))

    with pytest.raises(RuntimeError, match="duplicate"):
        ev.run_arm({}, "arm", ["A"], 1, twice, log=lambda t: None)


def test_relaunch_with_subset_keeps_scored_rows_and_never_reruns_them():
    """A=1.0, B=infra, C=0.0; relaunch with --cases B, then with A,B,C: the
    scored rows A and C are kept and never re-run."""

    calls: list = []
    run_cases = _runner({"B": ["bad"]}, calls)
    entry: dict = {}
    ev.run_arm(entry, "arm", ["A", "B", "C"], 1, run_cases, log=lambda t: None)
    entry["repeats"]["0"]["rows"][2].update(S=0.0, success=False)  # C scored 0.0
    ev.run_arm(entry, "arm", ["B"], 1, run_cases, layout=["A", "B", "C"], log=lambda t: None)
    rows = entry["repeats"]["0"]["rows"]
    assert [r["case_id"] for r in rows] == ["A", "B", "C"]
    assert rows[2]["S"] == 0.0 and rows[1]["S"] == 1.0
    assert entry["repeats"]["0"]["aggregate"]["cases_scored"] == 3
    before = len(calls)
    ev.run_arm(entry, "arm", ["A", "B", "C"], 1, run_cases, log=lambda t: None)
    assert len(calls) == before  # nothing re-run, C keeps its 0.0
    assert entry["repeats"]["0"]["rows"][2]["S"] == 0.0


# ==========================================================================
# arm identity, run fingerprint, case identity, provenance
# ==========================================================================


def test_check_meta_refuses_mixed_programs_and_round_caps():
    entry = {"meta": {"source_sha256": "aa", "max_rounds": 64}, "repeats": {"0": {"rows": [_ok("A")]}}}
    ev._check_meta("x", entry, {"source_sha256": "aa", "max_rounds": 64, "file": "p.py"})
    assert entry["meta"] == {"source_sha256": "aa", "max_rounds": 64, "file": "p.py"}
    with pytest.raises(SystemExit, match="source_sha256"):
        ev._check_meta("x", entry, {"source_sha256": "bb", "max_rounds": 64})
    with pytest.raises(SystemExit, match="max_rounds"):
        ev._check_meta("x", entry, {"source_sha256": "aa", "max_rounds": 32})
    fresh: dict = {"meta": {}, "repeats": {}}
    ev._check_meta("y", fresh, {"max_rounds": 4})
    assert fresh["meta"] == {"max_rounds": 4}


def test_fingerprint_and_case_identity_checks():
    budgets = SimpleNamespace(max_rounds=64, max_model_calls=1, max_completion_tokens=2, max_messages=3)
    fp = ev.run_fingerprint(agents=5, model="m", reasoning_effort=None, goal="g",
                            worker_contract="c", budgets=budgets)
    assert fp == {"agents": 5, "model": "m", "reasoning_effort": None, "goal": "g",
                  "worker_contract": "c", "temperature": 0.0,
                  "python_budgets": {"max_rounds": 64, "max_model_calls": 1,
                                     "max_completion_tokens": 2, "max_messages": 3}}
    fake = ev.run_fingerprint(agents=5, model="m", reasoning_effort=None, goal="g",
                              worker_contract="c", budgets=budgets, llm_provider="fake")
    assert fake == fp | {"llm_provider": "fake"}
    config: dict = {}
    ev.check_run_fingerprint(config, fp)
    assert config["fingerprint"] == fp
    ev.check_run_fingerprint(config, json.loads(json.dumps(fp)))  # as stored on disk
    with pytest.raises(SystemExit, match="agents"):
        ev.check_run_fingerprint(config, fp | {"agents": 2})
    with pytest.raises(SystemExit, match="reasoning_effort"):
        ev.check_run_fingerprint(config, fp | {"reasoning_effort": "high"})
    with pytest.raises(SystemExit, match="llm_provider"):
        ev.check_run_fingerprint(config, fake)
    with pytest.raises(SystemExit, match="no run fingerprint"):
        ev.check_run_fingerprint({}, fp, has_rows=True)

    result = {"config": {"case_meta": {"X@f1": {"path": "/dry/x.json", "instance_sha256": "aa"}}},
              "arms": {"a": {"repeats": {"0": {"rows": [{"case_id": "X@f1", "instance_sha256": "aa"}]}}}}}
    with pytest.raises(SystemExit, match="X@f1"):
        ev.check_case_identity(result, {"X@f1": {"instance_sha256": "bb", "path": "/real/x.json"}})
    merged = ev.check_case_identity(
        result, {"X@f1": {"instance_sha256": "aa", "path": "/dry/x.json"}, "Y": {"instance_sha256": "cc"}}
    )
    assert set(merged) == {"X@f1", "Y"} and merged["Y"] == {"instance_sha256": "cc"}
    unhashed = {"config": {}, "arms": {"a": {"repeats": {"0": {"rows": [{"case_id": "Z"}]}}}}}
    with pytest.raises(SystemExit, match="no instance sha256"):
        ev.check_case_identity(unhashed, {"Z": {"instance_sha256": "dd"}})


def test_instance_sha_and_code_provenance():
    a = SimpleNamespace(case_id="I-04", n_agents=2, shards=[1, 2])
    b = SimpleNamespace(case_id="I-04", n_agents=2, shards=[1, 3])
    assert ev.instance_sha256(a) == ev.instance_sha256(SimpleNamespace(**vars(a)))
    assert ev.instance_sha256(a) != ev.instance_sha256(b)
    code = ev.code_provenance(("queenbee.evaluate",))
    (sha,) = code["code_files"].values()
    assert sha == ev._sha256_bytes(Path(ev.__file__).read_bytes())
    assert len(code["code_sha256"]) == 64
    # every listed module exists (a renamed module would silently hash to None)
    full = ev.code_provenance(repo=ev.REPO_ROOT)
    assert None not in full["code_files"].values()
    assert len(full["code_files"]) == len(ev.CODE_MODULES)
    assert "src/queenbee/evaluate.py" in full["code_files"]
    assert ev.code_provenance(("queenbee.no_such_module",))["code_files"] == {
        "queenbee.no_such_module": None
    }


def test_git_provenance_reads_a_checkout_and_tolerates_none(tmp_path):
    unknown = {"git_sha": None, "git_dirty": None, "git_diff_sha256": None}
    assert ev.git_provenance(tmp_path) == unknown  # not a checkout
    if shutil.which("git") is None:
        return
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
           "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "f.txt").write_text("a\n")
    subprocess.run([*git, "add", "f.txt"], check=True)
    subprocess.run([*git, "commit", "-q", "--no-verify", "-m", "init"], check=True)
    clean = ev.git_provenance(repo)
    assert len(clean["git_sha"]) in (40, 64) and clean["git_dirty"] is False
    (repo / "f.txt").write_text("b\n")
    dirty = ev.git_provenance(repo)
    assert dirty["git_sha"] == clean["git_sha"] and dirty["git_dirty"] is True
    assert dirty["git_diff_sha256"] != clean["git_diff_sha256"]


# ==========================================================================
# cases: split and manifests
# ==========================================================================


def test_split_templates_are_the_fixed_holdout():
    assert ev.split_templates("test") == TEST12
    assert ev.split_templates("train") == DEV18
    with pytest.raises(ValueError, match="all"):
        ev.split_templates("all")


def test_check_split_guards_labels_templates_and_instance_ids():
    inst = {
        "I-01": SimpleNamespace(case_id="I-01"),
        "I-01@x5": SimpleNamespace(case_id="I-01@x5"),
        "I-01@o10": SimpleNamespace(case_id="I-01"),
        "I-04": SimpleNamespace(case_id="I-04"),
        "I-02@f1": SimpleNamespace(case_id="I-04"),  # a DEV label over a TEST file
        "Z-99": SimpleNamespace(case_id="Z-99"),
    }
    ev.check_split("train", ["I-01", "I-01@x5", "I-01@o10"], inst, {"I-01@o10": {"template": "I-01"}})
    ev.check_split("test", ["I-04"], inst, {})
    with pytest.raises(SystemExit, match="I-04"):
        ev.check_split("train", ["I-01", "I-04"], inst, {})
    with pytest.raises(SystemExit, match="I-01"):
        ev.check_split("test", ["I-04", "I-01"], inst, {})
    with pytest.raises(SystemExit, match="I-02@f1"):
        ev.check_split("train", ["I-02@f1"], inst, {"I-02@f1": {"template": "I-02"}})
    with pytest.raises(SystemExit, match="II-14"):
        ev.check_split("train", ["I-01"], inst, {"I-01": {"template": "II-14"}})
    for side in ("train", "test"):  # an unknown template is on neither side
        with pytest.raises(SystemExit, match="Z-99"):
            ev.check_split(side, ["Z-99"], inst, {})


def test_load_manifest_resolves_relative_paths_and_rejects_duplicates(tmp_path):
    (tmp_path / "inst").mkdir()
    man = tmp_path / "m.json"
    man.write_text(
        json.dumps(
            {
                "salt": "abc",
                "cases": [
                    "I-04",
                    {"case_id": "I-04@f1", "path": "inst/a.json", "template": "I-04", "pool": "fresh"},
                ],
            }
        )
    )
    entries = ev.load_manifest(man)
    assert entries[0] == {"case_id": "I-04"}
    assert entries[1]["path"] == str(tmp_path / "inst" / "a.json")
    assert entries[1]["pool"] == "fresh"
    assert [e["case_id"] for e in ev.filter_manifest(entries, ["I-04@f1"])] == ["I-04@f1"]
    with pytest.raises(ValueError, match="I-99"):
        ev.filter_manifest(entries, ["I-04", "I-99"])
    man.write_text(json.dumps([{"case_id": "A"}, "A"]))
    with pytest.raises(ValueError, match="duplicate"):
        ev.load_manifest(man)
    man.write_text(json.dumps({"cases": []}))
    with pytest.raises(ValueError, match="non-empty"):
        ev.load_manifest(man)


def test_manifest_instances_keep_labels_check_team_size_and_hash(tmp_path):
    path = tmp_path / "fresh_I-04_a.json"
    path.write_text(json.dumps(_instance_doc("I-04", 2, salt=40)))
    entries = [
        {"case_id": "I-04@f1", "path": str(path), "pool": "fresh"},
        {"case_id": "I-05"},
    ]
    bench = {"I-05": SimpleNamespace(case_id="I-05", n_agents=2)}
    inst, meta = ev.resolve_manifest_instances(entries, bench, 2)
    assert inst["I-04@f1"].case_id == "I-04" and inst["I-04@f1"].n_agents == 2
    assert meta["I-04@f1"] == {
        "path": str(path),
        "pool": "fresh",
        "instance_case_id": "I-04",
        "template": "I-04",
    }
    assert meta["I-05"]["template"] == "I-05"
    with pytest.raises(ValueError, match="n=2"):
        ev.resolve_manifest_instances(entries[:1], bench, 5)
    with pytest.raises(ValueError, match="no path"):
        ev.resolve_manifest_instances([{"case_id": "I-09"}], bench, 2)
    pinned = dict(entries[0], instance_sha256=ev.instance_sha256(inst["I-04@f1"]))
    assert set(ev.resolve_manifest_instances([pinned], {}, 2)[0]) == {"I-04@f1"}
    with pytest.raises(ValueError, match="instance_sha256"):
        ev.resolve_manifest_instances([dict(pinned, instance_sha256="0" * 64)], {}, 2)
    with pytest.raises(ValueError, match="not a single Silo-Bench instance"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"cases": []}))
        ev.load_instance_file(bad)


# ==========================================================================
# main() end to end with a fake executor (no LLM)
# ==========================================================================


class Crash(BaseException):
    """Simulated hard crash (not an Exception: nothing catches it)."""


@pytest.fixture
def fake_exec(clean_env):
    """Replace the program executor; rows follow ``state`` per case id."""

    import queenbee.program.execute as pe

    clean_env.setenv("OPENAI_API_KEY", "test-key")
    state: dict = {"calls": [], "kw": [], "env": [], "infra": set(), "fail": set(), "crash": set()}

    def fake(**kw):
        (inst,), (seed,) = kw["instances"], kw["seeds"]
        state["calls"].append((kw["arm_label"].split(":")[0], inst.case_id))
        state["kw"].append(kw)
        state["env"].append((os.environ.get("OPENAI_TIMEOUT"), os.environ.get("OPENAI_CONNECT_TIMEOUT")))
        if inst.case_id in state["crash"]:
            raise Crash(inst.case_id)
        if inst.case_id in state["infra"]:
            row = pe._failure_facts({"error_type": "APIConnectionError", "message": "Error code: 503"})
        elif inst.case_id in state["fail"]:
            row = pe._failure_facts({"error_type": "AnswerFormatError", "message": "no answer"})
        else:
            row = {"infra": None, "success": False, "S": 0.5, "C": 50.0, "rounds_executed": 3}
        return [dict(row, case_id=inst.case_id, seed=seed)]

    clean_env.setattr(pe, "evaluate_python_source_on_cases", fake)
    return state


def _argv(bench: Path, out: Path, *extra: str, model: str = "model-a") -> list[str]:
    return ["--agents", "2", "--benchmarks-dir", str(bench), "--model", model,
            "--out", str(out), *extra]


def test_main_runs_every_template_of_the_split_side(tmp_path, fake_exec):
    bench = write_bench(tmp_path / "bench")
    out = tmp_path / "t.json"
    ev.main(_argv(bench, out, "--split", "test", "--seed-program", "v1", "--repeats", "1"))
    data = json.loads(out.read_text())
    assert data["config"]["cases"] == list(TEST12)
    rows = data["arms"]["v1"]["repeats"]["0"]["rows"]
    assert [r["case_id"] for r in rows] == list(TEST12)
    assert [r["seed"] for r in rows] == list(range(1000, 1012))
    out = tmp_path / "d.json"
    ev.main(_argv(bench, out, "--split", "train", "--seed-program", "v1", "--repeats", "1"))
    assert json.loads(out.read_text())["config"]["cases"] == list(DEV18)


def test_main_refuses_cases_outside_the_split_before_running(tmp_path, fake_exec):
    bench = write_bench(tmp_path / "bench")
    out = tmp_path / "e.json"
    with pytest.raises(SystemExit, match="I-04"):
        ev.main(_argv(bench, out, "--split", "train", "--cases", "I-01,I-04", "--seed-program", "v1"))
    with pytest.raises(SystemExit, match="I-01"):
        ev.main(_argv(bench, out, "--split", "test", "--cases", "I-01", "--seed-program", "v1"))
    man = tmp_path / "m.json"
    man.write_text(json.dumps([{"case_id": "I-02@f1", "path": "bench/I-04_n2.json"}]))
    with pytest.raises(SystemExit, match="I-02@f1"):
        ev.main(_argv(bench, out, "--split", "train", "--manifest", str(man), "--seed-program", "v1"))
    with pytest.raises(ValueError, match="I-99"):
        ev.main(_argv(bench, out, "--split", "train", "--cases", "I-99", "--seed-program", "v1"))
    with pytest.raises(SystemExit) as exc:  # --split has no default
        ev.main(_argv(bench, out, "--seed-program", "v1"))
    assert exc.value.code == 2
    assert fake_exec["calls"] == [] and not out.exists()
    # one --out file holds the cases of one side only
    ev.main(_argv(bench, out, "--split", "train", "--cases", "I-01", "--seed-program", "v1",
                  "--repeats", "1"))
    assert json.loads(out.read_text())["config"]["split"] == "train"
    fake_exec["calls"].clear()
    with pytest.raises(SystemExit, match="split train"):
        ev.main(_argv(bench, out, "--split", "test", "--cases", "I-04", "--seed-program", "v1",
                      "--repeats", "1"))
    assert fake_exec["calls"] == []


def test_main_records_provenance_and_reruns_infra_rows_only(tmp_path, fake_exec, monkeypatch):
    from queenbee.program.execute import seed_python_source

    bench = write_bench(tmp_path / "bench")
    write_bench(tmp_path / "bench", n_agents=5)
    out = tmp_path / "e.json"
    prog = tmp_path / "p.py"
    prog.write_text("# frozen program\n")
    argv = _argv(bench, out, "--split", "train", "--cases", "I-01,I-02,I-03", "--repeats", "1",
                 "--program", f"qb={prog}", "--seed-program", "v1", "--python-max-rounds", "12")
    fake_exec["infra"] = {"I-02"}
    fake_exec["fail"] = {"I-03"}
    assert ev.main(argv) == 0
    data = json.loads(out.read_text())
    cfg = data["config"]
    assert cfg["python_max_rounds"] == 12 and cfg["fingerprint"]["agents"] == 2
    assert cfg["fingerprint"]["python_budgets"]["max_rounds"] == 12
    assert cfg["fingerprint"]["model"] == "model-a" and cfg["model"] == "model-a"
    assert cfg["fingerprint"]["reasoning_effort"] is None and "llm_provider" not in cfg["fingerprint"]
    assert cfg["worker_contract"] == "message_only_v2" and cfg["goal"] == "all_agents"
    assert cfg["benchmarks_dir"] == str(bench.resolve())
    assert "created_at" in cfg and {"git_sha", "git_dirty"} <= set(cfg)
    assert len(cfg["invocations"]) == 1
    inv = cfg["invocations"][0]
    assert len(inv["code_sha256"]) == 64 and "src/queenbee/evaluate.py" in inv["code_files"]
    assert {"git_sha", "git_dirty", "git_diff_sha256"} <= set(inv)
    assert inv["argv"] == argv and inv["cases"] == ["I-01", "I-02", "I-03"]
    assert set(cfg["case_meta"]) == {"I-01", "I-02", "I-03"}
    qb = data["arms"]["qb"]
    assert qb["kind"] == "python"
    assert qb["meta"] == {"file": str(prog), "source_sha256": ev.source_sha256("# frozen program\n"),
                          "max_rounds": 12}
    rows = qb["repeats"]["0"]["rows"]
    assert [r["case_id"] for r in rows] == ["I-01", "I-02", "I-03"]
    assert [r["seed"] for r in rows] == [1000, 1001, 1002]
    assert [r["max_rounds"] for r in rows] == [12, 12, 12]
    assert rows[0]["instance_sha256"] == cfg["case_meta"]["I-01"]["instance_sha256"]
    assert rows[0]["template"] == "I-01"
    assert rows[1]["infra"] and rows[2]["infra"] is None
    assert rows[2]["S"] == 0.0 and rows[2]["execution_class"] == "algorithm_failure"
    seed = seed_python_source("message_only_v2")[2]
    assert data["arms"]["v1"]["meta"] == {"seed": "sfs_phase", "source_sha256": ev.source_sha256(seed),
                                          "max_rounds": 12}
    sources = {kw["arm_label"].split(":")[0]: kw["source"] for kw in fake_exec["kw"]}
    assert sources == {"qb": "# frozen program\n", "v1": seed}
    kw = fake_exec["kw"][0]
    assert kw["llm_provider"] == "openai" and kw["worker_model"] == "model-a"
    assert kw["worker_contract"] == "message_only_v2" and kw["goal"] == "all_agents"
    assert kw["max_parallel_agents"] == 2 and kw["request_timeout"] == 5400.0
    assert kw["budgets"].max_rounds == 12 and kw["max_parallel_cases"] == 1
    # the workers' HTTP timeout follows --request-timeout during the run only
    assert set(fake_exec["env"]) == {("5400", "30")}
    assert "OPENAI_TIMEOUT" not in os.environ and "OPENAI_CONNECT_TIMEOUT" not in os.environ

    # relaunch: only the infra row runs again, in both arms
    fake_exec["infra"] = set()
    fake_exec["calls"].clear()
    ev.main(argv)
    assert sorted(fake_exec["calls"]) == [("qb", "I-02"), ("v1", "I-02")]
    data = json.loads(out.read_text())
    rep = data["arms"]["qb"]["repeats"]["0"]
    assert rep["aggregate"]["infra"] == 0
    assert rep["rows"][1]["infra_history"] == ["APIConnectionError: Error code: 503"]
    assert rep["reruns"][0]["infra_retry"] == ["I-02"]
    assert len(data["config"]["invocations"]) == 2

    # another program under the arm name, or another fingerprint: refused before any run
    fake_exec["calls"].clear()
    prog.write_text("# a different program\n")
    with pytest.raises(SystemExit, match="source_sha256"):
        ev.main(argv)
    prog.write_text("# frozen program\n")
    with pytest.raises(SystemExit, match="fingerprint"):
        ev.main([*argv, "--model", "model-b"])
    with pytest.raises(SystemExit, match="fingerprint"):
        ev.main([*argv, "--agents", "5"])
    with pytest.raises(SystemExit, match="fingerprint"):
        ev.main([*argv, "--llm", "fake"])
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    with pytest.raises(SystemExit, match="reasoning_effort"):
        ev.main(argv)
    assert fake_exec["calls"] == []
    assert len(json.loads(out.read_text())["config"]["invocations"]) == 2


def test_main_checkpoints_each_case_so_a_crash_loses_only_in_flight_cases(tmp_path, fake_exec):
    bench = write_bench(tmp_path / "bench")
    out = tmp_path / "e.json"
    argv = _argv(bench, out, "--split", "train", "--cases", "I-01,I-02,I-03", "--repeats", "1",
                 "--seed-program", "v1", "--parallel-cases", "1")
    fake_exec["crash"] = {"I-02"}
    with pytest.raises(Crash):
        ev.main(argv)
    rows = json.loads(out.read_text())["arms"]["v1"]["repeats"]["0"]["rows"]
    assert {r["case_id"] for r in rows} == {"I-01", "I-03"}
    fake_exec["crash"] = set()
    fake_exec["calls"].clear()
    ev.main(argv)
    assert fake_exec["calls"] == [("v1", "I-02")]
    rep = json.loads(out.read_text())["arms"]["v1"]["repeats"]["0"]
    assert [r["case_id"] for r in rep["rows"]] == ["I-01", "I-02", "I-03"]
    assert rep["rows"][1]["seed"] == 1001 and rep["reruns"][0]["missing"] == ["I-02"]


def test_main_manifest_rows_carry_labels_pool_and_template(tmp_path, fake_exec):
    bench = write_bench(tmp_path / "bench")
    shutil.copy(bench / "I-01_n2.json", tmp_path / "f.json")
    man = tmp_path / "m.json"
    man.write_text(json.dumps([
        "I-02",
        {"case_id": "I-01@f1", "path": "f.json", "pool": "fresh"},
    ]))
    out = tmp_path / "e.json"
    ev.main(_argv(bench, out, "--split", "train", "--manifest", str(man), "--repeats", "1",
                  "--seed-program", "v1"))
    data = json.loads(out.read_text())
    assert data["config"]["cases"] == ["I-02", "I-01@f1"]
    assert data["config"]["manifest"]["n_cases"] == 2 and len(data["config"]["manifests"]) == 1
    rows = data["arms"]["v1"]["repeats"]["0"]["rows"]
    assert rows[1]["case_id"] == "I-01@f1" and rows[1]["pool"] == "fresh"
    assert rows[1]["template"] == "I-01" and rows[1]["instance_case_id"] == "I-01"
    assert rows[1]["seed"] == 1001
    with pytest.raises(ValueError, match="I-99"):
        ev.main(_argv(bench, out, "--split", "train", "--manifest", str(man), "--cases", "I-02,I-99",
                      "--repeats", "1", "--seed-program", "v1"))
    # entries with paths only: the benchmarks dir is not needed
    only = tmp_path / "only.json"
    only.write_text(json.dumps([{"case_id": "I-01@f1", "path": "f.json"}]))
    out = tmp_path / "only_out.json"
    ev.main(_argv(tmp_path / "absent", out, "--split", "train", "--manifest", str(only),
                  "--repeats", "1", "--seed-program", "v1"))
    assert json.loads(out.read_text())["config"]["cases"] == ["I-01@f1"]


def test_main_refuses_a_label_that_now_points_to_another_instance(tmp_path, fake_exec):
    bench = write_bench(tmp_path / "bench")
    (tmp_path / "dry").mkdir()
    (tmp_path / "real").mkdir()
    shutil.copy(bench / "I-01_n2.json", tmp_path / "dry" / "I-01@f1_n2.json")
    shutil.copy(bench / "I-02_n2.json", tmp_path / "real" / "I-01@f1_n2.json")
    entry = {"case_id": "I-01@f1", "path": "I-01@f1_n2.json", "template": "I-01", "pool": "FRESH"}
    for d in ("dry", "real"):
        (tmp_path / d / "m.json").write_text(json.dumps({"cases": [entry]}))
    out = tmp_path / "e.json"
    base = _argv(bench, out, "--split", "train", "--repeats", "1", "--seed-program", "v1")
    ev.main([*base, "--manifest", str(tmp_path / "dry" / "m.json")])
    fake_exec["calls"].clear()
    with pytest.raises(SystemExit, match="I-01@f1"):
        ev.main([*base, "--manifest", str(tmp_path / "real" / "m.json")])
    assert fake_exec["calls"] == []
    cfg = json.loads(out.read_text())["config"]
    assert cfg["case_meta"]["I-01@f1"]["path"].startswith(str(tmp_path / "dry"))
    # a superset (union) with one more label is accepted and runs only the
    # added case
    (tmp_path / "dry" / "m2.json").write_text(json.dumps({"cases": [entry, "I-02"]}))
    ev.main([*base, "--manifest", str(tmp_path / "dry" / "m2.json")])
    assert fake_exec["calls"] == [("v1", "I-02")]
    data = json.loads(out.read_text())
    assert data["config"]["cases"] == ["I-01@f1", "I-02"] and len(data["config"]["manifests"]) == 2


def test_main_takes_the_worker_model_from_the_environment(tmp_path, fake_exec, monkeypatch):
    bench = write_bench(tmp_path / "bench")
    out = tmp_path / "e.json"
    argv = ["--agents", "2", "--benchmarks-dir", str(bench), "--split", "train", "--cases", "I-01",
            "--seed-program", "v1", "--repeats", "1", "--max-parallel-agents", "1", "--out", str(out)]
    monkeypatch.setenv("QUEENBEE_WORKER_MODEL", "model-env")
    monkeypatch.setenv("OPENAI_TIMEOUT", "77")
    ev.main(argv)
    assert fake_exec["env"] == [("77", "30")]  # a set timeout is kept
    assert os.environ["OPENAI_TIMEOUT"] == "77" and "OPENAI_CONNECT_TIMEOUT" not in os.environ
    cfg = json.loads(out.read_text())["config"]
    assert cfg["model"] == "model-env" and cfg["fingerprint"]["model"] == "model-env"
    assert fake_exec["kw"][0]["worker_model"] == "model-env"
    assert fake_exec["kw"][0]["max_parallel_agents"] == 1


def test_main_real_run_needs_a_model_a_key_and_an_arm(tmp_path, clean_env, capsys):
    bench = write_bench(tmp_path / "bench")
    out = tmp_path / "e.json"
    base = ["--agents", "2", "--benchmarks-dir", str(bench), "--split", "test", "--out", str(out)]
    with pytest.raises(SystemExit) as exc:
        ev.main([*base, "--seed-program", "v1"])
    assert exc.value.code == 2 and "QUEENBEE_WORKER_MODEL" in capsys.readouterr().err
    clean_env.setenv("QUEENBEE_WORKER_MODEL", "model-a")
    with pytest.raises(SystemExit) as exc:
        ev.main([*base, "--seed-program", "v1"])
    assert exc.value.code == 2 and "OPENAI_API_KEY" in capsys.readouterr().err
    clean_env.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        ev.main(base)
    assert exc.value.code == 2 and "--seed-program" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        ev.main([*base, "--seed-program", "a", "--seed-program", "a"])
    assert exc.value.code == 2 and "duplicate arm" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        ev.main([*base, "--program", "no-equals-sign"])
    assert exc.value.code == 2 and "NAME=PATH" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        ev.main([*base, "--program", f"x={tmp_path / 'missing.py'}"])
    assert exc.value.code == 2 and "no file" in capsys.readouterr().err
    assert not out.exists()


def test_main_fake_llm_runs_the_seed_offline_end_to_end(tmp_path, clean_env):
    """--llm fake: real sandboxed execution with the offline worker."""

    from queenbee.evo.units import load_eval_rows

    bench = write_bench(tmp_path / "bench")
    out = tmp_path / "census.json"
    traces = tmp_path / "traces"
    clean_env.setenv("QB_TRACE_DIR", str(traces))
    argv = ["--llm", "fake", "--agents", "2", "--benchmarks-dir", str(bench), "--split", "train",
            "--cases", "I-01,II-11", "--seed-program", "v1", "--repeats", "2", "--out", str(out)]
    assert ev.main(argv) == 0
    data = json.loads(out.read_text())
    fp = data["config"]["fingerprint"]
    assert fp["llm_provider"] == "fake" and fp["model"] == "fake"
    for rep in ("0", "1"):
        rows = data["arms"]["v1"]["repeats"][rep]["rows"]
        assert [r["case_id"] for r in rows] == ["I-01", "II-11"]
        for row in rows:
            assert row["infra"] is None and row["execution_class"] == "completed"
            assert row["success"] is False and row["S"] == 0.0  # the offline worker solves nothing
            assert row["rounds_executed"] >= 1 and row["model_calls"] > 0
    assert len(list(traces.glob("*.json"))) == 4
    # read back the way the evolution loop reads census files
    rows = load_eval_rows([out])
    assert {unit: len(r) for unit, r in rows.arms["v1"].items()} == {"I-01@o2": 2, "II-11@o2": 2}
    # a relaunch with the same command is fully cached
    assert ev.main(argv) == 0
    assert len(list(traces.glob("*.json"))) == 4


def test_help_text_is_self_contained(capsys):
    with pytest.raises(SystemExit) as exc:
        ev.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "Evaluate frozen team programs on Silo-Bench cases." in text
    assert "--split {train,test}" in text and "--llm {openai,fake}" in text
    assert "QUEENBEE_WORKER_MODEL" in text and "--seed-program NAME" in text
