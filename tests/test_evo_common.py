"""Shared constants, TEST guards, loaders, the execution log, the default
executor and the behaviour fingerprint (``queenbee.evo.common``).

Offline: synthetic Silo-Bench-shaped instances written into ``tmp_path``;
the one sandbox run uses the ``fake`` worker provider.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import queenbee.program.execute as ex
from queenbee.bench.instance import BenchmarkInstance
from queenbee.evo import common as C
from queenbee.program.budgets import PythonRunBudgets

SRC = Path(__file__).resolve().parents[1] / "src"
SEALED = (
    "I-04", "I-05", "I-09", "I-10", "II-14", "II-15", "II-19", "II-20",
    "III-24", "III-25", "III-29", "III-30",
)
V1_SHA256 = "a302510d734091a01544e467e41389b28aed8ed45325ef80dcbc0176dba0d2e9"
#: Pinned behaviour fingerprints (exec-cache / duplicate keys).
V1_FP = "s:1c8d90dcf360c21da949ec3c"
V1_FP_N2_5_8_10 = "s:5679b252385112a695db0a0a"
V1_FP_SINK = "s:1fb092af3f927f4cc4173063"
IDLE_ROUND_FP = "s:24963f9841b9165c032bc885"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _instance_doc(case_id: str, n_agents: int) -> dict[str, Any]:
    shards = [[3 * i + 1, 3 * i + 2] for i in range(n_agents)]
    best = max(v for shard in shards for v in shard)
    return {
        "case_id": case_id,
        "case_name": "Global Max",
        "paradigm": "Paradigm I",
        "metadata": {"num_agents": n_agents, "output_type": "distributed",
                     "is_segmented": False},
        "task_description": ("Find the GLOBAL MAXIMUM across all agents' data. "
                             "You are Agent {agent_id} and hold: {input_shard}"),
        "agent_configs": [{"agent_id": i, "input_shard": shard, "expected_output": best}
                          for i, shard in enumerate(shards)],
    }


def _write(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _idle_round_variant(src: str) -> str:
    var = src.replace(
        '    {"kind": "broadcast_last", "rounds": 1},\n',
        '    {"kind": "broadcast_last", "rounds": 1},\n    {"kind": "wait", "rounds": 1},\n', 1)
    head = ("def phase_turn(kind, local_round, agent_id, n_agents, selected_primary,\n"
            "               known_source_count, inbox_count):\n")
    return var.replace(
        head, head + '    if kind == "wait":\n        return {"mode": "idle", "recipients": []}\n', 1)


# --------------------------------------------------------------------------- #
# template ids and the TEST guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("test_id", C.TEST_IDS)
def test_every_test_id_is_refused(test_id: str) -> None:
    with pytest.raises(C.LeakageGuardError):
        C.assert_not_test([test_id], where="t")
    with pytest.raises(C.LeakageGuardError):
        C.assert_not_test(f"{test_id}@x5", where="t")
    with pytest.raises(C.LeakageGuardError):
        C.assert_not_test([f"traces/{test_id}_abcdefabcdef_1.json"], where="t")
    with pytest.raises(C.LeakageGuardError):
        C.make_unit(test_id, "x5")


def test_test_ids_live_only_in_the_guard_constant() -> None:
    """Outside its guard tuple, the source of ``queenbee.evo.common`` names
    no TEST template, not even in a comment; TEST and DEV partition the 30
    templates."""

    source = Path(C.__file__).read_text()
    guard = source[source.index("TEST_IDS: tuple[str, ...] = ("):]
    guard = guard[: guard.index(")") + 1]
    rest = source.replace(guard, "")
    for test_id in C.TEST_IDS:
        assert not re.search(r"(?<![\w-])" + re.escape(test_id) + r"(?!\d)", rest), test_id
    assert C.TEST_IDS == SEALED
    assert set(C.TEST_IDS).isdisjoint(C.DEV_IDS)
    assert len(set(C.TEST_IDS)) == 12 and len(set(C.DEV_IDS)) == 18
    assert set(C.TEST_IDS) | set(C.DEV_IDS) == set(C.ALL_TEMPLATE_IDS)
    assert len(C.ALL_TEMPLATE_IDS) == 30
    assert C.DEV_IDS == tuple(t for t in C.ALL_TEMPLATE_IDS if t not in SEALED)


def test_worker_model_and_benchmarks_dir_come_from_the_environment(tmp_path: Path) -> None:
    """No built-in model id: the worker model is read from the environment
    once, at import; ``SILO_BENCH_DIR`` overrides the benchmarks directory
    (default: the vendored Silo-Bench copy under ``third_party/``)."""

    code = ("import queenbee.evo.common as c; from queenbee.paths import default_benchmarks_dir; "
            "print(repr((c.WORKER_MODEL, str(default_benchmarks_dir()))))")
    env = {k: v for k, v in os.environ.items()
           if k not in (C.PLANNER_MODEL_ENV, C.WORKER_MODEL_ENV, "SILO_BENCH_DIR")}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC), env.get("PYTHONPATH")) if p)

    def run(**extra: str) -> tuple[str, str]:
        out = subprocess.run([sys.executable, "-c", code], env={**env, **extra},
                             capture_output=True, text=True, check=True)
        return ast.literal_eval(out.stdout.strip())

    worker, bench = run()
    assert worker == ""
    assert Path(bench) == SRC.parent / "third_party" / "acl26-silo-bench" / "benchmarks"
    assert run(**{C.WORKER_MODEL_ENV: " worker-y ", "SILO_BENCH_DIR": str(tmp_path)}) == (
        "worker-y", str(tmp_path))


def test_dev_units_pass_the_guard() -> None:
    for template in C.DEV_IDS:
        C.assert_not_test([template, f"{template}@x5", f"{template}@o10"], where="t")
        for rung in C.RUNGS:
            assert C.make_unit(template, rung) == f"{template}@{rung}"
    assert C.template_of("II-11@x5") == "II-11" and C.rung_of("II-11@x5") == "x5"
    assert C.template_of("II-11") == "II-11" and C.rung_of("II-11") == "o5"
    assert C.templates_in("units II-11@x5, III-21 and I-01") == ["II-11", "III-21", "I-01"]
    assert C.templates_in("XII-110 I-011") == []
    C.assert_not_test(None, where="t")
    with pytest.raises(ValueError):
        C.make_unit("II-11", "o7")
    with pytest.raises(C.LeakageGuardError):
        C.make_unit("ORD-50", "x5")


# --------------------------------------------------------------------------- #
# ladder manifests
# --------------------------------------------------------------------------- #


def test_ladder_manifest_refuses_test_entries(tmp_path: Path) -> None:
    bad = _write(tmp_path / "manifest.json", [{"case_id": "II-14@x5", "path": "x5/II-14_x5.json",
                                               "template": "II-14", "rung": "x5"}])
    with pytest.raises(Exception, match="LEAKAGE_GUARD|TEST"):
        C.load_ladder_manifest(bad)
    # a dev instance stored under a TEST-named path is refused too
    _write(tmp_path / "II-14" / "II-11_x5.json", _instance_doc("II-11@x5", 5))
    hidden = _write(tmp_path / "manifest_hidden.json", [
        {"case_id": "II-11@x5", "path": "II-14/II-11_x5.json", "template": "II-11", "rung": "x5"}])
    with pytest.raises(C.LeakageGuardError):
        C.load_ladder_manifest(hidden)


def test_ladder_manifest_keeps_dev_x5_entries(tmp_path: Path) -> None:
    x5 = _write(tmp_path / "ladder" / "x5" / "II-11_x5.json", _instance_doc("II-11@x5", 5))
    _write(tmp_path / "ladder" / "II-11_n5.json", _instance_doc("II-11", 5))
    manifest = _write(tmp_path / "ladder" / "manifest_x5.json", [
        {"case_id": "II-11@x5", "path": "x5/II-11_x5.json", "template": "II-11", "rung": "x5"},
        {"case_id": "II-11@o5", "path": "II-11_n5.json", "template": "II-11", "rung": "o5"},
    ])
    paths = C.load_ladder_manifest(manifest)
    assert paths == {"II-11@x5": x5}

    pathless = _write(tmp_path / "ladder" / "manifest_pathless.json", [
        {"case_id": "II-12@x5", "template": "II-12", "rung": "x5"}])
    with pytest.raises(C.ManifestError):
        C.load_ladder_manifest(pathless)


# --------------------------------------------------------------------------- #
# trace index
# --------------------------------------------------------------------------- #


def test_trace_index_skips_test_traces_unread(tmp_path: Path, monkeypatch) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "I-04_abcdefabcdef_1.json").write_text("not json: never opened")
    (traces / "ORD-50_abcdefabcdef_1.json").write_text("{}")
    (traces / "I-01_short.json").write_text("{}")
    _write(traces / "I-01_abcdefabcdef_1.json", {
        "case_id": "I-01", "facts": {"S": 1.0, "C": 1234.5, "prompt_tokens": 900}})
    _write(traces / "II-11@x5_abcdefabcdef_2.json", {
        "case_id": "II-11@x5", "facts": {"S": 0.4, "C": 10.0, "prompt_tokens": 7}})
    _write(traces / "I-02_abcdefabcdef_3.json", {"case_id": "I-05", "facts": {}})
    read: list[str] = []
    real_read = C._read_json

    def spy(path: Path, default: Any = None) -> Any:
        read.append(Path(path).name)
        return real_read(path, default)

    monkeypatch.setattr(C, "_read_json", spy)
    index = C.TraceIndex([traces, tmp_path / "missing"])
    assert index.skipped_test == 1 and index.skipped_other == 1
    assert sorted(index.by_key) == [("I-01", "abcdefabcdef"), ("I-02", "abcdefabcdef"),
                                    ("II-11@x5", "abcdefabcdef")]
    row = {"S": 1.0, "C": 1234.5, "prompt_tokens": 900}
    assert index.match("I-01", "abcdefabcdef", row)["case_id"] == "I-01"
    assert index.match("I-01", "abcdefabcdef", {**row, "S": 0.0}) is None
    assert index.match("II-11@x5", "abcdefabcdef", {"S": 0.4, "C": 10, "prompt_tokens": 7})
    assert index.match("I-01", "000000000000", row) is None
    with pytest.raises(C.LeakageGuardError):  # a TEST case id inside a dev-named file
        index.load(traces / "I-02_abcdefabcdef_3.json")
    assert "I-04_abcdefabcdef_1.json" not in read


def test_same_num_and_json_helpers(tmp_path: Path) -> None:
    assert C._same_num(1, 1.0000001) and not C._same_num(1, 1.1)
    assert C._same_num(None, None) and not C._same_num(None, 0) and not C._same_num("x", "x")
    path = tmp_path / "a" / "b.json"
    C._write_json(path, {"k": [1, 2]})
    assert C._read_json(path) == {"k": [1, 2]} and not (tmp_path / "a" / "b.json.tmp").exists()
    assert C._read_json(tmp_path / "missing.json", {}) == {}
    path.write_text("{torn")
    assert C._read_json(path) is None


# --------------------------------------------------------------------------- #
# execution log
# --------------------------------------------------------------------------- #


def test_exec_log_appends_and_ignores_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "root" / "executions.jsonl"
    log = C.ExecLog(path)
    log.append({"key": "v1|I-01@o5|r0", "row": {"infra": "ConnectionError"}})
    log.append({"key": "v1|I-01@o5|r0", "row": {"infra": None, "S": 1.0}})
    log.append({"key": "v1|I-02@o5|r0", "row": {"infra": "timeout"}})
    assert len(log.for_key("v1|I-01@o5|r0")) == 2
    assert log.scored("v1|I-01@o5|r0")["row"]["S"] == 1.0
    assert log.scored("v1|I-02@o5|r0") is None and log.scored("nope") is None
    with path.open("a") as handle:
        handle.write('{"key": "v1|I-06@o5|r0", "row": ')  # killed mid-write
    again = C.ExecLog(path)
    assert [r["key"] for r in again.records] == ["v1|I-01@o5|r0"] * 2 + ["v1|I-02@o5|r0"]


# --------------------------------------------------------------------------- #
# default executor
# --------------------------------------------------------------------------- #


def _req(tmp_path: Path, instance: BenchmarkInstance, *, provider: str = "fake") -> dict[str, Any]:
    cfg = SimpleNamespace(llm_provider=provider, worker_model="worker-x",
                          request_timeout=30.0, python_max_rounds=C.PYTHON_MAX_ROUNDS)
    return {"cfg": cfg, "job": SimpleNamespace(program_id="v1", unit_id=f"{instance.case_id}@o5"),
            "source": C.v1_source(), "instance": instance, "seed": 4242,
            "artifacts_dir": tmp_path / "artifacts"}


def _bench_instance(n_agents: int = 2) -> BenchmarkInstance:
    doc = _instance_doc("I-01", n_agents)
    return BenchmarkInstance(
        benchmark="silo_bench", case_id="I-01", case_name=doc["case_name"],
        n_agents=n_agents, shards=[c["input_shard"] for c in doc["agent_configs"]],
        ground_truth=doc["agent_configs"][0]["expected_output"],
        task_prompt=doc["task_description"] + "\n**Output:** A single integer.",
        meta={"output_type": "distributed", "is_segmented": False,
              "expected_outputs": [c["expected_output"] for c in doc["agent_configs"]]},
    )


def test_real_execute_passes_the_job_through(tmp_path: Path, monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_eval(**kw: Any) -> list[dict[str, Any]]:
        seen.update(kw)
        return [{"S": 0.5, "case_id": kw["instances"][0].case_id}]

    monkeypatch.setattr(ex, "evaluate_python_source_on_cases", fake_eval)
    instance = _bench_instance(5)
    req = _req(tmp_path, instance, provider="openai")
    assert C.real_execute(req) == {"S": 0.5, "case_id": "I-01"}
    assert seen["instances"] == (instance,) and seen["seeds"] == (4242,)
    assert seen["arm_label"] == "v1:I-01@o5" and seen["source"] == req["source"]
    assert (seen["llm_provider"], seen["worker_model"]) == ("openai", "worker-x")
    assert (seen["goal"], seen["worker_contract"]) == ("all_agents", "message_only_v2")
    assert seen["max_parallel_cases"] == 1 and seen["max_parallel_agents"] == 5
    assert seen["request_timeout"] == 30.0 and seen["artifacts_dir"] == tmp_path / "artifacts"
    want = PythonRunBudgets.for_rounds(C.PYTHON_MAX_ROUNDS, n_agents=5)
    assert vars(seen["budgets"]) == vars(want)
    monkeypatch.setattr(C, "GOAL", "sink")
    C.real_execute(req)
    assert seen["goal"] == "sink"


def test_real_execute_runs_the_v1_program_offline(tmp_path: Path) -> None:
    instance = _bench_instance(2)
    row = C.real_execute(_req(tmp_path, instance))
    assert row["infra"] is None and row["execution_class"] == "completed"
    assert row["case_id"] == "I-01" and row["seed"] == 4242
    assert row["model_calls"] > 0 and "S" in row and isinstance(row.get("diag"), dict)


# --------------------------------------------------------------------------- #
# behaviour fingerprint
# --------------------------------------------------------------------------- #


def test_v1_source_is_the_seed_program() -> None:
    from queenbee.evo.seed import v1_seed_source

    src = C.v1_source()
    assert src == v1_seed_source()
    assert hashlib.sha256(src.encode("utf-8")).hexdigest() == V1_SHA256


def test_fingerprint_sees_an_inserted_idle_round() -> None:
    """``simulate_program_behavior`` drops all-idle rounds, so a program
    that idles after the final broadcast (the bodies delivered in that round
    are lost) looks like the seed program (v1) to it.  The exec-cache /
    duplicate key (``behavior_fingerprint``) must tell them apart."""

    from queenbee.evo.seed import evo_seed_source

    src = evo_seed_source()
    var = _idle_round_variant(src)
    assert var != src
    same = ex.simulate_program_behavior(var, n_agents=5, max_rounds=64) == \
        ex.simulate_program_behavior(src, n_agents=5, max_rounds=64)
    assert same  # the weakness being guarded against
    assert C.behavior_fingerprint(var) != C.behavior_fingerprint(src)
    assert C.behavior_fingerprint(src) == C.behavior_fingerprint(C.v1_source())
    assert C.behavior_fingerprint(var) == IDLE_ROUND_FP


def test_fingerprint_values_and_scheme_are_pinned(monkeypatch) -> None:
    v1 = C.v1_source()
    assert C.FP_SCHEME == "screen.behavior_fp@n2,5,10"
    assert re.fullmatch(r"s:[0-9a-f]{24}", C.behavior_fingerprint(v1) or "")
    assert C.behavior_fingerprint(v1) == V1_FP
    assert C.behavior_fingerprint(v1, ns=(2, 5, 8, 10)) == V1_FP_N2_5_8_10
    assert C.behavior_fingerprint(v1.replace("def main(", "def main(:", 1)) is None
    assert C.behavior_fingerprint("") is None
    monkeypatch.setattr(C, "GOAL", "sink")  # read at call time
    assert C.behavior_fingerprint(v1) == V1_FP_SINK
