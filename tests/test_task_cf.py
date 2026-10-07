"""The Count-Frequency task (``queenbee.tasks.cf``), offline.

Cases generated from their numeric seed, scoring and answer parsing, the
opt-in count-table JSON repair (host side and in the worker sandbox),
diagnosis cards and their rendering, row finalization and the evaluator
summary, the seed program, the S0 screen at n = 8, behaviour fingerprints,
planner texts, ids / split / thresholds / command-line defaults, an execution
through the real sandbox with the fake worker, end-to-end runs of the
loop (``--task cf --fake``) and the evaluator (``--task cf --llm fake``),
the guards of the shared paths under this task, and its isolation from
the default task.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import queenbee.tasks as tasks
from exp_graph.mas import count_table_json as cj
from queenbee.evo import api_card as ac
from queenbee.evo import common as C
from queenbee.evo import credit as cr
from queenbee.evo import diagnosis as dg
from queenbee.evo import ladder as LD
from queenbee.evo import loop as L
from queenbee.evo import prompt as ep
from queenbee.evo import race as R
from queenbee.evo import screen as S
from queenbee.evo import units as U
from queenbee.evo.seed import evo_seed_source
from queenbee.paths import default_benchmarks_dir
from queenbee.program.budgets import PythonRunBudgets
from queenbee.tasks import get_task, set_task, use_task
from queenbee.tasks.cf import CF_TASK, cases, scoring, texts
from queenbee.tasks.cf import diagnosis as cfd
from queenbee.tasks.cf import screen as cfs
from queenbee.tasks.cf import seed as cfseed

HAVE_BENCH = (default_benchmarks_dir() / "I-01_n5.json").is_file()
needs_bench = pytest.mark.skipif(
    not HAVE_BENCH, reason="the work-instruction lint reads the Silo-Bench development texts")

SEED = cfseed.seed_source()
TRAIN = ("CF-001", "CF-002", "CF-003", "CF-004")
VALID = ("CF-201", "CF-202", "CF-203")
TEST = tuple(f"CF-{s}" for s in range(301, 311))
CF_CLASSES = ("format", "ok", "content-loss", "scattered-wrong", "consensus-wrong")

#: sha256 of the seed program and of three case documents (sorted-key JSON).
SEED_SHA256 = "25c43c5832b49da70b03b10b1e4953b674e89caa1a00a822211ff078d69f5c40"
CASE_SHA256 = {
    "CF-001": "629ecc65c16f9303076f70525de6c46b87c18466d0328f38cb1afc10ec16dc7e",
    "CF-201": "77b9dc4314c9b05788539194655220fd1a61621a2335e331e2f8bec6084f3a06",
    "CF-301": "2c0ba921f7fd61bffe7957f0f4d51511aa94c759fcda6922cd2a28adda71d56e",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch):
    """Case files under the test's own directory; the Silo task active
    before and after every test."""

    monkeypatch.setenv(cases.CASES_DIR_ENV, str(tmp_path / "cf_cases"))
    assert get_task() is tasks.SILO_TASK
    yield
    leaked = get_task()
    if leaked is not tasks.SILO_TASK:
        set_task("silo")
        pytest.fail(f"test left task {leaked.name!r} active")


@pytest.fixture
def card_flags(monkeypatch):
    """The loop sets the card options process-wide: restore them afterwards."""

    from queenbee.evo.credit import MULTISET_ENV

    for name in (dg.LOCAL_ONLY_ENV, dg.ATTEMPT_ENV, MULTISET_ENV):
        monkeypatch.setenv(name, "0")
    return monkeypatch


def _with_phases(entries: list[dict[str, Any]]) -> str:
    return cfseed.build_program(entries)


# --------------------------------------------------------------------------- #
# registry, ids, rungs
# --------------------------------------------------------------------------- #


def test_task_is_registered_and_wires_its_ids_rung_and_goal() -> None:
    assert "cf" in tasks.available_tasks() and tasks.resolve_task("cf") is CF_TASK
    with use_task("cf"):
        assert (C.task_goal(), C.task_rungs(), C.task_rung_order(), L.rung_next()) == \
            ("sink", {"o8": 8}, {"o8": 0}, {})
        assert C.task_test_ids() == TEST and C.task_dev_ids() == TRAIN + VALID
        assert C.task_mint_n_agents() == 5
        assert C.fingerprint_ns() == (2, 5, 8, 10)
        assert C.fingerprint_scheme() == "screen.behavior_fp@n2,5,8,10"
        assert R.exec_eq("CF-001@o8") == 1 and R.unit_n_agents("CF-001@o8") == 8
        assert C.make_unit("CF-002", "o8") == "CF-002@o8"
        assert C.templates_in("traces/CF-305@o8_0123abcd4567_1.json") == ["CF-305"]
        assert cr.templates_in("[CF-202@o8] and CF-1234") == ["CF-202", "CF-1234"]
        with pytest.raises(C.LeakageGuardError, match="CF-301"):
            C.assert_not_test(["x/CF-301@o8"], where="test")
        with pytest.raises(RuntimeError, match="CF-310"):
            cr.assert_not_test(["CF-310"])
        assert LD.assert_dev_template("CF-201@o8") == "CF-201"
        with pytest.raises(LD.LadderLeakError, match="TEST"):
            LD.assert_dev_template("CF-305")
        with pytest.raises(LD.LadderLeakError, match="unknown"):
            LD.assert_dev_template("I-01")
        assert R.RaceConfig(fake=True, worker_model="w").worker_fingerprint()["goal"] == "sink"
        assert C.task_score_hooks() == {"score_fn": scoring.score_fn, "diag_fn": cfd.diag_fn}
        from queenbee.program.execute import task_worker_env

        assert task_worker_env() == {cj.REPAIR_ENV: "1"}
    assert cases.template_id(7) == "CF-007" and cases.seed_of("CF-203@o8") == 203
    assert cases.seed_of("CF-0203") is None and cases.seed_of("I-01@o5") is None


# --------------------------------------------------------------------------- #
# cases
# --------------------------------------------------------------------------- #


def test_cases_are_generated_deterministically_from_their_seed() -> None:
    assert cases.build_global_array(1)[:8] == [68, 32, 130, 60, 253, 230, 241, 194]
    assert cases.build_global_array(301)[:8] == [204, 252, 235, 120, 130, 60, 124, 67]
    for case_id, digest in CASE_SHA256.items():
        assert _sha(json.dumps(cases.case_document(case_id), sort_keys=True)) == digest
    arrays = {}
    for case_id in TRAIN + VALID + TEST:
        seed = cases.seed_of(case_id)
        array = cases.build_global_array(seed)
        gold, shards = cases.generate(seed)
        assert (gold, shards) == cases.generate(seed)
        assert [len(s) for s in shards] == [128] * 8 and sum(shards, []) == array
        assert all(0 <= v <= 255 for v in array)
        assert gold == {str(k): v for k, v in sorted(Counter(array).items())}
        assert list(gold) == sorted(gold, key=int) and sum(gold.values()) == 1024
        arrays[case_id] = tuple(array)
    assert len(set(arrays.values())) == len(arrays)
    instance = cases.build_instance("CF-002")
    assert (instance.benchmark, instance.case_id, instance.n_agents) == ("cf", "CF-002", 8)
    assert instance.task_prompt == cases.STATEMENT and instance.case_name == "Count Frequency"
    assert instance.ground_truth == cases.generate(2)[0] and not instance.segmented
    assert instance.meta["expected_outputs"] == [instance.ground_truth] * 8
    doc = cases.case_document("CF-002")
    assert doc["agent_configs"][3]["user_prompt"] == cases.STATEMENT.replace(
        "{agent_id}", "3").replace("{input_shard}", json.dumps(instance.shards[3]))
    assert "Communication Protocol" not in cases.STATEMENT
    with pytest.raises(ValueError, match="template id"):
        cases.case_document("I-01")


def test_case_files_are_written_on_request_and_verified_on_load(tmp_path: Path) -> None:
    import queenbee.evaluate as ev

    path = cases.instance_path("CF-002@o8")
    assert path == tmp_path / "cf_cases" / "CF-002.json" and path.is_file()
    assert cases.instance_path("CF-002") == path and cases.instance_path("I-01@o5") is None
    assert cases.load_instance(path) == cases.build_instance("CF-002")
    doc = json.loads(path.read_text())
    doc["agent_configs"][0]["expected_output"]["5"] = 99
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="differs from the generated case CF-002"):
        cases.load_instance(path)
    assert cases.load_instance(cases.instance_path("CF-002")) == cases.build_instance("CF-002")
    mine = cases.write_case("my-case", tmp_path / "own", seed=7)
    assert cases.load_instance(mine) == cases.build_instance("my-case", seed=7)
    assert cases.load_instance(mine).shards == cases.generate(7)[1]
    with use_task("cf"):
        assert ev.load_instance_file(path) == cases.build_instance("CF-002")
        pool = L.UnitPool({}, SimpleNamespace(T=TRAIN, V=VALID), benchmarks_dir=tmp_path / "none")
        instance, sha = pool.get("CF-003@o8")
        assert instance.case_id == "CF-003" and sha == ev.instance_sha256(cases.build_instance("CF-003"))
        assert pool.n_agents("CF-003@o8") == 8 and pool.side("CF-202@o8") == "V"
        with pytest.raises(L.LeakageGuardError, match="CF-301"):
            pool.get("CF-301@o8")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def test_score_definition_and_edge_cases() -> None:
    gold = cases.generate(1)[0]
    zero = math.sqrt(sum(v * v for v in gold.values()))
    assert scoring.cf_rmse({}, gold) == pytest.approx(zero)
    assert scoring.cf_rmse(dict(gold), gold) == 0.0
    assert scoring.cf_rmse(json.dumps(gold), gold) == 0.0
    assert scoring.cf_rmse(None, gold) is None and scoring.cf_rmse("not json", gold) is None
    assert scoring.cf_rmse([1, 2, 3], gold) is None
    off = dict(gold)
    off["7"] = off.get("7", 0) + 3
    off["300"] = 99  # outside the value domain: ignored
    assert scoring.cf_rmse(off, gold) == pytest.approx(3.0)
    assert scoring.cf_rmse({"5": "2", "6": 1.0, "7": True}, {"5": 2, "6": 1}) == 0.0
    assert scoring.score_from_rmse(None) == 0.0 and scoring.score_from_rmse(0.0) == 1.0
    assert scoring.score_from_rmse(5.0) == pytest.approx(math.exp(-1.0))
    assert scoring.score_from_rmse(float("nan")) == 0.0 and scoring.score_from_rmse(-1) == 0.0
    assert scoring.CF_DOMAIN == [str(v) for v in range(256)]
    assert cases.compute_rmse({"1": 2}, {"1": 2}, []) == 0.0
    assert cases.canonicalize_counts({"3": 2, 3: 1, "1": 0, "2": 2.9}) == {"2": 2, "3": 3}


def test_non_count_objects_are_format_rows_and_broken_braces_are_tables() -> None:
    gold = cases.generate(1)[0]
    for answer in ({}, {"counts": dict(gold)}, {"answer": "see above"}, {"0": "three"},
                   json.dumps({"counts": dict(gold)}), "UNKNOWN", [1, 2], None):
        facts = scoring.cf_apply({"S": 0.4}, submissions=[{"agent_id": 0, "answer": answer}],
                                 sink_id=0, ground_truth=gold)
        row = scoring.finalize_row(facts)
        assert facts["cf_rmse"] is None and facts["cf_format_fail"] is True, answer
        assert row["S"] == 0.0 and row["format_fail"] and R.is_format_row(row), answer
        assert row["diag"]["failure_class"] == "format" and row["diag"]["cf"], answer
    one = scoring.cf_apply({"S": 0.0}, submissions=[{"agent_id": 0, "answer": {"7": 3, "note": 1}}],
                           sink_id=0, ground_truth=gold)
    assert one["cf_rmse"] is not None and one["cf_format_fail"] is False and one["S"] > 0.0
    exact = scoring.cf_apply({"S": 0.0}, submissions=[SimpleNamespace(
        model_dump=lambda: {"agent_id": 0, "answer": json.dumps(gold)})], sink_id=0, ground_truth=gold)
    assert exact["cf_rmse"] == 0.0 and exact["S"] == 1.0 and exact["success"] is True
    assert exact["cf_sum"] == 1024 and exact["cf_wrong_bins"] == 0
    # another agent's answer is never scored
    other = scoring.cf_apply({}, submissions=[{"agent_id": 3, "answer": gold}], sink_id=0,
                             ground_truth=gold)
    assert other["cf_format_fail"] is True
    # a table whose closing brace is missing or mistyped (')') is read as written
    broken = json.dumps(gold)[:-1]
    repaired = scoring.cf_apply({}, submissions=[{"agent_id": 0, "answer": broken + ")"}],
                                sink_id=0, ground_truth=gold)
    assert repaired["cf_rmse"] == 0.0
    assert scoring.is_count_table({"255": 1}) and not scoring.is_count_table({"256": 1})


def test_finalize_row_and_evaluator_summary_fields() -> None:
    import queenbee.evaluate as ev

    infra = scoring.finalize_row({"infra": "ConnectionError: refused", "S": None})
    assert infra["format_fail"] is False and infra["cf_rmse"] is None and infra["submitted_total"] is None
    scored = scoring.finalize_row({"S": 0.5, "cf_rmse": 3.4, "cf_sum": 1020, "diag": {"cf": True}})
    assert (scored["S"], scored["format_fail"], scored["submitted_total"]) == (0.5, False, 1020)
    broke = scoring.finalize_row({"S": 0.0, "error": "BudgetError: model-call budget exhausted",
                                  "success": True, "diag": {"failure_class": "consensus-wrong"}})
    assert broke["S"] == 0.0 and broke["success"] is False and broke["stage_score"] == 0.0
    assert broke["diag"]["failure_class"] == "budget" and broke["diag"]["error_class"] == "budget"
    assert broke["format_fail"] is True
    rows = [{"cf_rmse": 3.0}, {"cf_rmse": 5.0}, {"cf_rmse": None, "S": 0.0},
            {"infra": "timeout", "cf_rmse": None}]
    assert scoring.summary_fields(rows) == {"mean_rmse": 4.0, "rmse_rows": 2, "format_fail": 1}
    assert scoring.summary_fields([]) == {"mean_rmse": None, "rmse_rows": 0, "format_fail": 0}
    silo = ev._aggregate([dict(r, S=0.5) for r in rows])
    assert "mean_rmse" not in silo and "mean_rmse" not in ev._summary([silo], rows)
    with use_task("cf"):
        agg = ev._aggregate(rows)
        assert (agg["mean_rmse"], agg["rmse_rows"], agg["format_fail"], agg["cases_scored"]) == \
            (4.0, 2, 1, 3)
        assert ev._summary([agg], rows)["mean_rmse"] == 4.0


# --------------------------------------------------------------------------- #
# count-table JSON repair
# --------------------------------------------------------------------------- #


def test_count_table_repair_accepts_exactly_the_brace_quirk() -> None:
    loads = cj.tolerant_loads
    assert loads('{"0":1,"3":2,"255":4)') == {"0": 1, "3": 2, "255": 4}
    assert loads('{"0":1,"3":2,"255":4') == {"0": 1, "3": 2, "255": 4}
    assert loads(' {"0": 1, "255": 4 ] ') == {"0": 1, "255": 4}
    assert loads(b'{"7": 3)') == {"7": 3}
    for bad in ['{"0":1,"x":2', '{"0":1, "1": "a"', 'The table is {"0":1', '{"counts": {"0":1}',
                '{"0":1,,"1":2', '[1,2', '{"0":1}}', '', '{"0":-1', '{"1234": 5', '{"1": 2.5']:
        with pytest.raises(ValueError):
            loads(bad)
    assert cj.repair_count_table('{"0":1}') is None and cj.repair_count_table(7) is None
    assert loads('{"0": 1}') == {"0": 1}
    assert json.loads is not cj.tolerant_loads  # the host never installs it
    assert cj.install_if_requested({}) is False and json.loads is not cj.tolerant_loads


def _bootstrap(tmp_path: Path, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    program = tmp_path / "program.py"
    program.write_text("import json\nprint(sorted(json.loads('{\"7\": 3, \"9\": 1)').items()))\n")
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"max_output_bytes": 10_000, "worker_contract": "message_only_v2"}))
    env = {"PYTHONPATH": os.pathsep.join(p for p in sys.path if p), "PATH": os.environ.get("PATH", ""),
           "PYTHONDONTWRITEBYTECODE": "1"} | env_extra
    return subprocess.run(
        [sys.executable, "-m", "exp_graph.mas.python_worker_bootstrap", "--program", str(program),
         "--auth", str(auth), "--ledger", str(tmp_path / "ledger.json")],
        env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=120)


def test_worker_sandbox_repairs_count_tables_only_when_asked(tmp_path: Path) -> None:
    on = _bootstrap(tmp_path, {cj.REPAIR_ENV: "1"})
    assert on.returncode == 0, on.stderr[-2000:]
    assert on.stdout.strip() == "[('7', 3), ('9', 1)]"
    off = _bootstrap(tmp_path, {})
    assert off.returncode != 0 and "JSONDecodeError" in off.stderr
    assert _bootstrap(tmp_path, {cj.REPAIR_ENV: "0"}).returncode != 0


# --------------------------------------------------------------------------- #
# diagnosis cards
# --------------------------------------------------------------------------- #


def test_scored_path_card_fields_and_rendering() -> None:
    inst = cases.build_instance("CF-003")
    gold = dict(inst.ground_truth)
    own = [Counter(str(v) for v in s) for s in inst.shards]
    answer = dict(gold)
    answer["5"] = answer.get("5", 0) + 2      # a combining error of +2 on one value
    bad1 = dict(own[1])
    bad1["9"] = bad1.get("9", 0) + 1          # agent 1 miscounted its own shard by 1
    messages = [
        {"src": 1, "dst": 0, "round_sent": 0, "round_delivered": 1, "source_ids": [1],
         "body": json.dumps(bad1) + "\nTOTAL=125"},  # a TOTAL line off by 4
        {"src": 3, "dst": 2, "round_sent": 0, "round_delivered": 1, "source_ids": [3],
         "body": "counts: " + ", ".join(f"{k}: {v}" for k, v in own[3].items()) + "\nTOTAL=128"},
        {"src": 2, "dst": 0, "round_sent": 1, "round_delivered": 2, "source_ids": [2, 3],
         "body": "{}"},
    ]
    output = {"submissions": [{"agent_id": 0, "answer": answer, "submitted_round": 3}],
              "messages": messages, "rounds_executed": 4}
    facts = scoring.cf_apply({"S": 0.0, "infra": None}, submissions=output["submissions"],
                             sink_id=0, ground_truth=inst.ground_truth)
    assert facts["cf_rmse"] == pytest.approx(2.0)
    assert facts["S"] == pytest.approx(math.exp(-2.0 / 5.0), abs=1e-6)
    assert facts["cf_sum"] == 1026 and facts["cf_wrong_bins"] == 1
    card = cfd.cf_diag(facts, output, inst, base={"submit_round": 3, "n_messages": 3,
                                                  "n_lost_messages": 0})
    assert card["cf_local_k"] == 2 and card["cf_local_err"] == pytest.approx(1.0)
    assert card["cf_local_est"] == pytest.approx(2.0) and card["cf_merge_excess"] == pytest.approx(0.0)
    assert card["failure_class"] == "scattered-wrong"
    assert (card["cf_msgs_tables"], card["cf_labels_bad"], card["cf_cover_bad"]) == (2, 1, 0)
    assert "cf_submit_added" not in card  # a body into agent 0 holds no table
    assert cfd.parse_body_table(messages[1]["body"]) == dict(own[3])
    assert cfd.parse_body_table('{"counts": {"4": 2, "300": 1}}') == {"4": 2}
    assert cfd.parse_body_table("nothing here") is None
    with use_task("cf"):
        clean = dg.sanitize_diag(card)
        assert dg.sanitize_diag(clean) == clean and clean["cf_rmse"] == pytest.approx(2.0)
        assert "cf" not in clean and "agent_correct" in clean
        line = dg.render_diag_card("CF-003@o8", [{"diag": card, "S": facts["S"]}])
        assert "rmse=2.00" in line and "sum_dev=+2" in line and "local=1.00(2/8)" in line
        assert "labels_bad=1/2" in line and "cover_bad=0/2" in line
        assert dg.failure_class_glossary()["consensus-wrong"] == cfd.CF_GLOSSARY["consensus-wrong"]
    assert "rmse=" not in dg.render_diag_card("CF-003@o8", [{"diag": card, "S": facts["S"]}])
    assert dg.sanitize_diag(card).get("cf_rmse") is None
    # whole shards lost -> content-loss; a shard counted twice -> consensus-wrong
    lost = dict(gold)
    for k, v in own[5].items():
        lost[k] = lost[k] - v
    f2 = scoring.cf_apply({"S": 0.0}, submissions=[{"agent_id": 0, "answer": lost}], sink_id=0,
                          ground_truth=gold)
    assert cfd.cf_diag(f2, None, inst)["failure_class"] == "content-loss"
    twice = dict(gold)
    for k, v in own[6].items():
        twice[k] = twice[k] + v
    f3 = scoring.cf_apply({"S": 0.0}, submissions=[{"agent_id": 0, "answer": twice}], sink_id=0,
                          ground_truth=gold)
    assert cfd.cf_diag(f3, None, inst)["failure_class"] == "consensus-wrong"
    exact = scoring.cf_apply({}, submissions=[{"agent_id": 0, "answer": gold}], sink_id=0,
                             ground_truth=gold)
    assert cfd.cf_diag(exact, None, inst)["failure_class"] == "ok"
    assert cfd.cf_diag({"infra": "timeout"}, None, inst)["failure_class"] == "infra"
    budget = cfd.cf_diag({"S": 0.0, "cf_rmse": None, "error": "BudgetError: calls"}, None, inst)
    assert (budget["failure_class"], budget["error_class"]) == ("budget", "budget")
    # submit_added: agent 0 adds one shard a second time
    into0 = [{"src": s, "dst": 0, "round_sent": 0, "round_delivered": 1, "source_ids": [s],
              "body": json.dumps(dict(own[s]))} for s in range(1, 8)]
    added = cfd.coverage_bookkeeping({"messages": into0}, inst, 1024 + 128)
    assert added["cf_submit_added"] == 128 and added["cf_cover_bad"] == 0


def test_diag_fn_trace_row_and_post_change_summary(card_flags) -> None:
    inst = cases.build_instance("CF-001")
    gold = dict(inst.ground_truth)
    output = {"submissions": [{"agent_id": 0, "answer": gold}], "messages": [], "rounds_executed": 2}
    facts = scoring.cf_apply({"S": 0.0, "infra": None}, submissions=output["submissions"], sink_id=0,
                             ground_truth=gold)
    card = cfd.diag_fn(output, inst, facts, source=SEED, goal="sink")
    assert card["cf"] and card["failure_class"] == "ok" and card["cf_rmse"] == 0.0
    plain = SimpleNamespace(shards=[[1]] * 8, meta={}, benchmark="silo_bench", task_prompt="",
                            ground_truth=1, case_id="I-01")
    assert not dg.diagnose_execution(None, plain, {"S": 0.0}).get("cf")
    assert not cfd.diag_fn(None, plain, {"S": 0.0}).get("cf")
    trace = {"case_id": "CF-001", "facts": dict(facts), "output": output}
    assert not cr.trace_row(trace, inst, source=SEED, goal="sink")["diag"].get("cf")
    with use_task("cf"):
        row = cr.trace_row(trace, inst, source=SEED, goal=C.task_goal())
    assert row["diag"]["cf"] and row["diag"]["failure_class"] == "ok" and "credit" in row
    # the ledger's post-change line leaves out the per-agent counts CF cards cannot have
    stub = SimpleNamespace(archive=SimpleNamespace(rows={"g001_s0": {"CF-001@o8": {"0": {"diag": card}}}}))
    entry = {"verdict": "refuted", "target_units": ["CF-001@o8"], "gen": 1,
             "hypothesis": {"mechanism": "redundant counting"}, "observed": {"CF-001@o8": -0.05}}
    with use_task("cf"):
        after = L.EvoRun._after_diag(stub, "g001_s0", ["CF-001@o8"])
        line = ep._after_lines([dict(entry, after_diag=after)], set())[0]
    assert after["class"] == "ok" and after["n_agents"] is None
    assert after["own_shard_only"] is None and after["answer_seen"] is None
    assert line.endswith("after the change: class=ok") and "n/a" not in line


# --------------------------------------------------------------------------- #
# the seed program and the S0 screen
# --------------------------------------------------------------------------- #


def test_seed_is_the_evo_seed_with_anchored_edits() -> None:
    from exp_graph.mas.python_code import validate_python_source
    from queenbee.program.genome import check_genome, genome_bounds, genome_region, splice_genome

    assert _sha(SEED) == SEED_SHA256 and SEED == cfseed.build_program(cfseed.SEED_PHASES)
    report = validate_python_source(SEED, worker_contract="message_only_v2")
    assert report.valid, report.errors
    genome = genome_region(SEED)
    assert genome and genome.startswith("PHASES = [")
    assert splice_genome(SEED, genome) == SEED and check_genome(genome, incumbent_source=SEED) == []
    evo = evo_seed_source()
    a, b = genome_bounds(SEED), genome_bounds(evo)
    assert SEED[:a[0]] == evo[:b[0]] and SEED[a[1]:] == evo[b[1]:]  # host prefix and main()
    assert 'information_goal == "sink"' not in genome and len(cfseed.WI_PARTIAL) <= 800
    assert '"scale_rounds_log2": True, "extra_rounds": 1' in genome
    with pytest.raises(RuntimeError, match="sink branch"):
        cfseed.build_program(cfseed.SEED_PHASES, base=SEED)
    for n, (calls, messages) in {8: (16, 7), 5: (11, 4), 2: (4, 1), 10: (21, 9)}.items():
        sim = S.simulate_readable_coverage(SEED, n, 64, goal="sink")
        assert sim.ok and (sim.calls, sim.messages, sim.lost_edges) == (calls, messages, 0), n
        assert sim.readable[0] == n == sim.known[0]


@needs_bench
def test_screen_runs_at_n8_with_the_submitter_coverage_rule() -> None:
    assert cfs.seed_calls() == 16
    short = _with_phases([{"kind": "tree_reduce", "rounds": 2, "wi": cfseed.WI_PARTIAL}])
    calls = _with_phases([{"kind": "tree_reduce", "rounds": 4}] * 4)  # 4 x 15 sends
    gather = _with_phases([{"kind": "gather_to_hub_digest", "rounds": 1, "wi": cfseed.WI_PARTIAL}])
    named = _with_phases([{"kind": "tree_reduce", "rounds": 3, "scale_rounds_log2": True,
                           "wi": "Report the count frequency of your data."}])
    with use_task("cf"):
        res = S.screen_for_task(SEED, known_fps={}, goal="all_agents")  # goal ignored: n = 8, sink
        per = res.per_n[8]
        assert res.ok and set(res.per_n) == {8} and res.lint_hits == [] and res.rank_penalty == 0.0
        assert per["readable"][0] == 8 == per["known"][0] and per["lost_edges"] == 0
        assert per["calls"] == 16 and res.readable5 == [] and res.pred_calls5 == 0
        key = S.fp_key_for_task(SEED)
        assert key == res.fp_key == S.program_fp_key(SEED, n_list=(8,), goal="sink")
        assert key.startswith("n8=") and "|" not in key
        dup = S.screen_for_task(SEED, known_fps={key: "v1"})
        assert not dup.ok and dup.reasons == ["duplicate-of:v1"] and dup.dup_of == "v1"
        evo = S.screen_for_task(evo_seed_source())
        assert evo.reasons == ["unread-content@n8: the submitting agent 0 submits with ids whose "
                               "bodies it never read (readable 1 < known 8)"]
        low = S.screen_for_task(short)
        assert low.reasons == ["coverage@n8: the submitting agent 0 reads 4 < 8 shards"]
        sim = S.simulate_readable_coverage(short, 8, 64, goal="sink")
        assert low.rank_penalty == pytest.approx(1 / 8 + sim.lost_edges / sim.messages, abs=1e-6)
        many = S.screen_for_task(calls)
        assert [r for r in many.reasons if r.startswith("calls@n8:")] == \
            [f"calls@n8: {many.per_n[8]['calls']} > 3x v1 (16)"]
        assert S.screen_for_task(gather).ok
        lint = S.screen_for_task(named)
        assert not lint.ok and "wi-lint: task-specific terms ['count frequency']" in lint.reasons
    # the coverage rule over every agent (all_agents) rejects the tree seed
    every = S.screen_program(SEED, n_list=(8,), goal="all_agents", v1_calls={8: 16},
                             vocabulary=cfs.vocabulary())
    assert not every.ok and any(r.startswith("coverage@n8: agents [1, 2") for r in every.reasons)


_TREE_HEAD = '''    rank = (agent_id - selected_primary) % n_agents
    step = 1 << local_round
'''
_N8_ONLY = '''    if n_agents == 8 and local_round == 0 and agent_id == 1:
        return {"mode": "send", "recipients": [0, 2]}
'''


@needs_bench
def test_behaviour_fingerprint_sees_edits_that_change_n8_only(tmp_path: Path) -> None:
    import threading

    assert SEED.count(_TREE_HEAD) == 1
    child = SEED.replace(_TREE_HEAD, _N8_ONLY + _TREE_HEAD)
    with use_task("cf"):
        fa, fb = C.behavior_fingerprint(SEED), C.behavior_fingerprint(child)
        assert fa and fb and fa != fb
        # without n = 8 the edit is invisible (same key at n = 2 / 5 / 10)
        assert C.behavior_fingerprint(SEED, ns=(2, 5, 10)) == C.behavior_fingerprint(child, ns=(2, 5, 10))
        res = S.screen_for_task(child, known_fps={S.fp_key_for_task(SEED): "v1"})
        assert res.ok, res.reasons
        fps = {}
        for name, src in (("v1", SEED), ("g000_s0", child)):
            racer = SimpleNamespace(_lock=threading.RLock(), _programs={}, root=tmp_path / name,
                                    _save_programs=lambda: None)
            fps[name] = R.Racer.register_program(racer, name, src).fp
    assert fps == {"v1": fa, "g000_s0": fb}
    assert C.behavior_fingerprint(SEED) not in (fa, fb)  # the Silo key (n = 2 / 5 / 10, all agents)


# --------------------------------------------------------------------------- #
# planner texts, split, thresholds, command-line defaults
# --------------------------------------------------------------------------- #


def test_planner_texts() -> None:
    b5 = PythonRunBudgets.for_rounds(64, n_agents=5)  # what the loop passes
    with use_task("cf"):
        assert ep._header() == texts.CF_HEADER and ep._unit_legend() == texts.CF_UNIT_LEGEND
        card = ep._api_card(budgets=b5, preserve_dups=False)
        assert card.startswith(ac.API_CARD_HEADER) and "every agent must be correct" not in card
        assert "S = exp(-RMSE / 5)" in card and "information_goal is \"sink\"" in card
        assert card.endswith("Budget caps of this run: max_rounds=64, max_model_calls=536, "
                             "max_messages=3560, max_completion_tokens=512000.\n")
        dups = ep._api_card(budgets=None, preserve_dups=True)
        assert dups.count(ac.PRESERVE_DUPS_RULE) == 1 and dups.startswith(ac.API_CARD_HEADER)
        assert ep._diag_cards({}, []).endswith("one shard a second time. " + cr.credit_legend()
                                                + "\nno diagnosed runs\n")
        statements = get_task().t_templates(["CF-001"])
    plain = texts.cf_api_card()
    for old, new in ((texts._SCORING_OLD, texts._SCORING_NEW), (texts._RULE5_OLD, texts._RULE5_NEW),
                     (texts._INTERP_OLD, texts._INTERP_NEW)):
        assert plain.count(new) == 1
        plain = plain.replace(new, old)
    assert plain == ac.API_CARD  # the card differs from the shared one in the three anchors only
    assert texts.render_api_card() == texts.cf_api_card()
    assert statements == [{"unit_id": "CF", "title": "Count Frequency",
                           "output_sentence": statements[0]["output_sentence"], "protocol_sentence": ""}]
    assert statements[0]["output_sentence"].startswith("A JSON object mapping each integer")
    assert ep._header() == ep._HEADER  # the default task again


def test_split_thresholds_and_command_line_defaults(tmp_path: Path, monkeypatch) -> None:
    import queenbee.evaluate as ev

    for var in (L.PLANNER_MODEL_ENV, L.WORKER_MODEL_ENV, "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert cases.default_split(5) == {"name": "cf", "T": list(TRAIN), "V": list(VALID)}
    with use_task("cf"):
        split = L.default_split({}, 9)
        assert (split.name, split.T, split.V) == ("cf", TRAIN, VALID)
        th = R.load_thresholds(None)
        assert (th.tau_r1, th.tau_2, th.tau_h, th.tau_hi, th.loss_max, th.refute_quanta) == \
            (0.0, 0.1, 0.15, 0.25, 1.0, 1.0)
        assert th.refute_margin("CF-001@o8") == pytest.approx(1 / 8)
        assert ev.split_templates("train") == TRAIN + VALID and ev.split_templates("test") == TEST
    with use_task("silo"):
        args, task = tasks.parse_task_args(L.build_parser(), [
            "--task", "cf", "--arm", "full", "--split-seed", "1", "--root", str(tmp_path / "r")])
        assert task is CF_TASK
        cfg = L.config_from_args(args)
        assert (cfg.budget, cfg.final_reserve, cfg.K, cfg.python_max_rounds) == (48.0, 12.0, 3, 64)
        assert cfg.vacuous_guard and cfg.local_only_diag and cfg.attempt_diag
        frozen = cfg.frozen()
        assert frozen["task"] == "cf" and frozen["vacuous_guard"] and frozen["attempt_diag"]
        # a real run needs a census, models and keys, never a ladder manifest or a split file
        assert L.missing_settings(cfg) == ["--units-from", f"--worker-model (or ${L.WORKER_MODEL_ENV})",
                                           f"--planner-model (or ${L.PLANNER_MODEL_ENV})",
                                           f"${L.DEFAULT_API_KEY_ENV}"]
        args, _ = tasks.parse_task_args(ev.build_parser(), [
            "--task", "cf", "--split", "test", "--seed-program", "s", "--out", str(tmp_path / "x")])
        assert (args.agents, args.python_max_rounds) == (8, 64)
        with pytest.raises(SystemExit):
            ev.main(["--task", "cf", "--split", "test", "--seed-program", "s", "--llm", "fake",
                     "--goal", "all_agents", "--out", str(tmp_path / "x.json")])
    assert not (tmp_path / "x.json").exists()
    silo = L.EvoConfig(root=tmp_path / "s", worker_model="w", planner_model="p")
    assert L.missing_settings(silo) == ["--units-from", "--ladder-manifest", "--split-file",
                                        f"${L.DEFAULT_API_KEY_ENV}"]


# --------------------------------------------------------------------------- #
# execution and end-to-end runs (fake worker, fake planner)
# --------------------------------------------------------------------------- #


def test_execution_through_the_sandbox_with_the_fake_worker(tmp_path: Path, monkeypatch,
                                                            card_flags) -> None:
    import queenbee.program.execute as ex

    seen: dict[str, Any] = {}
    real_payload = ex._build_python_execution_payload

    def spy(**kw: Any) -> dict[str, Any]:
        seen["payload"] = real_payload(**kw)
        return seen["payload"]

    class Recording(ex.CodeProcessRunner):
        def __init__(self, *args: Any, **kw: Any) -> None:
            seen["env"] = dict(kw.get("extra_env") or {})
            super().__init__(*args, **kw)

    monkeypatch.setattr(ex, "_build_python_execution_payload", spy)
    monkeypatch.setattr(ex, "CodeProcessRunner", Recording)
    monkeypatch.setenv("QB_TRACE_DIR", str(tmp_path / "traces"))
    instance = cases.build_instance("CF-001")
    req = {"cfg": SimpleNamespace(llm_provider="fake", worker_model="w", request_timeout=120.0,
                                  python_max_rounds=64),
           "job": SimpleNamespace(program_id="v1", unit_id="CF-001@o8"), "source": SEED,
           "instance": instance, "seed": 7, "artifacts_dir": tmp_path / "artifacts"}
    with use_task("cf"):
        row = C.task_execute(req)
    payload = seen["payload"]
    assert (payload["information_goal"], payload["selected_primary"], payload["n_agents"]) == ("sink", 0, 8)
    assert payload["worker_llm"]["provider"] == "fake" and seen["env"] == {cj.REPAIR_ENV: "1"}
    for key in ("S", "cf_rmse", "C", "model_calls", "n_messages", "infra", "format_fail",
                "submitted_total", "diag", "cf_format_fail"):
        assert key in row, key
    assert row["infra"] is None and row["model_calls"] == 16 and row["n_messages"] == 7
    assert row["format_fail"] is True and row["S"] == 0.0  # the offline worker submits no table
    assert row["diag"]["cf"] and row["diag"]["failure_class"] in CF_CLASSES
    assert row["diag"]["submit_round"] == 4 and row["diag"]["phase_tokens"]
    traces = list((tmp_path / "traces").glob(f"CF-001_{_sha(SEED)[:12]}_*.json"))
    assert len(traces) == 1
    trace = json.loads(traces[0].read_text())
    with use_task("cf"):
        traced = cr.trace_row(trace, instance, source=SEED, goal="sink")
    assert traced["diag"].get("cf") and traced.get("credit") is not None


@needs_bench
def test_loop_runs_the_task_end_to_end_offline(tmp_path: Path, monkeypatch, card_flags) -> None:
    for var in (L.PLANNER_MODEL_ENV, L.WORKER_MODEL_ENV):
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "root"
    argv = ["--task", "cf", "--arm", "full", "--split-seed", "1", "--budget", "20",
            "--final-reserve", "6", "--root", str(root), "--fake", "--parallel-cases", "4",
            "--infra-backoff", "0"]
    with use_task("silo"):
        summary = L.main(argv)
    assert summary["run_status"] == "ok" and summary["champion"]
    config = json.loads((root / "config.json").read_text())
    assert config["task"] == "cf" and config["K"] == 3 and config["budget"] == 20.0
    assert config["vacuous_guard"] and config["local_only_diag"] and config["attempt_diag"]
    assert (root / "programs" / "v1.py").read_text() == SEED
    split = json.loads((root / "state" / "split.json").read_text())
    assert (split["T"], split["V"]) == (list(TRAIN), list(VALID))
    champion = json.loads((root / "champion.json").read_text())
    assert _sha((root / "champion.py").read_text()) == champion["sha256"]
    assert champion["thresholds"]["tau_h"] == 0.15 and champion["thresholds"]["source"] == "cf"
    final = json.loads((root / "state" / "final.json").read_text())
    assert final["done"] and final["V_units"] and final["champion"] == summary["champion"]
    assert set(final["V_units"] + final["V_guards"]) <= {f"{t}@o8" for t in VALID}
    evolution = json.loads((root / "state" / "evolution.json").read_text())
    assert evolution["budget_matched"] and evolution["generations"] >= 2
    assert json.loads((root / "summary.json").read_text())["leak_hits"] == []
    records = [json.loads(x) for x in (root / "executions.jsonl").read_text().splitlines() if x.strip()]
    assert records and {r["unit_id"] for r in records} <= {f"{t}@o8" for t in TRAIN + VALID}
    for r in records:
        if r.get("purpose") == "vselect":
            assert r["unit_id"].split("@")[0] in VALID
        elif not r.get("external"):
            assert r["unit_id"].split("@")[0] in TRAIN
            assert "format_fail" in r["row"] and "submitted_total" in r["row"]
    gen0 = json.loads((root / "state" / "generation_000.json").read_text())
    keys = [s["screen"]["fp_key"] for s in gen0["slots"] if (s.get("screen") or {}).get("fp_key")]
    assert keys and all(k.startswith("n8=") and "|" not in k for k in keys)
    prompts = sorted((root / "prompts").glob("*_prompt.txt")) + sorted((root / "mints").glob("*/mint_prompt.txt"))
    assert prompts
    for path in prompts:
        text = path.read_text()
        assert text.startswith(texts.CF_HEADER) and texts.CF_UNIT_LEGEND in text
        assert "S = exp(-RMSE / 5)" in text and "every agent must be correct" not in text
        assert "[CF] Count Frequency" in text and "max_model_calls=536," in text
        assert not re.search(r"(?<![\w-])(CF-20[1-3]|CF-3\d\d)(?!\d)", text), path.name
    # a resume of the finished run executes nothing again
    before = len(records)
    with use_task("silo"):
        again = L.main(argv + ["--resume"])
    assert again["champion"] == summary["champion"]
    assert len((root / "executions.jsonl").read_text().splitlines()) == before


def _stub_execute(req: dict[str, Any]) -> dict[str, Any]:
    """A deterministic CF-shaped executor: the seed sits at RMSE 6, any
    other program at RMSE 2..6 from its genome hash, plus a (unit, rep)
    offset of up to 0.5; calls / messages / submit round from the program's
    simulated schedule; card and row fields from the task's own functions."""

    from queenbee.program.genome import genome_region

    source, job, instance = req["source"], req["job"], req["instance"]
    h = _sha(genome_region(source) or source)
    noise = int(_sha(f"{h}|{job.unit_id}|{int(job.rep)}")[:6], 16) % 501 / 1000.0
    rmse = (6.0 if source == SEED else 2.0 + int(h[:6], 16) % 400 / 100.0) + noise
    sim = S.simulate_readable_coverage(source, 8, 64, goal="sink")
    s = round(scoring.score_from_rmse(rmse), 6)
    row = {"infra": None, "execution_class": "completed", "success": False, "S": s,
           "stage_score": s, "C": 7000.0 * sim.calls, "prompt_tokens": 1500 * sim.calls,
           "completion_tokens": 5500 * sim.calls, "model_calls": sim.calls,
           "n_messages": sim.messages, "cf_rmse": rmse, "cf_sum": 1024 - int(rmse),
           "cf_wrong_bins": int(rmse * rmse), "cf_format_fail": False}
    row["diag"] = cfd.cf_diag(row, None, instance, base={
        "submit_round": sim.submit_round, "n_messages": sim.messages,
        "n_lost_messages": sim.lost_edges})
    return scoring.finalize_row(row)


@needs_bench
def test_loop_races_confirms_and_selects_on_validation_units(tmp_path: Path, card_flags) -> None:
    root = tmp_path / "root"
    with use_task("cf"):
        cfg = L.EvoConfig(root=root, arm="full", split_seed=1, budget=32.0, final_reserve=8.0,
                          fake=True, K=3, planner_model="p", worker_model="w", traces=False,
                          planner_backoff_s=0.0, infra_backoff_s=0.0, parallel_cases=2,
                          outage_wait_s=0.0, vacuous_guard=True, local_only_diag=True,
                          attempt_diag=True)
        summary = L.run_evo(cfg, L.EvoDeps(execute=_stub_execute, planner_client=L.FakeEvoPlanner()))
    assert summary["run_status"] == "ok" and summary["leak_hits"] == []
    assert summary["verdicts"].get("confirmed", 0) >= 1
    final = json.loads((root / "state" / "final.json").read_text())
    assert final["done"] and len(final["candidates"]) >= 2 and summary["champion"] in final["candidates"]
    records = [json.loads(x) for x in (root / "executions.jsonl").read_text().splitlines() if x.strip()]
    vselect = [r for r in records if r.get("purpose") == "vselect"]
    assert vselect and {r["unit_id"] for r in vselect} <= {f"{t}@o8" for t in VALID}
    others = [r for r in records if r.get("purpose") != "vselect"]
    assert others and {r["unit_id"] for r in others} <= {f"{t}@o8" for t in TRAIN}
    paid = [r for r in records if not r.get("external") and not r.get("cached")]
    assert paid and all(r["row"]["diag"]["cf"] and r["row"]["format_fail"] is False for r in paid)
    memory = json.loads((root / "state" / "memory.json").read_text())
    entries = memory["ledger"]["entries"] if isinstance(memory["ledger"], dict) else memory["ledger"]
    executed = [e for e in entries if e.get("after_diag")]
    assert executed and all(e["after_diag"]["own_shard_only"] is None for e in executed)
    prompts = [p.read_text() for p in sorted((root / "prompts").glob("*_prompt.txt"))]
    assert any("rmse=" in text and "sum_dev=" in text and "wrong_values=" in text for text in prompts)


def test_evaluator_runs_a_census_case_and_a_test_case_offline(tmp_path: Path, monkeypatch) -> None:
    import queenbee.evaluate as ev

    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "QB_TRACE_DIR", "QUEENBEE_WORKER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    census = tmp_path / "census.json"
    with use_task("silo"):
        assert ev.main(["--task", "cf", "--split", "train", "--cases", "CF-001", "--seed-program", "v1",
                        "--llm", "fake", "--repeats", "1", "--out", str(census)]) == 0
    data = json.loads(census.read_text())
    fp = data["config"]["fingerprint"]
    assert (fp["task"], fp["agents"], fp["goal"]) == ("cf", 8, "sink")
    [row] = data["arms"]["v1"]["repeats"]["0"]["rows"]
    assert row["case_id"] == "CF-001" and row["infra"] is None and row["model_calls"] == 16
    assert row["format_fail"] is True and row["cf_rmse"] is None and row["S"] == 0.0
    assert row["diag"]["cf"] and row["submitted_total"] is None
    assert data["arms"]["v1"]["meta"] == {"seed": "cf", "source_sha256": _sha(SEED), "max_rounds": 64}
    summary = data["arms"]["v1"]["summary"]
    assert (summary["mean_rmse"], summary["rmse_rows"], summary["format_fail"]) == (None, 0, 1)
    assert data["arms"]["v1"]["repeats"]["0"]["aggregate"]["format_fail"] == 1
    with use_task("cf"):
        units = U.load_units([census])
        unit = units["CF-001@o8"]
        assert (unit.n_agents, unit.rung, unit.v1_S) == (8, "o8", (0.0,))
        assert unit.instance_sha256 == ev.instance_sha256(cases.build_instance("CF-001"))
    champion = tmp_path / "champion.py"
    champion.write_text(SEED)
    test_out = tmp_path / "test.json"
    with use_task("silo"):
        assert ev.main(["--task", "cf", "--split", "test", "--cases", "CF-301", "--program",
                        f"champ={champion}", "--llm", "fake", "--repeats", "1",
                        "--out", str(test_out)]) == 0
        with pytest.raises(SystemExit, match="split"):
            ev.main(["--task", "cf", "--split", "test", "--cases", "CF-302", "--seed-program", "v1",
                     "--llm", "fake", "--repeats", "1", "--out", str(census)])
        with pytest.raises(SystemExit, match="outside the train"):
            ev.main(["--task", "cf", "--split", "train", "--cases", "CF-301", "--seed-program", "v1",
                     "--llm", "fake", "--repeats", "1", "--out", str(tmp_path / "leak.json")])
    result = json.loads(test_out.read_text())
    [trow] = result["arms"]["champ"]["repeats"]["0"]["rows"]
    assert trow["case_id"] == "CF-301" and trow["format_fail"] is True
    assert result["config"]["case_meta"]["CF-301"]["instance_sha256"] == \
        ev.instance_sha256(cases.build_instance("CF-301"))
    # TEST rows never enter the loop's census loader
    with use_task("cf"):
        with pytest.raises(LD.LadderLeakError, match="CF-301"):
            U.load_eval_rows([test_out])


# --------------------------------------------------------------------------- #
# guards of the shared paths under this task
# --------------------------------------------------------------------------- #


def test_evaluator_refuses_another_team_size(tmp_path: Path, monkeypatch, capsys) -> None:
    import queenbee.evaluate as ev

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "a5.json"
    with use_task("silo"):
        with pytest.raises(SystemExit):
            ev.main(["--task", "cf", "--split", "train", "--cases", "CF-001", "--agents", "5",
                     "--seed-program", "v1", "--llm", "fake", "--repeats", "1", "--out", str(out)])
    assert "--agents 5: the cases are instances of 8 agents" in capsys.readouterr().err
    assert not out.exists()


def _failing_trace(template: str, n: int) -> dict[str, Any]:
    gold = dict(cases.build_instance(template).ground_truth)
    answer = dict(gold)
    first = next(iter(answer))
    answer[first] += 3
    return {"case_id": template, "n_agents": 8, "source_sha": "0" * 12,
            "facts": {"S": 0.5, "C": 1000.0, "prompt_tokens": 100 + n, "infra": None},
            "output": {"submissions": [{"agent_id": 0, "answer": answer}], "messages": []},
            "ground_truth": gold}


@needs_bench
def test_trace_dirs_never_reclass_the_units_of_this_task(tmp_path: Path, card_flags) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    for n, template in enumerate(TRAIN):
        (traces / f"{template}_{'0' * 12}_{n}.json").write_text(json.dumps(_failing_trace(template, n)))
    units = {f"{t}@o8" for t in TRAIN}
    with use_task("cf"):
        # the Silo-Bench heuristics read the sink's table as a wrong answer
        assert set(U.structural_units_from_traces([traces])) == units
        run = L.EvoRun(L.EvoConfig(root=tmp_path / "r", fake=True, trace_dirs=(traces,)))
        assert run.near_miss_units == {} and run.structural_units == {}
        frozen = json.loads((tmp_path / "r" / "state" / "split.json").read_text())
        assert (frozen["near_miss_units"], frozen["structural_units"]) == ([], [])
    with use_task(dataclasses.replace(CF_TASK, name="cf-traces", trace_unit_evidence=True)):
        run = L.EvoRun(L.EvoConfig(root=tmp_path / "t", fake=True, trace_dirs=(traces,)))
        assert set(run.structural_units) == units


def test_write_split_needs_no_ladder_manifest_for_this_task(tmp_path: Path) -> None:
    census = tmp_path / "census.json"
    with use_task("cf"):
        census.write_text(json.dumps(L.fake_census()))
    with use_task("silo"):
        doc = L.main(["--task", "cf", "--split-seed", "4", "--write-split",
                      str(tmp_path / "split.json"), "--units-from", str(census)])
    assert (list(doc["T"]), list(doc["V"])) == (list(TRAIN), list(VALID))
    assert doc["ladder_manifests"] == []
    with pytest.raises(L.EvoStateError, match="--units-from and --ladder-manifest"):
        L.write_split_file(tmp_path / "silo.json", split_seed=1, unit_sources=[census],
                           ladder_manifests=[])


def test_loop_refuses_to_start_without_its_screen_vocabulary(tmp_path: Path) -> None:
    def missing() -> Any:
        raise FileNotFoundError("no development texts")

    with use_task(dataclasses.replace(CF_TASK, name="cf-novocab", wi_vocabulary=missing)):
        with pytest.raises(L.EvoStateError, match="vocabulary cannot be built.*no development texts"):
            L.EvoRun(L.EvoConfig(root=tmp_path / "r", fake=True))
    assert not (tmp_path / "r").exists()


def test_diag_fn_builds_the_card_when_the_builtin_card_fails(monkeypatch) -> None:
    import queenbee.program.execute as ex

    inst = cases.build_instance("CF-002")
    gold = dict(inst.ground_truth)
    output = {"submissions": [{"agent_id": 0, "answer": gold}], "messages": []}
    facts = scoring.cf_apply({"S": 0.0, "infra": None}, submissions=output["submissions"],
                             sink_id=0, ground_truth=gold)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("card bug")

    monkeypatch.setattr(dg, "diagnose_execution", broken)
    card = cfd.diag_fn(output, inst, facts)
    assert card["cf"] and card["failure_class"] == "ok" and card["submit_round"] is None
    row = dict(facts)
    ex._attach_diag(row, output, inst, diag_fn=CF_TASK.diag_fn)
    assert row["diag"]["cf"] and row["diag"]["cf_rmse"] == 0.0
    plain = SimpleNamespace(shards=[[1]] * 8, meta={}, benchmark="silo_bench", task_prompt="",
                            ground_truth=1, case_id="I-01")
    bare: dict[str, Any] = {"S": 0.0}
    ex._attach_diag(bare, None, plain, diag_fn=CF_TASK.diag_fn)
    # another task's instance: the hook calls the (failing) built-in card
    # directly, so no card, as without the hook
    assert "diag" not in bare


def test_synthetic_census_takes_the_answer_shape_of_the_task_instances() -> None:
    with use_task("cf"):
        census = L.fake_census()
        shape = dg.instance_answer_shape(cases.build_instance("CF-001"))
    rows = [r for rep in census["arms"]["v1"]["repeats"].values() for r in rep["rows"]]
    assert census["config"]["agents"] == 8 and {r["case_id"] for r in rows} == set(TRAIN + VALID)
    assert shape == "collection" and {r["diag"]["answer_shape"] for r in rows} == {shape}
    silo = L.fake_census({"I-01": [0.4], "II-11": [0.6], "III-21": [1.0]})
    assert [r["diag"]["answer_shape"] for r in silo["arms"]["v1"]["repeats"]["0"]["rows"]] == \
        ["scalar", "scalar", "collection"]


# --------------------------------------------------------------------------- #
# isolation from the default task
# --------------------------------------------------------------------------- #

_SILO_STATE = r'''
import hashlib, json, sys
mode = sys.argv[1]
import queenbee.tasks as tasks
import queenbee.evaluate
import queenbee.evo.loop
loaded_before = "queenbee.tasks.cf" in sys.modules
if mode != "plain":
    tasks.resolve_task("cf")
if mode == "used":
    from queenbee.evo import common as C, prompt as P
    from queenbee.tasks.cf import cases, scoring
    with tasks.use_task("cf"):
        inst = cases.build_instance("CF-001")
        scoring.cf_rmse(dict(inst.ground_truth), inst.ground_truth)
        P._header()
        C.behavior_fingerprint(tasks.get_task().seed_source())
from queenbee.evo import common, diagnosis, prompt, screen
from queenbee.evo.seed import evo_seed_source
from queenbee.program.execute import task_worker_env
seed = evo_seed_source()
state = {
    "task": tasks.get_task().name, "goal": common.task_goal(), "rungs": common.task_rungs(),
    "test": common.task_test_ids(), "dev": common.task_dev_ids(),
    "header": prompt._header(), "legend": prompt._unit_legend(),
    "card": prompt._api_card(budgets=None, preserve_dups=False),
    "glossary": dict(diagnosis.failure_class_glossary()),
    "fp": common.behavior_fingerprint(seed), "fp_key": screen.fp_key_for_task(seed),
    "env": task_worker_env(), "hooks": sorted(common.task_score_hooks()),
    "loads": f"{json.loads.__module__}.{json.loads.__qualname__}",
}
print(hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest())
print(loaded_before)
'''


def test_registering_and_using_the_task_leaves_the_default_task_untouched(tmp_path: Path) -> None:
    env = {"PYTHONPATH": os.pathsep.join(p for p in sys.path if p), "PATH": os.environ.get("PATH", ""),
           "PYTHONDONTWRITEBYTECODE": "1", cases.CASES_DIR_ENV: str(tmp_path / "cases")}
    if os.environ.get("TMPDIR"):
        env["TMPDIR"] = os.environ["TMPDIR"]
    states = {}
    for mode in ("plain", "registered", "used"):
        out = subprocess.run([sys.executable, "-c", _SILO_STATE, mode], env=env, cwd=str(tmp_path),
                             capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        digest, loaded = out.stdout.split()
        states[mode] = digest
        assert loaded == "False"  # the command-line modules never load the task by themselves
    assert len(set(states.values())) == 1, states


def test_repair_flag_reaches_only_the_sandboxes_of_this_task(tmp_path: Path, monkeypatch) -> None:
    from exp_graph.mas.python_code_runner import CodeProcessRunner

    import queenbee.program.mint as mt
    from queenbee.program.execute import task_worker_env

    monkeypatch.setenv(cj.REPAIR_ENV, "1")  # set in the host: never forwarded by itself
    payload = {"worker_llm": {"provider": "fake"}}
    assert cj.REPAIR_ENV not in CodeProcessRunner(extra_env=task_worker_env())._child_env(payload)
    with use_task("cf"):
        assert CodeProcessRunner(extra_env=task_worker_env())._child_env(payload)[cj.REPAIR_ENV] == "1"
    seen: list[dict[str, str]] = []

    class Recording(mt.CodeProcessRunner):
        def __init__(self, *args: Any, **kw: Any) -> None:
            seen.append(dict(kw.get("extra_env") or {}))
            super().__init__(*args, **kw)

    monkeypatch.setattr(mt, "CodeProcessRunner", Recording)
    budgets = PythonRunBudgets.for_rounds(64, n_agents=5)
    for name in ("silo", "cf"):  # the mint's dry-run sandbox
        with use_task(name):
            goal = C.task_goal()
            request = mt.build_planner_request(n_agents=5, goal=goal, worker_contract="message_only_v2")
            runtime = mt.build_mint_runtime(llm_provider="fake", worker_model="w", goal=goal,
                                            worker_contract="message_only_v2", max_parallel_agents=5,
                                            request_timeout=60.0, n_agents=5, budgets=budgets)
            with pytest.raises(ValueError, match="incumbent"):
                mt.mint_python_challenger(planner_client=None, planner_model="p", prompt="P",
                                          request=request, runtime=runtime, workdir=tmp_path / name,
                                          genome_only=True, incumbent_source="")
    assert seen == [{}, {cj.REPAIR_ENV: "1"}]
    assert json.loads is not cj.tolerant_loads
