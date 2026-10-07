"""Budget meter of a run (``queenbee.evo.budget.BudgetMeter``): per-phase
spend, persistence and resume, planner usage and the execution cap."""

from __future__ import annotations

import json
from pathlib import Path

from queenbee.evo.budget import PHASES, TRAIN_PHASES, BudgetMeter


def _row(S: float = 1.0, *, infra: str | None = None) -> dict:
    return {"infra": infra, "S": S, "C": 100.5, "prompt_tokens": 10,
            "completion_tokens": 5, "model_calls": 3}


def test_phase_names_are_the_persisted_keys(tmp_path: Path) -> None:
    assert PHASES == ("train_stage1", "train_val", "train_debias", "test", "test_baseline")
    assert TRAIN_PHASES == {"train_stage1", "train_val", "train_debias"}
    meter = BudgetMeter(tmp_path / "budget_meter.json")
    meter.save()
    state = json.loads((tmp_path / "budget_meter.json").read_text())
    assert sorted(state["phases"]) == sorted(PHASES)
    assert state["cap_basis"] == "scored_train_executions" and state["cap_executions"] == 0


def test_rows_are_charged_persisted_and_resumed(tmp_path: Path) -> None:
    path = tmp_path / "root" / "budget_meter.json"
    meter = BudgetMeter(path, cap_executions=0)
    assert meter.charge_rows("train_stage1", [_row(), _row(infra="502"), "not a row"]) == \
        {"executions": 2, "scored": 1, "cached_rows": 0}
    assert meter.charge_rows("train_stage1", [_row()], cached=True) == \
        {"executions": 0, "scored": 0, "cached_rows": 1}
    meter.charge_rows("train_val", [_row(0.5)])
    meter.charge_rows("test", [_row()])
    meter.add_counter("exec_eq_surcharge", 1)
    meter.add_counter("exec_eq_surcharge", 0)
    on_disk = json.loads(path.read_text())  # saved after every charge
    stage1 = on_disk["phases"]["train_stage1"]
    assert (stage1["executions"], stage1["scored"], stage1["infra"], stage1["cached_rows"]) == (2, 1, 1, 1)
    assert stage1["prompt_tokens"] == 10 and stage1["model_calls"] == 3 and stage1["C"] == 100.5
    assert on_disk["counters"] == {"tie_remint_executions": 0, "exec_eq_surcharge": 1}

    totals = BudgetMeter(path).snapshot()["totals"]  # resumed from disk
    assert totals["train_executions"] == 3 and totals["train_scored_executions"] == 2
    assert totals["train_cached_rows"] == 1 and totals["train_worker_tokens"] == 30
    assert totals["train_C"] == 201.0 and totals["test_executions"] == 1
    assert BudgetMeter(path, fresh=True).snapshot()["totals"]["train_executions"] == 0


def test_planner_usage_buckets(tmp_path: Path) -> None:
    meter = BudgetMeter(tmp_path / "m.json")
    meter.charge_planner([
        {"call": "mint", "model": "m1", "prompt_tokens": 100, "completion_tokens": 20, "wall_s": 1.25},
        {"call": "mint", "model": "m1"},
        "junk",
    ], round_index=3)
    meter.charge_planner([], round_index=4)
    planner = meter.snapshot()["planner"]
    assert planner["calls"] == 2 and planner["unknown_usage_calls"] == 1
    assert (planner["prompt_tokens"], planner["completion_tokens"]) == (100, 20)
    assert planner["wall_s"] == 1.25
    assert planner["by_call"]["mint"] == {"calls": 2, "prompt_tokens": 100, "completion_tokens": 20}
    assert set(planner["by_model"]) == {"m1"} and set(planner["by_round"]) == {"3"}
    assert meter.snapshot()["totals"]["planner_tokens"] == 120


def test_execution_cap_counts_scored_train_rows_only(tmp_path: Path) -> None:
    meter = BudgetMeter(None, cap_executions=2)
    meter.charge_rows("train_stage1", [_row(infra="timeout"), _row()])
    meter.charge_rows("test", [_row(), _row()])
    assert not meter.exhausted() and meter.train_executions() == 2
    meter.charge_rows("train_val", [_row()])
    assert meter.exhausted() and meter.snapshot()["exhausted"]
    assert not BudgetMeter(None).exhausted()
