"""Spend ledger of one run: worker executions, worker tokens, planner usage.

Every worker case-execution a run pays for is charged to exactly one phase:

- ``train_stage1``  evolution: R1 / R2 / guard / confirmation reps, fresh
  parent reps, curriculum and inheritance reps
  (:data:`queenbee.evo.race.PURPOSE_PHASE`)
- ``train_val``     the final reserve: V selection and the coverage top-up
  T reps
- ``train_debias``  re-runs that remove the selection bias from a chosen
  program's score; counted in the training totals (the evolution loop does
  not charge it)
- ``test``          executions of a selected program on TEST
- ``test_baseline`` reference arms on TEST (e.g. the seed program (v1))

The three ``train_*`` phases make up the training totals and the execution
cap; the evolution loop itself never charges a TEST phase.

Rows served from a cache cost nothing and are counted separately
(``cached_rows``) so "executions" always means paid work.  Planner calls
(``mint`` / ``remint``) are charged from the
:func:`queenbee.program.mint.planner_usage_entry` records the minting
function appends to its ``usage_log``; the evolution loop charges each
planner call lost to a kill mid-call as one call of unknown usage.

The meter persists to ``budget_meter.json`` after EVERY charge, so a crash
or resume never forgets spend already recorded.  The optional execution cap
counts SCORED (non-infra) train executions: an infrastructure failure says
nothing about the program, and letting it consume the cap would give an arm
on a less reliable network fewer real candidates than its peers.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

PHASES: tuple[str, ...] = (
    "train_stage1",
    "train_val",
    "train_debias",
    "test",
    "test_baseline",
)
TRAIN_PHASES: frozenset[str] = frozenset(
    {"train_stage1", "train_val", "train_debias"}
)


def _phase_zero() -> dict[str, Any]:
    return {
        "executions": 0,
        "scored": 0,
        "infra": 0,
        "cached_rows": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "C": 0.0,
        "model_calls": 0,
    }


def _planner_zero() -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "unknown_usage_calls": 0,
        "wall_s": 0.0,
        "by_call": {},
        "by_model": {},
        "by_round": {},
    }


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class BudgetMeter:
    """JSON-persisted, thread-safe spend ledger for one run directory."""

    def __init__(
        self,
        path: Path | None,
        *,
        cap_executions: int = 0,
        fresh: bool = False,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.cap_executions = max(0, int(cap_executions or 0))
        self._lock = threading.Lock()
        self.state = self._empty()
        if self.path is not None and self.path.exists() and not fresh:
            try:
                loaded = json.loads(self.path.read_text())
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                self._merge_loaded(loaded)
        self.state["cap_executions"] = self.cap_executions
        self.state["cap_basis"] = "scored_train_executions"

    # ------------------------------------------------------------------ #
    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "version": 1,
            "phases": {phase: _phase_zero() for phase in PHASES},
            "planner": _planner_zero(),
            "counters": {"tie_remint_executions": 0},
            "cap_executions": 0,
            "cap_basis": "scored_train_executions",
        }

    def _merge_loaded(self, loaded: dict[str, Any]) -> None:
        for phase, values in dict(loaded.get("phases") or {}).items():
            slot = self.state["phases"].setdefault(phase, _phase_zero())
            for key, value in dict(values or {}).items():
                slot[key] = value
        planner = dict(loaded.get("planner") or {})
        for key, value in planner.items():
            self.state["planner"][key] = value
        for key, value in dict(loaded.get("counters") or {}).items():
            self.state["counters"][key] = value

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ #
    def charge_rows(
        self,
        phase: str,
        rows: Iterable[dict[str, Any]],
        *,
        cached: bool = False,
    ) -> dict[str, int]:
        """Charge worker rows to ``phase``; cached rows cost zero."""

        rows = [row for row in rows if isinstance(row, dict)]
        delta = {"executions": 0, "scored": 0, "cached_rows": 0}
        with self._lock:
            slot = self.state["phases"].setdefault(phase, _phase_zero())
            for row in rows:
                if cached:
                    slot["cached_rows"] += 1
                    delta["cached_rows"] += 1
                    continue
                slot["executions"] += 1
                delta["executions"] += 1
                if row.get("infra"):
                    slot["infra"] += 1
                    continue
                slot["scored"] += 1
                delta["scored"] += 1
                slot["prompt_tokens"] += _as_int(row.get("prompt_tokens")) or 0
                slot["completion_tokens"] += (
                    _as_int(row.get("completion_tokens")) or 0
                )
                try:
                    slot["C"] += float(row.get("C") or 0.0)
                except (TypeError, ValueError):
                    pass
                slot["model_calls"] += _as_int(row.get("model_calls")) or 0
            self.save()
        return delta

    def charge_planner(
        self,
        entries: Iterable[dict[str, Any]],
        *,
        round_index: int | None = None,
    ) -> None:
        """Charge planner usage entries (``usage_log`` contract)."""

        entries = [entry for entry in entries if isinstance(entry, dict)]
        if not entries:
            return
        with self._lock:
            planner = self.state["planner"]
            for entry in entries:
                prompt = _as_int(entry.get("prompt_tokens"))
                completion = _as_int(entry.get("completion_tokens"))
                call = str(entry.get("call") or "unknown")
                model = str(entry.get("model") or "unknown")
                planner["calls"] += 1
                if prompt is None and completion is None:
                    planner["unknown_usage_calls"] += 1
                planner["prompt_tokens"] += prompt or 0
                planner["completion_tokens"] += completion or 0
                try:
                    planner["wall_s"] = round(
                        float(planner["wall_s"])
                        + float(entry.get("wall_s") or 0.0),
                        3,
                    )
                except (TypeError, ValueError):
                    pass
                for bucket_name, key in (
                    ("by_call", call),
                    ("by_model", model),
                    ("by_round", str(round_index)),
                ):
                    bucket = planner[bucket_name].setdefault(
                        key,
                        {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
                    )
                    bucket["calls"] += 1
                    bucket["prompt_tokens"] += prompt or 0
                    bucket["completion_tokens"] += completion or 0
            self.save()

    def add_counter(self, name: str, amount: int) -> None:
        if not amount:
            return
        with self._lock:
            counters = self.state["counters"]
            counters[name] = int(counters.get(name, 0)) + int(amount)
            self.save()

    # ------------------------------------------------------------------ #
    def train_executions(self, *, scored_only: bool = False) -> int:
        key = "scored" if scored_only else "executions"
        with self._lock:
            return sum(
                int(values.get(key, 0))
                for phase, values in self.state["phases"].items()
                if phase in TRAIN_PHASES
            )

    def exhausted(self) -> bool:
        """True once scored train executions reached the cap (0 = no cap)."""

        if self.cap_executions <= 0:
            return False
        return self.train_executions(scored_only=True) >= self.cap_executions

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = copy.deepcopy(self.state)
        phases = state["phases"]
        train = [values for phase, values in phases.items() if phase in TRAIN_PHASES]
        test = [
            values for phase, values in phases.items() if phase not in TRAIN_PHASES
        ]
        state["totals"] = {
            "train_executions": sum(int(v["executions"]) for v in train),
            "train_scored_executions": sum(int(v["scored"]) for v in train),
            "train_cached_rows": sum(int(v["cached_rows"]) for v in train),
            "train_worker_tokens": sum(
                int(v["prompt_tokens"]) + int(v["completion_tokens"])
                for v in train
            ),
            "train_C": round(sum(float(v["C"]) for v in train), 1),
            "test_executions": sum(int(v["executions"]) for v in test),
            "planner_tokens": int(state["planner"]["prompt_tokens"])
            + int(state["planner"]["completion_tokens"]),
        }
        state["exhausted"] = (
            self.cap_executions > 0
            and state["totals"]["train_scored_executions"] >= self.cap_executions
        )
        return state
