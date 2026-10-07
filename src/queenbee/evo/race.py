"""Racing in QueenBee-Evo: R1, R2 and confirmation reps on T units.

The :class:`Racer` is shared infrastructure: every arm races its candidates
the same way, whichever experience components the arm uses.

Jobs and the exec cache
-----------------------
A :class:`Job` is ``(program_id, unit_id, rep, purpose, batch_id)``; its
``key`` (``program|unit|r<rep>``) is the resume identity: a key with a scored
row is never executed again.  Rows are cached on (behaviour fingerprint,
unit, instance_sha256, rep) under the same worker configuration: a
behaviourally identical program asking for the same (unit, rep) reuses the
row at zero cost, inside a batch too.  Every execution (infra attempts and
cache hits included) is appended to ``executions.jsonl`` through
:class:`RaceLog` (a subclass of :class:`~queenbee.evo.common.ExecLog`).

Reps.  A job planned without an explicit rep (R1, R2, guard, V selection,
...) gets the next rep index of *that program* on the unit (0 for a program
with no row or planned job there), so a duplicate program hits the cache.  A
``parent_fresh`` or ``confirm`` job is an INDEPENDENT draw: the next rep
index of the behaviour fingerprint over every program and every external
(census) row, so it can never be served from the cache.  Rep allocation is
persisted per batch in ``plans.jsonl`` before anything runs: re-planning a
batch id after a crash (``kill -9`` in the middle of R2) returns the same
jobs, and the scored keys are skipped -- no execution is repeated.

Infra rows (``row["infra"]``) are re-run up to ``max_infra_reruns`` times and
never scored as 0; a job whose attempts are exhausted comes back with
``infra_final`` and ``S = None``.  Every paid attempt is charged to the
:class:`~queenbee.evo.budget.BudgetMeter` (cache hits as cached rows).  A
paid scored run costs :func:`exec_eq` execution-equivalents (an o10 run
counts 2); the meter's counter ``exec_eq_surcharge`` adds up the excess over
one.

Race
----
* **R1** (:meth:`Racer.race_r1`): 1 execution per candidate on its first
  target unit + a FRESH parent rep on every (parent, R1 unit) that has none
  this generation, in the same batch.  ``d1 = S_c - S_fresh(parent)``;
  survive iff ``d1 >= tau_r1`` and the row is not a format failure.  A
  missing fresh parent row (infra_final) is never replaced by the archived
  mean (regression to the mean): the candidate does not survive.
* **R2** (:meth:`Racer.race_r2`): top-1 survivor (top-2 when both have
  ``d1 >= tau_hi``; ties -> lower tokens, lower screen rank penalty) x
  {``n_r2_units`` (default 2) other T frontier units, answer shape
  different from the target where possible, other template, chosen by a
  stable hash (never by S), + 1 rotating guard}.  ``D`` = mean dS over the
  R1 unit and the R2 units; high-confidence tier iff ``D >= tau_2`` and the
  guard lost no more than ``loss_max`` / n.
* **Confirmation** (:meth:`Racer.race_confirm`): 1 fresh rep of a new front
  winner on the unit it wins (the winners come from
  :meth:`queenbee.evo.memory.Archive.newly_front_winning`).

dS baselines: on the R1 unit ONLY the parent's FRESH reps of this generation
(no fresh rep -> no dS there, status ``no_fresh_parent``); on the R2 units
the parent's fresh reps, else its archived mean over every scored row
(census included, ``"archived"``), else the same baseline of the seed
program (v1), usually its census mean (``"v1_archived"``: an evolved parent
was never run on most R2 units).  ``D`` needs a dS on every R2 unit.

:meth:`Racer.outcome` folds a candidate's rows of one generation into the
per-candidate record the ledger needs (``observed`` {unit: dS}, ``target_dS``,
``D``, guard, tier) and :func:`host_verdict` turns it into the verdict
(confirmed / refuted / inconclusive / screened).

Thresholds
----------
:class:`Thresholds` defaults to the provisional values (tau_r1 0, tau_2 =
tau_h 0.2, tau_hi 0.4, loss_max 1 quantum); the active task or a
calibration file may replace them (:func:`load_thresholds`).

Leakage: every job unit must be a DEV template (TEST and unknown template
ids raise :class:`~queenbee.evo.common.LeakageGuardError` before anything
runs); with a split, every training purpose (:data:`T_PURPOSES`: R1 / R2 /
guard / confirm / fresh parent reps, curriculum, inherit, topup) needs a T
unit and ``vselect`` a V unit (``census`` may be either).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from queenbee.evo import common as _common
from queenbee.evo.common import BudgetExceeded, ExecLog, LeakageGuardError
from queenbee.tasks import get_task

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

EPS = 1e-9
GOAL = _common.GOAL
WORKER_MODEL = _common.WORKER_MODEL
PYTHON_MAX_ROUNDS = _common.PYTHON_MAX_ROUNDS
MAX_INFRA_RERUNS = 2

#: Job purposes; any other purpose (``test`` included) is refused: TEST never
#: runs here.
PURPOSES: tuple[str, ...] = (
    "census", "curriculum", "parent_fresh", "r1", "r2", "guard", "confirm", "vselect",
    "inherit", "topup",
)
#: BudgetMeter phase of each purpose: ``train_val`` = the loop's final
#: reserve (V selection and the ``topup`` T reps, ``loop.FINAL_PURPOSES``),
#: ``train_stage1`` = evolution.
PURPOSE_PHASE: dict[str, str] = {p: "train_stage1" for p in PURPOSES}
PURPOSE_PHASE["vselect"] = "train_val"
PURPOSE_PHASE["topup"] = "train_val"
#: Purposes that need an independent draw (never served from the cache).
FRESH_PURPOSES: frozenset[str] = frozenset({"parent_fresh", "confirm"})
#: Purposes allowed on each split side (census covers every dev template).
#: ``topup`` = the loop's ``--final-coverage-topup`` T reps (only run with
#: that flag).
T_PURPOSES: frozenset[str] = frozenset(
    {"curriculum", "parent_fresh", "r1", "r2", "guard", "confirm", "inherit", "topup"}
)
V_PURPOSES: frozenset[str] = frozenset({"vselect"})

#: Provisional thresholds: the defaults when neither the active task nor a
#: calibration file sets them.
PROVISIONAL: dict[str, float] = {
    "tau_r1": 0.0, "tau_2": 0.2, "tau_h": 0.2, "tau_hi": 0.4, "loss_max": 1.0,
}
VERDICTS: tuple[str, ...] = ("confirmed", "refuted", "inconclusive", "screened")
#: A proposal is refuted only when its target dS is at least this many quanta
#: (1/n) BELOW the fresh parent: a tie is no evidence against the hypothesis
#: (with a near-binary S, two runs of one program tie in at least half of
#: all pairs; a parent rep at 1.0 leaves no headroom).
#: ``Thresholds(refute_quanta=0)`` gives the strict rule "refuted if mean
#: dS <= 0".
REFUTE_QUANTA = 1.0

_BATCH_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")
_GEN_RE = re.compile(r"^g(\d+)")


class RaceError(RuntimeError):
    """A malformed job / candidate / plan (nothing ran)."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else None


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def template_of(unit_id: Any) -> str:
    return str(unit_id or "").split("@", 1)[0].strip()


def rung_of(unit_id: Any) -> str:
    text = str(unit_id or "")
    return text.split("@", 1)[1].strip() if "@" in text else "o5"


def tier_of(unit_id: Any) -> str:
    return template_of(unit_id).split("-", 1)[0]


def unit_n_agents(unit_id: Any) -> int:
    """Agents of a unit from its rung (``o5`` / ``x5`` -> 5, ``o10`` -> 10)."""

    match = re.match(r"^[ox](\d+)$", rung_of(unit_id))
    return int(match.group(1)) if match else 5


def exec_eq(unit_id: Any) -> int:
    """Execution-equivalents of one run: the active task's ``rung_cost`` of
    the rung when it names it, else 2 for a team of 10 or more agents (an
    o10 execution counts 2) and 1 otherwise."""

    cost = get_task().rung_cost
    rung = rung_of(unit_id)
    if cost is not None and rung in cost:
        return int(cost[rung])
    return 2 if unit_n_agents(unit_id) >= 10 else 1


def unit_sort_key(unit_id: Any) -> tuple:
    template = template_of(unit_id)
    tier, _, num = template.partition("-")
    tiers = {"I": 0, "II": 1, "III": 2}
    try:
        number = int(num)
    except ValueError:
        number = 99
    rung = rung_of(unit_id)
    return (tiers.get(tier, 9), number, unit_n_agents(unit_id), rung, str(unit_id))


def _stable_rank(*parts: Any) -> str:
    return _sha256("|".join(str(p) for p in parts))


def gen_of_batch(batch_id: Any) -> int | None:
    match = _GEN_RE.match(str(batch_id or ""))
    return int(match.group(1)) if match else None


def batch_id_for(gen: int, stage: str) -> str:
    return f"g{int(gen)}:{stage}"


def row_S(row: Mapping[str, Any] | None) -> float | None:
    """S of a scored row; None for infra / missing rows (never 0)."""

    if not isinstance(row, Mapping) or is_infra(row) or row.get("infra_final"):
        return None
    value = row.get("S")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_infra(row: Mapping[str, Any] | None) -> bool:
    if not isinstance(row, Mapping):
        return False
    return bool(row.get("infra")) or str(row.get("execution_class") or "") == "infra"


def is_format_row(row: Mapping[str, Any] | None) -> bool:
    """A scored row whose failure is the answer / contract format."""

    if not isinstance(row, Mapping) or is_infra(row):
        return False
    diag = row.get("diag")
    if isinstance(diag, Mapping):
        if diag.get("failure_class") == "format" or diag.get("error_class") == "format":
            return True
    try:
        from queenbee.program.execute import classify_row_error

        return classify_row_error(dict(row)) == "format"
    except Exception:  # noqa: BLE001 - classification is advisory
        return "answerformaterror" in str(row.get("error") or "").lower()


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #


def assert_dev_unit(unit_id: Any, *, where: str) -> str:
    """Refuse TEST (and unknown) templates; returns the unit id."""

    uid = str(unit_id or "")
    _common.assert_not_test([uid], where=where)
    from queenbee.evo.credit import assert_not_test as credit_guard
    from queenbee.evo.ladder import LadderLeakError, assert_dev_template

    try:
        credit_guard([uid], where=where)
        assert_dev_template(uid)
    except (LadderLeakError, RuntimeError) as exc:
        raise LeakageGuardError(f"LEAKAGE_GUARD ({where}): {exc}") from exc
    if "@" not in uid:
        raise RaceError(f"{where}: {uid!r} is not a unit id (template@rung)")
    return uid


def _split_sides(split: Any) -> tuple[set[str], set[str]] | None:
    if split is None:
        return None
    t = _get(split, "T")
    v = _get(split, "V")
    if t is None and v is None:
        return None
    return {str(x) for x in (t or ())}, {str(x) for x in (v or ())}


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Thresholds:
    """Racing and verdict thresholds.

    ``loss_max`` is in quanta: a guard may drop by at most ``loss_max / n``
    (one quantum, 1/n, by default)."""

    tau_r1: float = PROVISIONAL["tau_r1"]
    tau_2: float = PROVISIONAL["tau_2"]
    tau_h: float = PROVISIONAL["tau_h"]
    tau_hi: float = PROVISIONAL["tau_hi"]
    loss_max: float = PROVISIONAL["loss_max"]
    fpr: float | None = None
    tpr: float | None = None
    calib_sha256: str | None = None
    source: str = "provisional"
    #: refuted iff target dS <= -refute_quanta / n (``REFUTE_QUANTA``).
    refute_quanta: float = REFUTE_QUANTA
    report: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def guard_allowance(self, unit_id: Any) -> float:
        return float(self.loss_max) / float(unit_n_agents(unit_id))

    def refute_margin(self, unit_id: Any = None) -> float:
        """How far below the fresh parent a target dS must be to refute
        (``refute_quanta`` quanta of the unit; n = 5 without a unit)."""

        n = unit_n_agents(unit_id) if unit_id else 5
        return float(self.refute_quanta) / float(n)

    def guard_ok(self, drop: float | None, unit_id: Any) -> bool | None:
        if drop is None:
            return None
        return float(drop) <= self.guard_allowance(unit_id) + EPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau_r1": self.tau_r1, "tau_2": self.tau_2, "tau_h": self.tau_h,
            "tau_hi": self.tau_hi, "loss_max": self.loss_max, "fpr": self.fpr,
            "tpr": self.tpr, "calib_sha256": self.calib_sha256, "source": self.source,
            "refute_quanta": self.refute_quanta, "report": dict(self.report),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Thresholds":
        data = dict(data or {})
        kw: dict[str, Any] = {}
        for key in ("tau_r1", "tau_2", "tau_h", "tau_hi", "loss_max", "refute_quanta"):
            if data.get(key) is not None:
                kw[key] = float(data[key])
        for key in ("fpr", "tpr"):
            if data.get(key) is not None:
                kw[key] = float(data[key])
        if data.get("calib_sha256"):
            kw["calib_sha256"] = str(data["calib_sha256"])
        if data.get("source"):
            kw["source"] = str(data["source"])
        if isinstance(data.get("report"), Mapping):
            kw["report"] = dict(data["report"])
        return cls(**kw)

    @classmethod
    def provisional(cls) -> "Thresholds":
        return cls()


def load_thresholds(value: Any = None) -> Thresholds:
    """Thresholds from a Thresholds / mapping / JSON path (None = the active
    task's ``thresholds``, else provisional).  A calibration JSON may nest
    them under ``"thresholds"``."""

    if value is None:
        value = get_task().thresholds
    if value is None:
        return Thresholds.provisional()
    if isinstance(value, Thresholds):
        return value
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if isinstance(value, Mapping) and isinstance(value.get("thresholds"), Mapping):
        value = value["thresholds"]
    return Thresholds.from_dict(value)


def host_verdict(outcome: Mapping[str, Any] | None, thresholds: Thresholds | None = None) -> str:
    """The verdict of a proposal from its :meth:`Racer.outcome` record.

    * ``screened``: never executed (S0 reject, mint failure, duplicate);
    * ``refuted``: target dS <= -``refute_quanta`` / n (one quantum below
      the fresh parent by default; ``refute_quanta = 0`` gives the strict
      rule "dS <= 0", see :data:`REFUTE_QUANTA`); n = agents of the
      R1 unit;
    * ``confirmed``: target dS >= tau_h and the guard did not drop by
      more than ``loss_max`` / n (the guard runs in R2, so a candidate that
      never reached R2 cannot be confirmed; without a scored guard row it
      is not confirmed either, unless R2 found no guard unit to run and the
      racer has ``vacuous_guard`` on);
    * ``inconclusive``: everything else (incl. infra-only rows and ties).

    ``target_dS`` is the dS on the R1 unit (R1 + confirmation reps vs the
    fresh parent; :meth:`Racer.outcome`)."""

    th = thresholds or Thresholds()
    if not outcome or not outcome.get("executed"):
        # infra-only rows are not the program's fault: never "screened"
        return "inconclusive" if (outcome or {}).get("status") == "infra_final" else "screened"
    target = outcome.get("target_dS")
    if target is None:
        return "inconclusive"
    unit = outcome.get("r1_unit") or (list(outcome.get("target_units") or []) or [None])[0]
    if float(target) <= -th.refute_margin(unit) + EPS:
        return "refuted"
    if float(target) >= float(th.tau_h) - EPS and outcome.get("guard_ok") is True:
        return "confirmed"
    return "inconclusive"


# --------------------------------------------------------------------------- #
# Jobs, programs, log
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Job:
    """One execution request.  ``key`` ignores purpose / batch: the
    (program, unit, rep) triple is the resume identity (the exec cache
    matches on the behaviour fingerprint instead of the program id)."""

    program_id: str
    unit_id: str
    rep: int
    purpose: str
    batch_id: str = ""

    @property
    def key(self) -> str:
        return f"{self.program_id}|{self.unit_id}|r{int(self.rep)}"

    def to_dict(self) -> dict[str, Any]:
        return {"program_id": self.program_id, "unit_id": self.unit_id,
                "rep": int(self.rep), "purpose": self.purpose, "batch_id": self.batch_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Job":
        return cls(str(data["program_id"]), str(data["unit_id"]), int(data["rep"]),
                   str(data.get("purpose") or "r1"), str(data.get("batch_id") or ""))


def coerce_job(job: Any) -> Job:
    if isinstance(job, Job):
        return job
    if isinstance(job, Mapping):
        return Job.from_dict(job)
    # any object with the Job attributes (duck typing)
    return Job(str(_get(job, "program_id")), str(_get(job, "unit_id")), int(_get(job, "rep", 0)),
               str(_get(job, "purpose", "r1")), str(_get(job, "batch_id", "")))


@dataclass
class RaceProgram:
    program_id: str
    source: str
    fp: str | None
    source_sha256: str
    arm: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"program_id": self.program_id, "fp": self.fp,
                "source_sha256": self.source_sha256, "arm": self.arm, "meta": self.meta}


class RaceLog(ExecLog):
    """The append-only execution log plus external (census) rows, which
    feed the exec cache and the dS baselines but never count as spend.

    ``worker_cfg``: the sha of the racer's worker configuration
    (:meth:`RaceConfig.worker_cfg_sha`).  Every record carries one and the
    exec cache only serves rows of the SAME configuration (fail closed: a
    record without one is never served from the cache)."""

    worker_cfg: str | None = None

    def cache_lookup(self, fp: str | None, unit_id: str, instance_sha: str,
                     rep: int) -> dict | None:
        if not fp:
            return None
        with self._lock:
            for record in self.records:
                if (
                    record.get("fp") == fp
                    and record.get("unit_id") == unit_id
                    and record.get("instance_sha256") == instance_sha
                    and int(record.get("rep", -1)) == int(rep)
                    and not record.get("cached")
                    and not is_infra(record.get("row") or {})
                    and self.worker_cfg is not None
                    and record.get("worker_cfg") == self.worker_cfg
                ):
                    return record
        return None

    def exec_eq_scored(self) -> float:
        with self._lock:
            return float(sum(
                float(r.get("exec_eq") or 1) for r in self.records
                if not r.get("cached") and not r.get("external")
                and not is_infra(r.get("row") or {})
            ))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.records)
        by_purpose: dict[str, dict[str, float]] = {}
        external = 0
        for r in records:
            if r.get("external"):
                external += 1
                continue
            slot = by_purpose.setdefault(
                str(r.get("purpose")),
                {"executions": 0, "scored": 0, "infra": 0, "cached": 0, "exec_eq": 0},
            )
            if r.get("cached"):
                slot["cached"] += 1
                continue
            slot["executions"] += 1
            if is_infra(r.get("row") or {}):
                slot["infra"] += 1
            else:
                slot["scored"] += 1
                slot["exec_eq"] += int(r.get("exec_eq") or 1)
        return {
            "by_purpose": by_purpose,
            "executions": sum(s["executions"] for s in by_purpose.values()),
            "scored": sum(s["scored"] for s in by_purpose.values()),
            "infra": sum(s["infra"] for s in by_purpose.values()),
            "cached": sum(s["cached"] for s in by_purpose.values()),
            "exec_eq_scored": sum(s["exec_eq"] for s in by_purpose.values()),
            "external_rows": external,
        }

    def select(self, pred: Callable[[Mapping[str, Any]], bool]) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.records if pred(r)]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class RaceConfig:
    """Execution settings, handed to the executor as ``req["cfg"]``.
    :func:`queenbee.evo.common.real_execute` (the default executor unless
    the active task supplies its own ``execute``) reads ``llm_provider``,
    ``worker_model``, ``request_timeout`` and ``python_max_rounds``."""

    fake: bool = False
    worker_model: str = WORKER_MODEL
    python_max_rounds: int = PYTHON_MAX_ROUNDS
    parallel_cases: int = 6
    #: worker request timeout (s); large x5 / o10 cases can take over half
    #: an hour
    request_timeout: float = 5400.0
    max_infra_reruns: int = MAX_INFRA_RERUNS
    infra_backoff_s: float = 60.0
    traces: bool = True
    #: R2 extra units must come from another tier than the target (when at
    #: least ``n_r2_units`` such templates are available); off by default.
    r2_other_tier: bool = False
    n_r2_units: int = 2
    #: ``--vacuous-guard`` (off by default): when R2 finds no guard unit to
    #: run (typically an empty guard pool: with weak workers the seed program
    #: solves no T unit stably, so there is nothing solved to protect), the
    #: guard condition holds vacuously instead of failing closed, so a clear
    #: R1 gain that reached R2 can be confirmed.
    vacuous_guard: bool = False

    @property
    def llm_provider(self) -> str:
        return "fake" if self.fake else "openai"

    def worker_fingerprint(self) -> dict[str, Any]:
        """What makes two executions of one program exchangeable (the exec
        cache and external rows must agree on it)."""

        return {"llm_provider": self.llm_provider, "worker_model": self.worker_model,
                "python_max_rounds": int(self.python_max_rounds), "goal": _common.task_goal(),
                "worker_contract": _common.WORKER_CONTRACT}

    def worker_cfg_sha(self) -> str:
        return _sha256(json.dumps(self.worker_fingerprint(), sort_keys=True))[:16]

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


def normalize_candidate(cand: Any) -> dict[str, Any]:
    """A proposal as the racer needs it: ``program_id``, ``parent_id`` (first
    of ``parent_ids``), ``target_units`` (>= 1; the first is the R1 unit),
    ``brief_id``, ``kind``, ``rank_penalty`` (screen), ``target_class``."""

    pid = _get(cand, "program_id")
    if not pid:
        raise RaceError("candidate without program_id")
    parents = list(_get(cand, "parent_ids", []) or [])
    parent = _get(cand, "parent_id") or (parents[0] if parents else None)
    if not parent:
        raise RaceError(f"candidate {pid}: no parent")
    targets = [str(u) for u in (_get(cand, "target_units", []) or []) if u]
    brief = _get(cand, "brief")
    if not targets and brief is not None:
        targets = [str(u) for u in (_get(brief, "target_units", []) or []) if u]
    if not targets:
        raise RaceError(f"candidate {pid}: no target unit (the brief must name one)")
    screen = _get(cand, "screen", {}) or {}
    penalty = _get(cand, "rank_penalty")
    if penalty is None:
        penalty = _get(screen, "rank_penalty", 0.0)
    return {
        "program_id": str(pid),
        "parent_id": str(parent),
        "target_units": list(dict.fromkeys(targets)),
        "brief_id": _get(cand, "brief_id") or _get(brief, "brief_id"),
        "kind": _get(cand, "kind") or _get(brief, "kind"),
        "target_class": _get(cand, "target_class") or _get(brief, "target_class"),
        "rank_penalty": float(penalty or 0.0),
    }


def select_for_r2(r1_cands: Sequence[Mapping[str, Any]], th: Thresholds) -> list[str]:
    """R2 entry: the top-1 survivor by d1 (top-2 when both have
    d1 >= tau_hi).  Ties: lower R1 tokens, then lower screen rank penalty,
    then proposal order."""

    survivors = [c for c in r1_cands if c.get("survive")]
    order = {c["program_id"]: i for i, c in enumerate(r1_cands)}

    def key(c: Mapping[str, Any]) -> tuple:
        cost = c.get("C")
        return (-float(c["d1"]), float(cost) if cost is not None else math.inf,
                float(c.get("rank_penalty") or 0.0), order[c["program_id"]])

    ranked = sorted(survivors, key=key)
    if not ranked:
        return []
    chosen = [ranked[0]["program_id"]]
    if len(ranked) >= 2 and float(ranked[0]["d1"]) >= th.tau_hi - EPS \
            and float(ranked[1]["d1"]) >= th.tau_hi - EPS:
        chosen.append(ranked[1]["program_id"])
    return chosen


def pick_r2_units(
    program_id: str,
    target_unit: str,
    t_frontier: Sequence[str],
    *,
    gen: int,
    shapes: Mapping[str, Any] | None = None,
    k: int = 2,
    other_tier: bool = False,
    exclude: Iterable[str] = (),
) -> list[str]:
    """``k`` other T frontier units: never the target's template, answer shape
    different from the target where possible (and another tier with
    ``other_tier``), distinct templates, ordered by a stable hash of
    (gen, program, unit) -- never by S, so the choice cannot chase noise."""

    shapes = dict(shapes or {})
    banned = set(exclude) | {target_unit}
    pool = [u for u in dict.fromkeys(t_frontier)
            if u not in banned and template_of(u) != template_of(target_unit)]
    if other_tier:
        other = [u for u in pool if tier_of(u) != tier_of(target_unit)]
        if len({template_of(u) for u in other}) >= k:
            pool = other
    tshape = shapes.get(target_unit)
    pref = [u for u in pool if tshape and shapes.get(u) and shapes[u] != tshape]
    rest = [u for u in pool if u not in set(pref)]
    ordered = sorted(pref, key=lambda u: _stable_rank(gen, program_id, u)) + \
        sorted(rest, key=lambda u: _stable_rank(gen, program_id, u))
    out: list[str] = []
    for unit in ordered:
        if template_of(unit) in {template_of(o) for o in out}:
            continue
        out.append(unit)
        if len(out) >= k:
            break
    return out


def pick_guard(guards: Sequence[str], *, gen: int, slot: int = 0,
               exclude_templates: Iterable[str] = ()) -> str | None:
    """The rotating guard: saturated T units in unit order, index
    ``(gen + slot) mod len`` over those whose template is not in use."""

    banned = set(exclude_templates)
    pool = sorted({g for g in guards if template_of(g) not in banned}, key=unit_sort_key)
    if not pool:
        return None
    return pool[(int(gen) + int(slot)) % len(pool)]


# --------------------------------------------------------------------------- #
# Racer
# --------------------------------------------------------------------------- #


class Racer:
    """Job scheduling, exec cache, fresh parent reps, paired dS and the
    R1 / R2 / confirmation stages (module docstring).  Everything it knows
    is on disk under ``root`` (``executions.jsonl``, ``plans.jsonl``,
    ``race_programs.json`` + ``programs/``, ``budget_meter.json``), so a new
    Racer on the same root resumes where a killed one stopped."""

    def __init__(
        self,
        root: str | Path,
        *,
        resolver: Any,
        config: RaceConfig | None = None,
        execute: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        meter: Any = None,
        thresholds: Thresholds | Mapping[str, Any] | str | Path | None = None,
        split: Any = None,
        credit_from_trace: Callable[..., Mapping[str, Any] | None] | None = None,
        v1_id: str = "v1",
    ) -> None:
        from queenbee.evo.budget import BudgetMeter

        if resolver is None:
            raise RaceError("Racer needs a resolver (unit id -> (instance, instance_sha256))")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or RaceConfig()
        self.resolver = resolver
        self.execute = execute or _common.task_execute
        self.meter = meter if meter is not None else BudgetMeter(
            self.root / "budget_meter.json", cap_executions=0)
        self.thresholds = load_thresholds(thresholds)
        self.sides = _split_sides(split)
        self.credit_from_trace = credit_from_trace
        self.v1_id = str(v1_id)
        self.worker_cfg = self.config.worker_cfg_sha()
        self.log = RaceLog(self.root / "executions.jsonl")
        self.log.worker_cfg = self.worker_cfg
        self._plans: dict[str, dict[str, Any]] = {}
        self._programs: dict[str, RaceProgram] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._claimed_traces: set[str] = set()
        self.warnings: list[str] = []
        self._plans_path = self.root / "plans.jsonl"
        self._load_plans()
        self._load_programs()
        other = sorted({str(r.get("worker_cfg")) for r in self.log.records
                        if r.get("worker_cfg") != self.worker_cfg})
        if other:
            self.warnings.append(
                f"executions.jsonl holds rows of other worker configs {other} (current "
                f"{self.worker_cfg}); they are never served from the exec cache")

    # ------------------------------------------------------------------ #
    # Programs
    # ------------------------------------------------------------------ #
    def _programs_index(self) -> Path:
        return self.root / "race_programs.json"

    def _load_programs(self) -> None:
        index = _common._read_json(self._programs_index(), {}) or {}
        for pid, rec in dict(index).items():
            path = self.root / "programs" / f"{pid}.py"
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if _sha256(source) != rec.get("source_sha256"):
                raise RaceError(f"program {pid}: source file changed on disk")
            self._programs[pid] = RaceProgram(pid, source, rec.get("fp"), rec["source_sha256"],
                                              rec.get("arm"), dict(rec.get("meta") or {}))

    def _save_programs(self) -> None:
        _common._write_json(self._programs_index(),
                            {pid: p.record() for pid, p in sorted(self._programs.items())})

    def register_program(self, program_id: str, source: str, *, fp: str | None | bool = True,
                         arm: str | None = None, meta: Mapping[str, Any] | None = None) -> RaceProgram:
        """Register a program and store its source as ``programs/<id>.py``
        (idempotent for the same source; a different source under a known
        id raises).  ``fp=True`` computes the behaviour
        fingerprint (:func:`queenbee.evo.common.behavior_fingerprint`:
        ``screen.behavior_fp`` at the active task's team sizes); pass a
        string to reuse one, ``False`` or None for no cache."""

        pid = str(program_id)
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,120}", pid):
            raise RaceError(f"bad program id {pid!r}")
        sha = _sha256(source)
        with self._lock:
            old = self._programs.get(pid)
            if old is not None:
                if old.source_sha256 != sha:
                    raise RaceError(f"program {pid} already registered with another source")
                return old
        if fp is True:
            fp_value = _common.behavior_fingerprint(source)
        elif fp is False:
            fp_value = None
        else:
            fp_value = fp
        program = RaceProgram(pid, source, fp_value, sha, arm, dict(meta or {}))
        path = self.root / "programs" / f"{pid}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(source, encoding="utf-8")
            os.replace(tmp, path)
        with self._lock:
            self._programs[pid] = program
            self._save_programs()
        return program

    def program(self, program_id: str) -> RaceProgram:
        try:
            return self._programs[str(program_id)]
        except KeyError as exc:
            raise RaceError(f"unknown program {program_id!r} (register_program first)") from exc

    @property
    def programs(self) -> dict[str, RaceProgram]:
        return dict(self._programs)

    # ------------------------------------------------------------------ #
    # Units
    # ------------------------------------------------------------------ #
    def instance(self, unit_id: str) -> tuple[Any, str]:
        return self.resolver.get(unit_id)

    def instance_sha(self, unit_id: str) -> str:
        return self.instance(unit_id)[1]

    def _guard_job(self, job: Job) -> None:
        assert_dev_unit(job.unit_id, where=f"race job {job.key}")
        if job.purpose not in PURPOSES:
            raise RaceError(f"job {job.key}: purpose {job.purpose!r} not in {PURPOSES}")
        if self.sides is None:
            return
        t_side, v_side = self.sides
        template = template_of(job.unit_id)
        if job.purpose in T_PURPOSES and template not in t_side:
            raise LeakageGuardError(
                f"LEAKAGE_GUARD: {job.purpose} job on {job.unit_id} (not a T template)")
        if job.purpose in V_PURPOSES and template not in v_side:
            raise LeakageGuardError(
                f"LEAKAGE_GUARD: {job.purpose} job on {job.unit_id} (not a V template)")
        if job.purpose == "census" and template not in (t_side | v_side):
            raise LeakageGuardError(f"LEAKAGE_GUARD: census job on {job.unit_id}")

    # ------------------------------------------------------------------ #
    # External rows (census): exec cache + dS baselines, never spend
    # ------------------------------------------------------------------ #
    def check_worker_config(self, config: Mapping[str, Any] | None, *, where: str) -> None:
        """Raise :class:`RaceError` when a result file's / row's worker
        configuration (``model`` / ``worker_model``, ``worker_contract``,
        ``goal``, ``python_max_rounds`` or ``python_budgets.max_rounds``,
        also under ``fingerprint``) disagrees with the racer's; absent keys
        are not checked."""

        if not isinstance(config, Mapping):
            return
        want = self.config.worker_fingerprint()
        flat = dict(config)
        if isinstance(config.get("fingerprint"), Mapping):
            flat = {**dict(config["fingerprint"]), **{k: v for k, v in config.items()
                                                      if k != "fingerprint"}}
        budgets = flat.get("python_budgets")
        rounds = flat.get("python_max_rounds")
        if rounds is None and isinstance(budgets, Mapping):
            rounds = budgets.get("max_rounds")
        have = {
            "worker_model": flat.get("worker_model") or flat.get("model"),
            "worker_contract": flat.get("worker_contract"),
            "goal": flat.get("goal"),
            "python_max_rounds": int(rounds) if isinstance(rounds, (int, float))
            and not isinstance(rounds, bool) else None,
        }
        bad = {k: (v, want[k]) for k, v in have.items() if v is not None and v != want[k]}
        if bad:
            raise RaceError(f"{where}: worker config mismatch {bad} (have, racer)")

    def register_external_rows(self, program_id: str, unit_id: str,
                               rows: Sequence[Mapping[str, Any]], *,
                               label: str = "census",
                               config: Mapping[str, Any] | None = None) -> int:
        """Record already-paid rows of a registered program (census),
        rep i = the i-th scored row unless the row carries ``rep``.
        Idempotent per (program, unit, rep).  Returns rows added.

        ``config``: the result file's configuration (e.g. the ``"config"``
        of a :mod:`queenbee.evaluate` result file): a worker model /
        contract / goal / max-rounds mismatch raises
        (:meth:`check_worker_config`); so does a row that names another
        model.  The rows are stamped with the racer's worker config sha
        (they feed the exec cache)."""

        assert_dev_unit(unit_id, where="external rows")
        program = self.program(program_id)
        self.check_worker_config(config, where=f"external rows {program_id}|{unit_id}")
        sha = self.instance_sha(unit_id)
        added = 0
        index = 0
        for row in rows:
            if not isinstance(row, Mapping) or is_infra(row):
                continue
            cfg_keys = {k: row[k] for k in ("model", "worker_model", "worker_contract",
                                            "python_max_rounds") if row.get(k) is not None}
            if "python_max_rounds" not in cfg_keys and row.get("max_rounds") is not None:
                cfg_keys["python_max_rounds"] = row["max_rounds"]
            self.check_worker_config(cfg_keys, where=f"external row {program_id}|{unit_id}")
            rep = row.get("rep")
            rep = int(rep) if isinstance(rep, int) and not isinstance(rep, bool) else index
            index += 1
            key = f"{program.program_id}|{unit_id}|r{rep}"
            if self.log.for_key(key):
                continue
            if row.get("instance_sha256") and row["instance_sha256"] != sha:
                raise RaceError(f"external row {key}: instance sha mismatch")
            self.log.append({
                "key": key, "stage": label, "batch_id": label, "gen": None,
                "program_id": program.program_id, "arm": program.arm, "unit_id": unit_id,
                "template": template_of(unit_id), "rung": rung_of(unit_id), "rep": rep,
                "purpose": "census", "attempt": 0, "cached": False, "cached_from": None,
                "external": True, "fp": program.fp, "source_sha256": program.source_sha256,
                "instance_sha256": sha, "worker_cfg": self.worker_cfg,
                "exec_eq": exec_eq(unit_id), "at": _now(),
                "wall_s": None, "trace": None, "row": dict(row),
            })
            added += 1
        return added

    # ------------------------------------------------------------------ #
    # Plans (persisted rep allocation)
    # ------------------------------------------------------------------ #
    def _load_plans(self) -> None:
        if not self._plans_path.exists():
            return
        for line in self._plans_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # torn last line (kill -9)
            self._plans[str(rec["batch_id"])] = rec

    def plan(self, batch_id: str) -> list[Job] | None:
        rec = self._plans.get(str(batch_id))
        return None if rec is None else [Job.from_dict(j) for j in rec["jobs"]]

    def _max_rep(self, pred: Callable[[Mapping[str, Any]], bool]) -> int:
        reps = [int(r.get("rep", -1)) for r in self.log.select(pred)]
        for rec in self._plans.values():
            for job in rec["jobs"]:
                if pred({**job, "planned": True}):
                    reps.append(int(job["rep"]))
        return max(reps) if reps else -1

    def _next_rep(self, program_id: str, unit_id: str, purpose: str,
                  local: dict[tuple, int]) -> int:
        program = self.program(program_id)
        if purpose in FRESH_PURPOSES and program.fp:
            same_fp = {pid for pid, p in self._programs.items() if p.fp == program.fp}
            key = ("fp", program.fp, unit_id)

            def pred(r: Mapping[str, Any]) -> bool:
                if r.get("unit_id") != unit_id:
                    return False
                if r.get("planned"):
                    return r.get("program_id") in same_fp
                return r.get("fp") == program.fp or r.get("program_id") in same_fp
        else:
            key = ("pid", program_id, unit_id)

            def pred(r: Mapping[str, Any]) -> bool:
                return r.get("unit_id") == unit_id and r.get("program_id") == program_id

        base = max(self._max_rep(pred), local.get(key, -1))
        if purpose in FRESH_PURPOSES and program.fp:
            # a fresh draw must also clear every rep index the program itself used
            base = max(base, self._max_rep(
                lambda r: r.get("unit_id") == unit_id and r.get("program_id") == program_id))
        rep = base + 1
        local[key] = rep
        if key[0] == "fp":
            local[("pid", program_id, unit_id)] = max(local.get(("pid", program_id, unit_id), -1), rep)
        return rep

    def plan_batch(self, batch_id: str, requests: Sequence[Any], *,
                   gen: int | None = None) -> list[Job]:
        """Jobs of a batch with reps allocated and PERSISTED (idempotent: a
        known batch id returns its stored jobs).  ``requests``: Jobs, or
        ``(program_id, unit_id, purpose[, rep])`` tuples / mappings with
        ``rep`` None for automatic allocation (module docstring)."""

        batch_id = str(batch_id)
        with self._lock:
            stored = self._plans.get(batch_id)
        if stored is not None:
            jobs = [Job.from_dict(j) for j in stored["jobs"]]
            want = [(str(_get_req(r, 0, "program_id")), str(_get_req(r, 1, "unit_id")),
                     str(_get_req(r, 2, "purpose"))) for r in requests]
            have = [(j.program_id, j.unit_id, j.purpose) for j in jobs]
            if want and set(want) - set(have):
                self.warnings.append(
                    f"plan {batch_id}: re-plan asked for {sorted(set(want) - set(have))} "
                    "not in the stored plan (stored plan kept)")
            return jobs
        local: dict[tuple, int] = {}
        jobs: list[Job] = []
        seen: set[str] = set()
        for req in requests:
            if isinstance(req, Job):
                job = Job(req.program_id, req.unit_id, req.rep, req.purpose, batch_id)
            else:
                pid = str(_get_req(req, 0, "program_id"))
                unit = str(_get_req(req, 1, "unit_id"))
                purpose = str(_get_req(req, 2, "purpose"))
                rep = _get_req(req, 3, "rep")
                self._guard_job(Job(pid, unit, 0, purpose, batch_id))
                self.program(pid)
                if rep is None:
                    rep = self._next_rep(pid, unit, purpose, local)
                job = Job(pid, unit, int(rep), purpose, batch_id)
            self._guard_job(job)
            if job.key in seen:
                continue
            seen.add(job.key)
            jobs.append(job)
        rec = {"batch_id": batch_id, "gen": gen if gen is not None else gen_of_batch(batch_id),
               "at": _now(), "jobs": [j.to_dict() for j in jobs]}
        with self._lock:
            self._plans_path.parent.mkdir(parents=True, exist_ok=True)
            with self._plans_path.open("a") as handle:
                handle.write(json.dumps(rec) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._plans[batch_id] = rec
        return jobs

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def exec_eq_spent(self) -> float:
        """Scored execution-equivalents paid through this racer (o10 = 2;
        infra, cache hits and external rows excluded)."""

        return self.log.exec_eq_scored()

    def run_batch(self, batch_id: str, requests: Sequence[Any], *, gen: int | None = None,
                  cap_exec_eq: float | None = None) -> list[dict[str, Any]]:
        jobs = self.plan_batch(batch_id, requests, gen=gen)
        return self.run(jobs, cap_exec_eq=cap_exec_eq)

    def run(self, jobs: Sequence[Any], *, stage: str | None = None,
            cap_exec_eq: float | None = None) -> list[dict[str, Any]]:
        """``Racer.run(jobs) -> [ExecRow]`` (one per job, in order).

        Resumable (scored keys are returned, not re-run), exec cache on
        (fp, instance sha, rep) including inside the batch, infra re-runs,
        pre-flight cap on scored execution-equivalents (``BudgetExceeded``,
        nothing ran)."""

        jobs = [coerce_job(j) for j in jobs]
        for job in jobs:
            self._guard_job(job)
            self.program(job.program_id)
        with self._run_lock:
            results: dict[str, dict[str, Any]] = {}
            todo: list[Job] = []
            unique: dict[str, Job] = {}
            for job in jobs:
                unique.setdefault(job.key, job)
            for job in unique.values():
                program = self.program(job.program_id)
                sha = self.instance_sha(job.unit_id)
                hit = self.log.scored(job.key)
                if hit is not None:
                    results[job.key] = hit
                    continue
                cached = None
                if job.purpose not in FRESH_PURPOSES:
                    cached = self.log.cache_lookup(program.fp, job.unit_id, sha, job.rep)
                if cached is not None:
                    record = self._record(job, program, sha, cached["row"], attempt=0,
                                          cached_from=cached["key"], trace=cached.get("trace"))
                    self.log.append(record)
                    self.meter.charge_rows(PURPOSE_PHASE[job.purpose], [record["row"]], cached=True)
                    results[job.key] = record
                    continue
                if self._attempts(job.key) >= 1 + int(self.config.max_infra_reruns):
                    last = (self.log.for_key(job.key) or [{}])[-1]
                    results[job.key] = dict(last) | {"infra_final": True}
                    continue
                todo.append(job)
            leaders: dict[tuple, Job] = {}
            followers: list[tuple[Job, Job]] = []
            run_now: list[Job] = []
            for job in todo:
                program = self.program(job.program_id)
                ckey = (program.fp, job.unit_id, self.instance_sha(job.unit_id), int(job.rep))
                if program.fp and job.purpose not in FRESH_PURPOSES and ckey in leaders:
                    followers.append((job, leaders[ckey]))
                    continue
                if program.fp:
                    leaders.setdefault(ckey, job)
                run_now.append(job)
            if run_now and cap_exec_eq is not None:
                projected = self.exec_eq_spent() + sum(exec_eq(j.unit_id) for j in run_now)
                if projected > float(cap_exec_eq) + EPS:
                    raise BudgetExceeded(
                        f"{len(run_now)} job(s) would bring scored execution-equivalents "
                        f"to {projected:g} > cap {cap_exec_eq:g}; nothing ran")
            try:
                if run_now:
                    self._execute_batch(run_now, stage or _batch_label(run_now), results)
            finally:
                for job, leader in followers:
                    lead = self.log.scored(leader.key)
                    if lead is None:
                        results.setdefault(job.key, (self.log.for_key(leader.key) or [{}])[-1]
                                           | {"infra_final": True, "follower_of": leader.key})
                        continue
                    program = self.program(job.program_id)
                    record = self._record(job, program, self.instance_sha(job.unit_id),
                                          lead["row"], attempt=0, cached_from=lead["key"],
                                          trace=lead.get("trace"))
                    self.log.append(record)
                    self.meter.charge_rows(PURPOSE_PHASE[job.purpose], [record["row"]], cached=True)
                    results[job.key] = record
            return [self.exec_row(results.get(job.key) or {}, job) for job in jobs]

    def _attempts(self, key: str) -> int:
        return sum(1 for r in self.log.for_key(key) if not r.get("cached") and not r.get("external"))

    def _execute_batch(self, todo: Sequence[Job], label: str, results: dict[str, dict]) -> None:
        cfg = self.config
        safe = _BATCH_SAFE.sub("_", label)[:80] or "batch"
        saved = os.environ.get("QB_TRACE_DIR")
        # QB_TRACE_DIR is read only by the execution path (never by a planner
        # / mint dry run), and batches are serialised by ``_run_lock``.
        trace_dir = self.root / "traces" / safe
        if cfg.traces:
            trace_dir.mkdir(parents=True, exist_ok=True)
            os.environ["QB_TRACE_DIR"] = str(trace_dir)
        else:
            os.environ.pop("QB_TRACE_DIR", None)
        try:
            workers = max(1, min(int(cfg.parallel_cases), len(todo)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._run_one, job, trace_dir): job for job in todo}
                for future, job in futures.items():
                    results[job.key] = future.result()
        finally:
            if saved is None:
                os.environ.pop("QB_TRACE_DIR", None)
            else:
                os.environ["QB_TRACE_DIR"] = saved

    def _record(self, job: Job, program: RaceProgram, instance_sha: str,
                row: Mapping[str, Any], *, attempt: int, cached_from: str | None = None,
                wall_s: float | None = None, trace: str | None = None) -> dict[str, Any]:
        return {
            "key": job.key,
            "stage": job.batch_id or job.purpose,
            "batch_id": job.batch_id,
            "gen": gen_of_batch(job.batch_id),
            "program_id": job.program_id,
            "arm": program.arm,
            "unit_id": job.unit_id,
            "template": template_of(job.unit_id),
            "rung": rung_of(job.unit_id),
            "rep": int(job.rep),
            "purpose": job.purpose,
            "attempt": int(attempt),
            "cached": cached_from is not None,
            "cached_from": cached_from,
            "external": False,
            "fp": program.fp,
            "source_sha256": program.source_sha256,
            "instance_sha256": instance_sha,
            "worker_cfg": self.worker_cfg,
            "exec_eq": exec_eq(job.unit_id),
            "at": _now(),
            "wall_s": wall_s,
            "trace": trace,
            "row": dict(row),
        }

    def _run_one(self, job: Job, trace_dir: Path) -> dict[str, Any]:
        cfg = self.config
        program = self.program(job.program_id)
        instance, sha = self.instance(job.unit_id)
        record: dict[str, Any] = {}
        while self._attempts(job.key) < 1 + int(cfg.max_infra_reruns):
            attempt = self._attempts(job.key)
            if record and cfg.infra_backoff_s > 0:
                time.sleep(float(cfg.infra_backoff_s) * attempt)
            started = time.time()
            req = {
                "cfg": cfg, "job": job, "source": program.source, "instance": instance,
                "unit_id": job.unit_id, "seed": int(_sha256(job.key)[:6], 16),
                "artifacts_dir": self.root / "artifacts" / _BATCH_SAFE.sub("_", job.batch_id or "adhoc")
                / job.program_id / f"{job.unit_id.replace('@', '_')}_r{job.rep}_a{attempt}",
            }
            try:
                row = dict(self.execute(req))
            except (LeakageGuardError, RaceError):
                raise
            except Exception as exc:  # noqa: BLE001 - host-side: never the program's S
                # the default executor (``common.task_execute`` ->
                # ``real_execute`` -> ``evaluate_python_source_on_cases``)
                # turns every per-case failure into a row itself, so an
                # exception that escapes ``execute`` is host-side (setup,
                # budgets, I/O): an infra row, re-run, never scored 0.
                row = _failure_row(exc)
            row.setdefault("unit_id", job.unit_id)
            trace = None
            if not is_infra(row):
                trace = self._enrich(row, instance, program, started, trace_dir)
            record = self._record(job, program, sha, row, attempt=attempt,
                                  wall_s=round(time.time() - started, 2), trace=trace)
            self.log.append(record)
            self.meter.charge_rows(PURPOSE_PHASE[job.purpose], [row])
            if not is_infra(row) and exec_eq(job.unit_id) > 1:
                self.meter.add_counter("exec_eq_surcharge", exec_eq(job.unit_id) - 1)
            if not is_infra(row):
                break
        if record and is_infra(record.get("row") or {}):
            # every attempt was infra: the job is exhausted (never scored 0)
            record = dict(record) | {"infra_final": True}
        return record

    def _enrich(self, row: dict[str, Any], instance: Any, program: RaceProgram,
                started: float, trace_dir: Path) -> str | None:
        """Copy the diagnosis card and the credit fields of this execution's
        trace into ``row`` (:meth:`exec_row` sanitizes both); returns the
        trace path (relative to the root when possible) or None."""

        trace_rel = None
        if self.config.traces and "credit" not in row:
            path = self._find_trace(trace_dir, instance, program, started, row)
            if path is not None:
                try:
                    trace_rel = str(path.relative_to(self.root))
                except ValueError:
                    trace_rel = str(path)
                try:
                    trace = _common._read_json(path) or {}
                    enrich = self.credit_from_trace
                    if enrich is None:
                        from queenbee.evo.credit import trace_row as enrich_fn

                        traced = enrich_fn(trace, instance, source=program.source,
                                           goal=_common.task_goal())
                    else:
                        traced = enrich(trace, instance, program.source)
                    traced = dict(traced or {})
                    if isinstance(traced.get("diag"), Mapping):  # current classifier wins
                        row["diag"] = dict(traced["diag"])
                    if traced.get("credit") is not None:
                        row["credit"] = traced["credit"]
                except LeakageGuardError:
                    raise
                except Exception as exc:  # noqa: BLE001 - enrichment is advisory
                    row["enrich_error"] = type(exc).__name__
        return trace_rel

    def _find_trace(self, trace_dir: Path, instance: Any, program: RaceProgram,
                    started: float, row: Mapping[str, Any]) -> Path | None:
        pattern = f"{getattr(instance, 'case_id', '')}_{program.source_sha256[:12]}_*.json"
        truth = json.dumps(getattr(instance, "ground_truth", None), sort_keys=True, default=str)
        best: tuple[float, Path] | None = None
        for path in trace_dir.glob(pattern):
            if str(path) in self._claimed_traces:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime < started - 1.0:
                continue
            data = _common._read_json(path) or {}
            if json.dumps(data.get("ground_truth"), sort_keys=True, default=str) != truth:
                continue
            facts = data.get("facts") or {}
            if not _common._same_num(facts.get("S"), row.get("S")):
                continue
            if row.get("C") is not None and facts.get("C") is not None \
                    and not _common._same_num(facts.get("C"), row.get("C")):
                continue
            if best is None or mtime > best[0]:
                best = (mtime, path)
        if best is None:
            return None
        with self._lock:
            self._claimed_traces.add(str(best[1]))
        return best[1]

    def exec_row(self, record: Mapping[str, Any], job: Job | None = None) -> dict[str, Any]:
        """The ExecRow of a log record: the executor's row + {unit_id,
        program_id, fp, rep, batch_id, purpose, gen, key, cached,
        infra_final, exec_eq, diag (``sanitize_diag``), credit
        (``sanitize_credit``)}.  An infra row keeps ``S = None`` (never
        scored 0)."""

        from queenbee.evo.diagnosis import ensure_diag, sanitize_diag
        from queenbee.evo.credit import sanitize_credit

        row = dict(record.get("row") or {})
        job = job or Job(str(record.get("program_id")), str(record.get("unit_id")),
                         int(record.get("rep") or 0), str(record.get("purpose") or ""),
                         str(record.get("batch_id") or ""))
        out = {k: v for k, v in row.items() if k not in ("diag", "credit")}
        infra = is_infra(row) or bool(record.get("infra_final")) or not record
        n = unit_n_agents(job.unit_id)
        out.update({
            "unit_id": job.unit_id,
            "program_id": job.program_id,
            "fp": record.get("fp"),
            "rep": int(job.rep),
            "batch_id": job.batch_id,
            "purpose": job.purpose,
            "gen": gen_of_batch(job.batch_id),
            "key": job.key,
            "cached": bool(record.get("cached")),
            "cached_from": record.get("cached_from"),
            "external": bool(record.get("external")),
            "infra_final": bool(record.get("infra_final")) or not record,
            "exec_eq": exec_eq(job.unit_id),
            "instance_sha256": record.get("instance_sha256"),
            "source_sha256": record.get("source_sha256"),
            "trace": record.get("trace"),
        })
        if infra:
            out["S"] = None
            out["infra"] = row.get("infra") or ("missing" if not record else "infra_final")
            out["diag"] = None
            out["credit"] = None
        else:
            out["diag"] = sanitize_diag(row.get("diag")) or ensure_diag(row)
            out["credit"] = sanitize_credit(row.get("credit"), n_agents=n) \
                if row.get("credit") is not None else None
        return out

    # ------------------------------------------------------------------ #
    # Row views
    # ------------------------------------------------------------------ #
    def scored_rows(self, program_id: str, unit_id: str, *, gen: int | None = None,
                    purposes: Iterable[str] | None = None,
                    exclude_gen: int | None = None) -> list[dict[str, Any]]:
        """Scored records (one per rep, latest wins) of a program on a unit."""

        wanted = set(purposes) if purposes is not None else None
        by_rep: dict[int, dict[str, Any]] = {}
        for rec in self.log.select(lambda r: r.get("program_id") == program_id
                                   and r.get("unit_id") == unit_id):
            if is_infra(rec.get("row") or {}):
                continue
            if gen is not None and rec.get("gen") != gen:
                continue
            if exclude_gen is not None and rec.get("gen") == exclude_gen:
                continue
            if wanted is not None and rec.get("purpose") not in wanted:
                continue
            by_rep[int(rec.get("rep", 0))] = rec
        return [by_rep[k] for k in sorted(by_rep)]

    def has_fresh(self, parent_id: str, unit_id: str, gen: int, *,
                  exclude_batch: str | None = None) -> bool:
        return any(
            rec.get("batch_id") != exclude_batch
            for rec in self.scored_rows(parent_id, unit_id, gen=gen, purposes=("parent_fresh",))
        ) or any(
            j.program_id == parent_id and j.unit_id == unit_id and j.purpose == "parent_fresh"
            for bid, rec in self._plans.items()
            if rec.get("gen") == gen and bid != exclude_batch
            for j in (Job.from_dict(x) for x in rec["jobs"])
        )

    def _baseline_rows(self, parent_id: str, unit_id: str,
                       gen: int) -> tuple[list[dict[str, Any]], str]:
        # THIS generation's fresh reps only: a fresh rep of another
        # generation entered the archive and helped choose this generation's
        # targets and parents, so pooling it would bring back the selection
        # bias the fresh rep exists to remove.
        fresh = [r["row"] for r in
                 self.scored_rows(parent_id, unit_id, gen=gen, purposes=("parent_fresh",))
                 if row_S(r["row"]) is not None]
        if fresh:
            return fresh, "fresh"
        archived = [r["row"] for r in self.scored_rows(parent_id, unit_id)
                    if row_S(r["row"]) is not None]
        if archived:
            return archived, "archived"
        return [], "missing"

    def baseline(self, parent_id: str, unit_id: str, gen: int) -> tuple[float | None, str]:
        """The parent's S on a unit for paired dS: its FRESH reps of this
        generation, else its archived mean over every scored row.  This is
        the overridable rule: :meth:`outcome` pairs the R1 unit with it only
        when a fresh parent rep exists (:meth:`fresh_parent_S`) and falls
        back to the seed program (v1) on the R2 units (:meth:`r2_baseline`)."""

        rows, src = self._baseline_rows(parent_id, unit_id, gen)
        if not rows:
            return None, src
        return _mean(row_S(r) for r in rows), src

    def baseline_C(self, parent_id: str, unit_id: str, gen: int) -> float | None:
        """The parent's token cost C on a unit, same row choice as
        :meth:`baseline` (fresh reps, else archived), for the token-cost
        difference dC (economize proposals)."""

        rows, _src = self._baseline_rows(parent_id, unit_id, gen)
        c = [float(r["C"]) for r in rows if r.get("C") is not None]
        return _mean(c) if c else None

    def fresh_parent_S(self, parent_id: str, unit_id: str, gen: int) -> list[float]:
        """The parent's scored FRESH reps on a unit in generation ``gen``."""

        vals = [row_S(r["row"]) for r in
                self.scored_rows(parent_id, unit_id, gen=gen, purposes=("parent_fresh",))]
        return [v for v in vals if v is not None]

    def r2_baseline(self, parent_id: str, unit_id: str, gen: int) -> tuple[float | None, str]:
        """dS baseline on a non-R1 unit: :meth:`baseline` (the parent's fresh
        / archived rows, or a subclass's rule), else the seed program's (v1)
        baseline, usually its census rows -- an evolved parent only has rows
        on its own few units and R2 units are chosen by hash, so without this
        fallback most R2 units would have no dS and ``D`` (which needs a dS on
        every R2 unit) would rarely exist."""

        base, src = self.baseline(parent_id, unit_id, gen)
        if base is None and str(parent_id) != self.v1_id:
            ref, ref_src = self.baseline(self.v1_id, unit_id, gen)
            if ref is not None:
                return ref, f"v1_{ref_src}"
        return base, src

    def r2_baseline_C(self, parent_id: str, unit_id: str, gen: int) -> float | None:
        c = self.baseline_C(parent_id, unit_id, gen)
        if c is None and str(parent_id) != self.v1_id:
            c = self.baseline_C(self.v1_id, unit_id, gen)
        return c

    def guard_baseline(self, parent_id: str, unit_id: str,
                       v1_id: str = "v1") -> tuple[float | None, str]:
        for pid, label in ((parent_id, "parent"), (v1_id, "v1")):
            vals = [row_S(r["row"]) for r in self.scored_rows(pid, unit_id)]
            vals = [s for s in vals if s is not None]
            if vals:
                return _mean(vals), label
        return 1.0, "assumed_saturated"

    # ------------------------------------------------------------------ #
    # Race stages
    # ------------------------------------------------------------------ #
    def race_r1(self, gen: int, candidates: Sequence[Any], *, tag: str = "r1",
                cap_exec_eq: float | None = None) -> dict[str, Any]:
        """R1: each candidate x its first target unit x 1 rep, plus a fresh
        parent rep on every (parent, R1 unit) lacking one this generation --
        ONE batch.  Returns ``{"batch_id", "candidates": [...], "survivors",
        "r2_selected", "rows"}``."""

        th = self.thresholds
        cands = [normalize_candidate(c) for c in candidates]
        batch_id = batch_id_for(gen, tag)
        requests: list[tuple] = []
        fresh: list[tuple[str, str]] = []
        for c in cands:
            unit = c["target_units"][0]
            requests.append((c["program_id"], unit, "r1", None))
            pair = (c["parent_id"], unit)
            if pair not in fresh and not self.has_fresh(*pair, gen, exclude_batch=batch_id):
                fresh.append(pair)
        requests += [(p, u, "parent_fresh", None) for p, u in fresh]
        rows = self.run_batch(batch_id, requests, gen=gen, cap_exec_eq=cap_exec_eq)
        jobs = {(j.program_id, j.unit_id, j.purpose): j for j in self.plan(batch_id) or []}
        by_key = {r["key"]: r for r in rows}
        out: list[dict[str, Any]] = []
        for c in cands:
            unit = c["target_units"][0]
            job = jobs.get((c["program_id"], unit, "r1"))
            row = by_key.get(job.key) if job else None
            if row is None and job is not None:
                row = self.exec_row(self.log.scored(job.key) or {}, job)
            s = row_S(row)
            base, src = self.baseline(c["parent_id"], unit, gen)
            if src != "fresh":
                base = None  # never the archived mean for R1 (regression to the mean)
            fmt = is_format_row(row)
            d1 = None if s is None or base is None else round(s - base, 6)
            if s is None:
                status = "infra_final"
            elif base is None:
                status = "no_fresh_parent"
            elif fmt:
                status = "format_fail"
            elif d1 < th.tau_r1 - EPS:
                status = "below_tau_r1"
            else:
                status = "survived"
            out.append({
                **c, "r1_unit": unit, "S": s, "fresh_S": base, "d1": d1,
                "C": (row or {}).get("C"), "format_fail": fmt,
                "survive": status == "survived", "status": status,
                "key": job.key if job else None,
            })
        return {
            "gen": int(gen), "batch_id": batch_id, "thresholds": th.to_dict(),
            "candidates": out,
            "survivors": [c["program_id"] for c in out if c["survive"]],
            "r2_selected": select_for_r2(out, th),
            "rows": rows,
        }

    def race_r2(self, gen: int, r1: Mapping[str, Any] | Sequence[Any], *,
                t_frontier: Sequence[str], guards: Sequence[str] = (),
                shapes: Mapping[str, Any] | None = None, tag: str = "r2",
                selected: Sequence[str] | None = None,
                cap_exec_eq: float | None = None) -> dict[str, Any]:
        """R2 on the R1 selection (``r1["r2_selected"]`` unless ``selected``):
        ``n_r2_units`` other T frontier units + 1 rotating guard per
        candidate, one batch (a resumed batch keeps its stored units).
        Returns ``{"batch_id", "plan": {pid: {"units", "guard"}}, "outcomes":
        {pid: outcome}, "rows"}``."""

        cfg = self.config
        if isinstance(r1, Mapping):
            cands = {c["program_id"]: c for c in r1.get("candidates") or []}
            chosen = list(selected if selected is not None else r1.get("r2_selected") or [])
        else:
            norm = [normalize_candidate(c) for c in r1]
            cands = {c["program_id"]: c for c in norm}
            chosen = list(selected if selected is not None else cands)
        batch_id = batch_id_for(gen, tag)
        requests: list[tuple] = []
        plan: dict[str, dict[str, Any]] = {}
        stored = self.plan(batch_id)
        for slot, pid in enumerate(chosen):
            c = cands[pid]
            target = c["target_units"][0]
            if stored is not None:
                units = [j.unit_id for j in stored if j.program_id == pid and j.purpose == "r2"]
                gl = [j.unit_id for j in stored if j.program_id == pid and j.purpose == "guard"]
                guard = gl[0] if gl else None
            else:
                units = pick_r2_units(pid, target, t_frontier, gen=gen, shapes=shapes,
                                      k=cfg.n_r2_units, other_tier=cfg.r2_other_tier)
                used = {template_of(u) for u in [target, *units]}
                guard = pick_guard(guards, gen=gen, slot=slot, exclude_templates=used)
            plan[pid] = {"target": target, "units": units, "guard": guard}
            requests += [(pid, u, "r2", None) for u in units]
            if guard:
                requests.append((pid, guard, "guard", None))
        rows = self.run_batch(batch_id, requests, gen=gen, cap_exec_eq=cap_exec_eq) if requests else []
        outcomes = {pid: self.outcome(cands[pid], gen, r2_plan=plan[pid]) for pid in chosen}
        return {"gen": int(gen), "batch_id": batch_id, "plan": plan,
                "outcomes": outcomes, "rows": rows}

    def race_confirm(self, gen: int, winners: Sequence[Any], *, tag: str = "confirm",
                     cap_exec_eq: float | None = None) -> dict[str, Any]:
        """1 fresh rep of each new front winner on the unit it wins
        (``winners``: ``{"program_id", "unit_id"}`` records or pairs)."""

        pairs: list[tuple[str, str]] = []
        for w in winners:
            if isinstance(w, (tuple, list)):
                pid, unit = str(w[0]), str(w[1])
            else:
                pid, unit = str(_get(w, "program_id")), str(_get(w, "unit_id"))
            if (pid, unit) not in pairs:
                pairs.append((pid, unit))
        batch_id = batch_id_for(gen, tag)
        if not pairs:
            return {"gen": int(gen), "batch_id": batch_id, "confirm": {}, "rows": []}
        rows = self.run_batch(batch_id, [(p, u, "confirm", None) for p, u in pairs],
                              gen=gen, cap_exec_eq=cap_exec_eq)
        by_pair = {(r["program_id"], r["unit_id"]): r for r in rows}
        confirm = {}
        for pid, unit in pairs:
            row = by_pair.get((pid, unit))
            all_s = [row_S(r["row"]) for r in self.scored_rows(pid, unit)]
            all_s = [s for s in all_s if s is not None]
            confirm[f"{pid}|{unit}"] = {"program_id": pid, "unit_id": unit,
                                        "S_confirm": row_S(row), "S_mean": _r(_mean(all_s)),
                                        "n_reps": len(all_s)}
        return {"gen": int(gen), "batch_id": batch_id, "confirm": confirm, "rows": rows}

    def race_generation(self, gen: int, candidates: Sequence[Any], *,
                        t_frontier: Sequence[str], guards: Sequence[str] = (),
                        shapes: Mapping[str, Any] | None = None,
                        winners_fn: Callable[[list[dict[str, Any]]], Sequence[Any]] | None = None,
                        cap_exec_eq: float | None = None) -> dict[str, Any]:
        """R1 -> R2 -> confirmation in sequence (for a caller that does not
        overlap minting with racing).  ``winners_fn(rows_so_far)`` returns the
        new front winners (e.g.
        ``archive.add_rows(rows); archive.newly_front_winning()``); without
        it no confirmation runs.  Outcomes cover every candidate."""

        r1 = self.race_r1(gen, candidates, cap_exec_eq=cap_exec_eq)
        r2 = self.race_r2(gen, r1, t_frontier=t_frontier, guards=guards, shapes=shapes,
                          cap_exec_eq=cap_exec_eq)
        rows = list(r1["rows"]) + list(r2["rows"])
        conf = {"confirm": {}, "rows": []}
        if winners_fn is not None:
            winners = list(winners_fn(rows) or [])
            if winners:
                conf = self.race_confirm(gen, winners, cap_exec_eq=cap_exec_eq)
                rows += conf["rows"]
        cands = {c["program_id"]: c for c in r1["candidates"]}
        outcomes = {pid: self.outcome(c, gen, r2_plan=r2["plan"].get(pid))
                    for pid, c in cands.items()}
        return {"gen": int(gen), "r1": r1, "r2": r2, "confirm": conf,
                "outcomes": outcomes, "rows": rows}

    # ------------------------------------------------------------------ #
    # Outcome (paired dS) -> ledger input
    # ------------------------------------------------------------------ #
    def outcome(self, candidate: Any, gen: int, *,
                r2_plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Per-candidate record of generation ``gen`` from the log:
        ``observed`` {T unit: dS vs parent} (mean over the candidate's reps
        of this generation, guard excluded; the R1 unit only against this
        generation's FRESH parent rep, other units against
        :meth:`r2_baseline`), ``target_dS`` (the R1 unit: R1 +
        confirmation reps; ``target_dS_all`` = mean over every executed
        brief target unit, reported only), ``target_dC`` (token cost vs the
        fresh parent on the R1 unit), ``d1``, ``reached_r2``, ``D`` (mean dS
        over the COMPLETE R2 unit set -- R1 unit + ``n_r2_units``; None with
        ``D_note`` otherwise), ``guard`` / ``guard_ok`` (None without a scored
        guard row; True with ``vacuous_guard`` when R2 found no guard unit to
        run), ``high_tier``, ``executed``, ``status`` (``no_fresh_parent``
        when the R1 pairing is missing) and the host ``verdict``."""

        th = self.thresholds
        c = candidate if isinstance(candidate, Mapping) and "r1_unit" in candidate \
            else normalize_candidate(candidate)
        pid, parent = c["program_id"], c["parent_id"]
        r1_unit = c["target_units"][0]
        recs = self.log.select(lambda r: r.get("program_id") == pid and r.get("gen") == gen)
        scored: dict[str, dict[int, dict[str, Any]]] = {}
        guard_rows: dict[str, list[float]] = {}
        any_infra_final = False
        purposes: set[str] = set()
        for rec in recs:
            if is_infra(rec.get("row") or {}):
                if self._attempts(rec["key"]) >= 1 + int(self.config.max_infra_reruns) \
                        and self.log.scored(rec["key"]) is None:
                    any_infra_final = True
                continue
            purposes.add(str(rec.get("purpose")))
            if rec.get("purpose") == "guard":
                s = row_S(rec["row"])
                if s is not None:
                    guard_rows.setdefault(rec["unit_id"], []).append(s)
                continue
            scored.setdefault(rec["unit_id"], {})[int(rec.get("rep", 0))] = rec
        observed: dict[str, float] = {}
        per_unit: dict[str, dict[str, Any]] = {}
        fresh_ok = bool(self.fresh_parent_S(parent, r1_unit, gen))
        for unit, reps in scored.items():
            s_vals = [row_S(r["row"]) for r in reps.values()]
            s_vals = [s for s in s_vals if s is not None]
            if not s_vals:
                continue
            if unit == r1_unit:
                # the R1 unit is paired with THIS generation's fresh parent rep:
                # without one, no dS there -- never the archived mean in its
                # place (regression to the mean)
                if fresh_ok:
                    base, src = self.baseline(parent, unit, gen)
                    base_c = self.baseline_C(parent, unit, gen)
                else:
                    base, src, base_c = None, "no_fresh_parent", None
            else:
                base, src = self.r2_baseline(parent, unit, gen)
                base_c = self.r2_baseline_C(parent, unit, gen)
            s_mean = _mean(s_vals)
            c_vals = [float(r["row"]["C"]) for r in reps.values()
                      if (r.get("row") or {}).get("C") is not None]
            c_mean = _mean(c_vals) if c_vals else None
            per_unit[unit] = {"S": _r(s_mean), "n": len(s_vals), "baseline": _r(base),
                              "baseline_src": src, "C": _r(c_mean, 1),
                              "baseline_C": _r(base_c, 1),
                              "dC": _r(c_mean - base_c, 1) if c_mean is not None
                              and base_c is not None and src != "no_fresh_parent" else None}
            if base is not None:
                observed[unit] = round(s_mean - base, 6)
        r1_rows = [r for r in scored.get(r1_unit, {}).values() if r.get("purpose") in ("r1",)]
        d1 = None
        if r1_rows and fresh_ok:
            base, _src = self.baseline(parent, r1_unit, gen)
            s = row_S(r1_rows[0]["row"])
            if s is not None and base is not None:
                d1 = round(s - base, 6)
        # the verdict's target dS is the R1 unit's (R1 + confirmation reps vs
        # the fresh parent), one fixed unit; other target units that R2
        # happened to land on are reported, not scored
        target_dS = _r(observed[r1_unit], 6) if r1_unit in observed else None
        targets = [u for u in c["target_units"] if u in observed]
        target_dS_all = _r(_mean(observed[u] for u in targets), 6) if targets else None
        reached_r2 = "r2" in purposes or "guard" in purposes
        plan = dict(r2_plan or {})
        extra = plan.get("units")
        if extra is None:
            extra = sorted(
                (u for u, reps in scored.items() if u != r1_unit
                 and any(r.get("purpose") == "r2" for r in reps.values())),
                key=unit_sort_key,
            )
        r2_units = [r1_unit] + [u for u in extra if u != r1_unit]
        D = None
        D_note = None
        if reached_r2:
            missing = [u for u in r2_units if u not in observed]
            want = 1 + int(self.config.n_r2_units)
            if missing:
                D_note = f"no dS on {missing}"
            elif len(r2_units) < want:
                D_note = f"{len(r2_units)} of {want} R2 units"
            else:
                D = _r(_mean(observed[u] for u in r2_units), 6)
        guard_unit = plan.get("guard") or (next(iter(guard_rows)) if guard_rows else None)
        guard: dict[str, Any] = {"unit_id": guard_unit}
        guard_ok: bool | None = None
        if guard_unit and guard_rows.get(guard_unit):
            s_g = _mean(guard_rows[guard_unit])
            base_g, src_g = self.guard_baseline(parent, guard_unit, self.v1_id)
            drop = None if base_g is None else round(base_g - s_g, 6)
            guard_ok = th.guard_ok(drop, guard_unit)
            guard.update({"S": _r(s_g), "baseline": _r(base_g), "baseline_src": src_g,
                          "drop": drop, "ok": guard_ok,
                          "allowance": _r(th.guard_allowance(guard_unit))})
        elif reached_r2 and not guard_unit and "guard" in plan:
            if getattr(self.config, "vacuous_guard", False):
                # ``vacuous_guard``: R2 found no guard unit to run (typically an
                # empty guard pool: nothing solved to protect), so the "no
                # regression" condition holds vacuously
                guard_ok = True
                guard["note"] = "no guard unit available (vacuous: nothing solved to protect)"
            else:
                # no guard unit to run: fail closed -- the guard condition
                # cannot be checked, so neither confirmed nor high tier
                guard["note"] = "no guard unit available (not checked)"
        high_tier = bool(reached_r2 and D is not None and D >= th.tau_2 - EPS and guard_ok is True)
        executed = bool(scored) or bool(guard_rows)
        status = "executed" if executed else ("infra_final" if any_infra_final else "not_executed")
        if executed and r1_unit in scored and r1_unit not in observed:
            status = "no_fresh_parent"
        out = {
            "program_id": pid, "parent_id": parent, "gen": int(gen),
            "brief_id": c.get("brief_id"), "kind": c.get("kind"),
            "target_units": list(c["target_units"]), "target_class": c.get("target_class"),
            "r1_unit": r1_unit, "d1": d1, "observed": observed, "per_unit": per_unit,
            "target_units_executed": targets, "target_dS": target_dS,
            "target_dS_all": target_dS_all,
            "target_dC": (per_unit.get(r1_unit) or {}).get("dC"),
            "reached_r2": reached_r2, "r2_units": r2_units if reached_r2 else [],
            "D": D, "D_note": D_note, "guard": guard, "guard_ok": guard_ok,
            "high_tier": high_tier,
            "confirm_units": sorted(u for u, reps in scored.items()
                                    if any(r.get("purpose") == "confirm" for r in reps.values())),
            "executed": executed, "status": status,
        }
        out["verdict"] = host_verdict(out, th)
        return out

    # ------------------------------------------------------------------ #
    def summary(self) -> dict[str, Any]:
        meter = self.meter.snapshot() if hasattr(self.meter, "snapshot") else {}
        log = self.log.summary()
        totals = (meter.get("totals") or {}) if isinstance(meter, Mapping) else {}
        return {
            "log": log,
            "exec_eq_spent": self.exec_eq_spent(),
            "meter_train_executions": totals.get("train_executions"),
            "meter_agrees_with_log": (
                totals.get("train_executions") in (None, log["executions"])
            ),
            "plans": len(self._plans),
            "programs": len(self._programs),
            "thresholds": self.thresholds.to_dict(),
            "warnings": list(self.warnings),
        }


def _get_req(req: Any, index: int, name: str) -> Any:
    if isinstance(req, Mapping):
        return req.get(name)
    if isinstance(req, (tuple, list)):
        return req[index] if len(req) > index else None
    return getattr(req, name, None)


def _batch_label(jobs: Sequence[Job]) -> str:
    ids = sorted({j.batch_id for j in jobs if j.batch_id})
    return ids[0] if len(ids) == 1 else (ids[0] + "+" if ids else "adhoc")


def _failure_row(exc: BaseException) -> dict[str, Any]:
    """An exception raised by the executor itself: an infra row (host
    side), never an ``algorithm_failure`` with S = 0.  A leakage guard /
    race error is re-raised, never turned into a row (also for wrappers
    that call this from their own ``except``)."""

    if isinstance(exc, (LeakageGuardError, RaceError)):
        raise exc
    return {"infra": f"host:{type(exc).__name__}: {exc}"[:500], "execution_class": "infra",
            "S": None}


__all__ = [
    "EPS",
    "FRESH_PURPOSES",
    "Job",
    "PROVISIONAL",
    "PURPOSES",
    "PURPOSE_PHASE",
    "REFUTE_QUANTA",
    "RaceConfig",
    "RaceError",
    "RaceLog",
    "RaceProgram",
    "Racer",
    "Thresholds",
    "VERDICTS",
    "assert_dev_unit",
    "batch_id_for",
    "coerce_job",
    "exec_eq",
    "gen_of_batch",
    "host_verdict",
    "is_format_row",
    "is_infra",
    "load_thresholds",
    "normalize_candidate",
    "pick_guard",
    "pick_r2_units",
    "row_S",
    "select_for_r2",
    "unit_n_agents",
    "unit_sort_key",
]
