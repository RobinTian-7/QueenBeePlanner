"""Evolve the team program of a team of frozen worker LLMs (QueenBee-Evo).

One invocation is one evolution run of one arm on one train / validation
split of the development templates, starting from the seed program (v1).
Each generation, a planner LLM writes K challengers, each a rewrite of a
parent program's genome.  A challenger must pass zero-cost static checks
(the S0 screen) before it races its parent on train (T) units: R1 runs it
once on its target unit and compares it with a fresh run of the parent
there (paired dS); the best one or two R1 survivors go on to R2 on two more
T units and a guard unit.  Every proposal gets a recorded verdict.  Units
are template@rung, e.g. II-11@x5 (o5 / o10: the benchmark's 5- / 10-agent
instances; x5: 5 agents with twice the data); when few unsolved T units
remain, templates the best program solves move up a rung.  Once the
evolution budget is spent, a single final selection compares the best
programs with v1 on validation (V) units and writes the champion to the
run directory (champion.json, champion.py).  TEST templates are never used.

Arms: full keeps the three experience components (diagnosis cards,
hypothesis ledger, program archive); mf_elite switches all three off.
--no-diag / --no-ledger / --no-archive switch one component off.

The budget counts scored executions in execution-equivalents (an o10
execution counts 2); --final-reserve of --budget pays for final selection.
A run directory is resumable with --resume: nothing scored is executed
twice.  Exit codes: 0 done; 3 a stop cap (--max-generations,
--max-idle-generations) fired before the budget was spent (a failed run);
75 the worker or planner service stayed unreachable (state saved; run the
command again with --resume to continue); 2 a usage error; 1 any other
error.

Inputs of a Silo-Bench run (their sha256 are recorded in config.json; a
resume on other inputs is refused):

  --units-from       the census: a queenbee.evaluate result JSON with the
                     seed program as arm v1 on every development unit
                     (unit classes and dS baselines come from it)
  --ladder-manifest  the x5 instances (python -m queenbee.evo.ladder)
  --split-file       the T / V split, written once per split seed with
                     --write-split so that every arm uses the same split

Typical use:

    python -m queenbee.evo.loop --write-split SPLIT.json --split-seed S \\
        --units-from CENSUS.json --ladder-manifest MANIFEST.json
    python -m queenbee.evo.loop --arm full --split-seed S --root RUN_DIR \\
        --units-from CENSUS.json --ladder-manifest MANIFEST.json \\
        --split-file SPLIT.json --planner-model MODEL --worker-model MODEL
    python -m queenbee.evo.loop --arm full --split-seed 1 --budget 30 \\
        --root RUN_DIR --fake

--fake runs offline with a fake worker provider and a fake planner (on a
synthetic census when --units-from is not given).  --task selects the task
family (default silo; see queenbee.tasks); a task other than silo is
recorded in config.json and a resume under another task is refused.
Workers and planner call OpenAI-compatible endpoints through
OPENAI_BASE_URL / OPENAI_API_KEY (the planner may use --planner-base-url /
--planner-api-key-env); model names default to $QUEENBEE_PLANNER_MODEL /
$QUEENBEE_WORKER_MODEL.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from queenbee.evo import memory as _memory
from queenbee.evo import common as _common
from queenbee.evo import race as _race
from queenbee.evo.prompt import (
    ARM_FLAGS,
    EvoFlags,
    arm_brief,
    build_evo_prompt,
    read_mint_hypothesis,
    t_template_statements,
)
from queenbee.paths import default_benchmarks_dir
from queenbee.program.mint import (
    DEFAULT_PLANNER_DEADLINE_S,
    DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
)
from queenbee.tasks import DEFAULT_TASK, add_task_argument, get_task, parse_task_args

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TEST_IDS = _common.TEST_IDS

#: ``mf_elite`` is ``full`` with diagnosis, ledger and archive switched off.
ARMS: tuple[str, ...] = ("full", "mf_elite")
#: Environment variables naming the models when the CLI flags do not.
PLANNER_MODEL_ENV = _common.PLANNER_MODEL_ENV
WORKER_MODEL_ENV = _common.WORKER_MODEL_ENV
#: Model name recorded by a ``--fake`` run that names no model.
FAKE_MODEL = "fake"
#: Planner API key variable when ``--planner-api-key-env`` is not given;
#: workers always read it (OpenAI SDK convention).
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
GOAL = _common.GOAL
WORKER_CONTRACT = _common.WORKER_CONTRACT
PYTHON_MAX_ROUNDS = _common.PYTHON_MAX_ROUNDS

#: Defaults: budget and final reserve (execution-equivalents), proposals per
#: generation, planner calls in flight (also the maximum), infra re-runs of
#: one job, planner error retries per process session, re-mints after a
#: planner deadline.
DEFAULT_BUDGET = 80.0
FINAL_RESERVE = 20.0
K_DEFAULT = 4
PLANNER_CONCURRENCY = 3
MAX_INFRA_RERUNS = 2
MAX_PLANNER_ERROR_RETRIES = 2
MAX_DEADLINE_RETRIES = 1

#: Batch stages inside a generation (racer batch ids ``g<g>:<stage>``), in order.
STAGES: tuple[str, ...] = ("r1", "r2", "confirm", "curr")
#: Final-selection reps (V frontier units) and the V-guard rep.
FINAL_REPS: tuple[int, ...] = (1, 2)
FINAL_GUARD_REP = 1
V_PURPOSE = "vselect"

#: Final selection: candidates besides v1, the T units a candidate must have,
#: V frontier units, V guards, the tie margin on mean V S (ties go to fewer
#: tokens) and the largest S loss the leader may show on a V guard.
N_FINAL_CANDIDATES = 3
MIN_FINAL_UNITS = 3
N_V_UNITS = 3
N_V_GUARDS = 2
TIE_EPS = 0.05
V_GUARD_LOSS_MAX = 0.2
#: Do-no-harm veto at final selection: a candidate whose mean S on any measured
#: T unit is below v1's mean there by more than this never becomes champion
#: (V may not cover every answer shape -- e.g. no per-agent-segment template --
#: so a T regression on such a unit is the only evidence of harm).
T_HARM_MAX = 0.4
#: ... measured only on T units where v1 is stable: >= 2 reps, all >= this.
T_HARM_STABLE_MIN = 0.8

#: Curriculum: saturated templates move up a rung when the current best has
#: fewer than this many frontier T units left; the default rung moves (a
#: task may define its own).
CURRICULUM_MIN_FRONTIER = 4
RUNG_NEXT: dict[str, str] = {"o5": "x5", "x5": "o10"}
#: At most this many confirmation reps per generation.
MAX_CONFIRM_PER_GEN = 1

#: Stop cap for a pathological planner: this many CONSECUTIVE generations
#: without a scored execution (every proposal S0-rejected / duplicate /
#: failed) stop evolution -- recorded as a FAILED (not budget-matched) run.
#: A worker or planner outage never makes a generation idle: the run stops
#: resumable first (:class:`InfraOutage`).
DEFAULT_MAX_IDLE_GENERATIONS = 10
DEFAULT_MAX_GENERATIONS = 60
#: The cheapest R1 at the start of a generation (1 candidate + its fresh
#: parent rep on an o5 unit): below this no generation is planned or minted.
MIN_GEN_COST = 2.0
#: Planner errors of one slot, of any kind and over all process sessions:
#: one error more than this fails the proposal (a deterministic error must
#: not loop).  Within one session ``MAX_PLANNER_ERROR_RETRIES`` retries
#: apply: past them an outage error (:func:`is_outage_error`) stops the run
#: resumable (``planner_outage``) and any other error fails the proposal.
MAX_PLANNER_ERROR_TOTAL = 9
#: Worker outage detection: this many CONNECTIVITY infra rows in a row (no
#: scored row in between) over >= OUTAGE_MIN_UNITS distinct units.  A unit-
#: or program-specific infra failure (one unit for every program, e.g. an
#: HTTP timeout on a large instance) is not an outage: the racer re-runs it
#: (``--max-infra-reruns``, default 2) and, if every attempt fails, it ends
#: infra_final (never scored 0).
OUTAGE_STREAK = 3
OUTAGE_MIN_UNITS = 2
#: Infra texts that mean "the service is unreachable" (HTTP 502 / 503,
#: refused or reset connections, name resolution failures).
_CONNECTIVITY_MARKERS: tuple[str, ...] = (
    "502", "503", "bad gateway", "service unavailable",
    "connection", "connecterror", "refused", "unreachable", "reset by peer",
    "name resolution", "network is",
)
#: OutageGuard: how long an outage is waited out before the run stops
#: resumable, and the first probe delay (doubling, at most 600 s), seconds.
DEFAULT_OUTAGE_WAIT_S = 1800.0
DEFAULT_OUTAGE_BACKOFF_S = 60.0
#: Final selection: replacement rounds for infra-exhausted V jobs.
MAX_FINAL_REPLACEMENTS = 2
#: Economize verdict: confirmed at equal S with <= this token ratio.
ECON_TOKEN_RATIO = 0.8
#: ``--econ-neutral``: the ledger note of an economize proposal that kept S
#: but saved no tokens (verdict ``inconclusive``, never ``refuted``).
ECON_NOT_ECONOMIZED = "not_economized"
#: ``--final-coverage-topup K``: purpose / batch of the T reps that top up
#: under-covered programs before the final plan, the K of the bare flag, and
#: how many V candidates the reserve must still pay for after the top-up (it
#: never spends more).
TOPUP_PURPOSE = "topup"
TOPUP_BATCH = "final:topup"
DEFAULT_TOPUP_K = 2
TOPUP_KEEP_CANDIDATES = 2
#: Purposes paid from the final reserve (``spent("final")``).
FINAL_PURPOSES: tuple[str, ...] = (V_PURPOSE, TOPUP_PURPOSE)
#: ``config.json`` keys of the optional behaviours (written only when on).
OPTION_KEYS: tuple[str, ...] = ("econ_neutral", "final_coverage_topup",
                                "explore_on_lockin", "preserve_dups")
#: CLI exit codes.
EXIT_RESUME_LATER = 75
EXIT_FAILED_RUN = 3

EPS = 1e-9
_BATCH_RE = re.compile(r"^g(\d+):([a-z0-9_]+)$")

LeakageGuardError = _common.LeakageGuardError
BudgetExceeded = _common.BudgetExceeded
RaceError = _race.RaceError
Job = _race.Job


class EvoStateError(RuntimeError):
    """A run cannot start or resume consistently (a missing input, a config
    that differs on resume, an instance or minted source whose sha256 does
    not match the recorded one, ...)."""


class InfraOutage(BaseException):
    """The worker or planner service is unreachable (HTTP 5xx, connection
    failures).

    Raised INSTEAD of burning job attempts or proposals: the run stops
    without marking evolution or final selection done, and ``--resume``
    later continues where it stopped (nothing scored is repeated).  A
    ``BaseException`` on purpose: the racer turns every ``Exception`` of an
    executor into an infra row, which would burn one of the job's attempts."""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else None


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _append_jsonl(path: Path, record: Mapping[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def template_of(unit_id: Any) -> str:
    return _common.template_of(unit_id)


def rung_of(unit_id: Any) -> str:
    return _common.rung_of(unit_id)


def rung_next() -> dict[str, str]:
    """Curriculum moves of the active task (default :data:`RUNG_NEXT`)."""

    moves = get_task().rung_next
    return RUNG_NEXT if moves is None else dict(moves)


def assert_not_test(ids: Any, *, where: str) -> None:
    """TEST guard (raises :class:`LeakageGuardError`)."""

    _common.assert_not_test(ids, where=where)


def batch_id(gen: int, stage: str) -> str:
    """Racer batch id of a generation stage (``g3:r1``)."""

    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return _race.batch_id_for(int(gen), stage)


def batch_key(batch: Any) -> tuple[int, int] | None:
    """``"g3:r2" -> (3, 1)``; None for anything else (census, final)."""

    match = _BATCH_RE.match(str(batch or ""))
    if not match or match.group(2) not in STAGES:
        return None
    return int(match.group(1)), STAGES.index(match.group(2))


def _unit_order(unit_id: str) -> tuple:
    return _race.unit_sort_key(unit_id)


def _file_sha(path: Any) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _pinned(paths: Iterable[Any]) -> list[list[str | None]]:
    return [[str(p), _file_sha(p)] for p in paths]


#: Planner exceptions that mean "service unreachable", not "bad request"
#: (on top of the transport markers of ``queenbee.program.execute``).
_OUTAGE_MARKERS: tuple[str, ...] = (
    "502", "503", "504", "bad gateway",
    "service unavailable", "gateway", "connection", "unreachable", "refused",
    "timed out", "timeout",
)


def is_outage_error(exc: BaseException) -> bool:
    """True for transport failures, HTTP 5xx and timeouts (retry later),
    False for errors a retry cannot fix (e.g. a malformed request or a
    rejected key)."""

    from queenbee.program.execute import _is_transport_error

    if _is_transport_error(exc):  # type: ignore[arg-type]
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _OUTAGE_MARKERS)


def is_connectivity_row(row: Mapping[str, Any]) -> bool:
    """An infra row whose error says the service was unreachable."""

    text = " ".join(str(row.get(k) or "") for k in ("infra", "error", "error_type")).lower()
    return any(marker in text for marker in _CONNECTIVITY_MARKERS)


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # torn last line (kill -9)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def lost_inflight(path: Path) -> list[dict[str, Any]]:
    """Start markers without an end marker (calls lost to a kill -9)."""

    starts: dict[str, dict[str, Any]] = {}
    ended: set[str] = set()
    for rec in _jsonl_records(path):
        if rec.get("event") == "start":
            starts[str(rec.get("id"))] = rec
        elif rec.get("event") == "end":
            ended.add(str(rec.get("id")))
    return [rec for cid, rec in starts.items() if cid not in ended]


# --------------------------------------------------------------------------- #
# Economize verdict (a cost claim, not a quality claim)
# --------------------------------------------------------------------------- #


def econ_no_saving(outcome: Mapping[str, Any], token_ratio: float | None) -> bool:
    """An executed economize proposal that kept S on its targets (target dS
    not below 0) but used at least the parent's tokens."""

    target = outcome.get("target_dS")
    return bool(outcome.get("executed")) and target is not None and token_ratio is not None \
        and float(target) >= -EPS and float(token_ratio) >= 1.0 - EPS


def economize_verdict(outcome: Mapping[str, Any], *, token_ratio: float | None,
                      thresholds: Any = None, neutral_no_saving: bool = False) -> str:
    """Economize briefs target saturated units, so dS ~ 0 is their success,
    not a refutation.  confirmed: S not lower on the targets, tokens <=
    ``ECON_TOKEN_RATIO`` x the parent's, guard ok; refuted: S lower or no
    token saving; inconclusive otherwise.  A proposal that was not executed
    gets the general host verdict (``race.host_verdict``).

    ``neutral_no_saving`` (``--econ-neutral``): no token saving at an S that
    held is ``inconclusive``, not ``refuted`` -- a failed cost claim says
    nothing against the mechanism (e.g. a one-round mesh that ties its parent
    at S = 1.0 would otherwise be recorded as a refuted mechanism)."""

    if neutral_no_saving and econ_no_saving(outcome, token_ratio):
        return "inconclusive"
    if not outcome.get("executed"):
        return _race.host_verdict(outcome, thresholds)
    target = outcome.get("target_dS")
    if target is None or token_ratio is None:
        return "inconclusive"
    if float(target) < -EPS or float(token_ratio) >= 1.0 - EPS:
        return "refuted"
    if float(token_ratio) <= ECON_TOKEN_RATIO + EPS and outcome.get("guard_ok") is True:
        return "confirmed"
    return "inconclusive"


# --------------------------------------------------------------------------- #
# Prompt value redaction (a false-positive audit hit must not drop a proposal)
# --------------------------------------------------------------------------- #


def redact_evidence_values(text: str, values: Sequence[Any],
                           indices: Iterable[int] | None = None,
                           *, min_chars: int = 4) -> tuple[str, int]:
    """Replace the audited forms of ``values[indices]`` (all when None) by
    ``[redacted]`` inside the prompt's evidence sections only (the sections
    ``prompt.audit_evo_prompt`` scans for values); returns (text, n)."""

    from queenbee.evo.prompt import EVO_EVIDENCE_SECTION_HEADERS
    from queenbee.program.mint import _value_forms

    wanted = list(range(len(values))) if indices is None else sorted(set(int(i) for i in indices))
    forms: set[str] = set()
    for i in wanted:
        if 0 <= i < len(values):
            forms.update(f for f in _value_forms(values[i], min_chars=min_chars) if len(f) >= min_chars)
    patterns = [re.compile(r"(?<![\w.])" + re.escape(f) + r"(?![\w])")
                for f in sorted(forms, key=len, reverse=True)]
    headers = tuple(EVO_EVIDENCE_SECTION_HEADERS)
    out: list[str] = []
    inside = False
    count = 0
    for line in str(text).splitlines(keepends=True):
        stripped = line.lstrip()
        starts = any(stripped.startswith(h) for h in headers)
        if starts:
            inside = True
        elif inside and stripped.startswith("==="):
            inside = False
        if inside:
            for rx in patterns:
                line, n = rx.subn("[redacted]", line)
                count += n
        out.append(line)
    return "".join(out), count


# --------------------------------------------------------------------------- #
# Executor adapter: pause on an outage, never burn attempts on it
# --------------------------------------------------------------------------- #


class OutageGuard:
    """Wraps the racer's executor: pause on an outage, never burn attempts on it.

    A streak of ``streak`` infra rows with a CONNECTIVITY signature (HTTP
    502 / 503, refused or reset connections; :func:`is_connectivity_row`)
    over >= ``min_units`` distinct units, with no scored row in between, is
    an outage: the thread pauses (backoff up to ``wait_s``) and re-probes its
    own job WITHOUT returning the infra rows to the racer (so they cost none
    of the job's attempts); a scored probe resumes normal work, else
    :class:`InfraOutage` stops the run (resumable).  Other infra rows (one
    flaky job, one unit that times out for every program) go to the racer
    as usual (re-run up to ``max_infra_reruns`` times, never scored 0).
    Every call is bracketed by start / end markers in ``marker_path`` so
    executions lost to a kill -9 are counted on resume."""

    def __init__(self, execute: Callable[[Mapping[str, Any]], Any], *,
                 streak: int = OUTAGE_STREAK, min_units: int = OUTAGE_MIN_UNITS,
                 wait_s: float = DEFAULT_OUTAGE_WAIT_S,
                 backoff_s: float = DEFAULT_OUTAGE_BACKOFF_S,
                 marker_path: Path | None = None,
                 on_event: Callable[..., None] | None = None) -> None:
        self.execute = execute
        self.streak_n = max(1, int(streak))
        self.min_units = max(1, int(min_units))
        self.wait_s = max(0.0, float(wait_s))
        self.backoff_s = max(0.0, float(backoff_s))
        self.marker_path = Path(marker_path) if marker_path else None
        self.on_event = on_event
        self._lock = threading.Lock()
        self._marker_lock = threading.Lock()
        self._streak: list[str] = []
        self._tripped = threading.Event()
        self.reason: str | None = None
        self.probes = 0
        self.pauses = 0

    def _event(self, name: str, **data: Any) -> None:
        if self.on_event is not None:
            try:
                self.on_event(name, **data)
            except Exception:  # noqa: BLE001 - events are advisory
                pass

    def _mark(self, record: Mapping[str, Any]) -> None:
        if self.marker_path is not None:
            _append_jsonl(self.marker_path, dict(record) | {"at": _now()}, self._marker_lock)

    @staticmethod
    def _key(req: Mapping[str, Any]) -> str:
        job = req.get("job")
        return str(getattr(job, "key", None) or req.get("unit_id") or "?")

    @staticmethod
    def _unit(req: Mapping[str, Any]) -> str:
        job = req.get("job")
        return str(getattr(job, "unit_id", None) or req.get("unit_id") or "?")

    def _run(self, req: Mapping[str, Any], *, probe: bool = False) -> dict[str, Any]:
        call_id = uuid.uuid4().hex
        self._mark({"id": call_id, "event": "start", "key": self._key(req), "probe": probe})
        try:
            try:
                return dict(self.execute(req))
            except Exception as exc:  # noqa: BLE001 - the racer's own classifier
                return dict(_race._failure_row(exc))
        finally:
            self._mark({"id": call_id, "event": "end"})

    def _ok(self) -> None:
        with self._lock:
            self._streak.clear()

    def _is_outage(self, unit: str) -> bool:
        with self._lock:
            self._streak.append(unit)
            return (len(self._streak) >= self.streak_n
                    and len(set(self._streak)) >= self.min_units)

    def __call__(self, req: Mapping[str, Any]) -> dict[str, Any]:
        if self._tripped.is_set():
            raise InfraOutage(self.reason or "worker outage")
        key = self._key(req)
        row = self._run(req)
        if not _race.is_infra(row):
            self._ok()
            return row
        if not is_connectivity_row(row) or not self._is_outage(self._unit(req)):
            return row  # not (yet) an outage: the racer re-runs it (max_infra_reruns)
        with self._lock:
            self.pauses += 1
        self._event("outage_pause", key=key, infra=str(row.get("infra"))[:200],
                    wait_s=self.wait_s)
        deadline = time.time() + self.wait_s
        k = 0
        while not self._tripped.is_set() and time.time() < deadline:
            delay = min(self.backoff_s * (2 ** k), 600.0, max(0.0, deadline - time.time()))
            k += 1
            if self._tripped.wait(delay):
                break
            with self._lock:
                self.probes += 1
            row = self._run(req, probe=True)
            if not _race.is_infra(row):
                self._ok()
                self._event("outage_recovered", key=key, probes=k)
                return row
        with self._lock:
            if self.reason is None:
                self.reason = (f"worker outage: {self.streak_n}+ infra rows in a row "
                               f"(last: {str(row.get('infra'))[:160]})")
        self._tripped.set()
        raise InfraOutage(self.reason)


# --------------------------------------------------------------------------- #
# The loop's racer (evo.race.Racer plus the merge rule)
# --------------------------------------------------------------------------- #


class _LoopRacer(_race.Racer):
    """``race.Racer`` + the loop's merge rule: a merge child is compared, per
    unit, against the BEST of its parents (``baseline`` of each parent --
    fresh reps, else archived rows; the racer's own v1 fallback applies when
    none has rows).  A merge's R1 unit is one its second parent leads, so
    against the mutated parent alone a mere copy of the second parent would
    count as a gain.  Everything else is the base racer unchanged (the v1
    fallback on R2 units, ``baseline_C``)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = threading.local()

    def parents_of(self, program_id: str) -> list[str]:
        prog = self._programs.get(str(program_id))
        return [str(p) for p in ((prog.meta if prog is not None else {}) or {}).get("parent_ids") or []]

    def merge_baseline(self, parents: Sequence[str], unit_id: str,
                       gen: int) -> tuple[float | None, str]:
        best = _race.Racer.baseline(self, parents[0], unit_id, gen)
        for other in parents[1:]:
            s, src = _race.Racer.baseline(self, other, unit_id, gen)
            if s is not None and (best[0] is None or s > best[0] + EPS):
                best = (s, f"merge:{other}:{src}")
        return best

    def baseline(self, parent_id: str, unit_id: str, gen: int) -> tuple[float | None, str]:
        parents = getattr(self._ctx, "parents", None)
        if parents and len(parents) > 1 and str(parent_id) == parents[0]:
            return self.merge_baseline(parents, unit_id, gen)
        return super().baseline(parent_id, unit_id, gen)

    def outcome(self, candidate: Any, gen: int, *,
                r2_plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
        pid = str(_race._get(candidate, "program_id") or "")
        parents = self.parents_of(pid)
        previous = getattr(self._ctx, "parents", None)
        self._ctx.parents = parents
        try:
            out = super().outcome(candidate, gen, r2_plan=r2_plan)
        finally:
            self._ctx.parents = previous
        observed = out.get("observed") or {}
        out["baseline_srcs"] = {u: (v or {}).get("baseline_src")
                                for u, v in (out.get("per_unit") or {}).items()}
        out["n_observed_units"] = len(observed)
        out["n_r2_units_observed"] = sum(1 for u in out.get("r2_units") or [] if u in observed)
        if len(parents) > 1:
            out["baseline_rule"] = "merge: best parent per unit"
        return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def arm_flags(arm: str, *, no_diag: bool = False, no_ledger: bool = False,
              no_archive: bool = False) -> EvoFlags:
    """The arm's diagnosis / ledger / archive flags (``evo.prompt.ARM_FLAGS``)
    with the ``--no-diag`` / ``--no-ledger`` / ``--no-archive`` switches
    applied."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose {ARMS}")
    base = ARM_FLAGS[arm]
    return EvoFlags(
        diag=base.diag and not no_diag,
        ledger=base.ledger and not no_ledger,
        archive=base.archive and not no_archive,
    )


def parent_policy(flags: EvoFlags) -> str:
    """``specialist`` (with the archive) or ``incumbent`` -- the parent rule
    of ``memory.Scheduler`` for these flags (recorded in the config)."""

    return "specialist" if flags.archive else "incumbent"


def _model_name(value: str | None, env: str, *, fake: bool) -> str | None:
    """``value``, else ``$env``, else :data:`FAKE_MODEL` for a ``--fake`` run."""

    name = str(value or os.environ.get(env) or "").strip()
    return name or (FAKE_MODEL if fake else None)


def _source_label(source: Any) -> str:
    """How a census source is named in the run files."""

    return "<synthetic census>" if isinstance(source, Mapping) else str(source)


def _pinned_sources(sources: Iterable[Any]) -> list[list[str | None]]:
    """:func:`_pinned` for census sources; an in-memory census document is
    pinned by the sha256 of its canonical JSON."""

    return [[_source_label(s), _sha256(json.dumps(s, sort_keys=True))]
            if isinstance(s, Mapping) else [str(s), _file_sha(s)] for s in sources]


@dataclass
class EvoConfig:
    """Settings of one evolution run (mostly the CLI flags); :meth:`frozen`
    lists the ones a resume may not change."""

    root: Path
    arm: str = "full"
    split_seed: int = 1
    budget: float = DEFAULT_BUDGET
    final_reserve: float = FINAL_RESERVE
    no_diag: bool = False
    no_ledger: bool = False
    no_archive: bool = False
    fake: bool = False
    resume: bool = False
    K: int = K_DEFAULT
    split_file: Path | None = None
    unit_sources: tuple[Path, ...] = ()
    ladder_manifests: tuple[Path, ...] = ()
    trace_dirs: tuple[Path, ...] = ()
    thresholds_file: Path | None = None
    #: Silo-Bench ``benchmarks`` directory; None = ``default_benchmarks_dir()``.
    benchmarks_dir: Path | None = None
    #: Model names; None = ``$QUEENBEE_PLANNER_MODEL`` / ``$QUEENBEE_WORKER_MODEL``.
    planner_model: str | None = None
    planner_effort: str | None = "high"
    worker_model: str | None = None
    #: Planner endpoint and API key variable; None = the OpenAI SDK environment.
    planner_base_url: str | None = None
    planner_api_key_env: str | None = None
    python_max_rounds: int = PYTHON_MAX_ROUNDS
    parallel_cases: int = 6
    planner_concurrency: int = PLANNER_CONCURRENCY
    #: Worker / planner request timeout in seconds, per LLM call; one call
    #: with long reasoning on a large x5 / o10 instance can be very slow,
    #: hence the generous default.
    request_timeout: float = 5400.0
    connect_attempts: int = 8
    connect_backoff_s: float = 8.0
    max_infra_reruns: int = MAX_INFRA_RERUNS
    infra_backoff_s: float = 60.0
    planner_backoff_s: float = 60.0
    #: Planner output cap (tokens) and wall clock (s) per call; 0 = none,
    #: None = ``$QB_PLANNER_MAX_COMPLETION_TOKENS`` / ``$QB_PLANNER_DEADLINE_S``,
    #: else the default.
    planner_max_completion_tokens: int | None = DEFAULT_PLANNER_MAX_COMPLETION_TOKENS
    planner_deadline_s: float | None = DEFAULT_PLANNER_DEADLINE_S
    traces: bool = True
    async_mint: bool = True
    max_generations: int = DEFAULT_MAX_GENERATIONS
    max_idle_generations: int = DEFAULT_MAX_IDLE_GENERATIONS
    #: Arm name of the seed program's rows in the census files.
    v1_arm: str = "v1"
    #: Worker outage handling (OutageGuard): pause this long (backoff) before
    #: the run stops resumable; 0 = stop at once.
    outage_wait_s: float = DEFAULT_OUTAGE_WAIT_S
    outage_backoff_s: float = DEFAULT_OUTAGE_BACKOFF_S
    #: Inherited programs (``--inherit-program NAME=PATH``, e.g. the champion
    #: of another run) join this run's archive as roots.  Only their genome is
    #: taken: the code around it is v1's.  Each gets ONE paid rep on every T
    #: frontier unit (purpose ``inherit``, charged to the evolution budget, so
    #: arms stay budget-matched) and is then an eligible parent, specialist
    #: and final candidate like any confirmed program.  () = the run starts
    #: from the seed program alone.
    inherit_programs: tuple[tuple[str, Path], ...] = ()
    #: Templates whose gold answers the benchmark generator rounds to a
    #: precision the task text never states (on Silo-Bench: II-12, III-27,
    #: III-28).  This is a property of the benchmark, not of the worker, so
    #: their units that v1 does not solve are classed ``irreducible`` (never
    #: targeted) even when the traces show no near-miss.  Applied when the run
    #: starts; the unit evidence is then recorded in ``state/split.json`` and
    #: re-used on every resume.
    irreducible_templates: tuple[str, ...] = ()
    #: ``--vacuous-guard``: with NO guard unit in the pool (the seed program
    #: solves no T unit stably), the verdict's guard condition holds vacuously
    #: instead of failing closed (``RaceConfig.vacuous_guard``).
    vacuous_guard: bool = False
    #: ``--local-only-diag``: diagnosis cards flag wrong agents that answered
    #: from their own shard alone (class ``local-only``;
    #: ``diagnosis.LOCAL_ONLY_ENV``).
    local_only_diag: bool = False
    #: ``--attempt-diag``: cards count wrong agents that already held the
    #: right answer (``diagnosis.ATTEMPT_ENV``), and ledger entries of
    #: executed proposals keep the child's card summary after the change
    #: (``after_diag``) for the planner.
    attempt_diag: bool = False
    #: Optional behaviours (off by default; recorded in ``config.json`` only when on).
    #: ``--econ-neutral``: an economize proposal that kept S but saved no
    #: tokens is ``inconclusive`` (ledger note ``not_economized``), never a
    #: refuted mechanism (economize_verdict).
    econ_neutral: bool = False
    #: ``--final-coverage-topup K``: before the final plan, up to K
    #: high-scoring programs measured on < 3 T units get fresh T reps on more
    #: units (paid from the final reserve, never below the V cost of
    #: ``TOPUP_KEEP_CANDIDATES`` candidates) so they can become candidates;
    #: 0 = off.
    final_coverage_topup: int = 0
    #: ``--explore-on-lockin``: while the ledger's confirmed entries
    #: concentrate in one mechanism family (``memory.Ledger.lockin``), one
    #: slot per generation is an explore brief asking for another family
    #: (``memory.Scheduler.explore_on_lockin``; arms with the ledger only).
    explore_on_lockin: bool = False
    #: ``--preserve-dups``: the API card's forwarded-raw-data rule (keep
    #: every item, repeated values included, in order) and multiset raw_kept%
    #: + dup_lost on the diagnosis cards (``credit.MULTISET_ENV``).
    preserve_dups: bool = False
    #: Task family (``queenbee.tasks``); None = the active task.  Recorded in
    #: ``config.json`` when it is not the default task.
    task: str | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.task = str(self.task) if self.task else get_task().name
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose {ARMS}")
        inherited = []
        for item in self.inherit_programs or ():
            name, path = (item.split("=", 1) if isinstance(item, str) else item)
            name = str(name).strip()
            if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", name):
                raise ValueError(f"inherit program name {name!r}: use [A-Za-z0-9_-]{{1,40}}")
            inherited.append((name, Path(path)))
        if len({n for n, _p in inherited}) != len(inherited):
            raise ValueError("inherit program names must be distinct")
        self.inherit_programs = tuple(inherited)
        self.irreducible_templates = tuple(str(t) for t in (self.irreducible_templates or ()) if str(t))
        self.vacuous_guard = bool(self.vacuous_guard)
        self.local_only_diag = bool(self.local_only_diag)
        self.attempt_diag = bool(self.attempt_diag)
        self.econ_neutral = bool(self.econ_neutral)
        self.final_coverage_topup = int(self.final_coverage_topup or 0)
        if self.final_coverage_topup < 0:
            raise ValueError("final_coverage_topup must be >= 0")
        self.explore_on_lockin = bool(self.explore_on_lockin)
        self.preserve_dups = bool(self.preserve_dups)
        self.benchmarks_dir = (Path(self.benchmarks_dir) if self.benchmarks_dir is not None
                               else default_benchmarks_dir())
        self.planner_model = _model_name(self.planner_model, PLANNER_MODEL_ENV, fake=bool(self.fake))
        self.worker_model = _model_name(self.worker_model, WORKER_MODEL_ENV, fake=bool(self.fake))
        self.planner_base_url = str(self.planner_base_url).strip() if self.planner_base_url else None
        self.planner_api_key_env = (str(self.planner_api_key_env).strip()
                                    if self.planner_api_key_env else None)
        for name in ("planner_max_completion_tokens", "planner_deadline_s"):
            value = getattr(self, name)
            if value is not None and float(value) < 0:
                raise ValueError(f"{name} must be >= 0 (0 = none)")
        self.unit_sources = tuple(Path(p) for p in self.unit_sources)
        self.ladder_manifests = tuple(Path(p) for p in self.ladder_manifests)
        self.trace_dirs = tuple(Path(p) for p in self.trace_dirs)
        if self.split_file is not None:
            self.split_file = Path(self.split_file)
        if self.thresholds_file is not None:
            self.thresholds_file = Path(self.thresholds_file)
        self.budget = float(self.budget)
        self.final_reserve = float(self.final_reserve)
        if not 0 <= self.final_reserve < self.budget:
            raise ValueError(f"final_reserve {self.final_reserve} must be in [0, budget)")
        if int(self.K) < 1:
            raise ValueError("K must be >= 1")
        if not 1 <= int(self.planner_concurrency) <= PLANNER_CONCURRENCY:
            raise ValueError(f"planner_concurrency must be in 1..{PLANNER_CONCURRENCY}")
        if int(self.max_idle_generations) < 1:
            raise ValueError("max_idle_generations must be >= 1")

    def resolved_unit_sources(self) -> list[Any]:
        """``--units-from`` (pinned by sha256 in ``config.json``); a
        ``--fake`` run without it uses :func:`fake_census`."""

        if self.unit_sources:
            return list(self.unit_sources)
        return [fake_census()] if self.fake else []

    def inputs(self) -> dict[str, Any]:
        """The inputs that decide units, split, dS baselines or thresholds:
        each file with its sha256 (a resume on other bytes is refused), the
        trace and benchmark directories by path."""

        return {
            "unit_sources": _pinned_sources(self.resolved_unit_sources()),
            "ladder_manifests": _pinned(self.ladder_manifests),
            "split_file": _pinned([self.split_file])[0] if self.split_file else None,
            "thresholds_file": _pinned([self.thresholds_file])[0] if self.thresholds_file else None,
            "trace_dirs": [str(p) for p in self.trace_dirs],
            "benchmarks_dir": str(self.benchmarks_dir),
        }

    @property
    def flags(self) -> EvoFlags:
        return arm_flags(self.arm, no_diag=self.no_diag, no_ledger=self.no_ledger,
                         no_archive=self.no_archive)

    @property
    def evolution_budget(self) -> float:
        return self.budget - self.final_reserve

    @property
    def llm_provider(self) -> str:
        return "fake" if self.fake else "openai"

    @property
    def variant(self) -> str:
        """Arm label with its switched-off components and ``inherit`` when
        programs are inherited (e.g. ``full-nodiag``)."""

        parts = [self.arm]
        parts += [name for name, on in (("nodiag", self.no_diag), ("noledger", self.no_ledger),
                                        ("noarchive", self.no_archive)) if on]
        if self.inherit_programs:
            parts.append("inherit")  # the archive starts with inherited programs
        return "-".join(parts)

    def effective_planner_deadline(self) -> float | None:
        """The planner's per-call deadline in seconds as the mint applies it
        (None = no deadline)."""

        if self.planner_deadline_s is not None:
            return float(self.planner_deadline_s) or None
        raw = os.environ.get("QB_PLANNER_DEADLINE_S", "").strip()
        try:
            value = float(raw) if raw else 0.0
        except ValueError:
            value = 0.0
        return value if value > 0 else DEFAULT_PLANNER_DEADLINE_S

    def frozen(self) -> dict[str, Any]:
        """The fields a resume must not change (written to ``config.json``)."""

        flags = self.flags
        return {
            "arm": self.arm, "variant": self.variant,
            "flags": {"diag": flags.diag, "ledger": flags.ledger, "archive": flags.archive},
            "parent_policy": parent_policy(flags),
            "split_seed": int(self.split_seed),
            "split_file": str(self.split_file) if self.split_file else None,
            "budget": self.budget, "final_reserve": self.final_reserve, "K": int(self.K),
            "fake": bool(self.fake),
            "planner_model": self.planner_model, "planner_effort": self.planner_effort,
            "worker_model": self.worker_model,
            "python_max_rounds": int(self.python_max_rounds),
            "thresholds_file": str(self.thresholds_file) if self.thresholds_file else None,
            "max_idle_generations": int(self.max_idle_generations),
            "inputs": self.inputs(),
        } | (
            {"inherit_programs": [{"name": n, "path": str(p), "sha256": _file_sha(p)}
                                  for n, p in self.inherit_programs]}
            if self.inherit_programs else {}) | (
            {"irreducible_templates": list(self.irreducible_templates)}
            if self.irreducible_templates else {}) | (
            {"vacuous_guard": True} if self.vacuous_guard else {}) | (
            {"local_only_diag": True} if self.local_only_diag else {}) | (
            {"attempt_diag": True} if self.attempt_diag else {}) | (
            # optional behaviours: keys absent when off
            {"econ_neutral": True} if self.econ_neutral else {}) | (
            {"final_coverage_topup": self.final_coverage_topup}
            if self.final_coverage_topup else {}) | (
            {"explore_on_lockin": True} if self.explore_on_lockin else {}) | (
            {"preserve_dups": True} if self.preserve_dups else {}) | (
            {"task": self.task} if self.task != DEFAULT_TASK else {})


# --------------------------------------------------------------------------- #
# Units (instance resolver for the racer + T/V/TEST guards)
# --------------------------------------------------------------------------- #


class UnitPool:
    """Dev units, the split, instance resolution (``get(unit) -> (instance,
    sha)``: the racer's resolver) and the side guards.

    The evolution pool is every T unit whose v1 class is ``frontier`` or
    ``guard`` plus the units the curriculum promoted (``extra``)."""

    def __init__(self, units: Mapping[str, Any], split: Any, *,
                 benchmarks_dir: Path, manifest_paths: Mapping[str, Path] | None = None) -> None:
        assert_not_test(list(units), where="unit pool")
        self.units = dict(units)
        self.split = split
        self.benchmarks_dir = Path(benchmarks_dir)
        self.manifest_paths = {str(k): Path(v) for k, v in dict(manifest_paths or {}).items()}
        self.extra: dict[str, dict[str, Any]] = {}
        self._cache: dict[str, tuple[Any, str]] = {}
        self._shape: dict[str, str | None] = {}
        self._lock = threading.Lock()

    # -- sides / guards ---------------------------------------------------- #
    def side(self, unit_id: str) -> str | None:
        assert_not_test([unit_id], where="unit side")
        template = template_of(unit_id)
        if template in self.split.T:
            return "T"
        if template in self.split.V:
            return "V"
        return None

    def assert_side(self, unit_id: str, side: str, *, where: str) -> None:
        got = self.side(unit_id)
        if got != side:
            raise LeakageGuardError(
                f"LEAKAGE_GUARD ({where}): {unit_id!r} is on side {got!r}, not {side!r}"
            )

    # -- facts ------------------------------------------------------------- #
    def n_agents(self, unit_id: str) -> int:
        return int(_common.task_rungs().get(rung_of(unit_id)) or 5)

    def exec_eq(self, unit_id: str) -> int:
        return int(_race.exec_eq(unit_id))

    def klass(self, unit_id: str) -> str:
        unit = self.units.get(unit_id)
        return str(getattr(unit, "klass", "unresolved")) if unit is not None else "unresolved"

    def pool_T(self) -> list[str]:
        out = [u for u, unit in self.units.items()
               if self.side(u) == "T" and getattr(unit, "klass", None) in ("frontier", "guard")]
        out += [u for u in self.extra if self.side(u) == "T"]
        return sorted(dict.fromkeys(out), key=_unit_order)

    def census_frontier_T(self) -> list[str]:
        return [u for u in self.pool_T() if self.klass(u) == "frontier"]

    def guard_T(self) -> list[str]:
        return [u for u in self.pool_T() if self.klass(u) == "guard"]

    def _v_ordered(self, klass: str) -> list[str]:
        units = sorted((u for u, unit in self.units.items()
                        if self.side(u) == "V" and getattr(unit, "klass", None) == klass),
                       key=_unit_order)
        first, rest, seen = [], [], set()
        for u in units:  # distinct templates first (a fixed order, never by score)
            (rest if template_of(u) in seen else first).append(u)
            seen.add(template_of(u))
        return first + rest

    def V_frontier(self) -> list[str]:
        return self._v_ordered("frontier")

    def V_guard(self) -> list[str]:
        return self._v_ordered("guard")

    # -- instances --------------------------------------------------------- #
    def path(self, unit_id: str) -> Path:
        assert_not_test([unit_id], where="instance path")
        unit = self.units.get(unit_id)
        if unit is not None and getattr(unit, "instance_path", None):
            return Path(unit.instance_path)
        if unit_id in self.manifest_paths:
            return self.manifest_paths[unit_id]
        resolve = get_task().instance_path
        if resolve is not None:
            path = resolve(unit_id)
            if path is None:
                raise EvoStateError(f"{unit_id}: no instance (the task resolves none)")
            return Path(path)
        if rung_of(unit_id) in ("o5", "o10"):
            return self.benchmarks_dir / f"{template_of(unit_id)}_n{self.n_agents(unit_id)}.json"
        raise EvoStateError(f"{unit_id}: no instance (not in any ladder manifest)")

    def has_instance(self, unit_id: str) -> bool:
        try:
            return self.path(unit_id).is_file()
        except EvoStateError:
            return False

    def get(self, unit_id: str) -> tuple[Any, str]:
        with self._lock:
            if unit_id in self._cache:
                return self._cache[unit_id]
        from queenbee.evaluate import instance_sha256, load_instance_file

        if self.side(unit_id) is None:
            raise LeakageGuardError(f"{unit_id}: not a dev unit of this split")
        instance = load_instance_file(self.path(unit_id))
        assert_not_test([getattr(instance, "case_id", "")], where="instance")
        if int(instance.n_agents) != self.n_agents(unit_id):
            raise EvoStateError(f"{unit_id}: instance has n={instance.n_agents}")
        sha = instance_sha256(instance)
        want = getattr(self.units.get(unit_id), "instance_sha256", None)
        if want and want != sha:
            raise EvoStateError(f"{unit_id}: instance sha {sha[:12]} != recorded {str(want)[:12]}")
        entry = (instance, sha)
        with self._lock:
            self._cache[unit_id] = entry
        return entry

    def instance(self, unit_id: str) -> Any:
        return self.get(unit_id)[0]

    def instance_sha(self, unit_id: str) -> str:
        return self.get(unit_id)[1]

    def shape(self, unit_id: str) -> str | None:
        if unit_id not in self._shape:
            from queenbee.evo.diagnosis import instance_answer_shape

            try:
                self._shape[unit_id] = instance_answer_shape(self.instance(unit_id))
            except (OSError, EvoStateError, ValueError):
                self._shape[unit_id] = None
        return self._shape[unit_id]


# --------------------------------------------------------------------------- #
# Fake planner (--fake / tests): valid, behaviour-changing genome variants
# --------------------------------------------------------------------------- #

_PARENT_GENOME_RE = re.compile(r"=== PARENT GENOME[^\n]*\n```python\n(.*?)\n```", re.S)
_PHASES_BLOCK_RE = re.compile(r"^PHASES = \[\n(.*?)^\]\n", re.S | re.M)
_CLASS_RE = re.compile(r"Failure class assigned to this brief: ([a-z\-]+)")
_TARGETS_RE = re.compile(r"^Target units: (.*)$", re.M)
_UNIT_RE = re.compile(r"\b(I{1,3}-\d{2}@[ox]\d+)\b")
_ANY_UNIT_RE = re.compile(r"(?<![\w-])([A-Za-z0-9_\-]+@[ox]\d+)(?!\w)")


def _target_units(text: str) -> list[str]:
    """Unit ids in ``text``: Silo-Bench ids, or the ``<template>@<rung>`` ids
    of a task with its own template ids."""

    if get_task().template_pattern is None:
        return _UNIT_RE.findall(text)
    return [u for u in _ANY_UNIT_RE.findall(text) if _common.templates_in(u.split("@", 1)[0])]

_FAKE_WIS: tuple[str, ...] = (
    "Carry forward every data item you have read so far, verbatim, then state your current best answer. Note {n}.",
    "Before you answer, restate each value you hold and each value you received. Note {n}.",
)


def _format_phase(phase: Mapping[str, Any]) -> str:
    parts = []
    for key, value in phase.items():
        if isinstance(value, bool):
            text = "True" if value else "False"
        elif isinstance(value, (int, float)):
            text = repr(value)
        else:
            text = json.dumps(str(value))
        parts.append(f"{json.dumps(str(key))}: {text}")
    return "    {" + ", ".join(parts) + "},"


def fake_mutation(genome: str, index: int) -> tuple[str, str]:
    """``(new genome, mechanism)``: one deterministic structural edit of the
    ``PHASES`` list (a phase ``wi``, a leading mesh round, a digest round
    before the last phase).  Always simulable and S0-clean on the seed
    program."""

    match = _PHASES_BLOCK_RE.search(genome)
    if not match:
        return genome, "no-op (no PHASES block)"
    try:
        phases = ast.literal_eval("[" + match.group(1) + "]")
    except (ValueError, SyntaxError):
        return genome, "no-op (unparsable PHASES)"
    phases = [dict(p) for p in phases]
    kinds = [str(p.get("kind")) for p in phases]
    choice = index % 4
    n = 1 + (index * 7919) % 997
    if choice == 1 and len(phases) < 5 and kinds[:1] != ["mesh"]:
        phases.insert(0, {"kind": "mesh", "rounds": 1})
        mechanism = "add one leading mesh round"
    elif choice == 2 and len(phases) < 6 and "digest" not in kinds and len(phases) >= 1:
        phases.insert(len(phases) - 1, {"kind": "digest", "rounds": 1})
        mechanism = "add a digest round before the last phase"
    else:
        target = 0 if choice in (0, 1, 2) else len(phases) - 1
        phases[target] = dict(phases[target]) | {"wi": _FAKE_WIS[choice % 2].format(n=n)}
        mechanism = f"set the work instruction of phase {target}"
    body = "\n".join(_format_phase(p) for p in phases) + "\n"
    return genome[: match.start(1)] + body + genome[match.end(1):], mechanism


class FakeEvoPlanner:
    """Offline planner: reads the PARENT GENOME block of an evo prompt and
    answers ``HYPOTHESIS`` + a :func:`fake_mutation` of it.  The mutation is
    keyed on the prompt hash and the n-th call with that prompt, so it is
    deterministic under any planner concurrency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: Counter = Counter()
        self.calls = 0

    def complete(self, prompt: str, model_name: str | None = None,
                 temperature: float | None = None, json_mode: bool = False, **_: Any) -> Any:
        key = _sha256(prompt)
        with self._lock:
            self.calls += 1
            nth = self._seen[key]
            self._seen[key] += 1
        match = _PARENT_GENOME_RE.search(str(prompt))
        genome = match.group(1) if match else ""
        index = int(key[:8], 16) + nth
        new_genome, mechanism = fake_mutation(genome, index)
        cls_match = _CLASS_RE.search(prompt)
        found = _TARGETS_RE.search(prompt)
        targets = _target_units(found.group(1) if found else "")
        hypothesis = {
            "target_units": targets[:3],
            "failure_class": cls_match.group(1) if cls_match else "divergent",
            "mechanism": mechanism,
            "predicted_dS": {u: 0.2 for u in targets[:3]},
        }
        return SimpleNamespace(
            text="HYPOTHESIS: " + json.dumps(hypothesis) + "\n```python\n"
            + new_genome.rstrip("\n") + "\n```\n",
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": 400},
        )


#: v1 S per rep of the synthetic census, by a template's position in its tier
#: (the last two templates of every tier are solved).
_FAKE_CENSUS_S: tuple[tuple[float, float], ...] = (
    (0.4, 0.6), (0.6, 0.6), (0.4, 0.4), (0.8, 0.6),
)


def _fake_census_table() -> dict[str, list[float]]:
    by_tier: dict[str, list[str]] = {}
    for template in _common.task_dev_ids():
        by_tier.setdefault(template.split("-")[0], []).append(template)
    table: dict[str, list[float]] = {}
    for templates in by_tier.values():
        for i, template in enumerate(templates):
            solved = i >= len(templates) - 2
            table[template] = [1.0, 1.0] if solved else list(_FAKE_CENSUS_S[i % len(_FAKE_CENSUS_S)])
    return table


def fake_census(table: Mapping[str, Sequence[float]] | None = None) -> dict[str, Any]:
    """A seed-program census in the evaluator's result format, for offline
    runs: v1 rows on the o5 unit of every template of ``table`` (template ->
    S of each rep; default: every dev template, the last two of each tier
    solved), with synthetic diagnosis fields.  Under a task with its own
    rungs the rows are on its smallest ``o`` rung; under a task with its own
    template ids the answer shape is the one of the task's instance."""

    table = {str(t): [float(s) for s in values]
             for t, values in (table if table is not None else _fake_census_table()).items()}
    assert_not_test(list(table), where="synthetic census")
    own_ids = get_task().template_pattern is not None
    shapes = _task_answer_shapes(table) if own_ids else {}
    reps: dict[str, Any] = {}
    for rep in range(max((len(v) for v in table.values()), default=0)):
        rows = []
        for template, values in table.items():
            if rep >= len(values):
                continue
            s = values[rep]
            tier = template.split("-")[0]
            klass = "ok" if s >= 1.0 else ("divergent" if tier == "II" else "consensus-wrong")
            shape = shapes.get(template) if own_ids else ("collection" if tier == "III" else "scalar")
            rows.append({"case_id": template, "S": s, "C": 15000.0, "model_calls": 10,
                         "infra": None, "prompt_tokens": 5000, "completion_tokens": 10000,
                         "diag": {"S": s, "failure_class": klass, "answer_shape": shape}})
        reps[str(rep)] = {"rows": rows}
    return {"config": {"agents": _fake_census_agents()}, "arms": {"v1": {"repeats": reps}}}


def _task_answer_shapes(templates: Iterable[str]) -> dict[str, str | None]:
    """Answer shape of each template's instance under the active task (None
    where the task resolves no instance)."""

    resolve = get_task().instance_path
    out: dict[str, str | None] = {}
    for template in templates:
        shape = None
        if resolve is not None:
            try:
                from queenbee.evaluate import load_instance_file
                from queenbee.evo.diagnosis import instance_answer_shape

                path = resolve(template)
                shape = None if path is None else instance_answer_shape(load_instance_file(path))
            except Exception:  # noqa: BLE001 - a synthetic field only
                shape = None
        out[template] = shape
    return out


def _fake_census_agents() -> int:
    """Team size of the synthetic census rows: 5, or the smallest ``o`` rung
    of a task with its own rungs."""

    rungs = get_task().rungs
    if rungs is None:
        return 5
    return min((int(n) for r, n in rungs.items() if r.startswith("o")), default=5)


# --------------------------------------------------------------------------- #
# Dependencies (injectable for tests)
# --------------------------------------------------------------------------- #


@dataclass
class EvoDeps:
    """``execute``: one execution, ``req -> row`` (the shape of
    ``queenbee.evo.common.real_execute``); None = the active task's executor,
    else ``real_execute`` (fake worker provider in a ``--fake`` run).
    ``planner_client``: the planner (``complete(prompt, ...)``; None = fake or
    real by config).  ``seed_source`` / ``t_templates``: None = the active
    task's, else the built-in ones.  ``racer_factory(run) -> race.Racer``
    overrides the racer."""

    execute: Callable[[Mapping[str, Any]], dict] | None = None
    planner_client: Any = None
    seed_source: Callable[[], str] | None = None
    t_templates: Callable[[Sequence[str]], list] | None = None
    racer_factory: Callable[["EvoRun"], Any] | None = None


def _default_seed_source() -> str:
    """The active task's seed program, else the Silo-Bench seed program
    (``queenbee.evo.seed``)."""

    seed = get_task().seed_source
    if seed is not None:
        return seed()
    from queenbee.evo.seed import evo_seed_source

    return evo_seed_source()


def missing_settings(cfg: EvoConfig, deps: EvoDeps | None = None) -> list[str]:
    """What a real (non ``--fake``) run still lacks: pinned inputs, model
    names and API keys (the parts injected through ``deps`` need none).
    A task that resolves its own instances needs no ladder manifest, one
    with a default split no split file.  Empty when the run may start."""

    if cfg.fake:
        return []
    deps = deps or EvoDeps()
    task = get_task()
    real_workers = deps.execute is None and deps.racer_factory is None
    out: list[str] = []
    if real_workers:
        needed = [("--units-from", cfg.unit_sources)]
        if task.instance_path is None:
            needed.append(("--ladder-manifest", cfg.ladder_manifests))
        if task.default_split is None:
            needed.append(("--split-file", cfg.split_file))
        out += [flag for flag, value in needed if not value]
    if not cfg.worker_model:
        out.append(f"--worker-model (or ${WORKER_MODEL_ENV})")
    keys: list[str] = []
    if deps.planner_client is None:
        if not cfg.planner_model:
            out.append(f"--planner-model (or ${PLANNER_MODEL_ENV})")
        keys.append(cfg.planner_api_key_env or DEFAULT_API_KEY_ENV)
    if real_workers:
        keys.append(DEFAULT_API_KEY_ENV)
    out += [f"${key}" for key in dict.fromkeys(keys) if not os.environ.get(key)]
    return out


def default_split(units: Any, split_seed: int) -> Any:
    """The T / V split of ``split_seed`` when no split file is given: the
    active task's ``default_split``, else ``evo.units.make_split``."""

    from queenbee.evo.units import TemplateSplit, make_split

    hook = get_task().default_split
    if hook is None:
        return make_split(units, int(split_seed))
    data = dict(hook(int(split_seed)))
    assert_not_test(list(data.get("T") or []) + list(data.get("V") or []), where="task split")
    return TemplateSplit(str(data.get("name") or f"s{int(split_seed)}"),
                         tuple(data["T"]), tuple(data["V"]))


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


#: Mint statuses after which a slot is never minted again.
FINAL_MINT_STATUSES = frozenset({"ok", "mint_failed", "planner_error_final"})


class EvoRun:
    """One evolution run rooted at ``cfg.root`` (see the module docstring)."""

    def __init__(self, cfg: EvoConfig, deps: EvoDeps | None = None) -> None:
        from queenbee.evo.budget import BudgetMeter

        active = get_task().name
        if cfg.task != active:
            raise EvoStateError(f"the config names task {cfg.task!r} but {active!r} is "
                                "active (activate it with queenbee.tasks.set_task)")
        self.cfg = cfg
        self.deps = deps or EvoDeps()
        self.root = cfg.root
        # card options are process-wide switches: the cards are built inside
        # the executors
        if cfg.local_only_diag:
            from queenbee.evo.diagnosis import LOCAL_ONLY_ENV

            os.environ[LOCAL_ONLY_ENV] = "1"
        if cfg.attempt_diag:
            from queenbee.evo.diagnosis import ATTEMPT_ENV

            os.environ[ATTEMPT_ENV] = "1"
        if cfg.preserve_dups:  # multiset raw_kept% + dup_lost on the cards
            from queenbee.evo.credit import MULTISET_ENV

            os.environ[MULTISET_ENV] = "1"
        self._check_inputs()
        self._check_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.flags = cfg.flags
        self.policy = parent_policy(self.flags)
        self._events_lock = threading.Lock()
        self._mints_lock = threading.Lock()
        self._planner_lock = threading.Lock()
        self._marker_lock = threading.Lock()
        self._planner: Any = None
        self._pending: dict[str, Future] = {}
        self._t_statements: list[dict[str, str]] | None = None
        self._forbidden: dict[str, list[Any]] = {}
        self._v1_fp_key: str | None = None
        self._r1_cache: dict[int, dict[str, Any]] = {}
        self.guard: OutageGuard | None = None
        self.meter = BudgetMeter(self.root / "budget_meter.json", cap_executions=0, fresh=True)
        self._setup_units()
        self._setup_racer()
        self._setup_memory()
        self._rebuild_meter()
        self._planner_pool = ThreadPoolExecutor(max_workers=int(cfg.planner_concurrency),
                                                thread_name_prefix="evo-planner")

    # -- root / config ------------------------------------------------------- #
    def _check_inputs(self) -> None:
        """Real runs must name their inputs (no runtime glob may decide the
        units, split or dS baselines), their models and their API keys --
        checked before anything is written to the root.  A task's own S0
        vocabulary must load, in every run."""

        cfg = self.cfg
        missing = missing_settings(cfg, self.deps)
        if missing:
            raise EvoStateError(
                "a real run needs " + ", ".join(missing)
                + " (generate the split once with --write-split; --fake runs offline)")
        task = get_task()
        if task.wi_vocabulary is not None:  # without it every proposal fails S0
            try:
                task.wi_vocabulary()
            except Exception as exc:  # noqa: BLE001 - reported before the root is written
                raise EvoStateError(f"task {task.name!r}: the S0 work-instruction vocabulary "
                                    f"cannot be built ({type(exc).__name__}: {exc})") from exc
        if cfg.fake or self.deps.execute is not None or self.deps.racer_factory is not None:
            return
        for path in list(cfg.unit_sources) + list(cfg.ladder_manifests) + [cfg.split_file]:
            if path is not None and not Path(path).is_file():
                raise EvoStateError(f"input file not found: {path}")

    def _check_root(self) -> None:
        path = self.root / "config.json"
        frozen = self.cfg.frozen()
        if path.is_file():
            if not self.cfg.resume:
                raise EvoStateError(f"{self.root} already holds a run; pass --resume to continue it")
            old = _read_json(path, {}) or {}
            diff = {k: (old.get(k), v) for k, v in frozen.items() if old.get(k) != v}
            # the optional behaviours (and a task other than the default) are in
            # frozen() only when set: dropping one on a resume is a config change too
            diff.update({k: (old[k], None) for k in (*OPTION_KEYS, "task")
                         if k in old and k not in frozen})
            if diff:
                raise EvoStateError(f"resume with a different config: {diff}")
        else:
            if self.root.exists() and any(self.root.iterdir()) and not self.cfg.resume:
                raise EvoStateError(f"{self.root} is not empty; pass --resume or use a new root")
            _write_json(path, frozen | {"created_at": _now()})

    def event(self, name: str, **data: Any) -> None:
        _append_jsonl(self.root / "events.jsonl", {"at": _now(), "event": name, **data},
                      self._events_lock)

    # -- units / split / census ------------------------------------------------ #
    def _manifests(self) -> list[Path]:
        """``--ladder-manifest``, else the run's own x5 ladder (the 18 dev
        templates, default salt: deterministic and offline; a census built on
        another ladder makes ``load_units`` refuse the instance mismatch)."""

        manifests = list(self.cfg.ladder_manifests)
        if manifests or get_task().dev_ids is not None:
            # the built-in ladder covers the Silo-Bench templates only
            return manifests
        built = self.root / "ladder" / "manifest_x5.json"
        if not built.is_file():
            from queenbee.evo.ladder import build_ladder

            build_ladder(None, rung="x5", n_agents=5, out_dir=self.root / "ladder",
                         benchmarks_dir=self.cfg.benchmarks_dir)
        return [built]

    def _setup_units(self) -> None:
        from queenbee.evo.units import (
            TemplateSplit, load_units, near_miss_units_from_traces,
            structural_units_from_traces,
        )

        cfg = self.cfg
        sources = cfg.resolved_unit_sources()
        manifests = self._manifests()
        # Answer-free trace evidence (dev traces only): a unit where at least
        # half of the failing traces are precision / format near-misses is
        # irreducible; a zero-S unit where most failures are organizational is
        # frontier, not dead.
        split_path = self.root / "state" / "split.json"
        frozen = _read_json(split_path)
        if isinstance(frozen, dict) and "near_miss_units" in frozen and "structural_units" in frozen:
            # a started run keeps the trace evidence it recorded (the trace dirs
            # are not pinned by content: a resume must never re-class a unit mid-run)
            near = {str(u): 1 for u in frozen.get("near_miss_units") or []}
            struct = {str(u): 1 for u in frozen.get("structural_units") or []}
        else:
            evidence = bool(cfg.trace_dirs) and get_task().trace_unit_evidence
            near = near_miss_units_from_traces(cfg.trace_dirs) if evidence else {}
            struct = structural_units_from_traces(cfg.trace_dirs) if evidence else {}
            for t in cfg.irreducible_templates:  # benchmark rounding floor
                for rung in _common.task_rungs():
                    near[f"{t}@{rung}"] = max(int(near.get(f"{t}@{rung}", 0) or 0), 99)
                    struct.pop(f"{t}@{rung}", None)
        self.near_miss_units = dict(near)
        self.structural_units = dict(struct)
        units = load_units(sources, v1_arm=cfg.v1_arm, manifests=manifests,
                           benchmarks_dir=cfg.benchmarks_dir, near_miss_units=near,
                           structural_units=struct)
        split_path = self.root / "state" / "split.json"
        frozen = _read_json(split_path)
        if isinstance(frozen, dict) and frozen.get("T"):
            split = TemplateSplit(str(frozen.get("name") or "frozen"), tuple(frozen["T"]),
                                  tuple(frozen["V"]))
        else:
            if cfg.split_file is not None:
                data = json.loads(Path(cfg.split_file).read_text())
                assert_not_test(list(data.get("T") or []) + list(data.get("V") or []),
                                where="split file")
                split = TemplateSplit(str(data.get("name") or Path(cfg.split_file).stem),
                                      tuple(data["T"]), tuple(data["V"]))
            else:
                split = default_split(units, int(cfg.split_seed))
            _write_json(split_path, split.to_dict() | {
                "frozen_at": _now(), "split_seed": int(cfg.split_seed),
                "unit_sources": [_source_label(p) for p in sources],
                "manifests": [str(p) for p in manifests],
                "near_miss_units": sorted(near),
                "structural_units": sorted(struct),
            })
        assert_not_test(list(split.T) + list(split.V), where="split")
        if set(split.T) & set(split.V):
            raise LeakageGuardError("split: T and V overlap")
        manifest_paths: dict[str, Path] = {}
        for manifest in manifests:
            manifest_paths.update(_common.load_ladder_manifest(manifest))
        self.pool = UnitPool(units, split, benchmarks_dir=cfg.benchmarks_dir,
                             manifest_paths=manifest_paths)
        self._load_pool_extras()
        self._load_census(sources)

    def _load_census(self, sources: Sequence[Any]) -> None:
        """The seed program's census rows by unit; a T row whose trace is
        found gets its diag + credit fields recomputed from it.  T rows become
        the racer's external rows; V rows are read only by final selection."""

        from queenbee.evo.units import load_eval_rows

        rows = load_eval_rows(sources)
        v1_rows = rows.arms.get(self.cfg.v1_arm, {})
        dirs = list(self.cfg.trace_dirs)
        index = _common.TraceIndex(dirs)
        task_seed = get_task().seed_source  # the program the census ran
        plain = _common.v1_source() if task_seed is None else task_seed()
        plain_sha12 = _sha256(plain)[:12]
        self.census_T: dict[str, list[dict[str, Any]]] = {}
        self.census_V: dict[str, list[dict[str, Any]]] = {}
        for unit, items in v1_rows.items():
            side = self.pool.side(unit)
            if side is None:
                continue
            enriched = []
            for row in items:
                row = dict(row)
                trace = index.match(str(row.get("case_id")), plain_sha12, row)
                if trace is not None and side == "T":
                    self._attach_trace(row, trace, unit, plain)
                enriched.append(row)
            (self.census_T if side == "T" else self.census_V)[unit] = enriched

    def _attach_trace(self, row: dict[str, Any], trace: Mapping[str, Any], unit: str,
                      source: str) -> None:
        from queenbee.evo.credit import trace_row

        try:
            traced = trace_row(trace, self.pool.instance(unit), source=source,
                               goal=_common.task_goal())
        except LeakageGuardError:
            raise
        except Exception as exc:  # noqa: BLE001 - enrichment is advisory
            row["enrich_error"] = type(exc).__name__
            return
        if isinstance(traced.get("diag"), Mapping):  # this code's classifier, not the census file's
            row["diag"] = dict(traced["diag"])
        if traced.get("credit") is not None:
            row["credit"] = traced["credit"]

    def _curriculum_unit(self, unit_id: str, klass: str = "unresolved",
                         instance_sha256: str | None = None) -> Any:
        from queenbee.evo.units import Unit

        return Unit(unit_id=unit_id, template=template_of(unit_id), rung=rung_of(unit_id),
                    n_agents=self.pool.n_agents(unit_id) if hasattr(self, "pool")
                    else int(_common.task_rungs().get(rung_of(unit_id)) or 5),
                    instance_path=str(self.pool.path(unit_id)) if hasattr(self, "pool") else None,
                    instance_sha256=instance_sha256 or None, klass=klass)

    def _load_pool_extras(self) -> None:
        """``state/pool.json`` extras: curriculum promotions (class
        ``unresolved``)."""

        pool_state = _read_json(self.root / "state" / "pool.json", {}) or {}
        for uid, info in dict(pool_state.get("extra") or {}).items():
            self.pool.extra[uid] = dict(info)
            if uid not in self.pool.units:
                self.pool.units[uid] = self._curriculum_unit(uid)

    def _inherit_programs(self) -> None:
        """Inherited programs (``cfg.inherit_programs``), once per root
        (``state/inherit.json``; a resume re-uses it).  Each inherited genome
        is re-hosted on v1's host code (the text around v1's genome region),
        must name no TEST / V template, is registered with the racer (id
        ``inh_<name>``) and the archive (gen -1, parent v1), and gets one paid
        rep (purpose ``inherit``) on every T frontier unit, charged to the
        evolution budget.  The measured units count as confirmed, so the
        program is an eligible parent / specialist from generation 0 on."""

        if not self.cfg.inherit_programs:
            return
        path = self.root / "state" / "inherit.json"
        marker = _read_json(path) or {}
        if marker.get("done"):
            return
        from queenbee.program.genome import genome_region

        host = self.v1.source
        g_v1 = genome_region(host)
        if not g_v1 or host.count(g_v1) != 1:
            raise EvoStateError("inherit: cannot locate v1's genome region")
        pre, post = host[: host.find(g_v1)], host[host.find(g_v1) + len(g_v1):]
        forbidden = sorted(set(_common.task_test_ids()) | set(self.pool.split.V))
        pids: list[str] = []
        info: dict[str, Any] = {}
        for name, src_path in self.cfg.inherit_programs:
            raw = Path(src_path).read_text(encoding="utf-8")
            genome = genome_region(raw)
            if not genome:
                raise EvoStateError(f"inherit {name}: no genome region in {src_path}")
            hits = [t for t in forbidden if re.search(rf"(?<![A-Za-z0-9-]){re.escape(t)}(?![0-9])", genome)]
            if hits:
                raise EvoStateError(f"inherit {name}: genome names TEST / V templates {hits}")
            source = pre + genome + post
            pid = f"inh_{name}"
            program = self.racer.register_program(pid, source, arm=self.cfg.arm, meta={
                "gen": -1, "kind": "inherited", "inherited_from": str(src_path),
                "parent_ids": [self.cfg.v1_arm]})
            self.archive.add_program(pid, gen=-1, arm=self.cfg.arm, parent_ids=[self.cfg.v1_arm],
                                     brief_kind="inherited", fp=program.fp,
                                     source_sha256=program.source_sha256, reached_r2=True)
            pids.append(pid)
            info[pid] = {"name": name, "path": str(src_path), "source_sha256": program.source_sha256,
                         "rehosted": source != raw}
        units = list(self.t_frontier())
        requests = [(pid, unit, "inherit", 0) for pid in pids for unit in units]
        rows = (self.racer.run_batch("inherit:seed", requests,
                                     cap_exec_eq=self.cfg.evolution_budget)
                if requests else [])
        self.add_rows(list(rows))
        for pid in pids:
            measured = [u for u in units if self.archive.unit_stats(pid).get(u)]
            self.archive.confirmed[pid] = sorted(set(self.archive.confirmed.get(pid) or []) | set(measured))
            info[pid]["units"] = {u: (self.archive.unit_stats(pid).get(u) or {}).get("S") for u in units}
        _write_json(path, {"done": True, "at": _now(), "programs": info, "units": units,
                           "spent_evolution": self.spent("evolution")})
        self.save_memory("inherit")
        self.event("inherited", programs=info, units=units, spent=self.spent("evolution"))

    def _race_config(self) -> _race.RaceConfig:
        cfg = self.cfg
        return _race.RaceConfig(
            fake=cfg.fake, worker_model=cfg.worker_model,
            python_max_rounds=cfg.python_max_rounds, parallel_cases=cfg.parallel_cases,
            request_timeout=cfg.request_timeout, max_infra_reruns=cfg.max_infra_reruns,
            infra_backoff_s=cfg.infra_backoff_s, traces=cfg.traces,
            vacuous_guard=bool(cfg.vacuous_guard),
        )

    # -- racer + memory ------------------------------------------------------------ #
    def _setup_racer(self) -> None:
        cfg = self.cfg
        if self.deps.racer_factory is not None:
            self.racer = self.deps.racer_factory(self)
        else:
            rc = self._race_config()
            self.guard = OutageGuard(
                self.deps.execute or _common.task_execute, wait_s=cfg.outage_wait_s,
                backoff_s=cfg.outage_backoff_s, marker_path=self.root / "exec_inflight.jsonl",
                on_event=self.event,
            )
            self.racer = _LoopRacer(
                self.root, config=rc, resolver=self.pool, execute=self.guard,
                meter=self.meter, thresholds=cfg.thresholds_file, split=self.pool.split,
            )
        self.th = self.racer.thresholds
        seed = (self.deps.seed_source or _default_seed_source)()
        self.v1 = self.racer.register_program("v1", seed, arm="seed",
                                              meta={"gen": -1, "parent_ids": [], "role": "seed"})
        for unit, rows in sorted(self.census_T.items()):
            self.racer.register_external_rows("v1", unit, rows, label="census")

    def _setup_memory(self) -> None:
        data = _read_json(self.root / "state" / "memory.json")
        T = list(self.pool.split.T)
        if isinstance(data, dict) and data.get("archive"):
            self.archive = _memory.Archive.from_dict(data["archive"])
            self.ledger = _memory.Ledger.from_dict(data.get("ledger") or {})
            self.scheduler = _memory.Scheduler.from_dict(data["scheduler"])
        else:
            k = int(self.cfg.K)
            self.archive = _memory.Archive(v1_id="v1", t_templates=T)
            self.ledger = _memory.Ledger()
            self.scheduler = _memory.Scheduler(self.flags, arm=self.cfg.arm, k=k,
                                               n_target=max(1, k - 1) if k > 1 else 1)
        # an option of the run config (not in memory.json): set it on every load
        self.scheduler.explore_on_lockin = bool(self.cfg.explore_on_lockin)
        self.archive.add_program("v1", gen=-1, arm="seed", fp=self.v1.fp,
                                 source_sha256=self.v1.source_sha256)
        # idempotent re-derivation from the logs / generation files: a kill
        # between a generation file and memory.json loses nothing
        self.archive.add_rows(self._t_rows(self.racer.log.records))
        gen = 0
        while True:
            state = self.gen_state(gen)
            if not state:
                break
            for brief, slot in zip(state.get("briefs") or [], state.get("slots") or []):
                self._archive_slot(gen, brief, slot)
            gen += 1

    def save_memory(self, where: str) -> None:
        _write_json(self.root / "state" / "memory.json", {
            "archive": self.archive.to_dict(), "ledger": self.ledger.to_dict(),
            "scheduler": self.scheduler.to_dict(), "at": _now(), "where": where,
        })

    def _t_rows(self, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """ExecRows of T units (the archive never holds a V row)."""

        out = []
        for rec in records:
            unit = rec.get("unit_id")
            if not unit or self.pool.side(str(unit)) != "T":
                continue
            out.append(self.racer.exec_row(rec) if "row" in rec else dict(rec))
        return out

    def add_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self.archive.add_rows([r for r in rows if r.get("unit_id")
                               and self.pool.side(str(r["unit_id"])) == "T"])

    def _rebuild_meter(self) -> None:
        """Re-derive the meter from the logs (a kill between an append and
        its charge can never leave them disagreeing)."""

        for rec in list(self.racer.log.records):
            if rec.get("external"):
                continue
            row = rec.get("row") or {}
            phase = _race.PURPOSE_PHASE.get(str(rec.get("purpose")), "train_stage1")
            self.meter.charge_rows(phase, [row], cached=bool(rec.get("cached")))
            if not rec.get("cached") and not _race.is_infra(row) and int(rec.get("exec_eq") or 1) > 1:
                self.meter.add_counter("exec_eq_surcharge", int(rec.get("exec_eq")) - 1)
        # worker executions lost in flight to a kill -9 (re-run on resume;
        # their usage is unknown): counted, never silently dropped
        lost_exec = lost_inflight(self.root / "exec_inflight.jsonl")
        self.meter.add_counter("lost_inflight_executions", len(lost_exec))
        mints = self.root / "mints.jsonl"
        call_ids: set[str] = set()
        if mints.is_file():
            seen: set[str] = set()
            for line in mints.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                token = str(rec.get("call_id") or line.strip())  # one line per planner call
                if token in seen:
                    continue
                seen.add(token)
                if rec.get("call_id"):
                    call_ids.add(str(rec["call_id"]))
                self.meter.charge_planner(rec.get("usage") or [], round_index=rec.get("gen"))
        # planner calls started but never recorded (kill -9 mid-call): one
        # unknown-usage call each
        lost = [r for r in _jsonl_records(self.root / "planner_inflight.jsonl")
                if r.get("event") == "start" and str(r.get("id")) not in call_ids]
        self.meter.charge_planner([{"call": "lost_inflight", "model": self.cfg.planner_model}
                                   for _ in lost], round_index=None)
        self.meter.add_counter("lost_inflight_planner_calls", len(lost))
        self.meter.save()

    # -- budget ------------------------------------------------------------------------ #
    def _paid(self) -> list[dict[str, Any]]:
        return [r for r in list(self.racer.log.records)
                if not r.get("external") and not r.get("cached")
                and not _race.is_infra(r.get("row") or {})]

    def spent(self, scope: str = "evolution") -> float:
        """Scored paid execution-equivalents: ``evolution`` (every purpose
        but :data:`FINAL_PURPOSES`), ``final`` (V selection and the
        ``--final-coverage-topup`` T reps) or ``total``."""

        total = 0.0
        for rec in self._paid():
            # V selection (and the --final-coverage-topup T reps) are paid
            # from the final reserve
            final = rec.get("purpose") in FINAL_PURPOSES
            if (scope == "evolution" and final) or (scope == "final" and not final):
                continue
            total += float(rec.get("exec_eq") or 1)
        return total

    def spent_upto(self, gen: int, stage: str) -> float:
        """Evolution spend of the batches up to (gen, stage) inclusive."""

        cut = (int(gen), STAGES.index(stage))
        total = 0.0
        for rec in self._paid():
            key = batch_key(rec.get("batch_id"))
            if key is not None and key <= cut:
                total += float(rec.get("exec_eq") or 1)
        return total

    def evo_remaining(self) -> float:
        return self.cfg.evolution_budget - self.spent("evolution")

    # -- units in play ----------------------------------------------------------------- #
    def current_best(self) -> str:
        """Best shrunk T score (> 0) among programs with >= 3 T units, else v1."""

        top = self.archive.top_by_shrunk(1, min_units=MIN_FINAL_UNITS)
        if top and (self.archive.shrunk_score(top[0]) or 0.0) > EPS:
            return top[0]
        return "v1"

    def _S_best(self, best: str, unit: str) -> float | None:
        st = self.archive.unit_stats(best).get(unit) or self.archive.unit_stats("v1").get(unit)
        return None if st is None else float(st["S"])

    def t_frontier(self) -> list[str]:
        """Census frontier T units + curriculum units the current best has
        not saturated (the units in play)."""

        out = list(self.pool.census_frontier_T())
        best = self.current_best()
        for unit in self.pool.extra:
            if self.pool.side(unit) != "T" or unit in out:
                continue
            s = self._S_best(best, unit)
            if s is not None and s < 1.0 - EPS:
                out.append(unit)
        return sorted(dict.fromkeys(out), key=_unit_order)

    def shapes(self) -> dict[str, str | None]:
        return {u: self.pool.shape(u) for u in self.pool.pool_T()}

    # -- generation state files --------------------------------------------------------- #
    def gen_path(self, gen: int) -> Path:
        return self.root / "state" / f"generation_{int(gen):03d}.json"

    def gen_state(self, gen: int) -> dict[str, Any] | None:
        return _read_json(self.gen_path(gen))

    def save_gen(self, gen: int, state: Mapping[str, Any]) -> None:
        _write_json(self.gen_path(gen), dict(state) | {"updated_at": _now()})

    # -- prompts -------------------------------------------------------------------------- #
    def t_statements(self) -> list[dict[str, str]]:
        if self._t_statements is None:
            T = list(self.pool.split.T)
            assert_not_test(T, where="t_templates")
            task_templates = get_task().t_templates
            if self.deps.t_templates is not None:
                self._t_statements = list(self.deps.t_templates(T))
            elif task_templates is not None:
                self._t_statements = list(task_templates(T))
            else:
                self._t_statements = t_template_statements(T, benchmarks_dir=self.cfg.benchmarks_dir,
                                                           n_agents=_common.task_mint_n_agents())
        return self._t_statements

    def forbidden_values_T(self) -> list[Any]:
        from queenbee.evo.credit import forbidden_values_of

        values: list[Any] = []
        for unit in self.pool.pool_T():
            if unit not in self._forbidden:
                try:
                    self._forbidden[unit] = list(forbidden_values_of(self.pool.instance(unit)))
                except (OSError, EvoStateError, ValueError):
                    self._forbidden[unit] = []
            values.extend(self._forbidden[unit])
        return values

    def build_prompt(self, brief: Mapping[str, Any], t_units: Sequence[str]) -> tuple[str, int]:
        """``(prompt, n_redactions)``.  A TEST / V id anywhere, or a held-out
        template in a structured field, is a genuine leak: the run stops
        (:class:`LeakageGuardError`).  A T ground-truth VALUE in an evidence
        section (e.g. a token count that equals an answer) is redacted and
        the prompt re-audited, so a false positive never drops a proposal
        (and never more of one arm's than another's)."""

        from queenbee.evo.prompt import audit_evo_prompt, build_evo_prompt_blocks
        from queenbee.program.budgets import PythonRunBudgets

        needed = [str(p) for p in brief.get("parent_ids") or []]
        needed += [str(e["program_id"]) for e in brief.get("exemplars") or []]
        sources = {pid: self.racer.program(pid).source for pid in dict.fromkeys(needed)}
        # without the archive the failure map / rows are the PARENT's view (its
        # rows, else v1's census rows), never other programs' (memory.prompt_state)
        state = _memory.prompt_state(
            archive=self.archive, ledger=self.ledger, flags=self.flags, sources=sources,
            t_templates=self.t_statements(), forbidden_case_ids=list(self.pool.split.V),
            units=list(t_units),
            budgets=PythonRunBudgets.for_rounds(self.cfg.python_max_rounds,
                                                n_agents=_common.task_mint_n_agents()),
            parent_id=needed[0] if needed else None,
        )
        if self.cfg.preserve_dups:  # the API card's forwarded-raw-data rule
            state["preserve_dups"] = True
        view = arm_brief(brief, self.flags)
        held_out = list(self.pool.split.V)
        values = self.forbidden_values_T()
        try:
            return build_evo_prompt(state, view, self.flags, forbidden_case_ids=held_out,
                                    forbidden_values=values), 0
        except LeakageGuardError:
            raise
        except RuntimeError as exc:
            first = exc
        where = f"prompt {brief.get('slot_id') or brief.get('brief_id')}"
        try:
            blocks = build_evo_prompt_blocks(state, view, self.flags, forbidden_case_ids=held_out)
        except RuntimeError as exc:  # a held-out template reached a structured field
            raise LeakageGuardError(f"LEAKAGE_GUARD ({where}): {exc}") from exc
        text = "\n".join(body.rstrip("\n") + "\n" for _name, body in blocks)
        ids = sorted(set(_common.task_test_ids()) | set(held_out))
        hits = audit_evo_prompt(text, forbidden_case_ids=ids, forbidden_values=values)
        id_hits = [h for h in hits if "case_id" in h]
        if id_hits or not hits:
            raise LeakageGuardError(f"LEAKAGE_GUARD ({where}): {id_hits or first}") from first
        text, n = redact_evidence_values(text, values,
                                         [h["value_index"] for h in hits if "value_index" in h])
        left = audit_evo_prompt(text, forbidden_case_ids=ids, forbidden_values=values)
        if left or n == 0:
            raise LeakageGuardError(f"LEAKAGE_GUARD ({where}): {len(left)} hit(s) after redaction")
        return text, n

    # -- planning ------------------------------------------------------------------------- #
    def plan_generation(self, gen: int, *, after: str) -> dict[str, Any]:
        """Briefs + prompts of generation ``gen`` from the current archive /
        ledger (written to disk before any planner call and re-used on a
        resume; the scheduler's parent uses are committed idempotently)."""

        existing = self.gen_state(gen)
        if existing and existing.get("briefs") is not None:
            self.scheduler.commit(existing["briefs"], self.archive)
            return existing
        t_units = self.t_frontier() or self.pool.pool_T()
        briefs = self.scheduler.briefs(int(gen), archive=self.archive, ledger=self.ledger,
                                       t_units=t_units, shapes=self.shapes())
        prompt_dir = self.root / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for brief in briefs:
            brief["slot_id"] = f"g{int(gen):03d}_s{int(brief['slot'])}"
            units = list(brief.get("target_units") or [])
            assert_not_test(units, where="brief")
            for unit in units:
                self.pool.assert_side(unit, "T", where="brief")
            text, redacted = self.build_prompt(brief, t_units)  # a genuine leak raises
            if redacted:
                brief["prompt_redacted"] = int(redacted)
                self.event("prompt_redacted", gen=gen, slot_id=brief["slot_id"], n=int(redacted))
            path = prompt_dir / f"{brief['slot_id']}_prompt.txt"
            path.write_text(text)
            brief["prompt_path"] = str(path.relative_to(self.root))
            brief["prompt_sha256"] = _sha256(text)
        state = {"gen": int(gen), "briefs": briefs, "planned_after": after, "t_units": t_units,
                 "planned_at": _now(), "done": False,
                 "flags": dataclasses.asdict(self.flags), "policy": self.policy}
        self.save_gen(gen, state)
        self.scheduler.commit(briefs, self.archive)
        self.save_memory(f"g{gen} planned")
        self.event("gen_planned", gen=gen, after=after,
                   briefs=[{k: b.get(k) for k in ("slot_id", "kind", "target_units", "target_class",
                                                  "parent_ids", "fallback")} for b in briefs])
        return state

    # -- minting ------------------------------------------------------------------------- #
    def planner_client(self) -> Any:
        if self.deps.planner_client is not None:
            return self.deps.planner_client
        with self._planner_lock:
            if self._planner is None:
                if self.cfg.fake:
                    self._planner = FakeEvoPlanner()
                else:
                    deadline = self.cfg.effective_planner_deadline() or 0.0
                    http_timeout = max(float(self.cfg.request_timeout), deadline + 60.0)
                    wall = max(float(self.cfg.request_timeout) * 1.2, deadline + 120.0)
                    saved = os.environ.get("OPENAI_TIMEOUT")
                    os.environ["OPENAI_TIMEOUT"] = str(int(http_timeout))
                    try:
                        self._planner = self._real_planner_client(wall)
                    finally:
                        if saved is None:
                            os.environ.pop("OPENAI_TIMEOUT", None)
                        else:
                            os.environ["OPENAI_TIMEOUT"] = saved
        return self._planner

    def _real_planner_client(self, wall_s: float) -> Any:
        """The planner's OpenAI-compatible client,
        ConnectRetry(Limiter(Timeout(OpenAI(retries=0)))), on
        ``--planner-base-url`` / ``--planner-api-key-env`` when given, else
        on the SDK environment (``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``)."""

        from queenbee.program.clients import _build_arm_client

        return _build_arm_client(
            "openai", timeout_s=wall_s, reasoning_effort=self.cfg.planner_effort,
            connect_attempts=int(self.cfg.connect_attempts),
            connect_backoff_s=self.cfg.connect_backoff_s,
            base_url=self.cfg.planner_base_url, api_key_env=self.cfg.planner_api_key_env,
        )

    def _slot_result(self, slot_id: str) -> dict[str, Any] | None:
        return _read_json(self.root / "mints" / slot_id / "result.json")

    def _slot_prev(self, slot_id: str) -> dict[str, Any] | None:
        """The slot's latest result: ``result.json``, else (a kill during a
        retry, after the previous try was moved aside) the newest
        ``<slot>.try<n>/result.json`` -- so a resumed mint continues the try
        and error counters instead of restarting them."""

        prev = self._slot_result(slot_id)
        if isinstance(prev, dict):
            return prev
        workdir = self.root / "mints" / slot_id
        latest = None
        n = 1
        while workdir.with_name(f"{workdir.name}.try{n}").exists():
            rec = _read_json(workdir.with_name(f"{workdir.name}.try{n}") / "result.json")
            if isinstance(rec, dict):
                latest = rec
            n += 1
        return latest

    def _planner_mark(self, record: Mapping[str, Any]) -> None:
        _append_jsonl(self.root / "planner_inflight.jsonl", dict(record) | {"at": _now()},
                      self._marker_lock)

    def mint_slot(self, gen: int, brief: Mapping[str, Any]) -> dict[str, Any]:
        """One proposal (planner thread).  Resumable: a final result is
        returned as is; a planner deadline is re-minted once.  Outage errors
        (transport failures, HTTP 5xx, timeouts; :func:`is_outage_error`) are
        retried up to ``MAX_PLANNER_ERROR_RETRIES`` times per session, then
        the slot returns ``planner_outage`` (NOT final: the main thread stops
        the run resumable) -- an outage never costs a proposal; other errors
        are retried as often and then make the slot final, and so does any
        error past ``MAX_PLANNER_ERROR_TOTAL`` over all sessions.  Every
        planner call gets a ``call_id`` (start marker before the call, the
        mints.jsonl record after it) so the meter counts each call once and
        a call lost to a kill -9 as one unknown-usage call."""

        from queenbee.program.mint import (
            MintFailed,
            build_mint_runtime,
            build_planner_request,
            mint_python_challenger,
        )
        from queenbee.program.budgets import PythonRunBudgets

        slot_id = str(brief["slot_id"])
        workdir = self.root / "mints" / slot_id
        prev = self._slot_prev(slot_id)
        if isinstance(prev, dict) and prev.get("status") in FINAL_MINT_STATUSES \
                and (workdir / "result.json").is_file():
            return prev
        base: dict[str, Any] = {"slot_id": slot_id, "gen": int(gen), "brief_id": brief["brief_id"],
                                "kind": brief["kind"], "parent_ids": list(brief["parent_ids"]),
                                "started_at": _now(), "planner_model": self.cfg.planner_model}
        prompt = (self.root / brief["prompt_path"]).read_text()
        parent = self.racer.program(str(brief["parent_ids"][0]))
        n, goal = _common.task_mint_n_agents(), _common.task_goal()
        budgets = PythonRunBudgets.for_rounds(self.cfg.python_max_rounds, n_agents=n)
        request = build_planner_request(n_agents=n, goal=goal, worker_contract=WORKER_CONTRACT)
        runtime = build_mint_runtime(
            llm_provider=self.cfg.llm_provider, worker_model=self.cfg.worker_model, goal=goal,
            worker_contract=WORKER_CONTRACT, max_parallel_agents=n,
            request_timeout=self.cfg.request_timeout, n_agents=n, budgets=budgets,
        )
        prev = prev or {}
        errors_total = int(prev.get("planner_error_total", prev.get("planner_error_attempts")) or 0)
        deadlines = int(prev.get("deadline_attempts") or 0)
        tries = int(prev.get("try") or 0)
        session_outages = 0
        session_errors = 0
        while True:
            tries += 1
            if workdir.is_dir():  # a stale reply must never be read as this try's
                n = 1
                while workdir.with_name(f"{workdir.name}.try{n}").exists():
                    n += 1
                os.replace(workdir, workdir.with_name(f"{workdir.name}.try{n}"))
            usage: list[dict[str, Any]] = []
            hypothesis: dict[str, Any] = {}
            started = time.time()
            call_id = uuid.uuid4().hex
            record = dict(base) | {"call_id": call_id}
            status: str
            self._planner_mark({"id": call_id, "event": "start", "slot_id": slot_id, "try": tries})
            try:
                source = mint_python_challenger(
                    planner_client=self.planner_client(), planner_model=self.cfg.planner_model,
                    prompt=prompt, request=request, runtime=runtime, workdir=workdir,
                    usage_log=usage, prompt_dump_dir=workdir, genome_only=True,
                    incumbent_source=parent.source, hypothesis_out=hypothesis,
                    max_completion_tokens=self.cfg.planner_max_completion_tokens,
                    deadline_s=self.cfg.planner_deadline_s,
                )
                # registered with the racer by the MAIN thread (collect_mints):
                # the racer's rep allocation iterates its programs unlocked
                cand = workdir / "candidate.py"
                cand.write_text(source, encoding="utf-8")
                status = "ok"
                record.update({"program_id": slot_id, "source_sha256": _sha256(source),
                               "candidate_path": str(cand.relative_to(self.root))})
            except MintFailed as exc:
                record.update({"fail_kind": exc.kind, "reason": str(exc.reason)[:600]})
                if exc.kind == "planner_deadline":
                    deadlines += 1
                    status = "mint_failed" if deadlines > MAX_DEADLINE_RETRIES else "planner_deadline"
                else:
                    status = "mint_failed"
                if not hypothesis and isinstance(exc.hypothesis, dict):
                    hypothesis = dict(exc.hypothesis)
            except LeakageGuardError as exc:
                record.update({"status": "leakage_guard", "try": tries, "usage": usage,
                               "reason": str(exc)[:600], "finished_at": _now()})
                _append_jsonl(self.root / "mints.jsonl", record, self._mints_lock)
                self.meter.charge_planner(usage, round_index=gen)
                raise
            except Exception as exc:  # noqa: BLE001 - transport / planner infrastructure
                errors_total += 1
                outage = is_outage_error(exc)
                if outage:
                    session_outages += 1
                else:
                    session_errors += 1
                if errors_total > MAX_PLANNER_ERROR_TOTAL:
                    status = "planner_error_final"
                elif outage:
                    status = "planner_outage" if session_outages > MAX_PLANNER_ERROR_RETRIES \
                        else "planner_error"
                else:
                    status = "planner_error_final" if session_errors > MAX_PLANNER_ERROR_RETRIES \
                        else "planner_error"
                record["reason"] = f"{type(exc).__name__}: {exc}"[:600]
                record["outage"] = outage
            record.update({
                "status": status, "try": tries, "usage": usage,
                "planner_error_attempts": errors_total, "planner_error_total": errors_total,
                "deadline_attempts": deadlines,
                "hypothesis": read_mint_hypothesis(workdir) or hypothesis or None,
                "wall_s": round(time.time() - started, 1), "finished_at": _now(),
            })
            _write_json(workdir / "result.json", record)
            _append_jsonl(self.root / "mints.jsonl", record, self._mints_lock)
            # charged AFTER the append: the meter rebuilt from mints.jsonl on a
            # resume can never disagree with it
            self.meter.charge_planner(usage, round_index=gen)
            if status in FINAL_MINT_STATUSES or status == "planner_outage":
                return record
            if status == "planner_error" and self.cfg.planner_backoff_s > 0:
                time.sleep(float(self.cfg.planner_backoff_s) * max(session_outages, session_errors))

    def launch_mints(self, gen: int, state: Mapping[str, Any]) -> None:
        self.planner_client()  # built once, before the planner threads
        for brief in state.get("briefs") or []:
            sid = str(brief["slot_id"])
            prev = self._slot_result(sid)
            if isinstance(prev, dict) and prev.get("status") in FINAL_MINT_STATUSES:
                continue
            if sid in self._pending and not self._pending[sid].done():
                continue
            self._pending[sid] = self._planner_pool.submit(self.mint_slot, gen, dict(brief))

    def collect_mints(self, gen: int, state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every slot's final mint result (main thread).  A slot whose
        planner stayed unreachable stops the run resumable
        (:class:`InfraOutage`) instead of counting as a failed proposal."""

        self.launch_mints(gen, state)
        briefs = {str(b["slot_id"]): b for b in state.get("briefs") or []}
        results = []
        for sid, brief in briefs.items():
            future = self._pending.pop(sid, None)
            rec = dict((future.result() if future is not None else self._slot_result(sid))
                       or {"slot_id": sid, "status": "missing"})
            if rec.get("status") == "ok":
                self._register_minted(gen, brief, rec)
            results.append(rec)
        outages = [r["slot_id"] for r in results if r.get("status") == "planner_outage"]
        if outages:
            self.event("planner_outage", gen=gen, slots=outages)
            raise InfraOutage(f"planner unreachable for {outages} (retries exhausted this session)")
        return results

    def _register_minted(self, gen: int, brief: Mapping[str, Any], rec: dict[str, Any]) -> None:
        """Register a minted program with the racer (main thread only),
        after checking its source against the sha256 of its mint record."""

        pid = str(rec["program_id"])
        if pid in self.racer.programs:
            return
        source = (self.root / str(rec["candidate_path"])).read_text(encoding="utf-8")
        if _sha256(source) != rec.get("source_sha256"):
            raise EvoStateError(f"{pid}: minted source changed on disk")
        program = self.racer.register_program(pid, source, arm=self.cfg.arm, meta={
            "gen": int(gen), "slot": brief["slot"], "brief_id": brief["brief_id"],
            "kind": brief["kind"], "parent_ids": list(brief["parent_ids"]),
        })
        rec["fp"] = program.fp

    # -- screening (S0) ---------------------------------------------------------------------- #
    def v1_fp_key(self) -> str | None:
        if self._v1_fp_key is None:
            from queenbee.evo.screen import fp_key_for_task

            self._v1_fp_key = fp_key_for_task(self.v1.source)
        return self._v1_fp_key

    def _archive_slot(self, gen: int, brief: Mapping[str, Any], slot: Mapping[str, Any]) -> None:
        """Register a minted program's metadata with the archive
        (idempotent; ``Archive.add_program`` merges)."""

        pid = slot.get("program_id")
        if slot.get("status") != "ok" or not pid or pid not in self.racer.programs:
            return
        program = self.racer.program(str(pid))
        self.archive.add_program(
            program.program_id, gen=int(gen), arm=self.cfg.arm,
            parent_ids=list(brief["parent_ids"]), brief_id=brief["brief_id"],
            brief_kind=brief["kind"], fp=program.fp, source_sha256=program.source_sha256,
            hypothesis=dict(slot.get("hypothesis") or {}) or None,
            target_units=list(brief.get("target_units") or []),
            target_class=brief.get("target_class"),
        )

    def known_fps(self, before_gen: int) -> dict[str, str]:
        known: dict[str, str] = {}
        key = self.v1_fp_key()
        if key:
            known[key] = "v1"
        for g in range(int(before_gen)):
            for slot in (self.gen_state(g) or {}).get("slots") or []:
                fk = (slot.get("screen") or {}).get("fp_key")
                if slot.get("s0_pass") and fk:
                    known.setdefault(fk, str(slot["program_id"]))
        return known

    def screen_generation(self, gen: int, state: dict[str, Any],
                          mints: Sequence[Mapping[str, Any]]) -> None:
        """S0 on every minted proposal, in slot order.  Deterministic: the
        known fingerprints are v1's and those of the S0-passed proposals of
        the finished generations and of this generation's earlier slots; the
        ledger's tabu list (arms with the ledger) is that of the finished
        generations."""

        from queenbee.evo.screen import screen_for_task

        briefs = state.get("briefs") or []
        if state.get("slots") is not None and len(state["slots"]) == len(briefs):
            # a kill between the generation file and memory.json: re-add the
            # programs (idempotent) so parent_ids / fp are never lost
            for brief, slot in zip(briefs, state["slots"]):
                self._archive_slot(gen, brief, slot)
            return
        known = self.known_fps(gen)
        slots: list[dict[str, Any]] = []
        for brief, mint in zip(briefs, mints):
            hyp = dict(mint.get("hypothesis") or {})
            slot: dict[str, Any] = {
                "slot_id": brief["slot_id"], "brief_id": brief["brief_id"], "slot": brief["slot"],
                "kind": brief["kind"], "status": mint.get("status"),
                "program_id": mint.get("program_id"), "parent_id": brief["parent_ids"][0],
                "hypothesis": hyp or None, "s0_pass": False,
            }
            if mint.get("status") == "ok" and mint.get("program_id") in self.racer.programs:
                program = self.racer.program(str(mint["program_id"]))
                parent = self.racer.program(str(brief["parent_ids"][0]))
                self._archive_slot(gen, brief, slot)
                tabu = (self.ledger.tabu_fn(parent.source, target_class=brief.get("target_class"))
                        if self.flags.ledger else None)
                try:
                    result = screen_for_task(program.source, known_fps=known,
                                             hypothesis=hyp or None, tabu=tabu)
                    screen = result.to_dict()
                except Exception as exc:  # noqa: BLE001 - a screen crash is a rejection
                    screen = {"ok": False, "reasons": [f"screen_error:{type(exc).__name__}"]}
                slot["screen"] = {k: screen.get(k) for k in (
                    "ok", "reasons", "penalties", "rank_penalty", "readable5", "readable10",
                    "lost_edges5", "pred_calls5", "pred_calls10", "lint_hits", "dup_of", "fp_key")}
                slot["s0_pass"] = bool(screen.get("ok"))
                if slot["s0_pass"] and screen.get("fp_key"):
                    known.setdefault(str(screen["fp_key"]), str(slot["program_id"]))
            else:
                slot["screen"] = {"ok": False, "reasons": [f"mint:{mint.get('status')}"]}
            slots.append(slot)
        state["slots"] = slots
        self.save_gen(gen, state)
        self.save_memory(f"g{gen} screened")
        self.event("screened", gen=gen, passed=[s["slot_id"] for s in slots if s["s0_pass"]],
                   rejected={s["slot_id"]: (s.get("screen") or {}).get("reasons") for s in slots
                             if not s["s0_pass"]})

    # -- racing ------------------------------------------------------------------------------ #
    def _candidate(self, state: Mapping[str, Any], slot: Mapping[str, Any]) -> dict[str, Any]:
        brief = (state.get("briefs") or [])[int(slot["slot"])]
        return {"program_id": slot["program_id"], "parent_ids": list(brief["parent_ids"]),
                "target_units": list(brief.get("target_units") or []), "brief_id": brief["brief_id"],
                "kind": brief["kind"], "target_class": brief.get("target_class"),
                "rank_penalty": float((slot.get("screen") or {}).get("rank_penalty") or 0.0)}

    def _r1_cost(self, gen: int, cands: Sequence[Mapping[str, Any]]) -> float:
        cost = 0.0
        fresh: set[tuple[str, str]] = set()
        for c in cands:
            unit = c["target_units"][0]
            cost += self.pool.exec_eq(unit)
            pair = (c["parent_ids"][0], unit)
            if pair not in fresh and not self.racer.has_fresh(
                    *pair, gen, exclude_batch=batch_id(gen, "r1")):
                fresh.add(pair)
                cost += self.pool.exec_eq(unit)
        return cost

    def race_r1(self, gen: int, state: dict[str, Any]) -> str:
        """R1 + the fresh parent reps (one racer batch).  Returns a status."""

        r1 = state.get("r1") or {}
        if "cands" not in r1:
            planned = [self._candidate(state, s) for s in state.get("slots") or [] if s.get("s0_pass")]
            planned = [c for c in planned if c["target_units"]]
            remaining = self.evo_remaining()
            kept: list[dict[str, Any]] = []
            dropped: list[str] = []
            for cand in planned:  # slot order; a candidate the budget cannot pay for is dropped
                if self._r1_cost(gen, kept + [cand]) <= remaining + EPS:
                    kept.append(cand)
                else:
                    dropped.append(cand["program_id"])
            r1 = {"cands": kept, "dropped_budget": dropped, "planned_at": _now()}
            state["r1"] = r1
            self.save_gen(gen, state)
        if not r1["cands"]:
            state["r1"]["status"] = "budget_stop" if r1.get("dropped_budget") else "no_candidates"
            self.save_gen(gen, state)
            return state["r1"]["status"]
        res = self._r1_result(gen, state)
        self.add_rows(res["rows"])
        plan = self.racer.plan(batch_id(gen, "r1")) or []
        state["r1"].update({
            "status": "done", "survivors": res["survivors"], "r2_selected": res["r2_selected"],
            "results": {c["program_id"]: {k: c.get(k) for k in (
                "r1_unit", "S", "fresh_S", "d1", "survive", "status", "C", "format_fail",
                "merge_bar", "d1_vs_first_parent")}
                for c in res["candidates"]},
            # one fresh parent rep per distinct (parent, R1 unit) pair that had
            # none this generation
            "fresh_parent_reps": sum(1 for j in plan if j.purpose == "parent_fresh"),
        })
        self.save_gen(gen, state)
        self.save_memory(f"g{gen} r1")
        self.event("r1_done", gen=gen, d1={k: v["d1"] for k, v in state["r1"]["results"].items()})
        return "done"

    def _r1_result(self, gen: int, state: Mapping[str, Any]) -> dict[str, Any]:
        """``Racer.race_r1`` of the candidate list stored in the generation
        file (idempotent: a re-call returns the stored plan's scored rows)."""

        if gen not in self._r1_cache:
            res = self.racer.race_r1(
                gen, list((state.get("r1") or {}).get("cands") or []),
                cap_exec_eq=self.cfg.evolution_budget)
            self._r1_cache[gen] = self._merge_r1(gen, state, res)
        return self._r1_cache[gen]

    def _merge_r1(self, gen: int, state: Mapping[str, Any], res: dict[str, Any]) -> dict[str, Any]:
        """A merge child's d1 is measured against the BEST parent on its R1
        unit: max(fresh rep of the mutated parent, the other parent's S
        there).  The R1 unit of a merge is one the second parent wins, so
        against the first parent alone a mere copy of the second would
        survive with d1 >= 1/n.  Deterministic from the logs (resume-safe)."""

        th = self.th
        loop_cands = {c["program_id"]: c for c in (state.get("r1") or {}).get("cands") or []}
        changed = False
        for c in res.get("candidates") or []:
            parents = list((loop_cands.get(c["program_id"]) or {}).get("parent_ids") or [])
            if len(parents) < 2 or c.get("S") is None or c.get("fresh_S") is None:
                continue
            bar, src = float(c["fresh_S"]), "fresh"
            for other in parents[1:]:
                s, s_src = self.racer.baseline(other, c["r1_unit"], gen)
                if s is not None and s > bar + EPS:
                    bar, src = float(s), f"merge:{other}:{s_src}"
            c["merge_bar"] = {"S": _r(bar), "src": src, "fresh_parent_S": c["fresh_S"]}
            if src == "fresh":
                continue
            c["d1_vs_first_parent"] = c["d1"]
            c["d1"] = round(float(c["S"]) - bar, 6)
            if c.get("format_fail"):
                status = "format_fail"
            elif c["d1"] < th.tau_r1 - EPS:
                status = "below_tau_r1"
            else:
                status = "survived"
            c["status"], c["survive"] = status, status == "survived"
            changed = True
        if changed:
            res["survivors"] = [c["program_id"] for c in res["candidates"] if c["survive"]]
            res["r2_selected"] = _race.select_for_r2(res["candidates"], th)
        return res

    def _r2_cost(self, gen: int, r1res: Mapping[str, Any], selected: Sequence[str],
                 t_frontier: Sequence[str], guards: Sequence[str],
                 shapes: Mapping[str, Any]) -> float:
        """The exact cost of R2 for ``selected`` (the racer's own unit rule)."""

        cfg = self.racer.config
        cands = {c["program_id"]: c for c in r1res.get("candidates") or []}
        cost = 0.0
        for slot, pid in enumerate(selected):
            target = cands[pid]["target_units"][0]
            units = _race.pick_r2_units(pid, target, t_frontier, gen=gen, shapes=shapes,
                                        k=cfg.n_r2_units, other_tier=cfg.r2_other_tier)
            used = {template_of(u) for u in [target, *units]}
            guard = _race.pick_guard(guards, gen=gen, slot=slot, exclude_templates=used)
            cost += sum(self.pool.exec_eq(u) for u in units)
            cost += self.pool.exec_eq(guard) if guard else 0
        return cost

    def race_r2(self, gen: int, state: dict[str, Any]) -> None:
        """R2 of the R1 survivors the racer selected, dropping the last ones
        while the evolution budget cannot pay for them (the selection is
        stored in the generation file); outcomes go to the archive."""

        r2 = state.get("r2") or {}
        r1res = self._r1_result(gen, state)
        if "selected" not in r2:
            t_frontier = self.t_frontier()
            guards = self.pool.guard_T()
            shapes = self.shapes()
            selected = list(r1res.get("r2_selected") or [])
            remaining = self.evo_remaining()
            skipped: list[str] = []
            while selected and self._r2_cost(gen, r1res, selected, t_frontier, guards,
                                             shapes) > remaining + EPS:
                skipped.append(selected.pop())
            r2 = {"selected": selected, "skipped_budget": skipped, "t_frontier": t_frontier,
                  "guards": guards, "planned_at": _now()}
            state["r2"] = r2
            self.save_gen(gen, state)
        if r2["selected"]:
            res = self.racer.race_r2(gen, r1res, t_frontier=r2["t_frontier"], guards=r2["guards"],
                                     shapes=self.shapes(), selected=r2["selected"],
                                     cap_exec_eq=self.cfg.evolution_budget)
            self.add_rows(res["rows"])
            for outcome in res["outcomes"].values():
                self.archive.update_outcome(outcome)
            state["r2"]["plan"] = res["plan"]
        state["r2"]["status"] = "done"
        self.save_gen(gen, state)
        self.save_memory(f"g{gen} r2")
        self.event("r2_done", gen=gen, selected=r2["selected"])

    def confirm(self, gen: int, state: dict[str, Any]) -> None:
        """At most ``MAX_CONFIRM_PER_GEN`` confirmation rep per generation: a
        program other than v1 that leads a unit it is not yet confirmed on,
        and was never sent to confirmation, gets one fresh rep there (widest
        lead first; ``memory.Archive.newly_front_winning``), if the evolution
        budget can pay for it."""

        conf = state.get("confirm") or {}
        if "winners" not in conf:
            winners = self.archive.newly_front_winning(limit=MAX_CONFIRM_PER_GEN)
            winners = [w for w in winners if self.pool.exec_eq(w["unit_id"]) <= self.evo_remaining() + EPS]
            conf = {"winners": winners, "planned_at": _now()}
            state["confirm"] = conf
            self.save_gen(gen, state)
        winners = list(conf["winners"])
        for w in winners:  # marked only after the list is saved (idempotent)
            requested = self.archive.confirm_requested.setdefault(w["program_id"], [])
            if w["unit_id"] not in requested:
                requested.append(w["unit_id"])
        results: dict[str, bool] = {}
        if winners:
            res = self.racer.race_confirm(gen, winners, cap_exec_eq=self.cfg.evolution_budget)
            self.add_rows(res["rows"])
            results = self.archive.resolve_confirmations(winners)
        state["confirm"]["results"] = results
        self.save_gen(gen, state)
        self.save_memory(f"g{gen} confirm")
        self.event("confirm_done", gen=gen, results=results)

    # -- verdicts ---------------------------------------------------------------------------- #
    def verdicts(self, gen: int, state: dict[str, Any]) -> None:
        """A host verdict for EVERY proposal of the generation (ledger)."""

        r1 = state.get("r1") or {}
        raced = r1.get("status") == "done"
        cands = {c["program_id"]: c for c in (self._r1_result(gen, state)["candidates"]
                                              if raced else [])}
        plans = dict((state.get("r2") or {}).get("plan") or {})
        dropped = set(r1.get("dropped_budget") or [])
        verdicts: dict[str, str] = {}
        for slot in state.get("slots") or []:
            brief = (state.get("briefs") or [])[int(slot["slot"])]
            pid = slot.get("program_id")
            parent_src = self.racer.program(str(brief["parent_ids"][0])).source
            child_src = self.racer.program(pid).source if pid and pid in self.racer.programs else None
            common = dict(gen=int(gen), brief=brief, hypothesis=slot.get("hypothesis"),
                          parent_source=parent_src, child_source=child_src, thresholds=self.th)
            if not slot.get("s0_pass"):
                reasons = list((slot.get("screen") or {}).get("reasons") or [])
                dup = any(str(r).startswith("duplicate-of") for r in reasons)
                status = ("duplicate" if dup else "screened") if slot.get("status") == "ok" \
                    else str(slot.get("status") or "mint_failed")
                entry = self.ledger.record(pid or slot["slot_id"], outcome=None,
                                           screen_reasons=reasons, status=status, **common)
            elif pid in cands:
                outcome = self._outcome(gen, brief, cands[pid], plans.get(pid))
                self.archive.update_outcome(outcome)
                entry = self.ledger.record(pid, outcome=outcome, status=outcome.get("status"), **common)
                self._annotate_entry(entry, outcome)
                if self.cfg.attempt_diag:
                    after = self._after_diag(pid, entry.get("target_units") or [])
                    if after:
                        entry["after_diag"] = after
            else:
                # passed S0 but never raced (usually: the evolution budget ran
                # out before its R1): a host verdict of "screened" with its own
                # status, so it never enters an executed-proposal denominator
                reason = "not raced: evolution budget exhausted" if pid in dropped or not raced \
                    else "not raced"
                outcome = {"executed": False, "status": "not_raced_budget", "observed": {},
                           "target_dS": None}
                entry = self.ledger.record(pid, outcome=outcome, screen_reasons=[reason],
                                           status="not_raced_budget", **common)
            verdicts[slot["slot_id"]] = entry["verdict"]
        state["verdicts"] = verdicts
        self.save_gen(gen, state)
        self.save_memory(f"g{gen} verdicts")
        self.event("verdicts", gen=gen, verdicts=verdicts)

    def _outcome(self, gen: int, brief: Mapping[str, Any], cand: Mapping[str, Any],
                 r2_plan: Mapping[str, Any] | None) -> dict[str, Any]:
        """``Racer.outcome`` + the loop's R1 d1 (merge bar) + the economize
        cost verdict (an economize proposal succeeds by keeping S at fewer
        tokens)."""

        outcome = dict(self.racer.outcome(cand, gen, r2_plan=r2_plan))
        if cand.get("merge_bar") is not None:
            outcome["d1"] = cand.get("d1")
            outcome["merge_bar"] = cand.get("merge_bar")
        if brief.get("kind") == "economize":
            ratio = self._token_ratio(gen, str(brief["parent_ids"][0]), outcome)
            outcome["token_ratio"] = _r(ratio)
            outcome["verdict_rule"] = "economize"
            neutral = bool(self.cfg.econ_neutral)
            outcome["verdict"] = economize_verdict(outcome, token_ratio=ratio, thresholds=self.th,
                                                   neutral_no_saving=neutral)
            if neutral and econ_no_saving(outcome, ratio):
                outcome["econ_result"] = ECON_NOT_ECONOMIZED  # S kept, no token saving
        return outcome

    def _token_ratio(self, gen: int, parent: str, outcome: Mapping[str, Any]) -> float | None:
        """Mean child tokens / mean parent tokens over the executed target
        units with a paired parent cost (the outcome's ``baseline_C``: the
        fresh parent rep on the R1 unit; elsewhere the parent's fresh or
        archived rows, else v1's)."""

        child: list[float] = []
        base: list[float] = []
        per_unit = outcome.get("per_unit") or {}
        for unit in outcome.get("target_units_executed") or []:
            info = per_unit.get(unit) or {}
            c, b = info.get("C"), info.get("baseline_C")
            if b is None and info.get("baseline_src") != "no_fresh_parent" \
                    and hasattr(self.racer, "baseline_C"):
                b = self.racer.baseline_C(parent, unit, gen)
            if c is not None and b:
                child.append(float(c))
                base.append(float(b))
        if not child:
            return None
        return (sum(child) / len(child)) / (sum(base) / len(base))

    def _after_diag(self, pid: str, units: Sequence[str]) -> dict[str, Any] | None:
        """``--attempt-diag``: the child's own cards on its target units,
        summarized answer-free (dominant class; max own-shard-only /
        answer-seen counts; team size)."""

        from collections import Counter as _Counter

        from queenbee.evo.diagnosis import sanitize_diag

        cards = []
        for unit in units:
            for row in ((self.archive.rows.get(str(pid)) or {}).get(str(unit)) or {}).values():
                card = sanitize_diag((row or {}).get("diag"))
                if card and card.get("failure_class") and card.get("failure_class") != "infra":
                    cards.append(card)
        if not cards:
            return None
        classes = _Counter(str(c["failure_class"]) for c in cards)
        n = max((len(c["agent_correct"]) for c in cards if c.get("agent_correct")), default=None)
        out: dict[str, Any] = {"class": classes.most_common(1)[0][0], "n_agents": n,
                               "runs": len(cards)}
        for key, name in (("n_local_only", "own_shard_only"), ("n_answer_seen", "answer_seen")):
            out[name] = max(int(c.get(key) or 0) for c in cards)
        if n is None and not get_task().per_agent_cards:
            # no card carries per-agent correctness: the per-agent counts are undefined
            out["own_shard_only"] = out["answer_seen"] = None
        return out

    @staticmethod
    def _annotate_entry(entry: dict[str, Any], outcome: Mapping[str, Any]) -> None:
        """Ledger entry extras: the economize verdict rule and token ratio,
        how many units (and R2 units) the verdict rests on, the source of
        each unit's dS baseline, and the merge bar / rule of a merge."""

        if outcome.get("verdict_rule") == "economize":
            entry["verdict"] = outcome["verdict"]
            entry["verdict_rule"] = "economize"
            entry["token_ratio"] = outcome.get("token_ratio")
            if outcome.get("econ_result"):  # --econ-neutral only
                entry["econ_result"] = outcome["econ_result"]
        for key in ("n_observed_units", "n_r2_units_observed", "baseline_srcs", "merge_bar",
                    "baseline_rule"):
            if outcome.get(key) is not None:
                entry[key] = outcome[key]

    # -- curriculum ------------------------------------------------------------------------ #
    def curriculum(self, gen: int, state: dict[str, Any]) -> None:
        """Curriculum: when the current best has fewer than
        ``CURRICULUM_MIN_FRONTIER`` frontier T units left, saturated T
        templates move up one rung (:func:`rung_next`; one charged execution
        per move)."""

        cur = state.get("curriculum") or {}
        if "promotions" not in cur:
            best = self.current_best()
            frontier = self.t_frontier()
            left = [u for u in frontier if (self._S_best(best, u) or 0.0) < 1.0 - EPS]
            promotions: list[dict[str, Any]] = []
            requests: list[tuple] = []
            if len(left) < CURRICULUM_MIN_FRONTIER:
                by_template: dict[str, list[str]] = {}
                for unit in self.pool.pool_T():
                    by_template.setdefault(template_of(unit), []).append(unit)
                budget = self.evo_remaining()
                for template in sorted(by_template, key=lambda t: _unit_order(t + "@o5")):
                    if len(left) + len(promotions) >= CURRICULUM_MIN_FRONTIER:
                        break
                    units = by_template[template]
                    if any((self._S_best(best, u) if self._S_best(best, u) is not None else 0.0)
                           < 1.0 - EPS for u in units):
                        continue
                    order = _common.task_rung_order()
                    top = max(units, key=lambda u: order.get(rung_of(u), 0))
                    nxt = rung_next().get(rung_of(top))
                    if not nxt:
                        continue
                    new_unit = f"{template}@{nxt}"
                    if new_unit in self.pool.pool_T() or not self.pool.has_instance(new_unit):
                        continue
                    if self.pool.klass(new_unit) in ("irreducible", "dead"):
                        continue
                    # one execution per move: v1 when it has no row on the
                    # promoted unit (shrunk scores and the v1 fallback of the
                    # dS baselines need it), else the current best
                    runner = best if self.archive.unit_stats("v1").get(new_unit) else "v1"
                    new = [(runner, new_unit, "curriculum", None)]
                    cost = sum(self.pool.exec_eq(new_unit) for _ in new)
                    if cost > budget + EPS:
                        continue
                    budget -= cost
                    requests += new
                    promotions.append({"template": template, "from": top, "to": new_unit,
                                       "program_id": runner, "best": best})
            cur = {"best": best, "frontier_left": left, "promotions": promotions,
                   "requests": [list(r) for r in requests], "planned_at": _now()}
            state["curriculum"] = cur
            self.save_gen(gen, state)
        for promo in cur.get("promotions") or []:
            self.pool.extra.setdefault(promo["to"], {"gen": int(gen), "from": promo["from"]})
            if promo["to"] not in self.pool.units:
                self.pool.units[promo["to"]] = self._curriculum_unit(promo["to"])
        _write_json(self.root / "state" / "pool.json", {"extra": self.pool.extra})
        if cur.get("requests"):
            rows = self.racer.run_batch(batch_id(gen, "curr"), [tuple(r) for r in cur["requests"]],
                                        gen=gen, cap_exec_eq=self.cfg.evolution_budget)
            self.add_rows(rows)
        state["curriculum"]["status"] = "done"
        self.save_gen(gen, state)
        if cur.get("promotions"):
            self.event("curriculum", gen=gen, promotions=cur["promotions"])

    # -- one generation ------------------------------------------------------------------- #
    def run_generation(self, gen: int) -> str:
        """Run (or resume) generation ``gen``: plan, mint, S0 screen, R1, then
        R2 + confirmation (with async minting, generation g+1 is planned and
        minted while R2 runs), verdicts and curriculum.  Returns the status
        of R1 (``done``, ``budget_stop`` or ``no_candidates``)."""

        state = self.gen_state(gen)
        if state and state.get("done"):
            return str(state.get("status") or "done")
        state = self.plan_generation(gen, after=f"g{gen - 1} done")
        self.event("gen_start", gen=gen, spent_evolution=self.spent("evolution"))
        mints = self.collect_mints(gen, state)
        state = self.gen_state(gen) or state
        self.screen_generation(gen, state, mints)
        status = self.race_r1(gen, state)
        if status == "done":
            self.maybe_launch_next(gen)
            self.race_r2(gen, state)
            self.confirm(gen, state)
        self.verdicts(gen, state)
        if status == "done":
            self.curriculum(gen, state)
        best = self.current_best()
        paid = [r for r in self._paid() if (batch_key(r.get("batch_id")) or (-1,))[0] == gen]
        state.update({
            "done": True, "status": status,
            "best_after": {"program_id": best, "shrunk_T": _r(self.archive.shrunk_score(best))},
            "spent_evolution": self.spent("evolution"),
            "executed": len(paid),
            "exec_by_purpose": dict(Counter(str(r.get("purpose")) for r in paid)),
        })
        self.save_gen(gen, state)
        self.save_memory(f"g{gen} done")
        self.event("gen_done", gen=gen, status=status, best=best, spent=state["spent_evolution"])
        return status

    def maybe_launch_next(self, gen: int) -> None:
        """Asynchronous overlap: plan generation g+1 from the state after R1
        of g and mint it while R2 of g runs -- only while the evolution
        budget (as of R1 of g, so the decision is the same on a resume) can
        still pay for another R1."""

        if not self.cfg.async_mint or gen + 1 >= int(self.cfg.max_generations):
            return
        remaining = self.cfg.evolution_budget - self.spent_upto(gen, "r1")
        if remaining - self._inflight_estimate(gen) < MIN_GEN_COST - EPS:
            return
        state = self.plan_generation(gen + 1, after=f"g{gen} r1")
        self.launch_mints(gen + 1, state)
        self.event("async_mint_launched", gen=gen + 1, after_gen=gen)

    def _inflight_estimate(self, gen: int) -> float:
        """Rough upper bound of what the rest of generation g may spend: R2 of
        the selected survivors (3 units each, counted at 2 execution-
        equivalents per unit) plus one confirmation rep."""

        r1 = (self.gen_state(gen) or {}).get("r1") or {}
        n = len(r1.get("r2_selected") or [])
        return float(n * 3 * 2 + (1 if n else 0))

    def evolution_done(self) -> bool:
        return bool((_read_json(self.root / "state" / "evolution.json") or {}).get("done"))

    def _evolve(self) -> None:
        """Generations until the evolution budget cannot pay for another R1.

        Only the budget ends a run normally (arms stay budget-matched):
        proposals that are all S0-rejected / duplicates / failed mints cost
        planner calls, not the run.  The fixed stop caps
        ``max_generations`` and ``max_idle_generations`` (consecutive
        generations without a scored execution) stop a pathological run,
        recorded as FAILED (``budget_matched: false``).  Infrastructure
        outages never reach this point: they raise :class:`InfraOutage`
        (resumable, nothing marked done)."""

        self._inherit_programs()
        gen = 0
        idle = 0
        reason = "budget"
        while True:
            state = self.gen_state(gen)
            if state and state.get("done"):
                if state.get("status") == "budget_stop":
                    break
                idle = idle + 1 if not state.get("executed") else 0
                gen += 1
                continue
            if self.evo_remaining() < MIN_GEN_COST - EPS and not (state and state.get("r1")):
                break
            if gen >= int(self.cfg.max_generations):
                reason = "max_generations"
                break
            if idle >= int(self.cfg.max_idle_generations):
                reason = "idle"
                break
            status = self.run_generation(gen)
            if status == "budget_stop":
                break
            idle = idle + 1 if not (self.gen_state(gen) or {}).get("executed") else 0
            gen += 1
        for sid, future in list(self._pending.items()):  # async mints of a gen never raced
            try:
                future.result()
            except Exception:  # noqa: BLE001 - recorded in its result.json
                pass
            self._pending.pop(sid, None)
        orphan = self.gen_state(gen)
        if orphan and not orphan.get("done"):
            mints = self.collect_mints(gen, orphan)  # an unreachable planner raises (resumable)
            self.screen_generation(gen, orphan, mints)
            orphan["r1"] = {"cands": [], "status": "budget_stop",
                            "dropped_budget": [s["program_id"] for s in orphan["slots"] if s.get("s0_pass")]}
            self.verdicts(gen, orphan)
            orphan.update({"done": True, "status": "budget_stop" if reason == "budget" else "stopped",
                           "executed": 0})
            self.save_gen(gen, orphan)
            gen += 1
        remaining = self.evo_remaining()
        matched = reason == "budget"
        record = {
            "done": True, "reason": reason, "generations": gen,
            "status": "ok" if matched else "failed",
            "budget_matched": matched,
            "spent_evolution": self.spent("evolution"),
            "remaining_evolution": _r(remaining),
            "idle_generations_at_stop": idle,
            "finished_at": _now(),
        }
        if not matched:
            record["note"] = (f"stopped by the {reason} cap with {remaining:g} of "
                              f"{self.cfg.evolution_budget:g} evolution exec-eq unspent: "
                              "not budget-matched, report as a failed run")
        _write_json(self.root / "state" / "evolution.json", record)
        self.event("evolution_done", reason=reason, generations=gen, status=record["status"],
                   remaining=record["remaining_evolution"])

    # -- final selection (V, exactly once) ------------------------------------------------- #
    def final_selection(self) -> dict[str, Any]:
        """Validation, exactly once per run: the top 3 programs by shrunk T
        score (measured on >= 3 T units; :meth:`_final_candidates`) + v1, x
        ``V_frontier[:3]`` x reps 1, 2 (v1 re-uses its census rows and runs
        only the missing reps), ranked by mean V S.  The leader then runs
        <= 2 V guards (1 rep); a leader that fails them gives way to the next
        in order, down to v1, which needs no guard.

        Missing data is never ranked: an infra-exhausted V job gets a
        replacement rep (a new rep index, <= ``MAX_FINAL_REPLACEMENTS``
        rounds, never more scored rows than planned, so the reserve holds);
        a candidate still without a scored row on some V unit is
        ``incomplete`` and excluded; means are unit-balanced over the same V
        units for every ranked candidate, v1 included.  A service outage
        raises :class:`InfraOutage` before anything is marked done (resume
        re-runs what is missing)."""

        path = self.root / "state" / "final.json"
        final = _read_json(path) or {}
        if final.get("done"):
            return final
        if not final.get("planned_at"):
            # --final-coverage-topup: its T reps run (and are on disk) BEFORE
            # the final plan, so every T execution precedes the V ones and the
            # plan ranks with the top-up rows included
            topup = self.coverage_topup() if self.cfg.final_coverage_topup > 0 else None
            final = self._plan_final()
            if topup is not None:
                final["coverage_topup"] = {k: topup.get(k) for k in (
                    "programs", "units", "cost", "cap", "bar", "after")}
            _write_json(path, final)
            self.event("final_planned", candidates=final["candidates"], V_units=final["V_units"])
        # One batch per rep: two reps of one program on one unit in one batch
        # can finish in the same millisecond, and the QB_TRACE_DIR file name
        # (case, source sha, ms) would then hold only one of their traces.
        for rep in sorted({int(r[3]) for r in final.get("requests") or []}):
            self.racer.run_batch(f"final:vselect:r{rep}",
                                 [tuple(r) for r in final["requests"] if int(r[3]) == rep],
                                 cap_exec_eq=self.cfg.budget)
        final["replacements"] = self._replace_missing_v(final)
        scores = self._v_scores(final)
        final["V_scores"] = scores
        final["incomplete"] = [p for p in final["candidates"]
                               if not (scores.get(p) or {}).get("complete", True)]
        order = self._final_order(final["candidates"], scores)
        final["order"] = order
        champion = None
        guard_results = []
        for pid in order:
            if pid == "v1":
                champion = "v1"
                break
            passed, detail = self._v_guard(pid, final)
            guard_results.append({"program_id": pid, **detail})
            if passed:
                champion = pid
                break
        final.update({"guard_results": guard_results, "champion": champion or "v1",
                      "done": True, "finished_at": _now()})
        _write_json(path, final)
        self.write_champion(final)
        self.event("final_done", champion=final["champion"], incomplete=final["incomplete"])
        return final

    def _final_candidates(self) -> tuple[list[str], dict[str, Any]]:
        """(top-3 by shrunk T score among programs with >= 3 T units, minus
        T-harm vetoes and non-positive scores; the vetoes)."""

        pool = self.archive.top_by_shrunk(N_FINAL_CANDIDATES + 6, min_units=MIN_FINAL_UNITS)
        harm = {p: self._t_harm(p) for p in pool}
        vetoed = {p: h for p, h in harm.items() if h["loss"] > T_HARM_MAX + EPS}
        cands = [p for p in pool if p not in vetoed
                 and (self.archive.shrunk_score(p) or 0.0) > EPS][:N_FINAL_CANDIDATES]
        return cands, vetoed

    def _final_v_costs(self) -> tuple[list[str], list[str], list[tuple], float, float, float]:
        """(V units, V guards, v1's V requests, cost per candidate, guard
        cost, v1 cost) of the final plan."""

        v_units = self.pool.V_frontier()[:N_V_UNITS]
        v_guards = self.pool.V_guard()[:N_V_GUARDS]
        for unit in v_units + v_guards:
            self.pool.assert_side(unit, "V", where="final selection")
        v1_req: list[tuple] = []
        for unit in v_units:
            have = len([r for r in self.census_V.get(unit) or [] if _race.row_S(r) is not None])
            reps = list(FINAL_REPS[: max(0, len(FINAL_REPS) - have)])
            v1_req += [("v1", unit, V_PURPOSE, rep) for rep in reps]
        per_cand = sum(self.pool.exec_eq(u) for u in v_units) * len(FINAL_REPS)
        guard_cost = sum(self.pool.exec_eq(u) for u in v_guards)
        v1_cost = sum(self.pool.exec_eq(r[1]) for r in v1_req)
        return v_units, v_guards, v1_req, per_cand, guard_cost, v1_cost

    def _plan_final(self) -> dict[str, Any]:
        """The final plan: candidates (T-harm vetoes excluded; the last ones
        dropped while the remaining budget cannot pay for the V reps and the
        V guards), V units, V guards and the V requests, with the selection
        rule spelled out."""

        cands, vetoed = self._final_candidates()
        v_units, v_guards, v1_req, per_cand, guard_cost, v1_cost = self._final_v_costs()
        remaining = self.cfg.budget - self.spent("total")
        dropped = []
        while cands and v1_cost + per_cand * len(cands) + guard_cost > remaining + EPS:
            dropped.append(cands.pop())
        requests = v1_req + [(c, u, V_PURPOSE, rep) for c in cands for u in v_units
                             for rep in FINAL_REPS]
        return {
            "planned_at": _now(), "candidates": ["v1"] + cands, "dropped_budget": dropped,
            "vetoed_t_harm": vetoed, "t_harm_max": T_HARM_MAX,
            "shrunk_T": {p: _r(self.archive.shrunk_score(p)) for p in cands},
            "T_units": {p: sorted(self.archive.unit_stats(p), key=_unit_order) for p in cands},
            "V_units": v_units, "V_guards": v_guards, "requests": [list(r) for r in requests],
            "rule": ("top-3 by shrunk T score (>= 3 T units) + v1; unit-balanced mean S over "
                     f"V_frontier[:{N_V_UNITS}] x reps {list(FINAL_REPS)} (v1 from the census); "
                     "only candidates scored on every V unit are ranked (infra-exhausted jobs get "
                     f"<= {MAX_FINAL_REPLACEMENTS} replacement rounds); ties within {TIE_EPS} -> "
                     f"fewer tokens; the leader must not lose > {V_GUARD_LOSS_MAX} on "
                     f"{N_V_GUARDS} V guards (1 rep)"),
        }

    # -- coverage top-up before the final plan (--final-coverage-topup) ------------------ #
    def coverage_topup(self) -> dict[str, Any]:
        """``--final-coverage-topup K``, once per run (``state/topup.json``;
        resumable: the batch plan is idempotent).  A program measured on
        fewer than ``MIN_FINAL_UNITS`` T units can never be a final candidate
        however well it scored (e.g. S = 1.0 twice on its only unit).  Up to
        K such programs whose shrunk T score beats current candidate
        #``TOPUP_KEEP_CANDIDATES`` (and v1) and that pass the T-harm veto get
        one fresh rep (purpose ``topup``) on just enough further T units,
        BEFORE the final plan;
        the cost is paid from the final reserve and capped so V
        selection can still pay for ``TOPUP_KEEP_CANDIDATES`` candidates,
        v1's V reps and the V guards (with the default reserve a top-up can
        thus cost the third V slot).  T units only (racer side guard); V is
        untouched until the final plan."""

        path = self.root / "state" / "topup.json"
        rec = _read_json(path) or {}
        k = int(self.cfg.final_coverage_topup)
        if k <= 0 or rec.get("done"):
            return rec
        if not rec.get("planned_at"):
            rec = self._plan_topup(k)
            _write_json(path, rec)
            self.event("topup_planned", programs=rec["programs"], units=rec["units"],
                       cost=rec["cost"], cap=rec["cap"])
        if rec.get("requests"):
            rows = self.racer.run_batch(TOPUP_BATCH, [tuple(r) for r in rec["requests"]],
                                        cap_exec_eq=self.cfg.budget)
            self.add_rows(rows)
        rec.update({
            "done": True, "finished_at": _now(),
            "after": {p: {"n_T_units": self.archive.n_t_units(p),
                          "shrunk_T": _r(self.archive.shrunk_score(p))}
                      for p in rec.get("programs") or []},
            "spent_final": self.spent("final"),
        })
        _write_json(path, rec)
        self.save_memory("topup")
        self.event("topup_done", after=rec["after"])
        return rec

    def _topup_units(self, pid: str, need: int, lead_units: Sequence[str]) -> list[str]:
        """``need`` further T units for ``pid`` (deterministic): units with v1
        rows it has not run, cheapest first, then templates it has not run,
        then the current candidates' units (the same footing), then the
        frontier units in play, then the rest of the T pool."""

        mine = self.archive.unit_stats(pid)
        base = self.archive.unit_stats(self.archive.v1_id)
        measured = {template_of(u) for u in mine}
        order = list(dict.fromkeys(list(lead_units) + list(self.t_frontier())
                                   + sorted(self.pool.pool_T(), key=_unit_order)))
        free = [u for u in order if u not in mine and u in base and self.pool.side(u) == "T"]
        free.sort(key=lambda u: (self.pool.exec_eq(u), template_of(u) in measured,
                                 order.index(u)))
        out: list[str] = []
        for unit in free:  # one unit per template where possible
            if len(out) >= need:
                break
            if template_of(unit) not in {template_of(x) for x in out}:
                out.append(unit)
        for unit in free:
            if len(out) >= need:
                break
            if unit not in out:
                out.append(unit)
        for unit in out:
            self.pool.assert_side(unit, "T", where="coverage topup")
        return out

    def _plan_topup(self, k: int) -> dict[str, Any]:
        # the bar: the last current candidate whose V slot the reserve still
        # guarantees after a top-up (a top-up of the default 20-exec-eq reserve
        # can cost the third slot, so only a program already ahead of the
        # second candidate on its own evidence is worth it)
        current, _vetoed = self._final_candidates()
        bar = (self.archive.shrunk_score(current[TOPUP_KEEP_CANDIDATES - 1]) or 0.0) \
            if len(current) >= TOPUP_KEEP_CANDIDATES else 0.0
        pool = []
        for pid in self.archive.programs:
            if pid == self.archive.v1_id or pid not in self.racer.programs:
                continue
            if not 1 <= self.archive.n_t_units(pid) < MIN_FINAL_UNITS:
                continue
            score = self.archive.shrunk_score(pid)
            if score is None or score <= max(bar, 0.0) + EPS:
                continue
            if self._t_harm(pid)["loss"] > T_HARM_MAX + EPS:
                continue
            tokens = _mean(v["C"] for v in self.archive.unit_stats(pid).values()
                           if v.get("C") is not None)
            pool.append((-score, tokens if tokens is not None else float("inf"), pid))
        ranked = [pid for _s, _c, pid in sorted(pool)]
        _v_units, _v_guards, _v1_req, per_cand, guard_cost, v1_cost = self._final_v_costs()
        keep = v1_cost + per_cand * TOPUP_KEEP_CANDIDATES + guard_cost
        cap = self.cfg.budget - self.spent("total") - keep
        lead_units = [u for p in current
                      for u in sorted(self.archive.unit_stats(p), key=_unit_order)]
        chosen: list[str] = []
        units: dict[str, list[str]] = {}
        skipped: dict[str, str] = {}
        requests: list[list[Any]] = []
        cost = 0.0
        for pid in ranked:
            if len(chosen) >= k:
                break
            need = MIN_FINAL_UNITS - self.archive.n_t_units(pid)
            picked = self._topup_units(pid, need, lead_units)
            if len(picked) < need:
                skipped[pid] = "not enough T units with v1 rows"
                continue
            c = sum(self.pool.exec_eq(u) for u in picked)
            if cost + c > cap + EPS:
                skipped[pid] = "reserve"
                continue
            chosen.append(pid)
            units[pid] = picked
            requests += [[pid, u, TOPUP_PURPOSE, None] for u in picked]
            cost += c
        return {
            "planned_at": _now(), "k": int(k), "bar": _r(bar), "cap": _r(cap),
            "keep_candidates": TOPUP_KEEP_CANDIDATES, "current_candidates": current,
            "ranked": {p: _r(self.archive.shrunk_score(p)) for p in ranked},
            "programs": chosen, "units": units, "skipped": skipped,
            "requests": requests, "cost": cost,
            "rule": (f"up to {k} programs with 1..{MIN_FINAL_UNITS - 1} T units whose shrunk T "
                     f"score beats current final candidate #{TOPUP_KEEP_CANDIDATES} (and v1) "
                     f"and that pass the T-harm veto get one rep on {MIN_FINAL_UNITS} - n "
                     "further T units with "
                     "v1 rows (cheapest, new templates, the candidates' units first); cost <= "
                     f"the final reserve left after V for {TOPUP_KEEP_CANDIDATES} candidates"),
        }

    def _t_harm(self, pid: str) -> dict[str, Any]:
        """The largest drop of ``pid``'s mean S below v1's on any T unit both
        measured (T rows only; census rows count for v1)."""

        mine = self.archive.unit_stats(pid)
        base = self.archive.unit_stats(self.archive.v1_id)
        worst = {"loss": 0.0, "unit": None}
        for unit, st in mine.items():
            ref = base.get(unit)
            if ref is None or st.get("S") is None or ref.get("S") is None:
                continue
            # only where v1 is STABLY right (>= 2 reps, every rep >= 0.8): on a
            # unit v1 itself flips on, one candidate rep is noise, not harm
            ref_s = [float(r["S"]) for r in (self.archive.rows.get(self.archive.v1_id) or {})
                     .get(unit, {}).values() if r.get("S") is not None]
            if len(ref_s) < 2 or min(ref_s) < T_HARM_STABLE_MIN - EPS:
                continue
            loss = round(float(ref["S"]) - float(st["S"]), 4)
            if loss > worst["loss"]:
                worst = {"loss": loss, "unit": unit, "S": _r(st["S"]), "v1_S": _r(ref["S"])}
        return worst

    def _scored_v(self, pid: str, unit: str) -> list[dict[str, Any]]:
        return self.racer.scored_rows(pid, unit, purposes=(V_PURPOSE,))

    def _run_split(self, batch: str, requests: Sequence[tuple]) -> None:
        """Run replacement requests; repeated (program, unit) pairs go to
        separate batches (trace file names are per millisecond)."""

        seen: Counter = Counter()
        waves: dict[int, list[tuple]] = {}
        for req in requests:
            key = (req[0], req[1])
            waves.setdefault(seen[key], []).append(req)
            seen[key] += 1
        for j in sorted(waves):
            self.racer.run_batch(f"{batch}.{j}" if j else batch, waves[j],
                                 cap_exec_eq=self.cfg.budget)

    def _replace_missing_v(self, final: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Up to ``MAX_FINAL_REPLACEMENTS`` rounds of replacement reps for
        planned V jobs that ended without a scored row (infra); a (program,
        unit) pair never gets more scored rows than planned."""

        required = Counter((str(r[0]), str(r[1])) for r in final.get("requests") or [])
        rounds: list[dict[str, Any]] = []
        for k in range(1, MAX_FINAL_REPLACEMENTS + 1):
            missing: list[tuple] = []
            for (pid, unit), need in sorted(required.items()):
                have = len(self._scored_v(pid, unit))
                missing += [(pid, unit, V_PURPOSE, None)] * max(0, need - have)
            if not missing:
                break
            bid = f"final:vselect:fix{k}"
            self.event("final_replacement", batch=bid,
                       jobs=[f"{r[0]}|{r[1]}" for r in missing])
            self._run_split(bid, missing)
            rounds.append({"batch_id": bid, "jobs": [[r[0], r[1]] for r in missing]})
        return rounds

    def _v_rows(self, pid: str, unit: str) -> list[dict[str, Any]]:
        """Scored V-selection rows of ``pid`` on ``unit`` (v1: its census rows
        first)."""

        executed = [rec["row"] for rec in self._scored_v(pid, unit)]
        if pid == "v1":
            return [r for r in self.census_V.get(unit) or [] if _race.row_S(r) is not None] + executed
        return executed

    def _v_scores(self, final: Mapping[str, Any]) -> dict[str, Any]:
        """Per candidate: S per V unit, the unit-balanced mean S, mean tokens
        and ``complete`` (a scored row on every V unit)."""

        out: dict[str, Any] = {}
        for pid in final["candidates"]:
            per_unit = {}
            means: list[float] = []
            all_c: list[float] = []
            for unit in final["V_units"]:
                rows = self._v_rows(pid, unit)
                s = [x for x in (_race.row_S(r) for r in rows) if x is not None]
                all_c += [float(r["C"]) for r in rows if r.get("C") is not None]
                m = _mean(s)
                if m is not None:
                    means.append(m)
                per_unit[unit] = {"S": [round(x, 4) for x in s], "mean": _r(m), "n": len(s)}
            out[pid] = {"per_unit": per_unit, "mean_S": _r(_mean(means)),
                        "tokens": _r(_mean(all_c), 1), "n_units": len(means),
                        "complete": all(per_unit[u]["S"] for u in final["V_units"])}
        return out

    @staticmethod
    def _final_order(candidates: Sequence[str], scores: Mapping[str, Any]) -> list[str]:
        """Best (unit-balanced) mean V S first among candidates scored on
        EVERY V unit; candidates within ``TIE_EPS`` of the best are tied and
        ordered by fewer tokens; v1 last when it is not ranked."""

        pool = [p for p in candidates if (scores.get(p) or {}).get("mean_S") is not None
                and (scores.get(p) or {}).get("complete", True)]
        if not pool:
            return ["v1"]
        top = max(float(scores[p]["mean_S"]) for p in pool)
        tied = [p for p in pool if float(scores[p]["mean_S"]) >= top - TIE_EPS - EPS]
        rest = [p for p in pool if p not in tied]
        tied.sort(key=lambda p: (float(scores[p].get("tokens") or 0.0), -float(scores[p]["mean_S"]),
                                 list(candidates).index(p)))
        rest.sort(key=lambda p: (-float(scores[p]["mean_S"]), float(scores[p].get("tokens") or 0.0),
                                 list(candidates).index(p)))
        order = tied + rest
        if "v1" not in order:
            order.append("v1")
        return order

    def _v_guard(self, pid: str, final: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
        """The V-guard check of a final leader: it must not lose more than
        ``V_GUARD_LOSS_MAX`` against v1's census mean (1.0 without census
        rows) on any V guard (1 rep; an infra-exhausted guard gets
        replacement reps; a guard that stays unscored fails -- an unverified
        leader never becomes champion)."""

        guards = list(final.get("V_guards") or [])
        if not guards:
            return True, {"passed": True, "note": "no V guard units"}
        need = sum(self.pool.exec_eq(g) for g in guards if not self._scored_v(pid, g))
        if self.spent("total") + need > self.cfg.budget + EPS:
            return False, {"passed": False, "note": "budget exhausted before the V-guard check"}
        self.racer.run_batch(f"final:vguard:{pid}", [(pid, g, V_PURPOSE, FINAL_GUARD_REP) for g in guards],
                             cap_exec_eq=self.cfg.budget)
        for k in range(1, MAX_FINAL_REPLACEMENTS + 1):
            missing = [g for g in guards if not self._scored_v(pid, g)]
            if not missing:
                break
            self.racer.run_batch(f"final:vguard:{pid}:fix{k}",
                                 [(pid, g, V_PURPOSE, None) for g in missing],
                                 cap_exec_eq=self.cfg.budget)
        detail = {}
        passed = True
        for g in guards:
            rows = self._scored_v(pid, g)
            s = _race.row_S(rows[0]["row"]) if rows else None
            base = _mean(x for x in (_race.row_S(r) for r in self.census_V.get(g) or [])
                         if x is not None)
            base = 1.0 if base is None else base
            loss = None if s is None else round(base - s, 4)
            ok = loss is not None and loss <= V_GUARD_LOSS_MAX + EPS
            passed = passed and ok
            detail[g] = {"S": s, "v1_S": _r(base), "loss": loss, "ok": ok}
            if s is None:
                detail[g]["note"] = "unscored after replacements (infra): unverified"
        return passed, {"passed": passed, "units": detail}

    def lineage(self, pid: str) -> list[dict[str, Any]]:
        """The first-parent chain from ``pid`` back to the seed program."""

        chain = []
        seen: set[str] = set()
        while pid and pid not in seen and pid in self.racer.programs:
            seen.add(pid)
            program = self.racer.program(pid)
            meta = program.meta
            chain.append({"program_id": pid, "gen": meta.get("gen"), "kind": meta.get("kind"),
                          "brief_id": meta.get("brief_id"), "parent_ids": meta.get("parent_ids") or [],
                          "source_sha256": program.source_sha256, "fp": program.fp})
            parents = meta.get("parent_ids") or []
            pid = str(parents[0]) if parents else ""
        return chain

    def write_champion(self, final: Mapping[str, Any]) -> dict[str, Any]:
        """Write the champion: ``champion.py`` (its source) and
        ``champion.json`` (V scores, selection record, lineage, its latest
        ledger entry, budget used, thresholds)."""

        pid = str(final["champion"])
        program = self.racer.program(pid)
        entry = next((e for e in reversed(self.ledger.entries) if e.get("program_id") == pid), None)
        record = {
            "program_id": pid, "arm": self.cfg.arm, "variant": self.cfg.variant,
            "flags": dataclasses.asdict(self.flags), "policy": self.policy,
            "split": self.pool.split.to_dict(), "split_seed": int(self.cfg.split_seed),
            "budget": self.cfg.budget, "source": program.source, "sha256": program.source_sha256,
            "fp": program.fp, "V_scores": final["V_scores"].get(pid),
            "V_all": final["V_scores"], "V_units": final["V_units"],
            "selection": {"order": final["order"], "guard_results": final.get("guard_results"),
                          "rule": final["rule"], "candidates": final["candidates"],
                          "dropped_budget": final.get("dropped_budget"),
                          "incomplete": final.get("incomplete"),
                          "replacements": final.get("replacements")},
            "run_status": self.run_status(),
            "lineage": self.lineage(pid),
            "hypothesis": (entry or {}).get("hypothesis"),
            "ledger_entry": {k: (entry or {}).get(k) for k in ("gen", "verdict", "target_dS", "D",
                                                               "mech_key", "observed")} if entry else None,
            "budget_used": {"evolution_exec_eq": self.spent("evolution"),
                            "final_exec_eq": self.spent("final"),
                            "total_exec_eq": self.spent("total")},
            "thresholds": self.th.to_dict(), "frozen_at": _now(),
        }
        _write_json(self.root / "champion.json", record)
        (self.root / "champion.py").write_text(program.source)
        return record

    # -- driver ----------------------------------------------------------------------------- #
    def evolution_record(self) -> dict[str, Any]:
        return _read_json(self.root / "state" / "evolution.json") or {}

    def run_status(self) -> str:
        """``running`` until evolution is done, then ``ok`` (budget-matched)
        or ``failed:<cap>`` (see :meth:`_evolve`)."""

        rec = self.evolution_record()
        if not rec.get("done"):
            return "running"
        return "ok" if rec.get("budget_matched", True) else f"failed:{rec.get('reason')}"

    def run(self) -> dict[str, Any]:
        """Evolve (unless already done), run final selection and write
        ``summary.json``.  An :class:`InfraOutage` writes ``state/outage.json``
        and propagates."""

        self.event("run_start", arm=self.cfg.arm, variant=self.cfg.variant,
                   flags=dataclasses.asdict(self.flags), split=self.pool.split.to_dict(),
                   budget=self.cfg.budget, spent_total=self.spent("total"))
        try:
            if not self.evolution_done():
                self._evolve()
            final = self.final_selection()
        except InfraOutage as exc:
            # nothing is marked done: ``--resume`` continues where this stopped
            self._planner_pool.shutdown(wait=False, cancel_futures=True)
            _write_json(self.root / "state" / "outage.json",
                        {"at": _now(), "reason": str(exc)[:500],
                         "spent_total": self.spent("total")})
            self.event("infra_outage_stop", reason=str(exc)[:500])
            raise
        except BaseException:
            # do not block on in-flight planner calls (a mint is resumable)
            self._planner_pool.shutdown(wait=False, cancel_futures=True)
            raise
        self._planner_pool.shutdown(wait=True, cancel_futures=True)
        summary = self.summary(final)
        _write_json(self.root / "summary.json", summary)
        return summary

    def close(self) -> None:
        self._planner_pool.shutdown(wait=False, cancel_futures=True)

    def summary(self, final: Mapping[str, Any]) -> dict[str, Any]:
        from queenbee.program.mint import audit_prompt_leaks

        gens = []
        g = 0
        redacted = 0
        while True:
            state = self.gen_state(g)
            if not state:
                break
            gens.append({"gen": g, "status": state.get("status"), "verdicts": state.get("verdicts"),
                         "best_after": state.get("best_after"),
                         "spent_evolution": state.get("spent_evolution"),
                         "exec_by_purpose": state.get("exec_by_purpose"),
                         "fresh_parent_reps": (state.get("r1") or {}).get("fresh_parent_reps")})
            redacted += sum(1 for b in state.get("briefs") or [] if b.get("prompt_redacted"))
            g += 1
        dirs = [p for p in (self.root / "mints", self.root / "prompts") if p.is_dir()]
        leak_hits = (audit_prompt_leaks(dirs, list(self.pool.split.V) + list(_common.task_test_ids()))
                     if dirs else [])
        snap = self.meter.snapshot()
        entries = list(self.ledger.entries)
        executed = [e for e in entries if e.get("reached_r2")]
        evo = self.evolution_record()
        out = {
            "arm": self.cfg.arm, "variant": self.cfg.variant, "split": self.pool.split.to_dict(),
            "run_status": self.run_status(),
            "evolution": {k: evo.get(k) for k in ("reason", "status", "budget_matched",
                                                  "remaining_evolution", "generations")},
            "champion": final.get("champion"),
            "champion_V": (final.get("V_scores") or {}).get(final.get("champion") or "v1"),
            "final_incomplete": final.get("incomplete"),
            "generations": gens, "verdicts": dict(Counter(e.get("verdict") for e in entries)),
            "ledger": self.ledger.summary(),
            "units_behind_r2_verdicts": dict(Counter(int(e.get("n_r2_units_observed") or 0)
                                                     for e in executed)),
            "prompts_redacted": redacted,
            "spent": {"evolution": self.spent("evolution"), "final": self.spent("final"),
                      "total": self.spent("total"), "budget": self.cfg.budget},
            "meter": snap.get("totals"),
            "counters": snap.get("counters"),
            "planner": {k: snap["planner"].get(k)
                        for k in ("calls", "prompt_tokens", "completion_tokens", "unknown_usage_calls")},
            "racer": {k: v for k, v in self.racer.summary().items() if k != "thresholds"},
            "outage_guard": ({"pauses": self.guard.pauses, "probes": self.guard.probes}
                             if self.guard is not None else None),
            "leak_hits": leak_hits,
            "fake": self.cfg.fake,
        }
        if self.cfg.final_coverage_topup:  # key present only with --final-coverage-topup
            out["coverage_topup"] = final.get("coverage_topup")
        return out


# --------------------------------------------------------------------------- #
# Public entry point + CLI
# --------------------------------------------------------------------------- #


def run_evo(config: EvoConfig, deps: EvoDeps | None = None) -> dict[str, Any]:
    """Run (or resume) one evolution run; returns its summary.  Raises
    :class:`InfraOutage` when the service stays unreachable (resumable)."""

    return EvoRun(config, deps).run()


def write_split_file(out: Path, *, split_seed: int, unit_sources: Sequence[Path],
                     ladder_manifests: Sequence[Path],
                     benchmarks_dir: Path | None = None,
                     v1_arm: str = "v1",
                     trace_dirs: Sequence[Path] = ()) -> dict[str, Any]:
    """``--write-split``: draw the T / V split of ``split_seed``
    (:func:`default_split`) from the given census / ladder files and write it
    with their sha256, so every arm run with this seed uses the same split.
    An existing file holding another split is never overwritten."""

    from queenbee.evo.units import load_units

    task = get_task()
    # a task that resolves its own instances needs no ladder manifest
    needs_manifest = task.instance_path is None
    if not unit_sources or (needs_manifest and not ladder_manifests):
        raise EvoStateError("--write-split needs --units-from"
                            + (" and --ladder-manifest" if needs_manifest else ""))
    from queenbee.evo.units import (
        near_miss_units_from_traces,
        structural_units_from_traces,
    )

    evidence = bool(trace_dirs) and task.trace_unit_evidence
    near = near_miss_units_from_traces(trace_dirs) if evidence else {}
    struct = structural_units_from_traces(trace_dirs) if evidence else {}
    units = load_units(list(unit_sources), v1_arm=v1_arm, manifests=list(ladder_manifests),
                       benchmarks_dir=(Path(benchmarks_dir) if benchmarks_dir is not None
                                       else default_benchmarks_dir()),
                       near_miss_units=near, structural_units=struct)
    split = default_split(units, int(split_seed))
    assert_not_test(list(split.T) + list(split.V), where="write split")
    doc = split.to_dict() | {
        "split_seed": int(split_seed), "generated_at": _now(),
        "unit_sources": _pinned(unit_sources), "ladder_manifests": _pinned(ladder_manifests),
    }
    out = Path(out)
    if out.exists():
        old = _read_json(out) or {}
        if (list(old.get("T") or []), list(old.get("V") or [])) != (list(split.T), list(split.V)):
            raise EvoStateError(f"{out} exists with another split; refusing to overwrite")
        return old
    _write_json(out, doc)
    return doc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m queenbee.evo.loop",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default=None, choices=ARMS,
                   help="full: diagnosis + ledger + archive; mf_elite: all three off "
                        "(required unless --write-split)")
    p.add_argument("--no-diag", action="store_true", help="switch the diagnosis component off")
    p.add_argument("--no-ledger", action="store_true", help="switch the ledger component off")
    p.add_argument("--no-archive", action="store_true", help="switch the archive component off")
    p.add_argument("--split-seed", type=int, required=True,
                   help="seed of the train / validation template split")
    p.add_argument("--split-file", default=None,
                   help='pinned split JSON {"T": [...], "V": [...]} (required for real runs; '
                        "see --write-split)")
    p.add_argument("--write-split", default=None, metavar="OUT",
                   help="write the split of --split-seed drawn from --units-from / "
                        "--ladder-manifest to OUT and exit")
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                   help="execution-equivalents, final-selection reserve included "
                        "(default %(default)s)")
    p.add_argument("--final-reserve", type=float, default=FINAL_RESERVE,
                   help="part of --budget kept for final selection (default %(default)s)")
    p.add_argument("--root", default=None, help="run directory (required unless --write-split)")
    p.add_argument("--fake", action="store_true",
                   help="offline run: fake worker provider and fake planner; without "
                        "--units-from a synthetic census is used")
    p.add_argument("--resume", action="store_true", help="continue the run in --root")
    p.add_argument("-K", "--K", type=int, default=K_DEFAULT, dest="K",
                   help="proposals per generation (default %(default)s)")
    p.add_argument("--units-from", action="append", default=[], metavar="CENSUS",
                   help="census of the seed program: result JSON of queenbee.evaluate "
                        "(repeatable; required for real runs)")
    p.add_argument("--ladder-manifest", action="append", default=[], metavar="MANIFEST",
                   help="unit ladder manifest the census ran on (repeatable; required for "
                        "real runs)")
    p.add_argument("--trace-dir", action="append", default=[], metavar="DIR",
                   help="trace directory of the census runs (repeatable)")
    p.add_argument("--thresholds", default=None, metavar="JSON",
                   help="racing thresholds JSON (default: built-in values)")
    p.add_argument("--benchmarks-dir", default=None,
                   help="Silo-Bench benchmarks directory (default: $SILO_BENCH_DIR, else "
                        "third_party/acl26-silo-bench/benchmarks)")
    p.add_argument("--planner-model", default=None,
                   help=f"planner model (default: ${PLANNER_MODEL_ENV})")
    p.add_argument("--planner-effort", default="high",
                   help="reasoning effort sent with planner requests; an empty string omits "
                        "it (default %(default)s)")
    p.add_argument("--planner-base-url", default=None,
                   help="planner endpoint (default: $OPENAI_BASE_URL)")
    p.add_argument("--planner-api-key-env", default=None, metavar="ENV",
                   help=f"environment variable holding the planner API key "
                        f"(default: {DEFAULT_API_KEY_ENV})")
    p.add_argument("--worker-model", default=None,
                   help=f"worker model (default: ${WORKER_MODEL_ENV}); workers use "
                        f"$OPENAI_BASE_URL and ${DEFAULT_API_KEY_ENV}")
    p.add_argument("--parallel-cases", type=int, default=6,
                   help="executions run in parallel (default %(default)s)")
    p.add_argument("--planner-concurrency", type=int, default=PLANNER_CONCURRENCY,
                   help=f"planner calls in flight, 1..{PLANNER_CONCURRENCY} (default %(default)s)")
    p.add_argument("--max-concurrent", type=int, default=9,
                   help="cap on concurrent LLM requests made by this process "
                        "(default %(default)s)")
    p.add_argument("--request-timeout", type=float, default=5400.0,
                   help="worker / planner request timeout in seconds; OPENAI_TIMEOUT follows "
                        "it unless set (default %(default)s)")
    p.add_argument("--max-infra-reruns", type=int, default=MAX_INFRA_RERUNS,
                   help="re-runs of an execution that failed for infrastructure reasons "
                        "(default %(default)s)")
    p.add_argument("--infra-backoff", type=float, default=60.0,
                   help="seconds before such a re-run (default %(default)s)")
    p.add_argument("--outage-wait", type=float, default=DEFAULT_OUTAGE_WAIT_S,
                   help="seconds to wait out a worker outage before stopping resumable with "
                        "exit code 75 (default %(default)s)")
    p.add_argument("--outage-backoff", type=float, default=DEFAULT_OUTAGE_BACKOFF_S,
                   help="first probe delay in seconds during an outage (default %(default)s)")
    p.add_argument("--planner-max-tokens", type=int, default=DEFAULT_PLANNER_MAX_COMPLETION_TOKENS,
                   help="output token cap per planner call; 0 = none (default %(default)s)")
    p.add_argument("--planner-deadline", type=float, default=DEFAULT_PLANNER_DEADLINE_S,
                   help="wall clock per planner call in seconds; 0 = none (default %(default)s)")
    p.add_argument("--no-async", action="store_true",
                   help="mint generation g+1 only after generation g finished")
    p.add_argument("--no-traces", action="store_true", help="do not write execution traces")
    p.add_argument("--max-generations", type=int, default=DEFAULT_MAX_GENERATIONS,
                   help="stop cap: generations (default %(default)s)")
    p.add_argument("--max-idle-generations", type=int, default=DEFAULT_MAX_IDLE_GENERATIONS,
                   help="stop cap: consecutive generations without a scored execution "
                        "(default %(default)s)")
    p.add_argument("--inherit-program", action="append", default=[], metavar="NAME=PATH",
                   help="seed the archive with an earlier program: it joins as a root and "
                        "gets one paid run on every frontier train unit (repeatable)")
    p.add_argument("--vacuous-guard", action="store_true",
                   help="with no guard unit in the pool the verdict's guard condition holds "
                        "instead of failing")
    p.add_argument("--local-only-diag", action="store_true",
                   help="diagnosis cards flag wrong agents that answered from their own "
                        "shard alone")
    p.add_argument("--attempt-diag", action="store_true",
                   help="diagnosis cards count wrong agents that already held the right "
                        "answer, and ledger entries keep the child's diagnosis after the change")
    p.add_argument("--irreducible-templates", default="", metavar="T1,T2",
                   help="comma list of templates whose gold answers carry a rounding floor: "
                        "never targeted")
    p.add_argument("--econ-neutral", action="store_true",
                   help="an economize proposal that kept S but saved no tokens is "
                        "inconclusive, not refuted")
    p.add_argument("--final-coverage-topup", type=int, nargs="?", const=DEFAULT_TOPUP_K,
                   default=0, metavar="K",
                   help=f"before final selection, up to K high-scoring programs measured on "
                        f"fewer than {MIN_FINAL_UNITS} train units get runs on more units, "
                        f"paid from the final reserve (the bare flag means K={DEFAULT_TOPUP_K}; "
                        "default 0 = off)")
    p.add_argument("--explore-on-lockin", action="store_true",
                   help="while the confirmed improvements concentrate in one mechanism family, "
                        "one proposal per generation must try another family")
    p.add_argument("--preserve-dups", action="store_true",
                   help="the planner's API card asks that forwarded raw data keep repeated "
                        "values, and diagnosis cards count kept data with multiplicity")
    p.add_argument("--python-max-rounds", type=int, default=PYTHON_MAX_ROUNDS,
                   help="round cap of a team program (default %(default)s)")
    add_task_argument(p)
    return p


def config_from_args(args: argparse.Namespace) -> EvoConfig:
    return EvoConfig(
        root=Path(args.root), arm=args.arm, split_seed=int(args.split_seed),
        budget=float(args.budget), final_reserve=float(args.final_reserve),
        no_diag=bool(args.no_diag), no_ledger=bool(args.no_ledger), no_archive=bool(args.no_archive),
        fake=bool(args.fake), resume=bool(args.resume), K=int(args.K),
        split_file=Path(args.split_file) if args.split_file else None,
        unit_sources=tuple(Path(p) for p in args.units_from),
        ladder_manifests=tuple(Path(p) for p in args.ladder_manifest),
        trace_dirs=tuple(Path(p) for p in args.trace_dir),
        thresholds_file=Path(args.thresholds) if args.thresholds else None,
        benchmarks_dir=Path(args.benchmarks_dir) if args.benchmarks_dir else None,
        planner_model=args.planner_model or None, planner_effort=args.planner_effort or None,
        worker_model=args.worker_model or None,
        planner_base_url=args.planner_base_url or None,
        planner_api_key_env=args.planner_api_key_env or None,
        parallel_cases=int(args.parallel_cases),
        planner_concurrency=int(args.planner_concurrency), request_timeout=float(args.request_timeout),
        max_infra_reruns=int(args.max_infra_reruns), infra_backoff_s=float(args.infra_backoff),
        planner_backoff_s=0.0 if args.fake else 60.0,
        planner_max_completion_tokens=args.planner_max_tokens, planner_deadline_s=args.planner_deadline,
        traces=not args.no_traces, async_mint=not args.no_async,
        max_generations=int(args.max_generations),
        max_idle_generations=int(args.max_idle_generations),
        outage_wait_s=float(args.outage_wait), outage_backoff_s=float(args.outage_backoff),
        inherit_programs=tuple(getattr(args, "inherit_program", None) or ()),
        vacuous_guard=bool(getattr(args, "vacuous_guard", False)),
        local_only_diag=bool(getattr(args, "local_only_diag", False)),
        attempt_diag=bool(getattr(args, "attempt_diag", False)),
        irreducible_templates=tuple(t.strip() for t in str(getattr(args, "irreducible_templates", "") or "").split(",") if t.strip()),
        python_max_rounds=int(getattr(args, "python_max_rounds", None) or PYTHON_MAX_ROUNDS),
        task=getattr(args, "task", None),
        **optional_behaviours(args),
    )


def optional_behaviours(args: argparse.Namespace) -> dict[str, Any]:
    """The four optional behaviours of :class:`EvoConfig` from the CLI flags
    (all off by default)."""

    return {
        "econ_neutral": bool(getattr(args, "econ_neutral", False)),
        "final_coverage_topup": int(getattr(args, "final_coverage_topup", None) or 0),
        "explore_on_lockin": bool(getattr(args, "explore_on_lockin", False)),
        "preserve_dups": bool(getattr(args, "preserve_dups", False)),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, _task = parse_task_args(parser, argv)

    if args.write_split:
        doc = write_split_file(
            Path(args.write_split), split_seed=int(args.split_seed),
            unit_sources=[Path(p) for p in args.units_from],
            ladder_manifests=[Path(p) for p in args.ladder_manifest],
            benchmarks_dir=Path(args.benchmarks_dir) if args.benchmarks_dir else None,
            trace_dirs=[Path(p) for p in args.trace_dir],
        )
        print(json.dumps({"split_file": str(args.write_split), "T": doc.get("T"), "V": doc.get("V")},
                         indent=1), flush=True)
        return doc
    if not args.arm or not args.root:
        parser.error("--arm and --root are required (unless --write-split)")
    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    missing = missing_settings(cfg)
    if missing:
        parser.error("a real run needs " + ", ".join(missing) + " (--fake runs offline)")
    if not cfg.fake:
        from exp_graph.llm.concurrency import configure_global_llm_concurrency

        configure_global_llm_concurrency(int(args.max_concurrent))
    from queenbee.program.clients import http_timeout_defaults

    # the HTTP timeouts of planner and workers follow --request-timeout unless set
    with http_timeout_defaults(cfg.request_timeout):
        summary = run_evo(cfg)
    print(json.dumps({k: summary.get(k) for k in ("arm", "variant", "run_status", "champion",
                                                  "champion_V", "verdicts", "spent", "planner")},
                     indent=1, default=str), flush=True)
    return summary


def cli() -> None:
    """Exit codes: 0 finished (budget-matched), 3 finished but FAILED (a
    stop cap fired before the budget was spent), 75 service outage (state
    saved; re-run with ``--resume`` to continue), 2 a usage error, 1 any
    other abort.  An outage exits hard so it never waits for in-flight
    planner calls (everything is on disk)."""

    try:
        summary = main()
    except InfraOutage as exc:
        print(json.dumps({"status": "resume_later", "reason": str(exc)[:500]}), flush=True)
        sys.stderr.flush()
        os._exit(EXIT_RESUME_LATER)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - any abort: report, never wait on planner threads
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    if isinstance(summary, dict) and str(summary.get("run_status") or "ok").startswith("failed"):
        sys.exit(EXIT_FAILED_RUN)


__all__ = [
    "ARMS",
    "EvoConfig",
    "EvoDeps",
    "EvoRun",
    "EvoStateError",
    "FakeEvoPlanner",
    "InfraOutage",
    "Job",
    "OutageGuard",
    "UnitPool",
    "arm_flags",
    "batch_id",
    "batch_key",
    "cli",
    "econ_no_saving",
    "economize_verdict",
    "fake_census",
    "fake_mutation",
    "is_connectivity_row",
    "is_outage_error",
    "missing_settings",
    "optional_behaviours",
    "parent_policy",
    "redact_evidence_values",
    "run_evo",
    "write_split_file",
]


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    cli()
