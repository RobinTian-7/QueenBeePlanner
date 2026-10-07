"""Shared constants, TEST guards and execution helpers of QueenBee-Evo.

Template ids (the sealed TEST set and the DEV set), the rung table, the guards
that refuse TEST templates, the behaviour fingerprint (exec-cache / no-op /
duplicate key), the append-only execution log, the default executor and the
loaders for trace dumps and x5 ladder manifests.

The constants are the Silo-Bench values; the ``task_*`` readers return the
active task's values (:mod:`queenbee.tasks`), falling back to them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from queenbee.tasks import get_task

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The 30 Silo-Bench templates, in upstream order.
ALL_TEMPLATE_IDS: tuple[str, ...] = tuple(
    [f"I-{i:02d}" for i in range(1, 11)]
    + [f"II-{i}" for i in range(11, 21)]
    + [f"III-{i}" for i in range(21, 31)]
)
#: The 12 sealed TEST templates: never read, executed, prompted or listed.
#: This tuple is the only place in the module that names them.
TEST_IDS: tuple[str, ...] = (
    "I-04", "I-05", "I-09", "I-10", "II-14", "II-15", "II-19", "II-20",
    "III-24", "III-25", "III-29", "III-30",
)
#: The 18 DEV templates: every template that is not TEST.
DEV_IDS: tuple[str, ...] = tuple(t for t in ALL_TEMPLATE_IDS if t not in TEST_IDS)

#: Model ids have no built-in default: the CLI flags, else these variables.
PLANNER_MODEL_ENV = "QUEENBEE_PLANNER_MODEL"
WORKER_MODEL_ENV = "QUEENBEE_WORKER_MODEL"
WORKER_MODEL: str = (os.environ.get(WORKER_MODEL_ENV) or "").strip()

GOAL = "all_agents"
WORKER_CONTRACT = "message_only_v2"
PYTHON_MAX_ROUNDS = 64

# Rung tables: team size, cost in execution-equivalents and sort position
# (x5 holds twice the o5 data at the same team size).
RUNGS: dict[str, int] = {"o5": 5, "x5": 5, "o10": 10}
RUNG_COST: dict[str, int] = {"o5": 1, "x5": 1, "o10": 2}
RUNG_ORDER: dict[str, int] = {"o5": 0, "x5": 1, "o10": 2}
#: Team size of the mint dry run, the planner request and the prompt's budget line.
MINT_N_AGENTS = 5

# --------------------------------------------------------------------------- #
# The active task's values (read at call time)
# --------------------------------------------------------------------------- #


def task_goal() -> str:
    """The active task's information goal (default :data:`GOAL`)."""

    goal = get_task().goal
    return GOAL if goal is None else goal


def task_rungs() -> dict[str, int]:
    """Rung -> team size of the active task (default :data:`RUNGS`)."""

    rungs = get_task().rungs
    return RUNGS if rungs is None else dict(rungs)


def task_rung_order() -> dict[str, int]:
    """Rung -> sort position of the active task (default :data:`RUNG_ORDER`)."""

    order = get_task().rung_order
    return RUNG_ORDER if order is None else dict(order)


def task_test_ids() -> tuple[str, ...]:
    """The active task's sealed TEST template ids (default :data:`TEST_IDS`)."""

    ids = get_task().test_ids
    return TEST_IDS if ids is None else tuple(ids)


def task_dev_ids() -> tuple[str, ...]:
    """The active task's development template ids (default :data:`DEV_IDS`)."""

    ids = get_task().dev_ids
    if ids is None:
        return DEV_IDS
    return tuple(str(t) for t in (ids() if callable(ids) else ids))


def task_mint_n_agents() -> int:
    """Team size of the active task's mint dry run (default :data:`MINT_N_AGENTS`)."""

    n = get_task().mint_n_agents
    return MINT_N_AGENTS if n is None else int(n)


def _test_set() -> frozenset[str]:
    ids = get_task().test_ids
    return _TEST_SET if ids is None else frozenset(ids)


def _template_re() -> re.Pattern[str]:
    pattern = get_task().template_pattern
    return _TEMPLATE_RE if pattern is None else re.compile(pattern)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

_TEST_SET = frozenset(TEST_IDS)
_TEMPLATE_RE = re.compile(r"(?<![A-Za-z0-9-])(I{1,3}-\d{2})(?!\d)")


class LeakageGuardError(RuntimeError):
    """A held-out id reached a guarded code path: a sealed TEST template, a
    non-dev unit, a unit on the wrong side of the T / V split or a held-out
    template id in a planner prompt."""


class BudgetExceeded(RuntimeError):
    """A batch would exceed its execution cap (nothing ran)."""


class ManifestError(RuntimeError):
    """An x5 ladder manifest entry has no instance path.  (Instance files
    that do not match their entries raise
    :class:`queenbee.evo.ladder.LadderError`.)"""


def templates_in(value: Any) -> list[str]:
    """Every template id that occurs in ``str(value)`` (a unit id, a file
    name, free text)."""

    return _template_re().findall(str(value or ""))


def template_of(unit_id: Any) -> str:
    """``"II-11@x5"`` -> ``"II-11"``."""

    return str(unit_id or "").split("@", 1)[0].strip()


def rung_of(unit_id: Any) -> str:
    """``"II-11@x5"`` -> ``"x5"``; an id without ``@`` is an o5 unit."""

    text = str(unit_id or "")
    return text.split("@", 1)[1].strip() if "@" in text else "o5"


def assert_not_test(ids: Any, *, where: str) -> None:
    """Raise :class:`LeakageGuardError` when any id (or any template id inside
    it) is a sealed TEST template."""

    if ids is None:
        return
    if isinstance(ids, (str, bytes, Path)):
        ids = [ids]
    sealed = _test_set()
    bad = sorted(
        {t for item in ids for t in templates_in(item) if t in sealed}
        | {template_of(i) for i in ids if template_of(i) in sealed}
    )
    if bad:
        raise LeakageGuardError(
            f"LEAKAGE_GUARD ({where}): TEST templates {bad} are sealed and "
            "never used by evo"
        )


def make_unit(template: str, rung: str) -> str:
    """``(template, rung)`` -> unit id; refuses TEST and non-dev templates and
    unknown rungs."""

    assert_not_test([template], where="make_unit")
    if template not in task_dev_ids():
        raise LeakageGuardError(f"make_unit: {template!r} is not a dev template")
    if rung not in task_rungs():
        raise ValueError(f"unknown rung {rung!r}")
    return f"{template}@{rung}"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str))
    os.replace(tmp, path)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _same_num(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a is None and b is None


def _max_rounds(n_agents: int, python_max_rounds: int = PYTHON_MAX_ROUNDS) -> int:
    from queenbee.program.budgets import PythonRunBudgets

    return int(PythonRunBudgets.for_rounds(python_max_rounds, n_agents=n_agents).max_rounds)


# --------------------------------------------------------------------------- #
# Programs: the seed program (v1) and behaviour fingerprints
# --------------------------------------------------------------------------- #


def v1_source() -> str:
    """Source of the Silo-Bench seed program (v1) as the census runs it: the
    phase-structured seed of the ``message_only_v2`` worker contract.  The
    text evolution starts from (:func:`queenbee.evo.seed.evo_seed_source`)
    adds two capability extensions that leave the behaviour fingerprint
    unchanged."""

    from queenbee.program.execute import seed_python_source

    return seed_python_source(WORKER_CONTRACT, base="sfs_phase")[2]


#: Scheme of :func:`behavior_fingerprint` with its default agent counts.
FP_SCHEME = "screen.behavior_fp@n2,5,10"
#: Agent counts of :func:`behavior_fingerprint` (the task may set others).
FP_NS: tuple[int, ...] = (2, 5, 10)


def fingerprint_ns() -> tuple[int, ...]:
    """Agent counts of the active task's behaviour fingerprint."""

    ns = get_task().fp_ns
    return FP_NS if ns is None else tuple(ns)


def fingerprint_scheme() -> str:
    """Scheme id of the active task's behaviour fingerprint."""

    scheme = get_task().fp_scheme
    return FP_SCHEME if scheme is None else str(scheme)


def behavior_fingerprint(source: str, *, ns: Sequence[int] | None = None) -> str | None:
    """Exec-cache / no-op / duplicate key of a program.

    :func:`queenbee.evo.screen.behavior_fp` at every agent count, joined by
    :func:`queenbee.evo.screen.fp_key`: per-round edges, modes and work
    instructions PLUS the submit round, the ``main()`` variant and the
    globals ``main()`` reads, so a program that only inserts an idle round
    (bodies delivered in it are lost) differs from its parent.  ``ns``: agent
    counts (default: the active task's, :func:`fingerprint_ns`).  None when
    the program cannot be simulated."""

    from queenbee.evo.screen import behavior_fp, fp_key

    goal = task_goal()
    try:
        fps = {int(n): behavior_fp(source, int(n), max_rounds=_max_rounds(int(n)), goal=goal)
               for n in (fingerprint_ns() if ns is None else ns)}
        key = fp_key(fps)
    except Exception:  # noqa: BLE001 - unsimulable = no fingerprint
        return None
    return None if key is None else "s:" + _sha256(key)[:24]


# --------------------------------------------------------------------------- #
# Traces, instances, ladder manifests
# --------------------------------------------------------------------------- #


class TraceIndex:
    """Trace JSONs (``QB_TRACE_DIR`` dumps) indexed by (case_id, source
    sha12).  TEST traces are skipped by FILE NAME, before any file is opened;
    traces of non-dev templates are skipped too."""

    def __init__(self, dirs: Sequence[Path]) -> None:
        self.by_key: dict[tuple[str, str], list[Path]] = {}
        self.skipped_test = 0
        self.skipped_other = 0
        for directory in dirs:
            directory = Path(directory)
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                parts = path.stem.split("_")
                if len(parts) < 3:
                    continue
                case = parts[0]
                sealed = _test_set()
                if template_of(case) in sealed or any(
                    t in sealed for t in templates_in(path.name)
                ):
                    self.skipped_test += 1
                    continue
                if template_of(case) not in task_dev_ids():
                    self.skipped_other += 1
                    continue
                self.by_key.setdefault((case, parts[1]), []).append(path)
        self._cache: dict[Path, dict[str, Any]] = {}

    def load(self, path: Path) -> dict[str, Any] | None:
        if path not in self._cache:
            data = _read_json(path)
            if not isinstance(data, dict):
                return None
            assert_not_test([data.get("case_id")], where="trace load")
            self._cache[path] = data
        return self._cache[path]

    def match(self, case_id: str, sha12: str, row: Mapping[str, Any]) -> dict[str, Any] | None:
        """The trace of exactly this row (same S, C and prompt tokens)."""

        for path in self.by_key.get((str(case_id), str(sha12)), []):
            trace = self.load(path)
            facts = dict((trace or {}).get("facts") or {})
            if (
                _same_num(facts.get("S"), row.get("S"))
                and _same_num(facts.get("C"), row.get("C"))
                and _same_num(facts.get("prompt_tokens"), row.get("prompt_tokens"))
            ):
                return trace
        return None


def load_ladder_manifest(path: Path) -> dict[str, Path]:
    """x5 entries of a ladder manifest -> ``{"II-11@x5": instance path}``.
    Entries come from :func:`queenbee.evo.ladder.load_ladder_manifest` (TEST
    guard on every label and every file's own case id, instance-hash
    re-check); a TEST (or non-dev) template in the manifest raises."""

    from queenbee.evo.ladder import load_ladder_manifest as load_entries

    out: dict[str, Path] = {}
    for entry in load_entries(path):
        label = str(entry["case_id"])
        template = template_of(entry.get("template") or label)
        assert_not_test([label, template, entry.get("path", "")], where="ladder manifest")
        rung = str(entry.get("rung") or (rung_of(label) if "@" in label else "x5"))
        if rung != "x5" or template not in task_dev_ids():
            continue
        if not entry.get("path"):
            raise ManifestError(f"ladder manifest entry {label!r} has no path")
        out[make_unit(template, "x5")] = Path(entry["path"])
    return out


# --------------------------------------------------------------------------- #
# Execution log + default executor
# --------------------------------------------------------------------------- #


class ExecLog:
    """Append-only JSONL of every execution (infra rows and cache hits too).
    A torn last line (kill -9) is ignored on load."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except ValueError:
                    continue

    def append(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.records.append(record)

    def for_key(self, key: str) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.records if r.get("key") == key]

    def scored(self, key: str) -> dict[str, Any] | None:
        for record in reversed(self.for_key(key)):
            if not (record.get("row") or {}).get("infra"):
                return record
        return None


def task_score_hooks() -> dict[str, Any]:
    """The active task's ``score_fn`` / ``diag_fn`` as keyword arguments of
    ``evaluate_python_source_on_cases`` (only the ones it sets)."""

    task = get_task()
    return {name: fn for name, fn in (("score_fn", task.score_fn), ("diag_fn", task.diag_fn))
            if fn is not None}


def task_finalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The active task's ``finalize_row`` applied to one execution row (a
    copy of the row when the task sets none)."""

    finalize = get_task().finalize_row
    return dict(row) if finalize is None else dict(finalize(dict(row)))


def real_execute(req: Mapping[str, Any]) -> dict[str, Any]:
    """One execution of ``req["source"]`` on ``req["instance"]`` through
    ``queenbee.program.execute.evaluate_python_source_on_cases`` (the active
    task's goal, scoring hooks and ``finalize_row``).

    ``req["cfg"]`` supplies ``llm_provider``, ``worker_model``,
    ``request_timeout`` and ``python_max_rounds``; ``req["job"]`` supplies
    ``program_id`` and ``unit_id``."""

    from queenbee.program.execute import evaluate_python_source_on_cases
    from queenbee.program.budgets import PythonRunBudgets

    cfg = req["cfg"]
    instance = req["instance"]
    n = int(instance.n_agents)
    rows = evaluate_python_source_on_cases(
        source=req["source"],
        instances=(instance,),
        seeds=(int(req["seed"]),),
        arm_label=f"{req['job'].program_id}:{req['job'].unit_id}",
        llm_provider=cfg.llm_provider,
        worker_model=cfg.worker_model,
        goal=task_goal(),
        worker_contract=WORKER_CONTRACT,
        max_parallel_cases=1,
        max_parallel_agents=n,
        request_timeout=cfg.request_timeout,
        artifacts_dir=Path(req["artifacts_dir"]),
        budgets=PythonRunBudgets.for_rounds(cfg.python_max_rounds, n_agents=n),
        **task_score_hooks(),
    )
    return task_finalize_row(rows[0])


def task_execute(req: Mapping[str, Any]) -> dict[str, Any]:
    """The default executor: the active task's ``execute``, else
    :func:`real_execute`."""

    execute = get_task().execute
    return dict((real_execute if execute is None else execute)(req))
