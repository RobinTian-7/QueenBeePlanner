"""Evaluate frozen team programs on Silo-Bench cases.

Runs one or more *arms* on a list of cases for ``--repeats`` execution
repeats (cases in parallel) and writes one JSON result file with a row per
(arm, repeat, case).  An arm is one of:

* ``--program NAME=PATH``  a team program file (e.g. a run's champion.py)
* ``--seed-program NAME``  the seed program of the evolution loop (the loop
  reads the arm named ``v1`` of a census file)

Typical use::

    # census of the seed program on the cases of a ladder manifest
    python -m queenbee.evaluate --split train --agents 5 \\
        --manifest MANIFEST.json --seed-program v1 --model MODEL \\
        --out census_n5.json
    # an evolved champion on the held-out TEST templates
    python -m queenbee.evaluate --split test --agents 5 --repeats 3 \\
        --program champion=RUN_DIR/champion.py --model MODEL --out test_n5.json

Cases: ``--split train`` = the development templates (Silo-Bench: 18),
``--split test`` = the held-out TEST templates (Silo-Bench: 12), at team
size ``--agents``.  ``--cases`` narrows the list; ``--manifest`` reads a
JSON case list (``[{"case_id", "path", "template"?, "pool"?}, ...]`` or
``{"cases": [...]}``; ``path`` is relative to the manifest file), e.g. a
ladder manifest, and ``--cases`` then filters it.  Every case must belong
to the ``--split`` side, and one ``--out`` file holds the cases of one side
only.

``--task`` names the task family (default ``silo``): its template ids,
instances, seed program, goal, scoring, row fields, summary fields and
option defaults apply (``--task cf``: Count-Frequency, whose summaries add
the mean RMSE of the scored rows and the count of format failures).

Workers call an OpenAI-compatible endpoint (``OPENAI_BASE_URL`` /
``OPENAI_API_KEY``; model ``--model`` or ``QUEENBEE_WORKER_MODEL``).  Up to
``--parallel-cases`` x ``--max-parallel-agents`` worker requests are in
flight, each limited to ``--request-timeout`` seconds; unless already set,
``OPENAI_TIMEOUT`` (the workers' HTTP timeout) follows it and
``OPENAI_CONNECT_TIMEOUT`` is 30 s for the run.  ``--llm fake`` runs an
offline worker that never solves a task (no network, no key), to smoke-test
a command.  With ``QB_TRACE_DIR`` set, every completed execution also
writes a trace JSON there (it holds the case's ground truth).

Resume / integrity rules for one ``--out`` file:

* **One run fingerprint per file** (agents, worker model, worker reasoning
  effort ``OPENAI_REASONING_EFFORT``, goal, worker contract, temperature,
  python budgets; ``--task`` and ``--llm`` when not the defaults).  A
  relaunch whose fingerprint differs is refused: use another ``--out``.
* **One instance per case label.**  Every row and ``config.case_meta`` carry
  the sha256 of the instance content; a relaunch that maps a stored label to
  a different instance is refused.
* **Union of case lists.**  A relaunch may run a subset or a superset of the
  stored cases: requested cases that are missing, or whose row is a
  retryable infra row, are executed; every stored row is kept in ``rows``
  (a scored row is never replaced or hidden).  ``config.cases`` is the union.
* **Infra rows.**  Transport and endpoint failures (e.g. connection errors,
  timeouts, rate limits, HTTP 429 / 502 / 503) give an infra row (not
  scored); anything else is a scored algorithm failure (S = 0).  An infra
  row is re-run on relaunch at most ``--max-infra-reruns`` (default 3)
  times; after that it is marked ``infra_final`` and never re-run.
  Replaced infra messages stay in the row's ``infra_history`` and each
  re-run pass is logged in the repeat's ``reruns``.
* **Per-case checkpoints.**  Rows are merged and the file saved as each case
  finishes, so a crash loses at most the in-flight cases.
* **Per-arm identity.**  Each arm stores ``source_sha256`` and ``max_rounds``;
  a relaunch that would mix another program / round cap into the arm is
  refused.

Every row records the ``max_rounds`` it ran under; the row of a completed
run also records ``rounds_executed``.  Each invocation is logged in
``config.invocations`` with the git sha, a dirty flag, the sha256 of ``git
diff HEAD`` and the sha256 of every module that determines a row.  Row
``seed`` values are bookkeeping labels only: they never reach the program or
the workers, so a re-run is not a same-seed replay.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from queenbee.paths import default_benchmarks_dir
from queenbee.tasks import DEFAULT_TASK, add_task_argument, get_task, parse_task_args

# Row seed labels: TEST_SEED_BASE + REPEAT_SEED_STRIDE * repeat + case index
# (bookkeeping only, see :func:`repeat_seeds`).
TEST_SEED_BASE = 1_000
REPEAT_SEED_STRIDE = 100
DEFAULT_MAX_INFRA_RERUNS = 3
#: Worker contract of every program the evolution loop writes (and of its seed).
WORKER_CONTRACT = "message_only_v2"
#: Environment variable naming the worker model when ``--model`` is not given.
WORKER_MODEL_ENV = "QUEENBEE_WORKER_MODEL"
#: Root of the source checkout (git and code provenance).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Modules whose code determines an evaluation row (hashed per invocation).
CODE_MODULES = (
    "queenbee.evaluate",
    "queenbee.evo.common",
    "queenbee.evo.ladder",
    "queenbee.evo.diagnosis",
    "queenbee.program.execute",
    "queenbee.program.budgets",
    "queenbee.program.phase_seed",
    "queenbee.bench.config",
    "queenbee.bench.engine",
    "queenbee.bench.instance",
    "queenbee.bench.scoring",
    "queenbee.bench.silo_bench",
    "queenbee.bench.silo_metrics",
    "queenbee.bench.silo_protocol",
    "queenbee.bench.silo_scoring",
    "queenbee.bench.task_bridge",
    "queenbee.bench.task_view",
    "queenbee.tasks",
    "queenbee.tasks.base",
    "queenbee.tasks.silo",
    "queenbee.tasks.cf",
    "queenbee.tasks.cf.cases",
    "queenbee.tasks.cf.diagnosis",
    "queenbee.tasks.cf.scoring",
    "queenbee.tasks.cf.seed",
    "exp_graph.llm.factory",
    "exp_graph.llm.openai_client",
    "exp_graph.llm.timeout",
    "exp_graph.mas.count_table_json",
    "exp_graph.mas.information_flow",
    "exp_graph.mas.leakage_audit",
    "exp_graph.mas.python_code",
    "exp_graph.mas.python_code_runner",
    "exp_graph.mas.python_code_generation",
    "exp_graph.mas.python_worker_bootstrap",
)


def _task_summary_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The active task's ``summary_fields`` of ``rows`` (none by default)."""

    hook = get_task().summary_fields
    return {} if hook is None else dict(hook(list(rows)))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One repeat's aggregate: infra rows are counted but excluded from the
    success rate and the means of S and C."""

    scored = [r for r in rows if not r.get("infra")]
    n = len(scored)
    out = {
        "cases_total": len(rows),
        "cases_scored": n,
        "infra": len(rows) - n,
        "infra_final": sum(1 for r in rows if r.get("infra") and r.get("infra_final")),
        "success_rate": (
            sum(bool(r.get("success")) for r in scored) / n if n else None
        ),
        "mean_S": (sum(float(r.get("S") or 0.0) for r in scored) / n if n else None),
        "mean_C": (sum(float(r.get("C") or 0.0) for r in scored) / n if n else None),
    }
    out.update(_task_summary_fields(rows))
    return out


def _summary(
    per_repeat: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]] = ()
) -> dict[str, Any]:
    """Arm summary over its repeat aggregates; the task's ``summary_fields``
    over ``rows`` (every row of the arm) are added."""

    out: dict[str, Any] = {"repeats": len(per_repeat)}
    for key in ("success_rate", "mean_S", "mean_C"):
        vals = [a[key] for a in per_repeat if a.get(key) is not None]
        if vals:
            out[key] = statistics.fmean(vals)
            out[f"{key}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    out.update(_task_summary_fields(rows))
    return out


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_provenance(repo: Path) -> dict[str, Any]:
    """HEAD sha, dirty flag (tracked files) and sha256 of ``git diff HEAD``
    of ``repo`` (``None`` values when ``repo`` is not a git checkout or git
    fails)."""

    unknown: dict[str, Any] = {"git_sha": None, "git_dirty": None, "git_diff_sha256": None}
    if not (Path(repo) / ".git").exists():
        return unknown

    def _git(*argv: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *argv],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    diff = _git("diff", "HEAD")
    return {
        "git_sha": sha.strip() if sha is not None else None,
        "git_dirty": (bool(status.strip()) if status is not None else None),
        "git_diff_sha256": (
            _sha256_bytes(diff.encode("utf-8")) if diff is not None else None
        ),
    }


def code_provenance(
    modules: Sequence[str] = CODE_MODULES, repo: Path | None = None
) -> dict[str, Any]:
    """sha256 of each module file (tracked or not) + one bundle digest."""

    files: dict[str, str | None] = {}
    for name in modules:
        path: str | None = None
        mod = sys.modules.get(name)
        if mod is not None:
            path = getattr(mod, "__file__", None)
        if path is None:
            try:
                spec = importlib.util.find_spec(name)
                path = spec.origin if spec is not None else None
            except (ImportError, ValueError):
                path = None
        key = name
        if path and repo is not None:
            try:
                key = str(Path(path).resolve().relative_to(repo.resolve()))
            except ValueError:
                key = name
        try:
            files[key] = _sha256_bytes(Path(path).read_bytes()) if path else None
        except OSError:
            files[key] = None
    bundle = _sha256_bytes(
        json.dumps(sorted(files.items()), separators=(",", ":")).encode("utf-8")
    )
    return {"code_sha256": bundle, "code_files": files}


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instance_sha256(instance: Any) -> str:
    """sha256 of an instance's full content (canonical JSON)."""

    if dataclasses.is_dataclass(instance) and not isinstance(instance, type):
        payload: Any = dataclasses.asdict(instance)
    elif isinstance(instance, Mapping):
        payload = dict(instance)
    else:
        payload = dict(vars(instance))
    text = json.dumps(
        payload, sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":")
    )
    return source_sha256(text)


# --------------------------------------------------------------------------
# cases: the fixed template split and manifests
# --------------------------------------------------------------------------


def split_templates(split: str) -> tuple[str, ...]:
    """Templates of one side of the active task's fixed split, in order:
    ``test`` = the held-out TEST templates (Silo-Bench: 12), ``train`` = the
    development templates (Silo-Bench: 18)."""

    from queenbee.evo.common import task_dev_ids, task_test_ids

    if split == "test":
        return tuple(task_test_ids())
    if split == "train":
        return tuple(task_dev_ids())
    raise ValueError(f"unknown split {split!r} (train | test)")


def check_split(
    split: str,
    cases: Sequence[str],
    instances: Mapping[str, Any],
    case_meta: Mapping[str, Mapping[str, Any]],
) -> None:
    """Refuse cases outside the ``split`` side: a case's label, its manifest
    template and its instance's own case id must all name a template of that
    side (a unit label maps to its template: ``'II-11@x5' -> 'II-11'``)."""

    from queenbee.evo.ladder import template_of

    allowed = set(split_templates(split))
    bad: list[str] = []
    for label in cases:
        names = (
            label,
            (case_meta.get(label) or {}).get("template"),
            getattr(instances.get(label), "case_id", None),
        )
        outside = sorted({template_of(str(n)) for n in names if n} - allowed)
        if outside:
            bad.append(f"{label} ({', '.join(outside)})")
    if bad:
        raise SystemExit(f"--split {split}: cases outside the {split} templates: {bad}")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Normalized manifest entries ``{case_id, path?, template?, pool?}``.

    Accepts a JSON list or ``{"cases": [...]}``; an entry is a case-id string
    (resolved like a ``--cases`` label: the task's ``instance_path``, else
    the benchmarks dir) or a mapping with ``case_id`` and an optional
    instance ``path`` (relative to the manifest's directory).
    Labels (``case_id``) must be unique: they key the result rows.
    """

    path = Path(path)
    data = json.loads(path.read_text())
    items = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError(f"manifest {path}: expected a non-empty case list")
    entries: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            entry: dict[str, Any] = {"case_id": item}
        elif isinstance(item, Mapping) and item.get("case_id"):
            entry = {k: v for k, v in item.items() if v is not None}
            entry["case_id"] = str(entry["case_id"])
            if entry.get("path"):
                p = Path(str(entry["path"]))
                entry["path"] = str(p if p.is_absolute() else (path.parent / p))
        else:
            raise ValueError(f"manifest {path}: bad entry {item!r}")
        entries.append(entry)
    labels = [e["case_id"] for e in entries]
    dupes = sorted({c for c in labels if labels.count(c) > 1})
    if dupes:
        raise ValueError(f"manifest {path}: duplicate case_id labels {dupes}")
    return entries


def filter_manifest(
    entries: Sequence[Mapping[str, Any]], wanted: Sequence[str]
) -> list[dict[str, Any]]:
    """Manifest entries restricted to ``wanted`` labels (manifest order);
    labels absent from the manifest are an error, never silently dropped."""

    labels = {str(e["case_id"]) for e in entries}
    unknown = [c for c in wanted if c not in labels]
    if unknown:
        raise ValueError(f"--cases not in the manifest: {unknown}")
    keep = set(wanted)
    return [dict(e) for e in entries if e["case_id"] in keep]


def load_instance_file(path: str | Path) -> Any:
    """One instance file -> instance: the active task's ``load_instance``,
    else a Silo-Bench instance JSON -> ``BenchmarkInstance`` (sanitized)."""

    from queenbee.bench.silo_bench import SiloBenchAdapter

    path = Path(path)
    loader = get_task().load_instance
    if loader is not None:
        return loader(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(data, dict) and "case_id" in data and "agent_configs" in data):
        raise ValueError(f"{path}: not a single Silo-Bench instance")
    return SiloBenchAdapter(path.parent)._to_instance(data)


def resolve_manifest_instances(
    entries: Sequence[Mapping[str, Any]],
    bench_instances: Mapping[str, Any],
    n_agents: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """``(label -> instance, label -> case meta)`` for manifest entries.  An
    entry's ``instance_sha256`` (e.g. a ladder manifest's) must match the
    instance it resolves to."""

    instances: dict[str, Any] = {}
    case_meta: dict[str, dict[str, Any]] = {}
    for entry in entries:
        label = str(entry["case_id"])
        if entry.get("path"):
            inst = load_instance_file(entry["path"])
        elif label in bench_instances:
            inst = bench_instances[label]
        else:
            raise ValueError(f"manifest case {label!r}: no path and not in benchmarks")
        if int(inst.n_agents) != int(n_agents):
            raise ValueError(
                f"manifest case {label!r}: instance has n={inst.n_agents}, "
                f"--agents={n_agents}"
            )
        want = entry.get("instance_sha256")
        if want and str(want) != instance_sha256(inst):
            raise ValueError(
                f"manifest case {label!r}: the instance differs from the "
                "manifest's instance_sha256"
            )
        instances[label] = inst
        meta = {
            k: entry[k]
            for k in ("path", "template", "pool", "tier")
            if entry.get(k) is not None
        }
        if inst.case_id != label:
            meta["instance_case_id"] = inst.case_id
        meta.setdefault("template", entry.get("template") or inst.case_id)
        case_meta[label] = meta
    return instances, case_meta


# --------------------------------------------------------------------------
# run fingerprint + case identity (refuse to mix incompatible rows)
# --------------------------------------------------------------------------


def run_fingerprint(
    *,
    agents: int,
    model: str,
    reasoning_effort: str | None,
    goal: str,
    worker_contract: str,
    budgets: Any,
    llm_provider: str = "openai",
    task: str = DEFAULT_TASK,
) -> dict[str, Any]:
    """Everything that must be identical for rows of one ``--out`` file
    (``llm_provider`` is recorded when it is not ``openai``, ``task`` when it
    is not the default task)."""

    fp: dict[str, Any] = {
        "agents": int(agents),
        "model": str(model),
        "reasoning_effort": None if reasoning_effort is None else str(reasoning_effort),
        "goal": str(goal),
        "worker_contract": str(worker_contract),
        "temperature": 0.0,
        "python_budgets": {
            k: int(getattr(budgets, k))
            for k in ("max_rounds", "max_model_calls", "max_completion_tokens", "max_messages")
        },
    }
    if llm_provider != "openai":
        fp["llm_provider"] = str(llm_provider)
    if task != DEFAULT_TASK:
        fp["task"] = str(task)
    return fp


def check_run_fingerprint(
    config: dict[str, Any], fp: Mapping[str, Any], *, has_rows: bool = False
) -> None:
    """Store ``fp`` in ``config`` on the first launch; refuse a relaunch
    whose fingerprint differs from the stored one, and a file that has rows
    but no stored fingerprint."""

    stored = config.get("fingerprint")
    if stored is None:
        if has_rows:
            raise SystemExit(
                "the result file has rows but no run fingerprint; use another --out"
            )
        config["fingerprint"] = dict(fp)
        return
    diffs = {
        k: (stored.get(k), fp.get(k))
        for k in sorted(set(stored) | set(fp))
        if stored.get(k) != fp.get(k)
    }
    if diffs:
        raise SystemExit(
            f"run fingerprint differs from the stored one {diffs} "
            "(stored, this launch); use another --out"
        )


def check_case_identity(
    result: Mapping[str, Any], current: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Refuse a relaunch that maps a stored case label to another instance.

    ``current``: label -> {"instance_sha256", "path"?, ...} of this launch.
    Stored identity = ``config.case_meta[label].instance_sha256`` and the
    rows' ``instance_sha256``; a label with rows but no stored hash is
    refused too.  Returns the merged case meta (stored labels kept, current
    ones updated)."""

    config = result.get("config") or {}
    stored_meta: dict[str, dict[str, Any]] = {
        str(k): dict(v) for k, v in (config.get("case_meta") or {}).items()
    }
    row_sha: dict[str, set[str]] = {}
    labels_with_rows: set[str] = set()
    for entry in (result.get("arms") or {}).values():
        for rep in (entry.get("repeats") or {}).values():
            for row in _all_rows(rep):
                label = str(row.get("case_id"))
                labels_with_rows.add(label)
                if row.get("instance_sha256"):
                    row_sha.setdefault(label, set()).add(str(row["instance_sha256"]))
    problems: list[str] = []
    merged = {k: dict(v) for k, v in stored_meta.items()}
    for label, cur in current.items():
        sha = str(cur["instance_sha256"])
        old = stored_meta.get(label) or {}
        known = set(row_sha.get(label, set()))
        if old.get("instance_sha256"):
            known.add(str(old["instance_sha256"]))
        if known and known != {sha}:
            problems.append(f"{label}: stored instance {sorted(known)} != {sha}")
            continue
        if not known and label in labels_with_rows:
            problems.append(f"{label}: stored rows carry no instance sha256")
            continue
        merged[label] = dict(old) | dict(cur)
    if problems:
        raise SystemExit(
            "case labels map to different instances than the stored rows "
            f"(use another --out): {problems}"
        )
    return merged


# --------------------------------------------------------------------------
# resume: re-run only infra / missing rows of a repeat
# --------------------------------------------------------------------------


def repeat_seeds(cases: Sequence[str], rep: int) -> dict[str, int]:
    """Seed label of every case of repeat ``rep`` (by position in ``cases``)."""

    return {
        c: TEST_SEED_BASE + REPEAT_SEED_STRIDE * int(rep) + i
        for i, c in enumerate(cases)
    }


def _all_rows(done: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Rows of a repeat."""

    return list((done or {}).get("rows") or [])


def _rows_by_case(done: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """case -> row (the first row of a case)."""

    out: dict[str, dict[str, Any]] = {}
    for row in _all_rows(done):
        out.setdefault(str(row.get("case_id")), row)
    return out


def failed_attempts(row: Mapping[str, Any]) -> int:
    """How many executions of this row's case ended in infra."""

    return len(row.get("infra_history") or []) + (1 if row.get("infra") else 0)


def retryable(row: Mapping[str, Any], max_infra_reruns: int) -> bool:
    """An infra row that is not ``infra_final`` and has re-runs left."""

    return bool(row.get("infra")) and not row.get("infra_final") and (
        failed_attempts(row) < 1 + int(max_infra_reruns)
    )


def plan_repeat(
    done: Mapping[str, Any] | None,
    cases: Sequence[str],
    *,
    max_infra_reruns: int = DEFAULT_MAX_INFRA_RERUNS,
) -> list[str]:
    """Cases of a repeat that still need an execution (missing, or an infra
    row with re-runs left), in ``cases`` order.  Empty = nothing to do."""

    have = _rows_by_case(done)
    return [
        c
        for c in cases
        if c not in have or retryable(have[c], max_infra_reruns)
    ]


def merge_repeat(
    done: Mapping[str, Any] | None,
    layout: Sequence[str],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    wall_s: float,
    at: str | None = None,
    max_infra_reruns: int = DEFAULT_MAX_INFRA_RERUNS,
) -> dict[str, Any]:
    """Merge freshly executed rows into a (possibly absent) repeat.

    Union semantics: every stored row is kept; a freshly executed row
    replaces only the stored row of its case (which must be infra:
    :func:`plan_repeat` never schedules a scored row again), moving the
    stored infra message to ``infra_history``.  An infra row that used up
    ``1 + max_infra_reruns`` executions is marked ``infra_final``.  Rows are
    laid out in ``layout`` order, stored rows of cases outside ``layout``
    follow."""

    have = _rows_by_case(done)
    fresh: dict[str, dict[str, Any]] = {}
    for r in new_rows:
        case_id = str(r["case_id"])
        old = have.get(case_id)
        if old is not None and not old.get("infra"):
            raise ValueError(f"refusing to replace the scored row of {case_id}")
        row = dict(r)
        history = list((old or {}).get("infra_history") or [])
        if old is not None and old.get("infra"):
            history.append(old["infra"])
        if history:
            row["infra_history"] = history
        if row.get("infra") and failed_attempts(row) >= 1 + int(max_infra_reruns):
            row["infra_final"] = True
        fresh[case_id] = row
    order = list(dict.fromkeys([*map(str, layout), *have, *fresh]))
    rows = [
        fresh[c] if c in fresh else dict(have[c])
        for c in order
        if c in fresh or c in have
    ]
    out: dict[str, Any] = {
        "rows": rows,
        "aggregate": _aggregate(rows),
        "wall_s": round(float((done or {}).get("wall_s") or 0.0) + float(wall_s), 1),
    }
    reruns = list((done or {}).get("reruns") or [])
    if done is not None and new_rows:
        reruns.append(
            {
                "at": at or _utc_now(),
                "cases": [str(r["case_id"]) for r in new_rows],
                "missing": [str(r["case_id"]) for r in new_rows if str(r["case_id"]) not in have],
                "infra_retry": [str(r["case_id"]) for r in new_rows if str(r["case_id"]) in have],
                "still_infra": [str(r["case_id"]) for r in new_rows if r.get("infra")],
                "wall_s": round(float(wall_s), 1),
            }
        )
    if reruns:
        out["reruns"] = reruns
    return out


RunCases = Callable[
    [int, list[str], list[int], int, Callable[[dict[str, Any]], None]], None
]


def run_arm(
    entry: dict[str, Any],
    name: str,
    cases: Sequence[str],
    repeats: int,
    run_cases: RunCases,
    *,
    layout: Sequence[str] | None = None,
    max_infra_reruns: int = DEFAULT_MAX_INFRA_RERUNS,
    save: Callable[[], None] = lambda: None,
    log: Callable[[str], None] = lambda text: print(text, flush=True),
) -> None:
    """Fill ``entry["repeats"]`` for ``repeats`` repeats, executing only the
    rows each repeat still needs.

    ``run_cases(rep, todo, seeds, attempt, emit)`` must call ``emit(row)``
    once per todo case (``row["case_id"]`` set; any order, any thread);
    ``attempt`` 0 = first pass of the repeat, k = k-th later pass.  Each
    emitted row is merged into the repeat and ``save()``d immediately."""

    layout = list(dict.fromkeys([*(layout or []), *cases]))
    entry.setdefault("repeats", {})
    for rep in range(int(repeats)):
        key = str(rep)
        done = entry["repeats"].get(key)
        todo = plan_repeat(done, cases, max_infra_reruns=max_infra_reruns)
        if not todo:
            final = [
                c for c, r in _rows_by_case(done).items() if r.get("infra") and c in set(cases)
            ]
            suffix = f" ({len(final)} infra_final: {final})" if final else ""
            log(f"[eval] {name} repeat {rep}: cached{suffix}")
            continue
        old = _rows_by_case(done)
        default = repeat_seeds(layout, rep)
        seeds = [int(old.get(c, {}).get("seed") or default[c]) for c in todo]
        attempt = 0 if done is None else len((done or {}).get("reruns") or []) + 1
        if done is not None:
            log(f"[eval] {name} r{rep}: running {len(todo)} case(s) {todo}")
        at = _utc_now()
        t0 = time.monotonic()
        emitted: list[dict[str, Any]] = []
        lock = threading.Lock()
        pending = set(todo)

        def emit(row: Mapping[str, Any], *, _key: str = key, _done: Any = done) -> None:
            case_id = str(row.get("case_id"))
            with lock:
                if case_id not in pending:
                    raise RuntimeError(f"{name}: unexpected or duplicate row {case_id!r}")
                pending.discard(case_id)
                emitted.append(dict(row))
                entry["repeats"][_key] = merge_repeat(
                    _done,
                    layout,
                    emitted,
                    wall_s=time.monotonic() - t0,
                    at=at,
                    max_infra_reruns=max_infra_reruns,
                )
                entry["summary"] = _summary(
                    [r["aggregate"] for r in entry["repeats"].values()],
                    [row for r in entry["repeats"].values() for row in _all_rows(r)],
                )
                save()

        run_cases(rep, list(todo), seeds, attempt, emit)
        if pending:
            raise RuntimeError(f"{name} r{rep}: no row for {sorted(pending)}")
        agg = entry["repeats"][key]["aggregate"]
        extra = "".join(
            f" {k}={v}"
            for k, v in _task_summary_fields(_all_rows(entry["repeats"][key])).items()
        )
        log(
            f"[eval] {name} r{rep}: SR={agg['success_rate']} S={agg['mean_S']} "
            f"C={agg['mean_C']} infra={agg['infra']} (final {agg['infra_final']}){extra}"
        )


def _check_meta(name: str, entry: dict[str, Any], meta: Mapping[str, Any]) -> None:
    """Refuse to merge rows of a different program / round cap into an arm;
    keys the arm does not record yet are filled in."""

    old = entry.setdefault("meta", {})
    for key in ("source_sha256", "max_rounds"):
        if key in old and key in meta and old[key] != meta[key]:
            raise SystemExit(
                f"arm {name!r}: stored {key}={old[key]!r} differs from this "
                f"invocation's {meta[key]!r}; use another arm name or --out"
            )
    for key, value in meta.items():
        old.setdefault(key, value)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", type=int, required=True, help="team size n")
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "test"),
        help="side of the task's fixed template split the cases come from: train = "
        "the development templates, test = the held-out TEST templates "
        "(without --cases / --manifest: all of them)",
    )
    parser.add_argument(
        "--cases",
        default="",
        help="comma-separated case labels (default: every template of --split; "
        "with --manifest: a filter of its labels)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="JSON case manifest ([{case_id, path, template?, pool?}] or "
        "{cases: [...]}); --cases then filters it",
    )
    parser.add_argument(
        "--benchmarks-dir",
        default=None,
        help="Silo-Bench benchmarks directory (default: $SILO_BENCH_DIR, else "
        "third_party/acl26-silo-bench/benchmarks)",
    )
    parser.add_argument(
        "--program",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="arm NAME runs the frozen program at PATH (repeatable)",
    )
    parser.add_argument(
        "--seed-program",
        action="append",
        default=[],
        metavar="NAME",
        help="arm NAME runs the evolution loop's seed program (repeatable)",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="execution repeats per case (default 3)"
    )
    parser.add_argument(
        "--max-infra-reruns",
        type=int,
        default=DEFAULT_MAX_INFRA_RERUNS,
        help="re-runs of an infra row across relaunches before it is marked "
        "infra_final (default 3)",
    )
    parser.add_argument(
        "--llm",
        choices=("openai", "fake"),
        default="openai",
        help="worker client: openai = an OpenAI-compatible endpoint "
        "(OPENAI_BASE_URL / OPENAI_API_KEY); fake = offline worker that never "
        "solves a task (no network, no key)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"worker model id (default: ${WORKER_MODEL_ENV}; required unless --llm fake)",
    )
    parser.add_argument(
        "--goal",
        choices=("all_agents", "sink"),
        default="all_agents",
        help="all_agents: every agent must answer; sink: agent 0 answers "
        "(default all_agents)",
    )
    parser.add_argument(
        "--python-max-rounds",
        type=int,
        default=64,
        help="round cap of a team program (default 64)",
    )
    parser.add_argument(
        "--parallel-cases", type=int, default=6, help="cases executed at once (default 6)"
    )
    parser.add_argument(
        "--max-parallel-agents",
        type=int,
        default=None,
        help="concurrent worker calls per case (default: --agents)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=5400.0,
        help="wall-clock limit of one worker request in seconds; also the "
        "workers' HTTP timeout unless OPENAI_TIMEOUT is set (default 5400)",
    )
    parser.add_argument("--out", required=True, help="result JSON (resumable)")
    add_task_argument(parser)
    return parser


def _arm_specs(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> list[tuple[str, str | None]]:
    """``(name, program path or None for the seed)`` per arm, validated."""

    specs: list[tuple[str, str | None]] = []
    for spec in args.program:
        name, sep, path = str(spec).partition("=")
        if not (sep and name and path):
            parser.error(f"--program expects NAME=PATH, got {spec!r}")
        if not Path(path).is_file():
            parser.error(f"--program {name}: no file {path}")
        specs.append((name, path))
    specs.extend((str(name), None) for name in args.seed_program)
    if not specs:
        parser.error("give at least one arm: --program NAME=PATH or --seed-program NAME")
    names = [name for name, _ in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        parser.error(f"duplicate arm names {dupes}")
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, task = parse_task_args(parser, argv,
                                 extra=lambda t: {"goal": t.goal} if t.goal else {})
    if int(args.agents) < 1:
        parser.error("--agents must be at least 1")
    if task.goal is not None and args.goal != task.goal:
        parser.error(f"task {task.name!r} is scored under --goal {task.goal}")
    model = str(args.model or os.environ.get(WORKER_MODEL_ENV) or "").strip() or None
    if args.llm == "fake":
        model = model or "fake"
    elif not model:
        parser.error(
            f"a real run needs a worker model: pass --model or set {WORKER_MODEL_ENV} "
            "(--llm fake runs offline)"
        )
    elif not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "a real run needs OPENAI_API_KEY (and OPENAI_BASE_URL for an endpoint "
            "other than api.openai.com); --llm fake runs offline"
        )
    arm_specs = _arm_specs(parser, args)
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT") or None
    max_parallel_agents = int(args.max_parallel_agents or args.agents)

    from queenbee.bench.silo_bench import SiloBenchAdapter
    from queenbee.evo.common import task_finalize_row, task_score_hooks
    from queenbee.program.budgets import PythonRunBudgets
    from queenbee.program.clients import http_timeout_defaults
    from queenbee.program.execute import (
        evaluate_python_source_on_cases,
        seed_python_source,
    )

    bench = (
        Path(args.benchmarks_dir) if args.benchmarks_dir else default_benchmarks_dir()
    ).resolve()

    def bench_instances() -> dict[str, Any]:
        adapter = SiloBenchAdapter(bench)
        return {
            inst.case_id: inst
            for inst in adapter.iter_instances(agent_counts=[args.agents])
        }

    def named_instances(labels: Sequence[str]) -> dict[str, Any]:
        """Instances of case labels: the task's ``instance_path``, else the
        benchmarks dir."""

        if task.instance_path is None:
            return bench_instances()
        out: dict[str, Any] = {}
        for label in labels:
            path = task.instance_path(label)
            if path is not None:
                out[label] = load_instance_file(path)
        return out

    explicit = list(dict.fromkeys(c.strip() for c in args.cases.split(",") if c.strip()))
    case_meta: dict[str, dict[str, Any]] = {}
    manifest_info: dict[str, Any] | None = None
    if args.manifest:
        entries = load_manifest(args.manifest)
        if explicit:
            entries = filter_manifest(entries, explicit)
        unresolved = [str(e["case_id"]) for e in entries if not e.get("path")]
        from_bench = named_instances(unresolved) if unresolved else {}
        instances, case_meta = resolve_manifest_instances(entries, from_bench, args.agents)
        cases = [e["case_id"] for e in entries]
        manifest_info = {
            "path": str(Path(args.manifest).resolve()),
            "sha256": _sha256_bytes(Path(args.manifest).read_bytes()),
            "n_cases": len(cases),
        }
    else:
        cases = explicit or list(split_templates(args.split))
        instances = named_instances(cases)
    missing = [c for c in cases if c not in instances]
    if missing:
        raise ValueError(f"unknown cases for n={args.agents} in {bench}: {missing}")
    if not cases:
        raise SystemExit("empty case list")
    other_sizes = sorted({int(instances[c].n_agents) for c in cases} - {int(args.agents)})
    if other_sizes:
        parser.error(f"--agents {args.agents}: the cases are instances of "
                     f"{', '.join(map(str, other_sizes))} agents")
    check_split(args.split, cases, instances, case_meta)
    for c in cases:
        meta = case_meta.setdefault(c, {"template": instances[c].case_id})
        meta["instance_sha256"] = instance_sha256(instances[c])

    budgets = PythonRunBudgets.for_rounds(args.python_max_rounds, n_agents=args.agents)

    def _arm_meta(source: str, extra: dict[str, Any]) -> dict[str, Any]:
        return extra | {
            "source_sha256": source_sha256(source),
            "max_rounds": int(budgets.max_rounds),
        }

    arms: list[tuple[str, str, dict[str, Any]]] = []
    for name, path in arm_specs:
        if path is not None:
            source = Path(path).read_text(encoding="utf-8")
            arms.append((name, source, _arm_meta(source, {"file": path})))
        # --seed-program: the task's own seed program, else the
        # phase-structured seed program (v1)
        elif task.seed_source is not None:
            source = task.seed_source()
            arms.append((name, source, _arm_meta(source, {"seed": task.name})))
        else:
            _origin, _title, source = seed_python_source(WORKER_CONTRACT)
            arms.append((name, source, _arm_meta(source, {"seed": "sfs_phase"})))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = out_path.parent / (out_path.stem + "_artifacts")
    result: dict[str, Any] = (
        json.loads(out_path.read_text()) if out_path.exists() else {}
    )
    is_new = not result
    result.setdefault("arms", {})
    config = result.setdefault("config", {})
    if config.get("split") not in (None, args.split):
        raise SystemExit(
            f"{out_path} holds --split {config['split']} cases; use another --out"
        )
    fp = run_fingerprint(
        agents=args.agents,
        model=model,
        reasoning_effort=reasoning_effort,
        goal=args.goal,
        worker_contract=WORKER_CONTRACT,
        budgets=budgets,
        llm_provider=args.llm,
        task=task.name,
    )
    check_run_fingerprint(
        config,
        fp,
        has_rows=any(
            (rep or {}).get("rows")
            for arm in result["arms"].values()
            for rep in (arm.get("repeats") or {}).values()
        ),
    )
    merged_case_meta = check_case_identity(result, case_meta)
    for name, _source, meta in arms:
        if name in result["arms"]:
            probe = json.loads(json.dumps(result["arms"][name]))
            _check_meta(name, probe, meta)  # refuse before anything runs

    now = _utc_now()
    prov = git_provenance(REPO_ROOT)
    code = code_provenance(repo=REPO_ROOT)
    if is_new:
        config.update({"created_at": now, "git_sha": prov["git_sha"], "git_dirty": prov["git_dirty"]})
    layout = list(dict.fromkeys([*(config.get("cases") or []), *cases]))
    config.update(
        {
            "agents": args.agents,
            "split": args.split,
            "cases": layout,
            "repeats": max(int(config.get("repeats") or 0), int(args.repeats)),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "goal": args.goal,
            "worker_contract": WORKER_CONTRACT,
            "python_max_rounds": int(budgets.max_rounds),
            "benchmarks_dir": str(bench),
            "updated_at": now,
        }
    )
    if manifest_info is not None:
        config["manifest"] = manifest_info
        known = [m.get("sha256") for m in config.setdefault("manifests", [])]
        if manifest_info["sha256"] not in known:
            config["manifests"].append(manifest_info | {"at": now})
    config["case_meta"] = merged_case_meta
    config.setdefault("invocations", []).append(
        {
            "at": now,
            **prov,
            "code_sha256": code["code_sha256"],
            "code_files": code["code_files"],
            "argv": list(sys.argv[1:] if argv is None else argv),
            "cases": list(cases),
            "max_infra_reruns": int(args.max_infra_reruns),
        }
    )

    save_lock = threading.Lock()

    def save() -> None:
        with save_lock:
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=1, default=str))
            tmp.replace(out_path)

    def row_extras(case_id: str, max_rounds: int) -> dict[str, Any]:
        extra: dict[str, Any] = {"max_rounds": max_rounds}
        meta = case_meta.get(case_id) or {}
        for key in ("pool", "template", "instance_case_id", "instance_sha256"):
            if key in meta:
                extra[key] = meta[key]
        return extra

    save()  # config/provenance of this invocation, before any execution
    # Defaults for this run, unless already set: OPENAI_TIMEOUT (the
    # workers' HTTP timeout) = --request-timeout, OPENAI_CONNECT_TIMEOUT = 30 s
    with http_timeout_defaults(args.request_timeout):
        for name, source, meta in arms:
            entry = result["arms"].setdefault(name, {"kind": "python", "meta": {}, "repeats": {}})
            _check_meta(name, entry, meta)

            def run_cases(
                rep: int,
                todo: list[str],
                seeds: list[int],
                attempt: int,
                emit: Callable[[dict[str, Any]], None],
                *,
                _name: str = name,
                _source: str = source,
            ) -> None:
                sub = f"r{rep}" if attempt == 0 else f"r{rep}_rerun{attempt}"

                def one(case_id: str, seed: int) -> None:
                    rows = evaluate_python_source_on_cases(
                        source=_source,
                        instances=(instances[case_id],),
                        seeds=(seed,),
                        arm_label=f"{_name}:r{rep}",
                        llm_provider=args.llm,
                        worker_model=model,
                        goal=args.goal,
                        worker_contract=WORKER_CONTRACT,
                        max_parallel_cases=1,
                        max_parallel_agents=max_parallel_agents,
                        request_timeout=args.request_timeout,
                        artifacts_dir=artifacts / _name.replace(":", "_") / sub,
                        budgets=budgets,
                        progress=lambda text: print(f"[eval] {text}", flush=True),
                        **task_score_hooks(),
                    )
                    emit(
                        task_finalize_row(rows[0])
                        | {"case_id": case_id, "seed": seed}
                        | row_extras(case_id, int(budgets.max_rounds))
                    )

                workers = max(1, min(int(args.parallel_cases), len(todo)))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(one, c, s) for c, s in zip(todo, seeds)]
                    for future in futures:
                        future.result()

            run_arm(
                entry,
                name,
                cases,
                args.repeats,
                run_cases,
                layout=layout,
                max_infra_reruns=args.max_infra_reruns,
                save=save,
            )
    save()
    print(json.dumps({k: v.get("summary") for k, v in result["arms"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
