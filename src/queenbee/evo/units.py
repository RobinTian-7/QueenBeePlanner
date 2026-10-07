"""Units of QueenBee-Evo: unit records, their classes from census rows,
the T / V split of the development templates, and the TEST guard.

A *unit* is ``(template, rung)`` with id ``"II-11@x5"`` (Silo-Bench rungs:
``o5`` = shipped upstream n=5 instance, ``x5`` = n=5 with twice the
per-agent data from :mod:`queenbee.evo.ladder`, ``o10`` = shipped n=10
instance, worth 2 execution-equivalents; the Count-Frequency task has the
single rung ``o8``).

Rows come from result JSONs of :mod:`queenbee.evaluate`: the census of the
seed program (arm ``v1``) and, optionally, other arms evaluated on the same
units (*reference arms*, used only by the classification below).  A row's
label maps to a unit as follows: ``"II-11@x5"`` is already a unit id; a bare
template label ``"II-11"`` is the upstream instance at the file's agent count
(``o<n>``, e.g. ``o5`` / ``o10``).  Infra rows are dropped (they are not
measurements).

Unit classes (``S̄_v1`` = mean S of the seed program (v1) over the unit's
scored reps):

=============  ==============================================================
frontier       0 < S̄_v1 < 1; or S̄_v1 = 0 and (some reference arm scored > 0
               on the unit, or v1 solved the same template at o5 -- at least
               one o5 rep with S = 1 -- or the unit's failing traces are
               mostly organizational)
guard          S̄_v1 = 1 on >= ``min_guard_reps`` (2) reps
dead           S̄_v1 = 0, at least one reference ran ON THIS UNIT and every
               reference scored 0, v1 did not solve the template at o5, and
               the unit's failing traces are not mostly organizational
irreducible    S̄_v1 < 1 and, on some rung of the template, at least half of
               the failing traces are precision / format near-misses
               (the team computes the right values; the benchmark's ground truth
               is rounded to a per-task number of decimals the task text never
               states, e.g. 2 / 4 / 6 -- no task-agnostic program can choose it).
               Never targeted, never a guard; reported as a shared floor.
unresolved     not enough evidence yet: no v1 rows; S̄_v1 = 1 on fewer than
               ``min_guard_reps`` reps (a single rep never makes a guard); or
               S̄_v1 = 0 with no reference rows on the unit and no o5 or
               trace evidence ("every reference scores 0" needs a reference).
=============  ==============================================================

``irreducible`` takes precedence over ``frontier`` and ``dead``.

References count per unit: a reference that scored > 0 on
``<template>@o5`` says nothing about ``<template>@x5`` (twice the data), so
it does not make ``<template>@x5`` frontier; the only template-level
evidence of headroom is "v1 solves the same template at o5".  Trace
evidence is the second signal: a zero-S unit most of whose failing traces
are organizational (content loss, scattered / divergent / consensus-wrong
answers, shape mismatch -- not near-misses, not infra) is ``frontier``
although every reference arm scores 0 there: a failure of the team's
organization is exactly the headroom an evolved program targets.

Leakage guard: :func:`assert_dev` refuses every TEST template (raises
``LadderLeakError``); loading a result file that holds a TEST row raises
too -- TEST results never enter any evo structure.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from queenbee.evo.ladder import (
    DEV_TEMPLATE_IDS,
    TIERS,
    LadderLeakError,
    assert_dev_template,
    parse_rung,
    template_of,
    tier_of,
    upstream_path,
)

EPS = 1e-9
UNIT_CLASSES: tuple[str, ...] = ("frontier", "guard", "dead", "unresolved", "irreducible")
SIDES: tuple[str, ...] = ("T", "V")
DEFAULT_V1_ARM = "v1"
#: Arm-name markers (case-insensitive) of positive controls, e.g. a
#: deliberately degraded program: such arms are never "references" for the
#: dead rule.
NON_REFERENCE_MARKERS: tuple[str, ...] = ("degraded",)


# --------------------------------------------------------------------------- #
# Unit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Unit:
    """One unit: identity, instance, T / V side, class and the rows behind
    the class."""

    unit_id: str
    template: str
    rung: str
    n_agents: int
    instance_path: str | None = None
    instance_sha256: str | None = None
    side: str | None = None  # "T" | "V" | None (not assigned yet)
    klass: str = "unresolved"
    v1_S: tuple[float, ...] = ()
    ref_S: tuple[tuple[str, tuple[float, ...]], ...] = ()
    o5_solved: bool | None = None

    def __post_init__(self) -> None:
        assert_dev_template(self.template)
        if template_of(self.unit_id) != self.template:
            raise ValueError(f"unit id {self.unit_id!r} does not match {self.template!r}")
        if parse_rung(self.rung)[1] != int(self.n_agents):
            raise ValueError(f"{self.unit_id}: rung {self.rung} vs n={self.n_agents}")
        if self.klass not in UNIT_CLASSES:
            raise ValueError(f"{self.unit_id}: klass {self.klass!r}")
        if self.side is not None and self.side not in SIDES:
            raise ValueError(f"{self.unit_id}: side {self.side!r}")

    @property
    def tier(self) -> str:
        return tier_of(self.template)

    @property
    def v1_S_mean(self) -> float | None:
        return statistics.fmean(self.v1_S) if self.v1_S else None

    @property
    def exec_equivalents(self) -> float:
        """Execution-equivalents of one run (10 or more agents, e.g. o10,
        count 2)."""

        return 2.0 if int(self.n_agents) >= 10 else 1.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["v1_S"] = list(self.v1_S)
        out["ref_S"] = {arm: list(vals) for arm, vals in self.ref_S}
        out["v1_S_mean"] = self.v1_S_mean
        out["tier"] = self.tier
        return out


def make_unit_id(template: str, rung: str) -> str:
    parse_rung(rung)
    return f"{assert_dev_template(template)}@{rung}"


def parse_unit_id(uid: str) -> tuple[str, str]:
    """``'II-11@x5' -> ('II-11', 'x5')`` (dev templates only)."""

    template, sep, rung = str(uid).partition("@")
    if not sep:
        raise ValueError(f"{uid!r} is not a unit id (template@rung)")
    parse_rung(rung)
    return assert_dev_template(template), rung


def assert_dev(unit: Unit | str) -> Unit | str:
    """Refuse a TEST (or unknown) template; returns its argument."""

    if isinstance(unit, Unit):
        assert_dev_template(unit.template)
        assert_dev_template(unit.unit_id)
    else:
        assert_dev_template(str(unit))
    return unit


def row_unit_id(label: str, n_agents: int) -> str:
    """Unit id of a result-file row label, in a file run with ``n_agents``
    agents."""

    label = str(label)
    if "@" in label:
        template, rung = parse_unit_id(label)
        if parse_rung(rung)[1] != int(n_agents):
            raise ValueError(f"row {label!r} in an n={n_agents} result file")
        return f"{template}@{rung}"
    return make_unit_id(label, f"o{int(n_agents)}")


# --------------------------------------------------------------------------- #
# Loading evaluation rows
# --------------------------------------------------------------------------- #


@dataclass
class EvalRows:
    """Scored rows by arm and unit, plus per-unit instance identity."""

    arms: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    unit_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)

    def s_values(self, arm: str, uid: str) -> list[float]:
        return [float(r.get("S") or 0.0) for r in self.arms.get(arm, {}).get(uid, [])]


def _file_agents(doc: Mapping[str, Any], where: str) -> int:
    config = dict(doc.get("config") or {})
    agents = config.get("agents")
    if agents is None:
        agents = (config.get("fingerprint") or {}).get("agents")
    if agents is None:
        raise ValueError(f"{where}: result file does not record its agent count")
    return int(agents)


def load_eval_rows(sources: Iterable[str | Path | Mapping[str, Any]]) -> EvalRows:
    """Merge :mod:`queenbee.evaluate` result JSONs (paths or loaded dicts)
    into arm -> unit -> scored rows (repeat order).  Infra rows are dropped;
    a TEST row anywhere raises ``LadderLeakError``; one unit mapping to two
    different instances raises ``ValueError``."""

    out = EvalRows()
    for index, src in enumerate(sources):
        if isinstance(src, Mapping):
            doc, where = src, f"<doc {index}>"
        else:
            where = str(src)
            doc = json.loads(Path(src).read_text())
        out.inputs.append(where)
        n = _file_agents(doc, where)
        case_meta = dict((doc.get("config") or {}).get("case_meta") or {})
        for label, meta in case_meta.items():
            uid = row_unit_id(label, n)
            _merge_meta(out.unit_meta, uid, dict(meta or {}), where)
        for arm, entry in dict(doc.get("arms") or {}).items():
            table = out.arms.setdefault(str(arm), {})
            reps = dict((entry or {}).get("repeats") or {})
            for key in sorted(reps, key=lambda k: int(k) if str(k).isdigit() else 0):
                rep = reps[key] or {}
                by_case: dict[str, dict[str, Any]] = {}
                for row in rep.get("rows") or []:
                    by_case.setdefault(str(row.get("case_id")), row)
                for row in by_case.values():
                    uid = row_unit_id(str(row.get("case_id")), n)  # raises on TEST
                    if row.get("instance_sha256"):
                        _merge_meta(
                            out.unit_meta, uid,
                            {"instance_sha256": row["instance_sha256"]}, where,
                        )
                    if row.get("infra"):
                        continue
                    table.setdefault(uid, []).append(dict(row))
    return out


def _merge_meta(store: dict[str, dict[str, Any]], uid: str, meta: dict[str, Any], where: str) -> None:
    old = store.setdefault(uid, {})
    sha = meta.get("instance_sha256")
    if sha and old.get("instance_sha256") and old["instance_sha256"] != sha:
        raise ValueError(f"{where}: unit {uid} maps to a different instance than an earlier input")
    for key, value in meta.items():
        if value is not None:
            old.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def classify_unit(
    v1_S: Sequence[float],
    ref_S: Mapping[str, Sequence[float]] | None = None,
    *,
    o5_solved: bool | None = None,
    min_guard_reps: int = 2,
    near_miss: bool = False,
    structural: bool = False,
) -> str:
    """frontier | guard | dead | unresolved | irreducible (module docstring table).

    ``near_miss``: on some rung of the template at least half of the
    failing traces are precision / format near-misses (``diagnosis`` class
    ``precision-near-miss``; :func:`near_miss_units_from_traces`): the team
    computes the right values but the benchmark's unstated rounding (2, 4
    or 6 decimals, depending on the task) decides, so a unit with
    ``S̄_v1 < 1`` is ``irreducible``.  ``structural``: more than half of
    the unit's failing traces are organizational failures (content loss,
    scattered / divergent answers ...; :func:`structural_units_from_traces`):
    a zero-S unit is learnable (``frontier``) even when every reference arm
    scores 0 there."""

    vals = [float(s) for s in v1_S]
    if not vals:
        return "unresolved"
    mean = statistics.fmean(vals)
    if near_miss and mean < 1.0 - EPS:
        return "irreducible"
    if EPS < mean < 1.0 - EPS:
        return "frontier"
    if mean >= 1.0 - EPS:
        return "guard" if len(vals) >= int(min_guard_reps) else "unresolved"
    refs = {k: [float(s) for s in v] for k, v in dict(ref_S or {}).items() if v}
    if any(s > EPS for v in refs.values() for s in v) or o5_solved or structural:
        return "frontier"
    if refs:  # every reference that ran here scored 0 (and no o5 / trace evidence)
        return "dead"
    return "unresolved"  # no reference: "every reference scores 0" is unsupported


def reference_arms_of(arms: Iterable[str], v1_arm: str = DEFAULT_V1_ARM) -> list[str]:
    """Default reference arms: every arm except v1 and positive controls."""

    return sorted(
        a for a in arms
        if a != v1_arm and not any(m in a.lower() for m in NON_REFERENCE_MARKERS)
    )


def load_units(
    sources: Iterable[str | Path | Mapping[str, Any]] = (),
    *,
    v1_arm: str = DEFAULT_V1_ARM,
    reference_arms: Sequence[str] | None = None,
    manifests: Iterable[str | Path] = (),
    min_guard_reps: int = 2,
    benchmarks_dir: str | Path | None = None,
    near_miss_units: Iterable[str] = (),
    structural_units: Iterable[str] = (),
) -> dict[str, Unit]:
    """Units (id -> Unit, sorted) from census result files.

    ``manifests``: ladder manifests (``evo.ladder.build_ladder`` output) --
    their units are included even before any row exists (``unresolved``)
    and give ``instance_path`` / ``instance_sha256``.  An upstream (``o<n>``)
    unit's path defaults to the shipped file.  ``near_miss_units`` /
    ``structural_units``: unit ids with near-miss / organizational trace
    evidence (:func:`near_miss_units_from_traces`,
    :func:`structural_units_from_traces`); near-miss evidence marks every
    rung of its template."""

    from queenbee.evaluate import load_manifest

    rows = load_eval_rows(sources)
    meta: dict[str, dict[str, Any]] = {k: dict(v) for k, v in rows.unit_meta.items()}
    for manifest in manifests:
        for entry in load_manifest(manifest):
            uid = str(entry["case_id"])
            parse_unit_id(uid)  # dev guard + shape
            _merge_meta(
                meta, uid,
                {"path": entry.get("path"), "instance_sha256": entry.get("instance_sha256")},
                str(manifest),
            )
    refs = list(reference_arms) if reference_arms is not None else reference_arms_of(rows.arms, v1_arm)
    unit_ids = set(meta)
    for table in rows.arms.values():
        unit_ids.update(table)
    # "v1 solves the template at o5": at least one o5 v1 rep with S = 1.
    o5_solved: dict[str, bool] = {}
    for uid, table_rows in rows.arms.get(v1_arm, {}).items():
        template, rung = parse_unit_id(uid)
        if rung == "o5" and table_rows:
            o5_solved[template] = any(float(r.get("S") or 0.0) >= 1.0 - EPS for r in table_rows)
    # The rounding convention is a TEMPLATE property (the generator's
    # round(x, k)), so near-miss evidence on any rung marks every rung.
    near_templates = {parse_unit_id(str(u))[0] for u in near_miss_units}
    struct_set = {str(u) for u in structural_units}
    units: dict[str, Unit] = {}
    for uid in sorted(unit_ids, key=_unit_sort_key):
        template, rung = parse_unit_id(uid)
        n = parse_rung(rung)[1]
        v1 = tuple(rows.s_values(v1_arm, uid))
        ref = tuple((arm, tuple(rows.s_values(arm, uid))) for arm in refs if rows.s_values(arm, uid))
        solved = o5_solved.get(template) if rung != "o5" else None
        path = meta.get(uid, {}).get("path")
        if path is None and rung.startswith("o"):
            candidate = upstream_path(template, n, benchmarks_dir)
            path = str(candidate) if candidate.is_file() else None
        units[uid] = Unit(
            unit_id=uid,
            template=template,
            rung=rung,
            n_agents=n,
            instance_path=path,
            instance_sha256=meta.get(uid, {}).get("instance_sha256"),
            klass=classify_unit(v1, dict(ref), o5_solved=solved, min_guard_reps=min_guard_reps,
                                near_miss=template in near_templates,
                                structural=uid in struct_set),
            v1_S=v1,
            ref_S=ref,
            o5_solved=solved,
        )
    return units


def _unit_sort_key(uid: str) -> tuple:
    template, rung = parse_unit_id(uid)
    kind, n = parse_rung(rung)
    return (*_template_key(template), n * 2 + (kind == "x"), rung)


def frontier_units(
    units: Mapping[str, Unit] | Iterable[Unit],
    *,
    templates: Iterable[str] | None = None,
    rungs: Iterable[str] | None = None,
) -> list[Unit]:
    items = units.values() if isinstance(units, Mapping) else units
    keep_t = set(templates) if templates is not None else None
    keep_r = set(rungs) if rungs is not None else None
    return [
        u for u in items
        if u.klass == "frontier"
        and (keep_t is None or u.template in keep_t)
        and (keep_r is None or u.rung in keep_r)
    ]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TemplateSplit:
    """T / V template sets (V task types never appear in T)."""

    name: str
    T: tuple[str, ...]
    V: tuple[str, ...]
    swaps: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for t in (*self.T, *self.V):
            assert_dev_template(t)
        if set(self.T) & set(self.V):
            raise ValueError(f"split {self.name}: T and V overlap")

    def side_of(self, unit_or_template: Unit | str) -> str | None:
        template = unit_or_template.template if isinstance(unit_or_template, Unit) \
            else assert_dev_template(str(unit_or_template))
        if template in self.T:
            return "T"
        if template in self.V:
            return "V"
        return None

    def assert_side(self, unit_or_template: Unit | str, side: str) -> None:
        """Raise unless the unit's template is on ``side`` (TEST always raises)."""

        got = self.side_of(unit_or_template)
        if got != side:
            raise RuntimeError(
                f"LEAKAGE_GUARD: {unit_or_template!r} is on side {got!r} of split "
                f"{self.name}, not {side!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "T": list(self.T), "V": list(self.V),
                "swaps": [list(s) for s in self.swaps], "notes": list(self.notes)}


def _frontier_count(units: Iterable[Unit], templates: Iterable[str], rungs: Iterable[str] | None) -> int:
    return len(frontier_units(list(units), templates=templates, rungs=rungs))


def _template_key(template: str) -> tuple:
    """Silo-Bench ids by (tier, number); other task ids after them, by
    prefix and number."""

    tier, sep, num = template.partition("-")
    if tier in TIERS and sep and num.isdigit():
        return TIERS.index(tier), int(num)
    return len(TIERS), tier, int(num) if num.isdigit() else -1, template


def make_split(
    units: Mapping[str, Unit] | Iterable[Unit],
    seed: int,
    *,
    n_v_per_tier: int = 2,
    min_v_frontier: int = 3,
    min_t_frontier: int = 5,
    rungs: Iterable[str] | None = None,
    max_tries: int = 20000,
) -> TemplateSplit:
    """The T / V split of the Silo-Bench development templates for ``seed``:
    V = ``n_v_per_tier`` templates per tier (I / II / III; 6 in all by
    default), T = the remaining ones; V needs >= ``min_v_frontier``
    frontier units, T >= ``min_t_frontier``.  Deterministic in ``seed``
    (rejection sampling on ``random.Random('queenbee-evo-split|<seed>')``);
    raises ``ValueError`` when no draw within ``max_tries`` qualifies.  A
    task with its own fixed split supplies ``TaskSpec.default_split``
    instead."""

    pool = [assert_dev(u) for u in (units.values() if isinstance(units, Mapping) else units)]
    rung_list = list(rungs) if rungs is not None else None
    by_tier = {tier: sorted((t for t in DEV_TEMPLATE_IDS if tier_of(t) == tier), key=_template_key)
               for tier in TIERS}
    rng = random.Random(f"queenbee-evo-split|{int(seed)}")
    for attempt in range(int(max_tries)):
        V: list[str] = []
        for tier in TIERS:
            V.extend(rng.sample(by_tier[tier], n_v_per_tier))
        T = [t for t in DEV_TEMPLATE_IDS if t not in set(V)]
        if (_frontier_count(pool, V, rung_list) >= min_v_frontier
                and _frontier_count(pool, T, rung_list) >= min_t_frontier):
            return TemplateSplit(
                f"s{int(seed)}",
                tuple(sorted(T, key=_template_key)),
                tuple(sorted(V, key=_template_key)),
                notes=(f"draw {attempt + 1}",),
            )
    raise ValueError(
        f"split seed {seed}: no draw in {max_tries} meets V>={min_v_frontier} / "
        f"T>={min_t_frontier} frontier units"
    )


def units_summary(units: Mapping[str, Unit]) -> dict[str, Any]:
    """Counts per class and rung, and the unit ids of every class but
    ``unresolved`` (for logs)."""

    by: dict[str, dict[str, int]] = {}
    for u in units.values():
        by.setdefault(u.rung, {k: 0 for k in UNIT_CLASSES})[u.klass] += 1
    return {
        "per_rung": by,
        "frontier": [u.unit_id for u in units.values() if u.klass == "frontier"],
        "guard": [u.unit_id for u in units.values() if u.klass == "guard"],
        "dead": [u.unit_id for u in units.values() if u.klass == "dead"],
        "irreducible": [u.unit_id for u in units.values() if u.klass == "irreducible"],
    }


__all__ = [
    "DEFAULT_V1_ARM",
    "EvalRows",
    "LadderLeakError",
    "SIDES",
    "TemplateSplit",
    "UNIT_CLASSES",
    "Unit",
    "assert_dev",
    "classify_unit",
    "failure_profile_from_traces",
    "frontier_units",
    "load_eval_rows",
    "load_units",
    "make_split",
    "make_unit_id",
    "near_miss_units_from_traces",
    "parse_unit_id",
    "reference_arms_of",
    "structural_units_from_traces",
    "row_unit_id",
    "units_summary",
]


def near_miss_units_from_traces(trace_dirs: Iterable[str | Path]) -> dict[str, int]:
    """Unit id -> number of near-miss traces, for the units where at least
    half of the failing traces are near-misses (the ``irreducible``
    evidence of :func:`classify_unit`; answer-free: only counts leave).
    See :func:`failure_profile_from_traces`."""

    prof = failure_profile_from_traces(trace_dirs)
    return {u: p["near_miss"] for u, p in prof.items()
            if p["near_miss"] and 2 * p["near_miss"] >= p["failing"]}


def structural_units_from_traces(trace_dirs: Iterable[str | Path]) -> dict[str, int]:
    """Unit id -> number of failing traces that are organizational (not a
    near-miss, not infra), for the units where more than half of the failing
    traces are such (the ``structural`` evidence of :func:`classify_unit`)."""

    prof = failure_profile_from_traces(trace_dirs)
    return {u: p["failing"] - p["near_miss"] for u, p in prof.items()
            if 2 * (p["failing"] - p["near_miss"]) > p["failing"]}


def failure_profile_from_traces(trace_dirs: Iterable[str | Path]) -> dict[str, dict[str, int]]:
    """Unit id -> {"failing": scored, non-infra traces with S < 1 and at
    least one wrong agent, "near_miss": those in which at least half of the
    wrong agents are precision / format near-misses}.

    Trace JSONs are the ``QB_TRACE_DIR`` dumps (``case_id``, ``n_agents``,
    ``facts``, ``output``, ``ground_truth``).  o-rung ids are derived from the
    trace's agent count; traces of TEST (or unknown) templates are skipped."""

    import json as _json
    from types import SimpleNamespace

    from queenbee.evo import diagnosis as _dg

    out: dict[str, dict[str, int]] = {}
    for d in trace_dirs:
        for path in sorted(Path(d).glob("**/*.json")):
            try:
                rec = _json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            case = str(rec.get("case_id") or "")
            n = rec.get("n_agents")
            facts = rec.get("facts") or {}
            if not case or facts.get("S") is None or float(facts.get("S") or 0.0) >= 1.0 - EPS:
                continue
            if facts.get("infra"):
                continue
            uid = case if "@" in case else f"{case}@o{int(n or 5)}"
            try:
                parse_unit_id(uid)  # dev guard (raises on TEST)
            except Exception:
                continue
            subs = ((rec.get("output") or {}).get("submissions") or [])
            answers = [s.get("answer") for s in sorted(subs, key=lambda s: s.get("agent_id", 0))]
            truth = rec.get("ground_truth")
            inst = SimpleNamespace(meta={}, ground_truth=truth)
            expected = (truth.get("per_agent_values") if isinstance(truth, dict) else None)
            wrong = []
            for i, a in enumerate(answers):
                target = expected[i] if isinstance(expected, list) and i < len(expected) else truth
                if a is None or _dg._canonical(a) != _dg._canonical(target):
                    wrong.append(i)
            if not wrong:
                continue
            prof = out.setdefault(uid, {"failing": 0, "near_miss": 0})
            prof["failing"] += 1
            flags = _dg._near_miss_flags(inst, answers, wrong) or []
            if flags and 2 * sum(flags) >= len(wrong):
                prof["near_miss"] += 1
    return out
