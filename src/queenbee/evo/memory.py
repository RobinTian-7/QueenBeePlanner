"""Experience memory of QueenBee-Evo and the brief scheduler that reads it.

Two of the three experience components live here, the program archive and
the hypothesis ledger (the diagnosis cards are in
:mod:`queenbee.evo.diagnosis`), next to the scheduler that turns them into
briefs.  The three classes are JSON-serialisable and computed by the host
(no LLM curator):

* :class:`Archive` (program archive) -- every executed program's scored
  rows per unit; the **per-unit Pareto front** with a noise margin of one
  quantum, 1/n for n agents (GEPA-style: a program is on the front of unit u
  when no program beats it there by a full quantum; the *leader* of u beats
  every other program on u by >= 1/n); the **specialist parent** of a
  target unit (the best eligible program on it); **merge pairs** of front
  specialists that win disjoint units; the **shrunk T score**
  (:meth:`Archive.shrunk_score`); confirmation bookkeeping (until
  confirmed, a program can be a parent at most once); **contrastive
  exemplars** (one program that fixed the failure class, one that failed
  on it) and the best-per-unit rows the failure map is built from.
* :class:`Ledger` (hypothesis ledger) -- one entry per proposal with the
  host verdict (confirmed / refuted / inconclusive / screened,
  :func:`queenbee.evo.race.host_verdict`); **mechanism tags extracted by the
  host** from the parent -> child source (:func:`mechanism_tags`, structural;
  :func:`queenbee.program.mint.summarize_source_diff` for the logged diff
  summary) plus the planner's stated mechanism, normalised by
  :func:`queenbee.evo.prompt.normalize_mechanism`; the data of the prompt's
  ledger block (``prompt._block_ledger``: entries + tabu); the tabu list =
  (class, mechanism) pairs refuted twice INDEPENDENTLY (not against one
  shared fresh parent rep; economize refutations excluded), with a screen
  hook (:meth:`Ledger.tabu_fn`).  UCB pulls exclude proposals that were never
  measured for reasons outside the planner's control (infra, budget).
* :class:`Scheduler` -- K briefs per generation (default 4): ``n_target``
  target slots (default 3; the loop uses max(1, K - 1)) and K - n_target
  rotating slots (merge -> economize -> explore, cycled by generation:
  ``gen mod 3`` for a single rotating slot).  Target slots pick cells of
  the failure map (answer shape x failure class, built from the diagnosis
  cards) by UCB, with a penalty per refuted attempt from the ledger; the
  parent is the target unit's specialist in the archive.  A unit is closed
  (solved) only on ESTABLISHED evidence (>= 2 reps or a confirmation; the
  seed program's census rows always count), never on one lucky rep.  The
  slot schedule is the same in every arm; switching a component off only
  changes what fills a slot:

  ===============  =========================================================
  ``--no-diag``    targets = the lowest-S units (best established S̄, or the
                   incumbent's S without the archive), no class, no cells
  ``--no-ledger``  no refutation penalty, no tabu (verdicts still recorded)
  ``--no-archive`` parent = best-mean incumbent (shrunk T score), no
                   exemplars, the merge slot becomes explore; open units,
                   headroom prior and failure map read ONLY the incumbent's
                   rows (the seed program's census rows where it has none;
                   :meth:`Scheduler.view_program`), and the prompt state
                   only the parent's (:func:`prompt_state`)
  ===============  =========================================================

A brief is a dict with ``brief_id``, ``kind``, ``target_units``,
``target_class`` and ``parent_ids``, plus ``gen``, ``slot``, ``slot_kind``
(the schedule's kind before any fallback), ``cell``, ``target_shape``,
``exemplars``, ``wins``, ``fallback`` and ``flags`` (the arm's component
switches), and ``lockin`` on the slot that ``--explore-on-lockin``
reserves.  :meth:`Scheduler.briefs` is pure; :meth:`Scheduler.commit`
records parent uses and merged pairs once per brief id, so repeating it
after a resume is a no-op.
"""

from __future__ import annotations

import ast
import difflib
import math
import re
from collections import Counter
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from queenbee.evo.prompt import ARM_FLAGS, EvoFlags, coerce_flags, normalize_mechanism
from queenbee.evo.race import (
    EPS,
    Thresholds,
    assert_dev_unit,
    host_verdict,
    is_infra,
    template_of,
    unit_n_agents,
    unit_sort_key,
)

#: Kinds of the rotating slot, cycled by generation.
ROTATION: tuple[str, ...] = ("merge", "economize", "explore")
K_BRIEFS = 4
N_TARGET_SLOTS = 3
#: A cell (or, without diagnosis cards, a unit) whose headroom (1 - best S̄)
#: is below this fraction of the largest headroom in play never takes a
#: target slot: preferring distinct cells must not spend a proposal on a
#: 0.05-headroom unit while a 1.0-headroom unit is open.  Spare slots repeat
#: the eligible cells (spread by virtual loss) or cycle through the eligible
#: units instead.
MIN_HEADROOM_FRAC = 0.25
#: A (class, mechanism) pair refuted this many times becomes tabu.
#: Refutations that share one fresh parent rep (same parent, R1 unit and
#: generation) count once, economize refutations not at all.
TABU_REFUTATIONS = 2
#: A program's mean on a unit is ESTABLISHED (may close a unit, feeds the
#: failure map and the headroom prior) with this many scored reps, or once
#: confirmed there; the seed program (v1, census rows) always is.  A single
#: lucky rep of an unconfirmed program is exactly what the confirmation step
#: exists to check.
ESTABLISHED_REPS = 2
#: Ledger statuses that are not a UCB pull of their cell: the proposal was
#: never measured for reasons outside the planner's control.
NON_PULL_STATUSES = frozenset({"infra_final", "not_raced_budget", "no_fresh_parent",
                               "not_executed"})
#: A confirmation that produced no scored rep (every attempt infra) is
#: undecided; the unit is offered for confirmation again, up to this many
#: attempts in all.
MAX_CONFIRM_ATTEMPTS = 3
#: ``--explore-on-lockin``: the ledger is LOCKED IN when at least this many
#: confirmed entries exist and this share of them falls in one mechanism
#: family (:func:`mechanism_family`).
LOCKIN_MIN_CONFIRMED = 3
LOCKIN_SHARE = 0.6
#: Row keys kept in the archive (JSON-slim; diag / credit are answer-free).
ROW_KEYS: tuple[str, ...] = (
    "S", "C", "model_calls", "prompt_tokens", "completion_tokens", "success",
    "execution_class", "rep", "purpose", "batch_id", "gen", "key", "cached",
    "diag", "credit", "error",
)
_SAFE_TAG = re.compile(r"[^a-z0-9_]+")
_LEDGER_REF = re.compile(r"\bL(\d{1,4})\b")
_CLASS_CHARS = re.compile(r"[^a-z\-]")


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


def _fclass(value: Any) -> str:
    return _CLASS_CHARS.sub("", str(value or "").lower())[:24]


def margin(unit_id: Any) -> float:
    """The noise margin of a unit: one quantum, 1/n."""

    return 1.0 / float(unit_n_agents(unit_id))


# --------------------------------------------------------------------------- #
# Host-side mechanism tags (parent -> child source)
# --------------------------------------------------------------------------- #


def _tag(prefix: str, name: Any = "") -> str:
    body = _SAFE_TAG.sub("_", str(name or "").lower()).strip("_")[:20]
    return (f"{prefix}_{body}" if body else prefix)[:30]


def _phases_of(tree: ast.Module) -> list[dict[str, Any]] | None:
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == "PHASES" for t in targets):
            try:
                value = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001 - a computed PHASES is still a program
                return None
            if isinstance(value, (list, tuple)):
                return [dict(x) if isinstance(x, dict) else {} for x in value]
            return None
    return None


def _functions_of(tree: ast.Module) -> dict[str, ast.AST]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _others_of(tree: ast.Module) -> list[str]:
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
            continue  # docstrings / bare strings
        targets = node.targets if isinstance(node, ast.Assign) else []
        if any(isinstance(t, ast.Name) and t.id == "PHASES" for t in targets):
            continue
        out.append(ast.dump(node, include_attributes=False))
    return out


def _dump(node: ast.AST | None) -> str | None:
    return None if node is None else ast.dump(node, include_attributes=False)


def _mentions_wi(node: ast.AST | None) -> bool:
    if node is None:
        return False
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value == "work_instruction":
            return True
    return False


def _diff_tags(parent: str, child: str) -> list[str]:
    """Fallback tags from a unified line diff (unparseable source)."""

    tags: list[str] = []
    for line in difflib.unified_diff(parent.splitlines(), child.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")) or not line or line[0] not in "+-":
            continue
        body = line[1:]
        kind = re.search(r"""["']kind["']\s*:\s*["']([^"']+)["']""", body)
        if kind:
            tags.append(_tag("add" if line[0] == "+" else "drop", kind.group(1)))
        elif re.match(r"\s*def\s+phase_\w+", body) and line[0] == "+":
            tags.append("newkind")
        elif "work_instruction" in body or re.search(r"""["']wi["']""", body):
            tags.append("wi_edit")
    return sorted(dict.fromkeys(tags)) or ["unparsed_change"]


def mechanism_tags(parent_source: str | None, child_source: str | None) -> list[str]:
    """Structural mechanism tags of a parent -> child edit, computed by the
    host (never the planner's words).  Vocabulary (snake_case, <= 30 chars):

    ``none`` (identical sources), ``add_<kind>`` / ``drop_<kind>`` (PHASES
    kind multiset), ``rounds_<kind>`` (rounds / scale changed), ``reorder``,
    ``phases_computed`` (``PHASES`` is missing or not a literal on one side),
    ``wi_add`` / ``wi_edit`` / ``wi_drop`` (phase ``wi`` fields), ``wi_fn`` (a
    phase function newly emits ``work_instruction``), ``newkind`` (a new
    ``phase_*`` function), ``edit_<kind>`` (an existing phase function
    changed), ``dispatch`` (``phase_turn`` changed without a new kind),
    ``interp`` (the interpreter changed), ``helper`` (another function),
    ``other_code`` (other top-level code), ``cosmetic`` (text-only change).
    Source that does not parse gets tags from a line diff
    (``unparsed_change`` when none applies)."""

    parent = str(parent_source or "")
    child = str(child_source or "")
    if parent == child:
        return ["none"]
    try:
        ptree, ctree = ast.parse(parent), ast.parse(child)
    except SyntaxError:
        return _diff_tags(parent, child)
    tags: list[str] = []
    pp, cp = _phases_of(ptree), _phases_of(ctree)
    if pp is not None and cp is not None:
        pk = [str(p.get("kind", "")) for p in pp]
        ck = [str(p.get("kind", "")) for p in cp]
        pc, cc = Counter(pk), Counter(ck)
        for kind in sorted(set(pc) | set(cc)):
            if cc[kind] > pc[kind]:
                tags.append(_tag("add", kind))
            elif pc[kind] > cc[kind]:
                tags.append(_tag("drop", kind))
        if pc == cc and pk != ck:
            tags.append("reorder")
        # same-kind occurrences matched in order
        seen: Counter = Counter()
        pocc = {}
        for p in pp:
            kind = str(p.get("kind", ""))
            pocc[(kind, seen[kind])] = p
            seen[kind] += 1
        seen = Counter()
        for c in cp:
            kind = str(c.get("kind", ""))
            p = pocc.get((kind, seen[kind]))
            seen[kind] += 1
            cwi = str(c.get("wi") or "")
            if p is None:
                if cwi:
                    tags.append("wi_add")
                continue
            if (p.get("rounds"), bool(p.get("scale_rounds_to_agents"))) != \
                    (c.get("rounds"), bool(c.get("scale_rounds_to_agents"))):
                tags.append(_tag("rounds", kind))
            pwi = str(p.get("wi") or "")
            if cwi and not pwi:
                tags.append("wi_add")
            elif pwi and not cwi:
                tags.append("wi_drop")
            elif pwi != cwi:
                tags.append("wi_edit")
        for (kind, index), p in pocc.items():
            if index >= Counter(ck)[kind] and p.get("wi"):
                tags.append("wi_drop")
    elif pp != cp:
        tags.append("phases_computed")
    pf, cf = _functions_of(ptree), _functions_of(ctree)
    new_kinds = [n for n in cf if n.startswith("phase_") and n != "phase_turn" and n not in pf]
    if new_kinds:
        tags.append("newkind")
    changed = [n for n in cf if n in pf and _dump(cf[n]) != _dump(pf[n])]
    for name in changed:
        if name == "phase_turn":
            if not new_kinds:
                tags.append("dispatch")
        elif name.startswith("phase_"):
            tags.append(_tag("edit", name[len("phase_"):]))
        elif name in ("plan_communication_turn", "plan_submit_round"):
            tags.append("interp")
        elif name != "main":
            tags.append("helper")
    added_other = [n for n in cf if n not in pf and not n.startswith("phase_")]
    if added_other:
        tags.append("helper")
    if any(_mentions_wi(cf[n]) and not _mentions_wi(pf.get(n)) for n in cf
           if n.startswith("phase_") and n != "phase_turn"):
        tags.append("wi_fn")
    if _others_of(ptree) != _others_of(ctree):
        tags.append("other_code")
    tags = [t[:30] for t in dict.fromkeys(tags)]
    return tags or ["cosmetic"]


#: Mechanism FAMILIES of host tags (``--explore-on-lockin``): an edit that
#: changes who talks to whom / when (PHASES kinds, rounds, order, a phase
#: function, the dispatcher or the interpreter) is ``topology``; otherwise
#: one that rewrites work instructions (``wi_*`` tags) is ``instructions``;
#: anything else (helpers, other top-level code, cosmetic) is ``code``.
MECHANISM_FAMILIES: tuple[str, ...] = ("topology", "instructions", "code")
_TOPOLOGY_TAG_PREFIXES: tuple[str, ...] = ("add_", "drop_", "rounds_", "edit_")
_TOPOLOGY_TAGS = frozenset({"reorder", "newkind", "dispatch", "interp", "phases_computed"})
_INSTRUCTION_TAGS = frozenset({"wi_add", "wi_edit", "wi_drop", "wi_fn"})


def mechanism_family(tags: Sequence[str] | None) -> str | None:
    """``topology`` > ``instructions`` > ``code`` (the first that applies)
    of an edit's host tags (:func:`mechanism_tags`); None without a change
    (no tags / ``none``)."""

    clean = [normalize_mechanism(t).replace(" ", "_") for t in tags or []]
    clean = [t for t in clean if t and t != "none"]
    if not clean:
        return None
    if any(t in _TOPOLOGY_TAGS or t.startswith(_TOPOLOGY_TAG_PREFIXES) for t in clean):
        return "topology"
    if any(t in _INSTRUCTION_TAGS for t in clean):
        return "instructions"
    return "code"


def mech_key(tags: Sequence[str] | None, claim: Any = None) -> str:
    """The ledger's mechanism key, the same rule as ``prompt._ledger_key``:
    normalised host tags joined by ``+`` (sorted), else the normalised
    planner claim, else ``unlabeled``; clipped to 80 chars."""

    clean = [normalize_mechanism(t)[:30] for t in (tags or []) if normalize_mechanism(t)]
    if clean:
        mech = "+".join(sorted(dict.fromkeys(clean)))
    else:
        mech = normalize_mechanism(claim)[:60] or "unlabeled"
    return mech[:80]


def diff_summary(parent_source: str | None, child_source: str | None) -> str | None:
    """``summarize_source_diff(python_source_diff(parent, child))``; None
    when that fails."""

    from queenbee.program.mint import python_source_diff, summarize_source_diff

    try:
        return summarize_source_diff(
            python_source_diff(str(parent_source or ""), str(child_source or ""), max_lines=160))
    except Exception:  # noqa: BLE001 - a log field, never fatal
        return None


def edit_size(parent_source: str | None, child_source: str | None) -> dict[str, int]:
    """Added / removed line counts of the edit (logged in its ledger entry)."""

    plus = minus = 0
    for line in difflib.unified_diff(str(parent_source or "").splitlines(),
                                     str(child_source or "").splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            plus += 1
        elif line.startswith("-"):
            minus += 1
    return {"added": plus, "removed": minus}


# --------------------------------------------------------------------------- #
# Program archive
# --------------------------------------------------------------------------- #


def _rid_key(rid: str) -> tuple[int, int, str]:
    """Numeric order of archive row ids (``r2`` < ``r10``; ``x<i>`` last)."""

    match = re.fullmatch(r"([rx])(\d+)", str(rid))
    if match:
        return (0 if match.group(1) == "r" else 1, int(match.group(2)), "")
    return (2, 0, str(rid))


def _slim(row: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: row.get(k) for k in ROW_KEYS if row.get(k) is not None}
    if out.get("S") is not None:
        out["S"] = float(out["S"])
    return out


class Archive:
    """Per-unit Pareto archive of executed programs (see the module docstring).

    ``t_templates`` (the T side of the split) is REQUIRED: the archive fails
    closed -- without it nothing would tell a V row from a T row."""

    def __init__(self, *, v1_id: str = "v1", t_templates: Iterable[str] | None = None) -> None:
        self.v1_id = str(v1_id)
        templates = sorted({str(t) for t in (t_templates or ())})
        if not templates:
            raise ValueError("Archive needs the T templates (t_templates): it never "
                             "guesses which rows are T (fail closed)")
        self.t_templates: list[str] = templates
        self._t_set = frozenset(templates)
        self.programs: dict[str, dict[str, Any]] = {}
        self.rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        self.confirm_requested: dict[str, list[str]] = {}
        self.confirmed: dict[str, list[str]] = {}
        self.confirm_failed: dict[str, list[str]] = {}
        #: pid -> units whose confirmation produced no scored rep (one entry
        #: per undecided attempt; retried while < MAX_CONFIRM_ATTEMPTS)
        self.confirm_undecided: dict[str, list[str]] = {}
        self.parent_uses: dict[str, int] = {}
        self._stats_cache: dict[str, dict[str, dict[str, Any]]] = {}

    # ------------------------------------------------------------------ #
    def is_t(self, unit_id: str) -> bool:
        return template_of(unit_id) in self._t_set

    def add_program(self, program_id: str, **meta: Any) -> dict[str, Any]:
        pid = str(program_id)
        rec = self.programs.setdefault(pid, {
            "program_id": pid, "arm": None, "gen": None, "parent_ids": [], "brief_id": None,
            "brief_kind": None, "fp": None, "source_sha256": None, "hypothesis": None,
            "target_units": [], "target_class": None, "reached_r2": False,
            "high_tier": False, "D": None, "target_dS": None, "verdict": None,
        })
        for key, value in meta.items():
            if value is None:
                continue
            if key in ("parent_ids", "target_units"):
                value = [str(v) for v in value]
            rec[key] = value
        return rec

    def add_rows(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Add scored execution rows (``program_id``, ``unit_id``, ``S``);
        infra rows and rows without S are skipped; (program, unit, rep) is
        the identity.  Returns the number of rows added."""

        added = 0
        for row in rows:
            if not isinstance(row, Mapping) or is_infra(row) or row.get("infra_final"):
                continue
            if row.get("S") is None:
                continue
            pid, unit = row.get("program_id"), row.get("unit_id")
            if not pid or not unit:
                continue
            assert_dev_unit(unit, where="archive rows")
            pid, unit = str(pid), str(unit)
            if pid not in self.programs:
                self.add_program(pid)
            table = self.rows.setdefault(pid, {}).setdefault(unit, {})
            rep = row.get("rep")
            rid = f"r{int(rep)}" if isinstance(rep, int) and not isinstance(rep, bool) \
                else f"x{len(table)}"
            if rid in table:
                continue
            table[rid] = _slim(row)
            added += 1
            self._stats_cache.pop(pid, None)
        return added

    def update_outcome(self, outcome: Mapping[str, Any]) -> None:
        """Fold a ``Racer.outcome`` record into the program's entry."""

        pid = str(outcome.get("program_id"))
        rec = self.add_program(pid)
        rec["reached_r2"] = bool(rec.get("reached_r2") or outcome.get("reached_r2"))
        rec["high_tier"] = bool(rec.get("high_tier") or outcome.get("high_tier"))
        for key in ("D", "target_dS", "verdict"):
            if outcome.get(key) is not None:
                rec[key] = outcome[key]

    # ------------------------------------------------------------------ #
    def unit_rows(self, program_id: str, unit_id: str) -> list[dict[str, Any]]:
        table = (self.rows.get(str(program_id)) or {}).get(str(unit_id)) or {}
        return [table[k] for k in sorted(table, key=_rid_key)]

    def unit_stats(self, program_id: str, *, t_only: bool = True) -> dict[str, dict[str, Any]]:
        pid = str(program_id)
        if pid not in self._stats_cache:
            out: dict[str, dict[str, Any]] = {}
            for unit, table in (self.rows.get(pid) or {}).items():
                s = [float(r["S"]) for r in table.values() if r.get("S") is not None]
                if not s:
                    continue
                c = [float(r["C"]) for r in table.values() if r.get("C") is not None]
                out[unit] = {"S": sum(s) / len(s), "n": len(s), "C": _mean(c)}
            self._stats_cache[pid] = out
        stats = self._stats_cache[pid]
        return {u: v for u, v in stats.items() if not t_only or self.is_t(u)}

    def units(self, *, t_only: bool = True) -> list[str]:
        found = {u for pid in self.rows for u in self.unit_stats(pid, t_only=t_only)}
        return sorted(found, key=unit_sort_key)

    def is_confirmed(self, program_id: str) -> bool:
        return str(program_id) == self.v1_id or bool(self.confirmed.get(str(program_id)))

    def is_established(self, program_id: str, unit_id: str) -> bool:
        """The seed program (v1) always; another program with >=
        ``ESTABLISHED_REPS`` scored reps on the unit or confirmed there."""

        pid = str(program_id)
        if pid == self.v1_id:
            return True
        st = self.unit_stats(pid, t_only=False).get(str(unit_id))
        if not st:
            return False
        return int(st["n"]) >= ESTABLISHED_REPS or str(unit_id) in self.confirmed.get(pid, [])

    def robust_best(self, unit_id: str) -> float | None:
        """Best S̄ on a unit over ESTABLISHED programs only (a single lucky rep
        of an unconfirmed program neither closes a unit nor removes its
        headroom)."""

        means = self.unit_means(unit_id)
        vals = [s for p, s in means.items() if self.is_established(p, unit_id)]
        return max(vals) if vals else None

    def view_rows(self, program_id: str, unit_id: str) -> tuple[str | None, list[dict[str, Any]]]:
        """The rows one program's view has of a unit: its own, else the seed
        program's census rows (shared by every arm); ``(source program,
        rows)``, ``(None, [])`` when neither has any."""

        rows = self.unit_rows(program_id, unit_id)
        if rows:
            return str(program_id), rows
        rows = self.unit_rows(self.v1_id, unit_id)
        return (self.v1_id, rows) if rows else (None, [])

    def eligible_parent(self, program_id: str, extra_uses: Mapping[str, int] | None = None) -> bool:
        """The seed program (v1) and confirmed programs always; an unconfirmed
        program at most once (``extra_uses``: uses not yet committed, e.g.
        earlier briefs of the generation being planned)."""

        pid = str(program_id)
        if self.is_confirmed(pid):
            return True
        uses = int(self.parent_uses.get(pid, 0)) + int((extra_uses or {}).get(pid, 0))
        return uses < 1

    def note_parent_use(self, program_id: str) -> None:
        pid = str(program_id)
        self.parent_uses[pid] = int(self.parent_uses.get(pid, 0)) + 1

    # ------------------------------------------------------------------ #
    # Pareto front
    # ------------------------------------------------------------------ #
    def unit_means(self, unit_id: str) -> dict[str, float]:
        out = {}
        for pid in self.rows:
            st = self.unit_stats(pid, t_only=False).get(unit_id)
            if st is not None:
                out[pid] = st["S"]
        return out

    def pareto(self, units: Iterable[str] | None = None) -> dict[str, Any]:
        """Per T unit: best S̄, the front (programs within one quantum of the
        best), the leader (beats every other measured program by >= 1/n;
        None when nobody does or only one program is measured)."""

        wanted = [u for u in (units if units is not None else self.units())
                  if self.is_t(u)]
        per_unit: dict[str, dict[str, Any]] = {}
        members: set[str] = set()
        for unit in sorted(dict.fromkeys(wanted), key=unit_sort_key):
            means = self.unit_means(unit)
            if not means:
                continue
            m = margin(unit)
            best = max(means.values())
            front = sorted((p for p, s in means.items() if s >= best - m + EPS),
                           key=lambda p: (-means[p], p))
            leader = None
            if len(means) > 1:
                top = max(means, key=lambda p: (means[p], p == self.v1_id))
                if all(means[top] >= s + m - EPS for p, s in means.items() if p != top):
                    leader = top
            per_unit[unit] = {"best": round(best, 6), "front": front, "leader": leader,
                              "margin": round(m, 6), "n_programs": len(means)}
            members.update(front)
        return {"units": per_unit, "programs": sorted(members)}

    def wins(self, program_id: str, pareto: Mapping[str, Any] | None = None) -> list[str]:
        pareto = pareto or self.pareto()
        return [u for u, e in pareto["units"].items() if e["leader"] == str(program_id)]

    def newly_front_winning(self, *, mark: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
        """New front winners to confirm: programs other than the seed program
        that lead a unit they have not been confirmed on and have no
        confirmation request on record (an undecided request is withdrawn,
        see :meth:`resolve_confirmations`); one unit per program (the widest
        lead), widest leads first.  ``mark`` records the returned requests;
        ``limit`` caps the list."""

        pareto = self.pareto()
        best_unit: dict[str, tuple[float, str]] = {}
        for unit, e in pareto["units"].items():
            pid = e["leader"]
            if not pid or pid == self.v1_id:
                continue
            if self.confirm_requested.get(pid) or unit in self.confirmed.get(pid, []):
                continue
            means = self.unit_means(unit)
            runner_up = max((s for p, s in means.items() if p != pid), default=0.0)
            lead = round(means[pid] - runner_up, 6)
            if pid not in best_unit or (lead, unit) > best_unit[pid]:
                best_unit[pid] = (lead, unit)
        out = [{"program_id": pid, "unit_id": unit, "lead": round(lead, 6)}
               for pid, (lead, unit) in sorted(best_unit.items(),
                                               key=lambda kv: (-kv[1][0], kv[0]))]
        if limit is not None:
            out = out[: max(0, int(limit))]
        if mark:
            for item in out:
                self.confirm_requested.setdefault(item["program_id"], []).append(item["unit_id"])
        return out

    def resolve_confirmations(self, winners: Iterable[Any]) -> dict[str, bool | None]:
        """Decide confirmations once their rows are in the archive.  A winner
        is confirmed on its unit when it is still on the front there (merged
        mean) AND the confirmation rep itself (the latest scored
        ``purpose == "confirm"`` row) beats its parent's (else the seed
        program's) mean on the unit by one quantum: an independent
        replication of the lead.  With neither mean on the unit, the front
        condition alone decides.

        No scored confirmation rep (every attempt was infra) -> ``None``,
        undecided: neither confirmed nor failed, and NEVER decided on the
        merged mean (that would reuse the lucky rep being checked).  The
        request is withdrawn so :meth:`newly_front_winning` offers it again,
        at most ``MAX_CONFIRM_ATTEMPTS`` attempts in all.  Returns
        ``{"<program>|<unit>": True | False | None}``."""

        pareto = self.pareto()
        out: dict[str, bool | None] = {}
        for w in winners:
            pid = str(_get(w, "program_id") if not isinstance(w, (tuple, list)) else w[0])
            unit = str(_get(w, "unit_id") if not isinstance(w, (tuple, list)) else w[1])
            self.confirm_requested.setdefault(pid, [])
            if unit not in self.confirm_requested[pid]:
                self.confirm_requested[pid].append(unit)
            if unit in self.confirmed.get(pid, []) or unit in self.confirm_failed.get(pid, []):
                out[f"{pid}|{unit}"] = unit in self.confirmed.get(pid, [])  # already decided
                continue
            conf = [r for r in self.unit_rows(pid, unit)
                    if r.get("purpose") == "confirm" and r.get("S") is not None]
            if not conf:
                tries = self.confirm_undecided.setdefault(pid, [])
                tries.append(unit)
                if tries.count(unit) < MAX_CONFIRM_ATTEMPTS:
                    self.confirm_requested[pid] = [u for u in self.confirm_requested[pid]
                                                   if u != unit]
                out[f"{pid}|{unit}"] = None
                continue
            entry = pareto["units"].get(unit) or {}
            means = self.unit_means(unit)
            parents = (self.programs.get(pid) or {}).get("parent_ids") or []
            ref = next((p for p in parents if p in means), self.v1_id if self.v1_id in means else None)
            s_rep = float(conf[-1]["S"])
            ok = pid in (entry.get("front") or []) and (
                ref is None or s_rep >= means[ref] + margin(unit) - EPS)
            bucket = self.confirmed if ok else self.confirm_failed
            if unit not in bucket.setdefault(pid, []):
                bucket[pid].append(unit)
            out[f"{pid}|{unit}"] = ok
        return out

    # ------------------------------------------------------------------ #
    # Scores, parents, merges
    # ------------------------------------------------------------------ #
    def n_t_units(self, program_id: str) -> int:
        return len(self.unit_stats(program_id))

    def shrunk_score(self, program_id: str) -> float | None:
        """Shrunk T score: sum_{u in U} (S̄_c(u) - S̄_v1(u)) / (|U| + 2) over
        the T units U that both the program and the seed program (v1) have
        rows on.  The +2 shrinks a program measured on few units toward v1,
        which scores 0 by definition; None when U is empty."""

        pid = str(program_id)
        if pid == self.v1_id:
            return 0.0
        st = self.unit_stats(pid)
        v1 = self.unit_stats(self.v1_id)
        units = [u for u in st if u in v1]
        if not units:
            return None
        return sum(st[u]["S"] - v1[u]["S"] for u in units) / (len(units) + 2)

    def _mean_tokens(self, pid: str) -> float:
        c = [v["C"] for v in self.unit_stats(pid).values() if v.get("C") is not None]
        return _mean(c) if c else math.inf

    def top_by_shrunk(self, k: int = 3, *, min_units: int = 3) -> list[str]:
        """Final-selection candidates: the top ``k`` programs other than the
        seed program with rows on >= ``min_units`` T units (a program that
        reached R2 has rows on its R1 unit and its R2 units, the guard
        included), best shrunk T score first (ties: fewer mean tokens)."""

        pool = []
        for pid, rec in self.programs.items():
            if pid == self.v1_id:
                continue
            if self.n_t_units(pid) < int(min_units):
                continue
            score = self.shrunk_score(pid)
            if score is None:
                continue
            pool.append((-score, self._mean_tokens(pid), pid))
        return [pid for _s, _c, pid in sorted(pool)[: max(0, int(k))]]

    def final_candidates(self, k: int = 3, *, min_units: int = 3) -> list[str]:
        """:meth:`top_by_shrunk` plus the seed program (v1), which is always a
        final-selection candidate."""

        return self.top_by_shrunk(k, min_units=min_units) + [self.v1_id]

    def best_incumbent(self, *, extra_uses: Mapping[str, int] | None = None,
                       exclude: Iterable[str] = ()) -> str:
        """The best-mean incumbent: highest shrunk T score among the seed
        program (v1) and the programs that reached R2 (or have >= 3 T
        units), eligible as a parent; ties prefer confirmed, then more T
        units, then the older program.  v1 when no program qualifies."""

        banned = set(exclude)
        pool = []
        for pid in dict.fromkeys([self.v1_id] + sorted(self.programs)):
            if pid in banned:
                continue
            rec = self.programs.get(pid) or {}
            if pid != self.v1_id and not (rec.get("reached_r2") or self.n_t_units(pid) >= 3):
                continue
            if not self.eligible_parent(pid, extra_uses):
                continue
            score = self.shrunk_score(pid)
            if score is None:
                continue
            pool.append((-score, not self.is_confirmed(pid), -self.n_t_units(pid),
                         rec.get("gen") if isinstance(rec.get("gen"), int) else -1, pid))
        if not pool:
            return self.v1_id
        return sorted(pool)[0][-1]

    def specialist(self, unit_id: str, *, extra_uses: Mapping[str, int] | None = None,
                   exclude: Iterable[str] = ()) -> str:
        """The specialist parent of a target unit: the eligible program with
        the best S̄ on it (ties: more reps, confirmed, higher shrunk score,
        older); the best incumbent when none is eligible."""

        banned = set(exclude)
        means = self.unit_means(unit_id)
        pool = []
        for pid, s in means.items():
            if pid in banned or not self.eligible_parent(pid, extra_uses):
                continue
            st = self.unit_stats(pid, t_only=False).get(unit_id) or {}
            rec = self.programs.get(pid) or {}
            gen = rec.get("gen") if isinstance(rec.get("gen"), int) else -1
            pool.append((-s, -int(st.get("n") or 0), not self.is_confirmed(pid),
                         -(self.shrunk_score(pid) or 0.0), gen, pid))
        if not pool:
            return self.best_incumbent(extra_uses=extra_uses, exclude=banned)
        return sorted(pool)[0][-1]

    def best_per_unit_rows(self, units: Iterable[str] | None = None, *,
                           program_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
        """unit -> rows of the program with the best ESTABLISHED S̄ there
        (:meth:`is_established`; over every program when none is), T units
        only: the rows the failure map is built from
        (``build_failure_map(best-per-unit rows)``).

        ``program_id``: that one program's view instead (its own rows, else
        the seed program's census rows; :meth:`view_rows`): the failure map
        of an arm without the archive, which keeps no memory of other
        programs."""

        out = {}
        for unit in (units if units is not None else self.units()):
            if not self.is_t(unit):
                continue
            if program_id is not None:
                _src, rows = self.view_rows(program_id, unit)
                if rows:
                    out[unit] = [dict(r) for r in rows]
                continue
            means = self.unit_means(unit)
            if not means:
                continue
            pool = {p: s for p, s in means.items() if self.is_established(p, unit)} or means
            best = sorted(pool, key=lambda p: (
                -pool[p], -len(self.unit_rows(p, unit)), p != self.v1_id, p))[0]
            out[unit] = [dict(r) for r in self.unit_rows(best, unit)]
        return out

    def failure_map(self, units: Iterable[str] | None = None, *,
                    program_id: str | None = None) -> dict[str, Any]:
        from queenbee.evo.diagnosis import build_failure_map

        return build_failure_map(self.best_per_unit_rows(units, program_id=program_id))

    def merge_pairs(self, *, exclude: Iterable[Iterable[str]] = (),
                    extra_uses: Mapping[str, int] | None = None,
                    units: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Complementary front specialists (not the seed program, both
        eligible parents): ``wins[a]`` = the units a LEADS (beats every
        measured program by >= 1/n, so b is not on their front -- measured
        there or not), ``wins[b]`` the same for b, both non-empty (disjoint:
        one leader per unit).  Leadership is used instead of head-to-head
        units because a program has rows on only ~4 units, so two programs
        rarely share a measured unit.  Pairs in ``exclude`` (already merged)
        and pairs with the same behaviour fingerprint are skipped.  Best
        pairs first (balanced, then larger, then higher combined shrunk
        score); the first parent is the one with the higher shrunk score."""

        pareto = self.pareto(units)
        banned = {frozenset(map(str, pair)) for pair in exclude}
        leads: dict[str, list[str]] = {}
        for unit, e in pareto["units"].items():
            if e.get("leader") and e["leader"] != self.v1_id:
                leads.setdefault(str(e["leader"]), []).append(unit)
        specialists = [p for p in leads if self.eligible_parent(p, extra_uses)]
        out = []
        for a, b in combinations(sorted(specialists), 2):
            if frozenset((a, b)) in banned:
                continue
            fa = (self.programs.get(a) or {}).get("fp")
            if fa and fa == (self.programs.get(b) or {}).get("fp"):
                continue
            wa = sorted(leads[a], key=unit_sort_key)
            wb = sorted(leads[b], key=unit_sort_key)
            if not wa or not wb:
                continue
            score = (self.shrunk_score(a) or 0.0) + (self.shrunk_score(b) or 0.0)
            # parent first = the higher shrunk score (it is mutated)
            first, second = (a, b) if (self.shrunk_score(a) or 0.0) >= (self.shrunk_score(b) or 0.0) \
                else (b, a)
            wins = {a: wa, b: wb}
            out.append({"parent_ids": [first, second],
                        "wins": {first: wins[first], second: wins[second]},
                        "_key": (-min(len(wa), len(wb)), -(len(wa) + len(wb)), -score, first, second)})
        out.sort(key=lambda d: d["_key"])
        for d in out:
            d.pop("_key")
        return out

    def exemplars(self, entries: Sequence[Mapping[str, Any]], *,
                  target_class: str | None = None, unit_id: str | None = None,
                  exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
        """Contrastive exemplars: the program that best fixed ``target_class``
        (confirmed first, then highest target dS > 0) and the clearest
        failure on it (refuted, lowest dS).  Without a class (diagnosis cards
        off, or no open cell in the failure map) the same rule runs on
        proposals whose R1 unit was ``unit_id``.
        Only programs in the archive; never the seed program or an excluded
        program (the parent)."""

        banned = set(map(str, exclude)) | {self.v1_id}
        if target_class:
            cls = _fclass(target_class)
            pool = [e for e in entries if _fclass(e.get("target_class")) == cls]
        elif unit_id:
            pool = [e for e in entries if (e.get("target_units") or [None])[0] == unit_id]
        else:
            return []
        pool = [e for e in pool if e.get("verdict") in ("confirmed", "refuted", "inconclusive")
                and e.get("target_dS") is not None and str(e.get("program_id")) in self.rows
                and str(e.get("program_id")) not in banned]
        fixed = sorted((e for e in pool if float(e["target_dS"]) > EPS),
                       key=lambda e: (e.get("verdict") != "confirmed", -float(e["target_dS"]),
                                      str(e.get("program_id"))))
        failed = sorted((e for e in pool if e.get("verdict") == "refuted"),
                        key=lambda e: (float(e["target_dS"]), str(e.get("program_id"))))
        out = []
        for role, items in (("fixed", fixed), ("failed", failed)):
            for e in items:
                if any(x["program_id"] == str(e["program_id"]) for x in out):
                    continue
                observed = {u: round(float(v), 4) for u, v in (e.get("observed") or {}).items()
                            if self.is_t(u)}
                out.append({"program_id": str(e["program_id"]), "role": role,
                            "failure_class": e.get("target_class") or target_class,
                            "observed": observed})
                break
        return out

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "v1_id": self.v1_id, "t_templates": self.t_templates,
            "programs": self.programs, "rows": self.rows,
            "confirm_requested": self.confirm_requested, "confirmed": self.confirmed,
            "confirm_failed": self.confirm_failed, "confirm_undecided": self.confirm_undecided,
            "parent_uses": self.parent_uses,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Archive":
        out = cls(v1_id=str(data.get("v1_id") or "v1"), t_templates=data.get("t_templates"))
        out.programs = {str(k): dict(v) for k, v in dict(data.get("programs") or {}).items()}
        out.rows = {str(p): {str(u): {str(r): dict(row) for r, row in dict(t).items()}
                             for u, t in dict(units).items()}
                    for p, units in dict(data.get("rows") or {}).items()}
        for name in ("confirm_requested", "confirmed", "confirm_failed", "confirm_undecided"):
            setattr(out, name, {str(k): list(v) for k, v in dict(data.get(name) or {}).items()})
        out.parent_uses = {str(k): int(v) for k, v in dict(data.get("parent_uses") or {}).items()}
        return out


# --------------------------------------------------------------------------- #
# Hypothesis ledger
# --------------------------------------------------------------------------- #


def _hypothesis_view(hyp: Any) -> dict[str, Any] | None:
    if not isinstance(hyp, Mapping):
        return None
    out: dict[str, Any] = {}
    units = hyp.get("target_units")
    if isinstance(units, (list, tuple)):
        out["target_units"] = [str(u)[:32] for u in units[:12]]
    if hyp.get("failure_class"):
        out["failure_class"] = _fclass(hyp["failure_class"])
    if hyp.get("mechanism"):
        out["mechanism"] = " ".join(str(hyp["mechanism"]).split())[:300]
    pred = hyp.get("predicted_dS")
    if isinstance(pred, Mapping):
        out["predicted_dS"] = {str(k)[:32]: float(v) for k, v in pred.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    elif isinstance(pred, (int, float)) and not isinstance(pred, bool):
        out["predicted_dS"] = float(pred)
    if hyp.get("predicted_effect"):
        out["predicted_effect"] = " ".join(str(hyp["predicted_effect"]).split())[:300]
    if hyp.get("raw"):
        out["raw"] = " ".join(str(hyp["raw"]).split())[:300]
    return out


class Ledger:
    """Host-verified hypothesis ledger (see the module docstring).
    ``entries`` are the dicts :meth:`record` builds, in proposal order
    (entry i is ``L<i+1>`` in the prompt's ledger block)."""

    def __init__(self, entries: Iterable[Mapping[str, Any]] | None = None) -> None:
        self.entries: list[dict[str, Any]] = [dict(e) for e in entries or []]

    # ------------------------------------------------------------------ #
    @staticmethod
    def pair(entry: Mapping[str, Any]) -> tuple[str, str]:
        return (_fclass(entry.get("target_class")) or "unknown",
                str(entry.get("mech_key") or mech_key(entry.get("mech_tags_host"),
                                                     entry.get("mech_claim_norm"))))

    @staticmethod
    def evidence_key(entry: Mapping[str, Any]) -> tuple:
        """The fresh-parent draw a refutation rests on: (parent, R1 unit,
        generation).  Proposals sharing it were paired against ONE parent rep,
        so their refutations are one piece of evidence; an entry that does not
        record all three counts on its own."""

        parent = (list(entry.get("parent_ids") or []) or [None])[0]
        unit = (list(entry.get("target_units") or []) or [None])[0]
        gen = entry.get("gen")
        if parent is None or unit is None or not isinstance(gen, int):
            return ("entry", str(entry.get("program_id")), str(gen))
        return (str(parent), str(unit), int(gen))

    @staticmethod
    def counts_as_refutation(entry: Mapping[str, Any]) -> bool:
        """A ``refuted`` entry that is evidence against a (class, mechanism)
        pair: not an economize proposal (its goal is equal S at lower cost,
        which the S verdict does not test)."""

        return entry.get("verdict") == "refuted" and \
            (entry.get("brief_kind") or entry.get("kind")) != "economize"

    def refuted_counts(self, *, before: int | None = None) -> Counter:
        """(class, mechanism) -> INDEPENDENT refutations (distinct
        :meth:`evidence_key`; economize excluded), over the first ``before``
        entries when given."""

        items = self.entries if before is None else self.entries[:before]
        seen: dict[tuple[str, str], set] = {}
        for e in items:
            if self.counts_as_refutation(e):
                seen.setdefault(self.pair(e), set()).add(self.evidence_key(e))
        return Counter({pair: len(keys) for pair, keys in seen.items()})

    def tabu(self, *, before: int | None = None) -> list[list[str]]:
        """(class, mechanism) pairs refuted independently at least
        ``TABU_REFUTATIONS`` times, as ``[class, mech]``."""

        counts = self.refuted_counts(before=before)
        return [[c, m] for (c, m), n in sorted(counts.items()) if n >= TABU_REFUTATIONS]

    def is_tabu(self, target_class: Any, mechanism_key: str) -> bool:
        key = (_fclass(target_class) or "unknown", str(mechanism_key))
        return any(tuple(t) == key for t in self.tabu())

    def tabu_fn(self, parent_source: str, *, target_class: Any = None):
        """Screen hook (``screen_program(tabu=...)``): ``(source, hypothesis)
        -> reason | None`` on the HOST tags of parent -> source and the brief
        class (else the hypothesis' class)."""

        tabu = {tuple(t) for t in self.tabu()}

        def check(source: str, hypothesis: Mapping[str, Any] | None) -> str | None:
            cls = _fclass(target_class) or _fclass((hypothesis or {}).get("failure_class")) \
                or "unknown"
            key = mech_key(mechanism_tags(parent_source, source),
                           normalize_mechanism((hypothesis or {}).get("mechanism")))
            if (cls, key) in tabu:
                return f"tabu: ({cls}, {key}) refuted twice"
            return None

        return check

    def lockin(self, *, min_confirmed: int = LOCKIN_MIN_CONFIRMED,
               share: float = LOCKIN_SHARE) -> dict[str, Any] | None:
        """``--explore-on-lockin``: ``{"family", "n", "of", "counts"}``
        when >= ``min_confirmed`` confirmed entries exist and the most common
        mechanism family (:func:`mechanism_family` of their host tags) holds
        >= ``share`` of them; else None.  Ties: the earlier family of
        :data:`MECHANISM_FAMILIES`."""

        fams = [mechanism_family(e.get("mech_tags_host")) for e in self.entries
                if e.get("verdict") == "confirmed"]
        fams = [f for f in fams if f]
        if len(fams) < max(1, int(min_confirmed)):
            return None
        counts = Counter(fams)
        top = max(MECHANISM_FAMILIES, key=lambda f: (counts[f], -MECHANISM_FAMILIES.index(f)))
        if counts[top] < float(share) * len(fams) - EPS:
            return None
        return {"family": top, "n": int(counts[top]), "of": len(fams),
                "counts": {f: int(counts[f]) for f in MECHANISM_FAMILIES if counts[f]}}

    # ------------------------------------------------------------------ #
    def record(
        self,
        program_id: str,
        *,
        gen: int,
        brief: Mapping[str, Any] | None = None,
        hypothesis: Mapping[str, Any] | None = None,
        parent_source: str | None = None,
        child_source: str | None = None,
        outcome: Mapping[str, Any] | None = None,
        thresholds: Thresholds | None = None,
        screen_reasons: Sequence[str] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Record (or, on resume, replace) the entry of one proposal.

        ``outcome``: ``Racer.outcome`` of the program's generation (None =
        never executed -> ``screened``); ``status``: e.g. ``screened``,
        ``mint_failed``, ``duplicate``.  Mechanism tags are computed from the
        sources (no child source -> no tags)."""

        brief = dict(brief or {})
        hyp = _hypothesis_view(hypothesis)
        brief_class = _fclass(brief.get("target_class")) or None
        hyp_class = (hyp or {}).get("failure_class") or None
        klass = brief_class or hyp_class or "unknown"
        tags = mechanism_tags(parent_source, child_source) if child_source else []
        claim = normalize_mechanism((hyp or {}).get("mechanism"))[:120]
        key = mech_key(tags, claim)
        prior_refuted = self.refuted_counts()
        existing = next((i for i, e in enumerate(self.entries)
                         if e.get("program_id") == str(program_id) and e.get("gen") == int(gen)),
                        None)
        if existing is not None:
            prior_refuted = self.refuted_counts(before=existing)
        outcome = dict(outcome or {})
        verdict = host_verdict(outcome, thresholds) if outcome else "screened"
        text = " ".join(str((hyp or {}).get(k) or "") for k in
                        ("mechanism", "predicted_effect", "raw"))
        cites = sorted({f"L{int(m)}" for m in _LEDGER_REF.findall(text)}, key=lambda s: int(s[1:]))
        predicted = (hyp or {}).get("predicted_dS")
        shape = brief.get("target_shape")
        cell = brief.get("cell")
        entry = {
            "program_id": str(program_id),
            "gen": int(gen),
            "brief_id": brief.get("brief_id"),
            "brief_kind": brief.get("kind"),
            "slot": brief.get("slot"),
            "parent_ids": [str(p) for p in brief.get("parent_ids") or []],
            "target_units": [str(u) for u in brief.get("target_units") or []],
            "target_class": klass,
            "brief_class": brief_class,
            "target_shape": shape,
            "cell": list(cell) if isinstance(cell, (list, tuple)) else None,
            "mech_tags_host": tags,
            "mech_claim_norm": claim,
            "mech_key": key,
            "diff_summary": diff_summary(parent_source, child_source) if child_source else None,
            "edit_size": edit_size(parent_source, child_source) if child_source else None,
            "hypothesis": hyp,
            "predicted": predicted if isinstance(predicted, Mapping) else (
                {"mean": predicted} if predicted is not None else {}),
            "observed": {str(u): round(float(v), 6) for u, v in (outcome.get("observed") or {}).items()},
            "target_dS": outcome.get("target_dS"),
            "target_dS_all": outcome.get("target_dS_all"),
            "target_dC": outcome.get("target_dC"),
            "d1": outcome.get("d1"),
            "reached_r2": bool(outcome.get("reached_r2")),
            "D": outcome.get("D"),
            "D_note": outcome.get("D_note"),
            "guard_ok": outcome.get("guard_ok"),
            "high_tier": bool(outcome.get("high_tier")),
            "verdict": verdict,
            "status": status or outcome.get("status") or ("screened" if not outcome else "executed"),
            "screen_reasons": [str(r)[:200] for r in screen_reasons or []],
            "repeat_of_refuted": prior_refuted[(klass, key)] >= 1,
            "repeat_of_tabu": prior_refuted[(klass, key)] >= TABU_REFUTATIONS,
            "cites_ledger": cites,
            "cites_class": bool(brief_class and hyp_class and brief_class == hyp_class),
        }
        if isinstance(brief.get("lockin"), Mapping):
            # --explore-on-lockin: the directive this proposal answered plus
            # the mechanism family of its own edit (key absent otherwise)
            entry["lockin"] = dict(brief["lockin"]) | {
                "child_family": mechanism_family(tags)}
        if existing is not None:
            self.entries[existing] = entry
        else:
            self.entries.append(entry)
        return entry

    # ------------------------------------------------------------------ #
    def cell_stats(self) -> dict[str, dict[str, Any]]:
        """``"shape|class"`` -> attempts, rewards (target dS; 0 for a
        screened proposal), refuted / confirmed counts (entries that carry a
        cell).  An entry without a measured target dS that was not screened
        (infra, not raced for budget, no fresh parent rep) is not a pull:
        an outage must not penalise a cell."""

        out: dict[str, dict[str, Any]] = {}
        for e in self.entries:
            cell = e.get("cell")
            if not cell:
                continue
            reward = e.get("target_dS")
            if str(e.get("status") or "") in NON_PULL_STATUSES:
                continue
            if reward is None and e.get("verdict") != "screened":
                continue
            key = f"{cell[0]}|{cell[1]}"
            slot = out.setdefault(key, {"n": 0, "rewards": [], "refuted": 0, "confirmed": 0})
            slot["n"] += 1
            slot["rewards"].append(float(reward) if reward is not None else 0.0)
            if e.get("verdict") == "refuted":
                slot["refuted"] += 1
            elif e.get("verdict") == "confirmed":
                slot["confirmed"] += 1
        return out

    def unit_attempts(self) -> Counter:
        """Proposals per R1 unit (the first brief target)."""

        return Counter((e.get("target_units") or [None])[0] for e in self.entries
                       if e.get("target_units"))

    def prompt_view(self, flags: Any = None) -> dict[str, Any]:
        """State fields of the prompt's ledger block (empty when the ledger
        is off)."""

        if not coerce_flags(flags).ledger:
            return {"ledger": [], "tabu": []}
        return {"ledger": [dict(e) for e in self.entries], "tabu": self.tabu()}

    @staticmethod
    def _rates(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        verdicts = Counter(e.get("verdict") for e in items)
        executed = [e for e in items if e.get("verdict") != "screened"]
        hits = [e for e in executed if e.get("target_dS") is not None]
        return {
            "n": len(items),
            "verdicts": dict(verdicts),
            "confirmed_rate": _r(verdicts["confirmed"] / len(executed)) if executed else None,
            "hit_rate_dS_ge": {tau: _r(sum(1 for e in hits if float(e["target_dS"]) >= tau - EPS)
                                       / len(hits)) for tau in (0.2,)} if hits else None,
        }

    def summary(self, *, gen: int | None = None) -> dict[str, Any]:
        """Verdict counts and rates; ``by_kind`` splits them by brief kind
        (a merge child is paired against its first parent on a unit the
        second parent leads, so its hit rate is not comparable to a target
        slot's)."""

        items = [e for e in self.entries if gen is None or e.get("gen") == gen]
        rates = self._rates(items)
        executed = [e for e in items if e.get("verdict") != "screened"]
        kinds = sorted({str(e.get("brief_kind") or "unknown") for e in items})
        return {
            **rates,
            "by_kind": {k: self._rates([e for e in items
                                        if str(e.get("brief_kind") or "unknown") == k])
                        for k in kinds},
            "repeat_of_refuted": sum(1 for e in items if e.get("repeat_of_refuted")),
            "repeat_of_tabu_executed": sum(1 for e in executed if e.get("repeat_of_tabu")),
            "citation_rate": _r(sum(1 for e in items if e.get("cites_ledger") or e.get("cites_class"))
                                / len(items)) if items else None,
            "tabu": self.tabu(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [dict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> "Ledger":
        if isinstance(data, Mapping):
            return cls(data.get("entries") or [])
        return cls(data)


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


def _closed(info: Mapping[str, Any]) -> bool:
    if info.get("closed") is not None:
        return bool(info["closed"])
    best = info.get("best")
    return best is not None and float(best) >= 1.0 - EPS


class Scheduler:
    """K briefs per generation (see the module docstring).  Pure ``briefs``
    + idempotent ``commit``; ``to_dict`` / ``from_dict`` for checkpoints."""

    def __init__(
        self,
        flags: Any = None,
        *,
        arm: str | None = None,
        v1_id: str = "v1",
        k: int = K_BRIEFS,
        n_target: int = N_TARGET_SLOTS,
        ucb_c: float = 0.5,
        prior_weight: float = 1.0,
        refute_penalty: float = 0.2,
        max_target_units: int = 3,
    ) -> None:
        if arm is not None:
            key = str(arm).strip().lower().replace("-", "_").replace("+", "_")
            if key not in ARM_FLAGS:
                raise ValueError(f"unknown arm {arm!r}; choose {sorted(ARM_FLAGS)}")
            flags = ARM_FLAGS[key] if flags is None else flags
            arm = key
        self.flags: EvoFlags = coerce_flags(flags)
        self.arm = arm
        self.v1_id = v1_id
        self.k = int(k)
        self.n_target = min(int(n_target), self.k)
        self.ucb_c = float(ucb_c)
        self.prior_weight = float(prior_weight)
        self.refute_penalty = float(refute_penalty)
        self.max_target_units = max(1, int(max_target_units))
        self.merged_pairs: list[list[str]] = []
        self.committed: list[str] = []
        self.issued: list[dict[str, Any]] = []
        #: ``--explore-on-lockin`` (off by default; set from the run config,
        #: not serialized): while the ledger is locked in (``Ledger.lockin``: the
        #: confirmed entries concentrate in one mechanism family) one slot per
        #: generation is an explore brief that asks for ANOTHER family -- the
        #: rotating slot when it is explore, else the last target slot.  Arms
        #: without the ledger never read it, so the option is inert there.
        self.explore_on_lockin = False

    # ------------------------------------------------------------------ #
    def slot_kinds(self, gen: int) -> list[str]:
        """The schedule (identical in every arm): n_target x target, then the
        rotating slot(s) merge -> economize -> explore by generation.  With
        ``explore_on_lockin`` :meth:`briefs` may turn the last target slot of
        a locked-in arm with the ledger into an explore brief, so the briefs
        then differ from those of an arm without the ledger."""

        rot = self.k - self.n_target
        return ["target"] * self.n_target + [
            ROTATION[(int(gen) + j) % len(ROTATION)] for j in range(rot)]

    def view_program(self, archive: Archive) -> str | None:
        """Whose rows the scheduler may look at: None = the whole archive
        (archive on); else ONE program, the best-mean incumbent, with the
        seed program's census rows where it has none.  An arm without the
        archive keeps no memory of other programs: it never sees another
        program's per-unit results, not even through the failure map, the
        open / closed units or the UCB headroom prior."""

        if self.flags.archive:
            return None
        return archive.best_incumbent()

    def _unit_info(self, archive: Archive, units: Sequence[str],
                   fmap: Mapping[str, Any] | None,
                   shapes: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
        """Per unit: ``best`` (headroom prior: the best ESTABLISHED S̄, or the
        view program's S̄; :meth:`view_program`), ``closed`` (solved: best
        >= 1 on established evidence -- never on one lucky rep), shape,
        class.  (Subclasses may override; an entry without ``closed`` is
        closed iff ``best >= 1``.)"""

        view = self.view_program(archive)
        fm_units = (fmap or {}).get("units") or {}
        info = {}
        for u in units:
            f = fm_units.get(u) or {}
            if view is None:
                best = archive.robust_best(u)
                closed = best is not None and best >= 1.0 - EPS
            else:
                src, rows = archive.view_rows(view, u)
                vals = [float(r["S"]) for r in rows if r.get("S") is not None]
                best = _mean(vals) if vals else None
                closed = bool(best is not None and best >= 1.0 - EPS and src is not None
                              and archive.is_established(src, u))
            info[u] = {"best": None if best is None else round(best, 6), "closed": closed,
                       "shape": (shapes or {}).get(u) or f.get("shape"),
                       "class": f.get("dominant")}
        return info

    def _cells(self, fmap: Mapping[str, Any] | None, units: Sequence[str],
               info: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        in_play = set(units)
        open_units = {u for u in units if not _closed(info[u])}
        cells: dict[tuple[str, str], list[str]] = {}
        for cell in (fmap or {}).get("cells") or []:
            members = [u for u in cell.get("units") or [] if u in in_play and u in open_units]
            if members:
                key = (str(cell.get("shape") or "unknown-shape"), str(cell.get("failure_class")))
                cells.setdefault(key, []).extend(members)
        covered = {u for m in cells.values() for u in m}
        for u in sorted(open_units - covered, key=unit_sort_key):
            key = (str(info[u]["shape"] or "unknown-shape"), str(info[u]["class"] or "unknown"))
            cells.setdefault(key, []).append(u)
        return [{"shape": s, "failure_class": c,
                 "units": sorted(dict.fromkeys(m), key=unit_sort_key)}
                for (s, c), m in sorted(cells.items())]

    def cell_value(self, cell: Mapping[str, Any], ledger: "Ledger",
                   info: Mapping[str, Mapping[str, Any]], virtual: Counter | None = None,
                   stats: Mapping[str, Any] | None = None) -> float:
        """UCB value of a cell: (rewards + w * headroom prior) / (n + w)
        + c * sqrt(ln(1 + N) / (1 + n)) - penalty * refuted (the penalty only
        with the ledger on; the bonus is c while N = 0).  Rewards are the
        target dS of earlier proposals on the cell; the prior is the cell's
        headroom (1 - mean best S̄ of its units); pulls already assigned this
        generation count as reward 0 (virtual loss)."""

        virtual = virtual if virtual is not None else Counter()
        stats = stats if stats is not None else ledger.cell_stats()
        total = sum(v["n"] for v in stats.values()) + sum(virtual.values())
        key = f"{cell['shape']}|{cell['failure_class']}"
        st = stats.get(key) or {"n": 0, "rewards": [], "refuted": 0}
        n = int(st["n"]) + virtual[key]
        bests = [info[u]["best"] for u in cell["units"] if info[u]["best"] is not None]
        prior = 1.0 - (_mean(bests) if bests else 0.0)
        mean = (sum(st["rewards"]) + self.prior_weight * prior) / (n + self.prior_weight)
        bonus = self.ucb_c * math.sqrt(math.log(1.0 + total) / (1.0 + n)) if total else self.ucb_c
        penalty = self.refute_penalty * int(st.get("refuted") or 0) if self.flags.ledger else 0.0
        return round(mean + bonus - penalty, 9)

    @staticmethod
    def _cell_headroom(cell: Mapping[str, Any], info: Mapping[str, Mapping[str, Any]]) -> float:
        bests = [info[u]["best"] for u in cell["units"] if info[u]["best"] is not None]
        return 1.0 - (_mean(bests) if bests else 0.0)

    def _ucb_pick(self, cells: list[dict[str, Any]], ledger: "Ledger",
                  info: Mapping[str, Mapping[str, Any]], virtual: Counter) -> dict[str, Any]:
        """Batch UCB over the cells with enough headroom
        (``MIN_HEADROOM_FRAC``): the best cell not yet picked this
        generation (distinct cells first); once every such cell is picked,
        virtual loss decides."""

        stats = ledger.cell_stats()
        room = {id(c): self._cell_headroom(c, info) for c in cells}
        top = max(room.values(), default=0.0)
        eligible = [c for c in cells if room[id(c)] >= MIN_HEADROOM_FRAC * top - EPS]
        fresh = [c for c in eligible if not virtual[f"{c['shape']}|{c['failure_class']}"]]
        # same rule as the unit ranking without diagnosis cards: low-headroom
        # cells never take a slot while an eligible one is open (virtual loss
        # spreads the repeats)
        pool = fresh or eligible or cells
        best = max(pool, key=lambda c: (self.cell_value(c, ledger, info, virtual, stats),
                                        len(c["units"]), c["failure_class"], c["shape"]))
        virtual[f"{best['shape']}|{best['failure_class']}"] += 1
        return best

    def _parent(self, unit: str | None, archive: Archive, uses: Counter) -> str:
        if self.flags.archive and unit is not None:
            return archive.specialist(unit, extra_uses=uses)
        return archive.best_incumbent(extra_uses=uses)

    def _exemplars(self, archive: Archive, ledger: "Ledger", parent: str,
                   target_class: str | None, unit: str | None) -> list[dict[str, Any]]:
        if not self.flags.archive:
            return []
        return archive.exemplars(ledger.entries, target_class=target_class,
                                 unit_id=None if target_class else unit, exclude=[parent])

    def briefs(
        self,
        gen: int,
        *,
        archive: Archive,
        ledger: "Ledger",
        t_units: Sequence[str],
        fmap: Mapping[str, Any] | None = None,
        shapes: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """The generation's K briefs (pure: nothing is mutated).  ``t_units``:
        the T units in play (in the loop: the frontier T units plus the
        curriculum units the current best has not solved); ``fmap``:
        the failure map over best-per-unit rows (built from the archive when
        diagnosis cards are on and none is given)."""

        gen = int(gen)
        units = sorted(dict.fromkeys(u for u in t_units if archive.is_t(u)), key=unit_sort_key)
        for u in units:
            assert_dev_unit(u, where="scheduler units")
        if not units:
            raise ValueError("scheduler needs at least one T unit")
        if self.flags.diag and fmap is None:
            fmap = archive.failure_map(units, program_id=self.view_program(archive))
        info = self._unit_info(archive, units, fmap if self.flags.diag else None, shapes)
        uses: Counter = Counter()
        r1_used: list[str] = []
        attempts = ledger.unit_attempts()
        out: list[dict[str, Any]] = []
        kinds = self.slot_kinds(gen)
        target_plans = self._target_plans(gen, archive, ledger, units, info, fmap, attempts)
        lock, lock_slot = self._lockin_slot(kinds, ledger)
        for slot, slot_kind in enumerate(kinds):
            brief: dict[str, Any] = {
                "brief_id": f"g{gen}-s{slot}", "gen": gen, "slot": slot,
                "slot_kind": slot_kind, "kind": slot_kind, "target_units": [],
                "target_class": None, "target_shape": None, "cell": None,
                "parent_ids": [], "exemplars": [], "wins": {}, "fallback": None,
                "flags": {"diag": self.flags.diag, "ledger": self.flags.ledger,
                          "archive": self.flags.archive},
            }
            if slot == lock_slot:
                # --explore-on-lockin: the reserved exploration slot (rotating
                # explore, or the last target slot turned explore) carries the
                # lock-in directive
                self._fill_rotating(brief, "explore", archive, ledger, units, info, fmap,
                                    attempts, uses, r1_used)
                if slot_kind != "explore":
                    brief["fallback"] = "lockin"
                brief["lockin"] = {k: lock[k] for k in ("family", "n", "of")}
            elif slot_kind == "target":
                plan = target_plans[slot % len(target_plans)]
                r1 = plan["units"][0]
                if r1 in r1_used:
                    alt = [u for u in plan["units"] if u not in r1_used]
                    if alt:
                        plan = {**plan, "units": alt + [u for u in plan["units"] if u not in alt]}
                        r1 = alt[0]
                parent = self._parent(r1, archive, uses)
                brief.update({
                    "target_units": plan["units"][: self.max_target_units],
                    "target_class": plan.get("class"), "target_shape": plan.get("shape"),
                    "cell": plan.get("cell"), "parent_ids": [parent],
                })
                brief["exemplars"] = self._exemplars(archive, ledger, parent,
                                                     plan.get("class"), r1)
            else:
                self._fill_rotating(brief, slot_kind, archive, ledger, units, info, fmap,
                                    attempts, uses, r1_used)
            for p in brief["parent_ids"]:
                uses[p] += 1
            if brief["target_units"]:
                r1_used.append(brief["target_units"][0])
            out.append(brief)
        return out

    def _lockin_slot(self, kinds: Sequence[str],
                     ledger: "Ledger") -> tuple[dict[str, Any] | None, int | None]:
        """(lock-in record, reserved slot) for ``explore_on_lockin``: the
        rotating explore slot when the schedule has one, else the last target
        slot (at least one target slot always stays); (None, None) when the
        option is off, the arm has no ledger, or the ledger is not locked
        in."""

        if not getattr(self, "explore_on_lockin", False) or not self.flags.ledger:
            return None, None
        lock = ledger.lockin()
        if lock is None:
            return None, None
        if "explore" in kinds:
            return lock, list(kinds).index("explore")
        targets = [i for i, k in enumerate(kinds) if k == "target"]
        if len(targets) >= 2:
            return lock, targets[-1]
        return None, None

    def _target_plans(self, gen: int, archive: Archive, ledger: "Ledger", units: Sequence[str],
                      info: Mapping[str, Mapping[str, Any]], fmap: Mapping[str, Any] | None,
                      attempts: Counter) -> list[dict[str, Any]]:
        n = max(1, self.n_target)
        if self.flags.diag:
            cells = self._cells(fmap, units, info)
            if cells:
                virtual: Counter = Counter()
                plans = []
                used: list[str] = []
                for _ in range(n):
                    cell = self._ucb_pick(cells, ledger, info, virtual)
                    ordered = sorted(cell["units"], key=lambda u: (
                        u in used, attempts[u], info[u]["best"] if info[u]["best"] is not None else 0.0,
                        unit_sort_key(u)))
                    used.append(ordered[0])
                    plans.append({"units": ordered, "class": cell["failure_class"],
                                  "shape": cell["shape"],
                                  "cell": [cell["shape"], cell["failure_class"]]})
                return plans
        # diagnosis cards off (--no-diag) or no open cell: the lowest-S units,
        # no class
        if self.flags.archive:
            score = {u: info[u]["best"] for u in units}
        else:
            parent = archive.best_incumbent()
            st = archive.unit_stats(parent)
            score = {u: (st.get(u) or {}).get("S", info[u]["best"]) for u in units}
        ranked = sorted(units, key=lambda u: (
            score[u] if score[u] is not None else 0.0, attempts[u], unit_sort_key(u)))
        room = {u: 1.0 - (score[u] if score[u] is not None else 0.0) for u in units}
        top = max(room.values(), default=0.0)
        ranked = [u for u in ranked if room[u] >= MIN_HEADROOM_FRAC * top - EPS] or ranked
        picks: list[str] = []
        for u in ranked:
            if template_of(u) not in {template_of(p) for p in picks}:
                picks.append(u)
            if len(picks) == n:
                break
        for u in ranked:
            if len(picks) >= n:
                break
            if u not in picks:
                picks.append(u)
        plans = []
        for i in range(n):
            first = picks[i % len(picks)]
            rest = [u for u in picks if u != first]
            plans.append({"units": [first] + rest, "class": None,
                          "shape": info[first]["shape"] if self.flags.diag else None,
                          "cell": None})
        return plans

    def _fill_rotating(self, brief: dict[str, Any], slot_kind: str, archive: Archive,
                       ledger: "Ledger", units: Sequence[str],
                       info: Mapping[str, Mapping[str, Any]], fmap: Mapping[str, Any] | None,
                       attempts: Counter, uses: Counter, r1_used: Sequence[str]) -> None:
        kind = slot_kind
        if kind == "merge":
            if not self.flags.archive:
                kind, brief["fallback"] = "explore", "no_archive"
            else:
                pairs = archive.merge_pairs(exclude=self.merged_pairs, extra_uses=uses,
                                            units=units)
                if not pairs:
                    kind, brief["fallback"] = "explore", "no_merge_pair"
                else:
                    pair = pairs[0]
                    a, b = pair["parent_ids"]
                    wins = pair["wins"]
                    targets = list(wins[b][:2]) + list(wins[a][:1])
                    klass = (info.get(targets[0]) or {}).get("class") if self.flags.diag else None
                    brief.update({"kind": "merge", "parent_ids": [a, b], "wins": wins,
                                  "target_units": targets[: self.max_target_units],
                                  "target_class": klass if klass not in ("ok", None) else None})
                    return
        parent = archive.best_incumbent(extra_uses=uses)
        brief["kind"] = kind
        brief["parent_ids"] = [parent]
        if kind == "economize":
            st = archive.unit_stats(parent)
            saturated = [u for u in units if (st.get(u) or {}).get("S", 0.0) >= 1.0 - EPS]
            pool = saturated or [u for u in units if u in st] or list(units)
            ordered = sorted(pool, key=lambda u: (
                -((st.get(u) or {}).get("C") or 0.0), u in r1_used, unit_sort_key(u)))
            brief["target_units"] = ordered[: self.max_target_units]
            return
        # explore: open units, those not yet an R1 unit this generation
        # first, then the least attempted
        open_units = [u for u in units if not _closed(info[u])] or list(units)
        ordered = sorted(open_units, key=lambda u: (u in r1_used, attempts[u], unit_sort_key(u)))
        brief["target_units"] = ordered[: self.max_target_units]

    # ------------------------------------------------------------------ #
    def commit(self, briefs: Sequence[Mapping[str, Any]], archive: Archive) -> int:
        """Record the parents' uses and merged pairs of issued briefs, once
        per brief id (a re-commit after a resume is a no-op)."""

        done = 0
        for b in briefs:
            bid = str(b.get("brief_id"))
            if bid in self.committed:
                continue
            for p in b.get("parent_ids") or []:
                archive.note_parent_use(p)
            if b.get("kind") == "merge" and len(b.get("parent_ids") or []) >= 2:
                self.merged_pairs.append(sorted(map(str, b["parent_ids"][:2])))
            self.committed.append(bid)
            self.issued.append({k: b.get(k) for k in (
                "brief_id", "gen", "slot", "slot_kind", "kind", "target_units",
                "target_class", "cell", "parent_ids", "fallback")})
            done += 1
        return done

    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": {"diag": self.flags.diag, "ledger": self.flags.ledger,
                      "archive": self.flags.archive},
            "arm": self.arm, "v1_id": self.v1_id,
            "k": self.k, "n_target": self.n_target, "ucb_c": self.ucb_c,
            "prior_weight": self.prior_weight, "refute_penalty": self.refute_penalty,
            "max_target_units": self.max_target_units,
            "merged_pairs": [list(p) for p in self.merged_pairs],
            "committed": list(self.committed), "issued": [dict(i) for i in self.issued],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scheduler":
        out = cls(
            dict(data.get("flags") or {}),
            v1_id=str(data.get("v1_id") or "v1"), k=int(data.get("k") or K_BRIEFS),
            n_target=int(data.get("n_target") or N_TARGET_SLOTS),
            ucb_c=float(data.get("ucb_c", 0.5)), prior_weight=float(data.get("prior_weight", 1.0)),
            refute_penalty=float(data.get("refute_penalty", 0.2)),
            max_target_units=int(data.get("max_target_units") or 3),
        )
        out.arm = data.get("arm")
        out.merged_pairs = [list(p) for p in data.get("merged_pairs") or []]
        out.committed = [str(x) for x in data.get("committed") or []]
        out.issued = [dict(x) for x in data.get("issued") or []]
        return out


# --------------------------------------------------------------------------- #
# Prompt state (prompt.build_evo_prompt's state interface)
# --------------------------------------------------------------------------- #


def prompt_state(
    *,
    archive: Archive,
    ledger: Ledger,
    flags: Any,
    sources: Mapping[str, str],
    t_templates: Sequence[Mapping[str, Any]],
    forbidden_case_ids: Iterable[str] = (),
    units: Iterable[str] | None = None,
    budgets: Any = None,
    parent_id: str | None = None,
) -> dict[str, Any]:
    """The ``state`` of ``prompt.build_evo_prompt``: T rows only (never a V
    unit), the failure map (only with diagnosis cards on), the ledger
    entries + tabu (only with the ledger on).

    With the archive on, the failure map is over the archive's best-per-unit
    rows and ``rows`` holds every program's T rows (exemplars, merge
    parents).  WITHOUT the archive nothing crosses from other programs:
    ``rows`` only holds the programs in ``sources`` (the brief's parents),
    and the failure map is the parent's view (``parent_id``: its rows, else
    the seed program's census rows) -- or, without ``parent_id``, left to
    ``build_evo_prompt``, which builds it from the parent's own rows."""

    flags = coerce_flags(flags)
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    visible = None if flags.archive else set(map(str, sources))
    for pid, table in archive.rows.items():
        if visible is not None and pid not in visible:
            continue
        rows[pid] = {u: archive.unit_rows(pid, u) for u in table if archive.is_t(u)}
    if not flags.diag:
        fmap = None
    elif flags.archive:
        fmap = archive.failure_map(units)
    else:
        fmap = archive.failure_map(units, program_id=parent_id) if parent_id else None
    state = {
        "t_templates": list(t_templates),
        "programs": {pid: {"source": src} for pid, src in sources.items()},
        "rows": rows,
        "fmap": fmap,
        "forbidden_case_ids": sorted(set(map(str, forbidden_case_ids))),
        "budgets": budgets,
    }
    state.update(ledger.prompt_view(flags))
    return state


__all__ = [
    "Archive",
    "ESTABLISHED_REPS",
    "K_BRIEFS",
    "LOCKIN_MIN_CONFIRMED",
    "LOCKIN_SHARE",
    "Ledger",
    "MAX_CONFIRM_ATTEMPTS",
    "MECHANISM_FAMILIES",
    "NON_PULL_STATUSES",
    "N_TARGET_SLOTS",
    "ROTATION",
    "Scheduler",
    "TABU_REFUTATIONS",
    "diff_summary",
    "edit_size",
    "margin",
    "mech_key",
    "mechanism_family",
    "mechanism_tags",
    "prompt_state",
]
