"""The QueenBee-Evo planner prompt.

``build_evo_prompt(state, brief, flags)`` assembles the genome-only mint
prompt from ordered blocks.  Blocks shown to EVERY arm:

    header, API card, all T templates' public statements, parent genome,
    parent per-unit S and cost, brief, output contract

Flag-gated blocks, one flag per experience component (each block is simply
absent when its flag is off):

    ``diag``     diagnosis cards: failure map, diagnosis cards (+ credit
                 fields), target diagnosis (host-assigned failure class of
                 the targets)
    ``ledger``   hypothesis ledger: digest (class x mechanism counts, <= 5
                 confirmed hypotheses verbatim; with ``--attempt-diag`` also
                 <= 5 unsuccessful attempts) + tabu list
    ``archive``  program archive: contrastive exemplars, merge parents

For one brief, everything outside the gated blocks is the same byte for
byte under every flag setting (pinned by ``tests/test_evo_prompt.py``),
except that a merge brief is rendered as an explore brief when the archive
is off.

The output contract asks for ``HYPOTHESIS: {target_units, failure_class,
mechanism, predicted_dS}`` plus ONE python genome block and the smallest
effective change; the reply goes through
:func:`queenbee.program.mint.mint_python_challenger` (``genome_only=True``).
:func:`parse_evo_hypothesis` keeps ``predicted_dS``, which the mint's own
:func:`queenbee.program.mint.sanitize_hypothesis` drops.

Leakage guard: V and TEST templates never enter a prompt.  TEST ids, and
the ``forbidden_case_ids`` (V) ids, are refused in every structured input
(``queenbee.evo.credit.assert_not_test`` plus a held-out check), and the
finished prompt is audited for TEST + V ids and, when given,
``forbidden_values`` (T ground truths) inside the evidence sections; any hit
raises ``RuntimeError`` before the prompt can be sent.

State interface (duck-typed; a Mapping or an object with these names):

    t_templates         [{unit_id: <template id>, title, output_sentence,
                          protocol_sentence}]  (see t_template_statements)
    programs            {program_id: str | {"source": str}
                          | {"source_path": path}}
    rows                {program_id: {unit_id: [ExecRow dict, ...]}} (rows
                        as built by ``queenbee.evo.race.Racer.exec_row``)
    fmap                optional failure map (``diag``; else built from the
                        parent's rows)
    ledger              [entry dict, ...] (``ledger``; the entries of
                        ``queenbee.evo.memory.Ledger``)
    tabu                [(failure_class, mechanism) | str] (``ledger``)
    forbidden_case_ids  V template ids (never allowed in a prompt)
    budgets             optional PythonRunBudgets for the API card
    preserve_dups       optional bool (``--preserve-dups``): adds the API
                        card's rule that forwarded raw data keeps repeated
                        values (absent = off)

Brief (Mapping or object): ``kind`` (target | merge | economize | explore),
``target_units``, ``target_class``, ``parent_ids`` (parent first; a merge
brief's second parent second), optional ``exemplars`` ([{program_id, role:
"fixed"|"failed", failure_class, observed: {unit: dS}}]) and ``wins``
({program_id: [units]}), optional ``lockin`` ({family, n, of}:
``--explore-on-lockin``; rendered as one BRIEF line, and :func:`arm_brief`
keeps it only when the ledger is on).  ``brief_id`` is bookkeeping and
never rendered.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from queenbee.evo.api_card import render_api_card
from queenbee.evo.credit import (
    TEST_CASE_IDS,
    assert_not_test,
    credit_legend,
    render_diag_cards_with_credit,
    sealed_case_ids,
)
from queenbee.paths import default_benchmarks_dir
from queenbee.program.mint import EVIDENCE_SECTION_HEADERS as _BASE
from queenbee.tasks import get_task

# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvoFlags:
    """One flag per experience component: ``diag`` (diagnosis cards),
    ``ledger`` (hypothesis ledger), ``archive`` (program archive)."""

    diag: bool = True
    ledger: bool = True
    archive: bool = True


ARM_FLAGS: dict[str, EvoFlags] = {
    "full": EvoFlags(True, True, True),
    "mf_elite": EvoFlags(False, False, False),
}


def coerce_flags(flags: Any) -> EvoFlags:
    """EvoFlags from an EvoFlags, an arm name (case-insensitive; ``-`` and
    ``+`` read as ``_``), a mapping with keys ``diag/ledger/archive``, their
    aliases ``C1/C2/C3`` or ``no_diag/no_ledger/no_archive``, or an object
    with ``diag/ledger/archive`` attributes (None = ``full``)."""

    if flags is None:
        return EvoFlags()
    if isinstance(flags, EvoFlags):
        return flags
    if isinstance(flags, str):
        key = flags.strip().lower().replace("-", "_").replace("+", "_")
        if key not in ARM_FLAGS:
            raise ValueError(f"unknown arm {flags!r}; choose {sorted(ARM_FLAGS)}")
        return ARM_FLAGS[key]
    if isinstance(flags, Mapping):
        out = {"diag": True, "ledger": True, "archive": True}
        for name, alias in (("diag", "C1"), ("ledger", "C2"), ("archive", "C3")):
            if name in flags:
                out[name] = bool(flags[name])
            elif alias in flags:
                out[name] = bool(flags[alias])
            if flags.get(f"no_{name}"):
                out[name] = False
        return EvoFlags(**out)
    return EvoFlags(
        diag=bool(getattr(flags, "diag", True)),
        ledger=bool(getattr(flags, "ledger", True)),
        archive=bool(getattr(flags, "archive", True)),
    )


#: Block names each flag gates.
GATED_BLOCKS: dict[str, tuple[str, ...]] = {
    "diag": ("failure_map", "diag_cards", "target_diagnosis"),
    "ledger": ("ledger",),
    "archive": ("exemplars", "merge_parents"),
}

BRIEF_KINDS: tuple[str, ...] = ("target", "merge", "economize", "explore")

# --------------------------------------------------------------------------- #
# Leak-audit sections
# --------------------------------------------------------------------------- #

#: Every prompt section that carries execution evidence or task text.
EVO_EVIDENCE_SECTION_HEADERS: tuple[str, ...] = tuple(dict.fromkeys(
    tuple(_BASE) + (
        "=== FAILURE MAP",
        "=== DIAGNOSIS CARDS",
        "=== TASK TEMPLATES",
        "=== PARENT RECORD",
        "=== BRIEF",
        "=== TARGET DIAGNOSIS",
        "=== LEDGER",
        "=== CONTRASTIVE EXEMPLARS",
        "=== MERGE PARENTS",
    )
))

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

_UNIT_CHARS = re.compile(r"[^A-Za-z0-9_@\-]")
_CLASS_CHARS = re.compile(r"[^a-z\-]")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _unit(value: Any) -> str:
    return _UNIT_CHARS.sub("", str(value or ""))[:32]


def _fclass(value: Any) -> str:
    return _CLASS_CHARS.sub("", str(value or "").lower())[:24]


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def normalize_mechanism(text: Any) -> str:
    """Dedupe key: lowercase alphanumerics/underscores, single-spaced."""

    return " ".join(re.sub(r"[^a-z0-9_]+", " ", str(text or "").lower()).split())


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _signed(value: float) -> str:
    return f"{value:+.2f}"


def _program_source(state: Any, program_id: Any) -> str | None:
    programs = _get(state, "programs", {}) or {}
    record = programs.get(program_id) if isinstance(programs, Mapping) else None
    if record is None:
        return None
    if isinstance(record, str):
        return record
    source = _get(record, "source")
    if isinstance(source, str) and source:
        return source
    path = _get(record, "source_path")
    if path:
        try:
            return Path(path).read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def _genome_text(source: str | None) -> str:
    from queenbee.program.genome import genome_region

    if not source:
        raise ValueError("program source missing")
    genome = genome_region(source)
    if genome is None:
        raise ValueError("program source has no genome region")
    return genome.rstrip()


def _is_infra(row: Mapping[str, Any]) -> bool:
    if row.get("infra"):
        return True
    if str(row.get("execution_class") or "") == "infra":
        return True
    diag = row.get("diag")
    return isinstance(diag, Mapping) and diag.get("failure_class") == "infra"


def _unit_rows(state: Any, program_id: Any) -> dict[str, list[dict[str, Any]]]:
    rows = _get(state, "rows", {}) or {}
    by_unit = rows.get(program_id) if isinstance(rows, Mapping) else None
    out: dict[str, list[dict[str, Any]]] = {}
    for unit_id, items in (by_unit or {}).items():
        unit = _unit(unit_id)
        if not unit:
            continue
        out[unit] = [r for r in (items or []) if isinstance(r, Mapping)]
    return out


def _row_tokens(row: Mapping[str, Any]) -> float | None:
    value = _num(row.get("C"))
    if value is not None:
        return value
    prompt = _num(row.get("prompt_tokens"))
    completion = _num(row.get("completion_tokens"))
    if prompt is None and completion is None:
        return None
    return (prompt or 0.0) + (completion or 0.0)


def _row_calls(row: Mapping[str, Any]) -> float | None:
    for key in ("model_calls", "worker_model_calls"):
        value = _num(row.get(key))
        if value is not None:
            return value
    return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def unit_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean S / calls / tokens over one unit's non-infra rows."""

    scored = [r for r in rows if isinstance(r, Mapping) and not _is_infra(r)]
    s_values = [v for v in (_num(r.get("S")) for r in scored) if v is not None]
    calls = [v for v in (_row_calls(r) for r in scored) if v is not None]
    tokens = [v for v in (_row_tokens(r) for r in scored) if v is not None]
    return {
        "n": len(s_values),
        "S": _mean(s_values),
        "calls": _mean(calls),
        "tokens": _mean(tokens),
    }


def _forbidden_ids(state: Any, extra: Any) -> list[str]:
    ids = set(sealed_case_ids())
    for source in (_get(state, "forbidden_case_ids", ()) or (), extra or ()):
        if isinstance(source, str):
            source = [source]
        ids |= {str(c).strip() for c in source if str(c).strip()}
    return sorted(ids)


def _check_ids(ids: Iterable[Any], forbidden: set[str], where: str) -> None:
    from queenbee.evo.credit import templates_in

    ids = list(ids)
    assert_not_test(ids, where=where)
    for item in ids:
        for template in templates_in(item):
            if template in forbidden:
                raise RuntimeError(
                    f"LEAKAGE_GUARD: held-out template {template!r} reached {where}"
                )


# --------------------------------------------------------------------------- #
# Public statements of the T templates
# --------------------------------------------------------------------------- #


def _markdown_section(text: str, heading: str) -> str:
    rx = re.compile(
        r"\*\*" + re.escape(heading) + r":?\*\*(.*?)(?=\n[ \t]*\*\*|\Z)",
        flags=re.DOTALL | re.IGNORECASE,
    )
    match = rx.search(str(text or ""))
    return " ".join(match.group(1).split()) if match else ""


def template_statement(
    case_id: str, raw_task_description: str, *, max_chars: int = 320,
    case_name: str = "",
) -> dict[str, str]:
    """``{unit_id, title, output_sentence, protocol_sentence}`` of one
    template, from PUBLIC text only: the Task title and the Output section of
    the sanitized statement
    (:func:`queenbee.bench.silo_bench.sanitize_task_description`), and the
    lines of the raw Communication Protocol section that mention submitting.
    That section's ``Topology:`` lines (the leak for which the sanitizer
    drops the whole section) are never carried."""

    from queenbee.bench.silo_bench import sanitize_task_description

    assert_not_test([case_id], where="template_statement")
    statement = sanitize_task_description(str(raw_task_description or ""))
    title = _markdown_section(statement, "Task")
    if not title:
        head = re.search(r"\*\*Task:?\s*(.*?)\*\*", statement)
        title = head.group(1).strip() if head else ""
    title = title or str(case_name or case_id)
    protocol = ""
    match = re.search(
        r"\*\*Communication Protocol:?\*\*(.*?)(?=\n[ \t]*\*\*|\Z)",
        str(raw_task_description or ""),
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        lines = [
            line.strip() for line in match.group(1).splitlines()
            if line.strip()
            and not line.strip().lower().startswith("topology")
            and "submit" in line.lower()
        ]
        protocol = " ".join(lines)
    return {
        "unit_id": str(case_id),
        "title": title[:120],
        "output_sentence": _markdown_section(statement, "Output")[:max_chars],
        "protocol_sentence": protocol[:max_chars],
    }


def t_template_statements(
    case_ids: Iterable[str],
    *,
    benchmarks_dir: str | Path | None = None,
    n_agents: int = 5,
) -> list[dict[str, str]]:
    """Public statements of the given (T) templates, read from the
    Silo-Bench ``<case>_n<n>.json`` files in ``benchmarks_dir`` (default
    :func:`queenbee.paths.default_benchmarks_dir`).  TEST ids raise; pass T
    ids only."""

    ids = [str(c) for c in case_ids]
    assert_not_test(ids, where="t_template_statements")
    bench = Path(benchmarks_dir) if benchmarks_dir else default_benchmarks_dir()
    out = []
    for case_id in ids:
        data = json.loads(
            (bench / f"{case_id}_n{int(n_agents)}.json").read_text(encoding="utf-8")
        )
        out.append(template_statement(
            case_id, data.get("task_description", ""),
            case_name=data.get("case_name", ""),
        ))
    return out


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #

_HEADER = (
    "QUEENBEE-EVO PROGRAM DESIGN\n"
    "You improve ONE multi-agent team program for information-siloed tasks: "
    "n worker agents each hold a private data shard of one task and every "
    "agent must submit the correct answer. A frozen worker model executes "
    "the program; you write ONLY its GENOME (see the API CARD). The host "
    "runs your proposal on training units and compares it with a fresh run "
    "of its parent on the same units.\n"
)

_UNIT_LEGEND = (
    "Unit ids are <template>@<rung>: o5 = the upstream 5-agent instance, "
    "x5 = 5 agents with twice the data per agent, o10 = the upstream "
    "10-agent instance."
)


def _header() -> str:
    """The active task's header (default :data:`_HEADER`)."""

    header = get_task().header
    return _HEADER if header is None else header


def _unit_legend() -> str:
    legend = get_task().unit_legend
    return _UNIT_LEGEND if legend is None else legend


def _api_card(*, budgets: Any, preserve_dups: bool) -> str:
    """The API card text, rendered by the active task's renderer (default
    :func:`render_api_card`)."""

    render = get_task().render_api_card
    return (render_api_card if render is None else render)(budgets=budgets,
                                                            preserve_dups=preserve_dups)


def _diag_cards(unit_rows: Mapping[str, list], targets: list[str]) -> str:
    """The active task's diagnosis-cards block (default :func:`_block_diag_cards`)."""

    block = get_task().diag_cards_block
    return (_block_diag_cards if block is None else block)(unit_rows, targets)


_BRIEF_TASK = {
    "target": "Raise S on the target units without lowering S on the other "
    "training units.",
    "merge": "Build one program that keeps the units each merge parent wins "
    "(see MERGE PARENTS) without lowering S elsewhere.",
    "economize": "Lower the parent's calls and tokens without lowering S on "
    "any training unit.",
    "explore": "Choose one weakness of the parent on the training units and "
    "change the program to address it without lowering S elsewhere.",
}

#: ``--explore-on-lockin``: the extra BRIEF line of the reserved explore
#: slot -- what the dominant mechanism family did, and what this proposal
#: must do instead (families as in :func:`queenbee.evo.memory.mechanism_family`).
_LOCKIN_DID = {
    "instructions": "only rewrote work instructions on an unchanged communication pattern",
    "topology": "changed the communication topology (PHASES kinds, their rounds or order, "
    "or who sends to whom)",
    "code": "only changed helper code, leaving the communication pattern and the work "
    "instructions as they were",
}
_LOCKIN_ASK = {
    "instructions": "change the communication topology itself (which PHASES kinds run, in "
    "which order and for how many rounds, and who sends to whom), not only the work "
    "instructions",
    "topology": "keep the parent's communication topology and change what the workers are "
    "asked to do (the work instructions) instead",
    "code": "change the communication topology or the work instructions instead of helper code",
}

_HYPOTHESIS_CLASSES = (
    "divergent", "consensus-wrong", "scattered-wrong", "local-only",
    "shape-mismatch", "format", "content-loss", "budget", "unknown",
)

OUTPUT_CONTRACT = (
    "=== OUTPUT CONTRACT ===\n"
    "Make the SMALLEST change you expect to be effective: keep every part of "
    "the parent genome that you do not deliberately change byte-identical "
    "(the host logs the size of your edit). A proposal whose communication "
    "pattern and work instructions equal those of an already-run program is "
    "not executed.\n"
    "Reply format (strict): first ONE line\n"
    'HYPOTHESIS: {"target_units": ["<unit id>", ...], "failure_class": '
    '"<class>", "mechanism": "<what you change and why it should raise S>", '
    '"predicted_dS": {"<unit id>": <number>, ...}}\n'
    "where failure_class is one of " + ", ".join(_HYPOTHESIS_CLASSES)
    + ", and predicted_dS maps each target unit to your predicted change in "
    "mean S against a fresh run of the parent (a number between -1 and 1).\n"
    "Then ONE ```python code block holding ONLY the complete new genome "
    "(from `PHASES = [` up to, not including, `def main`). Do not reproduce "
    "the host prefix or main(); no other code blocks.\n"
)


def _block_templates(state: Any, forbidden: set[str]) -> str:
    rows = []
    items = list(_get(state, "t_templates", []) or [])
    _check_ids([_get(i, "unit_id", "") for i in items], forbidden, "t_templates")
    for item in items:
        unit = _unit(_get(item, "unit_id", "")) or "?"
        block = f"[{unit}] {_clip(_get(item, 'title', ''), 120)}".rstrip()
        output = _clip(_get(item, "output_sentence", ""), 400)
        protocol = _clip(_get(item, "protocol_sentence", ""), 400)
        if output:
            block += f"\n  Output: {output}"
        if protocol:
            block += f"\n  Protocol: {protocol}"
        rows.append(block)
    if not rows:
        raise ValueError("state.t_templates is empty (all T templates are required)")
    return (
        "=== TASK TEMPLATES (every training template; public text only) ===\n"
        "The program is run on units drawn from ALL of these templates, so it "
        "must be task-generic.\n" + "\n".join(rows) + "\n"
    )


def _block_parent_genome(source: str) -> str:
    return (
        "=== PARENT GENOME (the program you modify) ===\n"
        f"```python\n{_genome_text(source)}\n```\n"
    )


def _fmt_tokens(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1000:.1f}k"


def _fmt_calls(value: float | None) -> str:
    """Mean calls as a whole number: a fractional mean (``53.6``) could
    coincide with an audited ground-truth float, a count below 1000 cannot
    (forms shorter than 4 characters are not audited)."""

    if value is None:
        return "n/a"
    return f"{float(value):.0f}"


def _block_parent_record(unit_rows: Mapping[str, list[Mapping[str, Any]]]) -> str:
    lines = [
        "=== PARENT RECORD (training units; mean over the parent's runs) ===",
        _unit_legend(),
    ]
    all_s: list[float] = []
    all_calls: list[float] = []
    all_tokens: list[float] = []
    for unit in sorted(unit_rows):
        summary = unit_summary(unit_rows[unit])
        if summary["S"] is None:
            lines.append(f"- {unit}: no scored run")
            continue
        all_s.append(summary["S"])
        if summary["calls"] is not None:
            all_calls.append(summary["calls"])
        if summary["tokens"] is not None:
            all_tokens.append(summary["tokens"])
        calls = _fmt_calls(summary["calls"])
        lines.append(
            f"- {unit}: S={summary['S']:.2f} ({summary['n']} run"
            f"{'s' if summary['n'] != 1 else ''}) calls={calls} "
            f"tokens={_fmt_tokens(summary['tokens'])}"
        )
    if not all_s:
        lines.append("- the parent has no scored training run yet")
    else:
        calls = _mean(all_calls)
        lines.append(
            f"Mean over units: S={sum(all_s) / len(all_s):.2f} calls="
            + _fmt_calls(calls)
            + f" tokens={_fmt_tokens(_mean(all_tokens))}"
        )
    return "\n".join(lines) + "\n"


def _block_failure_map(state: Any, unit_rows: Mapping[str, list]) -> str:
    from queenbee.evo.diagnosis import build_failure_map, render_failure_map

    fmap = _get(state, "fmap")
    if not isinstance(fmap, Mapping):
        fmap = build_failure_map(unit_rows)
    text = render_failure_map(fmap, max_chars=3000)
    text = text.replace("current incumbent", "parent").replace("incumbent", "parent")
    # the map covers the units in play only; the PARENT RECORD lists the rest
    return text.replace("Solved by the parent on every run:",
                        "Of the units in this map, solved by the parent on every run:")


def _optin_card_legend() -> str:
    """Legend of the opt-in card fields (``--local-only-diag``,
    ``--attempt-diag``); empty when both are off, so the block text is
    unchanged unless one of them is enabled."""

    from queenbee.evo.diagnosis import attempt_diag_enabled, local_only_enabled

    parts = []
    if local_only_enabled():
        parts.append("own_shard_only_agents = wrong agents whose answer is exactly the "
                     "answer of their OWN shard alone.")
    if attempt_diag_enabled():
        parts.append("answer_seen_agents = wrong agents whose final snapshot (the "
                     "messages delivered to them at the submit round, or their own last "
                     "message) already contained the correct answer value: the right "
                     "result reached them and they submitted something else.")
    return (" " + " ".join(parts)) if parts else ""


def _block_diag_cards(unit_rows: Mapping[str, list], targets: list[str]) -> str:
    cards = render_diag_cards_with_credit(unit_rows, max_chars=8000, order=targets)
    return (
        "=== DIAGNOSIS CARDS (the parent's runs on training units; "
        "answer-free) ===\n"
        "One line per run: S, failure class, agents_right = per-agent "
        "correctness bits (agent 0 first), distinct_answers, plurality "
        "right/wrong, submit round, msgs, lost_bodies = bodies delivered to "
        "an agent that was not called that round, tokens/phase = completion "
        "tokens per PHASES entry | submit. " + credit_legend() + _optin_card_legend() + "\n"
        + (cards if cards else "no diagnosed runs\n")
    )


def _block_target_diagnosis(
    state: Any, unit_rows: Mapping[str, list], brief: Mapping[str, Any]
) -> str:
    from queenbee.evo.diagnosis import (
        build_failure_map,
        failure_class_glossary,
    )

    fmap = _get(state, "fmap")
    if not isinstance(fmap, Mapping):
        fmap = build_failure_map(unit_rows)
    units = fmap.get("units") or {}
    lines = ["=== TARGET DIAGNOSIS (host; answer-free) ==="]
    target_class = _fclass(brief.get("target_class"))
    if target_class:
        gloss = failure_class_glossary().get(target_class)
        lines.append(
            f"Failure class assigned to this brief: {target_class}"
            + (f" - {gloss}" if gloss else "")
        )
    for unit in brief["target_units"]:
        entry = units.get(unit) or {}
        dominant = _fclass(entry.get("dominant")) or "unknown"
        lines.append(f"- {unit}: dominant class over the parent's runs = {dominant}")
    if len(lines) == 1:
        lines.append("- no target units in this brief")
    return "\n".join(lines) + "\n"


def _ledger_key(entry: Mapping[str, Any]) -> tuple[str, str]:
    fclass = _fclass(_get(entry, "target_class")) or "unknown"
    tags = [
        normalize_mechanism(t)[:30] for t in (_get(entry, "mech_tags_host", []) or [])
        if normalize_mechanism(t)
    ]
    if tags:
        mech = "+".join(sorted(dict.fromkeys(tags)))
    else:
        mech = normalize_mechanism(_get(entry, "mech_claim_norm", ""))[:60] or "unlabeled"
    return fclass, mech[:80]


def _mean_observed(entry: Mapping[str, Any]) -> float | None:
    observed = _get(entry, "observed", {}) or {}
    if not isinstance(observed, Mapping):
        return None
    values = [v for v in (_num(x) for x in observed.values()) if v is not None]
    return _mean(values)


def _tabu_label(item: Any) -> str | None:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return f"{_fclass(item[0]) or 'unknown'} x {normalize_mechanism(item[1])[:80]}"
    if isinstance(item, Mapping):
        return (
            f"{_fclass(item.get('failure_class') or item.get('class')) or 'unknown'}"
            f" x {normalize_mechanism(item.get('mechanism'))[:80]}"
        )
    text = normalize_mechanism(item)[:100]
    return text or None


def _after_lines(entries: Sequence[Any], forbidden: set[str]) -> list[str]:
    """Lines for the executed, not-confirmed entries that carry the child's
    post-change card summary (``after_diag``, recorded with
    ``--attempt-diag``); newest first, at most 5.  A ``not economized`` entry
    (``--econ-neutral``: S held, no token saving) is not an unsuccessful
    attempt and is left out (its dS still counts in the table)."""

    out = []
    for index, entry in enumerate(entries, start=1):
        after = _get(entry, "after_diag", None)
        verdict = str(_get(entry, "verdict", "") or "")
        if not isinstance(after, Mapping) or verdict in ("confirmed", "screened"):
            continue
        if _get(entry, "econ_result", None) == "not_economized":
            continue
        units = [_unit(u) for u in (_get(entry, "target_units", []) or [])]
        _check_ids(units, forbidden, "ledger")
        hyp = _get(entry, "hypothesis", {}) or {}
        statement = _get(hyp, "mechanism", "") or _get(entry, "mech_claim_norm", "")
        observed = _mean_observed(entry)
        n = after.get("n_agents")
        def frac(key: str) -> str:
            value = after.get(key)
            return f"{int(value)}/{int(n)}" if isinstance(value, int) and isinstance(n, int) else "n/a"
        bits = [f"class={_fclass(after.get('class')) or 'unknown'}"]
        if after.get("own_shard_only") is not None:
            bits.append(f"own_shard_only_agents={frac('own_shard_only')}")
        if after.get("answer_seen") is not None:
            bits.append(f"answer_seen_agents={frac('answer_seen')}")
        out.append(
            f"- L{index} (gen {_get(entry, 'gen', '?')}; targets {', '.join(u for u in units if u)}; "
            f"{verdict}): {_clip(statement, 220) or 'unlabeled'}"
            + (f"; observed mean dS={_signed(observed)}" if observed is not None else "")
            + "; after the change: " + ", ".join(bits)
        )
    return list(reversed(out))[:5]


def _block_ledger(state: Any, forbidden: set[str]) -> str | None:
    entries = [e for e in (_get(state, "ledger", []) or []) if e is not None]
    tabu_items = list(_get(state, "tabu", []) or [])
    if not entries and not tabu_items:
        return None  # nothing recorded yet (e.g. generation 0): no block, as with the ledger off
    lines = [
        "=== LEDGER (host-verified outcomes of earlier proposals; answer-free) ===",
        "Verdicts are computed by the host from executed runs: confirmed = the "
        "target units improved against a fresh parent run with no guard loss; "
        "refuted = no improvement; inconclusive = in between; screened = "
        "rejected before execution. Entry ids L<k> may be cited in your "
        "HYPOTHESIS.",
    ]
    if any(_get(e, "econ_result", None) == "not_economized" for e in entries):
        # only with --econ-neutral; otherwise the line is absent
        lines.append("not economized = an economize proposal that kept S on its "
                     "target units but saved no tokens: a cost result, not evidence "
                     "against its mechanism.")
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    confirmed: list[str] = []
    for index, entry in enumerate(entries, start=1):
        units = [_unit(u) for u in (_get(entry, "target_units", []) or [])]
        _check_ids(units, forbidden, "ledger")
        verdict = str(_get(entry, "verdict", "") or "")
        if verdict not in ("confirmed", "refuted", "inconclusive", "screened"):
            verdict = "inconclusive"
        if verdict == "inconclusive" and _get(entry, "econ_result", None) == "not_economized":
            verdict = "not economized"
        key = _ledger_key(entry)
        cell = cells.setdefault(key, {"confirmed": 0, "refuted": 0,
                                      "inconclusive": 0, "not economized": 0,
                                      "screened": 0, "dS": []})
        cell[verdict] += 1
        observed = _mean_observed(entry)
        if observed is not None and verdict != "screened":
            cell["dS"].append(observed)
        if verdict == "confirmed":
            hyp = _get(entry, "hypothesis", {}) or {}
            statement = _get(hyp, "mechanism", "") or _get(entry, "mechanism", "")
            gen = _get(entry, "gen", "?")
            confirmed.append(
                f"- L{index} (gen {gen}; targets {', '.join(u for u in units if u)}; "
                f"class {key[0]}): {_clip(statement, 300) or key[1]}"
                + (f"; observed mean dS={_signed(observed)}" if observed is not None else "")
            )
    if cells:
        lines.append("(failure class x mechanism) -> verdict counts, mean observed dS:")
        for (fclass, mech), cell in sorted(cells.items()):
            counts = ", ".join(
                f"{cell[v]} {v}" for v in
                ("confirmed", "refuted", "inconclusive", "not economized", "screened")
                if cell[v]
            )
            mean = _mean(cell["dS"])
            lines.append(
                f"- {fclass} x {mech}: {counts}"
                + (f"; mean dS={_signed(mean)} over {len(cell['dS'])} executed"
                   if mean is not None else "")
            )
    else:
        lines.append("- no ledger entries yet")
    lines.append("Confirmed hypotheses (verbatim; newest first; at most 5):")
    lines.extend(list(reversed(confirmed))[:5] or ["- none yet"])
    after = _after_lines(entries, forbidden)
    if after:
        lines.append("Unsuccessful attempts and the failure LEFT AFTER the change (the "
                     "child's own runs on its target units; newest first; at most 5):")
        lines.extend(after)
    tabu = [t for t in (_tabu_label(x) for x in tabu_items) if t]
    lines.append(
        "TABU (refuted twice; a proposal repeating one is rejected before "
        "execution):"
    )
    lines.extend([f"- {t}" for t in sorted(dict.fromkeys(tabu))] or ["- none"])
    return "\n".join(lines) + "\n"


def _observed_text(observed: Any) -> str:
    if not isinstance(observed, Mapping):
        return ""
    bits = []
    for unit in sorted(observed):
        value = _num(observed[unit])
        if value is not None and _unit(unit):
            bits.append(f"{_unit(unit)} {_signed(value)}")
    return ", ".join(bits)


def _block_exemplars(
    state: Any, brief: Mapping[str, Any], forbidden: set[str]
) -> str | None:
    exemplars = [e for e in (brief.get("exemplars") or []) if isinstance(e, Mapping)]
    if not exemplars:
        return None
    lines = [
        "=== CONTRASTIVE EXEMPLARS (from the archive; answer-free) ===",
        "Programs already executed on training units: one that improved the "
        "failure class of this brief and one that did not.",
    ]
    for index, item in enumerate(exemplars[:2]):
        pid = _clip(item.get("program_id"), 40)
        role = "improved" if str(item.get("role")) == "fixed" else "did not improve"
        observed = item.get("observed")
        if isinstance(observed, Mapping):
            _check_ids(list(observed), forbidden, "exemplars")
        head = (
            f"-- exemplar {chr(65 + index)} (program {pid}; {role} class "
            f"{_fclass(item.get('failure_class')) or 'unknown'}"
        )
        obs = _observed_text(observed)
        head += (f"; observed dS: {obs}" if obs else "") + ") --"
        lines.append(head)
        lines.append(
            f"```python\n{_genome_text(_program_source(state, item.get('program_id')))}\n```"
        )
    return "\n".join(lines) + "\n"


def _block_merge(
    state: Any, brief: Mapping[str, Any], forbidden: set[str]
) -> str | None:
    parents = list(brief.get("parent_ids") or [])
    if brief.get("kind") != "merge" or len(parents) < 2:
        return None
    wins = brief.get("wins") or {}

    def _wins(pid: Any) -> str:
        units = [_unit(u) for u in (wins.get(pid) or [])] if isinstance(wins, Mapping) else []
        _check_ids(units, forbidden, "merge wins")
        return ", ".join(u for u in units if u) or "n/a"

    lines = [
        "=== MERGE PARENTS (two specialists from the archive) ===",
        f"Merge parent 1 = the PARENT GENOME above; units it wins: {_wins(parents[0])}",
        f"Merge parent 2 (program {_clip(parents[1], 40)}); units it wins: "
        f"{_wins(parents[1])}",
        f"```python\n{_genome_text(_program_source(state, parents[1]))}\n```",
    ]
    return "\n".join(lines) + "\n"


def _block_brief(brief: Mapping[str, Any], unit_rows: Mapping[str, list]) -> str:
    kind = brief["kind"]
    lines = [
        "=== BRIEF (host-assigned for this proposal) ===",
        f"Kind: {kind}",
        f"Task: {_BRIEF_TASK[kind]}",
    ]
    lock = brief.get("lockin")
    if isinstance(lock, Mapping) and str(lock.get("family")) in _LOCKIN_ASK:
        family = str(lock["family"])
        lines.append(
            f"Exploration directive (host): {int(lock.get('n') or 0)} of the "
            f"{int(lock.get('of') or 0)} confirmed improvements so far {_LOCKIN_DID[family]}. "
            f"This proposal must use a DIFFERENT mechanism: {_LOCKIN_ASK[family]}."
        )
    targets = []
    for unit in brief["target_units"]:
        summary = unit_summary(unit_rows.get(unit) or [])
        score = "n/a" if summary["S"] is None else f"{summary['S']:.2f}"
        targets.append(f"{unit} (parent S={score})")
    lines.append(
        "Target units: " + (", ".join(targets) if targets else "none (all training units)")
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _coerce_brief(brief: Any, flags: EvoFlags) -> dict[str, Any]:
    kind = str(_get(brief, "kind", "explore") or "explore").strip().lower()
    if kind not in BRIEF_KINDS:
        raise ValueError(f"unknown brief kind {kind!r}; choose {BRIEF_KINDS}")
    if kind == "merge" and not flags.archive:
        kind = "explore"  # without the archive a merge slot becomes explore
    targets = []
    for unit in _get(brief, "target_units", []) or []:
        clean = _unit(unit)
        if clean and clean not in targets:
            targets.append(clean)
    parents = [p for p in (_get(brief, "parent_ids", []) or []) if p is not None]
    if not parents:
        raise ValueError("brief.parent_ids is empty (the parent comes first)")
    out = {
        "kind": kind,
        "target_units": targets,
        "target_class": _get(brief, "target_class"),
        "parent_ids": parents,
        "exemplars": list(_get(brief, "exemplars", []) or []),
        "wins": _get(brief, "wins", {}) or {},
    }
    if isinstance(_get(brief, "lockin"), Mapping):  # --explore-on-lockin only
        out["lockin"] = dict(_get(brief, "lockin"))
    return out


def arm_brief(brief: Any, flags: Any) -> dict[str, Any]:
    """The brief an arm receives, derived from the ``full`` arm's brief of
    the same slot: the SAME kind, target units and parents in every arm,
    minus what a switched-off component would supply.  Without ``diag`` the
    failure class is dropped (the targets stay); without ``ledger`` the
    lock-in directive is dropped; without ``archive`` the exemplars / merge
    wins are dropped and a merge slot becomes explore with its first parent
    only.  ``brief_id`` is kept (bookkeeping, never rendered).

    Use it for every arm other than ``full``: the arms' prompts for one slot
    then differ only in the gated blocks and, inside the BRIEF, in the
    lock-in line and a merge slot's kind."""

    flags = coerce_flags(flags)
    out: dict[str, Any] = {
        "kind": str(_get(brief, "kind", "explore") or "explore").strip().lower(),
        "target_units": list(_get(brief, "target_units", []) or []),
        "target_class": _get(brief, "target_class") if flags.diag else None,
        "parent_ids": list(_get(brief, "parent_ids", []) or []),
    }
    brief_id = _get(brief, "brief_id")
    if brief_id is not None:
        out["brief_id"] = brief_id
    if isinstance(_get(brief, "lockin"), Mapping) and flags.ledger:
        # --explore-on-lockin: a ledger-derived directive, so only with the ledger
        out["lockin"] = dict(_get(brief, "lockin"))
    if flags.archive:
        out["exemplars"] = list(_get(brief, "exemplars", []) or [])
        out["wins"] = dict(_get(brief, "wins", {}) or {})
    elif out["kind"] == "merge":
        out["kind"] = "explore"
        out["parent_ids"] = out["parent_ids"][:1]
    return out


def build_evo_prompt_blocks(
    state: Any,
    brief: Any,
    flags: Any = None,
    *,
    forbidden_case_ids: Any = (),
) -> list[tuple[str, str]]:
    """Ordered ``(name, text)`` blocks of the prompt; gated blocks appear only
    when their flag is on (and they have content)."""

    flags = coerce_flags(flags)
    forbidden = set(_forbidden_ids(state, forbidden_case_ids))
    b = _coerce_brief(brief, flags)
    _check_ids(b["target_units"], forbidden, "brief.target_units")
    parent_id = b["parent_ids"][0]
    parent_source = _program_source(state, parent_id)
    unit_rows = _unit_rows(state, parent_id)
    _check_ids(list(unit_rows), forbidden, "parent rows")

    blocks: list[tuple[str, str]] = [
        ("header", _header()),
        ("api_card", _api_card(budgets=_get(state, "budgets"),
                               preserve_dups=bool(_get(state, "preserve_dups", False)))),
        ("templates", _block_templates(state, forbidden)),
        ("parent_genome", _block_parent_genome(parent_source)),
        ("parent_record", _block_parent_record(unit_rows)),
    ]
    if flags.diag:
        blocks.append(("failure_map", _block_failure_map(state, unit_rows)))
        blocks.append(("diag_cards", _diag_cards(unit_rows, b["target_units"])))
    if flags.ledger:
        ledger = _block_ledger(state, forbidden)
        if ledger:
            blocks.append(("ledger", ledger))
    if flags.archive:
        exemplars = _block_exemplars(state, b, forbidden)
        if exemplars:
            blocks.append(("exemplars", exemplars))
        merge = _block_merge(state, b, forbidden)
        if merge:
            blocks.append(("merge_parents", merge))
    blocks.append(("brief", _block_brief(b, unit_rows)))
    if flags.diag:
        blocks.append(("target_diagnosis", _block_target_diagnosis(state, unit_rows, b)))
    blocks.append(("output_contract", OUTPUT_CONTRACT))
    return blocks


def audit_evo_prompt(
    text: str,
    *,
    forbidden_case_ids: Any = (),
    forbidden_values: Any = None,
    min_value_chars: int = 4,
) -> list[dict[str, Any]]:
    """Leak hits of a finished prompt: TEST + ``forbidden_case_ids`` anywhere
    in the text; ``forbidden_values`` only inside the evidence sections
    (program text elsewhere cannot trigger false value hits)."""

    from queenbee.program.mint import audit_text_leaks

    ids = sorted(set(sealed_case_ids()) | {str(c) for c in forbidden_case_ids or []})
    hits = audit_text_leaks(text, ids)
    if forbidden_values:
        hits += audit_text_leaks(
            text, (), forbidden_values=forbidden_values,
            sections=EVO_EVIDENCE_SECTION_HEADERS, min_value_chars=min_value_chars,
        )
    return hits


def build_evo_prompt(
    state: Any,
    brief: Any,
    flags: Any = None,
    *,
    forbidden_case_ids: Any = (),
    forbidden_values: Any = None,
) -> str:
    """The QueenBee-Evo mint prompt (see the module docstring).

    Raises ``RuntimeError`` (LEAKAGE_GUARD) when a TEST / held-out template
    or a forbidden value would reach the prompt, ``ValueError`` on a
    malformed state or brief."""

    blocks = build_evo_prompt_blocks(
        state, brief, flags, forbidden_case_ids=forbidden_case_ids
    )
    text = "\n".join(body.rstrip("\n") + "\n" for _name, body in blocks)
    hits = audit_evo_prompt(
        text,
        forbidden_case_ids=_forbidden_ids(state, forbidden_case_ids),
        forbidden_values=forbidden_values,
    )
    if hits:
        raise RuntimeError(f"LEAKAGE_GUARD: evo prompt leak audit hits {hits[:5]}")
    return text


# --------------------------------------------------------------------------- #
# Reply parsing (closed keys; keeps predicted_dS and '@' in unit ids)
# --------------------------------------------------------------------------- #

_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS\**\s*[:=]\s*", re.I)
_PRE_BRACE = re.compile(r"[\s*`]*(?:json)?[\s*`]*", re.I)


def _balanced(text: str, start: int) -> str | None:
    depth = 0
    quote: str | None = None
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _clip_dS(value: Any) -> float | None:
    """A predicted dS as a float in [-1, 1]; ``"30%"`` / ``"+30 %"`` mean
    0.3 (a percentage string is scaled, a bare number is taken as is)."""

    number = _num(value)
    if number is None and isinstance(value, str):
        text = value.strip()
        percent = text.endswith("%")
        try:
            number = float(text.rstrip("%").strip())
        except ValueError:
            return None
        if percent:
            number /= 100.0
    if number is None or not math.isfinite(number):
        return None
    return max(-1.0, min(1.0, number))


def parse_evo_hypothesis(text: Any) -> dict[str, Any] | None:
    """The reply's ``HYPOTHESIS: {...}`` as a closed-key dict:
    ``target_units`` (ids with ``@`` kept, at most 12), ``failure_class``,
    ``mechanism``, ``predicted_dS`` ({unit: float in [-1, 1]} or one float;
    ``predicted_ds`` is read too), and ``predicted_effect`` (free text, when
    the reply carries that key).  ``{"raw": ...}`` when unparseable; None
    without a HYPOTHESIS header."""

    if not isinstance(text, str):
        return None
    match = _HYPOTHESIS_RE.search(text)
    if not match:
        return None
    rest = text[match.end():]
    brace = rest.find("{")
    newline = rest.find("\n")
    # The dict may open on the header line after any decoration
    # (``**HYPOTHESIS:** {``, ``HYPOTHESIS: `{``), or on a later line after
    # only whitespace / markdown / a json fence (the mint's parse_hypothesis
    # accepts only whitespace there).
    if brace == -1 or not (
        newline == -1 or brace <= newline or _PRE_BRACE.fullmatch(rest[:brace])
    ):
        return {"raw": _clip(rest.split("\n", 1)[0], 300)}
    blob = _balanced(rest, brace)
    value: Any = None
    if blob:
        for loader in (json.loads, ast.literal_eval):
            try:
                value = loader(blob)
                break
            except Exception:  # noqa: BLE001 - try the next loader
                continue
    if not isinstance(value, dict):
        return {"raw": _clip(blob or rest.split("\n", 1)[0], 300)}
    out: dict[str, Any] = {}
    units = value.get("target_units")
    if isinstance(units, str):
        units = [u for u in re.split(r"[,\s]+", units) if u]
    if isinstance(units, (list, tuple)):
        out["target_units"] = [u for u in (_unit(x) for x in units[:12]) if u]
    fclass = _fclass(value.get("failure_class"))
    if fclass:
        out["failure_class"] = fclass
    mechanism = _clip(value.get("mechanism"), 300)
    if mechanism:
        out["mechanism"] = mechanism
    predicted = value.get("predicted_dS", value.get("predicted_ds"))
    if isinstance(predicted, Mapping):
        clean = {}
        for unit, delta in list(predicted.items())[:12]:
            number = _clip_dS(delta)
            if _unit(unit) and number is not None:
                clean[_unit(unit)] = number
        if clean:
            out["predicted_dS"] = clean
    else:
        number = _clip_dS(predicted)
        if number is not None:
            out["predicted_dS"] = number
    effect = _clip(value.get("predicted_effect"), 300)
    if effect:
        out["predicted_effect"] = effect
    return out or {"raw": _clip(blob, 300)}


def read_mint_hypothesis(workdir: str | Path) -> dict[str, Any] | None:
    """:func:`parse_evo_hypothesis` of the newest mint reply in a
    :func:`queenbee.program.mint.mint_python_challenger` workdir
    (``mint_reply_01.txt`` - the re-mint - before ``mint_reply_00.txt``)
    that carries a HYPOTHESIS header.

    Use it instead of the mint's ``hypothesis_out``, whose sanitizer
    (:func:`queenbee.program.mint.sanitize_hypothesis`) drops
    ``predicted_dS``."""

    for name in ("mint_reply_01.txt", "mint_reply_00.txt"):
        path = Path(workdir) / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        parsed = parse_evo_hypothesis(text)
        if parsed is not None:
            return parsed
    return None


__all__ = [
    "ARM_FLAGS",
    "BRIEF_KINDS",
    "EVO_EVIDENCE_SECTION_HEADERS",
    "EvoFlags",
    "GATED_BLOCKS",
    "OUTPUT_CONTRACT",
    "TEST_CASE_IDS",
    "arm_brief",
    "audit_evo_prompt",
    "build_evo_prompt",
    "build_evo_prompt_blocks",
    "coerce_flags",
    "normalize_mechanism",
    "parse_evo_hypothesis",
    "read_mint_hypothesis",
    "t_template_statements",
    "template_statement",
    "unit_summary",
]
