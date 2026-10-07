"""Genome region of a team program: locate, splice and check it.

The host owns the program prefix (imports, the two worker instruction
strings, ``reject_nonfinite``) and ``main()``; the planner writes only the
genome, the region between them.  These helpers find that region, splice a
planner-written genome into an incumbent program and check a genome before
splicing.
"""

from __future__ import annotations

import ast as _ast
import re

#: Names a genome may never define or re-bind, in addition to the host-owned
#: top-level names of the incumbent's prefix (:func:`_reserved_names`).
_GENOME_RESERVED = frozenset(
    {
        "main",
        "reject_nonfinite",
        "MESSAGE_INSTRUCTION",
        "SUBMIT_INSTRUCTION",
        "json",
        "sys",
        "create_llm_client",
    }
)
#: The two policy functions ``main()`` calls; every genome must define them.
_GENOME_REQUIRED = ("plan_submit_round", "plan_communication_turn")
_MAIN_LINE = re.compile(r"^def main\s*\(", re.M)
_PHASES_LINE = re.compile(r"^PHASES = \[", re.M)
_SUBMIT_DEF_LINE = re.compile(r"^def plan_submit_round\b", re.M)
_CODE_LINE = re.compile(r"^[ \t]*\S", re.M)


def genome_bounds(source: str) -> tuple[int, int] | None:
    """(start, end) character offsets of the genome region of ``source``.

    end = the ``def main(`` line.  start = the first non-blank line after
    the host prefix (which ends with ``def reject_nonfinite``); when that
    function is missing or the source does not parse, the ``PHASES = [``
    line, else the ``def plan_submit_round`` line.  None when no region
    exists.
    """

    if not isinstance(source, str):
        return None
    main = _MAIN_LINE.search(source)
    if not main:
        return None
    end = main.start()
    start: int | None = None
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        prefix_end = None
        for node in tree.body:
            if isinstance(node, _ast.FunctionDef) and node.name == "reject_nonfinite":
                prefix_end = getattr(node, "end_lineno", None)
                break
        if prefix_end:
            lines = source.splitlines(keepends=True)
            offset = sum(len(line) for line in lines[:prefix_end])
            found = _CODE_LINE.search(source, offset)
            if found and found.start() < end:
                start = source.rfind("\n", 0, found.start()) + 1
    if start is None:
        for pattern in (_PHASES_LINE, _SUBMIT_DEF_LINE):
            found = pattern.search(source, 0, end)
            if found:
                start = found.start()
                break
    if start is None or start >= end:
        return None
    return start, end


def genome_region(source: str) -> str | None:
    """The genome text of a full program (None when it has no region)."""

    bounds = genome_bounds(source)
    if bounds is None:
        return None
    return source[bounds[0]:bounds[1]]


def splice_genome(incumbent_source: str, genome: str) -> str:
    """incumbent prefix + ``genome`` + incumbent ``def main`` onward.

    Splicing the incumbent's own genome back reproduces the incumbent byte
    for byte (the region is normalized to end with two blank lines before
    ``def main``, the scaffold's own layout).
    """

    bounds = genome_bounds(incumbent_source)
    if bounds is None:
        raise ValueError("incumbent source has no genome region")
    start, end = bounds
    body = str(genome or "").strip("\n")
    if not body.strip():
        raise ValueError("empty genome")
    return incumbent_source[:start] + body + "\n\n\n" + incumbent_source[end:]


def _reserved_names(incumbent_source: str | None) -> set[str]:
    """Host-owned names: :data:`_GENOME_RESERVED` plus every top-level
    function, class, plain-assignment target and import of the incumbent's
    prefix."""
    names = set(_GENOME_RESERVED)
    bounds = genome_bounds(incumbent_source) if incumbent_source else None
    if bounds is None:
        return names
    try:
        tree = _ast.parse(incumbent_source[: bounds[0]])
    except SyntaxError:
        return names
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, _ast.Assign):
            for target in node.targets:
                if isinstance(target, _ast.Name):
                    names.add(target.id)
        elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def check_genome(genome: str, *, incumbent_source: str | None = None) -> list[str]:
    """Structural problems of a genome BEFORE splicing (empty = fine).

    Exact messages with genome line numbers, so a re-mint can fix precisely
    what failed: an empty genome, a syntax error, a top-level import, a
    top-level expression statement (a bare string is allowed), a top-level
    definition or assignment of a host-owned name, or a missing
    ``plan_submit_round`` / ``plan_communication_turn``.
    """

    text = str(genome or "").strip("\n")
    if not text.strip():
        return ["the genome is empty"]
    lines = text.splitlines()
    try:
        tree = _ast.parse(text)
    except SyntaxError as exc:
        where = exc.lineno or 0
        snippet = lines[where - 1].strip() if 0 < where <= len(lines) else ""
        return [
            f"SyntaxError at genome line {where}: {exc.msg}"
            + (f" -> `{snippet[:160]}`" if snippet else "")
        ]
    reserved = _reserved_names(incumbent_source)
    problems: list[str] = []
    defined: set[str] = set()
    for node in tree.body:
        line = getattr(node, "lineno", 0)
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            problems.append(f"genome line {line}: imports are host-owned; remove it")
            continue
        if isinstance(node, _ast.Expr) and not (
            isinstance(node.value, _ast.Constant)
            and isinstance(node.value.value, str)
        ):
            problems.append(
                f"genome line {line}: top-level expression statements are not "
                "allowed in the genome (main() is called by the host)"
            )
            continue
        names: list[str] = []
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (_ast.Assign, _ast.AnnAssign, _ast.AugAssign)):
            targets = (
                node.targets if isinstance(node, _ast.Assign) else [node.target]
            )
            for target in targets:
                for sub in _ast.walk(target):
                    if isinstance(sub, _ast.Name):
                        names.append(sub.id)
        for name in names:
            if name in reserved:
                problems.append(
                    f"genome line {line}: {name!r} is host-owned (prefix or "
                    "main); the genome may not define or rebind it"
                )
            defined.add(name)
    for required in _GENOME_REQUIRED:
        if required not in defined:
            problems.append(
                f"the genome must define {required}() (main() calls it)"
            )
    return problems
