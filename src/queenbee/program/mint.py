"""Planner side of QueenBee-Evo: mint a challenger program from one planner reply.

The planner answers with a ``HYPOTHESIS`` header and a genome block; the
host parses the reply, splices the genome into the incumbent (parent)
program, validates and dry-runs the result without any worker call, and
allows one re-mint with the exact diagnostic.  Also here: planner-call
metering, the per-call output cap and deadline, prompt leak audits and
source-diff summaries.
"""

from __future__ import annotations

import ast as _ast
import difflib
import json
import re
import threading
from pathlib import Path
from typing import Any

from exp_graph.mas.python_code import python_source_sha256, validate_python_source
from exp_graph.mas.python_code_generation import _dry_run_canaries, _dry_run_payload
from exp_graph.mas.python_code_runner import (
    CodeProcessRunner,
    PythonExecutionLimits,
)
from exp_graph.mas.schemas import MASRuntimeConfig, PlannerRequest

from queenbee.bench.engine import _resolved_python_execution_timeout
from queenbee.evo.diagnosis import clean_unit_id
from queenbee.program.budgets import PythonRunBudgets
from queenbee.program.execute import _is_transport_error, _run_config, task_worker_env
from queenbee.program.genome import (
    _MAIN_LINE,
    check_genome,
    genome_bounds,
    genome_region,
    splice_genome,
)

#: Planner output cap per call (tokens) when neither the caller nor
#: ``QB_PLANNER_MAX_COMPLETION_TOKENS`` sets one.
DEFAULT_PLANNER_MAX_COMPLETION_TOKENS = 96_000
#: Planner wall clock per call (seconds) when neither the caller nor
#: ``QB_PLANNER_DEADLINE_S`` sets one.
DEFAULT_PLANNER_DEADLINE_S = 3600.0


class MintFailed(RuntimeError):
    """A genome-mode mint produced no valid challenger.

    ``reason`` is a short answer-free diagnostic (validator / dry-run text
    about the planner's own program, or the planner deadline); ``hypothesis``
    the last HYPOTHESIS header parsed (or None); ``kind`` is ``"invalid"``
    (the mint and its one re-mint were both rejected: no genome extracted,
    genome check, validator or dry run) or ``"planner_deadline"`` (a planner
    call hit its wall-clock deadline; no re-mint follows); ``attempts`` holds
    one record per planner call (the ``attempts`` list of ``mint_diag.json``).
    """

    def __init__(
        self,
        reason: str,
        hypothesis: dict[str, Any] | None = None,
        *,
        kind: str = "invalid",
        attempts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.hypothesis = dict(hypothesis) if isinstance(hypothesis, dict) else None
        self.kind = str(kind)
        self.attempts = list(attempts or [])


class PlannerDeadlineExceeded(RuntimeError):
    """The planner call did not return within its per-call deadline."""


_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS\**\s*[:=]\s*", re.I)
# A header line: HYPOTHESIS at the START of a line (optionally behind
# markdown / comment decoration), so prose such as "my hypothesis: ..."
# earlier in a reply can never shadow the real header.
_HYPOTHESIS_LINE_RE = re.compile(
    r"^[ \t]*(?:[#>*_`-]+[ \t]*)*HYPOTHESIS\**[ \t]*[:=]\**[ \t]*", re.M
)
# The same header typed INSIDE a python block (colon form only, so a genome
# assignment such as ``hypothesis = ...`` is never touched).
_HYPOTHESIS_CODE_LINE_RE = re.compile(
    r"^[ \t]*(?:[#>*_`-]+[ \t]*)*HYPOTHESIS\**[ \t]*:\**[ \t]*", re.M | re.I
)
_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.S)
_GENOME_MARKERS = ("PHASES = [", "def plan_communication_turn", "def plan_submit_round")
_GENOME_MARKER_LINE = re.compile(
    r"^(?:PHASES = \[|def plan_communication_turn\b|def plan_submit_round\b)",
    re.M,
)
_HOST_PREFIX_MARKERS = re.compile(
    r"^(?:MESSAGE_INSTRUCTION =|SUBMIT_INSTRUCTION =|def reject_nonfinite\b)",
    re.M,
)


def _balanced_json(text: str, start: int) -> str | None:
    """The balanced ``{...}`` text that opens at ``text[start]``, or None
    when it never closes.  Braces inside single- or double-quoted strings
    do not count (a backslash escapes the next character in a string)."""

    depth = 0
    in_str: str | None = None
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_str:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_str:
                in_str = None
            continue
        if char in "\"'":
            in_str = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def sanitize_hypothesis(hypothesis: Any) -> dict[str, Any] | None:
    """Closed-key, clipped copy of a HYPOTHESIS header (the planner's own text).

    Keeps ``target_units`` (sanitized unit ids, at most 12; a string is split
    on commas / whitespace) and ``failure_class`` / ``mechanism`` /
    ``predicted_effect`` / ``raw`` (whitespace collapsed, clipped); None when
    no key is kept."""

    if not isinstance(hypothesis, dict):
        return None
    out: dict[str, Any] = {}
    units = hypothesis.get("target_units")
    if isinstance(units, str):
        units = [u for u in re.split(r"[,\s]+", units) if u]
    if isinstance(units, (list, tuple)):
        clean = [clean_unit_id(u) for u in units][:12]
        out["target_units"] = [u for u in clean if u]
    for key, limit in (
        ("failure_class", 40),
        ("mechanism", 300),
        ("predicted_effect", 300),
        ("raw", 300),
    ):
        value = hypothesis.get(key)
        if value is not None and str(value).strip():
            out[key] = " ".join(str(value).split())[:limit]
    return out or None


def parse_hypothesis(text: str) -> dict[str, Any] | None:
    """The reply's ``HYPOTHESIS: {...}`` header, parsed and sanitized.

    Accepts JSON or a Python-literal dict (single quotes), on one line or
    spread over several; anything unparseable is kept as ``{"raw": ...}``.
    None when the reply has no HYPOTHESIS header at all.
    """

    if not isinstance(text, str):
        return None
    match = _pick_hypothesis_match(text)
    if not match:
        return None
    rest = text[match.end():]
    brace = rest.find("{")
    newline = rest.find("\n")
    if brace != -1 and (newline == -1 or brace <= newline or not rest[:brace].strip()):
        blob = _balanced_json(rest, brace)
        if blob:
            for loader in (json.loads, _ast.literal_eval):
                try:
                    value = loader(blob)
                except Exception:  # noqa: BLE001 - try the next loader
                    continue
                if isinstance(value, dict):
                    return sanitize_hypothesis(value) or {"raw": blob[:300]}
            return {"raw": " ".join(blob.split())[:300]}
    line = rest.split("\n", 1)[0].strip()
    return {"raw": line[:300]} if line else {"raw": ""}


def _pick_hypothesis_match(text: str) -> "re.Match[str] | None":
    """The HYPOTHESIS header to parse.

    Line-anchored headers come first: one directly followed by ``{`` (or
    with a blank rest of line and ``{`` as the next non-blank character)
    wins over prose, else the first one.  Only when no line starts with a
    header is the text searched unanchored, again preferring a match
    directly followed by ``{``."""

    anchored = list(_HYPOTHESIS_LINE_RE.finditer(text))
    if anchored:
        for match in anchored:
            if text[match.end():match.end() + 1] == "{" or (
                not text[match.end():].split("\n", 1)[0].strip()
                and text[match.end():].lstrip()[:1] == "{"
            ):
                return match
        return anchored[0]
    unanchored = list(_HYPOTHESIS_RE.finditer(text))
    for match in unanchored:
        if text[match.end():match.end() + 1] == "{":
            return match
    return unanchored[0] if unanchored else None


def _strip_hypothesis_lines(code: str) -> str:
    """Drop HYPOTHESIS header lines (and a JSON object that starts on one)
    that a reply put INSIDE its code (the python block, or the unfenced
    remainder): they belong to the header, and left in the genome they
    break it."""

    pieces: list[str] = []
    pos = 0
    for match in _HYPOTHESIS_CODE_LINE_RE.finditer(code):
        if match.start() < pos:
            continue
        end = code.find("\n", match.end())
        if code[match.end():match.end() + 1] == "{":
            blob = _balanced_json(code, match.end())
            if blob:
                end = code.find("\n", match.end() + len(blob))
        end = len(code) if end == -1 else end + 1
        pieces.append(code[pos:match.start()])
        pos = end
    if not pieces:
        return code
    pieces.append(code[pos:])
    return "".join(pieces)


def _looks_like_full_file(code: str) -> bool:
    """True when a code block carries HOST scaffold (main() or the prefix's
    own definitions).  Imports alone do not make a full file: a genome-only
    block with a stray ``import`` must reach :func:`check_genome`, whose
    exact "imports are host-owned" diagnostic the re-mint needs."""

    return bool(_MAIN_LINE.search(code) or _HOST_PREFIX_MARKERS.search(code))


def _scaffold_drift(reply_source: str, incumbent_source: str) -> str | None:
    """Compare a full-file reply's host scaffold (the prefix before the
    genome and the ``main()`` part after it) with the incumbent's, ignoring
    surrounding whitespace: a note naming the drifted parts, or None."""

    got = genome_bounds(reply_source)
    want = genome_bounds(incumbent_source)
    if got is None or want is None:
        return None
    notes = []
    for label, a, b in (
        ("prefix", reply_source[: got[0]], incumbent_source[: want[0]]),
        ("main", reply_source[got[1]:], incumbent_source[want[1]:]),
    ):
        if a.strip() == b.strip():
            continue
        changed = [
            line for line in difflib.unified_diff(
                b.strip().splitlines(), a.strip().splitlines(),
                lineterm="", n=0,
            )
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]
        first = next((line for line in changed if line.startswith("+")), "")
        notes.append(
            f"{label}: {len(changed)} changed line(s)"
            + (f", first `{first[1:].strip()[:80]}`" if first else "")
        )
    if not notes:
        return None
    return (
        "the reply re-typed the host scaffold with drift; the drift was "
        "DISCARDED (host bytes kept): " + "; ".join(notes)
    )


def parse_genome_reply(
    text: str, *, incumbent_source: str | None = None
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """(hypothesis, genome, notes) from a genome-mode planner reply.

    The genome is the last fenced code block that carries a genome marker
    (``PHASES = [`` / ``def plan_communication_turn`` / ``def
    plan_submit_round``), or the unfenced remainder (an unclosed fence from a
    truncated reply is accepted).  A full-file reply is cut down to its
    genome region and any scaffold drift is reported and discarded.
    """

    notes: list[str] = []
    raw = str(text or "")
    hypothesis = parse_hypothesis(raw)
    if hypothesis is None:
        notes.append("no HYPOTHESIS header")
    code: str | None = None
    blocks = [m.group(2) for m in _FENCE_RE.finditer(raw)]
    candidates = [b for b in blocks if any(k in b for k in _GENOME_MARKERS)]
    if candidates:
        code = candidates[-1]
        if len(candidates) > 1:
            notes.append(f"{len(candidates)} code blocks; used the last one")
        stripped = _strip_hypothesis_lines(code)
        if stripped != code:
            notes.append("dropped a HYPOTHESIS line from inside the code block")
            code = stripped
    else:
        opening = re.search(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n", raw)
        remainder = raw[opening.end():] if opening else raw
        if opening:
            notes.append("unclosed code fence (reply may be truncated)")
        remainder = _strip_hypothesis_lines(remainder)
        if any(k in remainder for k in _GENOME_MARKERS):
            code = remainder
    if code is None:
        return hypothesis, None, notes
    if _looks_like_full_file(code):
        region = genome_region(code) if _MAIN_LINE.search(code) else None
        if region is not None:
            notes.append("full-file reply: took its genome region")
            if incumbent_source:
                drift = _scaffold_drift(code, incumbent_source)
                if drift:
                    notes.append(drift)
            code = region
        else:
            # Host scaffold without a usable region (typically a full file
            # truncated before main()): the genome starts at its first
            # genome marker; anything the reply typed before it is host
            # prefix and is discarded.
            marker = _GENOME_MARKER_LINE.search(code)
            if marker is None:
                notes.append(
                    "full-file reply without a recognizable genome region"
                )
                return hypothesis, None, notes
            tail = code[marker.start():]
            main = _MAIN_LINE.search(tail)
            if main:
                tail = tail[:main.start()]
            notes.append(
                "full-file reply without a usable main() (truncated?): took "
                "everything from its first genome line on"
            )
            code = tail
    genome = code.strip("\n")
    return hypothesis, (genome if genome.strip() else None), notes


_GENOME_OUTPUT_CONTRACT = (
    "Reply format (strict): first a line\n"
    'HYPOTHESIS: {"target_units": [...], "failure_class": "...", '
    '"mechanism": "...", "predicted_effect": "..."}\n'
    "then ONE ```python code block holding ONLY the complete new genome "
    "(from `PHASES = [` up to, not including, `def main`). Do not reproduce "
    "the host prefix or main(); no other code blocks."
)


def build_planner_request(
    *, n_agents: int, goal: str, worker_contract: str
) -> PlannerRequest:
    """The :class:`PlannerRequest` a mint's dry run is built from (team size,
    information goal, worker contract)."""

    return PlannerRequest(
        n_agents=n_agents,
        information_goal=goal,
        python_worker_contract=worker_contract,
    )


def build_mint_runtime(
    *,
    llm_provider: str,
    worker_model: str,
    goal: str,
    worker_contract: str,
    max_parallel_agents: int,
    request_timeout: float,
    n_agents: int,
    budgets: PythonRunBudgets | None = None,
) -> MASRuntimeConfig:
    """Runtime settings of a mint's dry run: the worker model, contract and
    per-case budgets of a real run, with the execution timeout the benchmark
    engine derives for ``n_agents``."""

    budgets = budgets or PythonRunBudgets()
    cfg = _run_config(
        llm_provider=llm_provider,
        worker_model=worker_model,
        goal=goal,
        worker_contract=worker_contract,
        max_parallel_agents=max_parallel_agents,
        request_timeout=request_timeout,
        n_agents=n_agents,
        budgets=budgets,
    )
    return MASRuntimeConfig(
        llm_provider=llm_provider,
        model_name=worker_model,
        temperature=0.0,
        max_parallel_agents=max(1, int(max_parallel_agents)),
        python_worker_contract=worker_contract,
        python_execution_timeout=_resolved_python_execution_timeout(
            cfg, n_agents=n_agents
        ),
        python_max_rounds=budgets.max_rounds,
        python_max_model_calls=budgets.max_model_calls,
        python_max_completion_tokens=budgets.max_completion_tokens,
        python_max_messages=budgets.max_messages,
        information_goal=goal,
    )


def _complete_with_transport_retry(
    client: Any,
    prompt: str,
    *,
    model_name: str,
    attempts: int = 3,
    backoff_s: float = 20.0,
) -> Any:
    """Planner call with an outer retry on transport-shaped failures.

    The planner client itself retries connect-class failures only (and only
    when built with ``connect_attempts > 1``).  Here every error that
    :func:`queenbee.program.execute._is_transport_error` classifies as
    transport-shaped (connection errors, timeouts including the wall-clock
    guard's, rate limits, HTTP 500 / 502 / 503 and the like) is retried, up
    to ``attempts`` calls with a sleep of ``backoff_s`` x attempt in between;
    any other error propagates at once."""

    import time as _time

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return client.complete(
                prompt, model_name=model_name, temperature=0.0,
                json_mode=False,
            )
        except Exception as exc:  # noqa: BLE001 - classified below
            if not _is_transport_error(exc):
                raise
            last = exc
            if attempt < attempts - 1:
                _time.sleep(backoff_s * (attempt + 1))
    raise last  # type: ignore[misc]


def planner_usage_entry(
    call: str,
    model: str,
    response: Any,
    wall_s: float,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    """One ``usage_log`` record for a planner call.

    Token counts come from the response's ``usage`` (exp_graph
    ``LLMUsage`` or a plain dict); a missing or non-integer field is None.
    ``reasoning_tokens`` is added only when the usage carries it as an
    integer.  ``error`` (a failed call: no response, tokens None) is
    recorded as an extra ``error`` key so failed and timed-out calls stay
    visible in cost reporting.
    """

    usage = getattr(response, "usage", None)

    def _tokens(name: str) -> int | None:
        if usage is None:
            return None
        value = (
            usage.get(name) if isinstance(usage, dict)
            else getattr(usage, name, None)
        )
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return int(value)

    entry: dict[str, Any] = {
        "call": str(call),
        "model": str(model),
        "prompt_tokens": _tokens("prompt_tokens"),
        "completion_tokens": _tokens("completion_tokens"),
        "wall_s": round(max(0.0, float(wall_s)), 3),
    }
    reasoning = _tokens("reasoning_tokens")
    if reasoning is not None:
        entry["reasoning_tokens"] = reasoning
    if error:
        entry["error"] = str(error)[:300]
    return entry


def _metered_planner_call(
    client: Any,
    prompt: str,
    *,
    model: str,
    call: str,
    usage_log: list | None,
) -> Any:
    """``_complete_with_transport_retry`` + one usage entry, success OR failure.

    A call that raises (timeout, HTTP error, exhausted transport retries)
    still appends an entry (tokens None, ``error`` set) before re-raising, so
    a failed planner call stays visible in the meter.  The entry covers the
    whole call: attempts retried inside it are not logged separately.
    """

    import time as _time

    started = _time.monotonic()
    try:
        response = _complete_with_transport_retry(
            client, prompt, model_name=model
        )
    except Exception as exc:
        if usage_log is not None:
            usage_log.append(
                planner_usage_entry(
                    call, model, None, _time.monotonic() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        raise
    if usage_log is not None:
        usage_log.append(
            planner_usage_entry(
                call, model, response, _time.monotonic() - started
            )
        )
    return response


_CAP_LOCK = threading.Lock()
_CAP_STATE: dict[int, list[Any]] = {}


def _env_number(name: str, kind: Any) -> Any:
    """Positive number from env ``name`` (unset / invalid / <= 0 -> None)."""

    import os

    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = kind(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _capped_client(client: Any) -> Any | None:
    """The first client along the ``_inner`` links (at most 10 deep) that
    exposes ``_max_completion_tokens``, or None.  In the planner client
    chain that is the exp_graph ``OpenAIChatClient`` at the bottom."""

    node = client
    for _depth in range(10):
        if node is None:
            return None
        if hasattr(node, "_max_completion_tokens"):
            return node
        node = getattr(node, "_inner", None)
    return None


class _completion_cap:
    """Per-call planner output cap (reference-counted, thread-safe).

    The planner client chain has no per-call token parameter, so the cap is
    set on the innermost OpenAI client for the duration of the call and the
    original value is restored when the LAST concurrent override ends.  A
    client without the attribute (fakes, stubs) is left untouched."""

    def __init__(self, client: Any, cap: int | None) -> None:
        self.target = _capped_client(client) if cap else None
        self.cap = int(cap) if cap else None

    def __enter__(self) -> "_completion_cap":
        if self.target is None:
            return self
        with _CAP_LOCK:
            state = _CAP_STATE.get(id(self.target))
            if state is None:
                state = [self.target._max_completion_tokens, 0]
                _CAP_STATE[id(self.target)] = state
            state[1] += 1
            self.target._max_completion_tokens = self.cap
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.target is None:
            return
        with _CAP_LOCK:
            state = _CAP_STATE.get(id(self.target))
            if state is None:
                return
            state[1] -= 1
            if state[1] <= 0:
                self.target._max_completion_tokens = state[0]
                _CAP_STATE.pop(id(self.target), None)


def _planner_call(
    client: Any,
    prompt: str,
    *,
    model: str,
    call: str,
    usage_log: list | None,
    max_completion_tokens: int | None = None,
    deadline_s: float | None = None,
) -> Any:
    """Metered planner call with an optional output cap and a HARD per-call
    wall-clock deadline.

    With neither set this is exactly :func:`_metered_planner_call`.  The
    deadline bounds the whole call including transport retries; when it
    passes, the call is abandoned (daemon thread), one usage entry with
    ``error`` is logged, and :class:`PlannerDeadlineExceeded` is raised - it
    is not retried here.
    """

    if max_completion_tokens is None and not deadline_s:
        return _metered_planner_call(
            client, prompt, model=model, call=call, usage_log=usage_log
        )
    import time as _time

    def _log(response: Any, wall: float, error: str | None = None) -> None:
        if usage_log is not None:
            usage_log.append(
                planner_usage_entry(call, model, response, wall, error=error)
            )

    started = _time.monotonic()
    if not deadline_s:
        try:
            with _completion_cap(client, max_completion_tokens):
                response = _complete_with_transport_retry(
                    client, prompt, model_name=model
                )
        except Exception as exc:
            _log(None, _time.monotonic() - started, f"{type(exc).__name__}: {exc}")
            raise
        _log(response, _time.monotonic() - started)
        return response

    result: list[Any] = []
    failure: list[BaseException] = []
    # Abandonment hand-off: once the deadline fires, the still-running call
    # fills the already-logged usage entry in place when (if) it returns,
    # so tokens an abandoned call was billed for do not stay None.
    state_lock = threading.Lock()
    state: dict[str, Any] = {"done": False, "entry": None}

    def _run() -> None:
        try:
            with _completion_cap(client, max_completion_tokens):
                result.append(
                    _complete_with_transport_retry(
                        client, prompt, model_name=model
                    )
                )
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            failure.append(exc)
        finally:
            with state_lock:
                state["done"] = True
                entry = state["entry"]
            if entry is not None and result:
                late = planner_usage_entry(
                    call, model, result[0], _time.monotonic() - started
                )
                for key in ("prompt_tokens", "completion_tokens",
                            "reasoning_tokens"):
                    if late.get(key) is not None:
                        entry[key] = late[key]
                entry["late_wall_s"] = late["wall_s"]
                entry["abandoned"] = True

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(float(deadline_s))
    wall = _time.monotonic() - started
    with state_lock:
        abandoned = not state["done"]
        if abandoned:
            message = (
                f"planner call exceeded its {float(deadline_s):.0f}s deadline"
            )
            state["entry"] = planner_usage_entry(
                call, model, None, wall,
                error=f"PlannerDeadlineExceeded: {message}",
            )
    if abandoned:
        if usage_log is not None:
            usage_log.append(state["entry"])
        raise PlannerDeadlineExceeded(message)
    if failure:
        exc = failure[0]
        _log(None, wall, f"{type(exc).__name__}: {exc}")
        raise exc
    _log(result[0], wall)
    return result[0]


def _dump_prompt(directory: Path | None, name: str, text: str) -> None:
    """Best-effort prompt dump for the leak audit (never breaks minting)."""

    if directory is None:
        return
    try:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(text, encoding="utf-8")
    except OSError:
        pass


#: Prompt sections that carry execution evidence: the scope a leak audit can
#: be restricted to (``sections=``).
EVIDENCE_SECTION_HEADERS: tuple[str, ...] = (
    "=== FAILURE MAP",
    "=== DIAGNOSIS",
    "=== TASK TEMPLATES",
)


def _section_text(text: str, headers: Any) -> str:
    """Concatenation of every section starting at one of ``headers`` and
    running to the next line that starts a new ``===`` block (or EOF); the
    whole text when ``headers`` is empty."""

    wanted = [str(h) for h in headers or [] if str(h)]
    if not wanted:
        return text
    chunks: list[str] = []
    lines = text.splitlines(keepends=True)
    inside = False
    for line in lines:
        stripped = line.lstrip()
        starts = any(stripped.startswith(h) for h in wanted)
        if starts:
            inside = True
        elif inside and stripped.startswith("===") and not starts:
            inside = False
        if inside:
            chunks.append(line)
    return "".join(chunks)


def _value_forms(value: Any, *, min_chars: int) -> list[str]:
    """Distinctive string forms of a forbidden (answer / ground-truth) value.

    A scalar contributes its stripped text (None and booleans contribute
    nothing).  A list or tuple contributes its JSON text (compact and with
    default spacing), a dict its compact sorted-key JSON, and both,
    recursively, their first 200 members (a dict's values).  Forms shorter
    than ``min_chars`` are skipped (a bare ``3`` is everywhere in a
    prompt)."""

    forms: set[str] = set()

    def _add(item: Any) -> None:
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, (list, tuple)):
            try:
                forms.add(json.dumps(list(item), separators=(",", ":")))
                forms.add(json.dumps(list(item)))
            except (TypeError, ValueError):
                pass
            for member in list(item)[:200]:
                _add(member)
            return
        if isinstance(item, dict):
            try:
                forms.add(json.dumps(item, separators=(",", ":"), sort_keys=True))
            except (TypeError, ValueError):
                pass
            for member in list(item.values())[:200]:
                _add(member)
            return
        text = str(item).strip()
        if text:
            forms.add(text)

    _add(value)
    return sorted(f for f in forms if len(f) >= int(min_chars))


def audit_text_leaks(
    text: str,
    forbidden_case_ids: Any = (),
    *,
    forbidden_values: Any = None,
    sections: Any = None,
    min_value_chars: int = 4,
) -> list[dict[str, Any]]:
    """In-memory counterpart of :func:`audit_prompt_leaks` (no ``path``).

    Hits: ``{"case_id", "count"}`` for forbidden case ids and ``{"value_index",
    "form_chars", "count"}`` for forbidden values (the value itself is never
    echoed, so an audit log cannot become a leak)."""

    scope = _section_text(str(text or ""), sections) if sections else str(text or "")
    hits: list[dict[str, Any]] = []
    ids = sorted({str(c).strip() for c in forbidden_case_ids or [] if str(c).strip()})
    for case in ids:
        rx = re.compile(r"(?<![\w-])" + re.escape(case) + r"(?!\d)")
        count = len(rx.findall(scope))
        if count:
            hits.append({"case_id": case, "count": count})
    for index, value in enumerate(list(forbidden_values or [])):
        for form in _value_forms(value, min_chars=min_value_chars):
            rx = re.compile(r"(?<![\w.])" + re.escape(form) + r"(?![\w])")
            count = len(rx.findall(scope))
            if count:
                hits.append(
                    {"value_index": index, "form_chars": len(form), "count": count}
                )
    return hits


def audit_prompt_leaks(
    paths: Any,
    forbidden_case_ids: Any,
    *,
    pattern: str = "*prompt*.txt",
    forbidden_values: Any = None,
    sections: Any = None,
    min_value_chars: int = 4,
) -> list[dict[str, Any]]:
    """Scan dumped prompts for forbidden (validation / TEST) case ids.

    ``paths``: a directory (searched recursively with ``pattern``), a file,
    or an iterable of either.  A case id matches only as a whole token
    (``II-13`` does not match inside ``III-13`` or ``II-130``).  Returns one
    ``{"path", "case_id", "count"}`` dict per hit; empty list = clean.

    Optional: ``forbidden_values`` (e.g. the training units' ground-truth
    answers) are searched as whole tokens in their distinctive forms
    (>= ``min_value_chars`` chars); a hit is ``{"path", "value_index",
    "form_chars", "count"}`` - the value itself is never echoed.
    ``sections`` restricts BOTH scans to the named prompt sections (see
    :data:`EVIDENCE_SECTION_HEADERS` - the failure map, diagnosis cards,
    task templates), so program text elsewhere in the prompt cannot trigger
    false value hits.  With neither, only case ids are searched, over the
    whole file.  The default pattern also covers ``remint_prompt.txt``.
    """

    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[Path] = []
    for item in paths or []:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob(pattern) if p.is_file()))
        elif path.is_file():
            files.append(path)
    ids = sorted({str(c).strip() for c in forbidden_case_ids or [] if str(c).strip()})
    patterns = {
        case: re.compile(r"(?<![\w-])" + re.escape(case) + r"(?!\d)")
        for case in ids
    }
    hits: list[dict[str, Any]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if sections:
            text = _section_text(text, sections)
        for case, rx in patterns.items():
            count = len(rx.findall(text))
            if count:
                hits.append({"path": str(path), "case_id": case, "count": count})
        if forbidden_values:
            for hit in audit_text_leaks(
                text, (), forbidden_values=forbidden_values,
                min_value_chars=min_value_chars,
            ):
                hits.append({"path": str(path), **hit})
    return hits


def mint_python_challenger(
    *,
    planner_client: Any,
    planner_model: str,
    prompt: str,
    request: PlannerRequest,
    runtime: MASRuntimeConfig,
    workdir: Path,
    usage_log: list | None = None,
    prompt_dump_dir: Path | None = None,
    genome_only: bool = True,
    incumbent_source: str | None = None,
    max_completion_tokens: int | None = None,
    deadline_s: float | None = None,
    hypothesis_out: dict[str, Any] | None = None,
) -> str:
    """Mint one challenger: a planner genome spliced into the incumbent.

    The reply is a ``HYPOTHESIS`` line + a genome block; the host extracts
    the genome (from a full-file reply too, discarding any scaffold drift),
    splices it into ``incumbent_source`` (:func:`splice_genome`), validates
    and dry-runs it, so no worker token is spent on a program that cannot
    run.  On failure the planner re-mints ONCE with the exact diagnostic
    appended (``remint_prompt.txt``); if that also fails, :class:`MintFailed`
    is raised.  ``hypothesis_out`` (a dict) is filled in place with the
    parsed HYPOTHESIS header.

    ``usage_log``: when a list is given, one :func:`planner_usage_entry`
    dict is appended per planner call (``call`` = ``"mint"`` or
    ``"remint"``); a call that raises is still logged (tokens None,
    ``error`` set) before the exception propagates.  ``prompt_dump_dir``:
    when set, the exact prompts are written there (``mint_prompt.txt``,
    ``remint_prompt.txt``) for :func:`audit_prompt_leaks`.

    ``max_completion_tokens`` caps each planner call's output and
    ``deadline_s`` is a hard per-call wall clock (a hit raises
    ``MintFailed(kind="planner_deadline")``; it is not retried here).  When
    not passed they come from ``QB_PLANNER_MAX_COMPLETION_TOKENS`` /
    ``QB_PLANNER_DEADLINE_S``, else :data:`DEFAULT_PLANNER_MAX_COMPLETION_TOKENS`
    / :data:`DEFAULT_PLANNER_DEADLINE_S`; an explicit 0 disables either.
    Genome mode is the only minting mode (``genome_only`` must be True).
    """

    if not genome_only:
        raise ValueError("only genome-mode minting is available (genome_only=True)")
    if max_completion_tokens is None:
        max_completion_tokens = (
            _env_number("QB_PLANNER_MAX_COMPLETION_TOKENS", int)
            or DEFAULT_PLANNER_MAX_COMPLETION_TOKENS
        )
    if deadline_s is None:
        deadline_s = (
            _env_number("QB_PLANNER_DEADLINE_S", float) or DEFAULT_PLANNER_DEADLINE_S
        )
    workdir.mkdir(parents=True, exist_ok=True)
    runner = CodeProcessRunner(
        PythonExecutionLimits(
            timeout_seconds=runtime.python_execution_timeout,
            cpu_seconds=runtime.python_cpu_seconds,
            memory_mb=runtime.python_memory_mb,
            max_output_bytes=runtime.python_max_output_bytes,
        ),
        extra_env=task_worker_env(),
    )
    if not incumbent_source or genome_bounds(incumbent_source) is None:
        raise ValueError(
            "genome_only minting needs incumbent_source with a genome region"
        )
    return _mint_genome(
        planner_client=planner_client,
        planner_model=planner_model,
        prompt=prompt,
        request=request,
        runtime=runtime,
        runner=runner,
        workdir=workdir,
        usage_log=usage_log,
        prompt_dump_dir=prompt_dump_dir,
        incumbent_source=incumbent_source,
        max_completion_tokens=max_completion_tokens,
        deadline_s=deadline_s,
        hypothesis_out=hypothesis_out,
    )


def _genome_line_note(
    source: str, incumbent_source: str, line: Any
) -> str | None:
    """Locate a spliced-program line: "genome line k" inside the genome
    region, else "host scaffold line N", followed by the line's text; None
    for a missing or invalid line number, or when neither source has a
    genome region."""

    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        return None
    bounds = genome_bounds(source) or genome_bounds(incumbent_source)
    if bounds is None:
        return None
    offset = source[: bounds[0]].count("\n")
    lines = source.splitlines()
    text = lines[line - 1].strip() if line <= len(lines) else ""
    local = line - offset
    body_lines = source[bounds[0]:bounds[1]].count("\n")
    where = (
        f"genome line {local}"
        if 1 <= local <= body_lines
        else f"host scaffold line {line}"
    )
    return where + (f": `{text[:160]}`" if text else "")


_MESSAGE_LINE = re.compile(r"^\s*(\d+):\d+:\s")


def _validate_spliced(
    source: str,
    *,
    incumbent_source: str,
    worker_contract: str,
    runner: Any,
    request: Any,
    runtime: Any,
) -> dict[str, Any] | None:
    """None when the spliced program passes the validator, the static
    work-instruction length check and the fake dry-run; else an exact,
    answer-free diagnostic dict (``{"stage", "message"}``)."""

    validation = validate_python_source(source, worker_contract=worker_contract)
    if not validation.valid:
        errors = []
        for error in (validation.errors or [])[:4]:
            line = error.get("line")
            if not isinstance(line, int) or isinstance(line, bool):
                # pyflakes-style messages carry the position in their text
                # ("86:16: undefined name 'X'") instead of the line field.
                found = _MESSAGE_LINE.match(str(error.get("message") or ""))
                line = int(found.group(1)) if found else None
            note = _genome_line_note(source, incumbent_source, line)
            errors.append(
                f"{error.get('error_type') or 'ValidationError'}: "
                f"{str(error.get('message') or 'source failed validation')[:300]}"
                + (f" ({note})" if note else "")
            )
        return {
            "stage": "validator",
            "message": "the fail-closed validator rejected the spliced "
            "program: " + " | ".join(errors or ["unknown validation error"]),
        }
    try:  # static PHASES "wi" strings: the runtime rejects > 800 chars at execution
        from queenbee.evo.screen import WI_MAX_CHARS, static_phase_wis

        long_wis = [len(w) for w in static_phase_wis(source) if len(w) > WI_MAX_CHARS]
    except Exception:  # noqa: BLE001 - advisory lint; the screen re-checks
        long_wis = []
    if long_wis:
        return {
            "stage": "validator",
            "message": (
                f"a PHASES work instruction has {max(long_wis)} characters; the runtime "
                f"rejects any work_instruction longer than {WI_MAX_CHARS} characters "
                "(DataFlowError) - shorten it"
            ),
        }
    dry = runner.run(
        source,
        _dry_run_payload(request, runtime),
        agent_canaries=_dry_run_canaries(request.n_agents),
    )
    if dry.runtime_success:
        return None
    failure = dry.failure or {"error_type": "RuntimeError", "message": "fake dry-run failed"}
    return {
        "stage": "dry_run",
        "message": (
            f"the fake dry-run (n_agents={request.n_agents}, max_rounds="
            f"{runtime.python_max_rounds}) raised "
            f"{failure.get('error_type')}: {str(failure.get('message'))[:400]} "
            "- a returned action or submit round broke the runtime law of "
            "the API card"
        ),
    }


def _remint_notice(diagnostic: dict[str, Any], genome: str | None) -> str:
    """The text appended to the prompt for the re-mint: the failed stage,
    its exact diagnostic, the rejected genome (clipped) and the output
    contract again."""

    text = (
        "\n\n=== YOUR PREVIOUS REPLY WAS REJECTED BEFORE ANY EXECUTION "
        "(one retry) ===\n"
        f"Stage: {diagnostic.get('stage')}\n"
        f"Exact diagnostic: {diagnostic.get('message')}\n"
    )
    if genome:
        text += (
            "Your rejected genome:\n```python\n" + genome.strip("\n")[:9000]
            + "\n```\n"
        )
    text += (
        "Fix exactly this problem while keeping your design idea (do not "
        "regress to the incumbent). " + _GENOME_OUTPUT_CONTRACT
    )
    return text


def _mint_genome(
    *,
    planner_client: Any,
    planner_model: str,
    prompt: str,
    request: Any,
    runtime: Any,
    runner: Any,
    workdir: Path,
    usage_log: list | None,
    prompt_dump_dir: Path | None,
    incumbent_source: str,
    max_completion_tokens: int | None,
    deadline_s: float | None,
    hypothesis_out: dict[str, Any] | None,
) -> str:
    """Genome-mode mint: at most TWO planner calls (mint + one re-mint).

    Every reply, extracted genome and spliced attempt is written to
    ``workdir`` (``mint_reply_NN.txt``, ``mint_genome_NN.py``,
    ``mint_attempt_NN.py``) together with ``mint_diag.json``."""

    worker_contract = runtime.python_worker_contract
    attempts: list[dict[str, Any]] = []
    hypothesis: dict[str, Any] | None = None
    current = prompt
    _dump_prompt(prompt_dump_dir, "mint_prompt.txt", prompt)
    diagnostic: dict[str, Any] | None = None

    def _write(name: str, text: str) -> None:
        try:
            (workdir / name).write_text(text, encoding="utf-8")
        except OSError:
            pass

    def _finish(status: str) -> None:
        _write(
            "mint_diag.json",
            json.dumps(
                {
                    "interface": "genome",
                    "status": status,
                    "hypothesis": hypothesis,
                    "attempts": attempts,
                },
                indent=1,
                default=str,
            ),
        )

    for attempt in range(2):
        call = "mint" if attempt == 0 else "remint"
        try:
            response = _planner_call(
                planner_client, current, model=planner_model, call=call,
                usage_log=usage_log,
                max_completion_tokens=max_completion_tokens,
                deadline_s=deadline_s,
            )
        except PlannerDeadlineExceeded as exc:
            attempts.append({"attempt": attempt, "stage": "planner_deadline",
                             "message": str(exc)})
            _finish("planner_deadline")
            raise MintFailed(
                str(exc), hypothesis, kind="planner_deadline", attempts=attempts
            ) from exc
        text = str(getattr(response, "text", "") or "")
        _write(f"mint_reply_{attempt:02d}.txt", text)
        parsed, genome, notes = parse_genome_reply(
            text, incumbent_source=incumbent_source
        )
        if parsed is not None:
            hypothesis = parsed
            if hypothesis_out is not None:
                hypothesis_out.clear()
                hypothesis_out.update(parsed)
        record: dict[str, Any] = {"attempt": attempt, "notes": notes,
                                  "reply_chars": len(text)}
        source: str | None = None
        if genome is None:
            diagnostic = {
                "stage": "extract",
                "message": (
                    "the reply contained no genome (empty reply, or no "
                    "```python block with `PHASES = [` / `def "
                    "plan_communication_turn`); it may have been truncated - "
                    "keep the reasoning short and the genome compact"
                ),
            }
        else:
            _write(f"mint_genome_{attempt:02d}.py", genome + "\n")
            problems = check_genome(genome, incumbent_source=incumbent_source)
            if problems:
                diagnostic = {
                    "stage": "genome",
                    "message": " | ".join(problems[:6]),
                }
            else:
                source = splice_genome(incumbent_source, genome)
                _write(f"mint_attempt_{attempt:02d}.py", source)
                diagnostic = _validate_spliced(
                    source,
                    incumbent_source=incumbent_source,
                    worker_contract=worker_contract,
                    runner=runner,
                    request=request,
                    runtime=runtime,
                )
        record["diagnostic"] = diagnostic
        attempts.append(record)
        if diagnostic is None and source is not None:
            _finish("ok")
            return source
        if attempt == 0:
            current = prompt + _remint_notice(diagnostic or {}, genome)
            _dump_prompt(prompt_dump_dir, "remint_prompt.txt", current)
    _finish("mint_failed")
    reason = (
        f"{(diagnostic or {}).get('stage', 'unknown')}: "
        f"{(diagnostic or {}).get('message', 'mint failed')}"
    )
    raise MintFailed(reason[:600], hypothesis, kind="invalid", attempts=attempts)


# Changed lines that carry no mechanism (signature continuations, guards,
# idle returns): skipped by the diff summary so that its item budget goes to
# the routing logic.
_DIFF_BOILERPLATE = re.compile(
    r"^(?:[\w\s,=]*\):"
    r"|else:|pass|try:|recipients = \[\]"
    r"|if n_agents <= 1:|return \{\"mode\": \"idle\", \"recipients\": \[\]\}"
    r"|return \{'mode': 'idle', 'recipients': \[\]\})$"
)


def summarize_source_diff(
    diff: Any, *, max_items: int = 8, width: int = 110, max_chars: int = 700
) -> str | None:
    """Compact structural summary of a unified source diff
    (:func:`python_source_diff`), as recorded in hypothesis-ledger entries.

    PHASES entries (changed lines with a ``kind`` key: phase kind / rounds /
    work instruction) and added or removed ``def`` names come first, since
    they carry the mechanism; other changed code lines follow (at most 3
    when there are structural edits), ``max_items`` items in all.  Each item
    is clipped to ``width`` characters and the summary to ``max_chars``.
    Answer-free by construction: the diff is program text, which is
    case-agnostic.
    """

    if not diff:
        return None
    lines = [str(line) for line in diff]
    if lines == ["no source change"]:
        return "change: none (source byte-identical)"
    plus = minus = 0
    truncated = False
    phase_items: list[str] = []
    def_items: list[str] = []
    other: list[str] = []
    for line in lines:
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("... ("):
            truncated = True
            continue
        if not line or line[0] not in "+-":
            continue
        sign, body = line[0], line[1:].strip()
        if sign == "+":
            plus += 1
        else:
            minus += 1
        if not body or body.startswith("#"):
            continue
        if '"kind"' in body or "'kind'" in body:
            phase_items.append(sign + body.rstrip(","))
        elif body.startswith("def "):
            def_items.append(f"{sign}def {body[4:].split('(')[0].strip()}")
        elif not _DIFF_BOILERPLATE.match(body):
            item = sign + " ".join(body.split())
            if item not in other:
                other.append(item)
    items = (phase_items + def_items)[:max_items]
    # Structural edits present -> body lines are supporting detail only.
    other_cap = 3 if items else max_items
    items += other[: max(0, min(other_cap, max_items - len(items)))]
    extra = len(phase_items) + len(def_items) + len(other) - len(items)
    clipped = [
        item if len(item) <= width else item[: width - 3] + "..."
        for item in items
    ]
    text = " | ".join(clipped)
    if extra > 0:
        text += f" | ...+{extra} more changed lines"
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    head = (
        f"change (+{plus}/-{minus} lines"
        f"{', diff truncated' if truncated else ''})"
    )
    return f"{head}: {text}" if text else head


def python_source_diff(
    incumbent_source: str, challenger_source: str, *, max_lines: int = 40
) -> list[str]:
    """Unified diff (one line of context) from the incumbent to the
    challenger source, clipped to ``max_lines`` plus a ``... (N more diff
    lines)`` marker; ``["no source change"]`` when the two are equal.
    Answer-free: it holds program text only."""

    if incumbent_source == challenger_source:
        return ["no source change"]
    diff = list(
        difflib.unified_diff(
            incumbent_source.splitlines(),
            challenger_source.splitlines(),
            fromfile=f"incumbent:{python_source_sha256(incumbent_source)[:12]}",
            tofile=f"challenger:{python_source_sha256(challenger_source)[:12]}",
            lineterm="",
            n=1,
        )
    )
    if len(diff) > max_lines:
        omitted = len(diff) - max_lines
        diff = diff[:max_lines] + [f"... ({omitted} more diff lines)"]
    return diff
