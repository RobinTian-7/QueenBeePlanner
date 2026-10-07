"""Credit fields of the diagnosis cards.

Two per-agent signals that :mod:`queenbee.evo.diagnosis` does not have:

* **readable coverage** - for every agent, how many agents' shard CONTENT it
  had actually read by submit time (directly or relayed).  Unlike the
  runtime's ``known_source_count`` (which counts source *ids*, credited on
  delivery), a message only counts when its recipient was called in the
  round it was delivered (or it was delivered at the submit barrier), and a
  relayed message carries what its sender had *read*, not the ids it had
  merely been credited with.  The called set comes from the runtime call
  ledger when given, else from re-simulating the program's policy
  (cross-checked edge by edge against the trace), else from the trace's
  senders alone.
* **raw-input retention** - for every agent, the fraction of the OTHER
  agents' distinct raw shard items that occur in the message bodies the
  agent actually read.  Items: numbers (matched at the shard's precision or
  rounded / truncated to 1 .. 8 decimals), letter-only strings of 12 or
  more letters as overlapping 4-grams, every other string as one whole
  token (case-insensitive), dict values (and numeric dict keys).  Computed
  from inputs and delivered bodies only - never from an answer or the
  ground truth.  With ``MULTISET_ENV`` set (``--preserve-dups``) items
  count WITH their multiplicity and a ``dup_lost`` count per agent reports
  repeated values that went missing although the value itself arrived.

Leak safety: :func:`sanitize_credit` is the single whitelist every consumer
reads through (ints / floats in closed ranges, a closed ``basis`` word).
Nothing textual from a shard, a body or an answer can survive it.

The module also holds a TEST guard (:func:`assert_not_test`, used by the
prompt builder and the racer as well) and helpers that audit rendered text
for leaks and turn execution traces into rows.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from queenbee.evo.common import TEST_IDS
from queenbee.tasks import get_task

# --------------------------------------------------------------------------- #
# TEST guard (no TEST template may reach any evolution code path)
# --------------------------------------------------------------------------- #

#: The 12 sealed TEST templates (the shared tuple of :mod:`queenbee.evo.common`).
TEST_CASE_IDS: tuple[str, ...] = tuple(TEST_IDS)
_TEST_SET = frozenset(TEST_CASE_IDS)
_TEMPLATE_RE = re.compile(r"(?<![A-Za-z0-9-])(I{1,3}-\d{2})(?!\d)")


def sealed_case_ids() -> tuple[str, ...]:
    """The active task's sealed TEST ids (default :data:`TEST_CASE_IDS`)."""

    ids = get_task().test_ids
    return TEST_CASE_IDS if ids is None else tuple(ids)


def templates_in(value: Any) -> list[str]:
    """Every template id of the active task inside ``value`` (a case id, a
    unit id such as ``II-11@x5``, a path, ...); Silo-Bench ids are a level
    and two digits (``I-01``, ``II-11``, ``III-21``)."""

    pattern = get_task().template_pattern
    rx = _TEMPLATE_RE if pattern is None else re.compile(pattern)
    return rx.findall(str(value or ""))


def assert_not_test(ids: Any, *, where: str = "evo") -> None:
    """Raise ``RuntimeError`` if any id (or any template inside it) is TEST."""

    if ids is None:
        return
    if isinstance(ids, (str, bytes, Path)):
        ids = [ids]
    sealed = _TEST_SET if get_task().test_ids is None else frozenset(sealed_case_ids())
    for item in ids:
        for template in templates_in(item):
            if template in sealed:
                raise RuntimeError(
                    f"LEAKAGE_GUARD: TEST template {template!r} (from "
                    f"{str(item)[:80]!r}) reached {where}"
                )


# --------------------------------------------------------------------------- #
# Trace plumbing
# --------------------------------------------------------------------------- #

#: Where the called set came from: the runtime call ledger, a re-simulation
#: of the program's policy, or the trace's senders.
CREDIT_BASES: tuple[str, ...] = ("ledger", "sim", "trace")


def _dump(output: Any) -> dict[str, Any] | None:
    if output is None:
        return None
    if hasattr(output, "model_dump"):
        try:
            output = output.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            return None
    return output if isinstance(output, dict) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _submit_round(data: Mapping[str, Any]) -> int | None:
    """The latest ``submitted_round`` among the agents' answers, else
    ``rounds_executed - 1``; None when neither is known."""
    rounds = []
    for item in data.get("submissions") or []:
        if isinstance(item, dict):
            rnd = item.get("submitted_round")
            if isinstance(rnd, int) and not isinstance(rnd, bool):
                rounds.append(rnd)
    if rounds:
        return max(rounds)
    executed = data.get("rounds_executed")
    if isinstance(executed, int) and not isinstance(executed, bool) and executed > 0:
        return executed - 1
    return None


def _messages(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in data.get("messages") or []:
        if not isinstance(item, dict):
            continue
        sent = _int(item.get("round_sent"))
        delivered = _int(item.get("round_delivered"))
        src = _int(item.get("src"))
        dst = _int(item.get("dst"))
        if None in (sent, delivered, src, dst):
            continue
        out.append({
            "round_sent": sent, "round_delivered": delivered,
            "src": src, "dst": dst, "body": str(item.get("body") or ""),
        })
    return out


def _called_from_ledger(calls: Any) -> set[tuple[int, int]] | None:
    """(round, agent) pairs of the ledger's send calls; None without a
    usable ledger."""
    if isinstance(calls, Mapping):  # the whole runtime ledger was passed
        calls = calls.get("calls")
    if not calls or isinstance(calls, (str, bytes)):
        return None
    called: set[tuple[int, int]] = set()
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        if str(call.get("mode")) not in ("send",):
            continue
        rnd, agent = _int(call.get("round")), _int(call.get("agent_id"))
        if rnd is not None and agent is not None:
            called.add((rnd, agent))
    return called


def simulate_sends(
    source: str, *, n_agents: int, submit_round: int, goal: str = "all_agents",
    selected_primary: int = 0,
) -> tuple[set[tuple[int, int]], set[tuple[int, int, int]]] | None:
    """(called (round, agent) pairs, edges (round, src, dst)) of ``source``'s
    communication policy, mirroring ``main()``'s counters exactly (id
    knowledge merges on delivery, whether read or not).  Executes ONLY the
    genome region (validator-vetted pure policy code), never ``main``.
    None when the policy cannot be simulated or its barrier differs from
    ``submit_round``."""

    try:
        from queenbee.program.genome import genome_region

        region = genome_region(source)
        if not region:
            return None
        namespace: dict[str, Any] = {}
        exec(region, namespace)  # noqa: S102 - vetted policy section only
        submit = int(namespace["plan_submit_round"](
            n_agents, int(submit_round) + 1, goal
        ))
        if submit != int(submit_round):
            return None
        turn = namespace["plan_communication_turn"]
        knowledge = [{a} for a in range(n_agents)]
        inbox = [0] * n_agents
        called: set[tuple[int, int]] = set()
        edges: set[tuple[int, int, int]] = set()
        for rnd in range(submit):
            round_edges = []
            for agent in range(n_agents):
                action = turn(rnd, agent, n_agents, goal, selected_primary,
                              len(knowledge[agent]), inbox[agent])
                if str(action.get("mode")) == "send":
                    called.add((rnd, agent))
                    for dst in action.get("recipients") or []:
                        round_edges.append((agent, int(dst)))
            snapshot = [set(k) for k in knowledge]
            fresh = [0] * n_agents
            for src, dst in round_edges:
                knowledge[dst] |= snapshot[src]
                fresh[dst] += 1
                edges.add((rnd, src, dst))
            inbox = fresh
        return called, edges
    except Exception:  # noqa: BLE001 - simulation is advisory
        return None


def _called_set(
    messages: list[dict[str, Any]],
    *,
    n_agents: int,
    submit_round: int,
    source: str | None,
    calls: Any,
    goal: str,
) -> tuple[set[tuple[int, int]], str]:
    """(called (round, agent) pairs, basis): the runtime ledger when given;
    else the policy re-simulation when its edges equal the trace's and it
    calls every sender; else the trace's senders."""
    ledger = _called_from_ledger(calls)
    if ledger is not None:
        return ledger, "ledger"
    senders = {(m["round_sent"], m["src"]) for m in messages}
    if source:
        sim = simulate_sends(
            source, n_agents=n_agents, submit_round=submit_round, goal=goal
        )
        if sim is not None:
            called, edges = sim
            trace_edges = {(m["round_sent"], m["src"], m["dst"]) for m in messages}
            if edges == trace_edges and senders <= called:
                return called, "sim"
    return senders, "trace"


def _readable_sets(
    messages: list[dict[str, Any]],
    called: set[tuple[int, int]],
    *,
    n_agents: int,
    submit_round: int,
    submitters: Iterable[int],
) -> tuple[list[set[int]], dict[int, list[int]]]:
    """Per-agent sets of sources whose content the agent read, and per agent
    the indices of the messages whose bodies it read.

    A body delivered at round r is read only when its recipient is called
    at r; at the submit round every submitter reads its inbox.  A message
    carries the set its sender had read by the end of the call that sent it
    (that call's inbox included)."""

    sets = [{a} for a in range(n_agents)]
    delivered: dict[tuple[int, int], list[int]] = defaultdict(list)
    sent_at: dict[int, list[int]] = defaultdict(list)
    for index, message in enumerate(messages):
        delivered[(message["round_delivered"], message["dst"])].append(index)
        sent_at[message["round_sent"]].append(index)
    carried: dict[int, frozenset[int]] = {}
    read: dict[int, list[int]] = defaultdict(list)
    for rnd in range(submit_round):
        updates: dict[int, set[int]] = {}
        for agent in range(n_agents):
            if (rnd, agent) not in called:
                continue
            knowledge = set(sets[agent])
            for index in delivered.get((rnd, agent), []):
                knowledge |= carried.get(index, frozenset({messages[index]["src"]}))
                read[agent].append(index)
            updates[agent] = knowledge
        for agent, knowledge in updates.items():
            sets[agent] = knowledge
        for index in sent_at.get(rnd, []):
            src = messages[index]["src"]
            if 0 <= src < n_agents:
                carried[index] = frozenset(sets[src])
    for agent in submitters:
        if not 0 <= agent < n_agents:
            continue
        for index in delivered.get((submit_round, agent), []):
            sets[agent] |= carried.get(index, frozenset({messages[index]["src"]}))
            read[agent].append(index)
    return sets, read


# --------------------------------------------------------------------------- #
# Raw items of a shard and their presence in bodies
# --------------------------------------------------------------------------- #

_NUM_TOKEN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
_NUMERIC_KEY = re.compile(r"^-?\d+(?:\.\d+)?$")
_WS = re.compile(r"\s+")
_GRAM = 4
_LONG_LETTERS = 12
_MAX_ITEMS = 20_000


def _put(out: set[tuple[str, Any]] | Counter, item: tuple[str, Any]) -> None:
    """Add one item to a set (distinct items) or a Counter (multiset)."""

    if isinstance(out, Counter):
        out[item] += 1
    else:
        out.add(item)


def _items(value: Any, out: set[tuple[str, Any]] | Counter, depth: int = 0) -> None:
    """Add the raw items of ``value`` to ``out``: ``("i", int)`` for
    integral numbers and numeric strings, ``("f", float)`` for other finite
    ones, ``("g", 4-gram)`` for letter-only strings of at least
    ``_LONG_LETTERS`` letters, ``("s", text)`` (lower-cased, whitespace
    collapsed) for every other string.  Recurses into lists, tuples, sets
    and dicts (values and numeric keys), bounded by ``_MAX_ITEMS`` and a
    nesting depth of 12; None, booleans and non-finite numbers are
    skipped."""
    if len(out) >= _MAX_ITEMS or depth > 12 or value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        _put(out, ("i", int(value)))
        return
    if isinstance(value, float):
        if math.isfinite(value):
            _put(out, ("i", int(value)) if value.is_integer() else ("f", float(value)))
        return
    if isinstance(value, str):
        text = _WS.sub(" ", value).strip()
        if not text:
            return
        if _NUMERIC_KEY.match(text):
            number = float(text)
            if math.isfinite(number):
                _put(out, ("i", int(number)) if number.is_integer() else ("f", number))
            return
        if text.isalpha() and len(text) >= _LONG_LETTERS:
            for start in range(len(text) - _GRAM + 1):
                _put(out, ("g", text[start:start + _GRAM]))
            return
        _put(out, ("s", text.lower()))
        return
    if isinstance(value, Mapping):
        for key, item in list(value.items())[:_MAX_ITEMS]:
            if isinstance(key, (int, float)) and not isinstance(key, bool):
                _items(key, out, depth + 1)
            elif isinstance(key, str) and _NUMERIC_KEY.match(key.strip()):
                _items(key, out, depth + 1)
            _items(item, out, depth + 1)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for item in list(value)[:_MAX_ITEMS]:
            _items(item, out, depth + 1)


def shard_items(shard: Any) -> set[tuple[str, Any]]:
    """Distinct raw items of one agent's input shard (answer-free)."""

    out: set[tuple[str, Any]] = set()
    _items(shard, out)
    return out


def shard_item_counts(shard: Any) -> Counter:
    """Raw items of one agent's input shard WITH their multiplicity
    (``--preserve-dups``): a number or whole-token string held twice counts
    2; the 4-grams of a long letter-only string count once each (they stand
    for one string, not for repeated values)."""

    out: Counter = Counter()
    _items(shard, out)
    for item in [i for i in out if i[0] == "g"]:
        out[item] = 1
    return out


def _number_keys(kind: str, value: Any) -> list[str]:
    """Text forms under which a number counts as present: an int with and
    without ``.0``; a float as its repr and rounded or truncated to 8 .. 1
    decimals."""
    if kind == "i":
        return [str(int(value)), f"{int(value)}.0"]
    x = float(value)
    keys = [repr(x)]
    for digits in range(8, 0, -1):
        scale = 10 ** digits
        keys.append(f"{x:.{digits}f}")
        keys.append(f"{math.trunc(x * scale) / scale:.{digits}f}")
    return keys


class _Bodies:
    """Presence index over the concatenated bodies one agent read."""

    def __init__(self, bodies: Iterable[str]) -> None:
        text = "\n".join(bodies)
        self.numbers = set()
        for token in _NUM_TOKEN.findall(text):
            if token in ("-0", "-0.0"):
                token = token[1:]
            self.numbers.add(token)
        self.lower = text.lower()
        self.grams = {text[i:i + _GRAM] for i in range(max(0, len(text) - _GRAM + 1))}
        self._strings: dict[str, bool] = {}

    def has(self, item: tuple[str, Any]) -> bool:
        kind, value = item
        if kind in ("i", "f"):
            return any(key in self.numbers for key in _number_keys(kind, value))
        if kind == "g":
            return value in self.grams
        if kind == "s":
            hit = self._strings.get(value)
            if hit is None:
                rx = re.compile(
                    r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])"
                )
                hit = bool(rx.search(self.lower))
                self._strings[value] = hit
            return hit
        return False


def _retention(
    agent: int,
    items: list[set[tuple[str, Any]]],
    bodies: list[str],
) -> float | None:
    """Fraction (3 decimals) of the other agents' distinct shard items that
    occur in ``bodies``; None when the other agents hold no items."""
    total = 0
    kept = 0
    index = _Bodies(bodies)
    for src, src_items in enumerate(items):
        if src == agent or not src_items:
            continue
        total += len(src_items)
        kept += sum(1 for item in src_items if index.has(item))
    if total == 0:
        return None
    return round(kept / total, 3)


#: Environment switch ("1" = on) of the ``--preserve-dups`` multiset
#: counting; process-wide like the diagnosis options, because credit is
#: computed inside the executors.  Off: retention counts distinct items and
#: credit carries no ``dup_lost``.
MULTISET_ENV = "QB_CREDIT_MULTISET"
#: ``dup_lost`` counts are clipped here (rendered numbers stay <= 3 chars).
DUP_LOST_MAX = 999


def multiset_enabled() -> bool:
    return os.environ.get(MULTISET_ENV, "") == "1"


class _BodyCounts(_Bodies):
    """Occurrence counts of raw items in ONE message body (multiset mode)."""

    def __init__(self, body: str) -> None:
        super().__init__([body])
        self.number_counts: Counter = Counter(
            token[1:] if token in ("-0", "-0.0") else token for token in _NUM_TOKEN.findall(body)
        )

    def count(self, item: tuple[str, Any]) -> int:
        kind, value = item
        if kind in ("i", "f"):
            return sum(self.number_counts[key] for key in dict.fromkeys(_number_keys(kind, value)))
        if kind == "g":
            return int(value in self.grams)
        if kind == "s":
            return len(_token_rx(value).findall(self.lower))
        return 0


@functools.lru_cache(maxsize=4096)
def _token_rx(value: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])")


def _sender_counts(bodies: Sequence[str], need: Mapping[tuple[str, Any], int]) -> Counter:
    """What ONE direct sender delivered of the needed items (multiset mode):
    the item counts summed over its MAXIMAL bodies.  A body whose counts
    another body of the same sender covers item by item is a cumulative
    re-send (a relay re-sending its growing list each round, or the same
    list again) and is dropped; bodies that each carry something the others
    lack are distinct pieces (pipelined forwarding: its own shard one round,
    the shard it received the next) and add up."""

    vecs: list[dict[tuple[str, Any], int]] = []
    for body in bodies:
        index = _BodyCounts(body)
        vec = {item: n for item in need if (n := index.count(item)) > 0}
        if vec:
            vecs.append(vec)
    # largest first (stable: equal totals keep the delivery order), so a body
    # can only be covered by one already kept; an identical re-send is covered
    vecs.sort(key=lambda v: -sum(v.values()))
    kept: list[dict[tuple[str, Any], int]] = []
    for vec in vecs:
        if any(all(n <= big.get(item, 0) for item, n in vec.items()) for big in kept):
            continue
        kept.append(vec)
    out: Counter = Counter()
    for vec in kept:
        out.update(vec)
    return out


def _retention_multiset(
    agent: int,
    counts: list[Counter],
    bodies_by_sender: Mapping[int, list[str]],
) -> tuple[float | None, int | None]:
    """(retention, dup_lost) of one agent with multiplicities.

    Needed = the other agents' shard items with multiplicity (a 4-gram of a
    long letter-only string is needed once in all: it stands for a string,
    and two strings that share a 4-gram are no repeated value).  Available
    = per direct sender, :func:`_sender_counts` (cumulative re-sends are not
    counted twice, pipelined pieces add up), summed over senders (a mesh
    delivers each holder's copy in its own body).  retention =
    sum(min(needed, available)) / sum(needed); dup_lost = the occurrences
    missing of values that DID arrive at least once (repeated values dropped
    on the way, e.g. a relay that read "append new items" as set semantics
    -- or items replaced by a summary that names each value once)."""

    need: Counter = Counter()
    for src, src_counts in enumerate(counts):
        if src == agent or not src_counts:
            continue
        need.update(src_counts)
    for item in [i for i in need if i[0] == "g"]:
        need[item] = 1
    total = sum(need.values())
    if total == 0:
        return None, None
    avail: Counter = Counter()
    for sender in sorted(bodies_by_sender):
        avail.update(_sender_counts(bodies_by_sender[sender], need))
    kept = sum(min(n, avail[item]) for item, n in need.items())
    lost = sum(n - avail[item] for item, n in need.items() if 0 < avail[item] < n)
    return round(kept / total, 3), int(lost)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


#: Why a run has no credit (closed set; ``row["credit_error"]``).
CREDIT_ERRORS: tuple[str, ...] = (
    "no_output", "no_agents", "no_submit_round", "exception",
)


def _credit_or_reason(
    output: Any,
    instance: Any,
    source: str | None,
    *,
    calls: Any,
    goal: str,
    selected_primary: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """``(credit, None)``, or ``(None, reason)`` with a reason from
    :data:`CREDIT_ERRORS`; never raises.  Under the ``all_agents`` goal
    every agent reads its submit-round inbox; under any other goal only
    ``selected_primary`` (the sink) does."""
    try:
        data = _dump(output)
        if data is None:
            return None, "no_output"
        n_agents = _int(getattr(instance, "n_agents", None)) or len(
            data.get("submissions") or []
        )
        if not n_agents or n_agents < 1:
            return None, "no_agents"
        submit_round = _submit_round(data)
        if submit_round is None:
            return None, "no_submit_round"
        messages = _messages(data)
        called, basis = _called_set(
            messages, n_agents=n_agents, submit_round=submit_round,
            source=source, calls=calls, goal=goal,
        )
        submitters = (
            range(n_agents) if goal == "all_agents" else [int(selected_primary)]
        )
        sets, read = _readable_sets(
            messages, called, n_agents=n_agents, submit_round=submit_round,
            submitters=submitters,
        )
        shards = list(getattr(instance, "shards", None) or [])
        if multiset_enabled():  # --preserve-dups: multiset retention + dup_lost
            counts = [shard_item_counts(shards[a]) if a < len(shards) else Counter()
                      for a in range(n_agents)]
            pairs = []
            for agent in range(n_agents):
                by_sender: dict[int, list[str]] = defaultdict(list)
                for i in read.get(agent, []):
                    by_sender[messages[i]["src"]].append(messages[i]["body"])
                pairs.append(_retention_multiset(agent, counts, by_sender))
            return {
                "readable": [len(s) for s in sets],
                "retention": [r for r, _lost in pairs],
                "dup_lost": [lost for _r, lost in pairs],
                "basis": basis,
            }, None
        items = [shard_items(shards[a]) if a < len(shards) else set()
                 for a in range(n_agents)]
        retention = [
            _retention(agent, items, [messages[i]["body"] for i in read.get(agent, [])])
            for agent in range(n_agents)
        ]
        return {
            "readable": [len(s) for s in sets],
            "retention": retention,
            "basis": basis,
        }, None
    except Exception:  # noqa: BLE001 - credit must never break a run
        return None, "exception"


def credit_fields(
    output: Any,
    instance: Any,
    source: str | None = None,
    *,
    calls: Any = None,
    goal: str = "all_agents",
    selected_primary: int = 0,
) -> dict[str, Any] | None:
    """``{"readable": [int per agent], "retention": [float|None per agent],
    "basis": "ledger"|"sim"|"trace"}`` of one executed run (plus
    ``"dup_lost": [int|None per agent]`` under :data:`MULTISET_ENV`), or
    None when no credit can be computed (no output, no agents, no submit
    round, or an internal error).  A failure is NEVER disguised as an empty
    credit, so counting rows whose credit is not None measures true
    coverage.

    ``output``: the program output (``PythonProgramOutput`` or its dict dump,
    e.g. a ``QB_TRACE_DIR`` record's ``output``); ``instance``: the benchmark
    instance (only ``case_id``, for the TEST guard, ``n_agents`` and
    ``shards`` are read); ``source``: the executed program's source, used to
    recover the called set (including sends to ``[]`` that leave no
    message); ``calls``: the runtime call ledger or its ``calls`` list
    (authoritative when given).  Raises only the TEST guard."""

    assert_not_test([getattr(instance, "case_id", "")], where="credit_fields")
    credit, _reason = _credit_or_reason(
        output, instance, source, calls=calls, goal=goal,
        selected_primary=selected_primary,
    )
    return credit


def sanitize_credit(credit: Any, *, n_agents: int | None = None) -> dict[str, Any] | None:
    """Closed whitelist of a credit dict (None when not a mapping).

    ``readable``: list of ints in ``[0, n_agents]``, else None per entry
    (the bound is 10000 without a positive ``n_agents``); ``retention``:
    list of floats in ``[0, 1]`` rounded to 3 decimals (else None per entry);
    ``dup_lost`` (only when present): list of non-negative ints clipped to
    ``DUP_LOST_MAX`` (else None per entry); ``basis``: one of
    :data:`CREDIT_BASES` or dropped.  A ``readable`` / ``retention`` value
    that is not a list of at most 128 entries becomes ``[]``; such a
    ``dup_lost`` is left out."""

    if not isinstance(credit, Mapping):
        return None
    cap = n_agents if isinstance(n_agents, int) and n_agents > 0 else 10_000
    out: dict[str, Any] = {}
    readable = credit.get("readable")
    if isinstance(readable, (list, tuple)) and len(readable) <= 128:
        out["readable"] = [
            int(v) if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= cap
            else None
            for v in readable
        ]
    else:
        out["readable"] = []
    retention = credit.get("retention")
    if isinstance(retention, (list, tuple)) and len(retention) <= 128:
        out["retention"] = [
            round(float(v), 3)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0
            else None
            for v in retention
        ]
    else:
        out["retention"] = []
    dup_lost = credit.get("dup_lost")
    if isinstance(dup_lost, (list, tuple)) and len(dup_lost) <= 128:  # multiset mode only
        out["dup_lost"] = [
            min(int(v), DUP_LOST_MAX)
            if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
            for v in dup_lost
        ]
    basis = credit.get("basis")
    if basis in CREDIT_BASES:
        out["basis"] = basis
    return out


def attach_credit(
    row: dict[str, Any],
    output: Any,
    instance: Any,
    source: str | None = None,
    *,
    calls: Any = None,
    goal: str = "all_agents",
    selected_primary: int = 0,
) -> dict[str, Any]:
    """``row["credit"] = sanitize_credit(credit_fields(...))`` in place;
    returns ``row``.  When no credit can be computed, ``row["credit"]`` is
    None and ``row["credit_error"]`` names why (one of
    :data:`CREDIT_ERRORS`).  Never raises except the TEST guard."""

    assert_not_test([getattr(instance, "case_id", "")], where="attach_credit")
    credit, reason = _credit_or_reason(
        output, instance, source, calls=calls, goal=goal,
        selected_primary=selected_primary,
    )
    row["credit"] = sanitize_credit(
        credit, n_agents=_int(getattr(instance, "n_agents", None))
    )
    if row["credit"] is None:
        row["credit_error"] = reason if reason in CREDIT_ERRORS else "exception"
    else:
        row.pop("credit_error", None)
    return row


# --------------------------------------------------------------------------- #
# Rendering: the diagnosis-card extension
# --------------------------------------------------------------------------- #

CREDIT_LEGEND = (
    "readable=[..]/n: per agent, how many agents' shard content it had read "
    "by submit time, directly or relayed (a body delivered to an agent that "
    "is not called that round is not read). raw_kept%=[..] (runs with S<1 "
    "only): per agent, the percentage of the other agents' distinct raw "
    "shard items (numbers, strings) that occur verbatim in the message "
    "bodies it read; it counts raw items only, so a body that carries a "
    "computed result instead of the items scores low even when nothing the "
    "task needs was lost. Token and message counts of 1000 or more are "
    "shown in thousands (1.3k)."
)

#: The passage of :data:`CREDIT_LEGEND` that the multiset legend replaces.
_LEGEND_DISTINCT = (
    "the percentage of the other agents' distinct raw "
    "shard items (numbers, strings) that occur verbatim in the message "
    "bodies it read;"
)
#: The legend under ``MULTISET_ENV``: raw items with multiplicity and the
#: ``dup_lost`` field.
CREDIT_LEGEND_MULTISET = CREDIT_LEGEND.replace(
    _LEGEND_DISTINCT,
    "the percentage of the other agents' raw shard items (numbers, strings) "
    "that occur verbatim in the message bodies it read, counted WITH "
    "multiplicity (a number or short string two agents hold, or one agent "
    "holds twice, must arrive twice; a long letter-only string counts once);",
) + (
    " dup_lost=[..] (runs with S<1 only, shown when not all 0): per agent, "
    "how many occurrences of raw values are missing although the value itself "
    "arrived, i.e. the bodies carry the value fewer times than the shards hold "
    "it (repeated values dropped on the way, or items replaced by a summary "
    "that names each value once)."
)


def credit_legend() -> str:
    """:data:`CREDIT_LEGEND`, or :data:`CREDIT_LEGEND_MULTISET` under
    ``MULTISET_ENV`` (without ``--preserve-dups`` prompts keep the plain
    legend)."""

    return CREDIT_LEGEND_MULTISET if multiset_enabled() else CREDIT_LEGEND


_RUN_LINE = re.compile(r"^  run(\d+):")
_PHASE_TOKENS = re.compile(r"tokens/phase=\[([^\]]*)\]")
_BIG_KEYED_INT = re.compile(r"\b(msgs|lost_bodies)=(\d{4,})(?![\w.])")
_BIG_INT = re.compile(r"(?<![\w.])(\d{4,})(?![\w.])")


def _k(value: int) -> str:
    """Counts >= 1000 in thousands (``1331 -> 1.3k``).

    Leak safety: the prompt's value audit matches forbidden forms of >= 4
    characters as whole tokens, so a raw 4-digit count can coincide with a
    ground-truth number; ``1.3k`` never can (the trailing letter ends no
    number token) and counts below 1000 are too short to be audited."""

    value = int(value)
    return str(value) if abs(value) < 1000 else f"{value / 1000:.1f}k"


def _safe_counts(line: str) -> str:
    """A diagnosis run line with its per-phase token counts and its
    ``msgs`` / ``lost_bodies`` counts in :func:`_k` form (``agents_right``
    bits and every other field untouched)."""

    def _tokens(match: re.Match) -> str:
        inner = _BIG_INT.sub(lambda m: _k(int(m.group(1))), match.group(1))
        return f"tokens/phase=[{inner}]"

    line = _PHASE_TOKENS.sub(_tokens, line)
    return _BIG_KEYED_INT.sub(lambda m: f"{m.group(1)}={_k(int(m.group(2)))}", line)


def render_credit_suffix(
    credit: Any, *, n_agents: int | None = None, show_retention: bool = True
) -> str:
    """`` readable=[5 5 3 5 5]/5 raw_kept%=[27 100 - 50 100]`` (empty when
    there is no credit), plus ``dup_lost=[..]`` when any count is non-zero.
    ``raw_kept%`` and ``dup_lost`` appear only with ``show_retention``.

    Every rendered number has at most 3 characters and the lists are
    space-separated (never a JSON list), so no credit field can coincide
    with an audited ground-truth form (>= 4 characters, JSON lists)."""

    clean = sanitize_credit(credit, n_agents=n_agents)
    if not clean:
        return ""
    parts = []
    readable = clean.get("readable") or []
    if readable:
        n = n_agents if isinstance(n_agents, int) and n_agents > 0 else len(readable)
        parts.append(
            "readable=[" + " ".join("?" if v is None else str(v) for v in readable)
            + f"]/{n}"
        )
    retention = clean.get("retention") or []
    if show_retention and retention and any(v is not None for v in retention):
        parts.append(
            "raw_kept%=["
            + " ".join("-" if v is None else str(int(round(v * 100))) for v in retention)
            + "]"
        )
    dup_lost = clean.get("dup_lost") or []  # multiset rows only
    if show_retention and any(v for v in dup_lost):
        parts.append(
            "dup_lost=[" + " ".join("-" if v is None else str(v) for v in dup_lost) + "]"
        )
    return (" " + " ".join(parts)) if parts else ""


def _row_n_agents(row: Mapping[str, Any]) -> int | None:
    credit = row.get("credit")
    if isinstance(credit, Mapping) and isinstance(credit.get("readable"), list):
        return len(credit["readable"]) or None
    return None


def _row_s(row: Mapping[str, Any]) -> float | None:
    for value in (row.get("S"), (row.get("diag") or {}).get("S")
                  if isinstance(row.get("diag"), Mapping) else None):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def render_credit_card(unit_id: str, rows: Iterable[Any]) -> str:
    """``diagnosis.render_diag_card`` with each run line extended by its
    credit fields (rows carry ``diag`` and ``credit``).

    Retention (``raw_kept%``) is shown only on runs with S < 1: on solved
    runs a low raw-item count is expected whenever bodies carry computed
    results, so it carries no failure signal there.  Token, message and
    lost-body counts >= 1000 are rewritten in thousands (see :func:`_k`)."""

    from queenbee.evo.diagnosis import render_diag_card

    rows = list(rows or [])
    block = render_diag_card(unit_id, rows)
    # render_diag_card keeps exactly the Mapping rows, in order.
    kept = [row for row in rows if isinstance(row, Mapping)]
    lines = block.split("\n")
    # The head line must carry the full unit id: a head that shows it without
    # '@' (II-11@x5 -> [II-11x5]) gets the '@' back.  diagnosis.clean_unit_id
    # keeps '@', so the head normally matches already and stays unchanged.
    shown = re.sub(r"[^A-Za-z0-9_\-]", "", str(unit_id))[:32] or "?"
    wanted = re.sub(r"[^A-Za-z0-9_@\-]", "", str(unit_id))[:32] or "?"
    if lines and lines[0].startswith(f"[{shown}]"):
        lines[0] = f"[{wanted}]" + lines[0][len(shown) + 2:]
    for position, line in enumerate(lines):
        match = _RUN_LINE.match(line)
        if not match:
            continue
        line = _safe_counts(line)
        index = int(match.group(1)) - 1
        if 0 <= index < len(kept):
            row = kept[index]
            s_value = _row_s(row)
            line += render_credit_suffix(
                row.get("credit"), n_agents=_row_n_agents(row),
                show_retention=s_value is not None and s_value < 1.0 - 1e-9,
            )
        lines[position] = line
    return "\n".join(lines)


def render_diag_cards_with_credit(
    rows_by_unit: Mapping[str, Iterable[Any]],
    *,
    max_chars: int = 3000,
    order: Iterable[str] | None = None,
) -> str:
    """Whole-unit blocks of :func:`render_credit_card` within ``max_chars``
    (``order``: units to show first; the rest follow sorted)."""

    first = [u for u in (order or []) if u in rows_by_unit]
    units = first + sorted(u for u in rows_by_unit if u not in set(first))
    text = ""
    for index, unit_id in enumerate(units):
        block = render_credit_card(unit_id, rows_by_unit[unit_id]) + "\n"
        if len(text) + len(block) > max_chars:
            text += f"... +{len(units) - index} more units\n"
            break
        text += block
    return text


# --------------------------------------------------------------------------- #
# Leak audit + trace helpers
# --------------------------------------------------------------------------- #


def forbidden_values_of(instance: Any) -> list[Any]:
    """The ground-truth / expected-output values of an instance (to audit
    rendered text against; never rendered themselves)."""

    values: list[Any] = []
    meta = getattr(instance, "meta", None) or {}
    if isinstance(meta, Mapping):
        values.extend(v for v in (meta.get("expected_outputs") or []) if v is not None)
    truth = getattr(instance, "ground_truth", None)
    if truth is not None:
        values.append(truth)
    return values


def audit_credit_text(
    text: str, forbidden_values: Any, forbidden_case_ids: Any = (),
    *, min_value_chars: int = 4,
) -> list[dict[str, Any]]:
    """Leak hits of ``text`` (:func:`queenbee.program.mint.audit_text_leaks`);
    TEST ids are always forbidden.  Empty list = clean."""

    from queenbee.program.mint import audit_text_leaks

    ids = sorted(set(sealed_case_ids()) | {str(c) for c in forbidden_case_ids or []})
    return audit_text_leaks(
        text, ids, forbidden_values=forbidden_values,
        min_value_chars=min_value_chars,
    )


def load_dev_instance(
    case_id: str, n_agents: int = 5, *, benchmarks_dir: str | Path | None = None
) -> Any:
    """One Silo-Bench development instance, loaded through
    :class:`queenbee.bench.silo_bench.SiloBenchAdapter` (which strips the
    benchmark's topology hints); a TEST case is refused."""

    assert_not_test([case_id], where="load_dev_instance")
    from queenbee.bench.silo_bench import SiloBenchAdapter
    from queenbee.paths import default_benchmarks_dir

    bench = Path(benchmarks_dir) if benchmarks_dir else default_benchmarks_dir()
    for inst in SiloBenchAdapter(bench).iter_instances(
        cases=[case_id], agent_counts=[int(n_agents)]
    ):
        return inst
    raise FileNotFoundError(f"no instance {case_id} n={n_agents} in {bench}")


def load_trace(path: str | Path) -> dict[str, Any]:
    """A ``QB_TRACE_DIR`` record (written by
    :func:`queenbee.program.execute._maybe_dump_trace`); a TEST case id or
    file name is refused."""

    record = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_not_test([record.get("case_id"), Path(path).name], where="load_trace")
    return record


def trace_row(
    trace: Mapping[str, Any],
    instance: Any,
    *,
    source: str | None = None,
    goal: str = "all_agents",
) -> dict[str, Any]:
    """A trace record -> ExecRow-shaped dict with ``diag`` and ``credit``,
    then the active task's ``trace_row`` hook (when it sets one).

    The trace's ``ground_truth`` is never copied into the row."""

    assert_not_test(
        [trace.get("case_id"), getattr(instance, "case_id", "")], where="trace_row"
    )
    from queenbee.evo.diagnosis import diagnose_execution, refine_card

    facts = dict(trace.get("facts") or {})
    row = {k: v for k, v in facts.items() if k != "diag"}
    row["case_id"] = trace.get("case_id")
    output = trace.get("output")
    diag = facts.get("diag")
    if not isinstance(diag, Mapping):
        diag = diagnose_execution(output, instance, facts, source=source, goal=goal)
    row["diag"] = refine_card(dict(diag), output, instance)
    attach_credit(row, output, instance, source, goal=goal)
    hook = get_task().trace_row
    if hook is not None:
        row = dict(hook(row, trace, instance, source=source, goal=goal))
    return row


__all__ = [
    "CREDIT_BASES",
    "CREDIT_ERRORS",
    "CREDIT_LEGEND",
    "CREDIT_LEGEND_MULTISET",
    "MULTISET_ENV",
    "TEST_CASE_IDS",
    "assert_not_test",
    "attach_credit",
    "audit_credit_text",
    "credit_fields",
    "credit_legend",
    "forbidden_values_of",
    "load_dev_instance",
    "load_trace",
    "render_credit_card",
    "render_credit_suffix",
    "render_diag_cards_with_credit",
    "multiset_enabled",
    "sanitize_credit",
    "sealed_case_ids",
    "shard_item_counts",
    "shard_items",
    "simulate_sends",
    "templates_in",
    "trace_row",
]
