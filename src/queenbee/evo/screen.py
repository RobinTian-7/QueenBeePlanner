"""QueenBee-Evo zero-cost screen S0: static checks that every proposed team
program must pass before any worker call is spent on it.

Readable coverage
-----------------
Counting source **ids** overstates what an agent knows: an id is merged into
an agent's ``source_ids`` whenever a message is *delivered*, whether or not
the agent ever reads the body.  The fixed ``message_only_v2`` runtime (the
program's ``main()``, as in ``exp_graph.mas.python_code``, plus the worker
bootstrap) only shows a delivered body to its recipient when

* the recipient is called (``mode == "send"``, including ``send`` to ``[]``,
  the digest primitive) in the round of delivery, or
* the round of delivery is the submit barrier (every submitter reads the
  inbox delivered at the barrier).

``reflect`` and ``idle`` are free and read nothing; a body delivered to an
agent that is not called that round is gone (its ids stay counted).  An
agent's memory is its own previous output, so what it has *read* is carried
forward in what it writes.

:func:`simulate_readable_coverage` replays ``main()`` exactly -- the same
per-round snapshot of source ids drives ``known_source_count`` /
``inbox_count`` into the program's own ``plan_communication_turn``; the same
action checks (modes, recipients, 800-char work instructions) and call /
message budgets raise the same errors -- and tracks, next to the id sets, a
*readable* set per agent: the shards whose content can have reached the
agent through bodies it actually read.  A message carries its sender's
readable set after the sender read its own inbox that round.  The result
gives per-agent readable and id coverage at the barrier, the lost-edge count
(equal to ``diagnosis._lost_messages`` under ``all_agents``; under ``sink``
a body delivered at the barrier to an agent that does not submit counts as
lost too), predicted worker calls (sends + submits) and messages, and every
work instruction the program emits.  A program executes its own
``main()``; :func:`main_semantics` recognises the fixed scaffold
(:data:`MAIN_V2`; every evolved program carries it, since the genome region
ends at ``def main``), and an unrecognised ``main()`` is a hard reject (it
cannot be simulated).  The tests check the simulator against each
program's real ``main()``, run in-process with a content-tracking stub
worker whose submitted answers are its read sets.

S0 checks
---------
The checks run in this order; ``ScreenResult.reasons`` collects the hard
rejections and ``ScreenResult.penalties`` the ranking penalties.

1. static validation (``validate_python_source``; the dry run is part of
   building the program in ``queenbee.program.mint``) and the ``main()``
   variant (unrecognised: reject);
2. simulation and readable coverage at every n in ``n_list``.  A program
   that fails to simulate is rejected (``runtime@n``: the error ``main()``
   would raise, or a crash or overrun of the policy code).  Every
   submitting agent must have read n shards at submit (every agent under
   the ``all_agents`` goal, only agent 0 under ``sink``).
   ``unread-content@n``: an agent submits with ids whose bodies it never
   read (readable < known: the counted-but-unread trap); ``coverage@n``:
   an agent read fewer than n shards with nothing unread (it never hears
   from some peers).  Both are hard.  A variant of the seed program (v1)
   that leaves agent 0 out of the final broadcast (readable = known =
   [1, 5, 5, 5, 5]) is therefore rejected for coverage only, which is
   correct: agent 0 is blind by design.  ``strict_coverage=False`` relaxes
   the rule: one short agent is a penalty, two or more stay hard (a
   near-silent program never passes S0);
3. lost edges: counted, penalty only;
4. predicted calls: penalty above 2x v1, reject above ``calls_reject_ratio``
   (3) x v1 (n=5: 20 / 30; n=10: 40 / 60; ``None``: penalty only);
5. work instructions: one longer than 800 characters rejects; genericity
   lint: a vocabulary extracted from the task names, titles and Output
   sentences of the 18 development templates (never TEST), minus a generic
   stoplist; any hit rejects;
6. behaviour-fingerprint duplicate (``known_fps``, e.g. from
   :func:`known_fps_for`): the program is not executed and the rows of
   ``dup_of`` are reused.  The fingerprint is the simulated schedule +
   barrier + ``main()`` variant + :func:`host_digest` (the globals
   ``main()`` reads, e.g. ``MESSAGE_INSTRUCTION``);
7. tabu: the proposal repeats a (failure class, mechanism) pair that the
   hypothesis ledger has refuted at least twice (checked only when the
   ledger is on).

The team sizes, call caps and vocabulary above are the Silo-Bench
defaults.  :func:`screen_for_task` screens under the active task's goal
and, when the task sets them, its own team sizes, seed program (the
call-cap reference) and vocabulary (:func:`task_screen_policy`); a task's
own ``screen_program`` hook takes its place.

Policy code runs under a wall-clock deadline (``timeout_s``, default
``DEFAULT_POLICY_TIMEOUT_S``): an infinite loop is a ``runtime@n`` reject,
not a hang.

Main entry points: :func:`screen_program` and :func:`screen_for_task` (the
active task's policy, :func:`task_screen_policy`),
:func:`simulate_readable_coverage`, :class:`ScreenResult`,
:func:`behavior_fp`, :func:`fp_key`, :func:`program_fp_key`,
:func:`fp_key_for_task`, :func:`known_fps_for`, :func:`host_digest`,
:func:`readable_from_messages`, :func:`sim_call_ledger`,
:func:`crosscheck_trace`, :func:`build_wi_vocabulary`,
:func:`lint_work_instructions`, :func:`v1_reference_calls`.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import ctypes
import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from queenbee.evo.common import task_goal
from queenbee.evo.ladder import DEV_TEMPLATE_IDS, assert_silo_dev_template
from queenbee.paths import default_benchmarks_dir
from queenbee.tasks import get_task

DEFAULT_MAX_ROUNDS = 64  # default --python-max-rounds: round cap of a team program
DEFAULT_N_LIST: tuple[int, ...] = (5, 10)
DEFAULT_GOAL = "all_agents"
WI_MAX_CHARS = 800
#: ``main()`` variants (``main_semantics``): the fixed message_only_v2
#: scaffold, and anything else.
MAIN_V2, MAIN_UNKNOWN = "v2", "unknown"
CALLS_PENALTY_RATIO = 2.0
CALLS_REJECT_RATIO = 3.0
#: With ``strict_coverage=False``, at most this many agents may read fewer
#: than n shards before coverage is a hard reject again.
LENIENT_MAX_SHORT_AGENTS = 1

# --------------------------------------------------------------------------- #
# Loading the policy section (never the program entrypoint)
# --------------------------------------------------------------------------- #


class PolicyLoadError(RuntimeError):
    """The program's policy functions could not be defined."""


def _is_main_call(node: ast.stmt) -> bool:
    value = node.value if isinstance(node, ast.Expr) else None
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == "main")


def _is_docstring(node: ast.stmt) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _silent_print(*_args: Any, **_kwargs: Any) -> None:
    return None


def load_policy(source: str) -> dict[str, Any]:
    """Namespace of the program's module body minus the entrypoint.

    Dropped: top-level imports (the validator allows only ``json`` /
    ``sys`` / the client factory; ``json`` is provided), the ``main``
    function, the bare ``main()`` call and docstrings -- so no stdin read, no
    client and no network ever happens.  Everything else runs exactly as the
    runtime would run it before ``main()``: a top-level ``if`` block or
    expression that rebinds ``PHASES`` (``if True: PHASES = [...]``,
    ``PHASES.append(...)``) is simulated, not silently skipped (the module
    ``__name__`` is not ``"__main__"``, so an ``if __name__ == "__main__":``
    entry guard stays inert)."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PolicyLoadError(f"SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    keep: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or _is_main_call(node) or _is_docstring(node):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main":
            continue
        keep.append(node)
    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "__name__": "queenbee_evo_screen_policy",
        "json": json,
        "print": _silent_print,  # never write to the host's stdout
    }
    try:
        exec(compile(ast.Module(body=keep, type_ignores=[]), "<evo-policy>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise PolicyLoadError(f"{type(exc).__name__}: {exc}") from exc
    for name in ("plan_submit_round", "plan_communication_turn"):
        if not callable(namespace.get(name)):
            raise PolicyLoadError(f"program defines no {name}()")
    return namespace


def _main_tail(source: str) -> str | None:
    lines = str(source).strip().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("def main")), None)
    if start is None:
        return None
    return "\n".join(line.rstrip() for line in lines[start:]).strip()


@lru_cache(maxsize=1)
def _canonical_tails() -> dict[str, str]:
    from exp_graph.mas.python_code import DEFAULT_MESSAGE_ONLY_V2_PROGRAM

    return {MAIN_V2: _main_tail(DEFAULT_MESSAGE_ONLY_V2_PROGRAM) or ""}


def main_semantics(source: str) -> str:
    """Which runtime ``main()`` the program carries (it executes its own).

    :data:`MAIN_V2` for the fixed message_only_v2 scaffold, which every
    evolved program carries: the planner writes only the genome region,
    which ends at ``def main``, and the host splices it into the parent
    program.  Anything else is :data:`MAIN_UNKNOWN` (the simulator cannot
    vouch for it)."""

    tail = _main_tail(source)
    if tail is None:
        return MAIN_UNKNOWN
    for name, text in _canonical_tails().items():
        if tail == text:
            return name
    return MAIN_UNKNOWN


# --------------------------------------------------------------------------- #
# Readable-coverage simulator (mirrors main())
# --------------------------------------------------------------------------- #


@dataclass
class ReadableSim:
    """One simulated execution of a program's communication schedule."""

    n_agents: int
    max_rounds: int
    goal: str
    semantics: str = MAIN_V2
    selected_primary: int = 0
    submit_round: int | None = None
    readable: list[int] = field(default_factory=list)
    readable_sets: list[list[int]] = field(default_factory=list)
    known: list[int] = field(default_factory=list)
    lost_edges: int = 0
    lost_by_agent: list[int] = field(default_factory=list)
    sends: int = 0
    submit_calls: int = 0
    messages: int = 0
    wis: list[str] = field(default_factory=list)
    rounds: list[list[dict[str, Any]]] = field(default_factory=list)
    message_log: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def calls(self) -> int:
        """Predicted worker calls: communication calls (``send``, incl. the
        digest send to ``[]``) + barrier submits."""

        return self.sends + self.submit_calls

    @property
    def unread(self) -> list[int]:
        """Per agent: ids counted at submit whose content was never read."""

        return [k - r for k, r in zip(self.known, self.readable)]

    def summary(self) -> dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "submit_round": self.submit_round,
            "readable": list(self.readable),
            "known": list(self.known),
            "lost_edges": self.lost_edges,
            "calls": self.calls,
            "sends": self.sends,
            "messages": self.messages,
            "n_wis": len(self.wis),
            "semantics": self.semantics,
            "error": self.error,
        }


def _default_budgets(max_rounds: int, n_agents: int) -> dict[str, int]:
    from queenbee.program.budgets import PythonRunBudgets

    b = PythonRunBudgets.for_rounds(max_rounds, n_agents=n_agents)
    return {"max_model_calls": int(b.max_model_calls), "max_messages": int(b.max_messages)}


class _SimError(Exception):
    pass


#: Wall-clock budget of one simulation (policy load + every policy call).
#: The seed program (v1) simulates in a few milliseconds at n = 10.
DEFAULT_POLICY_TIMEOUT_S = 10.0


class PolicyTimeout(BaseException):
    """Raised asynchronously into a simulation whose policy code overran its
    wall-clock budget (a BaseException, so the program's own ``except
    Exception`` cannot swallow it)."""


@contextlib.contextmanager
def _deadline(seconds: float | None) -> Iterator[None]:
    """Interrupt the current thread with :class:`PolicyTimeout` after
    ``seconds`` (``PyThreadState_SetAsyncExc``; works off the main thread,
    unlike ``SIGALRM``; re-armed every 0.2 s in case the program swallows
    it).  Pure-Python loops are interrupted; a single long C-level operation
    is not.  A pending, undelivered interrupt is cleared on exit."""

    set_async = getattr(getattr(ctypes, "pythonapi", None), "PyThreadState_SetAsyncExc", None)
    if not seconds or seconds <= 0 or set_async is None:
        yield
        return
    tid = ctypes.c_ulong(threading.get_ident())
    lock = threading.Lock()
    stop = threading.Event()
    state = {"done": False, "fired": False}

    def watchdog() -> None:
        if stop.wait(float(seconds)):
            return
        while True:
            with lock:
                if state["done"]:
                    return
                state["fired"] = True
                set_async(tid, ctypes.py_object(PolicyTimeout))
            if stop.wait(0.2):
                return

    thread = threading.Thread(target=watchdog, name="evo-screen-deadline", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        with lock:
            state["done"] = True
            if state["fired"]:
                set_async(tid, None)  # clear an interrupt not delivered yet


def simulate_readable_coverage(
    source: str,
    n_agents: int,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    *,
    goal: str = DEFAULT_GOAL,
    selected_primary: int = 0,
    budgets: Mapping[str, int] | None = None,
    timeout_s: float | None = DEFAULT_POLICY_TIMEOUT_S,
) -> ReadableSim:
    """Replay the program's schedule under ``main()`` semantics (see the
    module docstring).

    Never raises for a bad program: ``sim.error`` holds the runtime's error
    text (``DataFlowError`` / ``BudgetError`` as ``main()`` would raise it,
    ``PolicyError`` for a crash of the program's own policy code, or for
    policy code that runs longer than ``timeout_s`` seconds -- e.g. an
    infinite loop; ``None`` disables the deadline)."""

    n = int(n_agents)
    semantics = main_semantics(source)
    sim = ReadableSim(n_agents=n, max_rounds=int(max_rounds), goal=goal,
                      semantics=semantics,
                      selected_primary=int(selected_primary) % max(1, n))
    caps = dict(budgets) if budgets is not None else _default_budgets(int(max_rounds), n)
    try:
        with _deadline(timeout_s):
            ns = load_policy(source)
            _run(sim, ns, n, int(max_rounds), goal, int(selected_primary), caps)
    except PolicyLoadError as exc:
        sim.error = f"PolicyError: {exc}"
    except _SimError as exc:
        sim.error = str(exc)
    except PolicyTimeout:
        sim.error = f"PolicyError: policy code exceeded the {timeout_s:g}s screen deadline"
    return sim


def _run(
    sim: ReadableSim,
    ns: dict[str, Any],
    n: int,
    max_rounds: int,
    goal: str,
    primary: int,
    caps: Mapping[str, int],
) -> None:
    def policy(fn: str, *args: Any) -> Any:
        try:
            return ns[fn](*args)
        except Exception as exc:  # noqa: BLE001 - a crash of the program itself
            raise _SimError(f"PolicyError: {fn} raised {type(exc).__name__}: {exc}") from exc

    try:
        submit = int(policy("plan_submit_round", n, max_rounds, goal))
    except (TypeError, ValueError) as exc:
        raise _SimError(f"PolicyError: plan_submit_round returned a non-int ({exc})") from exc
    sim.submit_round = submit
    if submit < 0 or submit >= max_rounds:
        raise _SimError("DataFlowError: submit round outside the round budget")

    known: list[set[int]] = [{a} for a in range(n)]
    readable: list[frozenset[int]] = [frozenset({a}) for a in range(n)]
    sim.lost_by_agent = [0] * n
    pending: list[dict[str, Any]] = []
    seen_wis: dict[str, None] = {}

    for r in range(max_rounds):
        if r >= submit:
            break
        inbox = [[m for m in pending if m["round_delivered"] == r and m["dst"] == a] for a in range(n)]
        snap_known = [set(known[a]) for a in range(n)]
        for a in range(n):
            for m in inbox[a]:
                snap_known[a].update(m["source_ids"])
        actions: list[dict[str, Any]] = []
        planned_calls = planned_sends = 0
        for a in range(n):
            raw = policy(
                "plan_communication_turn",
                r, a, n, goal, primary, len(snap_known[a]), len(inbox[a]),
            )
            action = _check_action(raw, a, n)
            actions.append(action)
            if action["mode"] == "send":
                planned_calls += 1
            planned_sends += len(action["recipients"])
        if sim.sends + planned_calls > int(caps["max_model_calls"]):
            raise _SimError("BudgetError: communication call budget exhausted")
        if sim.messages + planned_sends > int(caps["max_messages"]):
            raise _SimError("BudgetError: communication message budget exhausted")

        new_readable = list(readable)
        next_pending: list[dict[str, Any]] = []
        for a in range(n):
            action = actions[a]
            if action["mode"] != "send":
                sim.lost_edges += len(inbox[a])
                sim.lost_by_agent[a] += len(inbox[a])
                continue
            content = set(readable[a])
            for m in inbox[a]:
                content.update(m["content"])
            new_readable[a] = frozenset(content)
            sim.sends += 1
            if action.get("work_instruction"):
                seen_wis.setdefault(action["work_instruction"], None)
            for d in action["recipients"]:
                message = {
                    "round_sent": r,
                    "round_delivered": r + 1,
                    "src": a,
                    "dst": d,
                    "source_ids": sorted(snap_known[a]),
                    "content": new_readable[a],
                }
                next_pending.append(message)
                sim.messages += 1
                sim.message_log.append(
                    {k: (sorted(v) if k == "content" else v) for k, v in message.items()}
                )
        sim.rounds.append(actions)
        known = snap_known
        readable = new_readable
        pending = next_pending

    final_inbox = [[m for m in pending if m["round_delivered"] == submit and m["dst"] == a] for a in range(n)]
    submitters = list(range(n)) if goal == "all_agents" else [primary % n]
    if sim.sends + len(submitters) > int(caps["max_model_calls"]):
        raise _SimError("BudgetError: synchronized submit call budget exhausted")
    final_readable = list(readable)
    for a in range(n):
        for m in final_inbox[a]:
            known[a].update(m["source_ids"])
        reads = final_inbox[a]
        if a in submitters:
            content = set(readable[a])
            for m in reads:
                content.update(m["content"])
            final_readable[a] = frozenset(content)
        else:
            sim.lost_edges += len(reads)
            sim.lost_by_agent[a] += len(reads)
    sim.submit_calls = len(submitters)
    sim.known = [len(k) for k in known]
    sim.readable = [len(s) for s in final_readable]
    sim.readable_sets = [sorted(s) for s in final_readable]
    sim.wis = list(seen_wis)


def _check_action(raw: Any, agent_id: int, n: int) -> dict[str, Any]:
    """main()'s per-action checks, same error texts; returns the control
    action a worker would see (``mode``, ``recipients``, optional
    ``work_instruction``)."""

    try:
        mode = str(raw["mode"])
        recipients = [int(x) for x in raw["recipients"]]
        wi = str(raw.get("work_instruction") or "")
    except Exception as exc:  # noqa: BLE001 - main() would crash the same way
        raise _SimError(f"PolicyError: malformed action {raw!r:.120} ({type(exc).__name__})") from exc
    if mode not in ("send", "reflect", "idle"):
        raise _SimError("DataFlowError: communication policy cannot submit")
    if mode != "send" and recipients:
        raise _SimError("DataFlowError: recipients require send mode")
    if len(set(recipients)) != len(recipients):
        raise _SimError("DataFlowError: duplicate recipients")
    for d in recipients:
        if d < 0 or d >= n or d == agent_id:
            raise _SimError("DataFlowError: recipient outside the agent range")
    if len(wi) > WI_MAX_CHARS:
        raise _SimError("DataFlowError: work_instruction exceeds 800 characters")
    control: dict[str, Any] = {"mode": mode, "recipients": recipients}
    if wi:
        control["work_instruction"] = wi
    return control


# --------------------------------------------------------------------------- #
# Traces: readable coverage from executed messages + cross-check
# --------------------------------------------------------------------------- #


def readable_from_messages(
    messages: Sequence[Mapping[str, Any]],
    n_agents: int,
    submit_round: int,
    *,
    calls: Sequence[Mapping[str, Any]] | None = None,
    goal: str = DEFAULT_GOAL,
    selected_primary: int = 0,
) -> dict[str, Any]:
    """Readable coverage recomputed from an executed run's messages.

    ``calls`` (the runtime call ledger, ``{"round", "agent_id", "mode"}``)
    gives the called set; without it the called set is inferred from the
    senders, exactly like ``diagnosis._lost_messages`` (a digest send to
    ``[]`` is then invisible)."""

    n = int(n_agents)
    called: set[tuple[int, int]] = set()
    if calls:
        for c in calls:
            if str(c.get("mode")) == "submit":
                continue
            called.add((int(c["round"]), int(c["agent_id"])))
    else:
        for m in messages:
            called.add((int(m["round_sent"]), int(m["src"])))
    readable: list[set[int]] = [{a} for a in range(n)]
    content: dict[int, frozenset[int]] = {}
    by_delivery: dict[int, list[int]] = {}
    by_sent: dict[int, list[int]] = {}
    for i, m in enumerate(messages):
        by_delivery.setdefault(int(m["round_delivered"]), []).append(i)
        by_sent.setdefault(int(m["round_sent"]), []).append(i)
    lost = 0
    for r in range(int(submit_round)):
        for i in by_delivery.get(r, []):
            dst = int(messages[i]["dst"])
            if (r, dst) in called:
                readable[dst] |= content.get(i, frozenset({int(messages[i]["src"])}))
            else:
                lost += 1
        for i in by_sent.get(r, []):
            content[i] = frozenset(readable[int(messages[i]["src"])])
    submitters = set(range(n)) if goal == "all_agents" else {int(selected_primary) % n}
    for i in by_delivery.get(int(submit_round), []):
        dst = int(messages[i]["dst"])
        if dst in submitters:
            readable[dst] |= content.get(i, frozenset({int(messages[i]["src"])}))
        else:
            lost += 1
    return {
        "readable": [len(s) for s in readable],
        "readable_sets": [sorted(s) for s in readable],
        "lost_edges": lost,
    }


def _trace_submit_round(output: Mapping[str, Any]) -> int | None:
    rounds = [
        s.get("submitted_round") for s in (output.get("submissions") or [])
        if isinstance(s, Mapping) and isinstance(s.get("submitted_round"), int)
    ]
    if rounds:
        return max(rounds)
    executed = output.get("rounds_executed")
    return int(executed) - 1 if isinstance(executed, int) and executed > 0 else None


def sim_call_ledger(sim: ReadableSim) -> list[dict[str, Any]]:
    """The runtime call ledger a simulated run implies (``{"round",
    "agent_id", "mode"}`` per worker call: every called communication action
    -- ``send``, incl. the digest send to ``[]`` -- then one ``submit`` per
    submitter).  The program is deterministic, so for an executed trace of
    the same program this is its call ledger, which traces do not carry.  No
    token fields: pass it as ``calls=`` to :func:`readable_from_messages` /
    ``diagnosis._lost_messages``, not as a token ledger."""

    if not sim.ok or sim.submit_round is None:
        return []
    calls: list[dict[str, Any]] = []
    for r, actions in enumerate(sim.rounds):
        for agent_id, action in enumerate(actions):
            if action["mode"] == "send":
                calls.append({"round": r, "agent_id": agent_id, "mode": action["mode"]})
    submitters = range(sim.n_agents) if sim.goal == "all_agents" else [sim.selected_primary]
    calls.extend({"round": sim.submit_round, "agent_id": a, "mode": "submit"} for a in submitters)
    return calls


def crosscheck_trace(
    source: str,
    record: Mapping[str, Any],
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    calls: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare the simulator with one executed trace (``QB_TRACE_DIR``
    record: ``{"n_agents", "output": {"messages", "submissions",
    "rounds_executed"}, ...}``): message edges + source ids, submit round,
    lost edges (vs ``diagnosis._lost_messages``) and readable coverage (vs
    :func:`readable_from_messages` on the trace).  Both sides are computed
    under the ``all_agents`` goal.

    ``QB_TRACE_DIR`` records carry no call ledger; without one the called
    set would be inferred from senders, which cannot see a digest ``send``
    to ``[]`` (it leaves no message).  So the trace-side numbers use
    ``calls`` when given, else the record's own ``calls`` / ``ledger.calls``
    when present, else the ledger the (deterministic) program implies
    (:func:`sim_call_ledger`; ``ledger_basis`` says which).  The edge
    comparison stays independent."""

    from queenbee.evo.diagnosis import _lost_messages

    output = dict(record.get("output") or {})
    n = int(record.get("n_agents") or len(output.get("submissions") or []))
    messages = [m for m in (output.get("messages") or []) if isinstance(m, Mapping)]
    submit = _trace_submit_round(output)
    sim = simulate_readable_coverage(source, n, max_rounds)

    def edge(m: Mapping[str, Any]) -> tuple[Any, ...]:
        return (int(m["round_sent"]), int(m["round_delivered"]), int(m["src"]),
                int(m["dst"]), tuple(int(x) for x in m.get("source_ids") or []))

    trace_edges = sorted(edge(m) for m in messages)
    sim_edges = sorted(edge(m) for m in sim.message_log)
    own = record.get("calls") or (record.get("ledger") or {}).get("calls")
    if calls is not None:
        ledger, basis = [dict(c) for c in calls], "given"
    elif own:
        ledger, basis = [dict(c) for c in own if isinstance(c, Mapping)], "trace"
    else:
        ledger, basis = sim_call_ledger(sim), "sim"
    from_trace = (
        readable_from_messages(messages, n, submit, calls=ledger or None)
        if submit is not None else None
    )
    lost_diag = _lost_messages(messages, ledger or None, submit)
    out = {
        "ledger_basis": basis if ledger else "senders",
        "n_agents": n,
        "sim_ok": sim.ok,
        "submit_sim": sim.submit_round,
        "submit_trace": submit,
        "edges_match": trace_edges == sim_edges,
        "n_messages_sim": len(sim_edges),
        "n_messages_trace": len(trace_edges),
        "lost_sim": sim.lost_edges,
        "lost_diag": lost_diag,
        "lost_trace": from_trace["lost_edges"] if from_trace else None,
        "readable_sim": list(sim.readable),
        "readable_trace": from_trace["readable"] if from_trace else None,
        "known_sim": list(sim.known),
    }
    out["agree"] = bool(
        sim.ok
        and out["submit_sim"] == out["submit_trace"]
        and out["edges_match"]
        and out["lost_sim"] == out["lost_diag"] == out["lost_trace"]
        and out["readable_sim"] == out["readable_trace"]
    )
    return out


# --------------------------------------------------------------------------- #
# Behaviour fingerprints
# --------------------------------------------------------------------------- #


def behavior_fp(
    source: str,
    n_agents: int,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    goal: str = DEFAULT_GOAL,
) -> str | None:
    """Behaviour fingerprint at ``n_agents`` (execution-cache and duplicate
    key).

    :func:`queenbee.program.execute.simulate_program_behavior` (per active
    round: edge set, modes, work instructions) canonicalised (prefix
    ``lb:``), plus the submit round, which that tuple does not carry (a
    trailing all-idle round would otherwise make two programs with
    different barriers look identical).  When that simulator declines a
    program, this module's own per-round control log is used (prefix
    ``rs:``).  The ``main()`` variant and :func:`host_digest` (the globals
    ``main()`` reads, e.g. ``MESSAGE_INSTRUCTION``) are part of the key:
    same schedule, other worker instructions = other program.  None when
    the program cannot be simulated."""

    from queenbee.program.execute import simulate_program_behavior

    sim = simulate_readable_coverage(source, n_agents, max_rounds, goal=goal)
    if not sim.ok:
        return None
    try:
        with _deadline(DEFAULT_POLICY_TIMEOUT_S):
            beh = simulate_program_behavior(
                source, n_agents=int(n_agents), max_rounds=int(max_rounds), goal=goal
            )
    except PolicyTimeout:  # pragma: no cover - the readable sim already ran it
        return None
    host = host_digest(source)
    if beh is not None:
        rounds = [
            [sorted([int(s), int(t)] for s, t in edges), list(modes), list(wis)]
            for edges, modes, wis in beh
        ]
        payload = {"scheme": "lineage", "n": int(n_agents), "submit": sim.submit_round,
                   "main": sim.semantics, "host": host, "rounds": rounds}
        prefix = "lb"
    else:
        payload = {"scheme": "sim", "n": int(n_agents), "submit": sim.submit_round,
                   "main": sim.semantics, "host": host, "rounds": sim.rounds}
        prefix = "rs"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest[:32]}"


#: Policy entry points: their behaviour is captured by the simulated
#: schedule, so their text is not part of the host digest (a text-only
#: rewrite of the policy interpreter that behaves identically, such as the
#: PHASES-interpreter extension in :mod:`queenbee.evo.seed`, keeps the
#: fingerprint).
_POLICY_ENTRY_POINTS = frozenset({"plan_submit_round", "plan_communication_turn"})


def _bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}


def host_digest(source: str) -> str | None:
    """Digest of what ``main()`` reads outside the simulated policy.

    ``main()`` runs the worker calls with module globals such as
    ``MESSAGE_INSTRUCTION`` / ``SUBMIT_INSTRUCTION`` / ``reject_nonfinite``:
    two programs with the same schedule but a different message instruction
    are different programs.  The digest covers the ``main`` text and every
    top-level binding of the globals ``main`` reads, transitively (a helper
    of a helper), except the two policy entry points (their behaviour is the
    simulated schedule).  Other top-level statements that are not simple
    bindings (``if`` / ``for`` / ``try`` / expressions) are included whole --
    they may rebind anything.  ``ast.unparse`` normalises layout and
    comments.  None when the source does not parse."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    main_nodes = [n for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"]
    bindings: dict[str, list[ast.stmt]] = {}
    opaque: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or _is_main_call(node) or _is_docstring(node):
            continue
        names = _bound_names(node)
        for name in names:
            bindings.setdefault(name, []).append(node)
        if not names:
            opaque.append(node)

    def reads(node: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    todo = set().union(*(reads(n) for n in main_nodes)) if main_nodes else set()
    seen: set[str] = set()
    included: dict[int, ast.stmt] = {}
    while todo:
        name = todo.pop()
        if name in seen or name in _POLICY_ENTRY_POINTS or name == "main":
            continue
        seen.add(name)
        for node in bindings.get(name, []):
            if id(node) not in included:
                included[id(node)] = node
                todo |= reads(node)
    parts = [ast.unparse(n) for n in main_nodes]
    parts += [ast.unparse(n) for n in tree.body if id(n) in included]
    parts += [ast.unparse(n) for n in opaque]
    return hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()[:32]


def fp_key(fps: Mapping[int, str | None]) -> str | None:
    """One duplicate key over several agent counts (None if any is None)."""

    if not fps or any(v is None for v in fps.values()):
        return None
    return "|".join(f"n{n}={fps[n]}" for n in sorted(fps))


def program_fp_key(
    source: str,
    *,
    n_list: Sequence[int] = DEFAULT_N_LIST,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    goal: str = DEFAULT_GOAL,
) -> str | None:
    """``ScreenResult.fp_key`` of one program without the rest of S0."""

    return fp_key({int(n): behavior_fp(source, int(n), max_rounds=max_rounds, goal=goal)
                   for n in n_list})


def known_fps_for(
    programs: Mapping[str, str] | Iterable[tuple[str, str]],
    *,
    n_list: Sequence[int] = DEFAULT_N_LIST,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    goal: str = DEFAULT_GOAL,
) -> dict[str, str]:
    """``known_fps`` for :func:`screen_program` from ``program_id -> source``
    (in order; the first program of a key wins).  Seed it with the parent and
    the seed program (v1) before the first proposal is screened and add each
    program that passes as it is screened, so a behaviour-identical proposal
    is ``duplicate-of:<id>`` (its rows are reused, nothing is executed).
    Programs that cannot be simulated are skipped."""

    items = programs.items() if isinstance(programs, Mapping) else programs
    out: dict[str, str] = {}
    for pid, source in items:
        key = program_fp_key(source, n_list=n_list, max_rounds=max_rounds, goal=goal)
        if key is not None:
            out.setdefault(key, str(pid))
    return out


# --------------------------------------------------------------------------- #
# Work-instruction genericity lint
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[a-z0-9_]+")
_TITLE = re.compile(r"\*\*Task:\s*(.*?)\*\*")
#: Dropped before any tokenised comparison (never vocabulary, never bigram parts).
FUNCTION_WORDS = frozenset(
    "a an and are as at be by e g eg for from how in is it its of on one or "
    "the their them they this to was were which who whose with".split()
)
#: Generic words removed from the auto-extracted vocabulary: output-shape /
#: format words, MAS vocabulary, and data-operation words a task-agnostic
#: work instruction may legitimately use (sum, count, sort, vote, union ...).
GENERIC_STOPLIST = frozenset(
    """
    agent agents all answer any array arrays average based boundary candidate
    complete components compute concatenated connected coordinate count counts
    cross decimal determine dict dictionary diff difference distinct
    distributed each element elements entire every excluding final find first
    float floating frequency fully global graph hash head including input
    integer integers item items iteration iterative last length list lists
    local longest mapping max maximum min minimum most moving name number
    numbers output pair pairs place places point portion portions position
    positions precision prefix rank ranking ranks received representing result
    results same score scores segment segments set sets should single size
    sort sorted sorting step string strings submit submits submitted sum sums
    top total triple triples tuple tuples union unique unit units value values
    vote votes voting window word words filtering
    """.split()
)
#: Name / title bigrams that are generic mechanism phrases, not task names.
GENERIC_BIGRAMS = frozenset({
    ("hash", "based"), ("cross", "boundary"), ("set", "union"),
    ("frequency", "count"), ("compute", "rank"),
})


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[:-2].endswith(("s", "x", "ch", "sh")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _words(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(str(text).lower()) if t not in FUNCTION_WORDS]


@dataclass(frozen=True)
class WIVocabulary:
    unigrams: frozenset[str]
    bigrams: frozenset[tuple[str, str]]
    templates: tuple[str, ...]
    source_texts: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unigrams": sorted(self.unigrams),
            "bigrams": sorted(" ".join(b) for b in self.bigrams),
            "templates": list(self.templates),
        }


def build_wi_vocabulary(
    benchmarks_dir: str | Path | None = None,
    templates: Iterable[str] = DEV_TEMPLATE_IDS,
    *,
    df_max: int = 3,
) -> WIVocabulary:
    """Task-specific vocabulary of the development templates (S0 check 5).

    Source texts per template (its Silo-Bench ``<template>_n5.json`` file):
    ``case_name``, the ``**Task: ...**`` title and the public
    ``**Output:**`` sentence.
    Unigrams: tokens of length >= 3 (not numeric), stemmed, minus
    ``GENERIC_STOPLIST`` and minus tokens found in more than ``df_max``
    templates (automatic genericity).  Bigrams: adjacent word pairs of the
    names and titles (function words dropped) minus ``GENERIC_BIGRAMS``.
    Silo-Bench TEST templates are refused (never read), whatever the active
    task."""

    from queenbee.evo.diagnosis import output_sentence

    bench = Path(benchmarks_dir) if benchmarks_dir else default_benchmarks_dir()
    chosen = [assert_silo_dev_template(t) for t in templates]
    df: dict[str, int] = {}
    uni: set[str] = set()
    bi: set[tuple[str, str]] = set()
    texts: list[tuple[str, str]] = []
    stop = {_stem(w) for w in GENERIC_STOPLIST}
    for template in chosen:
        data = json.loads((bench / f"{template}_n5.json").read_text(encoding="utf-8"))
        desc = str(data.get("task_description") or "")
        title_match = _TITLE.search(desc)
        name = str(data.get("case_name") or "")
        title = title_match.group(1) if title_match else ""
        output = output_sentence(desc) or ""
        texts.append((template, " / ".join((name, title, output))))
        seen: set[str] = set()
        for text in (name, title, output):
            for tok in _words(text):
                if len(tok) >= 3 and not tok.isdigit():
                    seen.add(tok)
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1
        uni.update(seen)
        for text in (name, title):
            words = _words(text)
            for a, b in zip(words, words[1:]):
                if (a, b) not in GENERIC_BIGRAMS:
                    bi.add((a, b))
    unigrams = frozenset(t for t in uni if t not in stop and df.get(t, 0) <= df_max)
    return WIVocabulary(
        unigrams=unigrams,
        bigrams=frozenset(bi),
        templates=tuple(chosen),
        source_texts=tuple(texts),
    )


@lru_cache(maxsize=4)
def default_wi_vocabulary(benchmarks_dir: str | None = None) -> WIVocabulary:
    return build_wi_vocabulary(benchmarks_dir)


def lint_work_instructions(
    texts: Iterable[str], vocab: WIVocabulary | None = None
) -> list[str]:
    """Sorted distinct vocabulary hits (``"palindrome"``, ``"prefix sum"``)."""

    vocab = vocab or default_wi_vocabulary()
    hits: set[str] = set()
    for text in texts:
        words = _words(text)
        hits.update(w for w in words if w in vocab.unigrams)
        hits.update(f"{a} {b}" for a, b in zip(words, words[1:]) if (a, b) in vocab.bigrams)
    return sorted(hits)


def static_phase_wis(source: str) -> list[str]:
    """``wi`` fields of every literal ``PHASES`` list bound at module level
    (used or not; also inside a top-level ``if`` / ``try`` block)."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    for top in tree.body:
        if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for node in ast.walk(top):
            if not (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PHASES" for t in node.targets
            )):
                continue
            try:
                phases = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                continue
            for phase in phases if isinstance(phases, list) else []:
                if isinstance(phase, dict) and phase.get("wi"):
                    out.append(str(phase["wi"]))
    return out


# --------------------------------------------------------------------------- #
# screen_program
# --------------------------------------------------------------------------- #


@dataclass
class ScreenResult:
    """Outcome of the S0 screen for one program, plus diagnostics.

    ``ok`` iff ``reasons`` (hard rejections) is empty.  A duplicate is not
    ok (it must not be executed): ``dup_of`` names the program whose rows
    are reused.  ``penalties`` / ``rank_penalty`` feed ranking only."""

    ok: bool
    reasons: list[str]
    readable5: list[int]
    readable10: list[int]
    lost_edges5: int
    pred_calls5: int
    pred_calls10: int
    lint_hits: list[str]
    dup_of: str | None
    penalties: list[str] = field(default_factory=list)
    rank_penalty: float = 0.0
    lost_edges10: int = 0
    known5: list[int] = field(default_factory=list)
    known10: list[int] = field(default_factory=list)
    pred_messages5: int = 0
    pred_messages10: int = 0
    fp5: str | None = None
    fp10: str | None = None
    fp_key: str | None = None
    wis: list[str] = field(default_factory=list)
    main_semantics: str = MAIN_V2
    per_n: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["per_n"] = {str(k): v for k, v in self.per_n.items()}
        return out


@lru_cache(maxsize=8)
def v1_reference_calls(
    n_list: tuple[int, ...] = DEFAULT_N_LIST, max_rounds: int = DEFAULT_MAX_ROUNDS
) -> dict[int, int]:
    """Predicted worker calls of the seed program (v1) per team size n
    (n=5: 10, n=10: 20)."""

    from queenbee.evo.seed import v1_seed_source

    v1 = v1_seed_source()
    out: dict[int, int] = {}
    for n in n_list:
        sim = simulate_readable_coverage(v1, n, max_rounds)
        if not sim.ok:  # pragma: no cover - the seed always simulates
            raise RuntimeError(f"v1 seed does not simulate at n={n}: {sim.error}")
        out[int(n)] = sim.calls
    return out


TabuSpec = Callable[[str, Mapping[str, Any] | None], "str | None"] | Iterable[tuple[str, str]]


def _tabu_reason(
    tabu: TabuSpec | None, source: str, hypothesis: Mapping[str, Any] | None
) -> str | None:
    if tabu is None:
        return None
    if callable(tabu):
        reason = tabu(source, hypothesis)
        return str(reason) if reason else None
    if not hypothesis:
        return None
    from queenbee.evo.prompt import normalize_mechanism

    pairs = {(str(c), normalize_mechanism(m)) for c, m in tabu}
    key = (str(hypothesis.get("failure_class") or ""), normalize_mechanism(hypothesis.get("mechanism")))
    if key in pairs:
        return f"tabu: ({key[0]}, {key[1]}) refuted twice"
    return None


def screen_program(
    source: str,
    *,
    n_list: Sequence[int] = DEFAULT_N_LIST,
    v1_calls: Mapping[int, int] | None = None,
    tabu: TabuSpec | None = None,
    known_fps: Mapping[str, str] | None = None,
    hypothesis: Mapping[str, Any] | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    goal: str = DEFAULT_GOAL,
    strict_coverage: bool = True,
    validate: bool = True,
    worker_contract: str = "message_only_v2",
    vocabulary: WIVocabulary | None = None,
    calls_reject_ratio: float | None = CALLS_REJECT_RATIO,
    timeout_s: float | None = DEFAULT_POLICY_TIMEOUT_S,
) -> ScreenResult:
    """Zero-cost screen S0 of one program (the module docstring lists the
    checks).

    ``v1_calls``: n -> predicted calls of the seed program (v1) (default:
    simulated from it).  ``known_fps``: ``fp_key`` -> program id of
    already-evaluated programs (:func:`known_fps_for` builds it; seed it with
    v1 so a no-op proposal is a ``duplicate-of:v1`` instead of an
    execution).  ``tabu``: a callable ``(source, hypothesis) -> reason|None``
    or ``(failure_class, mechanism)`` pairs matched against ``hypothesis``
    (``{"failure_class", "mechanism"}``, mechanisms compared after
    ``prompt.normalize_mechanism``).  ``strict_coverage`` (default True):
    every submitting agent must read n shards (every agent under
    ``all_agents``, agent ``selected_primary`` = 0 under ``sink``); with
    ``False`` one short agent is only a penalty.  ``calls_reject_ratio``:
    the hard call cap as a multiple of v1 (``None``: no hard cap, penalty
    only).  ``timeout_s``: per-simulation deadline for the program's policy
    code."""

    ns_list = tuple(int(n) for n in n_list)
    reasons: list[str] = []
    penalties: list[str] = []
    rank_penalty = 0.0

    if validate:
        from exp_graph.mas.python_code import validate_python_source

        report = validate_python_source(source, worker_contract=worker_contract)
        if not report.valid:
            first = report.errors[0] if report.errors else {}
            reasons.append(
                "validation: " + str(first.get("message") or first.get("code") or "invalid")[:200]
            )

    semantics = main_semantics(source)
    if semantics == MAIN_UNKNOWN:
        reasons.append(
            "main: unrecognised runtime main() (not the frozen message_only_v2 "
            "scaffold); its communication cannot be simulated"
        )
    ref_calls = dict(v1_calls) if v1_calls is not None else v1_reference_calls(ns_list, int(max_rounds))
    sims: dict[int, ReadableSim] = {}
    wis: dict[str, None] = {}
    for n in ns_list:
        sim = simulate_readable_coverage(source, n, max_rounds, goal=goal, timeout_s=timeout_s)
        sims[n] = sim
        if not sim.ok:
            reasons.append(f"runtime@n{n}: {sim.error}")
            continue
        for wi in sim.wis:
            wis.setdefault(wi, None)
        submitters = range(n) if goal == "all_agents" else (sim.selected_primary,)
        unread = [a for a, u in enumerate(sim.unread) if u > 0 and a in submitters]
        if unread:
            reasons.append(
                f"unread-content@n{n}: agents {unread} submit with ids whose bodies they "
                f"never read (readable {sim.readable} < known {sim.known})"
            )
        deficit = [a for a, r in enumerate(sim.readable) if r < n and a in submitters]
        if deficit and not unread:
            text = f"coverage@n{n}: agents {deficit} read fewer than {n} shards ({sim.readable})"
            hard = strict_coverage or len(deficit) > LENIENT_MAX_SHORT_AGENTS
            (reasons if hard else penalties).append(text)
        rank_penalty += len(deficit) / n
        if sim.lost_edges:
            penalties.append(f"lost-edges@n{n}: {sim.lost_edges}")
            rank_penalty += sim.lost_edges / max(1, sim.messages)
        base = ref_calls.get(n)
        if base:
            ratio = sim.calls / base
            if calls_reject_ratio is not None and ratio > calls_reject_ratio:
                reasons.append(f"calls@n{n}: {sim.calls} > {calls_reject_ratio:g}x v1 ({base})")
            elif ratio > CALLS_PENALTY_RATIO:
                penalties.append(f"calls@n{n}: {sim.calls} > {CALLS_PENALTY_RATIO:g}x v1 ({base})")
            rank_penalty += max(0.0, ratio - CALLS_PENALTY_RATIO)

    for wi in static_phase_wis(source):
        wis.setdefault(wi, None)
    for wi in wis:
        if len(wi) > WI_MAX_CHARS:
            reasons.append(f"wi-length: a work instruction has {len(wi)} > {WI_MAX_CHARS} chars")
            break
    hits = lint_work_instructions(wis, vocabulary)
    if hits:
        reasons.append(f"wi-lint: task-specific terms {hits}")

    fps = {n: (behavior_fp(source, n, max_rounds=max_rounds, goal=goal) if sims[n].ok else None)
           for n in ns_list}
    key = fp_key(fps)
    dup_of = None
    if key is not None and known_fps and key in known_fps:
        dup_of = str(known_fps[key])
        reasons.append(f"duplicate-of:{dup_of}")

    tabu_reason = _tabu_reason(tabu, source, hypothesis)
    if tabu_reason:
        reasons.append(tabu_reason)

    def field_of(n: int, attr: str, default: Any) -> Any:
        sim = sims.get(n)
        return getattr(sim, attr) if sim is not None and sim.ok else default

    return ScreenResult(
        ok=not reasons,
        reasons=reasons,
        readable5=list(field_of(5, "readable", [])),
        readable10=list(field_of(10, "readable", [])),
        lost_edges5=int(field_of(5, "lost_edges", 0)),
        pred_calls5=int(field_of(5, "calls", 0)),
        pred_calls10=int(field_of(10, "calls", 0)),
        lint_hits=hits,
        dup_of=dup_of,
        penalties=penalties,
        rank_penalty=round(rank_penalty, 6),
        lost_edges10=int(field_of(10, "lost_edges", 0)),
        known5=list(field_of(5, "known", [])),
        known10=list(field_of(10, "known", [])),
        pred_messages5=int(field_of(5, "messages", 0)),
        pred_messages10=int(field_of(10, "messages", 0)),
        fp5=fps.get(5),
        fp10=fps.get(10),
        fp_key=key,
        wis=list(wis),
        main_semantics=semantics,
        per_n={n: sims[n].summary() for n in ns_list},
    )


# --------------------------------------------------------------------------- #
# The active task's screen policy
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=8)
def _seed_reference_calls(seed: str, n_list: tuple[int, ...], max_rounds: int,
                          goal: str) -> tuple[tuple[int, int], ...]:
    out = []
    for n in n_list:
        sim = simulate_readable_coverage(seed, n, max_rounds, goal=goal)
        if not sim.ok:
            raise RuntimeError(f"the task's seed does not simulate at n={n}: {sim.error}")
        out.append((int(n), int(sim.calls)))
    return tuple(out)


def task_screen_policy(*, max_rounds: int = DEFAULT_MAX_ROUNDS) -> dict[str, Any]:
    """Keyword arguments of :func:`screen_program` under the active task:
    its goal and, when the task sets them, its team sizes (``n_list``), the
    predicted calls of its seed as the call-cap reference (``v1_calls``) and
    its work-instruction vocabulary."""

    task = get_task()
    goal = task_goal()
    policy: dict[str, Any] = {"goal": goal}
    if task.screen_ns is not None:
        policy["n_list"] = tuple(task.screen_ns)
    if task.seed_source is not None:
        ns = tuple(int(n) for n in policy.get("n_list", DEFAULT_N_LIST))
        policy["v1_calls"] = dict(_seed_reference_calls(task.seed_source(), ns,
                                                        int(max_rounds), goal))
    if task.wi_vocabulary is not None:
        policy["vocabulary"] = task.wi_vocabulary()
    return policy


def screen_for_task(source: str, **kwargs: Any) -> ScreenResult:
    """:func:`screen_program` under the active task's policy
    (:func:`task_screen_policy`; explicit ``kwargs`` win).  A task's own
    ``screen_program`` hook replaces it (same call)."""

    hook = get_task().screen_program
    if hook is not None:
        return hook(source, **kwargs)
    policy = task_screen_policy(max_rounds=int(kwargs.get("max_rounds", DEFAULT_MAX_ROUNDS)))
    return screen_program(source, **(policy | kwargs))


def fp_key_for_task(source: str, *, max_rounds: int = DEFAULT_MAX_ROUNDS) -> str | None:
    """:func:`program_fp_key` at the active task's team sizes and goal (the
    ``fp_key`` :func:`screen_for_task` reports).  A task's own
    ``program_fp_key`` hook replaces it (called with ``max_rounds``)."""

    task = get_task()
    if task.program_fp_key is not None:
        return task.program_fp_key(source, max_rounds=max_rounds)
    ns = DEFAULT_N_LIST if task.screen_ns is None else tuple(task.screen_ns)
    return program_fp_key(source, n_list=ns, max_rounds=max_rounds, goal=task_goal())


__all__ = [
    "CALLS_PENALTY_RATIO",
    "CALLS_REJECT_RATIO",
    "DEFAULT_POLICY_TIMEOUT_S",
    "GENERIC_BIGRAMS",
    "GENERIC_STOPLIST",
    "LENIENT_MAX_SHORT_AGENTS",
    "PolicyLoadError",
    "PolicyTimeout",
    "ReadableSim",
    "ScreenResult",
    "WIVocabulary",
    "behavior_fp",
    "build_wi_vocabulary",
    "crosscheck_trace",
    "default_wi_vocabulary",
    "fp_key",
    "fp_key_for_task",
    "host_digest",
    "known_fps_for",
    "lint_work_instructions",
    "load_policy",
    "main_semantics",
    "program_fp_key",
    "readable_from_messages",
    "screen_for_task",
    "screen_program",
    "sim_call_ledger",
    "simulate_readable_coverage",
    "static_phase_wis",
    "task_screen_policy",
    "v1_reference_calls",
]
