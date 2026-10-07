"""The QueenBee-Evo API card: the planner's reference to the runtime.

One text, included verbatim in every planner prompt and identical across
arms.  It states the runtime semantics of the fixed host program and of the
seed program's phase interpreter, the scoring and cost definitions and the
code rules.  It deliberately gives NO mechanism advice: no failure -> fix
mapping, no default work instruction, no recommended program shapes.

The two host-owned worker instruction strings are quoted verbatim from the
runtime (``exp_graph.mas.python_code``) and the code-law bullets that list
forbidden names and attributes are built from the validator's own
deny-lists, so the card cannot drift from what the workers actually read or
from what the validator rejects.  A task may supply its own renderer (the
Count-Frequency task edits three passages of this card).
"""

from __future__ import annotations

import textwrap

from exp_graph.mas.python_code import (
    MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION,
    MESSAGE_ONLY_V2_SUBMIT_INSTRUCTION,
)

# The validator's own deny-lists: the card lists exactly the names and
# attributes the validator rejects.
from exp_graph.mas.python_code import (
    _FORBIDDEN_ATTRIBUTE_NAMES as _VALIDATOR_ATTRIBUTES,
    _FORBIDDEN_NAMES as _VALIDATOR_NAMES,
)


def _rule(head: str, names: object, tail: str) -> str:
    """One wrapped ``  * head a, b, c tail`` bullet (sorted, deterministic).

    Dunder names are left out of the list: the names bullet covers them as
    "any dunder name", the attributes bullet as "any starting with '_'"."""

    words = ", ".join(
        sorted(str(n) for n in names if not str(n).startswith("__"))  # type: ignore[union-attr]
    )
    return textwrap.fill(
        f"{head}{words}{tail}", width=76, initial_indent="  * ",
        subsequent_indent="    ", break_on_hyphens=False, break_long_words=False,
    )

API_CARD_HEADER = "=== API CARD (runtime semantics; identical for every method) ==="

_CORE = """=== API CARD (runtime semantics; identical for every method) ===
PROGRAM LAYOUT
The HOST owns the program prefix (imports, the two worker instruction
strings quoted below, reject_nonfinite) and main() (payload I/O, the round
loop, worker calls, budgets, the submit barrier). You own the GENOME:
everything from `PHASES = [` up to (not including) `def main`. The host
splices your genome between its prefix and main(), then validates and
dry-runs the program before any paid execution. The genome must define:
  plan_submit_round(n_agents, max_rounds, information_goal) -> int S
    the synchronized barrier, 0 <= S < max_rounds; rounds 0..S-1
    communicate, round S submits.
  plan_communication_turn(round_idx, agent_id, n_agents, information_goal,
                          selected_primary, known_source_count,
                          inbox_count) -> dict
    called for every agent in every round r < S; returns
    {"mode": "send"|"reflect"|"idle", "recipients": [agent ids]} plus an
    optional "work_instruction" string (1..800 characters).
    known_source_count = number of agents whose ids this agent has been
    credited with so far (itself included; see rule 2); inbox_count =
    number of messages delivered to it at this round; selected_primary = 0.

PHASE INTERPRETER (the seed's plan_* functions; editable like any genome code)
- PHASES is an ordered list of at most 8 entries
  {"kind": str, "rounds": int, "scale_rounds_to_agents": bool, "wi": str}
  ("scale_rounds_to_agents" and "wi" are optional). An entry runs for
  "rounds" rounds, or n_agents - 1 rounds when "scale_rounds_to_agents" is
  true; entries run back to back from round 0, and S = the total number of
  phase rounds (capped at max_rounds - 1).
- In each phase round the interpreter calls phase_turn(kind, local_round,
  agent_id, n_agents, selected_primary, known_source_count, inbox_count),
  where local_round counts from 0 inside the phase. phase_turn dispatches on
  kind to a phase_<kind> function; an unknown kind returns reflect. New kinds
  are written as a new phase function plus a dispatch branch.
- Work instructions: a phase function may return its own
  "work_instruction". Otherwise, when the action is "send" and the running
  PHASES entry has a non-empty "wi", the interpreter attaches that "wi" as
  the action's work_instruction. Non-send actions never carry one.
- The digest kind: an agent whose inbox_count > 0 in that round does
  "send" with recipients [] (one worker call that reads its inbox and
  updates its own previous message; nothing is delivered); every other
  agent does "reflect".

RUNTIME SEMANTICS (exact; this is what main() does)
1. Only "send" calls the worker (one paid call). "reflect" and "idle" are
   identical free no-ops: no call, state unchanged.
2. A send's worker output becomes that agent's previous message and is
   delivered to every recipient at round r+1. A message delivered at round
   r is read only if its recipient does "send" at round r; otherwise its
   body is LOST, although its source ids still count toward the recipient's
   known_source_count. Messages delivered at round S are read by every
   submitter.
3. "send" with recipients [] is legal: the worker is called and its output
   replaces its own previous message, but nothing is delivered.
   Recipients must be distinct, in 0..n_agents-1 and never the sender;
   "reflect"/"idle" need recipients [].
4. An agent's memory is only its own previous message. A worker call sees
   exactly: a control header (its agent id, the round, S, its own action
   as PYTHON_CONTROL_JSON {mode, recipients, work_instruction}, the source
   ids it has been credited with), its previous message, the inbox
   delivered at this round, its private task prompt (task text + its own
   data shard), the optional PHASE_WORK_INSTRUCTION, and
   MESSAGE_INSTRUCTION. It never sees PHASES or any other agent's action.
5. At round S every agent is called once (submit) and sees the same
   control header (its action is the submit), its previous message, the
   inbox delivered at round S and its private submit prompt (task text +
   its own shard); it must return exactly one JSON value
   (strict json.loads; one format-repair retry). A submit call cannot carry
   a work instruction.
6. A work instruction is shown to that worker as PHASE_WORK_INSTRUCTION and
   defines what its message body must contain on that call (see
   MESSAGE_INSTRUCTION below).
7. Budgets are enforced by main(); exceeding one aborts the run (S = 0).

SCORING AND COST
- Every agent submits and every agent must be correct: S = the fraction of
  agents whose submitted answer is correct.
- Calls = sends + n submits. Cost = the tokens of all worker calls.

RULES
- Work instructions must be task-agnostic: the same text is shown on every
  task type, so it may not name a task, a task's data or its answer
  vocabulary (a host lint rejects task-specific instructions).
- The program must work for any n_agents and any task type.
- Code law (fail-closed validator; one violation rejects the program):
  * Statements not allowed: import, while, try/except, with, lambda,
    class, global/nonlocal, pass, del, yield (generators), async/await,
    the walrus operator (:=), nested functions, decorators,
    *args/**kwargs, recursion (also mutual), attribute assignment
    (x.y = ...).
  * Loops and comprehensions may iterate only over range(...) whose every
    argument is an int literal, n_agents, max_rounds, round_idx or a
    len()/min()/max() call; enumerate() of such an iterable; or a literal
    list/tuple/set. A loop over any other value (e.g. `for entry in
    PHASES:`) is rejected; index it: `for i in range(len(PHASES)):`.
{forbidden_names}
{forbidden_attributes}
  * The words "todo" and "notimplemented" may not appear anywhere in the
    program, strings included (any letter case).
  * Every name used must be defined.
  * Never define or rebind main, reject_nonfinite, MESSAGE_INSTRUCTION,
    SUBMIT_INSTRUCTION, json, sys or create_llm_client; the genome holds
    only assignments and function definitions (no top-level expression
    statements).
- Code style: build recipient lists with explicit
  `for r in range(n_agents):` loops.
"""

_CORE = _CORE.replace(
    "{forbidden_names}",
    _rule("Names that may not be used: ", _VALIDATOR_NAMES,
          ", and any dunder name."),
).replace(
    "{forbidden_attributes}",
    _rule("Attributes that may not be used: any starting with '_', and ",
          _VALIDATOR_ATTRIBUTES, " (so no list.remove; filter with a loop)."),
)

API_CARD = (
    _CORE
    + "\nHOST-OWNED WORKER INSTRUCTIONS (every worker call ends with one)\n"
    + "MESSAGE_INSTRUCTION (communication calls):\n"
    + MESSAGE_ONLY_V2_MESSAGE_INSTRUCTION.rstrip("\n")
    + "\nSUBMIT_INSTRUCTION (submit calls):\n"
    + MESSAGE_ONLY_V2_SUBMIT_INSTRUCTION.rstrip("\n")
    + "\n"
)


#: ``--preserve-dups``: one more RULES bullet, inserted after this anchor.
#: A statement about how the frozen workers read relay wording, not
#: mechanism advice: it says what a forwarding instruction must ask for,
#: never which topology to use.  It does not depend on the arm (pass the flag
#: to every arm being compared so their cards stay identical).
_PRESERVE_DUPS_ANCHOR = "- The program must work for any n_agents and any task type.\n"
PRESERVE_DUPS_RULE = (
    "- Forwarded raw data: a work instruction that asks a worker to forward,\n"
    "  relay or append raw shard data must ask for EVERY item, repeated values\n"
    "  included, in their original order (a value that occurs twice is forwarded\n"
    "  twice). Workers read wording such as \"append any new items\" as \"skip\n"
    "  values already listed\" and silently drop repeated values; never ask to\n"
    "  deduplicate, reorder or summarize raw data that is being forwarded.\n"
)


def render_api_card(*, budgets: object | None = None, preserve_dups: bool = False) -> str:
    """API_CARD, optionally followed by the run's numeric budget caps.

    ``budgets`` is anything with ``max_rounds`` / ``max_model_calls`` /
    ``max_messages`` / ``max_completion_tokens`` attributes (e.g.
    ``PythonRunBudgets``).  Callers that need arm-identical prompts pass the
    same budgets to every arm (or none).  ``preserve_dups`` adds
    :data:`PRESERVE_DUPS_RULE` to the RULES; without it the text before the
    budget line is exactly :data:`API_CARD`."""

    card = API_CARD
    if preserve_dups:
        if card.count(_PRESERVE_DUPS_ANCHOR) != 1:  # pragma: no cover - card text moved
            raise RuntimeError("API card: the preserve-dups anchor line is missing")
        card = card.replace(_PRESERVE_DUPS_ANCHOR, _PRESERVE_DUPS_ANCHOR + PRESERVE_DUPS_RULE)
    if budgets is None:
        return card
    fields = []
    for name in ("max_rounds", "max_model_calls", "max_messages",
                 "max_completion_tokens"):
        value = getattr(budgets, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            fields.append(f"{name}={value}")
    if not fields:
        return card
    return card + "Budget caps of this run: " + ", ".join(fields) + ".\n"


__all__ = ["API_CARD", "API_CARD_HEADER", "PRESERVE_DUPS_RULE", "render_api_card"]
