"""The Count-Frequency seed program: a tree reduction onto agent 0.

Built from the Silo-Bench seed program of QueenBee-Evo
(:func:`queenbee.evo.seed.evo_seed_source`; its host prefix and ``main()``
are left unchanged) by deterministic, anchored text edits of its genome
(every anchor must occur exactly as often as expected, else
``RuntimeError``):

1. its fixed ``information_goal == "sink"`` branch of
   ``plan_communication_turn`` is removed (it sends every shard to agent 0
   at round 0 and lets agent 0 ``reflect``, a free no-op that drops the
   delivered bodies, so agent 0 would submit from its own shard alone); the
   phases decide under every goal;
2. the phase round count (computed in both ``plan_communication_turn`` and
   ``plan_submit_round``) accepts the optional ``"scale_rounds_log2": True``
   entry field: the entry runs ``(n_agents - 1).bit_length()`` rounds plus
   its optional ``"extra_rounds"`` (3 + 1 at n = 8);
3. two phase kinds are added (function + dispatch branch): ``tree_reduce``
   (a binary reduction onto ``selected_primary``, ranks counted from it: at
   local round k a rank with ``rank % 2^(k+1) == 2^k`` sends to
   ``rank - 2^k``; a rank with ``rank % 2^(k+1) == 0`` tallies its own data
   at k = 0 (a send to ``[]``) and later digests its inbox when it is not
   empty) and ``gather_to_hub_digest`` (in every round of the phase each
   non-hub agent sends to the hub; the hub digests its inbox when it holds
   messages and reflects otherwise);
4. ``PHASES`` becomes one ``tree_reduce`` entry with the neutral work
   instruction :data:`WI_PARTIAL`; the extra round lets the root merge its
   last partial result in a send round, under that instruction, before it
   submits.

At n = 8 under the ``sink`` goal it makes 16 worker calls (15 sends + 1
submit) and agent 0 reads every shard.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping, Sequence

#: The one neutral, task-agnostic work instruction of the seed (it asks for
#: an exactly mergeable partial result and a self-check line, never names
#: the task).
WI_PARTIAL = (
    "Write one complete partial result that another worker can combine exactly with "
    "other partial results. Start from your previous message if you have one (it "
    "already includes your own data); otherwise start from your own data. Then "
    "combine in every partial result delivered in your inbox. Count the data of each "
    "source agent exactly once: never add a partial result whose source ids are "
    "already covered by what you hold. Write out every entry of the combined result "
    "in full, nothing omitted, no prose. End with one line TOTAL=<number of raw data "
    "items your result covers>."
)

#: PHASES of the seed (edit 4 of the module docstring).
SEED_PHASES: tuple[dict[str, Any], ...] = (
    {"kind": "tree_reduce", "rounds": 3, "scale_rounds_log2": True, "extra_rounds": 1,
     "wi": WI_PARTIAL},
)

# Anchor texts of the base program and the code the edits insert; _replace
# checks how often each anchor occurs (the round-count anchor twice).
_PHASES_OLD = '''PHASES = [
    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},
    {"kind": "broadcast_last", "rounds": 1},
]
'''

_SINK_BRANCH = '''    if information_goal == "sink":
        if agent_id == selected_primary:
            if inbox_count:
                return {"mode": "reflect", "recipients": []}
            return {"mode": "idle", "recipients": []}
        if round_idx == 0:
            return {"mode": "send", "recipients": [selected_primary]}
        return {"mode": "idle", "recipients": []}
'''

_ROUNDS_OLD = '''        rounds = int(phase.get("rounds", 1))
        if bool(phase.get("scale_rounds_to_agents")):
            rounds = n_agents - 1
        if rounds < 1:
'''

_ROUNDS_NEW = '''        rounds = int(phase.get("rounds", 1))
        if bool(phase.get("scale_rounds_to_agents")):
            rounds = n_agents - 1
        if bool(phase.get("scale_rounds_log2")):
            rounds = (n_agents - 1).bit_length() + int(phase.get("extra_rounds", 0))
        if rounds < 1:
'''

_FUNCTIONS_ANCHOR = "\n\n\ndef phase_turn(kind, "

_NEW_FUNCTIONS = '''


def phase_tree_reduce(local_round, agent_id, n_agents, selected_primary,
                      known_source_count, inbox_count):
    rank = (agent_id - selected_primary) % n_agents
    step = 1 << local_round
    if rank % (2 * step) == step:
        return {"mode": "send",
                "recipients": [(selected_primary + rank - step) % n_agents]}
    if rank % (2 * step) == 0:
        if local_round == 0 or inbox_count > 0:
            return {"mode": "send", "recipients": []}
    return {"mode": "idle", "recipients": []}


def phase_gather_to_hub_digest(local_round, agent_id, n_agents,
                               selected_primary, known_source_count,
                               inbox_count):
    hub = selected_primary % n_agents
    if agent_id == hub:
        if inbox_count > 0:
            return {"mode": "send", "recipients": []}
        return {"mode": "reflect", "recipients": []}
    return {"mode": "send", "recipients": [hub]}'''

_DISPATCH_OLD = '''    if kind == "digest":
        return phase_digest(local_round, agent_id, n_agents,
                            selected_primary, known_source_count,
                            inbox_count)
    return {"mode": "reflect", "recipients": []}
'''

_DISPATCH_NEW = '''    if kind == "digest":
        return phase_digest(local_round, agent_id, n_agents,
                            selected_primary, known_source_count,
                            inbox_count)
    if kind == "tree_reduce":
        return phase_tree_reduce(local_round, agent_id, n_agents,
                                 selected_primary, known_source_count,
                                 inbox_count)
    if kind == "gather_to_hub_digest":
        return phase_gather_to_hub_digest(local_round, agent_id, n_agents,
                                          selected_primary,
                                          known_source_count, inbox_count)
    return {"mode": "reflect", "recipients": []}
'''


def _replace(text: str, old: str, new: str, *, count: int, label: str) -> str:
    found = text.count(old)
    if found != count:
        raise RuntimeError(f"CF seed transform: anchor {label!r} found {found} times "
                           f"(expected {count})")
    return text.replace(old, new)


def format_phase(phase: Mapping[str, Any]) -> str:
    """One PHASES entry in the loop's layout: JSON keys, Python booleans,
    numbers as ``repr``, strings as JSON literals."""

    parts = []
    for key, value in phase.items():
        if isinstance(value, bool):
            text = "True" if value else "False"
        elif isinstance(value, (int, float)):
            text = repr(value)
        else:
            text = json.dumps(str(value))
        parts.append(f"{json.dumps(str(key))}: {text}")
    return "    {" + ", ".join(parts) + "},"


def phases_block(phases: Sequence[Mapping[str, Any]]) -> str:
    """The ``PHASES = [...]`` block of a program, one :func:`format_phase`
    line per entry."""

    return "PHASES = [\n" + "\n".join(format_phase(p) for p in phases) + "\n]\n"


def build_program(phases: Sequence[Mapping[str, Any]], *, base: str | None = None) -> str:
    """:func:`~queenbee.evo.seed.evo_seed_source` (or ``base``) with edits 1-3
    of the module docstring and ``phases`` as its PHASES."""

    if base is None:
        from queenbee.evo.seed import evo_seed_source

        base = evo_seed_source()
    text = _replace(base, _SINK_BRANCH, "", count=1, label="sink branch")
    text = _replace(text, _ROUNDS_OLD, _ROUNDS_NEW, count=2, label="phase round count")
    text = _replace(text, _FUNCTIONS_ANCHOR, _NEW_FUNCTIONS + _FUNCTIONS_ANCHOR, count=1,
                    label="phase_turn definition")
    text = _replace(text, _DISPATCH_OLD, _DISPATCH_NEW, count=1, label="dispatch tail")
    text = _replace(text, _PHASES_OLD, phases_block(phases), count=1, label="PHASES")
    return text


@lru_cache(maxsize=1)
def seed_source() -> str:
    """The Count-Frequency seed program described in the module docstring."""

    return build_program(SEED_PHASES)


__all__ = ["SEED_PHASES", "WI_PARTIAL", "build_program", "format_phase", "phases_block",
           "seed_source"]
