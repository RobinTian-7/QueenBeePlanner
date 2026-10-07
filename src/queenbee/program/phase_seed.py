"""The phase-structured seed program (v1) of the Silo-Bench task.

This module builds the base text of the seed program, the one the census
runs; :mod:`queenbee.evo.seed` adds two behaviour-preserving extensions
(phase work-instruction forwarding and the ``digest`` phase kind) to get the
text evolution starts from.

The genome of the seed program (the phase section below) is a declarative
``PHASES`` list interpreted by two policy functions (``plan_submit_round``,
``plan_communication_turn``), together with a small library of phase
primitives (``phase_relay``, ``phase_broadcast_last``, ``phase_mesh``,
``phase_gather_to_hub``, ``phase_broadcast_from_hub``,
``phase_one_peer_step``) and their dispatcher ``phase_turn``.  All of it is
evolvable source, so the planner can insert, remove, reorder or
re-parameterize phases, modify a primitive, or add a phase kind of its own
(a function plus a dispatch branch) that descendants inherit with the
program.

The seed is ``[{"kind": "relay", "rounds": 4, "scale_rounds_to_agents":
True}, {"kind": "broadcast_last", "rounds": 1}]``: a relay chain
0 -> 1 -> ... -> n-1 (``scale_rounds_to_agents`` overrides ``rounds`` with
n - 1, so every agent's data can reach the last agent at any team size),
then one round in which the last agent sends to everyone.  The team submits
after these n rounds (round 5 at n = 5), at most at round
``max_rounds - 1``.  Only the first 8 entries of ``PHASES`` are
interpreted.  Under the ``sink`` goal the phases only set the submit round:
every other agent sends to the selected primary in round 0 and then idles;
the primary reflects in a round in which its inbox is non-empty, else idles.
"""

from __future__ import annotations

# The evolvable phase section that replaces the default program's two policy
# functions.  This text is part of the seed program: editing it changes the
# program and its sha256.  It must pass the sandbox's fail-closed AST policy
# (``exp_graph.mas.python_code.validate_python_source``): no ``while`` loops,
# and ``for`` loops / comprehensions only over statically bounded iterators
# (a plain ``for phase in PHASES`` is refused), so the interpreters scan
# ``range(8)`` and stop at the end of the list.
_PHASE_SECTION = '''PHASES = [
    {"kind": "relay", "rounds": 4, "scale_rounds_to_agents": True},
    {"kind": "broadcast_last", "rounds": 1},
]


def phase_relay(local_round, agent_id, n_agents, selected_primary,
                known_source_count, inbox_count):
    if agent_id == local_round and agent_id + 1 < n_agents:
        return {"mode": "send", "recipients": [agent_id + 1]}
    return {"mode": "reflect", "recipients": []}


def phase_broadcast_last(local_round, agent_id, n_agents, selected_primary,
                         known_source_count, inbox_count):
    if agent_id == n_agents - 1:
        recipients = []
        for r in range(n_agents):
            if r != agent_id:
                recipients.append(r)
        return {"mode": "send", "recipients": recipients}
    return {"mode": "reflect", "recipients": []}


def phase_mesh(local_round, agent_id, n_agents, selected_primary,
               known_source_count, inbox_count):
    if known_source_count == n_agents:
        return {"mode": "reflect", "recipients": []}
    recipients = []
    for r in range(n_agents):
        if r != agent_id:
            recipients.append(r)
    return {"mode": "send", "recipients": recipients}


def phase_gather_to_hub(local_round, agent_id, n_agents, selected_primary,
                        known_source_count, inbox_count):
    hub = selected_primary % n_agents
    if agent_id == hub:
        return {"mode": "reflect", "recipients": []}
    return {"mode": "send", "recipients": [hub]}


def phase_broadcast_from_hub(local_round, agent_id, n_agents,
                             selected_primary, known_source_count,
                             inbox_count):
    hub = selected_primary % n_agents
    if agent_id == hub:
        recipients = []
        for r in range(n_agents):
            if r != hub:
                recipients.append(r)
        return {"mode": "send", "recipients": recipients}
    return {"mode": "reflect", "recipients": []}


def phase_one_peer_step(local_round, agent_id, n_agents, selected_primary,
                        known_source_count, inbox_count):
    if known_source_count == n_agents:
        return {"mode": "reflect", "recipients": []}
    steps = (n_agents - 1).bit_length()
    if steps < 1:
        steps = 1
    distance = 1 << (local_round % steps)
    peer = (agent_id + distance) % n_agents
    if peer == agent_id:
        return {"mode": "reflect", "recipients": []}
    return {"mode": "send", "recipients": [peer]}


def phase_turn(kind, local_round, agent_id, n_agents, selected_primary,
               known_source_count, inbox_count):
    if kind == "relay":
        return phase_relay(local_round, agent_id, n_agents,
                           selected_primary, known_source_count, inbox_count)
    if kind == "broadcast_last":
        return phase_broadcast_last(local_round, agent_id, n_agents,
                                    selected_primary, known_source_count,
                                    inbox_count)
    if kind == "mesh":
        return phase_mesh(local_round, agent_id, n_agents, selected_primary,
                          known_source_count, inbox_count)
    if kind == "gather_to_hub":
        return phase_gather_to_hub(local_round, agent_id, n_agents,
                                   selected_primary, known_source_count,
                                   inbox_count)
    if kind == "broadcast_from_hub":
        return phase_broadcast_from_hub(local_round, agent_id, n_agents,
                                        selected_primary,
                                        known_source_count, inbox_count)
    if kind == "one_peer_step":
        return phase_one_peer_step(local_round, agent_id, n_agents,
                                   selected_primary, known_source_count,
                                   inbox_count)
    return {"mode": "reflect", "recipients": []}


def plan_submit_round(n_agents, max_rounds, information_goal):
    if max_rounds <= 1 or n_agents <= 1:
        return 0
    total = 0
    for _i in range(8):
        if _i >= len(PHASES):
            break
        phase = PHASES[_i]
        rounds = int(phase.get("rounds", 1))
        if bool(phase.get("scale_rounds_to_agents")):
            rounds = n_agents - 1
        if rounds < 1:
            rounds = 1
        total = total + rounds
    if total > max_rounds - 1:
        total = max_rounds - 1
    return total


def plan_communication_turn(
    round_idx,
    agent_id,
    n_agents,
    information_goal,
    selected_primary,
    known_source_count,
    inbox_count,
):
    if information_goal == "sink":
        if agent_id == selected_primary:
            if inbox_count:
                return {"mode": "reflect", "recipients": []}
            return {"mode": "idle", "recipients": []}
        if round_idx == 0:
            return {"mode": "send", "recipients": [selected_primary]}
        return {"mode": "idle", "recipients": []}
    if n_agents == 1:
        return {"mode": "reflect", "recipients": []}
    offset = 0
    for _i in range(8):
        if _i >= len(PHASES):
            break
        phase = PHASES[_i]
        rounds = int(phase.get("rounds", 1))
        if bool(phase.get("scale_rounds_to_agents")):
            rounds = n_agents - 1
        if rounds < 1:
            rounds = 1
        if round_idx < offset + rounds:
            return phase_turn(str(phase.get("kind", "")),
                              round_idx - offset, agent_id, n_agents,
                              selected_primary, known_source_count,
                              inbox_count)
        offset = offset + rounds
    return {"mode": "reflect", "recipients": []}'''


def build_phase_seed_source(default_source: str) -> str:
    """Replace the default program's two policy functions (everything from
    ``def plan_submit_round`` up to ``def main``) with the phase section,
    keeping every other line (helpers, ``main``, contract strings)
    unchanged."""

    lines = default_source.strip().splitlines()
    start = next(
        i for i, l in enumerate(lines)
        if l.startswith("def plan_submit_round")
    )
    end = next(i for i, l in enumerate(lines) if l.startswith("def main"))
    grafted = lines[:start] + _PHASE_SECTION.splitlines() + ["", ""] + lines[end:]
    return "\n".join(grafted) + "\n"
