"""The seed program (v1) of QueenBee-Evo.

The seed program has two source texts with the same behaviour:

* :func:`v1_seed_source` -- the phase-structured seed of
  :func:`queenbee.program.execute.seed_python_source` (``base="sfs_phase"``:
  relay rounds, then ``broadcast_last``): the text the census runs.
* :func:`evo_seed_source` -- the same program with two capability
  extensions, applied by a DETERMINISTIC, anchored text transform
  (:func:`apply_evo_extensions`) that adds no task content and no default
  work instructions: the text evolution starts from, so the planner can use
  the extensions, which its API card documents, from the first generation.

A task that supplies its own seed (``TaskSpec.seed_source``) uses that
program for both instead.

The two extensions:

1. **The interpreter forwards a phase's ``wi``.**  When
   ``plan_communication_turn`` gets an action from ``phase_turn`` with
   ``mode == "send"``, the running phase has a non-empty ``"wi"`` field and
   the action carries no ``work_instruction`` of its own, the interpreter
   attaches ``phase["wi"]`` to a COPY of the action (a phase function that
   returns one shared dict from several phases never has one phase's ``wi``
   stick to another's sends).  A phase function may still return its own
   ``work_instruction`` (e.g. keyed on ``agent_id``); that one wins.
2. **The ``digest`` routing primitive.**  ``phase_digest``: an agent that
   received at least one message this round does ``send`` to ``[]`` (one
   worker call that reads its inbox and updates only its own previous
   message; nothing is delivered); every other agent does ``reflect``.
   Dispatched from ``phase_turn`` under ``kind == "digest"``.

Same behaviour: the base ``PHASES`` has no ``wi`` and never uses
``digest``, so the simulated behaviour (``simulate_program_behavior`` at
n in {2, 5, 10}) and every control JSON a worker sees are identical for the
two texts, and the host prefix and ``main()`` are untouched byte for byte
(pinned by ``tests/test_evo_seed.py``); the two texts also share one
behaviour fingerprint (exec-cache key).  The census rows of the base text
are therefore valid rows of the program evolution starts from.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from queenbee.program.execute import seed_python_source

EVO_SEED_ORIGIN = "seed:python_sfs_phase_evo"
EVO_SEED_TITLE = "Python phase-structured sfs (relay + broadcast_last) +wi +digest"

# --------------------------------------------------------------------------- #
# Transform anchors: each must occur EXACTLY once in the base seed text, so a
# changed base seed fails loudly instead of silently producing another seed.
# --------------------------------------------------------------------------- #

_INTERP_OLD = '''        if round_idx < offset + rounds:
            return phase_turn(str(phase.get("kind", "")),
                              round_idx - offset, agent_id, n_agents,
                              selected_primary, known_source_count,
                              inbox_count)
'''

_INTERP_NEW = '''        if round_idx < offset + rounds:
            action = phase_turn(str(phase.get("kind", "")),
                                round_idx - offset, agent_id, n_agents,
                                selected_primary, known_source_count,
                                inbox_count)
            if action.get("mode") == "send":
                if not action.get("work_instruction"):
                    wi = str(phase.get("wi", "") or "")
                    if wi:
                        action = dict(action)
                        action["work_instruction"] = wi
            return action
'''

_DIGEST_ANCHOR = "\n\n\ndef phase_turn(kind, "

_DIGEST_FUNCTION = '''


def phase_digest(local_round, agent_id, n_agents, selected_primary,
                 known_source_count, inbox_count):
    if inbox_count > 0:
        return {"mode": "send", "recipients": []}
    return {"mode": "reflect", "recipients": []}'''

_DISPATCH_OLD = '''    if kind == "one_peer_step":
        return phase_one_peer_step(local_round, agent_id, n_agents,
                                   selected_primary, known_source_count,
                                   inbox_count)
    return {"mode": "reflect", "recipients": []}
'''

_DISPATCH_NEW = '''    if kind == "one_peer_step":
        return phase_one_peer_step(local_round, agent_id, n_agents,
                                   selected_primary, known_source_count,
                                   inbox_count)
    if kind == "digest":
        return phase_digest(local_round, agent_id, n_agents,
                            selected_primary, known_source_count,
                            inbox_count)
    return {"mode": "reflect", "recipients": []}
'''


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"evo seed transform: anchor {label!r} found {count} times in the "
            "v1 seed (expected exactly 1); the upstream seed changed"
        )
    return text.replace(old, new, 1)


def apply_evo_extensions(v1_source: str) -> str:
    """The deterministic text transform from the base seed text to the text
    evolution starts from (see the module docstring).

    A pure function of its input; raises ``RuntimeError`` when an anchor is
    not found exactly once (never a silent partial transform)."""

    text = _replace_once(v1_source, _INTERP_OLD, _INTERP_NEW, label="interpreter")
    text = _replace_once(
        text, _DIGEST_ANCHOR, _DIGEST_FUNCTION + _DIGEST_ANCHOR,
        label="phase_turn definition",
    )
    text = _replace_once(text, _DISPATCH_OLD, _DISPATCH_NEW, label="dispatch tail")
    return text


def v1_seed_source() -> str:
    """The base text of the seed program (v1): the phase-structured seed of
    the ``message_only_v2`` worker contract, as the census runs it."""

    return seed_python_source("message_only_v2", base="sfs_phase")[2]


@lru_cache(maxsize=1)
def evo_seed_source() -> str:
    """The text evolution starts from under the default Silo-Bench task:
    :func:`v1_seed_source` + phase-``wi`` forwarding + ``digest``."""

    return apply_evo_extensions(v1_seed_source())


def evo_seed() -> tuple[str, str, str]:
    """``(origin, title, source)`` in ``seed_python_source``'s shape."""

    return EVO_SEED_ORIGIN, EVO_SEED_TITLE, evo_seed_source()


def evo_seed_sha256() -> str:
    return hashlib.sha256(evo_seed_source().encode("utf-8")).hexdigest()


__all__ = [
    "EVO_SEED_ORIGIN",
    "EVO_SEED_TITLE",
    "apply_evo_extensions",
    "evo_seed",
    "evo_seed_sha256",
    "evo_seed_source",
    "v1_seed_source",
]
