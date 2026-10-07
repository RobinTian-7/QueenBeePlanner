"""S0 screen of the Count-Frequency task.

:func:`queenbee.evo.screen.screen_program` at n = 8 under the ``sink``
goal, with two task-specific inputs: the call-cap reference is the seed
program's predicted worker calls at n = 8, and the work-instruction
vocabulary is the Silo-Bench development vocabulary plus this task's own
name, ``"count frequency"`` (building it needs the Silo-Bench files).
The coverage rule applies to the submitting agent 0 only: it must read all
8 shards, and every id it submits with must have been read.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from queenbee.tasks.cf.cases import N_AGENTS

GOAL = "sink"
PYTHON_MAX_ROUNDS = 64


@lru_cache(maxsize=1)
def seed_calls() -> int:
    """Predicted worker calls of the seed program at n = 8 (the call-cap
    reference)."""

    from queenbee.evo.screen import simulate_readable_coverage
    from queenbee.tasks.cf.seed import seed_source

    sim = simulate_readable_coverage(seed_source(), N_AGENTS, PYTHON_MAX_ROUNDS, goal=GOAL)
    if not sim.ok:
        raise RuntimeError(f"the CF seed does not simulate at n={N_AGENTS}: {sim.error}")
    return int(sim.calls)


@lru_cache(maxsize=1)
def vocabulary() -> Any:
    """The screen's work-instruction vocabulary (module docstring)."""

    from queenbee.evo import screen as _screen

    base = _screen.default_wi_vocabulary()
    return _screen.WIVocabulary(
        unigrams=base.unigrams,
        bigrams=frozenset(set(base.bigrams) | {("count", "frequency")}),
        templates=tuple(base.templates) + ("CF",),
        source_texts=tuple(base.source_texts) + (("CF", "Count Frequency"),),
    )


def _all_agent_penalty(summary: dict[str, Any], base_calls: int | None) -> float:
    """The n = 8 rank penalty counted over every agent: the fraction of
    agents that read fewer than n shards, unread deliveries (lost edges)
    per message, and the call ratio to the seed program beyond
    ``CALLS_PENALTY_RATIO``."""

    from queenbee.evo.screen import CALLS_PENALTY_RATIO

    n = N_AGENTS
    value = 0.0
    value += sum(1 for r in summary["readable"] if int(r) < n) / n
    if summary.get("lost_edges"):
        value += int(summary["lost_edges"]) / max(1, int(summary.get("messages") or 0))
    if base_calls:
        value += max(0.0, int(summary["calls"]) / base_calls - CALLS_PENALTY_RATIO)
    return round(value, 6)


def screen_program(source: str, **kwargs: Any) -> Any:
    """The task's ``screen_program`` (module docstring); ``n_list`` and
    ``goal`` arguments are ignored.  The coverage verdict names the
    submitting agent 0; the rank penalty is the all-agent penalty with the
    coverage deficit of the agents that do not submit taken out."""

    from queenbee.evo import screen as _screen

    kwargs.pop("n_list", None)
    kwargs.pop("goal", None)
    kwargs.setdefault("v1_calls", {N_AGENTS: seed_calls()})
    kwargs.setdefault("vocabulary", vocabulary())
    res = _screen.screen_program(source, n_list=(N_AGENTS,), goal=GOAL, **kwargs)
    summary = (res.per_n or {}).get(N_AGENTS) or {}
    if summary.get("error") is None and summary.get("readable"):
        readable, known = list(summary["readable"]), list(summary.get("known") or [])
        prefix = (f"coverage@n{N_AGENTS}", f"unread-content@n{N_AGENTS}")
        reasons = [r for r in res.reasons if not str(r).startswith(prefix)]
        r0 = int(readable[0])
        k0 = int(known[0]) if known else r0
        own: list[str] = []
        if r0 < k0:
            own.append(f"unread-content@n{N_AGENTS}: the submitting agent 0 submits with ids "
                       f"whose bodies it never read (readable {r0} < known {k0})")
        elif r0 < N_AGENTS:
            own.append(f"coverage@n{N_AGENTS}: the submitting agent 0 reads {r0} < {N_AGENTS} shards")
        deficit_all = sum(1 for r in readable if int(r) < N_AGENTS)
        base_calls = (kwargs.get("v1_calls") or {}).get(N_AGENTS)
        penalty = _all_agent_penalty(summary, base_calls) - deficit_all / N_AGENTS \
            + (1.0 / N_AGENTS if r0 < N_AGENTS else 0.0)
        res.reasons = own + reasons
        res.rank_penalty = round(max(0.0, penalty), 6)
        res.ok = not res.reasons
    return res


__all__ = ["GOAL", "screen_program", "seed_calls", "vocabulary"]
