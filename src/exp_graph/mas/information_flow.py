"""Per-agent information coverage of a final knowledge state.

``knowledge[j]`` is the set of source ids credited to agent ``j`` by the end
of the run: its own id plus the source ids of every message delivered to it,
whether or not the agent read that body (the sandbox runner reconstructs it
from the delivered messages).
"""

from __future__ import annotations


def coverage_by_agent(knowledge: list[set[int]]) -> list[float]:
    """Per agent, the fraction of the team whose source ids it holds:
    ``len(knowledge[j] & {0, ..., n - 1}) / n``, in ``[0, 1]`` (ids outside
    the team are ignored)."""
    n = len(knowledge)
    if n == 0:
        return []
    return [len(k & set(range(n))) / n for k in knowledge]
