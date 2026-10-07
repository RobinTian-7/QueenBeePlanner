"""Normalized cross-benchmark task instance.

One task of any benchmark is represented as per-agent private ``shards``
plus the expected global answer ``ground_truth``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkInstance:
    """One benchmark task instance, normalized across benchmarks.

    ``shards[i]`` is the private data held by agent ``i``.  ``ground_truth``
    is the expected global answer, which the task adapter
    (:mod:`queenbee.bench.task_bridge`) keeps out of agent-visible context.
    ``task_prompt`` is the task statement, a template that may contain the
    ``{agent_id}`` / ``{input_shard}`` placeholders; ``meta`` holds extra
    metadata (e.g. the ``is_segmented`` flag).
    """

    benchmark: str
    case_id: str
    case_name: str
    n_agents: int
    shards: list[Any]
    ground_truth: Any
    task_prompt: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require ``n_agents >= 1`` and exactly one shard per agent."""
        if self.n_agents < 1:
            raise ValueError("n_agents must be positive")
        if len(self.shards) != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} shards, got {len(self.shards)}"
            )

    @property
    def segmented(self) -> bool:
        """True iff each agent has its OWN expected answer (not one shared one).

        The per-agent answers are ``meta['expected_outputs']``.  Under the
        ``sink`` goal the engine grades a segmented instance's agent 0 against
        its own entry instead of the shared ``ground_truth``; under
        ``all_agents`` each agent is graded against its own entry whether or
        not the instance is segmented.
        """
        return bool(self.meta.get("is_segmented", False))
