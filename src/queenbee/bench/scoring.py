"""Benchmark-agnostic score record for one instance (not tied to a metric such as RMSE)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreResult:
    """Outcome of running one benchmark instance through the pipeline.

    ``success`` is the strict pass/fail signal and ``partial`` an optional graded
    correctness in [0, 1]. ``n_messages``, ``n_model_calls`` and ``tokens`` record
    cost; ``final_answer`` is the run's final answer (``queenbee.bench.engine``
    reports agent 0's) and ``extra`` holds benchmark-specific details of the run.
    """

    success: bool
    partial: float | None = None
    n_messages: int = 0
    n_model_calls: int = 0
    tokens: int = 0
    final_answer: Any = None
    extra: dict[str, Any] = field(default_factory=dict)
