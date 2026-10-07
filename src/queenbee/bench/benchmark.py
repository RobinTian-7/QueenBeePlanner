"""Benchmark adapter interface.

A :class:`BenchmarkAdapter` yields one benchmark's cases as normalized
:class:`~queenbee.bench.instance.BenchmarkInstance` objects; execution and
scoring (:mod:`queenbee.bench.engine`, :mod:`queenbee.program.execute`)
work on that normalized form rather than on a benchmark's own file layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from queenbee.bench.instance import BenchmarkInstance


class BenchmarkAdapter(ABC):
    """A source of normalized BenchmarkInstances for one benchmark."""

    name: str

    @abstractmethod
    def iter_instances(self, **filters: Any) -> Iterable[BenchmarkInstance]:
        """Yield instances, optionally narrowed by benchmark-specific filters."""
        ...
