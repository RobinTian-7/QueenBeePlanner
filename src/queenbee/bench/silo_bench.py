"""Silo-Bench loader: benchmarks/{Level}-{NN}_n{agents}.json -> BenchmarkInstance.

Level and agent count come from the file name (or, for a file with another
name, from its body); instances can be filtered by level, agent count and
case id. The loader also removes the benchmark's topology leak: the task
statement loses its Communication Protocol section and the annotated
optimal-topology fields never reach ``instance.meta``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from queenbee.bench.benchmark import BenchmarkAdapter
from queenbee.bench.instance import BenchmarkInstance

_FILENAME_RE = re.compile(r"(?P<level>[IVX]+)-(?P<num>\d+)_n(?P<agents>\d+)\.json$")

# The "**Communication Protocol:**" section of a task_description (heading up
# to the next "**Heading:**" or end of text). It spells out the benchmark's
# annotated topology ("Topology: Chain ...") in every agent prompt -- a direct
# topology leak, removed as a whole section.
_COMMUNICATION_PROTOCOL_RE = re.compile(
    r"[ \t]*\*\*Communication Protocol:?\*\*.*?(?=\n[ \t]*\*\*|\Z)",
    flags=re.DOTALL | re.IGNORECASE,
)

# Benchmark-annotation fields never carried into instance.meta: the optimal
# topology / message count / complexity ARE part of the evaluation answer. The
# benchmark files themselves are not modified; these keys are simply not copied.
_LEAKY_METADATA_KEYS = frozenset(
    {"optimal_topology", "optimal_message_count", "theoretical_complexity"}
)


def sanitize_task_description(text: str) -> str:
    """Drop the Communication Protocol section; keep the task semantics.

    The other sections (Task, Your Data / Data Distribution, Agent Ordering,
    Algorithm, Output) are kept verbatim, apart from collapsed blank-line
    runs and stripped outer whitespace; only the section that prescribes the
    communication topology is cut.
    """
    if not text:
        return text
    sanitized = _COMMUNICATION_PROTOCOL_RE.sub("", text)
    # Collapse the runs of blank lines the excision can leave behind.
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip() + ("\n" if sanitized.strip() else "")


class SiloBenchAdapter(BenchmarkAdapter):
    """Load Silo-Bench instances from a directory of benchmark JSON files."""

    name = "silo_bench"

    def __init__(self, benchmarks_dir: str | Path) -> None:
        self.benchmarks_dir = Path(benchmarks_dir)

    def iter_instances(
        self,
        *,
        levels: list[str] | None = None,
        agent_counts: list[int] | None = None,
        cases: list[str] | None = None,
    ) -> Iterable[BenchmarkInstance]:
        """Yield the instances of the directory's files (sorted by name) that
        pass the ``levels`` / ``agent_counts`` / ``cases`` filters (None = no
        filter); a missing directory raises ``FileNotFoundError``."""
        if not self.benchmarks_dir.is_dir():
            raise FileNotFoundError(
                f"Silo-Bench benchmarks dir not found: {self.benchmarks_dir}. "
                "Set SILO_BENCH_DIR or pass --benchmarks-dir."
            )
        for path in sorted(self.benchmarks_dir.glob("*.json")):
            resolved = self._resolve_file(path)
            if resolved is None:
                continue
            data, level, agents = resolved
            if levels is not None and level not in set(levels):
                continue
            if agent_counts is not None and agents not in set(agent_counts):
                continue
            if cases is not None and data.get("case_id") not in set(cases):
                continue
            yield self._to_instance(data)

    def _resolve_file(self, path: Path) -> tuple[dict, str, int] | None:
        """Return ``(data, level, agents)`` for a loadable instance file, else None.

        A canonical ``{Level}-{NN}_n{agents}.json`` filename gives level and agent
        count directly. Any other file (e.g. a vendored test fixture) is read and
        kept only if its body is structurally a single instance (has ``case_id`` +
        ``agent_configs``); ``level`` is then derived from the ``case_id`` prefix
        and ``agents`` from the body. Aggregate files such as
        ``benchmark_summary.json`` lack those keys and are skipped.
        """
        match = _FILENAME_RE.search(path.name)
        if match is not None:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data, match.group("level"), int(match.group("agents"))

        data = json.loads(path.read_text(encoding="utf-8"))
        if not (isinstance(data, dict) and "case_id" in data and "agent_configs" in data):
            return None
        case_id = str(data["case_id"])
        level = case_id.split("-", 1)[0] if "-" in case_id else case_id
        metadata = data.get("metadata", {})
        agents = int(metadata.get("num_agents", len(data["agent_configs"])))
        return data, level, agents

    def _to_instance(self, data: dict) -> BenchmarkInstance:
        """One instance document -> :class:`BenchmarkInstance` (shards,
        expected outputs, metadata).

        ``task_prompt`` is the sanitized statement and ``meta`` drops the
        annotated optimal-topology fields. ``expected_outputs`` stays in
        ``meta``: it is host-side scoring data, which the task bridge
        (:mod:`queenbee.bench.task_bridge`) puts only into the private
        scoring payload, never into model context.
        """
        agent_configs = data["agent_configs"]
        shards = [ac["input_shard"] for ac in agent_configs]
        expected_outputs = [ac.get("expected_output") for ac in agent_configs]
        metadata = {
            key: value
            for key, value in dict(data.get("metadata", {})).items()
            if key not in _LEAKY_METADATA_KEYS
        }
        # Per-agent expected answers (segmented tasks differ per agent; plain
        # tasks repeat the single global answer). Always carried so the engine
        # can grade per agent.
        metadata["expected_outputs"] = expected_outputs
        # Normalize the segmented flag to a real bool so the engine can branch on
        # it unconditionally (a file without the field, e.g. a synthetic one,
        # -> False).
        metadata["is_segmented"] = bool(metadata.get("is_segmented", False))
        return BenchmarkInstance(
            benchmark="silo_bench",
            case_id=data["case_id"],
            case_name=data.get("case_name", ""),
            n_agents=int(metadata.get("num_agents", len(agent_configs))),
            shards=shards,
            ground_truth=expected_outputs[0] if expected_outputs else None,
            task_prompt=sanitize_task_description(data.get("task_description", "")),
            meta=metadata,
        )
