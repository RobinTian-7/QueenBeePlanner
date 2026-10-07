"""Count-Frequency cases, generated from their seed at run time.

A case of seed ``s``: a global array of :data:`ARRAY_SIZE` integers drawn
with ``random.Random(s)`` (:data:`ARRAY_SIZE` calls of
``randint(VALUE_MIN, VALUE_MAX)``), split into :data:`N_AGENTS` contiguous
shards of equal size; the answer is the count table of the whole array
(string keys, ascending by value, values that never occur omitted).

Template ids are ``"CF-%03d" % seed``; a unit is ``<template>@o8`` (8
agents).  T side: CF-001 .. CF-004 (seeds 1-4); V side: CF-201 .. CF-203
(seeds 201-203); TEST: CF-301 .. CF-310 (seeds 301-310), sealed and never
part of the development split.

An instance is materialized on request as a Silo-Bench-schema JSON file
under :func:`cases_dir` (:func:`instance_path`); :func:`load_instance`
reads such a file and accepts it only when it equals the case its seed
generates.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from queenbee.bench.instance import BenchmarkInstance

N_AGENTS = 8
ARRAY_SIZE = 1024
VALUE_MIN, VALUE_MAX = 0, 255
RUNG = "o8"
TITLE = "Count Frequency"
#: ``BenchmarkInstance.benchmark`` of every case of this task.
BENCHMARK = "cf"

TRAIN_SEEDS: tuple[int, ...] = (1, 2, 3, 4)
VALIDATION_SEEDS: tuple[int, ...] = (201, 202, 203)
TEST_SEEDS: tuple[int, ...] = tuple(range(301, 311))

#: Finds template ids in any text (a unit id, a path, a prompt).
TEMPLATE_PATTERN = r"(?<![A-Za-z0-9-])(CF-\d{3,})(?!\d)"
_TEMPLATE_RE = re.compile(r"^CF-(\d{3,})$")

#: Environment variable that sets the directory of the materialized case
#: files (default: under the temp dir, :func:`cases_dir`).
CASES_DIR_ENV = "QUEENBEE_CF_CASES_DIR"


def template_id(seed: int) -> str:
    """``1 -> "CF-001"``."""

    return f"CF-{int(seed):03d}"


TRAIN_IDS: tuple[str, ...] = tuple(template_id(s) for s in TRAIN_SEEDS)
VALIDATION_IDS: tuple[str, ...] = tuple(template_id(s) for s in VALIDATION_SEEDS)
DEV_IDS: tuple[str, ...] = TRAIN_IDS + VALIDATION_IDS
TEST_IDS: tuple[str, ...] = tuple(template_id(s) for s in TEST_SEEDS)

#: The worker-visible statement (Task / Your Data / Output); ``{agent_id}``
#: and ``{input_shard}`` are filled in per agent.
STATEMENT = (
    f"**Task: {TITLE}**\n\n"
    f"A global array of {ARRAY_SIZE} integers, each between {VALUE_MIN} and {VALUE_MAX}, "
    f"is split across {N_AGENTS} agents in consecutive blocks of "
    f"{ARRAY_SIZE // N_AGENTS}. Compute how many times every integer occurs in the "
    "whole global array.\n\n"
    "**Your Data:**\n"
    "You (Agent {agent_id}) hold: {input_shard}\n\n"
    "**Output:**\n"
    "A JSON object mapping each integer that occurs in the global array (as a string "
    'key) to its total count, e.g. {"0": 4, "1": 1, "3": 7}. Integers that never occur '
    f"may be omitted. The counts must sum to {ARRAY_SIZE}.\n"
)


def template_of(case_or_unit_id: Any) -> str:
    """``"CF-001@o8" -> "CF-001"``."""

    return str(case_or_unit_id or "").split("@", 1)[0].strip()


def seed_of(case_or_unit_id: Any) -> int | None:
    """The seed of a template / unit id of this task (``"CF-201@o8" ->
    201``), None for any other id (also a non-canonical one such as
    ``"CF-0201"``)."""

    template = template_of(case_or_unit_id)
    match = _TEMPLATE_RE.match(template)
    if not match or template_id(int(match.group(1))) != template:
        return None
    return int(match.group(1))


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


def _sort_count_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def canonicalize_counts(counts: Mapping[Any, Any]) -> dict[str, int]:
    """String keys, ``int()`` counts, non-positive counts dropped, keys that
    stringify alike summed; sorted by integer value (other keys after)."""

    cleaned: dict[str, int] = {}
    for key, value in dict(counts).items():
        count = int(value)
        if count <= 0:
            continue
        cleaned[str(key)] = cleaned.get(str(key), 0) + count
    return {key: cleaned[key] for key in sorted(cleaned, key=_sort_count_key)}


def compute_rmse(pred_counts: Mapping[Any, Any], true_counts: Mapping[Any, Any],
                 domain_keys: list[str]) -> float:
    """Square root of the SUMMED squared count error over ``domain_keys``
    (both tables through :func:`canonicalize_counts`; a missing key counts
    0).  Not divided by the number of keys."""

    if not domain_keys:
        return 0.0
    pred = canonicalize_counts(pred_counts)
    truth = canonicalize_counts(true_counts)
    squared_error = sum(
        (float(pred.get(key, 0)) - float(truth.get(key, 0))) ** 2
        for key in domain_keys
    )
    return math.sqrt(squared_error)


def build_global_array(seed: int, *, array_size: int = ARRAY_SIZE,
                       value_min: int = VALUE_MIN, value_max: int = VALUE_MAX) -> list[int]:
    """``array_size`` draws of ``randint(value_min, value_max)`` from
    ``random.Random(seed)``."""

    if value_max < value_min:
        raise ValueError("value_max must be greater than or equal to value_min")
    rng = random.Random(seed)
    return [rng.randint(value_min, value_max) for _ in range(int(array_size))]


def split_into_shards(array: list[int], n_agents: int) -> list[list[int]]:
    """Contiguous shards; the first ``len(array) % n_agents`` are one longer."""

    if n_agents < 1:
        raise ValueError("n_agents must be positive")
    base_size, remainder = divmod(len(array), n_agents)
    shards: list[list[int]] = []
    offset = 0
    for agent_id in range(n_agents):
        size = base_size + (1 if agent_id < remainder else 0)
        shards.append(list(array[offset: offset + size]))
        offset += size
    return shards


def generate(seed: int) -> tuple[dict[str, int], list[list[int]]]:
    """``(gold count table, the N_AGENTS shards)`` of one seed."""

    array = build_global_array(int(seed))
    gold = canonicalize_counts(Counter(int(v) for v in array))
    shards = split_into_shards(array, N_AGENTS)
    if sum(len(s) for s in shards) != ARRAY_SIZE or sum(gold.values()) != ARRAY_SIZE:
        raise RuntimeError(f"seed {seed}: bad Count-Frequency case")
    return gold, shards


# --------------------------------------------------------------------------- #
# Case files and instances
# --------------------------------------------------------------------------- #


def case_document(case_id: str, seed: int | None = None) -> dict[str, Any]:
    """The Silo-Bench-schema document of one case (``seed`` defaults to the
    seed of ``case_id``)."""

    seed = seed_of(case_id) if seed is None else int(seed)
    if seed is None:
        raise ValueError(f"{case_id!r} is not a Count-Frequency template id")
    gold, shards = generate(seed)
    agents = [
        {
            "agent_id": i,
            "user_prompt": STATEMENT.replace("{agent_id}", str(i)).replace(
                "{input_shard}", json.dumps(shard)),
            "input_shard": shard,
            "expected_output": dict(gold),
        }
        for i, shard in enumerate(shards)
    ]
    return {
        "case_id": str(case_id),
        "case_name": TITLE,
        "metadata": {
            "num_agents": N_AGENTS,
            "output_type": "distributed",
            "is_segmented": False,
            "cf": {"seed": seed, "vmin": VALUE_MIN, "vmax": VALUE_MAX,
                   "array_size": ARRAY_SIZE},
            "cf_seed": seed,
            "array_size": ARRAY_SIZE,
            "value_min": VALUE_MIN,
            "value_max": VALUE_MAX,
            "generator": "queenbee.tasks.cf.cases.generate",
        },
        "task_description": STATEMENT,
        "agent_configs": agents,
    }


def instance_from_document(doc: Mapping[str, Any]) -> BenchmarkInstance:
    """A case document -> ``BenchmarkInstance`` (the Silo-Bench loader's
    field mapping)."""

    from queenbee.bench.silo_bench import sanitize_task_description

    agents = list(doc["agent_configs"])
    expected = [a.get("expected_output") for a in agents]
    meta = dict(doc.get("metadata") or {})
    meta["expected_outputs"] = expected
    meta["is_segmented"] = bool(meta.get("is_segmented", False))
    return BenchmarkInstance(
        benchmark=BENCHMARK,
        case_id=str(doc["case_id"]),
        case_name=str(doc.get("case_name") or ""),
        n_agents=int(meta.get("num_agents", len(agents))),
        shards=[a["input_shard"] for a in agents],
        ground_truth=expected[0] if expected else None,
        task_prompt=sanitize_task_description(str(doc.get("task_description") or "")),
        meta=meta,
    )


def build_instance(case_id: str, seed: int | None = None) -> BenchmarkInstance:
    """The instance of one case, straight from the generator."""

    return instance_from_document(case_document(case_id, seed))


def cases_dir() -> Path:
    """``$QUEENBEE_CF_CASES_DIR``, else ``<temp dir>/queenbee-cf-cases-<uid>``
    (``queenbee-cf-cases`` where there are no user ids)."""

    configured = os.environ.get(CASES_DIR_ENV)
    if configured:
        return Path(configured)
    getuid = getattr(os, "getuid", None)
    name = "queenbee-cf-cases" + (f"-{getuid()}" if getuid is not None else "")
    return Path(tempfile.gettempdir()) / name


def _document_text(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, indent=1) + "\n"


def write_case(case_id: str, directory: str | Path, seed: int | None = None) -> Path:
    """Write the case file ``<directory>/<case_id>.json`` (atomically; an
    identical file is left alone) and return its path."""

    text = _document_text(case_document(case_id, seed))
    path = Path(directory) / f"{case_id}.json"
    try:
        if path.read_text(encoding="utf-8") == text:
            return path
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=f".{case_id}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def instance_path(unit_or_case_id: str) -> Path | None:
    """The case file of a template / unit id of this task (written under
    :func:`cases_dir` unless an identical file is already there), None for
    any other id."""

    template = template_of(unit_or_case_id)
    if seed_of(template) is None:
        return None
    return write_case(template, cases_dir())


def load_instance(path: str | Path) -> BenchmarkInstance:
    """A case file -> instance.  The file must equal the case that its
    ``metadata.cf_seed`` (else the seed of its case id) generates under its
    own case id."""

    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not (isinstance(doc, dict) and doc.get("case_id") and doc.get("agent_configs")):
        raise ValueError(f"{path}: not a Count-Frequency case file")
    seed = (doc.get("metadata") or {}).get("cf_seed")
    try:
        want = case_document(str(doc["case_id"]), None if seed is None else int(seed))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: no Count-Frequency seed ({exc})") from exc
    if doc != want:
        raise ValueError(f"{path}: differs from the generated case {doc['case_id']} "
                         f"(seed {want['metadata']['cf_seed']})")
    return instance_from_document(want)


def default_split(split_seed: int = 0) -> dict[str, Any]:
    """T = the training templates, V = the validation templates (fixed:
    the split seed does not change them)."""

    return {"name": "cf", "T": list(TRAIN_IDS), "V": list(VALIDATION_IDS)}


__all__ = [
    "ARRAY_SIZE",
    "BENCHMARK",
    "CASES_DIR_ENV",
    "DEV_IDS",
    "N_AGENTS",
    "RUNG",
    "STATEMENT",
    "TEMPLATE_PATTERN",
    "TEST_IDS",
    "TITLE",
    "TRAIN_IDS",
    "VALIDATION_IDS",
    "VALUE_MAX",
    "VALUE_MIN",
    "build_global_array",
    "build_instance",
    "canonicalize_counts",
    "case_document",
    "cases_dir",
    "compute_rmse",
    "default_split",
    "generate",
    "instance_from_document",
    "instance_path",
    "load_instance",
    "seed_of",
    "split_into_shards",
    "template_id",
    "template_of",
    "write_case",
]
