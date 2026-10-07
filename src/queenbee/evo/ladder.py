"""Build Silo-Bench instances of the development templates at one rung and
write a case-list manifest for them.  At an x<n> rung (e.g. x5) each template
gets a generated n-agent instance with twice the per-agent data of the
upstream Silo-Bench files, in their JSON schema and with gold answers
re-derived by an independent checker; an o<n> rung (e.g. o10) generates
nothing and points the manifest at the upstream files themselves.  All 18
development templates are built unless --templates names some; TEST
templates are always refused.  Upstream files are read from --benchmarks-dir
(default: $SILO_BENCH_DIR, else the copy that scripts/fetch_silo_bench.sh
downloads); generated data are a fixed function of template, rung and --salt
(default dev-v1).  Typical use: "python -m queenbee.evo.ladder --rung x5
--out ladder" writes ladder/x5/<template>_x5.json and ladder/manifest_x5.json,
the case list that the evolution loop (--ladder-manifest) and the evaluator
(--manifest) read.

Rungs
-----
A unit is ``(template, rung)``, with unit id ``"<template>@<rung>"`` (e.g.
``"II-11@x5"``).

* ``o<n>``: the shipped upstream instance ``benchmarks/<template>_n<n>.json``
  (nothing is generated; the manifest points at the upstream file).
* ``x<n>``: ``n`` agents holding twice the upstream per-agent data.
  Upstream ties data size to team size: total ``L = k * n``, where ``k`` is
  the template's per-agent item count (every shard of a shipped ``_n5`` /
  ``_n10`` file holds ``k`` items, except in the two random-edge graphs,
  whose upstream generator drops duplicate edges: III-23 o5 holds 98 edges
  (19/19/19/19/22) and III-28 o5 holds 74 (14/14/14/14/18) instead of 100 /
  80).  ``x<n>`` keeps ``n`` and doubles ``L`` to ``2 * k * n``, so ``x5``
  is ``L = 10 * k`` over 5 agents (``2k`` per agent): exactly twice the o5
  data, 2.04x for III-23 and 2.16x for III-28.

A manifest's ``total_size`` is the actual item count of the file (o-rungs
included).

Generator
---------
The per-template builders (:data:`BUILDERS`) and their helpers
(:func:`rng_for`, :func:`split`, :func:`fill`, :func:`case`) form a
strong-scaling generator for the 18 development templates:
``builder(L, N, salt)`` draws a case of total size ``L`` for ``N`` agents
from a random stream keyed on ``(template, L)``, or on ``(template, L,
salt)`` when a salt is given.

The generated case (data and gold, in the generator's own layout:
:func:`build_raw`) is re-laid into the upstream schema by
:func:`to_upstream_schema`: the shipped files' key order, and
``case_name`` / ``paradigm`` / ``leetcode`` / ``evaluation_metrics`` /
``verification_logic_hint`` / ``format_description`` / system prompts copied
from the upstream file of the same template and ``n``.  The task statement
is the upstream file's own statement with only its size numbers replaced
(:data:`STATEMENT_SIZE_SLOTS`: II-18 node count, III-22 element count and
median position, III-23 / III-28 node counts).  The generator's own
statements are not used, so an x-rung differs from its o-rung in data size
only (those statements lack, e.g., I-03's answer example, I-06's
``global_xor`` formula, II-18's answer format, III-22's median position and
III-27's scoring step and tuple format).  Every other character of the
statement, numbers included, is upstream's.  User prompts render shards the
way the upstream files do (edges as tuples, integer vector ids).

Checks
------
Gold answers are re-derived from the written file by an independent checker
(:func:`recompute_gold`, which reads only the agents' shards and the public
statement) and graded with the Silo-Bench metrics
(:func:`queenbee.bench.silo_metrics.evaluate_silo_submissions`: S must be
1.0); every file is also loaded back through
:func:`queenbee.evaluate.load_instance_file` (the ``SiloBenchAdapter``)
before its manifest entry is written.

Leakage guard: every entry point refuses the TEST templates
(:func:`assert_dev_template` raises :class:`LadderLeakError`), and no TEST
template has a builder.

Usage::

    python -m queenbee.evo.ladder --rung x5 --out ladder
    python -m queenbee.evo.ladder --rung x5 --out DIR --templates II-11 III-21
    python -m queenbee.evo.ladder --rung o10 --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from queenbee.evo.common import ALL_TEMPLATE_IDS, DEV_IDS, TEST_IDS, task_dev_ids, task_test_ids
from queenbee.paths import default_benchmarks_dir

GENERATOR_VERSION = "evo-ladder-v1"
DEFAULT_SALT = "dev-v1"

# --------------------------------------------------------------------------- #
# Template universe and the TEST guard
# --------------------------------------------------------------------------- #

#: Sealed TEST templates (the shared tuple of :mod:`queenbee.evo.common`).
TEST_TEMPLATE_IDS: tuple[str, ...] = tuple(TEST_IDS)
#: The 18 DEV templates (the T and V sides of every split come from these only).
DEV_TEMPLATE_IDS: tuple[str, ...] = tuple(DEV_IDS)
TIERS: tuple[str, ...] = ("I", "II", "III")


class LadderError(RuntimeError):
    """A generated instance, an upstream file or a manifest entry failed a
    size, schema, gold, load or on-disk content check."""


class LadderLeakError(LadderError):
    """A TEST (or unknown) template reached the dev-set ladder."""


def template_of(case_or_unit_id: str) -> str:
    """``'II-11@x5' -> 'II-11'``; identity for a bare template id."""

    return str(case_or_unit_id).split("@", 1)[0]


def tier_of(case_or_unit_id: str) -> str:
    return template_of(case_or_unit_id).split("-", 1)[0]


def _assert_dev(case_or_unit_id: str, test_ids: Iterable[str], dev_ids: Iterable[str]) -> str:
    template = template_of(case_or_unit_id)
    if template in set(test_ids):
        raise LadderLeakError(
            f"LEAKAGE_GUARD: TEST template {template!r} ({case_or_unit_id!r}) "
            "refused by the QueenBee-Evo dev ladder"
        )
    if template not in set(dev_ids):
        raise LadderLeakError(f"unknown template {template!r} ({case_or_unit_id!r})")
    return template


def assert_dev_template(case_or_unit_id: str) -> str:
    """The template id of a case or unit id; raises :class:`LadderLeakError`
    when it is a TEST template or not a development template of the active
    task (Silo-Bench by default)."""

    return _assert_dev(case_or_unit_id, task_test_ids(), task_dev_ids())


def assert_silo_dev_template(case_or_unit_id: str) -> str:
    """Like :func:`assert_dev_template`, but always against the Silo-Bench
    ids, whatever the active task (for code that reads Silo-Bench files)."""

    return _assert_dev(case_or_unit_id, TEST_TEMPLATE_IDS, DEV_TEMPLATE_IDS)


# --------------------------------------------------------------------------- #
# Rungs
# --------------------------------------------------------------------------- #

_RUNG_RE = re.compile(r"^(?P<kind>[ox])(?P<n>\d+)$")


def parse_rung(rung: str) -> tuple[str, int]:
    """``'x5' -> ('x', 5)``; raises ``ValueError`` on anything else."""

    match = _RUNG_RE.match(str(rung))
    if not match or int(match.group("n")) < 2:
        raise ValueError(f"bad rung {rung!r} (expected o<n> or x<n>, n >= 2)")
    return match.group("kind"), int(match.group("n"))


def rung_n_agents(rung: str) -> int:
    return parse_rung(rung)[1]


def unit_id(template: str, rung: str) -> str:
    parse_rung(rung)
    return f"{assert_dev_template(template)}@{rung}"


# --------------------------------------------------------------------------- #
# Strong-scaling generator (the 18 dev templates only)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """[PLACEHOLDER] This system_prompt will be overridden at runtime by protocol-specific prompts.

Agent ID: {agent_id}
Total agents: {num_agents}
Max agent ID: {max_id}"""

#: Generator parameters per template: ``k`` = items per agent in the shipped
#: upstream files, ``unit`` = what one item is, ``out`` = answer size class,
#: ``minpa`` = minimum sensible items per agent.
SPEC: dict[str, dict[str, Any]] = {
    "I-01":  dict(k=20, unit="int",    out="O(1)",   minpa=2),
    "I-02":  dict(k=30, unit="word",   out="O(1)",   minpa=2),
    "I-03":  dict(k=25, unit="vote",   out="O(1)",   minpa=2),
    "I-06":  dict(k=20, unit="byte",   out="O(1)",   minpa=2),
    "I-07":  dict(k=25, unit="float",  out="O(1)",   minpa=2),
    "I-08":  dict(k=30, unit="int",    out="O(1)",   minpa=2),
    "II-11": dict(k=15, unit="int",    out="O(L/N)", minpa=2),
    "II-12": dict(k=15, unit="int",    out="O(L)",   minpa=3),
    "II-13": dict(k=15, unit="char",   out="O(1)",   minpa=8),
    "II-16": dict(k=10, unit="int",    out="O(1)",   minpa=3),
    "II-17": dict(k=15, unit="int",    out="O(L)",   minpa=2),
    "II-18": dict(k=8,  unit="node",   out="O(L)",   minpa=2),
    "III-21": dict(k=20, unit="int",    out="O(L/N)", minpa=2),
    "III-22": dict(k=25, unit="int",    out="O(1)",   minpa=3),
    "III-23": dict(k=20, unit="edge",   out="O(1)",   minpa=2),
    "III-26": dict(k=30, unit="int",    out="O(1)",   minpa=2),
    "III-27": dict(k=6,  unit="vector", out="O(1)",   minpa=2),
    "III-28": dict(k=16, unit="edge",   out="O(L)",   minpa=2),
}


def rng_for(case_id: str, L: int, salt: str | None = None) -> random.Random:
    """The deterministic random stream of one ``(case_id, L)``: seeded with
    the CRC32 of ``"<case_id>|<L>"``, or of ``"<case_id>|<L>|<salt>"`` when
    a ``salt`` is given."""

    key = f"{case_id}|{L}" if salt is None else f"{case_id}|{L}|{salt}"
    return random.Random(zlib.crc32(key.encode()) & 0xFFFFFFFF)


def split(seq, n: int):
    chunk = len(seq) // n
    return [seq[i * chunk: (i * chunk + chunk) if i < n - 1 else len(seq)] for i in range(n)]


def fill(t: str, **kw) -> str:
    for k, v in kw.items():
        t = t.replace("{" + k + "}", str(v))
    return t


def case(cid, name, paradigm, desc, shards, expected, N, L, topo,
         is_segmented=False, float_tol=0.01, extra_meta=None) -> dict:
    if is_segmented:
        assert isinstance(expected, list) and len(expected) == N, f"{cid}: bad segmented gold"
        per_agent = expected
        vlogic = ("len(agent_outputs) == num_agents and "
                  "all(agent_outputs[i] == expected_output[i] for i in range(num_agents))")
    else:
        per_agent = [expected] * N
        if isinstance(expected, float):
            vlogic = (f"len(agent_outputs) == num_agents and "
                      f"all(abs(agent_outputs[i] - expected_output) <= {float_tol} for i in range(num_agents))")
        else:
            vlogic = ("len(agent_outputs) == num_agents and "
                      "all(agent_outputs[i] == expected_output for i in range(num_agents))")

    def slen(s):
        if isinstance(s, (list, str)):
            return len(s)
        if isinstance(s, dict):
            if "users" in s:
                return len(s["users"]) + len(s["items"])
            if "A_rows" in s:
                return len(s["A_rows"])
        return 1

    meta = {
        "num_agents": N,
        "optimal_topology": topo,
        "theoretical_complexity": {"Paradigm I": "O(N) - MapReduce/Aggregation",
                                   "Paradigm II": "O(N) - Structured Mesh with Locality",
                                   "Paradigm III": "O(N log N) or O(N^2) - Global/Shuffle"}[paradigm],
        "output_type": "distributed",
        "is_segmented": is_segmented,
        "scaling_mode": "strong",
        "total_length": L,
        "length_unit": SPEC[cid]["unit"],
        "upstream_unit_k": SPEC[cid]["k"],
        "size_index_m": round(L / SPEC[cid]["k"], 3),
        "shard_lengths": [slen(s) for s in shards],
    }
    if extra_meta:
        meta.update(extra_meta)

    return {
        "case_id": cid, "case_name": name, "paradigm": paradigm,
        "metadata": meta,
        "task_description": desc,
        "agent_configs": [{
            "agent_id": i,
            "system_prompt": fill(SYSTEM_PROMPT, agent_id=i, num_agents=N, max_id=N - 1),
            "user_prompt": fill(desc, agent_id=i, num_agents=N, max_id=N - 1, input_shard=shards[i]),
            "input_shard": shards[i],
            "expected_output": per_agent[i],
        } for i in range(N)],
        "expected_output": {
            "type": "distributed", "per_agent_values": per_agent,
            "is_segmented": is_segmented, "verification_logic": vlogic,
            "format_description": "ALL agents must submit their results using submit_result(answer).",
        },
    }


P_STAR = "\n\n**Communication Protocol:**\nTopology: Star/Tree aggregation -> broadcast.\nAll agents must submit the same final answer using submit_result(answer)."
P_CHAIN = "\n\n**Communication Protocol:**\nTopology: Chain (0 <-> 1 <-> ... <-> {max_id}). Exchange boundaries with neighbors.\nAll agents must submit the same final answer using submit_result(answer)."
P_CHAIN_SEG = "\n\n**Communication Protocol:**\nTopology: Chain (0 -> 1 -> ... -> {max_id}).\nEach agent submits their OWN segment using submit_result(your_segment)."
P_MESH = "\n\n**Communication Protocol:**\nTopology: all-to-all / hash-partitioned shuffle.\nAll agents must submit the same final answer using submit_result(answer)."
P_MESH_SEG = "\n\n**Communication Protocol:**\nTopology: all-to-all shuffle.\nEach agent submits their OWN portion using submit_result(your_portion)."


# ======================================================================
# Paradigm I
# ======================================================================

def i01(L, N, salt=None):
    r = rng_for("I-01", L, salt)
    b = max(1000, 10 * L)          # scales with L so the max does not saturate at 1000
    d = [r.randint(-b, b) for _ in range(L)]
    t = """**Task: Global Maximum**

Find the GLOBAL MAXIMUM value across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Compute the maximum in your local shard
2. Coordinate with other agents to determine the global maximum

**Output:**
A single integer representing the global maximum value.""" + P_STAR
    return case("I-01", "Global Max", "Paradigm I", t, split(d, N), max(d), N, L,
                "Star (all agents -> leader) or Tree, then broadcast result")


def i02(L, N, salt=None):
    r = rng_for("I-02", L, salt)
    w = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    d = [r.choice(w) for _ in range(L)]
    for i in range(0, L, 10):
        d[i] = "apple"
    t = """**Task: Word Frequency Count**

Count the TOTAL occurrences of the word "apple" across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Count how many times "apple" appears in your local data
2. Coordinate with other agents to compute the global count

**Output:**
A single integer representing the total count of "apple".""" + P_STAR
    return case("I-02", "Word Frequency", "Paradigm I", t, split(d, N), d.count("apple"), N, L,
                "Star (all agents -> leader) then broadcast")


def i03(L, N, salt=None):
    r = rng_for("I-03", L, salt)
    c = ["Candidate_A", "Candidate_B", "Candidate_C", "Candidate_D"]
    v = [r.choice(c) for _ in range(L)]
    w = r.choice(c)
    for i in range(0, L, 2):
        v[i] = w
    cnt = {x: v.count(x) for x in c}
    t = """**Task: Distributed Voting**

Determine which candidate received the MOST votes across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Count the votes for each candidate in your local shard
2. Collaborate with other agents to determine the global winner

**Output:**
A string with the candidate name who received the most votes.""" + P_STAR
    return case("I-03", "Distributed Vote", "Paradigm I", t, split(v, N),
                max(cnt, key=cnt.get), N, L,
                "Star or Tree with histogram aggregation, then broadcast")


def i06(L, N, salt=None):
    r = rng_for("I-06", L, salt)
    d = [r.randint(0, 255) for _ in range(L)]
    x = 0
    for v in d:
        x ^= v
    t = """**Task: XOR Checksum**

Compute the XOR checksum of all data bytes across all agents.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Compute local XOR: local_xor = byte1 ^ byte2 ^ ... ^ byteN
2. Coordinate with other agents to compute global XOR
3. Property: XOR is associative and commutative

**Output:**
A single integer (0-255) representing the global XOR checksum.""" + P_STAR
    return case("I-06", "Checksum", "Paradigm I", t, split(d, N), x, N, L,
                "Star (all agents -> leader) then broadcast")


def i07(L, N, salt=None):
    r = rng_for("I-07", L, salt)
    d = [round(r.uniform(0, 100), 4) for _ in range(L)]   # 4 decimals: ~60% fewer tokens
    t = """**Task: Global Average**

Compute the average (mean) value of all numbers across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold: {input_shard}

**Algorithm:**
1. Compute local_sum and local_count
2. Coordinate with other agents to compute global_sum and global_count
3. global_average = global_sum / global_count

**Output:**
A floating-point number representing the global average (precision: 2 decimal places).""" + P_STAR
    return case("I-07", "Average Value", "Paradigm I", t, split(d, N),
                round(sum(d) / len(d), 2), N, L,
                "Star (all agents -> leader with (sum, count)) then broadcast")


def i08(L, N, salt=None):
    r = rng_for("I-08", L, salt)
    pool = list(range(max(2, L // 2)))
    d = [r.choice(pool) for _ in range(L)]
    t = """**Task: Global Distinct Count**

Count the total number of DISTINCT (unique) elements across all agents' data.

**Your Data:**
You (Agent {agent_id}) hold (may contain duplicates): {input_shard}

**Algorithm:**
1. Identify unique elements in your local data: local_set = set(your_data)
2. Coordinate with other agents to find the global set of distinct elements
3. Note: Elements may appear in multiple agents' data - need global deduplication

**Output:**
A single integer representing the total count of distinct elements.""" + P_STAR
    return case("I-08", "Set Union Size", "Paradigm I", t, split(d, N), len(set(d)), N, L,
                "Star or hash-based partitioning, then broadcast")


# ======================================================================
# Paradigm II
# ======================================================================

def ii11(L, N, salt=None):
    r = rng_for("II-11", L, salt)
    d = [r.randint(1, 50) for _ in range(L)]
    pref, c = [], 0
    for v in d:
        c += v
        pref.append(c)
    t = """**Task: Prefix Sum (Cumulative Sum)**

Compute the prefix sum (cumulative sum) of the entire array.

**Agent Ordering:**
Agent 0 -> Agent 1 -> ... -> Agent {max_id} (sequential segments)

**Your Data (Position {agent_id}):**
{input_shard}

**Algorithm:**
- prefix_sum[i] = sum of all elements from index 0 to i
- You need the TOTAL SUM from all previous agents to adjust your local prefix sums

**Output:**
Each agent submits THEIR portion of the prefix sum array.""" + P_CHAIN_SEG
    return case("II-11", "Prefix Sum", "Paradigm II", t, split(d, N), split(pref, N), N, L,
                "Chain (Agent 0 -> 1 -> 2 -> ... -> N-1)", is_segmented=True)


def ii12(L, N, salt=None):
    r = rng_for("II-12", L, salt)
    d = [r.randint(1, 100) for _ in range(L)]
    out = [round((d[i - 1] + d[i] + d[i + 1]) / 3.0, 2) for i in range(1, L - 1)]
    t = """**Task: Moving Average (Window=3)**

Compute a moving average with window size 3 for all valid positions.

**Agent Ordering:**
Agents in a LINE: 0 <-> 1 <-> 2 <-> ... <-> {max_id} (consecutive data)

**Your Data (Position {agent_id}):**
{input_shard}

**Algorithm:**
For each element (except first and last globally): avg = (prev + curr + next) / 3
- You need BOUNDARY elements from neighbors to compute averages at segment edges

**Output:**
The complete moving average array (excluding first and last positions).""" + P_CHAIN
    return case("II-12", "Moving Average", "Paradigm II", t, split(d, N), out, N, L,
                "Chain with bidirectional boundary exchange, then share result")


def ii13(L, N, salt=None):
    r = rng_for("II-13", L, salt)
    ch = [r.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(L)]
    pal = "racecar"
    p = r.randint(0, L - len(pal))
    for j, c in enumerate(pal):
        ch[p + j] = c
    s = "".join(ch)

    def longest(t_):                       # expand around each centre, O(L^2); a
        best = 1                           # brute-force O(L^3) scan takes ~10 s at L=5000
        for c in range(len(t_)):
            for lo, hi in ((c, c), (c, c + 1)):
                while lo >= 0 and hi < len(t_) and t_[lo] == t_[hi]:
                    lo -= 1
                    hi += 1
                best = max(best, hi - lo - 1)
        return best

    t = """**Task: Longest Palindrome (Cross-Boundary)**

Find the length of the longest palindromic substring, including palindromes spanning agent boundaries.

**Agent Ordering:**
Agents hold CONSECUTIVE parts of one string: 0 -> 1 -> ... -> {max_id}

**Your Data (Position {agent_id}):**
String segment: {input_shard}

**Algorithm:**
1. Find longest palindrome within your local segment
2. Exchange BOUNDARY CHARACTERS with neighbors to detect cross-boundary palindromes
3. Coordinate to find the global maximum length

**Output:**
A single integer representing the length of the longest palindrome.""" + P_CHAIN
    return case("II-13", "Longest Palindrome", "Paradigm II", t, split(s, N), longest(s), N, L,
                "Chain with prefix/suffix exchange, then broadcast result")


def ii16(L, N, salt=None):
    r = rng_for("II-16", L, salt)
    d = [r.randint(0, 10) for _ in range(L)]
    for i in range(1, L - 1, 5):
        d[i] = r.randint(0, 3)
    lm, rm = [0] * L, [0] * L
    lm[0] = d[0]
    for i in range(1, L):
        lm[i] = max(lm[i - 1], d[i])
    rm[L - 1] = d[L - 1]
    for i in range(L - 2, -1, -1):
        rm[i] = max(rm[i + 1], d[i])
    t = """**Task: Trapping Rain Water**

Given an elevation map, compute how much water can be trapped after raining.

**Agent Ordering:**
Agents hold CONSECUTIVE segments: 0 <-> 1 <-> ... <-> {max_id}

**Your Data (Position {agent_id}):**
Elevation heights: {input_shard}

**Algorithm:**
- water_at_i = min(max_height_to_left, max_height_to_right) - height[i]
- Requires TWO PASSES: left-to-right (get left_max), right-to-left (get right_max)

**Output:**
A single integer representing the total trapped water units.""" + P_CHAIN
    return case("II-16", "Trapping Rain", "Paradigm II", t, split(d, N),
                sum(max(0, min(lm[i], rm[i]) - d[i]) for i in range(L)), N, L,
                "Chain with bidirectional pass, then aggregate and broadcast")


def ii17(L, N, salt=None):
    r = rng_for("II-17", L, salt)
    d = [r.randint(1, 100) for _ in range(L)]
    t = """**Task: Difference Array**

Compute the difference array: diff[i] = arr[i] - arr[i-1] for all i >= 1.

**Agent Ordering:**
Agents hold CONSECUTIVE elements: 0 -> 1 -> ... -> {max_id}

**Your Data (Position {agent_id}):**
{input_shard}

**Algorithm:**
- Compute differences within your segment
- First element of your segment needs the LAST element from previous agent

**Output:**
The complete difference array.""" + P_CHAIN
    return case("II-17", "Diff Array", "Paradigm II", t, split(d, N),
                [d[i] - d[i - 1] for i in range(1, L)], N, L,
                "Chain (each agent -> next agent), then collect and broadcast")


def ii18(L, N, salt=None):
    """L = number of NODES (upstream: 8 per agent)."""
    r = rng_for("II-18", L, salt)
    nodes = list(range(L))
    r.shuffle(nodes)
    nxt = {nodes[i]: nodes[i + 1] for i in range(L - 1)}
    nxt[nodes[-1]] = -1
    ranks = {nid: i for i, nid in enumerate(nodes)}
    r.shuffle(nodes)
    shards = [[{"id": nid, "next": nxt[nid]} for nid in s] for s in split(nodes, N)]
    t = f"""**Task: List Ranking (Compute Rank from Head)**

Reconstruct a distributed linked list and compute each node's rank from the head.

**Data Distribution:**
- Total nodes: {L}, SCATTERED across agents
- Each node has "id" and "next" pointer (-1 for end)
- Forms ONE logical chain

**Your Data:**
Agent {{agent_id}} holds: {{input_shard}}

**Algorithm:**
1. Reconstruct the logical chain starting from head node
2. Compute each node's rank (head=0, next=1, ...)

**Output:**
Dictionary mapping node_id to its rank.""" + P_MESH
    return case("II-18", "List Ranking", "Paradigm II", t, shards,
                {str(k): v for k, v in sorted(ranks.items())}, N, L,
                "Chain or pointer-jumping tree")


# ======================================================================
# Paradigm III
# ======================================================================

def iii21(L, N, salt=None):
    r = rng_for("III-21", L, salt)
    d = [r.randint(1, max(1000, L)) for _ in range(L)]
    r.shuffle(d)
    t = """**Task: Distributed Sorting**

Sort the global dataset so that Agent 0 holds smallest values, Agent {max_id} holds largest.

**Data Distribution:**
RANDOM - your data may belong to other agents' final ranges.

**Your Data (RANDOM):**
Agent {agent_id} holds: {input_shard}

**Algorithm:**
1. Determine value ranges for each agent
2. SHUFFLE data between agents
3. Each agent sorts their final range

**Output:**
Each agent submits their sorted portion (concatenated result should be fully sorted).""" + P_MESH_SEG
    return case("III-21", "Distributed Sort", "Paradigm III", t, split(d, N),
                split(sorted(d), N), N, L,
                "All-to-all or sample-based partitioning", is_segmented=True)


def iii22(L, N, salt=None):
    r = rng_for("III-22", L, salt)
    hi = max(1000, L)          # keep median resolution as L grows
    d = [r.randint(1, hi) for _ in range(L)]
    r.shuffle(d)
    t = """**Task: Global Median (Iterative)**

Find the median value without collecting all data (use iterative consensus).

**Data Distribution:**
RANDOM across agents.

**Your Data (RANDOM):**
Agent {agent_id} holds: {input_shard}

**Algorithm (Iterative):**
1. Each agent computes local median/quantiles
2. Estimate global pivot from local medians
3. Each agent counts: values < pivot, = pivot, > pivot
4. Determine which range contains median, repeat until convergence

**Output:**
A single integer representing the global median.""" + P_MESH
    return case("III-22", "Median of Medians", "Paradigm III", t, split(d, N),
                sorted(d)[L // 2], N, L, "Iterative broadcast and aggregation")


def _uf(n):
    p = list(range(n))

    def find(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x
    return p, find


def iii23(L, N, salt=None):
    """L = number of EDGES; node count decoupled from N as L//2."""
    r = rng_for("III-23", L, salt)
    nn = max(2, L // 2)
    edges = []
    while len(edges) < L:
        u, v = r.randint(0, nn - 1), r.randint(0, nn - 1)
        if u != v:
            edges.append((u, v))
    r.shuffle(edges)
    p, find = _uf(nn)
    for u, v in edges:
        pu, pv = find(u), find(v)
        if pu != pv:
            p[pu] = pv
    t = f"""**Task: Graph Connected Components**

Find the number of connected components in an undirected graph with randomly distributed edges.

**Data Distribution:**
Total nodes: {nn}. Edges RANDOMLY distributed across agents.

**Your Data (RANDOM edges):**
Agent {{agent_id}} holds: {{input_shard}} (each edge is a pair (u, v))

**Algorithm:**
Use Union-Find. Coordinate to merge components connected through edges held by different agents.

**Output:**
A single integer representing the number of connected components.""" + P_MESH
    return case("III-23", "Graph Components", "Paradigm III", t,
                [[list(e) for e in s] for s in split(edges, N)],
                len({find(i) for i in range(nn)}), N, L,
                "Hash-based partitioning or all-to-all coordination",
                extra_meta={"num_nodes": nn})


def iii26(L, N, salt=None):
    r = rng_for("III-26", L, salt)
    pool = list(range(max(2, L // 2)))
    d = [r.choice(pool) for _ in range(L)]
    r.shuffle(d)
    t = """**Task: Global Distinct Count (Hash-Based)**

Count the total DISTINCT elements using hash-based deduplication.

**Data Distribution:**
RANDOM with duplicates. Same element may appear in multiple agents.

**Your Data (RANDOM with duplicates):**
Agent {agent_id} holds: {input_shard}

**Algorithm:**
1. Hash-based partitioning: bucket_id = hash(element) % {num_agents}
2. SHUFFLE elements to owner agents
3. Each agent deduplicates their buckets
4. Aggregate counts

**Output:**
A single integer representing the global distinct count.""" + P_MESH
    return case("III-26", "Global Distinct", "Paradigm III", t, split(d, N), len(set(d)), N, L,
                "Hash-based shuffle (all-to-all by hash)")


def iii27(L, N, salt=None):
    """L = total VECTORS; users = items = L//2 (upstream ties both to N)."""
    r = rng_for("III-27", L, salt)
    nu = ni = max(1, L // 2)
    uv = {i: [round(r.uniform(0, 1), 4) for _ in range(5)] for i in range(nu)}
    iv = {i: [round(r.uniform(0, 1), 4) for _ in range(5)] for i in range(ni)}
    alld = [("user", k, v) for k, v in uv.items()] + [("item", k, v) for k, v in iv.items()]
    r.shuffle(alld)
    shards = []
    for s in split(alld, N):
        a = {"users": {}, "items": {}}
        for ty, k, v in s:
            a["users" if ty == "user" else "items"][k] = v
        shards.append(a)
    sc = []
    for u, a in uv.items():
        for i, b in iv.items():
            sc.append((sum(x * y for x, y in zip(a, b)), u, i))
    sc.sort(key=lambda x: (-x[0], x[1], x[2]))
    top = [[u, i, round(s, 4)] for s, u, i in sc[:3]]
    t = f"""**Task: Collaborative Filtering (Top User-Item Pairs)**

Find the top 3 user-item pairs with highest compatibility scores (dot product).

**Data Distribution:**
{nu} users and {ni} items, vectors RANDOMLY distributed. Each has a 5-dim feature vector.

**Your Data (RANDOM):**
Agent {{agent_id}} holds: {{input_shard}} (dict with "users" and "items" keys)

**Algorithm:**
1. Compute dot products for ALL (user, item) pairs
2. Find top 3 pairs with highest scores

**Output:**
List of top 3 triples: [[user_id, item_id, score], ...]""" + P_MESH
    return case("III-27", "Collaborative Filtering", "Paradigm III", t, shards, top, N, L,
                "Broadcast or hash-based partitioning",
                extra_meta={"num_users": nu, "num_items": ni})


def iii28(L, N, salt=None):
    """L = number of directed EDGES; nodes = L//2."""
    r = rng_for("III-28", L, salt)
    nn = max(2, L // 2)
    g = {i: [] for i in range(nn)}
    edges = []
    guard = 0
    while len(edges) < L and guard < L * 20:
        guard += 1
        u, v = r.randint(0, nn - 1), r.randint(0, nn - 1)
        if u != v and v not in g[u]:
            g[u].append(v)
            edges.append((u, v))
    r.shuffle(edges)
    d_ = 0.85
    init = {i: 1.0 / nn for i in range(nn)}
    new = {i: (1 - d_) / nn for i in range(nn)}
    for node in range(nn):
        od = len(g[node])
        if od:
            for tgt in g[node]:
                new[tgt] += d_ * init[node] / od
    t = f"""**Task: PageRank (One Iteration)**

Compute one iteration of PageRank on a directed graph.

**Data Distribution:**
Total nodes: {nn}. Edges RANDOMLY distributed. Initial ranks: uniform 1/{nn}.

**Your Data (RANDOM edges):**
Agent {{agent_id}} holds: {{input_shard}}

**Algorithm:**
rank'[v] = (1-d)/{nn} + d * sum(rank[u]/out_degree[u]) for all u->v
where d = 0.85 (damping factor)

**Output:**
Dictionary mapping node_id to rank_value.""" + P_MESH
    return case("III-28", "PageRank Step", "Paradigm III", t,
                [[list(e) for e in s] for s in split(edges, N)],
                {str(k): round(v, 6) for k, v in sorted(new.items())}, N, len(edges),
                "Node-based partitioning with message routing",
                extra_meta={"num_nodes": nn})


BUILDERS: dict[str, Callable[..., dict]] = {
    "I-01": i01, "I-02": i02, "I-03": i03, "I-06": i06, "I-07": i07, "I-08": i08,
    "II-11": ii11, "II-12": ii12, "II-13": ii13, "II-16": ii16, "II-17": ii17, "II-18": ii18,
    "III-21": iii21, "III-22": iii22, "III-23": iii23, "III-26": iii26, "III-27": iii27,
    "III-28": iii28,
}

# ---- end of the generator ----------------------------------------------------


#: The size-dependent numbers of the upstream statements: per template,
#: ``(pattern, values)`` -- ``pattern`` matches exactly once in the upstream
#: statement with one group per number, ``values(meta)`` gives the
#: replacement numbers from the generated case's metadata.  Every other
#: number of the upstream statements is size-free (step numbers, window 3,
#: top 3, 5-dim, 0-255, d = 0.85; the tests compare statements with digits
#: normalised).
STATEMENT_SIZE_SLOTS: dict[str, tuple[tuple[str, Callable[[Mapping[str, Any]], tuple[int, ...]]], ...]] = {
    "II-18": ((r"Total nodes: (\d+),", lambda m: (int(m["total_length"]),)),),
    "III-22": ((r"Total elements: (\d+), median at position (\d+)\.",
                lambda m: (int(m["total_length"]), int(m["total_length"]) // 2)),),
    "III-23": ((r"Total nodes: (\d+)\.", lambda m: (int(m["num_nodes"]),)),),
    "III-28": ((r"Total nodes: (\d+)\.", lambda m: (int(m["num_nodes"]),)),
               (r"\(1-d\)/(\d+) ", lambda m: (int(m["num_nodes"]),))),
}


def _sub_groups(text: str, pattern: str, values: Sequence[int], where: str) -> str:
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1 or len(matches[0].groups()) != len(values):
        raise LadderError(f"{where}: size slot {pattern!r} found {len(matches)} times")
    match = matches[0]
    out, pos = [], 0
    for index, value in enumerate(values, start=1):
        out.append(text[pos:match.start(index)])
        out.append(str(int(value)))
        pos = match.end(index)
    out.append(text[pos:])
    return "".join(out)


def sized_statement(template: str, upstream_desc: str, meta: Mapping[str, Any],
                    raw_desc: str | None = None) -> str:
    """The upstream statement with its size numbers set from ``meta`` (the
    generated case's metadata).  When ``raw_desc`` (the generator's own
    statement) states the same slot, its number must agree: the data and
    that statement come from the same builder call."""

    template = assert_dev_template(template)
    desc = str(upstream_desc)
    for pattern, values_of in STATEMENT_SIZE_SLOTS.get(template, ()):
        values = values_of(meta)
        if raw_desc is not None:
            stated = re.search(pattern, raw_desc)
            if stated and tuple(int(g) for g in stated.groups()) != tuple(values):
                raise LadderError(
                    f"{template}: the generator states {stated.groups()} for {pattern!r}, "
                    f"metadata says {values}"
                )
        desc = _sub_groups(desc, pattern, values, template)
    return desc


def render_shard(template: str, shard: Any) -> Any:
    """A shard as the upstream files render it in ``user_prompt`` (Python
    ``str``): graph edges as tuples, III-27 vector ids as ints.  Only the
    file's ``user_prompt`` uses it; workers get ``input_shard``."""

    if template in ("III-23", "III-28"):
        return [tuple(edge) for edge in shard]
    if template == "III-27":
        return {part: {int(k): v for k, v in dict(shard[part]).items()} for part in ("users", "items")}
    return shard


def shard_size(shard: Any) -> int:
    """Items in one shard, counted as the generator counts them (list /
    string length, III-27 users + items)."""

    if isinstance(shard, (list, str)):
        return len(shard)
    if isinstance(shard, Mapping) and "users" in shard:
        return len(shard["users"]) + len(shard["items"])
    return 1


def total_size(template: str, rung: str) -> int:
    """Nominal problem size L of a rung: ``2 * k * n`` for ``x<n>`` (the
    generator's L), ``k * n`` for ``o<n>`` (upstream's nominal size; the
    shipped graph files hold fewer edges, see the module docstring --
    manifests record the actual count)."""

    kind, n = parse_rung(rung)
    k = int(SPEC[assert_dev_template(template)]["k"])
    return (2 if kind == "x" else 1) * k * n


def build_raw(template: str, L: int, n_agents: int, *, salt: str | None) -> dict[str, Any]:
    """The generator's case dict for ``template`` at total size ``L`` over
    ``n_agents`` (the generator's own layout, not upstream's); fewer than
    the template's minimum items per agent raises :class:`LadderError`."""

    template = assert_dev_template(template)
    per = L // n_agents
    if L < n_agents or per < int(SPEC[template]["minpa"]):
        raise LadderError(
            f"{template}: L={L} over n={n_agents} gives {per} per agent "
            f"(< minimum {SPEC[template]['minpa']})"
        )
    return BUILDERS[template](L, n_agents, salt)


# --------------------------------------------------------------------------- #
# Upstream schema
# --------------------------------------------------------------------------- #

UPSTREAM_TOP_KEYS = (
    "case_id", "case_name", "paradigm", "leetcode", "metadata",
    "task_description", "agent_configs", "expected_output", "evaluation_metrics",
)
UPSTREAM_META_KEYS = (
    "num_agents", "optimal_topology", "optimal_message_count",
    "theoretical_complexity", "output_type", "is_segmented",
)
UPSTREAM_AGENT_KEYS = (
    "agent_id", "system_prompt", "user_prompt", "input_shard", "expected_output",
)
UPSTREAM_EXPECTED_KEYS = (
    "type", "per_agent_values", "is_segmented", "verification_logic",
    "verification_logic_hint", "format_description",
)


def upstream_path(template: str, n_agents: int, benchmarks_dir: str | Path | None = None) -> Path:
    bench = Path(benchmarks_dir) if benchmarks_dir else default_benchmarks_dir()
    return bench / f"{assert_dev_template(template)}_n{int(n_agents)}.json"


def _load_upstream(template: str, n_agents: int, benchmarks_dir: str | Path | None) -> dict[str, Any]:
    path = upstream_path(template, n_agents, benchmarks_dir)
    if not path.is_file():
        raise LadderError(f"upstream skeleton {path} missing (needed for the schema)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if tuple(data) != UPSTREAM_TOP_KEYS or tuple(data["metadata"]) != UPSTREAM_META_KEYS:
        raise LadderError(f"{path}: not in the upstream schema")
    return data


def to_upstream_schema(
    raw: Mapping[str, Any],
    *,
    template: str,
    rung: str,
    benchmarks_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Re-lay a generated case (:func:`build_raw`) into the shipped files'
    schema.

    Static fields (case name, paradigm, leetcode, evaluation metrics,
    topology, message-count and complexity annotations, verification hint,
    format description, system prompts) AND the task statement come from the
    upstream file of the same template and agent count, the statement with
    its size numbers replaced (:func:`sized_statement`); data and gold come
    from ``raw``."""

    template = assert_dev_template(template)
    n = int(raw["metadata"]["num_agents"])
    up = _load_upstream(template, n, benchmarks_dir)
    desc = sized_statement(template, up["task_description"], raw["metadata"],
                           raw_desc=str(raw["task_description"]))
    segmented = bool(raw["metadata"]["is_segmented"])
    per_agent = list(raw["expected_output"]["per_agent_values"])
    shards = [ac["input_shard"] for ac in raw["agent_configs"]]
    configs = []
    for i in range(n):
        configs.append(
            {
                "agent_id": i,
                "system_prompt": up["agent_configs"][i]["system_prompt"],
                "user_prompt": fill(desc, agent_id=i, num_agents=n, max_id=n - 1,
                                    input_shard=render_shard(template, shards[i])),
                "input_shard": shards[i],
                "expected_output": per_agent[i],
            }
        )
    up_meta = up["metadata"]
    return {
        "case_id": unit_id(template, rung),
        "case_name": up["case_name"],
        "paradigm": up["paradigm"],
        "leetcode": up["leetcode"],
        "metadata": {
            "num_agents": n,
            "optimal_topology": up_meta["optimal_topology"],
            "optimal_message_count": up_meta["optimal_message_count"],
            "theoretical_complexity": up_meta["theoretical_complexity"],
            "output_type": "distributed",
            "is_segmented": segmented,
        },
        "task_description": desc,
        "agent_configs": configs,
        "expected_output": {
            "type": "distributed",
            "per_agent_values": per_agent,
            "is_segmented": segmented,
            "verification_logic": raw["expected_output"]["verification_logic"],
            "verification_logic_hint": up["expected_output"]["verification_logic_hint"],
            "format_description": up["expected_output"]["format_description"],
        },
        "evaluation_metrics": up["evaluation_metrics"],
    }


# --------------------------------------------------------------------------- #
# Independent gold checker (shards + public statement only)
# --------------------------------------------------------------------------- #


def _concat(shards: Sequence[Sequence[Any]]) -> list[Any]:
    out: list[Any] = []
    for s in shards:
        out.extend(s)
    return out


def _split_like(values: Sequence[Any], shards: Sequence[Sequence[Any]]) -> list[list[Any]]:
    out, pos = [], 0
    for s in shards:
        out.append(list(values[pos: pos + len(s)]))
        pos += len(s)
    if pos != len(values):
        raise LadderError("segment lengths do not cover the gold")
    return out


def _stmt_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise LadderError(f"statement lacks {pattern!r}")
    return int(match.group(1))


def _longest_palindrome(s: str) -> int:
    n, best = len(s), (1 if s else 0)
    for i in range(n):  # odd and even centres (written independently of the generator)
        for lo, hi in ((i - 1, i + 1), (i, i + 1)):
            while lo >= 0 and hi < n and s[lo] == s[hi]:
                lo, hi = lo - 1, hi + 1
            best = max(best, hi - lo - 1)
    return best


def _trapped_water(h: Sequence[int]) -> int:
    lo, hi, lmax, rmax, water = 0, len(h) - 1, 0, 0, 0  # two-pointer
    while lo < hi:
        if h[lo] < h[hi]:
            lmax = max(lmax, h[lo])
            water += lmax - h[lo]
            lo += 1
        else:
            rmax = max(rmax, h[hi])
            water += rmax - h[hi]
            hi -= 1
    return water


def recompute_gold(doc: Mapping[str, Any]) -> list[Any]:
    """Per-agent gold recomputed from the agents' shards and the statement
    only (never the stored gold), for a JSON-loaded upstream-format doc."""

    template = assert_dev_template(str(doc["case_id"]))
    text = str(doc["task_description"])
    shards = [ac["input_shard"] for ac in doc["agent_configs"]]
    n = len(shards)
    flat = _concat(shards) if template not in ("II-13", "III-27") else None
    if template == "I-01":
        gold: Any = max(flat)
    elif template == "I-02":
        gold = sum(1 for w in flat if w == "apple")
    elif template == "I-03":
        counts = Counter(flat).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            raise LadderError("I-03: tied vote winner")
        gold = counts[0][0]
    elif template == "I-06":
        gold = 0
        for v in flat:
            gold ^= int(v)
    elif template == "I-07":
        gold = round(sum(flat) / len(flat), 2)
    elif template in ("I-08", "III-26"):
        gold = len(set(flat))
    elif template == "II-11":
        run, pref = 0, []
        for v in flat:
            run += v
            pref.append(run)
        return _split_like(pref, shards)
    elif template == "II-12":
        gold = [round((flat[i - 1] + flat[i] + flat[i + 1]) / 3.0, 2) for i in range(1, len(flat) - 1)]
    elif template == "II-13":
        gold = _longest_palindrome("".join(shards))
    elif template == "II-16":
        gold = _trapped_water(flat)
    elif template == "II-17":
        gold = [flat[i] - flat[i - 1] for i in range(1, len(flat))]
    elif template == "II-18":
        nxt = {int(node["id"]): int(node["next"]) for node in flat}
        pointed = {v for v in nxt.values() if v != -1}
        heads = [k for k in nxt if k not in pointed]
        if len(heads) != 1:
            raise LadderError(f"II-18: {len(heads)} list heads")
        ranks, cur = {}, heads[0]
        while cur != -1:
            if cur in ranks:
                raise LadderError("II-18: cycle")
            ranks[cur] = len(ranks)
            cur = nxt[cur]
        if len(ranks) != len(nxt):
            raise LadderError("II-18: list does not cover every node")
        gold = {str(k): ranks[k] for k in sorted(ranks)}
    elif template == "III-21":
        return _split_like(sorted(flat), shards)
    elif template == "III-22":
        pos = _stmt_int(r"median at position (\d+)", text)
        total = _stmt_int(r"Total elements: (\d+)", text)
        if total != len(flat) or pos != len(flat) // 2:
            raise LadderError("III-22: statement size / position disagree with the data")
        gold = sorted(flat)[pos]
    elif template == "III-23":
        nn = _stmt_int(r"Total nodes: (\d+)", text)
        parent = list(range(nn))

        def root(x: int) -> int:
            while parent[x] != x:
                x = parent[x]
            return x

        for u, v in flat:
            ru, rv = root(int(u)), root(int(v))
            if ru != rv:
                parent[ru] = rv
        gold = len({root(i) for i in range(nn)})
    elif template == "III-27":
        users: dict[int, list[float]] = {}
        items: dict[int, list[float]] = {}
        for s in shards:
            users.update({int(k): v for k, v in s["users"].items()})
            items.update({int(k): v for k, v in s["items"].items()})
        scored = sorted(
            ((sum(x * y for x, y in zip(a, b)), u, i)
             for u, a in users.items() for i, b in items.items()),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        gold = [[u, i, round(s, 4)] for s, u, i in scored[:3]]
    elif template == "III-28":
        nn = _stmt_int(r"Total nodes: (\d+)", text)
        out: dict[int, list[int]] = {i: [] for i in range(nn)}
        for u, v in flat:
            out[int(u)].append(int(v))
        damping = 0.85
        rank = {i: (1 - damping) / nn for i in range(nn)}
        for u in range(nn):  # ascending u: same float summation order as the generator
            for v in out[u]:
                rank[v] += damping * (1.0 / nn) / len(out[u])
        gold = {str(k): round(rank[k], 6) for k in sorted(rank)}
    else:  # pragma: no cover - BUILDERS and this table cover the same 18
        raise LadderError(f"no checker for {template}")
    return [gold] * n


def _jdump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def verify_instance(doc: Mapping[str, Any]) -> None:
    """Check the upstream schema and the gold (independent checker and the
    Silo-Bench scorer); raises :class:`LadderError` on any mismatch."""

    cid = str(doc.get("case_id"))
    if tuple(doc) != UPSTREAM_TOP_KEYS:
        raise LadderError(f"{cid}: top-level keys {list(doc)} != upstream")
    if tuple(doc["metadata"]) != UPSTREAM_META_KEYS:
        raise LadderError(f"{cid}: metadata keys differ from upstream")
    if tuple(doc["expected_output"]) != UPSTREAM_EXPECTED_KEYS:
        raise LadderError(f"{cid}: expected_output keys differ from upstream")
    n = int(doc["metadata"]["num_agents"])
    configs = doc["agent_configs"]
    if [ac.get("agent_id") for ac in configs] != list(range(n)):
        raise LadderError(f"{cid}: agent_configs are not agents 0..{n - 1}")
    if any(tuple(ac) != UPSTREAM_AGENT_KEYS for ac in configs):
        raise LadderError(f"{cid}: agent_config keys differ from upstream")
    stored = [ac["expected_output"] for ac in configs]
    if _jdump(doc["expected_output"]["per_agent_values"]) != _jdump(stored):
        raise LadderError(f"{cid}: per_agent_values != agent expected_output")
    if "Communication Protocol" not in str(doc["task_description"]):
        raise LadderError(f"{cid}: statement lost its protocol section (upstream layout)")
    recomputed = recompute_gold(doc)
    for agent_id, (want, got) in enumerate(zip(recomputed, stored)):
        if _jdump(want) != _jdump(got):
            raise LadderError(
                f"{cid}: agent {agent_id} stored gold != independently recomputed gold"
            )
    from queenbee.bench.silo_metrics import evaluate_silo_submissions

    graded = evaluate_silo_submissions(case_id=cid, answers=stored, expected_outputs=recomputed)
    # S only: Silo-Bench's partial score P uses a strict LIS over value
    # positions, which is < 1 even for the exact gold when a sorted segment
    # repeats a value (III-21) -- a scorer property, not a gold defect.
    if graded["paper_S"] != 1.0:
        raise LadderError(f"{cid}: harness scorer rejects the gold answers")


# --------------------------------------------------------------------------- #
# build_ladder
# --------------------------------------------------------------------------- #


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _manifest_path(entry_path: Path, manifest_dir: Path) -> str:
    """Relative to the manifest for files under it (the ladder moves as one
    directory); absolute for the shipped upstream files."""

    target, base = entry_path.resolve(), manifest_dir.resolve()
    if target.is_relative_to(base):
        return os.path.relpath(target, base)
    return str(target)


def build_ladder(
    templates: Iterable[str] | None = None,
    rung: str = "x5",
    n_agents: int = 5,
    out_dir: str | Path | None = None,
    salt: str | None = DEFAULT_SALT,
    *,
    benchmarks_dir: str | Path | None = None,
    manifest_name: str | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Write the ``rung`` instances of ``templates`` (default: all 18 dev
    templates) and their case-list manifest; return the manifest entries.

    * ``x<n>``: generated (``L = 2 * k * n``, stream keyed on
      ``(template, L[, salt])``), written as
      ``<out_dir>/<rung>/<template>_<rung>.json`` in the upstream schema,
      re-loaded and gold-checked before the entry is recorded.
    * ``o<n>``: no file is written; the entry points at the shipped upstream
      instance.
    * An existing ``x<n>`` file with DIFFERENT content is never replaced
      silently (``LadderError``; ``overwrite=True`` replaces it); identical
      content is rewritten as is.

    Manifest ``<out_dir>/<manifest_name or 'manifest_<rung>.json'>``: a JSON
    list of ``{case_id: '<template>@<rung>', path, template, rung, n_agents,
    total_size (actual items in the file), instance_sha256, salt,
    generator}``; ``path`` is relative to the manifest directory for
    generated files and absolute for upstream files.  ``instance_sha256`` is
    :func:`queenbee.evaluate.instance_sha256` of the loaded instance (the
    value the evaluator stores per row).  Any TEST template raises
    ``LadderLeakError`` before anything is written."""

    from queenbee.evaluate import instance_sha256, load_instance_file

    if out_dir is None:
        raise ValueError("build_ladder needs out_dir")
    kind, rung_n = parse_rung(rung)
    if int(n_agents) != rung_n:
        raise ValueError(f"rung {rung!r} means n={rung_n}, got n_agents={n_agents}")
    wanted = list(templates) if templates is not None else list(DEV_TEMPLATE_IDS)
    chosen = [assert_dev_template(t) for t in wanted]  # guard before any write
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"duplicate templates in {wanted}")
    out = Path(out_dir)
    rung_dir = out / rung
    out.mkdir(parents=True, exist_ok=True)
    gen_info = {
        "version": GENERATOR_VERSION,
        "source_sha256": _source_sha256(),
    }
    entries: list[dict[str, Any]] = []
    for template in chosen:
        if kind == "o":
            path = upstream_path(template, rung_n, benchmarks_dir)
            if not path.is_file():
                raise LadderError(f"{path} missing")
            entry_salt = None
        else:
            raw = build_raw(template, total_size(template, rung), rung_n, salt=salt)
            doc = to_upstream_schema(raw, template=template, rung=rung,
                                     benchmarks_dir=benchmarks_dir)
            path = rung_dir / f"{template}_{rung}.json"
            text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
            if path.is_file() and path.read_text(encoding="utf-8") != text and not overwrite:
                raise LadderError(
                    f"{path} exists with different content (other salt / generator "
                    "version?); refusing to overwrite an instance that results or "
                    "manifests may already reference (overwrite=True to replace it)"
                )
            rung_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            verify_instance(json.loads(path.read_text(encoding="utf-8")))
            entry_salt = salt
        size = sum(shard_size(ac["input_shard"])
                   for ac in json.loads(path.read_text(encoding="utf-8"))["agent_configs"])
        inst = load_instance_file(path)
        if int(inst.n_agents) != rung_n:
            raise LadderError(f"{path}: loads with n={inst.n_agents}, want {rung_n}")
        if kind == "x" and inst.case_id != unit_id(template, rung):
            raise LadderError(f"{path}: loads as {inst.case_id!r}")
        if "Communication Protocol" in str(inst.task_prompt):
            raise LadderError(f"{path}: adapter did not sanitize the statement")
        entries.append(
            {
                "case_id": unit_id(template, rung),
                "path": _manifest_path(path, out),
                "template": template,
                "rung": rung,
                "n_agents": rung_n,
                "total_size": size,
                "instance_sha256": instance_sha256(inst),
                "salt": entry_salt,
                "generator": gen_info if kind == "x" else {"version": "upstream"},
            }
        )
    manifest = out / (manifest_name or f"manifest_{rung}.json")
    manifest.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    return entries


def load_ladder_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Manifest entries (paths resolved).  Every label and every file's own
    case id pass the TEST guard, and every file is re-checked against its
    entry (case id, team size, instance hash)."""

    from queenbee.evaluate import (
        instance_sha256,
        load_instance_file,
        load_manifest,
    )

    entries = load_manifest(path)
    for entry in entries:
        cid = str(entry["case_id"])
        template = assert_dev_template(cid)
        if not entry.get("path"):
            continue
        inst = load_instance_file(entry["path"])
        # the file's OWN case id is guarded too (a TEST file behind a dev id)
        own = assert_dev_template(str(inst.case_id))
        if own != template or ("@" in str(inst.case_id) and str(inst.case_id) != cid):
            raise LadderError(f"{cid}: manifest path holds instance {inst.case_id!r}")
        if "@" in cid and int(inst.n_agents) != rung_n_agents(cid.split("@", 1)[1]):
            raise LadderError(f"{cid}: instance has n={inst.n_agents}")
        if entry.get("instance_sha256") and instance_sha256(inst) != entry["instance_sha256"]:
            raise LadderError(f"{cid}: instance changed on disk")
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rung", default="x5")
    parser.add_argument("--agents", type=int, default=None, help="default: the rung's n")
    parser.add_argument("--templates", nargs="+", default=None, help="default: the 18 dev templates")
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--benchmarks-dir", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing instance files whose content differs")
    args = parser.parse_args(argv)
    n = args.agents if args.agents is not None else rung_n_agents(args.rung)
    entries = build_ladder(
        args.templates, rung=args.rung, n_agents=n, out_dir=args.out,
        salt=args.salt, benchmarks_dir=args.benchmarks_dir, overwrite=args.overwrite,
    )
    print(json.dumps({"rung": args.rung, "n_cases": len(entries),
                      "manifest": str(Path(args.out) / f"manifest_{args.rung}.json")}))
    return 0


__all__ = [
    "ALL_TEMPLATE_IDS",
    "BUILDERS",
    "DEFAULT_SALT",
    "DEV_TEMPLATE_IDS",
    "LadderError",
    "LadderLeakError",
    "SPEC",
    "TEST_TEMPLATE_IDS",
    "STATEMENT_SIZE_SLOTS",
    "assert_dev_template",
    "assert_silo_dev_template",
    "build_ladder",
    "build_raw",
    "load_ladder_manifest",
    "parse_rung",
    "recompute_gold",
    "render_shard",
    "rng_for",
    "rung_n_agents",
    "shard_size",
    "sized_statement",
    "template_of",
    "tier_of",
    "to_upstream_schema",
    "total_size",
    "unit_id",
    "upstream_path",
    "verify_instance",
]


if __name__ == "__main__":
    raise SystemExit(main())
