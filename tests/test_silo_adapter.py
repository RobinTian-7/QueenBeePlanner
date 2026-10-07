from __future__ import annotations

import json
from pathlib import Path

import pytest

from queenbee.bench.silo_bench import SiloBenchAdapter, sanitize_task_description

# Synthetic instances in the Silo-Bench file layout. Of the four instance
# files, two names match the canonical ``{Level}-{NN}_n{agents}.json``
# pattern; the other two do not and go through the loader's body fallback.
FIXTURES: dict[str, dict] = {
    "silo_I-01_n2.json": {
        "case_id": "I-01",
        "case_name": "Global Max",
        "paradigm": "Paradigm I",
        "metadata": {
            "num_agents": 2,
            "output_type": "distributed",
            "is_segmented": False,
            "theoretical_complexity": "O(N) - MapReduce/Aggregation",
        },
        "task_description": (
            "Find the GLOBAL MAXIMUM across all agents' data. "
            "You are Agent {agent_id} and hold: {input_shard}"
        ),
        "agent_configs": [
            {"agent_id": 0, "input_shard": [3, 1, 9, 2], "expected_output": 9},
            {"agent_id": 1, "input_shard": [5, 8, 4], "expected_output": 9},
        ],
    },
    "silo_III-21_n2.json": {
        "case_id": "III-21",
        "case_name": "Distributed Sort",
        "paradigm": "Paradigm III",
        "metadata": {"num_agents": 2, "output_type": "distributed", "is_segmented": False},
        "task_description": (
            "Return the globally sorted ascending list of all agents' values. "
            "You are Agent {agent_id} and hold: {input_shard}"
        ),
        "agent_configs": [
            {"agent_id": 0, "input_shard": [3, 1], "expected_output": [1, 2, 3, 4]},
            {"agent_id": 1, "input_shard": [4, 2], "expected_output": [1, 2, 3, 4]},
        ],
    },
    "silo_ORD-50_n2.json": {
        "case_id": "ORD-50",
        "case_name": "Boundary Join",
        "paradigm": "Paradigm II",
        "metadata": {"num_agents": 2, "output_type": "distributed", "is_segmented": False},
        "task_description": (
            "Count adjacent equal pairs across the whole array. "
            "You are Agent {agent_id} (Position {agent_id}) and hold: {input_shard}"
        ),
        "agent_configs": [
            {"agent_id": 0, "input_shard": [1, 2, 2], "expected_output": 2},
            {"agent_id": 1, "input_shard": [2, 5, 5], "expected_output": 2},
        ],
    },
    "silo_SEG_n2.json": {
        "case_id": "SEG-99",
        "case_name": "Segmented Prefix Sum",
        "paradigm": "Paradigm II",
        "metadata": {"num_agents": 2, "output_type": "distributed", "is_segmented": True},
        "task_description": (
            "Compute the running prefix sums of all agents' values in order. "
            "You are Agent {agent_id} and hold: {input_shard}."
        ),
        "agent_configs": [
            {"agent_id": 0, "input_shard": [1, 2, 3], "expected_output": [1, 3, 6]},
            {"agent_id": 1, "input_shard": [4, 5], "expected_output": [10, 15]},
        ],
    },
    # Aggregate files carry neither case_id nor agent_configs and are skipped.
    "benchmark_summary.json": {"total": 4, "levels": ["I", "II", "III"]},
}


def write_fixtures(directory: Path, fixtures: dict[str, dict] = FIXTURES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in fixtures.items():
        (directory / name).write_text(json.dumps(body), encoding="utf-8")
    return directory


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return write_fixtures(tmp_path / "benchmarks")


def test_loads_instance_from_json(data_dir: Path):
    adapter = SiloBenchAdapter(data_dir)
    instances = {inst.case_id: inst for inst in adapter.iter_instances()}
    assert set(instances) == {"I-01", "III-21", "SEG-99", "ORD-50"}
    gmax = instances["I-01"]
    assert gmax.benchmark == "silo_bench"
    assert gmax.n_agents == 2
    assert gmax.shards == [[3, 1, 9, 2], [5, 8, 4]]
    assert gmax.ground_truth == 9
    assert gmax.case_name == "Global Max"
    assert "{agent_id}" in gmax.task_prompt
    assert gmax.meta["expected_outputs"] == [9, 9]
    # Benchmark annotations that give away the intended structure are dropped.
    assert "theoretical_complexity" not in gmax.meta
    assert gmax.segmented is False
    assert instances["SEG-99"].segmented is True
    assert instances["SEG-99"].meta["expected_outputs"] == [[1, 3, 6], [10, 15]]


def test_filters_by_case(data_dir: Path):
    adapter = SiloBenchAdapter(data_dir)
    only = list(adapter.iter_instances(cases=["I-01"]))
    assert len(only) == 1
    assert only[0].case_id == "I-01"


def test_filters_by_level(data_dir: Path):
    adapter = SiloBenchAdapter(data_dir)
    level_iii = list(adapter.iter_instances(levels=["III"]))
    assert [inst.case_id for inst in level_iii] == ["III-21"]
    # Body fallback: the level is the case-id prefix.
    assert [i.case_id for i in adapter.iter_instances(levels=["SEG"])] == ["SEG-99"]


def test_filters_by_agent_count(tmp_path: Path):
    body = dict(FIXTURES["silo_I-01_n2.json"])
    three = dict(body)
    three["metadata"] = dict(body["metadata"], num_agents=3)
    three["agent_configs"] = [
        {"agent_id": i, "input_shard": [i], "expected_output": 2} for i in range(3)
    ]
    data = write_fixtures(
        tmp_path / "benchmarks", {"I-01_n2.json": body, "I-01_n3.json": three}
    )
    adapter = SiloBenchAdapter(data)
    assert [i.n_agents for i in adapter.iter_instances(agent_counts=[3])] == [3]
    assert [i.n_agents for i in adapter.iter_instances(agent_counts=[2])] == [2]
    assert len(list(adapter.iter_instances(agent_counts=[5]))) == 0


def test_loader_drops_protocol_section_and_optimal_structure(tmp_path: Path):
    body = dict(FIXTURES["silo_I-01_n2.json"])
    body["metadata"] = dict(
        body["metadata"], optimal_topology="Chain", optimal_message_count=1
    )
    body["task_description"] = (
        "**Task: Global Maximum**\nFind the maximum.\n\n"
        "**Communication Protocol:**\nTopology: Chain (0-1). Pass it on.\n\n"
        "**Output:** A single integer.\n"
    )
    data = write_fixtures(tmp_path / "benchmarks", {"I-01_n2.json": body})
    (inst,) = SiloBenchAdapter(data).iter_instances()
    assert "Communication Protocol" not in inst.task_prompt
    assert "Topology: Chain" not in inst.task_prompt
    assert "**Output:** A single integer." in inst.task_prompt
    assert "optimal_topology" not in inst.meta
    assert "optimal_message_count" not in inst.meta


def test_sanitize_task_description_removes_protocol_section():
    text = (
        "**Task: X**\n\nBody.\n\n**Algorithm:**\nDo it.\n\n"
        "**Communication Protocol:**\nTopology: Chain (0-1-2). Exchange.\n"
        "All agents must submit the same final answer.\n"
    )
    out = sanitize_task_description(text)
    assert "Communication Protocol" not in out
    assert "Topology: Chain" not in out
    assert "**Algorithm:**" in out and "Body." in out
    assert sanitize_task_description("") == ""


def test_missing_benchmarks_dir_names_the_override(tmp_path: Path):
    adapter = SiloBenchAdapter(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="SILO_BENCH_DIR"):
        list(adapter.iter_instances())


def test_default_benchmarks_dir(tmp_path: Path, monkeypatch):
    from queenbee import paths

    monkeypatch.setenv("SILO_BENCH_DIR", str(tmp_path / "silo" / "benchmarks"))
    assert paths.default_benchmarks_dir() == tmp_path / "silo" / "benchmarks"
    monkeypatch.delenv("SILO_BENCH_DIR")
    default = paths.default_benchmarks_dir()
    repo_root = Path(paths.__file__).resolve().parents[2]
    assert default == repo_root / "third_party" / "acl26-silo-bench" / "benchmarks"
