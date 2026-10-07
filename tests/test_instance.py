import pytest

from queenbee.bench.instance import BenchmarkInstance


def test_instance_happy_path():
    inst = BenchmarkInstance(
        benchmark="silo_bench",
        case_id="I-01",
        case_name="Global Max",
        n_agents=2,
        shards=[[1, 5, 3], [9, 2]],
        ground_truth=9,
        task_prompt="Find the global maximum.",
        meta={"output_type": "distributed"},
    )
    assert inst.n_agents == 2
    assert inst.shards[1] == [9, 2]
    assert inst.ground_truth == 9


def test_instance_rejects_shard_count_mismatch():
    with pytest.raises(ValueError, match="expected 3 shards"):
        BenchmarkInstance(
            benchmark="silo_bench",
            case_id="I-01",
            case_name="Global Max",
            n_agents=3,
            shards=[[1], [2]],
            ground_truth=2,
        )


def test_instance_rejects_nonpositive_agents():
    with pytest.raises(ValueError, match="n_agents must be positive"):
        BenchmarkInstance(
            benchmark="b", case_id="c", case_name="n",
            n_agents=0, shards=[], ground_truth=None,
        )
