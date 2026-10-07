"""Graded PARTIAL-CORRECTNESS scoring on Silo-Bench, and the scoring of an
executed team program.

Strict success / exact match is 1.0 or 0.0; ``partial`` carries a graded signal
in [0, 1]. ``silo_partial_score`` accepts the natural (answer, ground_truth,
output_type) triple and parses canonical-JSON strings back (the form
``canonical_answer`` renders), so it grades a raw value and the canonical key
that :mod:`queenbee.bench.task_bridge` keeps in the global task's private
scoring payload (``GROUND_TRUTH_KEY``) alike.  Also covered: Silo-Bench's own
LIS helper, ``score_protocol_answer``, and ``_score_python_execution`` under
both goals (``all_agents``, ``sink``), down to an offline sandboxed run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from exp_graph.mas.python_code import DEFAULT_MESSAGE_ONLY_V2_PROGRAM
from exp_graph.mas.python_code_generation import PythonCodePlanningResult
from exp_graph.mas.python_code_runner import CodeProcessRunner, PythonExecutionLimits

from queenbee.bench import silo_scoring
from queenbee.bench.config import RunConfig
from queenbee.bench.engine import (
    _build_python_execution_payload,
    _protocol_adapter,
    _resolved_python_execution_timeout,
    _score_python_execution,
)
from queenbee.bench.instance import BenchmarkInstance
from queenbee.bench.scoring import ScoreResult
from queenbee.bench.silo_protocol import SiloProtocolAdapter, score_protocol_answer
from queenbee.bench.silo_scoring import silo_partial_score
from queenbee.bench.task_bridge import canonical_answer


def _global_max(n_agents: int = 2) -> BenchmarkInstance:
    shards = [[3, 1, 9, 2], [5, 8, 4], [7], [6, 0]][:n_agents]
    return BenchmarkInstance(
        benchmark="silo_bench",
        case_id="I-01",
        case_name="Global Max",
        n_agents=n_agents,
        shards=shards,
        ground_truth=9,
        task_prompt=(
            "Find the GLOBAL MAXIMUM across all agents' data. "
            "You are Agent {agent_id} and hold: {input_shard}\n"
            "**Output:** A single integer."
        ),
        meta={
            "output_type": "distributed",
            "is_segmented": False,
            "expected_outputs": [9] * n_agents,
        },
    )


def _segmented() -> BenchmarkInstance:
    return BenchmarkInstance(
        benchmark="silo_bench",
        case_id="II-99",
        case_name="Segmented Prefix Sum",
        n_agents=2,
        shards=[[1, 2, 3], [4, 5]],
        ground_truth=[1, 3, 6],
        task_prompt="Report the prefix sums of your own positions: {input_shard}",
        meta={
            "output_type": "distributed",
            "is_segmented": True,
            "expected_outputs": [[1, 3, 6], [10, 15]],
        },
    )


# --------------------------------------------------------------------------- #
# silo_partial_score: grading dispatched on the value's structure
# --------------------------------------------------------------------------- #
def test_numeric_near_miss_is_partial():
    # answer 8 vs truth 9: graded, strictly between 0 and 1, and NOT a success.
    score = silo_partial_score(8, 9, "distributed")
    assert 0.0 < score < 1.0
    # 1 - |8-9|/max(1,9) = 1 - 1/9 ~= 0.888...
    assert abs(score - (1 - 1 / 9)) < 1e-9


def test_numeric_exact_is_one():
    assert silo_partial_score(9, 9, "distributed") == 1.0


def test_numeric_string_coercion_matches_int():
    # canonical_answer renders the truth as a JSON string "9"; the scorer must
    # parse it back so "9" and 9 compare/grade identically.
    truth_key = canonical_answer(9)
    assert isinstance(truth_key, str)
    assert silo_partial_score("8", truth_key, "distributed") == silo_partial_score(
        8, 9, "distributed"
    )


def test_numeric_far_miss_floors_at_zero():
    # A wildly off answer never goes negative.
    assert silo_partial_score(-100, 1, "distributed") == 0.0


def test_zero_truth_numeric():
    assert silo_partial_score(0, 0, "distributed") == 1.0
    assert silo_partial_score(1, 0, "distributed") < 1.0
    assert silo_partial_score(1, 0, "distributed") >= 0.0


def test_sorted_list_near_miss_is_partial():
    # [1,2,4,3] vs [1,2,3,4]: two positions match exactly; ordering quality high
    # but not perfect. Either way it must land strictly inside (0, 1).
    score = silo_partial_score([1, 2, 4, 3], [1, 2, 3, 4], "distributed")
    assert 0.0 < score < 1.0


def test_list_exact_is_one():
    assert silo_partial_score([1, 2, 3, 4], [1, 2, 3, 4], "distributed") == 1.0


def test_list_reversed_is_low_but_ordered_subseq_counts():
    # Fully reversed: no position matches, but the LIS-based ordering ratio still
    # credits the single-element longest increasing subsequence (1/4), so the
    # blended score is low yet positive.
    score = silo_partial_score([4, 3, 2, 1], [1, 2, 3, 4], "distributed")
    assert 0.0 < score < 0.5


def test_set_jaccard():
    # With the output_type hint "set", two lists are compared as sets.
    score = silo_partial_score([1, 2, 3], [2, 3, 4], "set")
    # Jaccard of {1,2,3} and {2,3,4} = |{2,3}| / |{1,2,3,4}| = 2/4 = 0.5
    assert abs(score - 0.5) < 1e-9


def test_dict_key_value_overlap():
    # {a:1, b:2, c:9} vs {a:1, b:5, c:9}: 2 of 3 key/value pairs match -> 2/3.
    score = silo_partial_score({"a": 1, "b": 2, "c": 9}, {"a": 1, "b": 5, "c": 9}, "distributed")
    assert abs(score - (2 / 3)) < 1e-9


def test_dict_exact_is_one():
    assert silo_partial_score({"a": 1, "b": 2}, {"a": 1, "b": 2}, "distributed") == 1.0


def test_string_token_overlap():
    # Non-numeric, non-JSON strings: normalized token overlap.
    score = silo_partial_score("the quick fox", "the quick brown fox", "distributed")
    assert 0.0 < score < 1.0


def test_string_exact_is_one():
    assert silo_partial_score("Candidate_D", "Candidate_D", "distributed") == 1.0


def test_unparseable_or_none_is_zero():
    assert silo_partial_score(None, 9, "distributed") == 0.0
    assert silo_partial_score("UNKNOWN", 9, "distributed") == 0.0


def test_type_mismatch_is_zero():
    # A scalar answer against a list truth cannot be graded sensibly -> 0.0.
    assert silo_partial_score(5, [1, 2, 3], "distributed") == 0.0



# --------------------------------------------------------------------------- #
# Silo's official LIS helper is loaded from the Silo-Bench checkout
# --------------------------------------------------------------------------- #
_OFFICIAL_LIS_SOURCE = """
def _longest_increasing_subsequence_length(seq):
    from bisect import bisect_left
    tails = []
    for x in seq:
        pos = bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)
        else:
            tails[pos] = x
    return len(tails)
"""


def _silo_checkout(root: Path, metrics_source: str | None) -> Path:
    """A Silo-Bench checkout layout; returns its ``benchmarks`` directory."""
    (root / "benchmarks").mkdir(parents=True)
    if metrics_source is not None:
        utils = root / "src" / "utils"
        utils.mkdir(parents=True)
        (utils / "metrics.py").write_text(metrics_source, encoding="utf-8")
    return root / "benchmarks"


def test_official_lis_is_loaded_from_the_silo_checkout(tmp_path, monkeypatch):
    bench = _silo_checkout(tmp_path / "silo", _OFFICIAL_LIS_SOURCE)
    monkeypatch.setenv("SILO_BENCH_DIR", str(bench))
    fn, official = silo_scoring._load_official_lis()
    assert official is True
    assert fn is not silo_scoring._local_lis_length
    assert fn([0, 2, 1, 3]) == 3


def test_official_lis_falls_back_to_the_local_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("SILO_BENCH_DIR", str(_silo_checkout(tmp_path / "bare", None)))
    assert silo_scoring._load_official_lis() == (silo_scoring._local_lis_length, False)
    # A helper that breaks the expected contract is not used either.
    broken = "def _longest_increasing_subsequence_length(seq):\n    return 0\n"
    monkeypatch.setenv("SILO_BENCH_DIR", str(_silo_checkout(tmp_path / "bad", broken)))
    assert silo_scoring._load_official_lis() == (silo_scoring._local_lis_length, False)


def test_official_lis_is_loaded_on_first_use(tmp_path, monkeypatch):
    bench = _silo_checkout(tmp_path / "silo", _OFFICIAL_LIS_SOURCE)
    monkeypatch.setenv("SILO_BENCH_DIR", str(bench))
    monkeypatch.setattr(silo_scoring, "_LIS", None)
    assert silo_scoring._lis_length([0, 2, 1, 3]) == 3
    fn, official = silo_scoring._LIS
    assert official is True and fn is not silo_scoring._local_lis_length


def test_local_lis_is_strictly_increasing():
    lis = silo_scoring._local_lis_length
    assert lis([]) == 0
    assert lis([0, 2, 1, 3]) == 3
    assert lis([4, 3, 2, 1]) == 1
    assert lis([1, 1, 1]) == 1
    assert lis([3, 1, 2, 5, 4, 6]) == 4


# --------------------------------------------------------------------------- #
# score_protocol_answer exposes partial alongside exact_match
# --------------------------------------------------------------------------- #
def test_score_protocol_answer_carries_partial():
    inst = BenchmarkInstance(
        benchmark="silo_bench",
        case_id="I-09",
        case_name="Top-K",
        n_agents=2,
        shards=[[1, 2], [3, 4]],
        ground_truth=[1, 2, 3, 4],
        meta={"output_type": "distributed"},
    )
    adapter = SiloProtocolAdapter(inst)
    global_task = adapter.build_global_task()

    near = adapter.score_protocol_answer([1, 2, 4, 3], global_task)
    assert near["exact_match"] is False
    assert near["primary_metric"] == 0.0  # strict signal: a near miss is 0
    assert 0.0 < near["partial"] < 1.0  # graded signal present

    exact = adapter.score_protocol_answer([1, 2, 3, 4], global_task)
    assert exact["exact_match"] is True
    assert exact["primary_metric"] == 1.0
    assert exact["partial"] == 1.0


def test_score_protocol_answer_module_function_matches_method():
    inst = BenchmarkInstance(
        benchmark="silo_bench",
        case_id="I-01",
        case_name="Global Max",
        n_agents=2,
        shards=[[3, 1, 9, 2], [5, 8, 4]],
        ground_truth=9,
        meta={"output_type": "distributed"},
    )
    adapter = SiloProtocolAdapter(inst)
    global_task = adapter.build_global_task()
    assert score_protocol_answer(8, global_task) == adapter.score_protocol_answer(
        8, global_task
    )



# --------------------------------------------------------------------------- #
# _score_python_execution: one executed team program -> ScoreResult
# --------------------------------------------------------------------------- #
_ABSENT = object()


class _Record:
    """Stand-in for the runtime's message / submission records."""

    def __init__(self, **data):
        self.__dict__.update(data)
        self._data = data

    def model_dump(self, mode: str = "python"):
        return dict(self._data)


def _planning(answers, *, knowledge, rounds=3, n_messages=2, calls=6):
    n = len(answers)
    submissions = [
        _Record(agent_id=i, answer=answer, submitted_round=rounds - 1)
        for i, answer in enumerate(answers)
        if answer is not _ABSENT
    ]
    messages = [
        _Record(round=0, sender=i % n, recipients=[(i + 1) % n], content=f"m{i}")
        for i in range(n_messages)
    ]
    output = SimpleNamespace(
        submissions=submissions, messages=messages, rounds_executed=rounds
    )
    execution = SimpleNamespace(
        output=output,
        final_knowledge=knowledge,
        authoritative_usage=SimpleNamespace(
            completion_tokens=60, prompt_tokens=40, model_calls=calls
        ),
        ledger={"submit_barrier": {"round": rounds - 1}},
    )
    return SimpleNamespace(execution=execution)


def _score(instance, goal, answers, *, knowledge=None):
    adapter = _protocol_adapter(instance, information_goal=goal)
    n = instance.n_agents
    return _score_python_execution(
        _planning(answers, knowledge=knowledge or [list(range(n))] * n),
        instance=instance,
        cfg=RunConfig(silo_eval_mode=goal, n_agents=n),
        task_adapter=adapter,
        global_task=adapter.build_global_task(),
        extra={"case_id": instance.case_id},
    )


def test_all_agents_success_requires_every_agent():
    score = _score(_global_max(), "all_agents", [9, 8], knowledge=[[0, 1], [1]])
    assert isinstance(score, ScoreResult)
    assert score.success is False
    assert score.extra["per_agent_correct"] == [True, False]
    assert score.extra["paper_S"] == 0.5
    # Level I quality is an exact match within a 1% relative tolerance, so 8
    # for 9 earns nothing.
    assert score.partial == 0.5 and score.extra["paper_P"] == 0.5
    assert score.final_answer == 9
    assert score.extra["information_goal"] == "all_agents"
    assert score.extra["information_coverage_by_agent"] == [1.0, 0.5]
    assert score.extra["mean_information_coverage"] == 0.75
    assert score.extra["all_agents_full_information"] is False
    # Cost metrics: C = output tokens per executed round, D = messages per
    # directed agent pair.
    assert score.extra["paper_C"] == 20.0
    assert score.extra["paper_D"] == 1.0
    assert (score.n_messages, score.n_model_calls, score.tokens) == (2, 6, 100)
    assert score.extra["python_submit_barrier"] == {"round": 2}
    assert [s["answer"] for s in score.extra["python_submissions"]] == [9, 8]

    solved = _score(_global_max(), "all_agents", [9, "9"])
    assert solved.success is True and solved.partial == 1.0
    assert solved.extra["all_agents_exact"] is True
    assert solved.extra["all_agents_full_information"] is True


def test_all_agents_missing_submission_is_wrong():
    score = _score(_global_max(), "all_agents", [9, _ABSENT])
    assert score.success is False
    assert score.extra["per_agent_answers"] == [9, None]
    assert score.extra["paper_S"] == 0.5


def test_segmented_all_agents_uses_each_agents_own_segment():
    good = _score(_segmented(), "all_agents", [[1, 3, 6], [10, 15]])
    assert good.success is True and good.partial == 1.0
    swapped = _score(_segmented(), "all_agents", [[10, 15], [1, 3, 6]])
    assert swapped.success is False
    assert swapped.extra["per_agent_correct"] == [False, False]


def test_sink_goal_grades_agent_zero_only():
    near = _score(_global_max(), "sink", [8, 9], knowledge=[[0], [0, 1]])
    assert near.success is False
    assert near.extra["sink_id"] == 0
    assert near.extra["sink_exact"] is False
    assert near.partial == pytest.approx(1 - 1 / 9)
    assert near.final_answer == 8
    assert near.extra["sink_information_coverage"] == 0.5
    assert "paper_S" not in near.extra

    hit = _score(_global_max(), "sink", [9, None])
    assert hit.success is True and hit.partial == 1.0
    assert hit.extra["information_goal"] == "sink"


def test_sink_goal_on_segmented_instance_uses_agent_zero_segment():
    hit = _score(_segmented(), "sink", [[1, 3, 6], None])
    assert hit.success is True and hit.partial == 1.0
    near = _score(_segmented(), "sink", [[1, 3, 7], [10, 15]])
    assert near.success is False
    assert 0.0 < near.partial < 1.0


def test_scoring_requires_program_output():
    planning = _planning([9, 9], knowledge=[[0], [1]])
    planning.execution.output = None
    adapter = _protocol_adapter(_global_max(), information_goal="all_agents")
    with pytest.raises(RuntimeError, match="lacks output"):
        _score_python_execution(
            planning,
            instance=_global_max(),
            cfg=RunConfig(silo_eval_mode="all_agents", n_agents=2),
            task_adapter=adapter,
            global_task=adapter.build_global_task(),
            extra={},
        )


# --------------------------------------------------------------------------- #
# End-to-end offline: payload -> sandboxed program -> ScoreResult
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("goal", ["all_agents", "sink"])
def test_offline_program_run_is_scored(goal, tmp_path):
    program = DEFAULT_MESSAGE_ONLY_V2_PROGRAM
    instance = _global_max(3)
    cfg = RunConfig(
        silo_eval_mode=goal,
        llm_provider="fake",
        model_name="fake",
        n_agents=3,
        max_parallel_agents=3,
        python_worker_contract="message_only_v2",
        request_timeout=30.0,
    )
    adapter = _protocol_adapter(instance, information_goal=goal)
    assert isinstance(adapter, SiloProtocolAdapter)
    global_task = adapter.build_global_task()
    payload = _build_python_execution_payload(
        instance=instance,
        cfg=cfg,
        task_adapter=adapter,
        global_task=global_task,
        n_agents=3,
    )
    runner = CodeProcessRunner(
        PythonExecutionLimits(
            timeout_seconds=_resolved_python_execution_timeout(cfg, n_agents=3),
            cpu_seconds=cfg.python_cpu_seconds,
            memory_mb=cfg.python_memory_mb,
            max_output_bytes=cfg.python_max_output_bytes,
        )
    )
    execution = runner.run(program, payload)
    assert execution.runtime_success, execution.failure
    planning = PythonCodePlanningResult(
        source=program,
        execution=execution,
        artifacts_dir=tmp_path,
        provenance="test",
        architect_prompt=None,
        attempts=[],
        planner_model_calls=0,
        repair_model_calls=0,
    )
    score = _score_python_execution(
        planning,
        instance=instance,
        cfg=cfg,
        task_adapter=adapter,
        global_task=global_task,
        extra={"case_id": instance.case_id},
    )
    # The offline worker never solves the task; the run is still fully scored.
    assert score.success is False
    assert isinstance(score.partial, float) and 0.0 <= score.partial <= 1.0
    assert score.extra["information_goal"] == goal
    assert score.n_model_calls > 0 and score.tokens > 0
    assert score.extra["rounds_executed"] >= 1
    assert canonical_answer(score.final_answer) != "9"
