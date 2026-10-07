"""Defaults of ``RunConfig`` and the shape of ``ScoreResult``."""

from queenbee.bench.config import RunConfig
from queenbee.bench.scoring import ScoreResult


def test_run_config_defaults():
    cfg = RunConfig()
    assert cfg.benchmark == "silo_bench"
    assert cfg.silo_eval_mode == "sink"
    assert cfg.llm_provider == "fake"
    assert cfg.model_name == "fake"
    assert cfg.base_url is None and cfg.api_key_env is None
    assert cfg.temperature == 0.0
    assert cfg.n_agents is None
    assert cfg.max_rounds == 4
    assert cfg.max_parallel_agents == 5
    assert cfg.request_timeout == 90.0
    assert cfg.python_worker_contract == "message_only_v2"
    assert cfg.python_execution_contract_version == "python_mas_v1"
    assert cfg.python_execution_timeout is None
    assert (cfg.python_cpu_seconds, cfg.python_memory_mb) == (10, 512)
    assert cfg.python_max_output_bytes == 1_000_000
    assert (
        cfg.python_max_model_calls,
        cfg.python_max_completion_tokens,
        cfg.python_max_messages,
    ) == (32, 20_000, 64)


def test_score_result_shape():
    score = ScoreResult(success=True, n_messages=4, n_model_calls=8, tokens=120)
    assert score.success is True
    assert score.partial is None
    assert score.n_messages == 4
    assert score.extra == {}
