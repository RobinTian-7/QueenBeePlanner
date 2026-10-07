"""Run configuration for executing one team program on one benchmark instance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    """How to run one team program on one instance."""

    benchmark: str = "silo_bench"
    # Whole-program wall-clock budget (s). None derives it from
    # request_timeout, max_rounds, n_agents and max_parallel_agents
    # (queenbee.bench.engine); a positive value sets an explicit cap (a
    # non-positive one is an error).
    python_execution_timeout: float | None = None
    python_cpu_seconds: int = 10
    python_memory_mb: int = 512
    python_max_output_bytes: int = 1_000_000
    python_execution_contract_version: str = "python_mas_v1"
    # Worker output contract; message_only_v2 is the only supported value.
    python_worker_contract: str = "message_only_v2"
    python_max_model_calls: int = 32
    python_max_completion_tokens: int = 20_000
    python_max_messages: int = 64
    n_agents: int | None = None
    max_rounds: int = 4
    # Intra-task concurrency. Calls from agents in the same logical round may
    # overlap; the next round still waits for the complete round snapshot.
    max_parallel_agents: int = 5
    llm_provider: str = "fake"
    model_name: str = "fake"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    # Per-request hard wall-clock timeout (seconds) for non-fake worker LLM
    # calls. A hung provider request is abandoned after this budget so the run
    # fails fast instead of hanging (see exp_graph.llm.timeout.TimeoutLLMClient);
    # a value <= 0 disables the guard. It also sizes the derived
    # python_execution_timeout. The fake client is instant, so it is never
    # wrapped.
    request_timeout: float = 90.0
    # Silo-Bench information goal (sink / all_agents). sink: all information
    # flows to one selected primary (agent 0), and only its answer is
    # submitted and scored; all_agents: every agent must hold the complete
    # information and answer correctly (success = every agent exact; a
    # majority does not count). It is the program's information_goal and
    # selects the scoring; queenbee.program.execute always sets it explicitly
    # (and renders the worker prompts for the same goal).
    silo_eval_mode: str = "sink"
