"""Runtime side of the program layer: seed, execution hooks, traces, imports.

Offline: programs run in the real sandbox with the ``fake`` worker provider
on a synthetic Silo-Bench-shaped instance.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

import queenbee.program.execute as ex
from queenbee.bench.instance import BenchmarkInstance
from queenbee.program.budgets import PythonRunBudgets

SRC = Path(__file__).resolve().parents[1] / "src"
SEED_SHA256 = "a302510d734091a01544e467e41389b28aed8ed45325ef80dcbc0176dba0d2e9"


def _instance(n_agents: int = 5, case_id: str = "I-01") -> BenchmarkInstance:
    shards = [[3, 1, 9, 2], [5, 8, 4], [7], [6, 0], [11, 10]][:n_agents]
    return BenchmarkInstance(
        benchmark="silo_bench",
        case_id=case_id,
        case_name="Global Max",
        n_agents=n_agents,
        shards=shards,
        ground_truth=11,
        task_prompt=(
            "Find the GLOBAL MAXIMUM across all agents' data. "
            "You are Agent {agent_id} and hold: {input_shard}\n"
            "**Output:** A single integer."
        ),
        meta={
            "output_type": "distributed",
            "is_segmented": False,
            "expected_outputs": [11] * n_agents,
        },
    )


def _seed() -> str:
    return ex.seed_python_source("message_only_v2")[2]


def _run(tmp_path: Path, source: str | None = None, **kw: Any) -> dict[str, Any]:
    kw.setdefault("instance", _instance())
    return ex.execute_python_source_on_case(
        source=_seed() if source is None else source,
        llm_provider="fake", worker_model="fake", goal="all_agents",
        worker_contract="message_only_v2", max_parallel_agents=5,
        request_timeout=30.0, artifacts_dir=tmp_path,
        budgets=PythonRunBudgets.for_rounds(64, n_agents=5), **kw,
    )


# --------------------------------------------------------------------------- #
# seed
# --------------------------------------------------------------------------- #


def test_seed_is_the_pinned_phase_structured_program() -> None:
    origin, title, source = ex.seed_python_source("message_only_v2", base="sfs_phase")
    assert origin == "seed:python_sfs_phase"
    assert title == "Python phase-structured sfs (relay + broadcast_last)"
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == SEED_SHA256
    for base in ("phase_sfs", "sfs-phase", " SFS_PHASE "):
        assert ex.seed_python_source("message_only_v2", base=base)[2] == source


@pytest.mark.parametrize(
    "contract, kwargs",
    [
        ("bogus_contract", {}),
        ("message_only_v2", {"base": "default"}),
        ("message_only_v2", {"base": "sfs"}),
    ],
)
def test_unavailable_seeds_fail_loudly(contract, kwargs) -> None:
    with pytest.raises(ValueError):
        ex.seed_python_source(contract, **kwargs)


# --------------------------------------------------------------------------- #
# execution and the score / diagnosis hooks
# --------------------------------------------------------------------------- #


def test_default_execution_scores_and_attaches_a_card(tmp_path) -> None:
    facts = _run(tmp_path)
    assert facts["infra"] is None and facts["execution_class"] == "completed"
    assert facts["model_calls"] == 10 and facts["n_messages"] == 8  # relay + broadcast
    assert facts["diag"]["S"] == facts["S"]
    assert facts["diag"]["submit_round"] == 5


def test_score_fn_replaces_the_facts_of_a_completed_run(tmp_path) -> None:
    calls: list[dict[str, Any]] = []

    def score_fn(facts, **kwargs):
        calls.append({"facts": dict(facts), **kwargs})
        return {**facts, "S": 0.25, "custom": True}

    inst = _instance()
    facts = _run(tmp_path, instance=inst, score_fn=score_fn)
    assert len(calls) == 1
    seen = calls[0]
    assert set(seen) == {"facts", "instance", "output", "payload", "score"}
    assert seen["instance"] is inst
    assert seen["payload"]["selected_primary"] == 0
    assert len(seen["output"].submissions) == 5
    assert seen["score"].n_model_calls == seen["facts"]["model_calls"]
    assert facts["S"] == 0.25 and facts["custom"] is True
    assert facts["diag"]["S"] == 0.25  # the card is built from the re-scored facts


def test_diag_fn_builds_the_card_on_every_path(tmp_path) -> None:
    seen: list[dict[str, Any]] = []

    def diag_fn(output, instance, facts, **context):
        seen.append({"output": output, "facts": dict(facts), **context})
        return {"custom_card": len(seen)}

    facts = _run(tmp_path, diag_fn=diag_fn)
    assert facts["diag"] == {"custom_card": 1}
    assert seen[0]["output"] is not None
    assert set(seen[0]) >= {"per_agent_correct", "ledger", "source", "goal"}
    # A run that fails in the sandbox: no re-scoring, the card still comes
    # from diag_fn (with no output).
    scored: list[Any] = []
    failed = _run(tmp_path, source="x = 1\n", diag_fn=diag_fn,
                  score_fn=lambda facts, **kw: scored.append(1) or facts)
    assert failed["execution_class"] == "algorithm_failure"
    assert failed["diag"] == {"custom_card": 2} and seen[1]["output"] is None
    assert scored == []


def test_a_failing_diag_fn_never_breaks_the_run(tmp_path) -> None:
    def broken(*_a, **_k):
        raise RuntimeError("card bug")

    facts = _run(tmp_path, diag_fn=broken)
    assert "diag" not in facts and facts["execution_class"] == "completed"


def test_evaluate_passes_the_hooks_and_keeps_row_bookkeeping(tmp_path) -> None:
    class Broken:
        case_id = "BROKEN"
        n_agents = "not-an-int"

    cards: list[str] = []

    def diag_fn(output, instance, facts, **context):
        cards.append(str(getattr(instance, "case_id", "?")))
        return {"card_for": getattr(instance, "case_id", "?")}

    rows = ex.evaluate_python_source_on_cases(
        source=_seed(), instances=(_instance(), Broken()), seeds=(7, 8),
        arm_label="seed", llm_provider="fake", worker_model="fake",
        goal="all_agents", worker_contract="message_only_v2",
        max_parallel_cases=2, max_parallel_agents=5, request_timeout=30.0,
        artifacts_dir=tmp_path, budgets=PythonRunBudgets.for_rounds(64, n_agents=5),
        score_fn=lambda facts, **kw: {**facts, "S": 0.5}, diag_fn=diag_fn,
    )
    assert [r["case_id"] for r in rows] == ["I-01", "BROKEN"]
    assert [r["seed"] for r in rows] == [7, 8]
    assert rows[0]["S"] == 0.5 and rows[0]["diag"] == {"card_for": "I-01"}
    assert rows[1]["execution_class"] == "algorithm_failure"
    assert rows[1]["error"].startswith("ValueError:")
    assert rows[1]["diag"] == {"card_for": "BROKEN"}
    with pytest.raises(ValueError):
        ex.evaluate_python_source_on_cases(
            source="x", instances=(1, 2), seeds=(1,), arm_label="a",
            llm_provider="fake", worker_model="fake", goal="all_agents",
            worker_contract="message_only_v2", max_parallel_cases=1,
            max_parallel_agents=1, request_timeout=1.0, artifacts_dir=tmp_path,
        )


def test_trace_file_contract(tmp_path, monkeypatch) -> None:
    trace_dir = tmp_path / "traces"
    monkeypatch.setenv("QB_TRACE_DIR", str(trace_dir))
    source = _seed()
    facts = _run(tmp_path, source=source)
    [path] = list(trace_dir.glob("*.json"))
    sha12 = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    case_id, sha, millis = path.stem.split("_")
    assert (case_id, sha) == ("I-01", sha12) and millis.isdigit()
    record = json.loads(path.read_text())
    assert set(record) == {"case_id", "n_agents", "source_sha", "facts", "output",
                           "ground_truth"}
    assert record["facts"] == json.loads(json.dumps(facts, default=str))
    assert record["ground_truth"] == 11 and record["n_agents"] == 5
    monkeypatch.delenv("QB_TRACE_DIR")
    _run(tmp_path, source=source)
    assert len(list(trace_dir.glob("*.json"))) == 1


def test_behaviour_simulation_of_the_seed() -> None:
    fp = ex.simulate_program_behavior(_seed(), n_agents=5, max_rounds=64)
    assert fp is not None and len(fp) == 5  # four relay rounds + one broadcast
    assert ex.simulate_program_behavior("x = 1\n", n_agents=5, max_rounds=64) is None


# --------------------------------------------------------------------------- #
# guarded imports
# --------------------------------------------------------------------------- #


def _guarded_queenbee_imports() -> list[tuple[str, int, str, list[str]]]:
    """Every ``from queenbee... import`` inside a ``try`` body under src/."""

    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    if (
                        isinstance(sub, ast.ImportFrom)
                        and sub.level == 0
                        and (sub.module or "").startswith("queenbee.")
                    ):
                        found.append((
                            str(path.relative_to(SRC)), sub.lineno, sub.module,
                            [alias.name for alias in sub.names],
                        ))
    return found


def test_guarded_imports_into_queenbee_resolve() -> None:
    """A guarded import falls back silently when its target moves; every one
    must resolve so no fallback is taken by accident."""

    guarded = _guarded_queenbee_imports()
    assert guarded, "expected guarded imports (e.g. the leak-audit headers)"
    broken = []
    for where, line, module, names in guarded:
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            broken.append(f"{where}:{line} {module}: {exc}")
            continue
        broken.extend(
            f"{where}:{line} {module}.{name}" for name in names if not hasattr(mod, name)
        )
    assert broken == []
