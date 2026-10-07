"""Silo-Bench metrics (:mod:`queenbee.bench.silo_metrics`): S and P of a run,
the per-agent quality rule of each level and the token / communication cost
formulas."""

from __future__ import annotations

import pytest

from queenbee.bench.silo_metrics import (
    evaluate_silo_submissions,
    silo_agent_quality,
    silo_communication_density,
    silo_token_consumption,
)


def test_level_ii_uses_positionwise_segment_accuracy() -> None:
    result = evaluate_silo_submissions(
        case_id="II-99",
        answers=[[1, 9, 3], [4]],
        expected_outputs=[[1, 2, 3], [4, 5]],
    )
    assert result["paper_S"] == 0.0
    assert result["per_agent_partial"] == pytest.approx([2 / 3, 1 / 2])
    assert result["paper_P"] == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_level_iii_uses_longest_correctly_ordered_subsequence() -> None:
    quality = silo_agent_quality(
        [2, 1, 3, 4],
        [1, 2, 3, 4],
        level="III",
    )
    assert quality == pytest.approx(3 / 4)


def test_level_i_uses_one_percent_tolerance() -> None:
    assert silo_agent_quality(100.9, 100, level="I") == 1.0
    assert silo_agent_quality(101.1, 100, level="I") == 0.0


def test_cost_formulas_do_not_multiply_density_by_rounds() -> None:
    assert silo_token_consumption(120, 3) == 40.0
    assert silo_communication_density(12, 4) == 1.0
    assert silo_communication_density(24, 4) == 2.0
